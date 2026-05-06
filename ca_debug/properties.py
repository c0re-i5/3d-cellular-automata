"""Property-based assertions over finished run bundles.

Where ``smell.py`` finds patterns *across* runs by SQL, ``properties.py``
asserts invariants *within* a single run by walking its timeseries / events /
snapshots. Failures are written back into the bundle's ``events.jsonl`` with
``kind="assertion_failed"`` so the smell report's ``events`` detector picks
them up automatically — one query gives you the union across the whole corpus.

Usage:

    # Check a single bundle (writes failures back to its events.jsonl):
    python -m ca_debug.properties check runs/20260506_080124_lenia_3d_abc

    # Check every bundle in runs/ (idempotent — duplicate failures are
    # detected and not re-appended):
    python -m ca_debug.properties check-all

    # Just print without writing:
    python -m ca_debug.properties check <path> --dry-run

Each property is a function ``(run) -> Iterable[Failure]``. Properties are
deliberately conservative — they should fire only when something is *clearly*
wrong, never on borderline cases (smell.py is for fuzzy heuristics).

Properties currently implemented:
    non_constant         every rule must change *something* in 80 steps
    bounded_below_inf    no NaN/Inf in measured channel
    not_instant_dead     alive_fraction shouldn't drop to zero in step 1
                         unless it started zero
    seed_is_respected    *only* asserted when ≥3 runs of the same rule at
                         the same size+steps are present with different
                         seeds; the rule must produce ≥2 distinct hashes
                         (this is run cross-bundle, not per-bundle).

Adding a property: write `check_<name>(run) -> list[Failure]`, append it to
PROPERTIES below.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from glob import glob
from pathlib import Path
from typing import Any, Callable, Iterable

from . import analyzer
from . import schema as S


# ── Failure dataclass ─────────────────────────────────────────────────
@dataclass
class Failure:
    property: str           # which check_*
    detail: str             # one-line summary
    step: int | None = None
    metrics: dict = field(default_factory=dict)

    def to_event(self) -> dict[str, Any]:
        """Render as an events.jsonl row."""
        return {
            "kind": "assertion_failed",
            "step": self.step,
            "property": self.property,
            "detail": self.detail,
            "metrics": self.metrics,
        }


# ── Per-bundle property checks ────────────────────────────────────────
def check_non_constant(run) -> list[Failure]:
    """A rule that produced ≥10 timeseries samples must change something.
    activity > 0 at any step is enough; total absence is the bug."""
    ts = run.timeseries
    if ts is None or len(ts) < 10 or 'activity' not in ts.columns:
        return []
    peak_activity = float(ts['activity'].max())
    if peak_activity > 0.0:
        return []
    return [Failure(
        property="non_constant",
        detail=f"activity stayed at exactly 0.0 across {len(ts)} samples "
               f"(rule never changed any cell)",
        metrics={"peak_activity": peak_activity, "n_samples": len(ts)},
    )]


def check_no_nan_inf(run) -> list[Failure]:
    """No NaN or Inf in the measured channel — those are unrecoverable
    numerical instabilities."""
    ts = run.timeseries
    if ts is None or len(ts) == 0:
        return []
    out = []
    if 'has_nan' in ts.columns and bool(ts['has_nan'].any()):
        first_step = int(ts.loc[ts['has_nan'], 'step'].iloc[0])
        out.append(Failure(
            property="no_nan", step=first_step,
            detail=f"NaN appeared in measured channel at step {first_step}",
        ))
    if 'has_inf' in ts.columns and bool(ts['has_inf'].any()):
        first_step = int(ts.loc[ts['has_inf'], 'step'].iloc[0])
        out.append(Failure(
            property="no_inf", step=first_step,
            detail=f"Inf appeared in measured channel at step {first_step}",
        ))
    return out


def check_not_instant_dead(run) -> list[Failure]:
    """If the rule started with a non-zero initial state but alive_fraction
    dropped to exactly 0 by step ≤2, the dynamics are killing the seed
    immediately — almost always a dt or sign bug."""
    ts = run.timeseries
    if ts is None or len(ts) < 3 or 'alive_fraction' not in ts.columns:
        return []
    early = ts[ts['step'] <= 2].sort_values('step')
    if len(early) < 2:
        return []
    started = float(early['alive_fraction'].iloc[0])
    later = float(early['alive_fraction'].iloc[-1])
    if started > 0.001 and later == 0.0:
        return [Failure(
            property="not_instant_dead", step=int(early['step'].iloc[-1]),
            detail=f"alive_fraction collapsed {started:.3f} → 0 by step "
                   f"{int(early['step'].iloc[-1])} — dynamics are erasing the seed",
            metrics={"start_alive": started},
        )]
    return []


def check_not_instant_saturated(run) -> list[Failure]:
    """If the rule starts with alive_fraction == 1.000 *exactly*, the
    `is_alive` predicate is mis-calibrated for the field range — the
    init's continuous-field values trip the discrete threshold from
    step 0. (This is the bug pattern we found in brusselator/schnakenberg.)"""
    ts = run.timeseries
    if ts is None or len(ts) == 0 or 'alive_fraction' not in ts.columns:
        return []
    first = ts.sort_values('step').iloc[0]
    af = float(first['alive_fraction'])
    if af >= 0.999:
        return [Failure(
            property="not_instant_saturated", step=int(first['step']),
            detail=f"alive_fraction = {af:.4f} at step {int(first['step'])} "
                   f"— predicate likely mis-calibrated for this field's range",
            metrics={"start_alive": af},
        )]
    return []


def check_finite_run(run) -> list[Failure]:
    """The recorder marked the run as crashed (manifest['complete'] = False).
    Any incomplete run is a bug."""
    if run.manifest.get('complete', True):
        return []
    return [Failure(
        property="finite_run",
        detail="run terminated without writing complete=True to manifest",
    )]


# ── Cross-bundle property checks (need a RunSet) ──────────────────────
def check_seed_respected(rs) -> dict[str, list[Failure]]:
    """For any rule with ≥3 runs that share size+steps but differ in seed,
    require the runs to produce ≥2 distinct *spatial fingerprints* of the
    final state. The fingerprint is a tuple of populated-by-default
    columns (alive_count, activity, com_x/y/z, rg, bbox_min/max_x/y/z,
    boundary_count) at the last recorded step.

    These quantities are sensitive to where mass sits in the grid, not just
    how much — so seed-driven spatial reorganisation will distinguish runs
    even when alive_count happens to coincide. The per-channel ch_* stats
    can't be used because the harness only fills them for the single
    measure-channel.

    Returns {run_id: [Failure, ...]} keyed by the first run in each
    violating group.
    """
    import pandas as pd
    import numpy as np

    fingerprint_cols = [
        'alive_count', 'activity',
        'com_x', 'com_y', 'com_z', 'rg',
        'bbox_min_x', 'bbox_min_y', 'bbox_min_z',
        'bbox_max_x', 'bbox_max_y', 'bbox_max_z',
        'boundary_count', 'surface_ratio',
    ]

    rows = []
    for r in rs.runs:
        ts = r.timeseries
        m = r.manifest
        if ts is None or len(ts) == 0:
            continue
        last = ts.sort_values('step').iloc[-1]
        fp = []
        for col in fingerprint_cols:
            v = last.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                fp.append(None)
            else:
                fp.append(round(float(v), 4))
        rows.append({
            'run_id': r.run_id,
            'rule':   m.get('rule'),
            'size':   m.get('size'),
            'seed':   m.get('seed'),
            'fp':     tuple(fp),
        })
    if not rows:
        return {}
    df = pd.DataFrame(rows).dropna(subset=['rule', 'size', 'seed'])
    if df.empty:
        return {}

    out: dict[str, list[Failure]] = {}
    for (rule, size), grp in df.groupby(['rule', 'size']):
        if grp['seed'].nunique() < 3:
            continue
        n_distinct = grp['fp'].nunique()
        if n_distinct == 1:
            first_run = grp['run_id'].iloc[0]
            out.setdefault(first_run, []).append(Failure(
                property="seed_respected",
                detail=f"all {len(grp)} runs of {rule} at size={size} "
                       f"produced bit-identical spatial fingerprints "
                       f"(alive, activity, com, rg, bbox, boundary) across "
                       f"{grp['seed'].nunique()} distinct seeds — init "
                       f"and/or dynamics RNG is ignored",
                metrics={"n_runs": int(len(grp)),
                         "n_seeds": int(grp['seed'].nunique())},
            ))
    return out


# ── Registry ──────────────────────────────────────────────────────────
PER_BUNDLE_PROPERTIES: list[Callable[[Any], list[Failure]]] = [
    check_non_constant,
    check_no_nan_inf,
    check_not_instant_dead,
    check_not_instant_saturated,
    check_finite_run,
]

CROSS_BUNDLE_PROPERTIES: list[Callable[[Any], dict[str, list[Failure]]]] = [
    check_seed_respected,
]


# ── Driver ────────────────────────────────────────────────────────────
def _existing_assertion_keys(events_path: Path) -> set[tuple[str, int | None]]:
    """Return (property, step) pairs already recorded in events.jsonl, so
    re-running is idempotent."""
    if not events_path.exists():
        return set()
    keys: set[tuple[str, int | None]] = set()
    with open(events_path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get('kind') == 'assertion_failed':
                keys.add((ev.get('property', ''), ev.get('step')))
    return keys


def _append_failures(events_path: Path, failures: Iterable[Failure]) -> int:
    """Append failures to events.jsonl, skipping duplicates. Returns count
    actually written."""
    existing = _existing_assertion_keys(events_path)
    n = 0
    with open(events_path, 'a') as f:
        for fail in failures:
            key = (fail.property, fail.step)
            if key in existing:
                continue
            f.write(json.dumps(fail.to_event()) + "\n")
            existing.add(key)
            n += 1
    return n


def check_run(run, *, dry_run: bool = False) -> list[Failure]:
    """Run all per-bundle properties on one run; return all failures."""
    out: list[Failure] = []
    for prop in PER_BUNDLE_PROPERTIES:
        try:
            out.extend(prop(run))
        except Exception as e:
            out.append(Failure(
                property=prop.__name__,
                detail=f"property crashed: {type(e).__name__}: {e}",
            ))
    if not dry_run and out:
        _append_failures(run.path / S.EVENTS_NAME, out)
    return out


def check_all(runs_root: str = S.DEFAULT_RUNS_ROOT, *,
              dry_run: bool = False) -> dict[str, list[Failure]]:
    """Run all properties (per-bundle + cross-bundle) against every run.
    Returns {run_id: [failures]}."""
    rs = analyzer.load_runs(f"{runs_root}/*")
    by_run: dict[str, list[Failure]] = {}

    # Per-bundle pass.
    for r in rs.runs:
        fails = check_run(r, dry_run=dry_run)
        if fails:
            by_run[r.run_id] = list(fails)

    # Cross-bundle pass.
    for cross in CROSS_BUNDLE_PROPERTIES:
        try:
            for run_id, fails in cross(rs).items():
                by_run.setdefault(run_id, []).extend(fails)
                if not dry_run:
                    # find the run dir to write into
                    matches = [r for r in rs.runs if r.run_id == run_id]
                    if matches:
                        _append_failures(
                            matches[0].path / S.EVENTS_NAME, fails)
        except Exception as e:
            print(f"[properties] {cross.__name__} crashed: {e}",
                  file=sys.stderr)

    return by_run


# ── CLI ───────────────────────────────────────────────────────────────
def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="ca_debug.properties",
        description="Run property assertions over run bundles.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_one = sub.add_parser("check", help="check a single bundle")
    p_one.add_argument("path")
    p_one.add_argument("--dry-run", action="store_true")

    p_all = sub.add_parser("check-all",
                           help="check every bundle in a runs/ tree")
    p_all.add_argument("--runs-root", default=S.DEFAULT_RUNS_ROOT)
    p_all.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    if args.cmd == "check":
        run = analyzer.load_run(args.path)
        fails = check_run(run, dry_run=args.dry_run)
        if not fails:
            print(f"{run.run_id}: ok")
        else:
            print(f"{run.run_id}: {len(fails)} failure(s)")
            for f in fails:
                step = f"step {f.step}" if f.step is not None else "—"
                print(f"  [{f.property}] {step}: {f.detail}")
    elif args.cmd == "check-all":
        by_run = check_all(runs_root=args.runs_root, dry_run=args.dry_run)
        if not by_run:
            print("all properties pass")
            return
        # group by property to print a useful summary, not 600 lines of bundles
        from collections import Counter
        counter = Counter()
        examples: dict[str, str] = {}
        for run_id, fails in by_run.items():
            for f in fails:
                counter[f.property] += 1
                examples.setdefault(f.property, f"{run_id}: {f.detail}")
        total = sum(counter.values())
        n_runs = len(by_run)
        print(f"=== property failures across {n_runs} runs ({total} total) ===")
        for prop, n in counter.most_common():
            print(f"  [{prop}] {n}")
            print(f"     e.g. {examples[prop]}")
        suffix = "  (dry-run; events.jsonl not modified)" if args.dry_run else ""
        print(suffix)


if __name__ == "__main__":
    _cli()
