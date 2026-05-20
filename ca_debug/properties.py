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
    """If the rule starts with alive_fraction == 1.000 *exactly* AND
    stays above 0.95 throughout the whole run, the `is_alive` predicate
    is mis-calibrated for the field range — every cell trips the
    discrete threshold from step 0 and never falls below it. (This is
    the bug pattern we found in brusselator/schnakenberg.)

    Rules that start saturated but then evolve interesting dynamics
    (fire burning out → low alive; sandpile relaxing to SOC → ~0.88)
    are NOT flagged: their later steps drop well below the saturation
    floor, so the threshold is detecting *something* — it's just too
    generous at t=0, which is harmless if the dynamics carry the
    signal afterwards.
    """
    ts = run.timeseries
    if ts is None or len(ts) == 0 or 'alive_fraction' not in ts.columns:
        return []
    sorted_ts = ts.sort_values('step')
    af0 = float(sorted_ts['alive_fraction'].iloc[0])
    if af0 < 0.999:
        return []
    # Sustained: minimum alive_fraction across the whole run still
    # essentially saturated. Anything that drops below 0.95 at any
    # point is doing real work and not actually a calibration bug.
    af_min = float(sorted_ts['alive_fraction'].min())
    if af_min < 0.95:
        return []
    return [Failure(
        property="not_instant_saturated", step=int(sorted_ts['step'].iloc[0]),
        detail=f"alive_fraction = {af0:.4f} at step 0 and stays "
               f"≥ {af_min:.3f} throughout — predicate likely "
               f"mis-calibrated for this field's range",
        metrics={"start_alive": af0, "min_alive": af_min},
    )]


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
def _state_fingerprint(ts_row) -> tuple:
    """Spatial fingerprint of a single timeseries row. Uses populated-by-
    default columns sensitive to where mass sits in the grid.

    The per-channel ch_* stats are omitted because the harness only fills
    them for the single measure-channel — most are NaN and would falsely
    distinguish runs that are actually identical.
    """
    import numpy as np
    cols = [
        'alive_count', 'activity',
        'com_x', 'com_y', 'com_z', 'rg',
        'bbox_min_x', 'bbox_min_y', 'bbox_min_z',
        'bbox_max_x', 'bbox_max_y', 'bbox_max_z',
        'boundary_count', 'surface_ratio',
    ]
    fp = []
    for c in cols:
        v = ts_row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            fp.append(None)
        else:
            fp.append(round(float(v), 4))
    return tuple(fp)


def _trajectory_fingerprint(ts) -> tuple:
    """Full-trajectory fingerprint: a tuple of (alive_count, activity)
    pairs across every sampled step. Two seeds are considered to have
    "the same trajectory" iff the entire sequence matches bit-for-bit.

    This is robust to rules whose initial state is below the alive
    threshold (so step-0 alive_count = 0 for every seed) but whose
    dynamics later diverge — those rules properly fail neither stage of
    seed_respected. Conversely, a rule that genuinely ignores its rng
    will produce identical trajectories at every step.
    """
    import numpy as np
    if ts is None or len(ts) == 0:
        return ()
    ts_sorted = ts.sort_values('step')
    out = []
    for _, row in ts_sorted.iterrows():
        ac = row.get('alive_count')
        ap = row.get('activity')
        ac_v = int(ac) if ac is not None and not (isinstance(ac, float) and np.isnan(ac)) else None
        ap_v = round(float(ap), 6) if ap is not None and not (isinstance(ap, float) and np.isnan(ap)) else None
        out.append((int(row['step']), ac_v, ap_v))
    return tuple(out)


def check_seed_respected(rs) -> dict[str, list[Failure]]:
    """Whole-trajectory seed-sensitivity check, run per (rule, size) cell
    with ≥3 distinct seeds:

      `seed_affects_trajectory`: across the full timeseries, the
        (alive_count, activity) sequence must vary across at least two
        of the seeds. If every seed produces a bit-identical trajectory
        at every sampled step, the rule's RNG is being ignored — either
        the init function ignores its rng, or the dynamics use no
        randomness and all seeds happen to share a trivial init (which
        is itself worth flagging if the init is supposed to be random).

    Why whole-trajectory rather than just step-0 fingerprint: many rules
    have initial state below the `alive_threshold` so step-0 alive_count
    is 0 regardless of seed (e.g. predator_prey_3d's food field starts
    at 0.02–0.07, all below 0.5). Those rules properly diverge at later
    steps; only the trajectory comparison catches the divergence.

    Caveat: rules with deterministic-by-design inits (analytic quantum
    wavefunctions, single-voxel seeds, fluid_quiescent zero start, …)
    that *also* have deterministic dynamics will legitimately fail this
    check. They're not bugs — they're physics. The smell report's
    severity tier lets a reviewer skim the list and dismiss them.

    Returns {run_id: [Failure, ...]} keyed by the first run in each
    violating group.
    """
    import pandas as pd

    rows = []
    for r in rs.runs:
        m = r.manifest
        if r.timeseries is None or len(r.timeseries) == 0:
            continue
        rows.append({
            'run_id': r.run_id,
            'rule':   m.get('rule'),
            'size':   m.get('size'),
            'seed':   m.get('seed'),
            'traj':   _trajectory_fingerprint(r.timeseries),
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
        if grp['traj'].nunique() != 1:
            continue
        # Trajectories are bit-identical. Decide whether that's a
        # genuine RNG-ignored bug or a degenerate run that other
        # checks (dead_rule / instant_saturation / sustained_saturation)
        # already cover. Two degenerate cases to ignore:
        #
        #   (a) Dead-at-this-size: alive_count is 0 for every step. The
        #       rule produces nothing above the alive_threshold here, so
        #       there's no dynamics to vary. The dead-rule smell handles
        #       this as a separate finding.
        #   (b) Converged-saturated: the trajectory ends with alive_count
        #       at ≥95% of grid volume. The rule filled the grid, which
        #       is a physical fixed point regardless of seed (e.g. Eden
        #       growth from a single voxel always fills the box). The
        #       saturation check handles this elsewhere.
        traj = grp['traj'].iloc[0]
        if not traj:
            continue
        alives = [t[1] for t in traj if t[1] is not None]
        if not alives:
            continue
        max_alive = max(alives)
        if max_alive == 0:
            continue  # case (a): dead at this size
        vol = int(size) ** 3
        if alives[-1] >= 0.95 * vol:
            continue  # case (b): converged saturated
        first_run = grp['run_id'].iloc[0]
        out.setdefault(first_run, []).append(Failure(
            property="seed_affects_trajectory",
            detail=f"all {len(grp)} runs of {rule} at size={size} "
                   f"produced bit-identical (alive_count, activity) "
                   f"trajectories across {grp['seed'].nunique()} "
                   f"distinct seeds for every sampled step — "
                   f"init and/or dynamics RNG is ignored",
            metrics={"n_runs": int(len(grp)),
                     "n_seeds": int(grp['seed'].nunique()),
                     "n_steps_sampled": len(grp['traj'].iloc[0])},
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
            except Exception:  # noqa: BLE001  malformed JSON, treat as missing
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
        except Exception as e:  # noqa: BLE001  property check crash, log and continue
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
        except Exception as e:  # noqa: BLE001  property check crash, log and continue
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
