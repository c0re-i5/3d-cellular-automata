"""Code-smell detector over a runs/ tree.

Codifies the ad-hoc SQL queries we ran during the bug-hunt session into a
single command:

    python -m ca_debug.smell                     # full report
    python -m ca_debug.smell --kind collisions   # one section
    python -m ca_debug.smell --json              # machine-readable

Each detector returns a list of `Finding` dicts:
    {
        "kind":     str,          # detector id
        "severity": "high"|"med"|"low",
        "subject":  str,          # rule / group label the finding is about
        "detail":   str,          # one-line human summary
        "metrics":  dict[str, float|str],  # supporting numbers
    }

The report aggregates findings, groups by severity, and prints a ranked
summary. JSON mode emits the raw list for downstream tooling (CI, dashboards,
diffing two runs trees).

Detectors are intentionally pure-SQL where possible so adding a new one is a
single function. The full set is registered in DETECTORS at the bottom.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Callable

from . import analyzer
from . import schema as S


# ── Finding dataclass ─────────────────────────────────────────────────
@dataclass
class Finding:
    kind: str
    severity: str           # 'high' | 'med' | 'low'
    subject: str
    detail: str
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ── Detector helpers ──────────────────────────────────────────────────
def _q(query: str, runs_root: str):
    """Run a DuckDB query, return a DataFrame (empty if no data).

    Errors print to stderr rather than aborting so a single broken detector
    doesn't crash the whole report; but the noise is loud enough that you'll
    notice during development.
    """
    try:
        return analyzer.sql(query, runs_root=runs_root)
    except Exception as e:
        print(f"[smell] query failed: {type(e).__name__}: {e}", file=sys.stderr)
        return _empty_df()


def _empty_df():
    import pandas as pd
    return pd.DataFrame()


# ── Detectors ─────────────────────────────────────────────────────────
def detect_metric_collisions(runs_root: str) -> list[Finding]:
    """Rules that produce bit-identical metric tuples — likely duplicate
    preset definitions or aliasing bugs. Filters out uninformative groups
    where the metrics are all zero (e.g. fractals that legitimately don't
    evolve all hash to the same dead-rule tuple)."""
    df = _q("""
        select score, gol_coherence_max, projection_complexity, slice_mi_max,
               count(distinct rule) as n_rules,
               string_agg(distinct rule, ', ') as rules
        from runs
        group by score, gol_coherence_max, projection_complexity, slice_mi_max
        having count(distinct rule) > 1
           and score > 0.15                       -- exclude saturation-floor cluster
           and (gol_coherence_max + projection_complexity + slice_mi_max) > 0.0
        order by n_rules desc, score desc
    """, runs_root)
    out = []
    seen: set[frozenset[str]] = set()
    for _, row in df.iterrows():
        rules = frozenset(row['rules'].split(', '))
        if len(rules) < 2 or rules in seen:
            continue
        seen.add(rules)
        out.append(Finding(
            kind="metric_collision",
            severity="high",
            subject=", ".join(sorted(rules)),
            detail=f"{len(rules)} rules share identical (score, gol, projC, mi)",
            metrics={
                "n_rules": len(rules),
                "score": float(row['score']),
                "projection_complexity": float(row['projection_complexity']),
            },
        ))
    return out


def detect_seed_invariance(runs_root: str) -> list[Finding]:
    """Rules whose final state is byte-identical across multiple seeds.
    Many are legitimate (analytic field inits, single-nucleus seeds,
    fractals); the report flags them as 'low' and lets the human triage."""
    df = _q("""
        select rule,
               count(distinct seed) as n_seeds,
               count(distinct round(final_alive,5)) as n_unique,
               round(avg(score),3) as mean_score
        from runs
        where size = (select min(size) from runs)
        group by rule
        having count(distinct seed) >= 3
           and count(distinct round(final_alive,5)) = 1
           and avg(score) > 0.0
        order by mean_score desc
    """, runs_root)
    out = []
    for _, row in df.iterrows():
        out.append(Finding(
            kind="seed_invariant",
            severity="low",
            subject=row['rule'],
            detail=f"identical final state across {int(row['n_seeds'])} seeds "
                   f"(may be by-design for analytic/fractal inits)",
            metrics={"n_seeds": int(row['n_seeds']),
                     "mean_score": float(row['mean_score'])},
        ))
    return out


def detect_seed_fragility(runs_root: str) -> list[Finding]:
    """Rules where the score swings wildly between seeds — RNG-dependent
    pathologies, or a knife-edge parameter regime."""
    df = _q("""
        select rule,
               round(avg(score),3) as mean_score,
               round(stddev(score),3) as sd_score,
               round(min(score),3) as min_score,
               round(max(score),3) as max_score,
               count(*) as n
        from runs
        where size = (select min(size) from runs)
        group by rule
        having count(distinct seed) >= 3 and stddev(score) > 0.10
        order by sd_score desc
    """, runs_root)
    out = []
    for _, row in df.iterrows():
        out.append(Finding(
            kind="seed_fragile",
            severity="med",
            subject=row['rule'],
            detail=f"score varies by {row['sd_score']:.2f} across seeds "
                   f"(min={row['min_score']:.2f}, max={row['max_score']:.2f})",
            metrics={"sd_score": float(row['sd_score']),
                     "min_score": float(row['min_score']),
                     "max_score": float(row['max_score'])},
        ))
    return out


def detect_size_collapse(runs_root: str) -> list[Finding]:
    """Rules whose score *decreases* substantially when the grid grows.
    Almost always an intensive-vs-extensive metric artifact (localized rules
    on a big grid look 'dead' under fraction-based scoring), but some are
    real bugs (init/dt that doesn't scale)."""
    # Need at least two distinct sizes in the corpus to compare.
    sizes_df = _q("select distinct size from runs order by size", runs_root)
    if len(sizes_df) < 2:
        return []
    s_min = int(sizes_df['size'].iloc[0])
    s_max = int(sizes_df['size'].iloc[-1])
    df = _q(f"""
        with by_size as (
            select rule,
                   avg(case when size = {s_min} then score end) as score_small,
                   avg(case when size = {s_max} then score end) as score_large
            from runs group by rule
        )
        select rule,
               round(score_small, 3) as score_small,
               round(score_large, 3) as score_large,
               round(score_large - score_small, 3) as delta
        from by_size
        where score_small is not null and score_large is not null
          and (score_large - score_small) < -0.15
        order by delta
    """, runs_root)
    out = []
    for _, row in df.iterrows():
        out.append(Finding(
            kind="size_collapse",
            severity="med",
            subject=row['rule'],
            detail=f"score drops {row['score_small']:.2f}→{row['score_large']:.2f} "
                   f"as grid grows (likely localized-rule fraction artifact)",
            metrics={"score_small": float(row['score_small']),
                     "score_large": float(row['score_large']),
                     "delta": float(row['delta'])},
        ))
    return out


def detect_instant_saturation(runs_root: str) -> list[Finding]:
    """Rules whose alive_fraction reaches 1.0 within the first few steps —
    init not breaking symmetry, threshold mis-tuned for the field range,
    or runaway dynamics. Cross-checked: only flag if the rule's *best*
    score across all sampled configs is also weak (otherwise the rule
    is intentionally a filling process like Eden, or starts saturated
    and burns down to interesting structure like fire/sandpile)."""
    df = _q("""
        with first_few as (
            select t.run_id,
                   max(case when t.step <= 5 then t.alive_fraction end) as early_alive
            from timeseries t
            group by t.run_id
        )
        select r.rule,
               round(avg(f.early_alive), 3) as early_alive,
               round(max(r.score), 3) as best_score,
               count(*) as n_runs
        from first_few f join runs r using (run_id)
        where f.early_alive >= 0.99
        group by r.rule
        having max(r.score) < 0.3
        order by early_alive desc
    """, runs_root)
    out = []
    for _, row in df.iterrows():
        out.append(Finding(
            kind="instant_saturation",
            severity="high",
            subject=row['rule'],
            detail=f"alive_fraction = {row['early_alive']:.3f} within first 5 steps "
                   f"and weak final score — init doesn't break symmetry, "
                   f"or threshold mis-tuned for field range",
            metrics={"early_alive": float(row['early_alive']),
                     "n_runs": int(row['n_runs'])},
        ))
    return out


def detect_dead_rules(runs_root: str) -> list[Finding]:
    """Rules that score 0 across every seed at every size — completely
    inert. May be init bugs, dt bugs, or shader bugs."""
    df = _q("""
        select rule, count(*) as n_runs,
               round(max(score), 3) as max_score
        from runs
        group by rule
        having max(score) < 0.05
        order by rule
    """, runs_root)
    out = []
    for _, row in df.iterrows():
        out.append(Finding(
            kind="dead_rule",
            severity="high",
            subject=row['rule'],
            detail=f"score never exceeds {row['max_score']:.3f} across "
                   f"{int(row['n_runs'])} runs — inert at every tested config",
            metrics={"max_score": float(row['max_score']),
                     "n_runs": int(row['n_runs'])},
        ))
    return out


def detect_nan_inf(runs_root: str) -> list[Finding]:
    """Numerical instability — any rule that produced NaN or Inf in the
    measured channel."""
    df = _q("""
        select rule, count(*) as n_bad
        from runs
        where has_nan or has_inf
        group by rule
        order by n_bad desc
    """, runs_root)
    out = []
    for _, row in df.iterrows():
        out.append(Finding(
            kind="numerical_instability",
            severity="high",
            subject=row['rule'],
            detail=f"produced NaN or Inf in {int(row['n_bad'])} runs",
            metrics={"n_bad": int(row['n_bad'])},
        ))
    return out


def detect_anomaly_events(runs_root: str) -> list[Finding]:
    """Rules whose runs emitted runtime anomaly events (recorder-side)."""
    # events.jsonl is per-run JSON-lines; not exposed as a parquet view.
    # Walk run dirs directly.
    from glob import glob
    from pathlib import Path
    counts: dict[str, dict[str, int]] = {}
    for ev_path in glob(f"{runs_root}/*/{S.EVENTS_NAME}"):
        run_dir = Path(ev_path).parent
        manifest = run_dir / S.MANIFEST_NAME
        if not manifest.exists():
            continue
        with open(manifest) as f:
            rule = json.load(f).get('rule', 'unknown')
        with open(ev_path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                kind = ev.get('kind', '')
                if kind in ('anomaly', 'assertion_failed'):
                    counts.setdefault(rule, {}).setdefault(kind, 0)
                    counts[rule][kind] += 1
    out = []
    for rule, kinds in sorted(counts.items()):
        for kind, n in sorted(kinds.items()):
            out.append(Finding(
                kind=f"event:{kind}",
                severity="high" if kind == 'assertion_failed' else "med",
                subject=rule,
                detail=f"{n} {kind} events recorded",
                metrics={"n_events": n},
            ))
    return out


# ── Registry + report ──────────────────────────────────────────────────
DETECTORS: dict[str, Callable[[str], list[Finding]]] = {
    "collisions":       detect_metric_collisions,
    "saturation":       detect_instant_saturation,
    "dead":             detect_dead_rules,
    "nan_inf":          detect_nan_inf,
    "size_collapse":    detect_size_collapse,
    "seed_fragile":     detect_seed_fragility,
    "seed_invariant":   detect_seed_invariance,
    "events":           detect_anomaly_events,
}

SEVERITY_ORDER = {"high": 0, "med": 1, "low": 2}


def run_all(runs_root: str = S.DEFAULT_RUNS_ROOT,
            kinds: list[str] | None = None) -> list[Finding]:
    selected = kinds or list(DETECTORS.keys())
    findings: list[Finding] = []
    for name in selected:
        det = DETECTORS.get(name)
        if det is None:
            print(f"[smell] unknown detector: {name}", file=sys.stderr)
            continue
        try:
            findings.extend(det(runs_root))
        except Exception as e:
            print(f"[smell] {name} failed: {e}", file=sys.stderr)
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.kind, f.subject))
    return findings


def format_report(findings: list[Finding], runs_root: str) -> str:
    """Pretty-print findings grouped by severity then kind."""
    if not findings:
        return f"=== ca_debug smell report ({runs_root}) ===\n  no findings — clean!"

    lines = [f"=== ca_debug smell report ({runs_root}) ==="]
    by_sev: dict[str, list[Finding]] = {}
    for f in findings:
        by_sev.setdefault(f.severity, []).append(f)

    sev_label = {"high": "HIGH", "med": "MED ", "low": "LOW "}
    sev_count = {sev: len(by_sev.get(sev, [])) for sev in ("high", "med", "low")}
    lines.append(f"  {sev_count['high']} high · {sev_count['med']} med · "
                 f"{sev_count['low']} low  "
                 f"({len(findings)} findings across {len(set(f.kind for f in findings))} detectors)")
    lines.append("")

    for sev in ("high", "med", "low"):
        items = by_sev.get(sev, [])
        if not items:
            continue
        lines.append(f"── {sev_label[sev]} ──")
        # group by kind under each severity
        by_kind: dict[str, list[Finding]] = {}
        for f in items:
            by_kind.setdefault(f.kind, []).append(f)
        for kind, group in by_kind.items():
            lines.append(f"  [{kind}] ({len(group)})")
            for f in group[:20]:        # cap noisy kinds
                # truncate long subjects (e.g. comma-separated rule lists)
                subj = f.subject if len(f.subject) <= 60 else f.subject[:57] + "..."
                lines.append(f"     · {subj}")
                lines.append(f"         {f.detail}")
            if len(group) > 20:
                lines.append(f"     · ... and {len(group) - 20} more")
        lines.append("")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────
def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="ca_debug.smell",
        description="Detect suspicious patterns in a runs/ tree.")
    ap.add_argument("--runs-root", default=S.DEFAULT_RUNS_ROOT,
                    help="Path to runs directory (default: runs/)")
    ap.add_argument("--kind", action="append",
                    choices=list(DETECTORS.keys()),
                    help="Run only this detector (repeatable). "
                         "Default: run all.")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON instead of pretty report")
    ap.add_argument("--list-detectors", action="store_true",
                    help="List available detectors and exit")
    args = ap.parse_args()

    if args.list_detectors:
        for name, fn in DETECTORS.items():
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {name:<18s}  {doc}")
        return

    findings = run_all(runs_root=args.runs_root, kinds=args.kind)

    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2))
    else:
        print(format_report(findings, args.runs_root))


if __name__ == "__main__":
    _cli()
