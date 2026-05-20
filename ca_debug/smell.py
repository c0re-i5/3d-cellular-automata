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
    except Exception as e:  # noqa: BLE001  sql query may fail, log and continue
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
        where kind = 'audit'
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
        where kind = 'audit'
          and size = (select min(size) from runs where kind='audit')
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
        where kind = 'audit'
          and size = (select min(size) from runs where kind='audit')
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
    sizes_df = _q("select distinct size from runs where kind='audit' order by size", runs_root)
    if len(sizes_df) < 2:
        return []
    s_min = int(sizes_df['size'].iloc[0])
    s_max = int(sizes_df['size'].iloc[-1])
    df = _q(f"""
        with by_size as (
            select rule,
                   avg(case when size = {s_min} then score end) as score_small,
                   avg(case when size = {s_max} then score end) as score_large
            from runs where kind='audit' group by rule
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
        where r.kind = 'audit' and f.early_alive >= 0.99
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
        where kind = 'audit'
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
        where kind = 'audit' and (has_nan or has_inf)
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


# ── Discoveries-source detectors ──────────────────────────────────────
# These read discoveries.json (search-output, not run-bundles) so they
# answer questions about the SEARCH process rather than the AUDIT process.
# Discoveries record (rule, params, score, init_variant) but no timeseries.
DEFAULT_DISCOVERIES = "discoveries.json"


def _load_discoveries(path: str) -> list[dict]:
    """Load discoveries.json; empty list if missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:  # noqa: BLE001  malformed JSON, treat as missing
        print(f"[smell] could not read {path}: {e}", file=sys.stderr)
        return []


def _by_rule(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r.get('rule', '?'), []).append(r)
    return out


def detect_param_insensitive(discoveries_path: str = DEFAULT_DISCOVERIES,
                             *, min_records: int = 5,
                             score_sd_threshold: float = 0.02) -> list[Finding]:
    """Per (rule, init_variant), if score barely moves across ≥min_records
    samples with random parameter draws, the rule is either insensitive
    to its tuned params or its scoring is saturated. Either way, search
    is wasting compute and the harness will mis-rank promotions.

    Wireworld is the canonical case: params span 7× ranges but score
    pins to 0.819 ± 0.001 across all 11 records.
    """
    import statistics
    recs = _load_discoveries(discoveries_path)
    if not recs:
        return []
    by: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        key = (r.get('rule', '?'), r.get('init_variant') or '<default>')
        by.setdefault(key, []).append(r)
    out = []
    for (rule, iv), rs in by.items():
        if len(rs) < min_records:
            continue
        scores = [r['score'] for r in rs if isinstance(r.get('score'), (int, float))]
        if len(scores) < min_records:
            continue
        sd = statistics.pstdev(scores)
        if sd >= score_sd_threshold:
            continue
        # Compute a quick param-spread sanity number: how varied are the
        # actual param vectors? If params themselves are constant the
        # finding is weak. We measure by counting unique tuples.
        param_keys = sorted({k for r in rs for k in (r.get('params') or {}).keys()})
        unique_params = {tuple(round(float((r.get('params') or {}).get(k, 0)), 4)
                               for k in param_keys) for r in rs}
        if len(unique_params) < max(2, len(rs) // 2):
            continue  # params themselves don't vary — not a rule-side bug
        out.append(Finding(
            kind="param_insensitive",
            severity="high",
            subject=f"{rule} ({iv})",
            detail=f"score sd = {sd:.4f} across {len(rs)} records with "
                   f"{len(unique_params)} distinct param vectors — "
                   f"score band [{min(scores):.3f}, {max(scores):.3f}]",
            metrics={"n_records": len(rs),
                     "score_sd": round(sd, 5),
                     "score_min": round(min(scores), 4),
                     "score_max": round(max(scores), 4),
                     "n_unique_params": len(unique_params)},
        ))
    return out


def detect_score_pinned(discoveries_path: str = DEFAULT_DISCOVERIES,
                        *, min_records: int = 10,
                        ceiling_threshold: float = 0.5) -> list[Finding]:
    """Rules whose discoveries never break above a low score ceiling
    despite many search attempts. Distinct from `param_insensitive` —
    the score *does* vary, it just never reaches the band the harness
    considers 'interesting' (≥0.5 by convention). Likely causes:
      * scoring weights mistuned for this rule's dynamic range
      * param ranges centered in a degenerate regime
      * rule itself genuinely uninteresting (then prune from search)

    Sandpile (max 0.32, n=8) and prisoners_dilemma (max 0.45, n=28) are
    the two current cases.
    """
    recs = _load_discoveries(discoveries_path)
    if not recs:
        return []
    out = []
    for rule, rs in _by_rule(recs).items():
        scores = [r['score'] for r in rs if isinstance(r.get('score'), (int, float))]
        if len(scores) < min_records:
            continue
        mx = max(scores)
        if mx >= ceiling_threshold:
            continue
        out.append(Finding(
            kind="score_pinned",
            severity="med",
            subject=rule,
            detail=f"max score = {mx:.3f} across {len(rs)} records "
                   f"(ceiling = {ceiling_threshold:.2f}) — scoring or "
                   f"param range needs review",
            metrics={"n_records": len(rs),
                     "score_max": round(mx, 4),
                     "score_median": round(sorted(scores)[len(scores)//2], 4)},
        ))
    return out


def detect_init_variant_redundant(discoveries_path: str = DEFAULT_DISCOVERIES,
                                  *, min_per_variant: int = 5,
                                  ks_p_threshold: float = 0.5) -> list[Finding]:
    """For rules with multiple init_variants, flag pairs whose score
    distributions are statistically indistinguishable (KS test p>0.5).
    Means one variant is dead weight in the search budget — either
    pick the cheaper one or differentiate them.

    Soft dependency on scipy; if missing, this detector silently
    returns nothing.
    """
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return []
    recs = _load_discoveries(discoveries_path)
    if not recs:
        return []
    out = []
    for rule, rs in _by_rule(recs).items():
        by_iv: dict[str, list[float]] = {}
        for r in rs:
            iv = r.get('init_variant') or '<default>'
            s = r.get('score')
            if isinstance(s, (int, float)):
                by_iv.setdefault(iv, []).append(float(s))
        ivs = [iv for iv, ss in by_iv.items() if len(ss) >= min_per_variant]
        if len(ivs) < 2:
            continue
        # All pairwise KS tests
        for i, a in enumerate(ivs):
            for b in ivs[i+1:]:
                stat, p = ks_2samp(by_iv[a], by_iv[b])
                if p < ks_p_threshold:
                    continue
                out.append(Finding(
                    kind="init_variant_redundant",
                    severity="low",
                    subject=f"{rule}: {a} ~ {b}",
                    detail=f"score distributions indistinguishable "
                           f"(KS p={p:.2f}, n={len(by_iv[a])}/{len(by_iv[b])})",
                    metrics={"ks_p": round(float(p), 4),
                             "ks_stat": round(float(stat), 4),
                             "n_a": len(by_iv[a]), "n_b": len(by_iv[b])},
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
                except Exception:  # noqa: BLE001  malformed JSON, treat as missing
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
# Detectors come in two flavours: those that read a runs_root (bundle-
# derived) and those that read a discoveries.json path. We tag them so
# the runner can dispatch the right argument.
DETECTORS: dict[str, Callable] = {
    "collisions":              detect_metric_collisions,
    "saturation":              detect_instant_saturation,
    "dead":                    detect_dead_rules,
    "nan_inf":                 detect_nan_inf,
    "size_collapse":           detect_size_collapse,
    "seed_fragile":            detect_seed_fragility,
    "seed_invariant":          detect_seed_invariance,
    "events":                  detect_anomaly_events,
    "param_insensitive":       detect_param_insensitive,
    "score_pinned":            detect_score_pinned,
    "init_variant_redundant":  detect_init_variant_redundant,
}

DISCOVERY_DETECTORS = {
    "param_insensitive", "score_pinned", "init_variant_redundant",
}

SEVERITY_ORDER = {"high": 0, "med": 1, "low": 2}


def run_all(runs_root: str = S.DEFAULT_RUNS_ROOT,
            kinds: list[str] | None = None,
            discoveries_path: str = DEFAULT_DISCOVERIES) -> list[Finding]:
    selected = kinds or list(DETECTORS.keys())
    findings: list[Finding] = []
    for name in selected:
        det = DETECTORS.get(name)
        if det is None:
            print(f"[smell] unknown detector: {name}", file=sys.stderr)
            continue
        try:
            arg = discoveries_path if name in DISCOVERY_DETECTORS else runs_root
            findings.extend(det(arg))
        except Exception as e:  # noqa: BLE001  detector may crash, log and continue
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
    ap.add_argument("--discoveries", default=DEFAULT_DISCOVERIES,
                    help="Path to discoveries.json (default: discoveries.json)")
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

    findings = run_all(runs_root=args.runs_root, kinds=args.kind,
                       discoveries_path=args.discoveries)

    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2))
    else:
        print(format_report(findings, args.runs_root))


if __name__ == "__main__":
    _cli()
