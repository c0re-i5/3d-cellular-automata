"""Batch triage: run microscope across every rule and produce a
ranked QA report.

Usage:

    python -m ca_debug.triage                # all rules, default budget
    python -m ca_debug.triage --skip-flagship
    python -m ca_debug.triage --rules wireworld_3d,sandpile_3d
    python -m ca_debug.triage --params 2 --seeds 1 --steps 48 --size 24
    python -m ca_debug.triage --json triage.json

For each rule we run a small (seeds × params × init_variants) cross-product
via ``ca_debug.microscope.microscope`` (with stdout silenced), grade the
returned report, and finally print a single sorted table:

    SEVERITY  RULE                       MAX   N_DEAD/N   FLAGS
    crit      genome_ca_3d               0.22   0/24      MAX_SCORE_LOW PARAMS_DEAD
    high      sandpile_3d                0.42   3/48      MAX_SCORE_SUB
    high      wireworld_3d               0.97   0/48      PARAMS_DEAD INIT_DEAD:wireworld_torus
    ...
    ok        game_of_life_3d            0.91   0/16

Severity levels (highest first):
  crit  - max < 0.30 OR all trials NaN/Inf OR every trial errored
  high  - max < 0.50, or PARAMS_DEAD, or INIT_DEAD, or DEAD_MAJORITY
  med   - SCORE_SAT (sd < 0.02 with non-trivial mean)
  ok    - no flags
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
from typing import Any


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _grade(report: dict[str, Any]) -> dict[str, Any]:
    """Reduce a microscope report to a single triage row."""
    secs = report.get('sections', {})
    iv_scores = secs.get('init_variant_scores', {})
    params_r = secs.get('param_correlations', {})
    anom = secs.get('anomalies', {}) or {}
    n_trials = report.get('n_trials', 0) or 0
    errors = report.get('errors', [])

    flags: list[str] = []
    severity = 'ok'

    # crit: every trial errored
    if errors and len(errors) >= n_trials:
        return {'severity': 'crit', 'flags': ['ALL_ERRORS'],
                'max_score': float('nan'), 'n_dead': 0, 'n_total': n_trials,
                'note': errors[0].get('error', '?')[:80] if errors else ''}

    # Aggregate max_score across init_variants
    max_score = 0.0
    for s in iv_scores.values():
        if s.get('max', 0) > max_score:
            max_score = s['max']

    n_dead = anom.get('n_dead', 0)
    n_nan = anom.get('n_nan', 0)
    n_inf = anom.get('n_inf', 0)

    # Numerical instabilities
    if (n_nan + n_inf) > 0:
        flags.append(f"NUMERIC_BAD({n_nan + n_inf})")
        severity = 'crit'

    # Score ceiling
    if max_score < 0.30:
        flags.append(f"MAX_SCORE_LOW({max_score:.2f})")
        if severity != 'crit':
            severity = 'crit'
    elif max_score < 0.50:
        flags.append(f"MAX_SCORE_SUB({max_score:.2f})")
        if severity == 'ok':
            severity = 'high'

    # Dead majority
    if n_trials and n_dead > 0.5 * n_trials:
        flags.append(f"DEAD_MAJORITY({n_dead}/{n_trials})")
        if severity == 'ok':
            severity = 'high'

    # Param plumbing: all |r| < 0.10 (only meaningful with enough samples)
    pcs = [r for r in params_r.values() if r is not None]
    if len(pcs) >= 2 and n_trials >= 6 and max(abs(r) for r in pcs) < 0.10:
        flags.append(f"PARAMS_DEAD(|r|<{max(abs(r) for r in pcs):.2f})")
        if severity == 'ok':
            severity = 'high'

    # Dead init_variants alongside healthy ones
    dead_iv = [iv for iv, s in iv_scores.items() if s.get('median', 0) < 0.20]
    alive_iv = [iv for iv, s in iv_scores.items() if s.get('median', 0) >= 0.50]
    if dead_iv and alive_iv:
        flags.append(f"INIT_DEAD:{','.join(dead_iv)}")
        if severity == 'ok':
            severity = 'high'

    # Score saturation (low variance with non-trivial mean) — only if
    # we'd otherwise call it ok.
    all_means = [s.get('mean', 0) for s in iv_scores.values()]
    all_sds = [s.get('sd', 0) for s in iv_scores.values()]
    if all_means and max(all_means) > 0.30 and all(sd < 0.02 for sd in all_sds):
        flags.append("SCORE_SAT")
        if severity == 'ok':
            severity = 'med'

    if not flags:
        flags = ['—']

    return {
        'severity': severity,
        'flags':    flags,
        'max_score': max_score,
        'n_dead':   n_dead,
        'n_total':  n_trials,
        'note':     '',
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_SEV_ORDER = {'crit': 0, 'high': 1, 'med': 2, 'ok': 3, 'err': 4}


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS
    if args.rules:
        wanted = [r.strip() for r in args.rules.split(',') if r.strip()]
        bad = [r for r in wanted if r not in RULE_PRESETS]
        if bad:
            raise SystemExit(f"unknown rules: {bad}")
        return wanted
    rules = list(RULE_PRESETS.keys())
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip]
    return rules


def _run_one(ctx, rule: str, args) -> tuple[dict, dict, str]:
    """Returns (report, grade, captured_stdout)."""
    from ca_debug.microscope import microscope
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            report = microscope(ctx, rule,
                                n_seeds=args.seeds, n_params=args.params,
                                size=args.size, steps=args.steps,
                                seed_base=args.seed_base)
    except Exception as e:
        tb = traceback.format_exc(limit=4)
        report = {'rule': rule, 'n_trials': 0,
                  'errors': [{'error': f"{type(e).__name__}: {e}"}],
                  'sections': {}, 'fatal': True, 'traceback': tb}
    grade = _grade(report)
    if report.get('fatal'):
        grade['severity'] = 'err'
        grade['flags'] = ['FATAL']
        grade['note'] = report['errors'][0]['error'][:80]
    return report, grade, buf.getvalue()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (_SEV_ORDER.get(r['grade']['severity'], 9),
                                       -len(r['grade']['flags']),
                                       r['rule']))
    print()
    print(f"{'SEV':5s}  {'RULE':30s}  {'MAX':>5s}  {'DEAD':>9s}  {'TIME':>5s}  FLAGS / NOTE")
    print("-" * 110)
    for r in rows:
        g = r['grade']
        max_s = f"{g['max_score']:.2f}" if g['max_score'] == g['max_score'] else "  nan"
        dead = f"{g['n_dead']}/{g['n_total']}"
        flags = " ".join(g['flags'])
        note = (" — " + g['note']) if g.get('note') else ""
        print(f"{g['severity']:5s}  {r['rule']:30s}  {max_s:>5s}  "
              f"{dead:>9s}  {r['elapsed']:>4.1f}s  {flags}{note}")
    print()
    # Counts
    counts: dict[str, int] = {}
    for r in rows:
        counts[r['grade']['severity']] = counts.get(r['grade']['severity'], 0) + 1
    print("Summary:", " ".join(f"{k}={counts[k]}"
                               for k in ('crit', 'high', 'med', 'ok', 'err')
                               if k in counts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(prog="ca_debug.triage",
        description="Batch-run microscope across all rules and produce a triage table.")
    ap.add_argument("--rules", default="",
                    help="Comma-separated subset (default: all)")
    ap.add_argument("--skip", default="",
                    help="Comma-separated rules to skip")
    ap.add_argument("--skip-flagship", action="store_true",
                    help="Skip flagship_* composed presets")
    ap.add_argument("--seeds", type=int, default=2,
                    help="Seeds per (param, init) cell (default: 2)")
    ap.add_argument("--params", type=int, default=3,
                    help="Random param vectors per rule (default: 3)")
    ap.add_argument("--size", type=int, default=24,
                    help="Grid size (default: 24)")
    ap.add_argument("--steps", type=int, default=48,
                    help="Steps per trial (default: 48)")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--json", default="",
                    help="If set, dump full per-rule reports + grades to this file")
    ap.add_argument("--verbose", action="store_true",
                    help="Print microscope's full per-rule output as it runs")
    args = ap.parse_args()

    rules = _select_rules(args)
    print(f"triage: {len(rules)} rules; budget = "
          f"{args.params}p × {args.seeds}s × inits  at size={args.size} steps={args.steps}")
    print()

    from test_harness import create_headless_context
    window, ctx = create_headless_context()

    rows: list[dict] = []
    all_reports: dict[str, Any] = {}
    t_start = time.time()
    try:
        for i, rule in enumerate(rules, 1):
            t0 = time.time()
            report, grade, captured = _run_one(ctx, rule, args)
            elapsed = time.time() - t0
            row = {'rule': rule, 'grade': grade, 'elapsed': elapsed}
            rows.append(row)
            all_reports[rule] = {'report': report, 'grade': grade}
            flag_str = " ".join(grade['flags'])
            print(f"  [{i:3d}/{len(rules)}] {grade['severity']:5s}  "
                  f"{rule:30s}  max={grade['max_score']:.2f}  "
                  f"dead={grade['n_dead']}/{grade['n_total']}  "
                  f"{elapsed:.1f}s  {flag_str}", flush=True)
            if args.verbose and captured:
                print(captured)
    finally:
        try:
            window.destroy()
        except Exception:
            pass

    print(f"\nDone in {time.time() - t_start:.1f}s.")
    _print_table(rows)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(all_reports, f, indent=2, default=str)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    _cli()
