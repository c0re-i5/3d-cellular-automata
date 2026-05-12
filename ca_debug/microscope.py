"""Per-rule microscope: drill into one CA's behaviour across the full
(seed × params × init_variant) cross-product and print a structured QA
report.

Designed for the eventual "go through every CA in microscopic detail"
QA pass — instead of eyeballing each rule, run:

    python -m ca_debug.microscope wireworld_3d
    python -m ca_debug.microscope sandpile_3d --seeds 5 --params 12

The report covers:

  1.  Rule overview (params, ranges, init_variants, dt, defaults)
  2.  Score landscape (per init_variant: n, mean, sd, min/max)
  3.  Per-parameter effect (Pearson correlation with score)
  4.  Init-variant differentiation (pairwise KS, if scipy available)
  5.  Seed sensitivity (sd of score across seeds with same params)
  6.  Numerical anomalies (NaN/Inf rates, dead/saturated rates)
  7.  Top-N most interesting (highest score) and bottom-N (lowest)
      with their full param vectors — so the human can drill further.

This module IMPORTS test_harness (which owns the GPU context) so it
must be invoked while a venv with the simulator deps is active.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from typing import Any


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r without numpy. Returns None if undefined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _summarise(scores: list[float]) -> dict[str, float]:
    return {
        'n':      len(scores),
        'mean':   statistics.mean(scores),
        'sd':     statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        'min':    min(scores),
        'max':    max(scores),
        'median': statistics.median(scores),
    }


def _fmt_summary(s: dict[str, float]) -> str:
    return (f"n={s['n']:3d}  mean={s['mean']:.3f}  sd={s['sd']:.3f}  "
            f"range=[{s['min']:.3f}, {s['max']:.3f}]  median={s['median']:.3f}")


def microscope(ctx, rule_name: str, *,
               n_seeds: int = 3, n_params: int = 8,
               size: int = 32, steps: int = 80,
               seed_base: int = 1000) -> dict[str, Any]:
    """Run the trial cross-product and return a structured report dict.

    ``ctx`` is the moderngl context produced by
    ``test_harness.create_headless_context``. The caller owns the
    context lifecycle.

    Cross-product: n_params random param vectors × n_seeds seeds × all
    init_variants. Each cell is one trial. Total trials =
    n_params * n_seeds * len(init_variants).
    """
    # Local imports — ca_debug normally avoids importing the simulator
    # to keep `python -m ca_debug.smell` light.
    import numpy as np
    from simulator import RULE_PRESETS, _resolve_composed_preset
    from test_harness import run_trial, randomize_params

    if rule_name not in RULE_PRESETS:
        raise SystemExit(f"unknown rule: {rule_name!r}")
    preset = _resolve_composed_preset(rule_name)
    init_variants = preset.get('init_variants') or [preset.get('init')]
    init_variants = [iv for iv in init_variants if iv]
    if not init_variants:
        init_variants = [None]

    # Pre-draw the param vectors so every init_variant sees the same
    # set — that way differences across variants reflect init effects
    # rather than param sampling noise.
    param_rng = np.random.RandomState(seed_base)
    param_vectors = [randomize_params(preset, param_rng)
                     for _ in range(n_params)]
    seeds = [seed_base + 1 + i for i in range(n_seeds)]

    print(f"\n=== microscope: {rule_name} ===")
    print(f"  init_variants:   {init_variants}")
    print(f"  param_ranges:    {preset.get('param_ranges', {})}")
    print(f"  trial cross:     {n_params} params × {n_seeds} seeds × "
          f"{len(init_variants)} inits  = "
          f"{n_params * n_seeds * len(init_variants)} trials at "
          f"size={size}, steps={steps}")
    print()

    records: list[dict[str, Any]] = []
    n_total = n_params * n_seeds * len(init_variants)
    n_done = 0
    for iv in init_variants:
        for pv_idx, params in enumerate(param_vectors):
            for seed in seeds:
                n_done += 1
                try:
                    res = run_trial(ctx, rule_name,
                                    size=size, seed=seed, steps=steps,
                                    sample_interval=max(5, steps // 16),
                                    params=params, dt=preset.get('dt'),
                                    init_override=iv)
                    rec = {
                        'init_variant': iv,
                        'pv_idx':       pv_idx,
                        'seed':         seed,
                        'params':       dict(params),
                        'score':        float(res.get('score', 0.0)),
                        'final_alive':  float(res.get('final_alive', 0.0)),
                        'mean_activity': float(res.get('mean_activity', 0.0)),
                        'has_nan':      bool(res.get('has_nan', False)),
                        'has_inf':      bool(res.get('has_inf', False)),
                    }
                except Exception as e:
                    rec = {'init_variant': iv, 'pv_idx': pv_idx,
                           'seed': seed, 'params': dict(params),
                           'error': f"{type(e).__name__}: {e}",
                           'score': float('nan')}
                records.append(rec)
                if n_done % 10 == 0 or n_done == n_total:
                    print(f"  [{n_done:3d}/{n_total}] done", flush=True)

    return _analyse(rule_name, records, preset, seeds, param_vectors)


def _analyse(rule_name: str, records: list[dict], preset: dict,
             seeds: list[int], param_vectors: list[dict]) -> dict[str, Any]:
    """Build the structured report; mostly delegated to small helpers."""
    print()
    report: dict[str, Any] = {
        'rule':       rule_name,
        'n_trials':   len(records),
        'errors':     [r for r in records if 'error' in r],
        'sections':   {},
    }
    ok = [r for r in records if 'error' not in r]
    if not ok:
        print("ALL TRIALS FAILED — see report['errors']")
        return report

    # ── 2. Score landscape per init_variant ─────────────────────────
    print("── (2) score landscape per init_variant ──")
    by_iv: dict[Any, list[dict]] = defaultdict(list)
    for r in ok:
        by_iv[r['init_variant']].append(r)
    iv_scores = {iv: [r['score'] for r in rs] for iv, rs in by_iv.items()}
    section_iv = {}
    for iv in sorted(by_iv.keys(), key=lambda x: str(x)):
        s = _summarise(iv_scores[iv])
        section_iv[str(iv)] = s
        print(f"  {str(iv):40s}  {_fmt_summary(s)}")
    report['sections']['init_variant_scores'] = section_iv
    print()

    # ── 3. Per-parameter effect (Pearson r vs score) ────────────────
    print("── (3) per-parameter effect on score (Pearson r, |r|>0.2 noted) ──")
    param_keys = sorted({k for r in ok for k in r['params'].keys()})
    section_params: dict[str, float | None] = {}
    for k in param_keys:
        xs = [float(r['params'][k]) for r in ok if k in r['params']]
        ys = [float(r['score']) for r in ok if k in r['params']]
        r_val = _pearson(xs, ys)
        section_params[k] = r_val
        flag = ""
        if r_val is None:
            flag = "(undefined)"
        elif abs(r_val) >= 0.5:
            flag = "  ◀── strong"
        elif abs(r_val) >= 0.2:
            flag = "  ◀── moderate"
        rstr = f"{r_val:+.3f}" if r_val is not None else "  n/a"
        print(f"  {k:30s}  r = {rstr}{flag}")
    report['sections']['param_correlations'] = section_params
    print()

    # ── 4. Init-variant differentiation (pairwise KS) ───────────────
    if len(by_iv) > 1:
        print("── (4) init_variant pairwise KS test ──")
        try:
            from scipy.stats import ks_2samp
            ivs = sorted(by_iv.keys(), key=lambda x: str(x))
            section_ks = []
            for i, a in enumerate(ivs):
                for b in ivs[i+1:]:
                    if len(iv_scores[a]) < 3 or len(iv_scores[b]) < 3:
                        continue
                    stat, p = ks_2samp(iv_scores[a], iv_scores[b])
                    note = "  ◀── DIFFERENT" if p < 0.05 else \
                           "  ◀── indistinguishable" if p > 0.5 else ""
                    print(f"  {str(a):20s} vs {str(b):20s}  "
                          f"p={p:.3f}  stat={stat:.3f}{note}")
                    section_ks.append({'a': str(a), 'b': str(b),
                                       'ks_p': float(p),
                                       'ks_stat': float(stat)})
            report['sections']['ks_pairwise'] = section_ks
        except ImportError:
            print("  (scipy not available — skipping)")
        print()

    # ── 5. Seed sensitivity (sd of score per param vector) ──────────
    print("── (5) seed sensitivity (sd of score across seeds, per param vector) ──")
    by_pv: dict[tuple, list[float]] = defaultdict(list)
    for r in ok:
        by_pv[(r['init_variant'], r['pv_idx'])].append(r['score'])
    sds = [statistics.pstdev(ss) for ss in by_pv.values() if len(ss) > 1]
    if sds:
        s = _summarise(sds)
        print(f"  per-cell sd over seeds:  {_fmt_summary(s)}")
        # Show high-variance cells
        worst = sorted(by_pv.items(),
                       key=lambda kv: -(statistics.pstdev(kv[1])
                                        if len(kv[1]) > 1 else 0))[:3]
        for (iv, pv_idx), ss in worst:
            if len(ss) < 2:
                continue
            sd = statistics.pstdev(ss)
            print(f"    iv={iv} pv={pv_idx}: scores={[round(x,3) for x in ss]}  sd={sd:.3f}")
        report['sections']['seed_sd_summary'] = s
    else:
        print("  (only 1 seed — cannot measure)")
    print()

    # ── 6. Numerical anomalies + dead/sat counts ────────────────────
    print("── (6) anomalies ──")
    n_nan = sum(1 for r in ok if r.get('has_nan'))
    n_inf = sum(1 for r in ok if r.get('has_inf'))
    n_dead = sum(1 for r in ok if r['score'] < 0.05)
    n_weak = sum(1 for r in ok if 0.05 <= r['score'] < 0.30)
    n_strong = sum(1 for r in ok if r['score'] >= 0.50)
    print(f"  NaN trials:        {n_nan} / {len(ok)}")
    print(f"  Inf trials:        {n_inf} / {len(ok)}")
    print(f"  dead   (<0.05):    {n_dead} / {len(ok)}")
    print(f"  weak   (0.05-0.30):{n_weak} / {len(ok)}")
    print(f"  strong (≥0.50):    {n_strong} / {len(ok)}")
    if len(report['errors']):
        print(f"  trial errors:      {len(report['errors'])}")
    report['sections']['anomalies'] = {
        'n_nan': n_nan, 'n_inf': n_inf,
        'n_dead': n_dead, 'n_weak': n_weak, 'n_strong': n_strong,
        'n_errors': len(report['errors']),
    }
    print()

    # ── 7. Top/bottom param vectors ─────────────────────────────────
    print("── (7) top 3 / bottom 3 by score ──")
    sorted_records = sorted(ok, key=lambda r: -r['score'])
    print("  TOP:")
    for r in sorted_records[:3]:
        print(f"    score={r['score']:.3f}  iv={r['init_variant']}  "
              f"seed={r['seed']}  alive={r['final_alive']:.2f}")
        print(f"      params={r['params']}")
    print("  BOTTOM:")
    for r in sorted_records[-3:]:
        print(f"    score={r['score']:.3f}  iv={r['init_variant']}  "
              f"seed={r['seed']}  alive={r['final_alive']:.2f}")
        print(f"      params={r['params']}")
    print()

    # ── Summary verdict ─────────────────────────────────────────────
    print("── verdict ──")
    verdict_lines = []
    all_scores = [r['score'] for r in ok]
    overall = _summarise(all_scores)
    if overall['max'] < 0.3:
        verdict_lines.append(f"  ⚠ MAX SCORE {overall['max']:.2f} — rule never produces interesting dynamics")
    elif overall['max'] < 0.5:
        verdict_lines.append(f"  ⚠ max score only {overall['max']:.2f} (below 'interesting' threshold 0.50) — scoring or param range needs review")
    if overall['sd'] < 0.02 and len(ok) >= 5:
        verdict_lines.append(f"  ⚠ SCORE SD {overall['sd']:.4f} — params or scoring saturated")
    if n_dead > 0.5 * len(ok):
        verdict_lines.append(f"  ⚠ {n_dead}/{len(ok)} trials are dead — most of the param space is non-functional")
    if n_nan + n_inf > 0:
        verdict_lines.append(f"  ⚠ {n_nan + n_inf} numerical instabilities")
    # All params have negligible effect on score → broken plumbing or
    # scoring saturated. Skip if too few records or undefined r values.
    pcs = [r for r in section_params.values() if r is not None]
    if pcs and len(pcs) >= 2 and max(abs(r) for r in pcs) < 0.10:
        verdict_lines.append(
            f"  ⚠ no parameter shows |r| ≥ 0.10 with score "
            f"(max |r| = {max(abs(r) for r in pcs):.3f}) — "
            f"params don't move the rule")
    # Multiple init_variants and at least one is dead — variant config
    # is wrong (some inits don't seed dynamics for this rule). Use
    # median (not max) so a single lucky outlier doesn't mask a variant
    # whose typical outcome is the score floor.
    iv_dead = [iv for iv, s in section_iv.items() if s['median'] < 0.20]
    iv_alive = [iv for iv, s in section_iv.items() if s['median'] >= 0.50]
    if iv_dead and iv_alive:
        verdict_lines.append(
            f"  ⚠ init_variant(s) {iv_dead} appear non-functional "
            f"(median < 0.20) while {iv_alive} reach ≥0.50 — wasted variants")
    if not verdict_lines:
        verdict_lines.append("  ✓ no obvious flags — rule appears healthy")
    for line in verdict_lines:
        print(line)
    report['verdict'] = [l.strip() for l in verdict_lines]

    return report


def _cli():
    ap = argparse.ArgumentParser(prog="ca_debug.microscope",
        description="Drill into one CA's behaviour across (seed × params × init).")
    ap.add_argument("rule", help="Rule name (e.g. wireworld_3d)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="Seeds per (param, init) cell (default: 3)")
    ap.add_argument("--params", type=int, default=8,
                    help="Random param vectors to draw (default: 8)")
    ap.add_argument("--size", type=int, default=32,
                    help="Grid size (default: 32)")
    ap.add_argument("--steps", type=int, default=80,
                    help="Steps per trial (default: 80)")
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="Base for seed generation; param RNG uses this seed "
                         "(default: 1000)")
    ap.add_argument("--json", action="store_true",
                    help="Also dump full report as JSON to stdout after "
                         "the human-readable section")
    args = ap.parse_args()

    # Lazy import — touching test_harness creates the GPU context.
    from test_harness import create_headless_context
    window, ctx = create_headless_context()
    try:
        report = microscope(ctx, args.rule,
                            n_seeds=args.seeds, n_params=args.params,
                            size=args.size, steps=args.steps,
                            seed_base=args.seed_base)
    finally:
        try:
            window.destroy()
        except Exception:
            pass
    if args.json:
        print()
        print("── JSON ──")
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    _cli()
