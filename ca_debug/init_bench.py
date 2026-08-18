"""Candidate-init bench: score new init variants before wiring them in.

Adding an init variant used to be a manual ritual — write the function,
guess a preset to attach it to, run the microscope, squint at a median,
and hope.  That ritual missed two failure modes repeatedly:

  1. The microscope samples *random* params.  For rules whose param
     space is mostly dead (GoL B/S thresholds, Lenia growth windows)
     every variant medians 0.000 including the shipped default, so the
     probe returns no signal and a perfectly good init looks dead.

  2. A rule is only interesting inside a finite step window.  Scoring a
     candidate at whatever horizon happened to be configured says
     nothing about the init and everything about the horizon.
     ``lichen_dense`` scores 0.824 at 50 steps and 0.100 at 2500 — it
     was being run at 2500, so every trial was rejected.

This module scores candidates at **default params across several seeds
and several horizons**, which gives signal on dead param spaces, and
prints the score-vs-horizon curve so a mis-horizoned rule is obvious
rather than being misread as a bad init.

Verdicts (per candidate, judged against the rule's own base init):

    pass     peak median >= PASS_BAR and within parity of base.
             -> wire into ``init_variants`` (search will sample it).
    explore  produces real structure but below the bar, or only in a
             narrow window.  -> wire into ``init_extras`` (browsable in
             the GUI, never sampled, costs no search budget).
    dead     floors at every horizon while a sibling scores well.
             -> keep it in init_extras anyway if it looks good on
             screen; scores are not the only reason to keep an init.

The bench also flags the horizon bug directly: if a candidate's peak
horizon is much shorter than the longest probed, it prints a
``mis-horizon`` note and the recommended step count.

Usage::

    python -m ca_debug.init_bench lichen --candidates lichen_territories
    python -m ca_debug.init_bench mycelium --candidates a,b --seeds 5
    python -m ca_debug.init_bench wave_3d --candidates wave_plane \\
        --horizons 50,100,200,400,800 --size 48
    python -m ca_debug.init_bench lenia_multi --candidates x --random-params 8

``--random-params N`` switches to the harsher microscope-style bar
(N random param vectors instead of preset defaults).  Use it for rules
with a broad healthy param space; it is misleading elsewhere.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time
import traceback

DEFAULT_HORIZONS = (50, 100, 200, 400)
DEFAULT_SEEDS = 3
# Canonical scale. Scale-sensitive families (Lenia, dendrite growth) score
# ~0 below this, which would drag candidate and base under the bar alike.
DEFAULT_SIZE = 64

# A candidate must peak at or above this to earn a search slot.
PASS_BAR = 0.50
# Below this, with a healthy sibling, the init is not pulling its weight.
DEAD_BAR = 0.20
# Fraction of the base init's peak a candidate must reach to count as parity.
PARITY_FRAC = 0.90
# Peak must beat the longest horizon by this factor to call it mis-horizoned.
MIS_HORIZON_FACTOR = 2.0


def _score_one(ctx, rule: str, init: str | None, size: int, seed: int,
               steps: int, params=None) -> dict:
    """One trial. Never raises; returns {'score': None, 'error': ...}."""
    from test_harness import run_trial
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_trial(ctx, rule, size=size, seed=seed, steps=steps,
                          params=params, init_override=init)
    except Exception as e:  # noqa: BLE001  probe must survive bad inits
        return {'score': None, 'error': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc()}
    return {
        'score': float(r['score']),
        'median_alive': float(r['median_alive']),
        'mean_activity': float(r['mean_activity']),
        'measure_mode': r.get('measure_mode'),
    }


def _param_vectors(rule: str, n: int, seed: int) -> list:
    """N random param dicts, or [None] to mean 'use preset defaults'."""
    if n <= 0:
        return [None]
    import numpy as np

    from simulator import _resolve_composed_preset
    from test_harness import randomize_params
    preset = _resolve_composed_preset(rule)
    rng = np.random.default_rng(seed)
    return [randomize_params(preset, rng) for _ in range(n)]


def bench_init(ctx, rule: str, init: str | None, *, size: int,
               seeds: list[int], horizons: list[int],
               param_vectors: list) -> dict:
    """Score one init across the seed x horizon (x param) cross-product."""
    by_horizon: dict[int, dict] = {}
    errors: list[str] = []
    for h in horizons:
        scores: list[float] = []
        alives: list[float] = []
        for sd in seeds:
            for pv in param_vectors:
                res = _score_one(ctx, rule, init, size, sd, h, params=pv)
                if res['score'] is None:
                    errors.append(res['error'])
                    continue
                scores.append(res['score'])
                alives.append(res['median_alive'])
        by_horizon[h] = {
            'median': statistics.median(scores) if scores else None,
            'best': max(scores) if scores else None,
            'median_alive': statistics.median(alives) if alives else None,
            'n': len(scores),
        }
    live = {h: v for h, v in by_horizon.items() if v['median'] is not None}
    peak_h = max(live, key=lambda h: live[h]['median']) if live else None
    return {
        'init': init,
        'by_horizon': by_horizon,
        'peak_horizon': peak_h,
        'peak_median': live[peak_h]['median'] if peak_h else None,
        'errors': errors[:3],
        'n_errors': len(errors),
    }


def _verdict(cand: dict, base_peak: float | None,
             horizons: list[int]) -> tuple[str, list[str]]:
    """Return (verdict, notes) for a benched candidate."""
    notes: list[str] = []
    peak = cand['peak_median']
    if peak is None:
        return 'err', ['every trial errored']

    # Mis-horizon: strong early, collapses by the longest horizon probed.
    longest = max(horizons)
    tail = cand['by_horizon'].get(longest, {}).get('median')
    if (tail is not None and peak >= DEAD_BAR
            and cand['peak_horizon'] != longest
            and peak >= MIS_HORIZON_FACTOR * max(tail, 1e-6)):
        notes.append(
            f"mis-horizon: peaks {peak:.3f}@{cand['peak_horizon']} but "
            f"{tail:.3f}@{longest} -- run this rule at "
            f"~{cand['peak_horizon']} steps")

    if peak >= PASS_BAR and (base_peak is None or peak >= PARITY_FRAC * base_peak):
        return 'pass', notes
    if peak < DEAD_BAR and base_peak is not None and base_peak >= PASS_BAR:
        notes.append(f'floors while base reaches {base_peak:.3f}')
        return 'dead', notes
    if base_peak is not None and peak > base_peak:
        notes.append(f'beats base ({base_peak:.3f}) but below {PASS_BAR:.2f} bar')
    return 'explore', notes


def bench(ctx, rule: str, candidates: list[str], *,
          size: int = DEFAULT_SIZE, n_seeds: int = DEFAULT_SEEDS,
          horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
          n_random_params: int = 0, seed_base: int = 1000) -> dict:
    """Bench candidates against the rule's declared base init."""
    from simulator import RULE_PRESETS, _resolve_composed_preset
    if rule not in RULE_PRESETS:
        raise SystemExit(f'unknown rule: {rule!r}')
    preset = _resolve_composed_preset(rule)
    base = preset.get('init')
    seeds = [seed_base + i for i in range(n_seeds)]
    horizons = list(horizons)
    pvs = _param_vectors(rule, n_random_params, seed_base)

    rows = [bench_init(ctx, rule, base, size=size, seeds=seeds,
                       horizons=horizons, param_vectors=pvs)]
    base_peak = rows[0]['peak_median']
    for c in candidates:
        rows.append(bench_init(ctx, rule, c, size=size, seeds=seeds,
                               horizons=horizons, param_vectors=pvs))

    for r in rows[1:]:
        r['verdict'], r['notes'] = _verdict(r, base_peak, horizons)
    rows[0]['verdict'] = 'base'
    rows[0]['notes'] = []
    return {
        'rule': rule,
        'base_init': base,
        'size': size,
        'seeds': seeds,
        'horizons': horizons,
        'n_random_params': n_random_params,
        'rows': rows,
    }


def _print_report(rep: dict) -> None:
    horizons = rep['horizons']
    mode = (f"{rep['n_random_params']} random params"
            if rep['n_random_params'] else 'default params')
    print(f"\ninit bench — {rep['rule']}  (base: {rep['base_init']})")
    print(f"  size={rep['size']}  seeds={len(rep['seeds'])}  {mode}")
    print()
    head = f"{'INIT':<30}" + ''.join(f"{('@' + str(h)):>9}" for h in horizons)
    print(head + f"{'PEAK':>14}  VERDICT")
    print('-' * (len(head) + 25))
    for r in rep['rows']:
        cells = ''
        for h in horizons:
            m = r['by_horizon'].get(h, {}).get('median')
            cells += f"{'  --  ':>9}" if m is None else f"{m:>9.3f}"
        peak = r['peak_median']
        peak_s = '  --  ' if peak is None else f'{peak:.3f}@{r["peak_horizon"]}'
        print(f"{str(r['init']):<30}{cells}{peak_s:>14}  {r['verdict']}")
        for n in r['notes']:
            print(f"{'':<30}  ! {n}")
        if r['n_errors']:
            print(f"{'':<30}  ! {r['n_errors']} errored: {r['errors'][0][:70]}")

    # Actionable summary: exactly what to paste into the preset.
    keep = [r['init'] for r in rep['rows'][1:] if r['verdict'] == 'pass']
    extra = [r['init'] for r in rep['rows'][1:]
             if r['verdict'] in ('explore', 'dead')]
    print()
    if keep:
        print(f"  init_variants += {keep}")
    if extra:
        print(f"  init_extras   += {extra}   (browsable, never sampled)")
    if not keep and not extra:
        print('  nothing to wire.')


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rule')
    ap.add_argument('--candidates', required=True,
                    help='Comma-separated candidate init function names '
                         '(without the init_ prefix, as used in INIT_FUNCS).')
    ap.add_argument('--size', type=int, default=DEFAULT_SIZE)
    ap.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--horizons', default=','.join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument('--random-params', type=int, default=0,
                    help='Sample N random param vectors instead of preset '
                         'defaults (harsher; misleading on dead param spaces).')
    ap.add_argument('--seed-base', type=int, default=1000)
    ap.add_argument('--json', help='Write the report dict to this path.')
    args = ap.parse_args(argv)

    cands = [c.strip() for c in args.candidates.split(',') if c.strip()]
    horizons = [int(h) for h in args.horizons.split(',') if h.strip()]

    from simulator import INIT_FUNCS
    unknown = [c for c in cands if c not in INIT_FUNCS]
    if unknown:
        raise SystemExit(f'unknown init(s), not in INIT_FUNCS: {unknown}')

    from test_harness import create_headless_context
    window, ctx = create_headless_context()  # noqa: F841  keep window alive

    t0 = time.perf_counter()
    rep = bench(ctx, args.rule, cands, size=args.size, n_seeds=args.seeds,
                horizons=horizons, n_random_params=args.random_params,
                seed_base=args.seed_base)
    rep['elapsed_s'] = time.perf_counter() - t0
    _print_report(rep)
    print(f"\n  {rep['elapsed_s']:.1f}s")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rep, f, indent=2)
        print(f'  wrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
