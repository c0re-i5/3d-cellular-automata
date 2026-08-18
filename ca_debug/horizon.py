"""Horizon sweep: find the step window where each rule is actually alive.

The horizon bug is a *class* of bug, not a list of incidents.  Its root
cause is structural: the search runs every rule for a single global step
count (500 by default), while the per-rule overrides that correct it
live in ``STEPS_BY_RULE`` inside gitignored shell scripts.  So the one
piece of knowledge that decides whether a rule can score at all is
untracked, unversioned, invisible to the harness and the GUI, and
trivially lost.

Many rules are interesting only inside a finite window and collapse to
a uniform field afterwards.  Past that window the spatial-uniformity
damper drives the score to the 0.100 floor, and since MIN_QUALITY is
0.10 *every* trial is rejected — the rule looks broken when only its
step count was wrong.  ``lichen`` scores 0.824 at 50 steps and 0.100 at
2500, and was being run at 2500.

This module sweeps a wide horizon grid per rule and recommends a
``search_steps`` value to store on the preset itself, which is the
actual fix: one tracked source of truth that the harness reads.

The recommendation is deliberately the *smallest* horizon that still
reaches within ``--tolerance`` of the peak, not the argmax.  Two
reasons: shorter runs cost less search budget, and the channel's
best-performing clips are 6-10s, which is exactly this early window.

Usage::

    python -m ca_debug.horizon --rules lichen,smoothlife_3d
    python -m ca_debug.horizon --horizons 50,100,200,400,800,1600
    python -m ca_debug.horizon --all --seeds 2 --json /tmp/horizon.json
    python -m ca_debug.horizon --all --only-misfit    # just the broken ones
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

DEFAULT_HORIZONS = (50, 100, 200, 500, 1000, 2000)
DEFAULT_SEEDS = 2
DEFAULT_SIZE = 64
# Fraction of peak a shorter horizon must still reach to be preferred.
DEFAULT_TOLERANCE = 0.90
# The global default the search uses when a rule says nothing.
GLOBAL_DEFAULT_STEPS = 500
# A rule is only *misfit* if running it at the global default actually costs
# score. Preferring a shorter horizon slightly is not a bug; losing 30% of
# the achievable score by running past the active window is.
MISFIT_LOSS = 0.70


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS, _resolve_composed_preset
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = []
    for r in sorted(RULE_PRESETS.keys()):
        try:
            preset = _resolve_composed_preset(r)
        except Exception:  # noqa: BLE001
            continue
        if preset.get('kind') == 'viewport':
            continue
        if preset.get('agent_count') or 'entity_arena' in preset:
            continue
        if (preset.get('passes') or [{}])[0].get('kind') == 'particle':
            continue
        if preset.get('particle_count'):
            continue
        if preset.get('audit_skip'):
            continue
        rules.append(r)
    return rules


def sweep_rule(ctx, rule: str, *, horizons: list[int], seeds: list[int],
               size: int, tolerance: float) -> dict:
    """Score one rule across the horizon grid; recommend search_steps."""
    from ca_debug.init_bench import _score_one
    by_h: dict[int, float | None] = {}
    for h in horizons:
        scores = []
        for sd in seeds:
            res = _score_one(ctx, rule, None, size, sd, h)
            if res['score'] is not None:
                scores.append(res['score'])
        by_h[h] = statistics.median(scores) if scores else None

    live = {h: s for h, s in by_h.items() if s is not None}
    if not live:
        return {'rule': rule, 'by_horizon': by_h, 'peak': None,
                'recommend': None, 'verdict': 'err'}

    peak_h = max(live, key=lambda h: live[h])
    peak = live[peak_h]
    # Prefer the earliest horizon that still reaches tolerance * peak.
    rec = min((h for h in sorted(live) if live[h] >= tolerance * peak),
              default=peak_h)
    at_default = live[min(live, key=lambda h: abs(h - GLOBAL_DEFAULT_STEPS))]

    if peak <= 0.105:
        verdict = 'dead'          # floors everywhere; not a horizon problem
    elif at_default >= MISFIT_LOSS * peak:
        verdict = 'ok'            # default horizon loses little; leave it
    elif rec < GLOBAL_DEFAULT_STEPS:
        verdict = 'misfit-short'  # collapses before the default is reached
    else:
        verdict = 'misfit-long'   # still climbing past the default
    return {
        'rule': rule,
        'by_horizon': by_h,
        'peak_horizon': peak_h,
        'peak': peak,
        'recommend': rec,
        'score_at_recommend': live[rec],
        'score_at_default': at_default,
        'loss_at_default': (1.0 - at_default / peak) if peak else 0.0,
        'verdict': verdict,
    }


def _print_report(rows: list[dict], horizons: list[int],
                  only_misfit: bool) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    print(f'horizon sweep — {len(rows)} rules')
    for k in ('misfit-short', 'misfit-long', 'dead', 'ok', 'err'):
        if counts.get(k):
            print(f'    {k:<14} {counts[k]:>3}')
    print()

    shown = [r for r in rows
             if not only_misfit or r['verdict'].startswith('misfit')]
    shown.sort(key=lambda r: (r['verdict'], -(r['peak'] or 0)))

    head = f"{'RULE':<32}" + ''.join(f"{('@' + str(h)):>8}" for h in horizons)
    print(head + f"{'REC':>7} {'LOSS':>6} {'VERDICT':>14}")
    print('-' * (len(head) + 30))
    for r in shown:
        cells = ''
        for h in horizons:
            s = r['by_horizon'].get(h)
            cells += f"{'   --':>8}" if s is None else f"{s:>8.3f}"
        rec = r['recommend']
        loss = r.get('loss_at_default') or 0.0
        print(f"{r['rule']:<32}{cells}{(rec or 0):>7} {loss:>5.0%} "
              f"{r['verdict']:>14}")

    patch = [(r['rule'], r['recommend']) for r in shown
             if r['verdict'].startswith('misfit')]
    if patch:
        print('\n  add to each preset (single tracked source of truth):')
        for rule, rec in sorted(patch, key=lambda t: t[0]):
            print(f"    {rule:<34} 'search_steps': {rec},")


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rules', help='Comma-separated rules.')
    ap.add_argument('--all', action='store_true',
                    help='Sweep every eligible rule (slow).')
    ap.add_argument('--horizons',
                    default=','.join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--size', type=int, default=DEFAULT_SIZE)
    ap.add_argument('--seed-base', type=int, default=1234)
    ap.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument('--only-misfit', action='store_true')
    ap.add_argument('--json')
    args = ap.parse_args(argv)

    if not args.rules and not args.all:
        ap.error('pass --rules or --all')

    horizons = [int(h) for h in args.horizons.split(',') if h.strip()]
    seeds = [args.seed_base + i for i in range(args.seeds)]

    from test_harness import create_headless_context
    window, ctx = create_headless_context()  # noqa: F841  keep window alive

    rules = _select_rules(args)
    rows = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f'\r[{i:>3}/{len(rules)}] {rule:<44}')
        sys.stdout.flush()
        rows.append(sweep_rule(ctx, rule, horizons=horizons, seeds=seeds,
                               size=args.size, tolerance=args.tolerance))
    sys.stdout.write('\r' + ' ' * 72 + '\r')
    elapsed = time.perf_counter() - t0

    _print_report(rows, horizons, args.only_misfit)
    print(f'\n  {elapsed:.1f}s')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'  wrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
