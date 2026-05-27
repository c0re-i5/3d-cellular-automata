"""Param-fuzz probe — random param-combination NaN/Inf hunter.

Probe #21 — Probe #12 (`param_endpoints`) varies ONE slider at a time
to its lo/hi while keeping every other slider at its default.  That
catches single-axis cliffs but misses interaction bugs where two
sliders only blow up together (classic example: a small denominator
slider × a large numerator slider → division-blowup that neither
endpoint test triggers alone).

This probe samples random POINTS inside the ``param_ranges`` box and
runs each for a small number of steps, flagging crashes and NaN/Inf.

Samples per rule:
  - 1 all-lo corner (every slider at its min)
  - 1 all-hi corner (every slider at its max)
  - 1 mid-point   (every slider at (lo+hi)/2)
  - --n random uniform samples in the box, deterministically seeded
    by (rule_name, sample_idx) so the sweep is reproducible.

Skips:
  - kind == 'viewport' (zoom compounds; same reason as Probe #12)
  - rules with no params or no ranges

Grades (per (rule, sample) test):
  err   construction or stepping crashed
  crit  NaN/Inf appeared in the grid within --cap steps
  ok    construction + replay clean

Per-rule grade is the worst sample grade.

Usage::

    python -m ca_debug.param_fuzz
    python -m ca_debug.param_fuzz --rules lichen,morphogen_3d
    python -m ca_debug.param_fuzz --n 16 --cap 20 --size 48
    python -m ca_debug.param_fuzz --severity err --json /tmp/fuzz.json
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
import traceback

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'ok': 2, 'skip': 3}


def _seed_for(rule: str, idx: int) -> int:
    h = hashlib.sha256(f'{rule}::{idx}'.encode()).digest()
    return int.from_bytes(h[:4], 'little')


def _run_once(ctx, rule, size, seed, params, dt, cap):
    from test_harness import HeadlessRunner
    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
        try:
            for _ in range(cap):
                r.step()
            g = np.asarray(r.read_grid())
            return bool(np.isfinite(g).all())
        finally:
            try: r.release()
            except Exception: pass  # noqa: BLE001


def _build_samples(base: dict, ranges: dict, n: int, rule: str):
    """Yield (tag, params_dict) for corner/mid/random samples.

    Only sliders that appear in both `params` and `ranges` are perturbed;
    others stay at their default value.
    """
    keys = [k for k in ranges if k in base
            and isinstance(ranges[k], (tuple, list)) and len(ranges[k]) == 2]

    def make(values_for_key):
        p = dict(base)
        for k in keys:
            p[k] = values_for_key(k)
        return p

    yield 'all_lo', make(lambda k: ranges[k][0])
    yield 'all_hi', make(lambda k: ranges[k][1])
    yield 'mid',    make(lambda k: 0.5 * (ranges[k][0] + ranges[k][1]))

    for i in range(n):
        rng = np.random.default_rng(_seed_for(rule, i))
        def pick(k, _rng=rng):
            lo, hi = ranges[k]
            return float(_rng.uniform(lo, hi))
        yield f'rand{i}', make(pick)


def _probe_rule(ctx, rule: str, size: int, seed: int,
                cap: int, n_random: int) -> dict:
    from simulator import _resolve_composed_preset
    try:
        preset = _resolve_composed_preset(rule)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'reason': f'resolve: {type(e).__name__}: {e}'}

    if preset.get('kind') == 'viewport':
        return {'rule': rule, 'grade': 'skip',
                'reason': 'viewport kind — zoom compounds each step'}

    params = preset.get('params') or {}
    ranges = preset.get('param_ranges') or {}
    if not params or not ranges:
        return {'rule': rule, 'grade': 'skip',
                'reason': 'no params or no ranges declared'}

    dt = preset.get('dt')
    tests: list[dict] = []
    worst = 'ok'
    worst_rank = _SEV_ORDER['ok']
    for tag, p in _build_samples(dict(params), ranges, n_random, rule):
        test = {'tag': tag}
        try:
            finite = _run_once(ctx, rule, size, seed, p, dt, cap)
        except Exception as e:  # noqa: BLE001
            test['grade'] = 'err'
            test['reason'] = f'{type(e).__name__}: {e}'
            test['tb'] = traceback.format_exc().splitlines()[-3:]
            test['params'] = p
        else:
            if not finite:
                test['grade'] = 'crit'
                test['reason'] = f'NaN/Inf within {cap} steps'
                test['params'] = p
            else:
                test['grade'] = 'ok'
        tests.append(test)
        rank = _SEV_ORDER.get(test['grade'], 9)
        if rank < worst_rank:
            worst_rank = rank
            worst = test['grade']
    return {'rule': rule, 'grade': worst,
            'n_tests': len(tests), 'tests': tests}


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = sorted(RULE_PRESETS.keys())
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--size', type=int, default=32,
                    help='Grid size for fuzz runs (default: 32).')
    ap.add_argument('--seed', type=int, default=1001,
                    help='IC seed (params vary across samples; IC is fixed).')
    ap.add_argument('--cap', type=int, default=10,
                    help='Steps per sample (default: 10).')
    ap.add_argument('--n', type=int, default=8,
                    help='Random samples per rule (default: 8). '
                         'Plus 3 deterministic corners (all_lo, all_hi, mid).')
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='crit',
                    help='Min severity to print (default: crit).')
    ap.add_argument('--json', help='Write per-rule report JSON.')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    _window, ctx = create_headless_context()

    rules = _select_rules(args)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<42}")
        sys.stdout.flush()
        rows.append(_probe_rule(ctx, rule, args.size, args.seed,
                                args.cap, args.n))
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    counts = {k: 0 for k in _SEV_ORDER}
    test_counts = {k: 0 for k in _SEV_ORDER}
    total_tests = 0
    for row in rows:
        counts[row['grade']] = counts.get(row['grade'], 0) + 1
        for t in row.get('tests') or []:
            test_counts[t['grade']] = test_counts.get(t['grade'], 0) + 1
            total_tests += 1

    print(f'\nparam-fuzz probe — {len(rules)} rules, '
          f'{total_tests} samples in {elapsed:.1f}s '
          f'(n={args.n} random + 3 corners per rule)')
    print('  by rule (worst sample grade):')
    for g in ('err', 'crit', 'ok', 'skip'):
        print(f'    {g:<5}  {counts.get(g, 0):>5}')
    print('  by sample:')
    for g in ('err', 'crit', 'ok'):
        print(f'    {g:<5}  {test_counts.get(g, 0):>5}')

    sev_cap = _SEV_ORDER[args.severity]
    flagged = [r for r in rows if _SEV_ORDER.get(r['grade'], 9) <= sev_cap]
    if flagged:
        print(f'\nflagged ({args.severity}+):  {len(flagged)} rules')
        for row in flagged:
            print(f'  [{row["grade"]:<4}] {row["rule"]}')
            if row.get('reason'):
                print(f'          {row["reason"]}')
            for t in row.get('tests') or []:
                if _SEV_ORDER.get(t['grade'], 9) > sev_cap:
                    continue
                print(f'          {t["tag"]:<8} {t["grade"]:<4}  '
                      f'{t.get("reason", "")}')
                if 'params' in t:
                    short = {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in t['params'].items()}
                    print(f'                params={short}')

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'rules': rows, 'elapsed_s': elapsed,
                       'size': args.size, 'cap': args.cap,
                       'n_random': args.n, 'seed': args.seed},
                      f, indent=2, default=str)
        print(f'\nwrote {args.json}')

    return 0 if counts.get('err', 0) == 0 and counts.get('crit', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
