"""Param-range endpoint sweep probe.

Probe #12 — each rule preset declares ``param_ranges`` defining the
min/max value of every slider exposed in the GUI.  A common bug class
is that an endpoint produces a degenerate run: divide-by-zero when a
denominator slider hits 0, overflow when an exponent goes to the max,
or NaN propagation from a clamped quantity that wasn't sanitised.

For each (rule × slider × endpoint) we set that one slider to its
endpoint value (lo and hi), keep every other slider at its preset
default, construct a small headless runner, step ``--cap`` frames,
and check for crash or NaN/Inf in the output state.

Grades (per (rule, param, endpoint) test):
  err   construction or stepping crashed.
  crit  NaN/Inf appeared in the grid within --cap steps.
  ok    construction + replay clean.

Per-rule grade is the worst test grade.

This complements Probe #10 (param-coherence): #10 checks that every
slider is wired to a shader uniform; #12 checks that every slider's
declared range is safe to use.  Together they validate the GUI/engine
contract end-to-end on the param-name surface.

Scope: voxel/agent/entity_arena/particle rules all work (we just pass
the params dict to HeadlessRunner — every rule's runner accepts it).
Skips ``kind == 'viewport'`` rules because endpoint zoom-rates compound
multiplicatively each step and produce degenerate views unrelated to
the slider being tested.

Usage::

    python -m ca_debug.param_endpoints
    python -m ca_debug.param_endpoints --rules nca_3d,lichen
    python -m ca_debug.param_endpoints --cap 20 --size 48
    python -m ca_debug.param_endpoints --severity err --json /tmp/pep.json
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

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'ok': 2, 'skip': 3}


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


def _probe_rule(ctx, rule: str, size: int, seed: int,
                cap: int) -> dict:
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
    base = dict(params)
    tests: list[dict] = []
    worst = 'ok'
    worst_rank = _SEV_ORDER['ok']
    for name, rng in ranges.items():
        if name not in params:
            # range declared for a key not in params — odd but skip;
            # Probe #10 would have caught this if it mattered.
            continue
        if not isinstance(rng, (tuple, list)) or len(rng) != 2:
            continue
        lo, hi = rng
        for tag, val in (('lo', lo), ('hi', hi)):
            test_params = dict(base)
            test_params[name] = val
            test = {'param': name, 'endpoint': tag, 'value': val}
            try:
                finite = _run_once(ctx, rule, size, seed,
                                   test_params, dt, cap)
            except Exception as e:  # noqa: BLE001
                test['grade'] = 'err'
                test['reason'] = f'{type(e).__name__}: {e}'
                test['tb'] = traceback.format_exc().splitlines()[-3:]
            else:
                if not finite:
                    test['grade'] = 'crit'
                    test['reason'] = f'NaN/Inf within {cap} steps'
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
                    help='Grid size for endpoint sweeps (default: 32).')
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--cap', type=int, default=10,
                    help='Steps per endpoint test (default: 10).')
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
        rows.append(_probe_rule(ctx, rule, args.size, args.seed, args.cap))
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

    print(f'\nparam-endpoints probe — {len(rules)} rules, '
          f'{total_tests} endpoint tests in {elapsed:.1f}s')
    print('  by rule (worst test grade):')
    for g in ('err', 'crit', 'ok', 'skip'):
        print(f'    {g:<5}  {counts.get(g, 0):>5}')
    print('  by test:')
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
                print(f'          [{t["grade"]:<4}] {t["param"]} @{t["endpoint"]}={t["value"]}  {t.get("reason", "")}')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'counts': counts, 'test_counts': test_counts,
                       'rows': rows, 'elapsed_s': elapsed,
                       'size': args.size, 'cap': args.cap},
                      fh, indent=2, default=str)
        print(f'\nwrote {args.json}')

    return 1 if (counts['err'] + counts['crit']) else 0


if __name__ == '__main__':
    sys.exit(main())
