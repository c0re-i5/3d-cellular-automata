"""dt-range endpoint sweep probe.

Probe #13 — most rules declare a ``dt_range = (dt_min, dt_max)`` in
the preset, defining the bounds of the GUI's "dt" slider.  The default
``dt`` is well within this range, but the endpoints exercise different
numerical regimes:

  * dt_min → near-quasi-static evolution; some Poisson-clock /
    discrete-event shaders that don't scale per-time misfire here.
  * dt_max → near-CFL-limit timestepping; integrators that aren't
    unconditionally stable can blow up to NaN.

For each rule × endpoint we set ``dt`` to the endpoint, keep every
slider at its preset default, run a small headless replay, and check
for crash or NaN/Inf in the output.

Grades:
  err   construction or stepping crashed.
  crit  NaN/Inf appeared in the grid within --cap steps.
  ok    construction + replay clean.

Note: this DOES NOT check correctness (the dt-convergence probe #3
already covers per-time scaling correctness); it only validates that
the declared dt range is *runnable* — a slider pulled to its endpoint
shouldn't blow up the simulation or crash the engine.

Skips ``kind == 'viewport'`` rules (dt has no semantic role there).

Usage::

    python -m ca_debug.dt_endpoints
    python -m ca_debug.dt_endpoints --cap 50 --severity ok
    python -m ca_debug.dt_endpoints --json /tmp/dtep.json
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
                'reason': 'viewport kind — dt has no semantic role'}

    if 'dt_range' not in preset:
        return {'rule': rule, 'grade': 'skip',
                'reason': 'no dt_range declared (no GUI dt slider)'}

    rng = preset['dt_range']
    if not isinstance(rng, (tuple, list)) or len(rng) != 2:
        return {'rule': rule, 'grade': 'skip',
                'reason': f'malformed dt_range: {rng!r}'}

    lo, hi = float(rng[0]), float(rng[1])
    params = dict(preset.get('params') or {})
    tests: list[dict] = []
    worst = 'ok'
    worst_rank = _SEV_ORDER['ok']
    for tag, val in (('lo', lo), ('hi', hi)):
        test = {'endpoint': tag, 'dt': val}
        try:
            finite = _run_once(ctx, rule, size, seed, params, val, cap)
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
            'dt_range': (lo, hi), 'tests': tests}


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
                    help='Grid size (default: 32).')
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--cap', type=int, default=20,
                    help='Steps per endpoint test (default: 20).')
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

    print(f'\ndt-endpoints probe — {len(rules)} rules, '
          f'{total_tests} dt-endpoint tests in {elapsed:.1f}s')
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
            print(f'  [{row["grade"]:<4}] {row["rule"]}  dt_range={row.get("dt_range")}')
            if row.get('reason'):
                print(f'          {row["reason"]}')
            for t in row.get('tests') or []:
                if _SEV_ORDER.get(t['grade'], 9) > sev_cap:
                    continue
                print(f'          [{t["grade"]:<4}] dt@{t["endpoint"]}={t["dt"]}  {t.get("reason", "")}')

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
