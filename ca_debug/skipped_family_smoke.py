"""Probe #24 — Smoke test of rule families that all other probes skip.

Probes #22 / #23 / determinism / etc. all skip:
    * viewport (raymarched fractals)
    * agent (langton-ant style head walkers)
    * entity_arena (population dynamics)
    * particle
    * element_ca (sandbox physics)

That leaves a 12-rule blind spot. This probe does a minimal robustness
check on those: for each rule, run a small (size, seed, steps) matrix
and flag NaN, crash, persistent-saturation, or all-zero outputs.

Detected signatures:

    nan         any non-finite output value
    err         crash on at least one (size, seed)
    zero        every channel returned identically 0 across the run
                (rule died or initial condition never propagated)
    constant    every voxel has the same value at end (uniform freeze)
    saturated   max |val| pinned to a clamp bound (1e3 / 255 / 100)
    huge        max |val| > 1e6 without hitting a clamp

Healthy rules return 'ok'. Per-rule severity = worst across the matrix.

Usage:
    python -m ca_debug.skipped_family_smoke
    python -m ca_debug.skipped_family_smoke --sizes 64,128 --seeds 0,42 --steps 60
    python -m ca_debug.skipped_family_smoke --rules sandbox --json /tmp/p24.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4}

_CLAMP_CANDIDATES = (1000.0, 255.0, 100.0, 1.0)
_CLAMP_TOL_FRAC   = 0.01
_HUGE_THRESHOLD   = 1e6


def _gather_rules(args):
    from simulator import RULE_PRESETS, _resolve_composed_preset
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    out = []
    for r in sorted(RULE_PRESETS.keys()):
        try:
            p = _resolve_composed_preset(r)
        except Exception:  # noqa: BLE001
            continue
        if (p.get('kind') == 'viewport'
                or p.get('agent_count')
                or 'entity_arena' in p
                or (p.get('passes') or [{}])[0].get('kind') == 'particle'
                or p.get('is_element_ca')):
            out.append(r)
    return out


def _summarize(g: np.ndarray) -> dict:
    nan = int(np.isnan(g).sum()) + int(np.isinf(g).sum())
    if nan == g.size:
        return {'nan': nan, 'max_abs': float('nan'), 'min': float('nan'),
                'max': float('nan'), 'std': float('nan'),
                'unique_count': 0}
    finite = g[np.isfinite(g)]
    # cheap "unique_count" — coarse rounding to spot uniform/constant fields
    rounded = np.round(finite, 6)
    unique = int(min(np.unique(rounded).size, 1024))
    return {'nan': nan,
            'max_abs': float(np.abs(finite).max()) if finite.size else 0.0,
            'min': float(finite.min()) if finite.size else 0.0,
            'max': float(finite.max()) if finite.size else 0.0,
            'std': float(finite.std()) if finite.size else 0.0,
            'unique_count': unique}


def _saturated_at(stat: dict) -> float | None:
    m = stat['max_abs']
    if not (m == m) or m <= 0:
        return None
    for c in _CLAMP_CANDIDATES:
        if abs(m - c) <= c * _CLAMP_TOL_FRAC:
            return c
    return None


def _run_one(ctx, rule: str, *, size: int, steps: int, seed: int):
    from test_harness import HeadlessRunner
    try:
        r = HeadlessRunner(ctx, rule, size=size, seed=seed)
    except Exception as e:  # noqa: BLE001
        return None, f'init: {type(e).__name__}: {e}'
    try:
        for _ in range(steps):
            r.step()
        g = np.asarray(r.read_grid()).copy()
        return _summarize(g), None
    except Exception as e:  # noqa: BLE001
        return None, f'step: {type(e).__name__}: {e}'
    finally:
        if hasattr(r, 'release'):
            try: r.release()
            except Exception: pass  # noqa: BLE001


def _grade(stat: dict | None, err: str | None) -> tuple[str, str]:
    if err is not None:
        return 'err', err[:60]
    assert stat is not None
    if stat['nan'] > 0:
        return 'crit', f"nan×{stat['nan']}"
    m = stat['max_abs']
    if m != m:  # NaN
        return 'crit', 'nan_all'
    if stat['unique_count'] <= 1:
        # Distinguish zero-everywhere vs constant-nonzero
        if abs(m) < 1e-10:
            return 'high', 'zero'
        return 'med', f'constant={m:.3g}'
    if m > _HUGE_THRESHOLD:
        return 'high', f'huge={m:.2e}'
    sat = _saturated_at(stat)
    if sat is not None and stat['std'] < 1e-6:
        return 'high', f'saturated@{sat}'
    return 'ok', 'ok'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', default='64,128',
                    help='Comma-separated grid sizes.')
    ap.add_argument('--seeds', default='0,42,8675309')
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--rules', help='Override rule subset.')
    ap.add_argument('--json', help='Write full matrix.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER), default='med')
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    import moderngl
    ctx = moderngl.create_standalone_context(require=430)

    rules = _gather_rules(args)
    print(f"\n=== skipped_family_smoke — sizes={sizes} seeds={seeds} "
          f"steps={args.steps} rules={len(rules)} ===\n", file=sys.stderr)

    t0 = time.time()
    all_results = []
    for rule in rules:
        matrix = []
        worst_sev = 'ok'
        worst_msg = ''
        worst_cell = None
        with contextlib.redirect_stdout(io.StringIO()):
            for s in sizes:
                for seed in seeds:
                    stat, err = _run_one(ctx, rule, size=s,
                                         steps=args.steps, seed=seed)
                    sev, msg = _grade(stat, err)
                    matrix.append({'size': s, 'seed': seed, 'sev': sev,
                                   'msg': msg, 'stat': stat, 'err': err})
                    if _SEV_ORDER[sev] < _SEV_ORDER[worst_sev]:
                        worst_sev = sev
                        worst_msg = msg
                        worst_cell = (s, seed)
        all_results.append({'rule': rule, 'status': worst_sev,
                            'msg': worst_msg, 'worst_cell': worst_cell,
                            'matrix': matrix})
        print(f"  {rule:<28}  {worst_sev:<5}  {worst_msg}", file=sys.stderr)

    print()
    all_results.sort(key=lambda r: (_SEV_ORDER[r['status']], r['rule']))
    threshold = _SEV_ORDER[args.severity]
    print(f"{'rule':<28}  {'sev':<5}  {'msg':<24}  worst@(size,seed)")
    print('-' * 80)
    counts = {k: 0 for k in _SEV_ORDER}
    for r in all_results:
        counts[r['status']] += 1
        if _SEV_ORDER[r['status']] > threshold:
            continue
        cell = r['worst_cell']
        cell_s = f'({cell[0]},{cell[1]})' if cell else '-'
        print(f"{r['rule']:<28}  {r['status']:<5}  {r['msg']:<24}  {cell_s}")
    print(f"\n[{time.time()-t0:.1f}s] " +
          ' '.join(f'{k}={v}' for k,v in counts.items() if v))

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'sizes': sizes, 'seeds': seeds, 'steps': args.steps,
                       'results': all_results}, f, indent=2, default=str)
        print(f"wrote {args.json}")

    return 0 if counts['err'] + counts['crit'] + counts['high'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
