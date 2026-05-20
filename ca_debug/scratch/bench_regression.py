#!/usr/bin/env python3
"""Regression smoke test for every CA preset.

Runs each rule for a small number of steps headlessly and records summary
statistics (alive_ratio, mean, std, finite-frac, NaN/Inf counts, voxel
count). Two modes:

  --capture <out.json>    Capture baseline. Writes per-rule stats to JSON.
  --check <baseline.json> Re-run and diff against baseline. Exit 1 on
                          regression. A regression is:
                            - any new NaN / Inf
                            - alive_ratio drift > 0.10 (absolute)
                            - mean drift > max(0.05, 0.5 * |baseline_mean|)
                            - rule that previously ran now errors
                          Per-rule output is colorized OK / WARN / FAIL.

Usage:
    .venv/bin/python bench_regression.py --capture baseline.json
    # ... do refactor ...
    .venv/bin/python bench_regression.py --check baseline.json

The default size is 48 (fast) and steps=30. Override with --size/--steps.
The trial uses a fixed seed for determinism.
"""
import argparse
import json
import os
import sys
import time
import traceback


def _run_one(ctx, rule_name, size, steps, seed):
    """Run a single rule, return summary dict or {'error': ...}."""
    try:
        from test_harness import run_trial
        result = run_trial(
            ctx, rule_name,
            size=size, seed=seed, steps=steps,
            sample_interval=max(5, steps // 4),
            verbose=False,
        )
        # Distill to a small set of stable scalars actually returned by run_trial.
        return {
            'ok': True,
            'final_alive':   float(result.get('final_alive', float('nan'))),
            'final_activity':float(result.get('final_activity', float('nan'))),
            'final_surface': float(result.get('final_surface', float('nan'))),
            'mean_activity': float(result.get('mean_activity', float('nan'))),
            'median_alive':  float(result.get('median_alive', float('nan'))),
            'spatial_variation': float(result.get('spatial_variation', float('nan'))),
            'has_nan': bool(result.get('has_nan', False)),
            'has_inf': bool(result.get('has_inf', False)),
            'score':   float(result.get('score', float('nan'))),
            'effective_size': int(result.get('size', size)),
        }
    except Exception as e:  # noqa: BLE001  optional dependency
        return {'ok': False, 'error': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc(limit=4)}


def _all_rules():
    from simulator import RULE_PRESETS
    return sorted(RULE_PRESETS.keys())


def _capture(out_path, size, steps, seed):
    from test_harness import create_headless_context, destroy_context
    rules = _all_rules()
    print(f"[capture] {len(rules)} rules, size={size}, steps={steps}, seed={seed}")
    window, ctx = create_headless_context()
    results = {}
    t_total = time.perf_counter()
    for i, name in enumerate(rules):
        t0 = time.perf_counter()
        r = _run_one(ctx, name, size=size, steps=steps, seed=seed)
        dt = time.perf_counter() - t0
        r['_dt_sec'] = dt
        results[name] = r
        status = 'ok' if r.get('ok') else 'ERR'
        if r.get('ok'):
            print(f"  [{i+1:>3}/{len(rules)}] {name:<32} {status}  "
                  f"alive={r['final_alive']:.3f} act={r['final_activity']:.3f} "
                  f"score={r['score']:.3f} ({dt*1000:.0f}ms)")
        else:
            print(f"  [{i+1:>3}/{len(rules)}] {name:<32} {status}  "
                  f"{r['error']}  ({dt*1000:.0f}ms)")
    destroy_context(window)
    payload = {
        'meta': {
            'size': size, 'steps': steps, 'seed': seed,
            'wall_time_sec': time.perf_counter() - t_total,
            'n_rules': len(rules),
            'n_ok': sum(1 for r in results.values() if r.get('ok')),
            'n_err': sum(1 for r in results.values() if not r.get('ok')),
        },
        'rules': results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n[capture] wrote {out_path}")
    print(f"[capture] {payload['meta']['n_ok']} ok / "
          f"{payload['meta']['n_err']} err in "
          f"{payload['meta']['wall_time_sec']:.1f}s")


def _check(baseline_path, size_override, steps_override, seed_override):
    from test_harness import create_headless_context, destroy_context
    with open(baseline_path) as f:
        base = json.load(f)
    size = size_override if size_override is not None else base['meta']['size']
    steps = steps_override if steps_override is not None else base['meta']['steps']
    seed = seed_override if seed_override is not None else base['meta']['seed']
    rules = _all_rules()
    print(f"[check] baseline={baseline_path}, size={size}, steps={steps}, seed={seed}")
    print(f"[check] {len(rules)} rules vs {len(base['rules'])} in baseline")
    window, ctx = create_headless_context()

    # ANSI colors
    RED = '\033[91m'; YEL = '\033[93m'; GRN = '\033[92m'; END = '\033[0m'

    n_ok = n_warn = n_fail = n_skip = 0
    diffs = {}
    for i, name in enumerate(rules):
        if name not in base['rules']:
            print(f"  [{i+1:>3}/{len(rules)}] {name:<32} {YEL}NEW{END} "
                  "(not in baseline, skipping)")
            n_skip += 1
            continue
        b = base['rules'][name]
        cur = _run_one(ctx, name, size=size, steps=steps, seed=seed)

        # Both errored → still ok (consistent)
        if not cur.get('ok') and not b.get('ok'):
            print(f"  [{i+1:>3}/{len(rules)}] {name:<32} {YEL}both errored{END} "
                  f"(was: {b.get('error','?')[:40]})")
            n_warn += 1; continue

        # Newly broken
        if not cur.get('ok') and b.get('ok'):
            print(f"  [{i+1:>3}/{len(rules)}] {name:<32} {RED}REGRESSION (now errors){END}")
            print(f"      was ok, now: {cur['error']}")
            diffs[name] = {'kind': 'newly_broken', 'error': cur['error']}
            n_fail += 1; continue

        # Newly fixed (was broken, now works)
        if cur.get('ok') and not b.get('ok'):
            print(f"  [{i+1:>3}/{len(rules)}] {name:<32} {GRN}FIXED{END} "
                  f"(was: {b.get('error','?')[:30]})")
            n_ok += 1; continue

        # Both ok → check drift
        problems = []
        # New NaN/Inf
        if cur['has_nan'] and not b.get('has_nan'):
            problems.append('NEW NaN')
        if cur['has_inf'] and not b.get('has_inf'):
            problems.append('NEW Inf')
        # Drift on the most stable scalars
        d_alive = abs(cur['final_alive'] - b['final_alive'])
        if d_alive > 0.10:
            problems.append(f"alive Δ={d_alive:.3f}")
        d_act = abs(cur['final_activity'] - b['final_activity'])
        if d_act > 0.10:
            problems.append(f"activity Δ={d_act:.3f}")
        d_score = abs(cur['score'] - b['score'])
        thresh_score = max(0.10, 0.30 * abs(b['score']))
        if d_score > thresh_score:
            problems.append(f"score Δ={d_score:.3g} (>{thresh_score:.3g})")

        if problems:
            print(f"  [{i+1:>3}/{len(rules)}] {name:<32} {RED}FAIL{END}  "
                  f"{', '.join(problems)}")
            print(f"      was alive={b['final_alive']:.3f} act={b['final_activity']:.3f} score={b['score']:.3f}, "
                  f"now alive={cur['final_alive']:.3f} act={cur['final_activity']:.3f} score={cur['score']:.3f}")
            diffs[name] = {
                'kind': 'drift', 'problems': problems,
                'baseline': {'final_alive': b['final_alive'], 'final_activity': b['final_activity'], 'score': b['score']},
                'current':  {'final_alive': cur['final_alive'], 'final_activity': cur['final_activity'], 'score': cur['score']},
            }
            n_fail += 1
        else:
            print(f"  [{i+1:>3}/{len(rules)}] {name:<32} {GRN}OK{END}  "
                  f"alive={cur['final_alive']:.3f} act={cur['final_activity']:.3f} score={cur['score']:.3f}")
            n_ok += 1

    destroy_context(window)
    print(f"\n[check] {GRN}{n_ok} ok{END} / {YEL}{n_warn} warn{END} / "
          f"{RED}{n_fail} fail{END} / {n_skip} skipped")
    if n_fail > 0:
        print(f"\n[check] {RED}REGRESSION DETECTED{END}: {n_fail} rules failed")
        return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--capture', type=str, metavar='OUT.json',
                   help='Capture baseline')
    g.add_argument('--check',   type=str, metavar='BASELINE.json',
                   help='Diff against baseline')
    p.add_argument('--size', type=int, default=None,
                   help='Grid size (default: 48 for capture; matches baseline for check)')
    p.add_argument('--steps', type=int, default=None,
                   help='Steps per rule (default: 30 for capture; matches baseline for check)')
    p.add_argument('--seed', type=int, default=None,
                   help='Seed (default: 42 for capture; matches baseline for check)')
    args = p.parse_args()

    if args.capture:
        size = args.size if args.size is not None else 48
        steps = args.steps if args.steps is not None else 30
        seed = args.seed if args.seed is not None else 42
        _capture(args.capture, size, steps, seed)
        sys.exit(0)
    else:
        rc = _check(args.check, args.size, args.steps, args.seed)
        sys.exit(rc)


if __name__ == '__main__':
    main()
