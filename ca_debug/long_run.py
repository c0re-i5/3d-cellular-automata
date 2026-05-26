"""long-run drift probe.

Most rules in the GUI are run for thousands of frames before being
observed.  A rule that's well-behaved over its first 50 frames can
still blow up, NaN out, or numerically drift over the long run --
especially PDE rules with marginally stable integrators, or any rule
that lacks a hard clamp on its state.

This probe steps each rule for many frames (default 1000) at modest
size, sampling the state at regular checkpoints to track:

  * appearance of NaN / Inf
  * unbounded value growth (max-abs ratio against early sample)
  * runaway std (a more sensitive growth signal that ignores any
    single hot spike)

Grades:

    err   crash during construction or stepping.
    crit  NaN or Inf appears at any checkpoint when none was
          present at the first checkpoint.
    high  late-run max-abs > 100x the early-run max-abs, OR
          late-run std > 100x early-run std (the field is growing
          without bound and will eventually overflow fp32).
    med   late-run max-abs > 10x the early-run max-abs, OR
          late-run std > 10x early-run std (suspicious drift, may
          be slowly diverging).
    ok    bounded ratios across the run.
    n/a   rule could not be skipped by the standard filters.

Rules that *die out* (all-zero at end of run) are not flagged --
extinction is a legitimate attractor for many CAs and not a bug.
Rules that *saturate* (reach a stable fixed point near their
operating scale) are not flagged either -- the ratio checks are
about growth, not stasis.

Usage::

    python -m ca_debug.long_run
    python -m ca_debug.long_run --rules lenia_3d,smoothlife_3d
    python -m ca_debug.long_run --size 48 --steps 2000 --severity high
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


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4, 'n/a': 5}


def _read_main(runner) -> np.ndarray:
    return np.asarray(runner.read_grid()).copy()


def _stats(grid: np.ndarray) -> dict:
    flat = grid.reshape(-1)
    n_nan = int((~np.isfinite(flat)).sum())
    n_inf = int(np.isinf(flat).sum())
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {'n_nan': n_nan, 'n_inf': n_inf,
                'max_abs': float('nan'), 'std': float('nan'),
                'mean': float('nan')}
    return {'n_nan': n_nan, 'n_inf': n_inf,
            'max_abs': float(np.max(np.abs(finite))),
            'std': float(finite.std()),
            'mean': float(finite.mean())}


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
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def _probe_rule(ctx, rule: str, size: int, steps: int, seed: int,
                n_samples: int) -> dict:
    from test_harness import HeadlessRunner
    sample_at = [int(round(i * steps / n_samples))
                 for i in range(1, n_samples + 1)]
    sample_set = set(sample_at)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = HeadlessRunner(ctx, rule, size=size, seed=seed)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'phase': 'construct',
                'error': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc()}
    samples: list[dict] = []
    crashed_at: int | None = None
    err_msg: str | None = None
    try:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for step_i in range(1, steps + 1):
                    r.step()
                    if step_i in sample_set:
                        s = _stats(_read_main(r))
                        s['step'] = step_i
                        samples.append(s)
        except Exception as e:  # noqa: BLE001
            crashed_at = step_i
            err_msg = f'{type(e).__name__}: {e}'
    finally:
        try: r.release()
        except Exception: pass  # noqa: BLE001

    out: dict = {'rule': rule, 'size': size, 'steps': steps,
                 'samples': samples}
    if crashed_at is not None:
        out['grade'] = 'err'
        out['phase'] = 'step'
        out['crashed_at'] = crashed_at
        out['error'] = err_msg
        return out
    if not samples:
        out['grade'] = 'err'
        out['error'] = 'no samples collected'
        return out

    # NaN/Inf appearing later when not present at first sample = crit.
    s0 = samples[0]
    later_bad = max(s['n_nan'] + s['n_inf'] for s in samples[1:]) \
        if len(samples) > 1 else 0
    initial_bad = s0['n_nan'] + s0['n_inf']
    if initial_bad == 0 and later_bad > 0:
        out['grade'] = 'crit'
        out['reason'] = f'NaN/Inf appeared after t=0 ({later_bad} cells)'
        return out
    if later_bad > 0 or initial_bad > 0:
        # Was bad from the start -- separate problem, but not a drift bug.
        out['grade'] = 'crit'
        out['reason'] = f'NaN/Inf present throughout (init {initial_bad}, late {later_bad})'
        return out

    # Growth ratios using early sample (first half) vs late sample (last).
    half = max(1, len(samples) // 2)
    early_max = max(s['max_abs'] for s in samples[:half])
    early_std = max(s['std'] for s in samples[:half])
    late_max = samples[-1]['max_abs']
    late_std = samples[-1]['std']
    # Avoid division by zero for rules whose early state is empty.
    eps = 1e-9
    max_ratio = late_max / max(early_max, eps)
    std_ratio = late_std / max(early_std, eps)
    # If the field died (early_max is also ~0), don't flag growth.
    field_floor = 1e-4
    out.update({'early_max': early_max, 'late_max': late_max,
                'early_std': early_std, 'late_std': late_std,
                'max_ratio': max_ratio, 'std_ratio': std_ratio})
    if early_max < field_floor and late_max < field_floor:
        out['grade'] = 'ok'
        out['reason'] = 'rule extinct (field stays ~0) -- not flagged'
        return out
    worst_ratio = max(max_ratio, std_ratio)
    if worst_ratio > 100.0:
        out['grade'] = 'high'
        out['reason'] = (
            f'unbounded growth: max_ratio={max_ratio:.1f}x '
            f'std_ratio={std_ratio:.1f}x')
        return out
    if worst_ratio > 10.0:
        out['grade'] = 'med'
        out['reason'] = (
            f'suspicious drift: max_ratio={max_ratio:.1f}x '
            f'std_ratio={std_ratio:.1f}x')
        return out
    out['grade'] = 'ok'
    return out


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--size', type=int, default=48,
                    help='Grid size (default: 48 to keep total runtime sane).')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--samples', type=int, default=10,
                    help='Number of state snapshots taken across the run.')
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='med',
                    help='Min severity to print (default: med).')
    ap.add_argument('--json', help='Write per-rule report JSON to this path.')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    window, ctx = create_headless_context()

    rules = _select_rules(args)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<42}")
        sys.stdout.flush()
        rows.append(_probe_rule(ctx, rule, args.size, args.steps,
                                args.seed, args.samples))
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    counts = {k: 0 for k in _SEV_ORDER}
    for r in rows:
        counts[r['grade']] = counts.get(r['grade'], 0) + 1

    rows_sorted = sorted(rows, key=lambda r: _SEV_ORDER.get(r['grade'], 9))
    min_sev = _SEV_ORDER[args.severity]

    print(f"long-run probe (size={args.size}, steps={args.steps}, "
          f"samples={args.samples}, seed={args.seed}) -- {elapsed:.1f}s")
    print(f"{'SEV':<6} {'RULE':<42}  NOTES")
    print('-' * 130)
    for r in rows_sorted:
        if _SEV_ORDER.get(r['grade'], 9) > min_sev:
            continue
        parts: list[str] = []
        if r.get('reason'):
            parts.append(r['reason'])
        if r.get('error'):
            parts.append(str(r['error'])[:80])
        if r.get('crashed_at') is not None:
            parts.append(f"crashed at step {r['crashed_at']}")
        note = '  '.join(parts) or '-'
        print(f"{r['grade']:<6} {r['rule']:<42}  {note}")
    summary = '  '.join(f'{k}={counts[k]}' for k in _SEV_ORDER
                        if counts.get(k))
    print(f"\nSummary: {summary}  (n={len(rows)})")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'args': vars(args), 'rows': rows,
                       'elapsed_s': elapsed}, f, indent=2, default=str)
        print(f"Wrote {args.json}")
    return 0 if counts.get('crit', 0) == 0 and counts.get('err', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
