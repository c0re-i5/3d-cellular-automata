"""Determinism probe.

Run each rule twice from the same seed / params / size for K steps and
compare the final grid (pair1 + pair2 when present).  On the same GPU
with the same shader and the same input, a CA evolution should be
bit-identical: differences betray race conditions in compute shaders
(non-atomic neighbour writes, missing ``barrier()`` in shared-memory
variants, multi-pass ordering bugs, ping-pong texture confusion, etc.).

We grade by ``max |diff|`` across all channels and pairs:

    crit   max_diff > 1e-2     (visibly different final state)
    high   max_diff > 1e-4     (small but persistent drift)
    med    max_diff > 1e-6     (last few mantissa bits only -- still a race)
    ok     max_diff == 0       (bit-identical, as expected)
    err    crash / NaN / size mismatch

Cells that are NaN in BOTH runs are treated as identical (NaN != NaN
under direct comparison, but a NaN at the *same* cell across two
identical-seed runs is itself deterministic).

Usage:
    python -m ca_debug.determinism
    python -m ca_debug.determinism --rules game_of_life_3d,lenia_3d
    python -m ca_debug.determinism --size 48 --steps 50 --json /tmp/det.json
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


# Severity buckets.  Order matters: lower index = more severe.
_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4, 'n/a': 5}


def _grade(max_diff: float, nan_only_a: int, nan_only_b: int) -> str:
    """Bucket the max-diff number; NaN-asymmetry forces 'crit'."""
    if nan_only_a + nan_only_b > 0:
        return 'crit'
    if max_diff == 0.0:
        return 'ok'
    if max_diff > 1e-2:
        return 'crit'
    if max_diff > 1e-4:
        return 'high'
    if max_diff > 1e-6:
        return 'med'
    return 'ok'


def _read_full_state(runner) -> list[np.ndarray]:
    """Return [pair1, pair2?] grids as numpy arrays.

    pair2 is included when the harness has bound it (multi-field rules).
    """
    grids = [np.asarray(runner.read_grid()).copy()]
    # pair2 isn't exposed by a method; read it raw if allocated.
    tex_a2 = getattr(runner, 'tex_a2', None)
    if tex_a2 is not None:
        try:
            src = (tex_a2 if getattr(runner, 'ping2', 0) == 0
                   else runner.tex_b2)
            raw = np.frombuffer(src.read(), dtype=runner._tex_np_dtype).reshape(
                runner.size, runner.size, runner.size, 4)
            grids.append(raw.astype(np.float32, copy=True))
        except Exception:  # noqa: BLE001  optional pair2 readback, never fatal
            pass
    return grids


def _diff_pair(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two same-shape grids; return diff stats.

    NaN-vs-NaN at the same cell is treated as equal; NaN-vs-finite is
    counted as a divergence."""
    if a.shape != b.shape:
        return {'shape_mismatch': (a.shape, b.shape),
                'max_diff': float('inf'),
                'n_diff_cells': int(np.prod(a.shape)),
                'nan_only_a': 0, 'nan_only_b': 0}
    nan_a = ~np.isfinite(a)
    nan_b = ~np.isfinite(b)
    nan_only_a = int((nan_a & ~nan_b).sum())
    nan_only_b = int((~nan_a & nan_b).sum())
    both_finite = ~nan_a & ~nan_b
    if both_finite.any():
        d = np.abs(a[both_finite] - b[both_finite])
        max_diff = float(d.max())
        # Count cells with ANY channel differing (channel is last axis).
        diff_mask = (a != b) & both_finite
        # Reduce-any across channel axis (axis -1) for the cell count.
        n_diff = int(np.any(diff_mask, axis=-1).sum())
    else:
        max_diff = 0.0
        n_diff = 0
    return {'max_diff': max_diff,
            'n_diff_cells': n_diff,
            'nan_only_a': nan_only_a,
            'nan_only_b': nan_only_b}


def _run_once(ctx, rule: str, *, size: int, steps: int, seed: int
              ) -> list[np.ndarray]:
    from test_harness import HeadlessRunner
    r = HeadlessRunner(ctx, rule, size=size, seed=seed)
    try:
        for _ in range(steps):
            r.step()
        return _read_full_state(r)
    finally:
        rel = getattr(r, 'release', None)
        if callable(rel):
            try: rel()
            except Exception: pass  # noqa: BLE001  GL release, never fatal


def _run_pair(ctx, rule: str, args) -> dict:
    """Two back-to-back runs with identical inputs; diff the outputs."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run_a = _run_once(ctx, rule, size=args.size,
                              steps=args.steps, seed=args.seed)
            run_b = _run_once(ctx, rule, size=args.size,
                              steps=args.steps, seed=args.seed)
    except Exception as e:  # noqa: BLE001  per-rule trial may crash, record error and continue
        return {'error': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc(),
                'grade': 'err', 'max_diff': float('nan'), 'pairs': []}
    if len(run_a) != len(run_b):
        return {'error': f'pair count mismatch: {len(run_a)} vs {len(run_b)}',
                'grade': 'err', 'max_diff': float('nan'), 'pairs': []}
    pairs = []
    max_diff = 0.0
    nan_only_a_total = 0
    nan_only_b_total = 0
    for i, (a, b) in enumerate(zip(run_a, run_b)):
        s = _diff_pair(a, b)
        s['pair_idx'] = i
        pairs.append(s)
        max_diff = max(max_diff, s['max_diff'])
        nan_only_a_total += s['nan_only_a']
        nan_only_b_total += s['nan_only_b']
    return {'grade': _grade(max_diff, nan_only_a_total, nan_only_b_total),
            'max_diff': max_diff,
            'pairs': pairs}


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS, _resolve_composed_preset
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = []
    for r in sorted(RULE_PRESETS.keys()):
        try:
            preset = _resolve_composed_preset(r)
        except Exception:  # noqa: BLE001  preset lookup failure -> caller falls back
            continue
        if preset.get('kind') == 'viewport':
            continue
        if preset.get('agent_count') or 'entity_arena' in preset:
            continue
        if (preset.get('passes') or [{}])[0].get('kind') == 'particle':
            continue
        rules.append(r)
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def main(argv=None):
    # Allow the harness to honour our requested size instead of auto-
    # bumping to the preset's default_size; non-deterministic rules
    # often differ only at small grids where racing-cells overlap.
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--size', type=int, default=32)
    ap.add_argument('--steps', type=int, default=30)
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
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<40}")
        sys.stdout.flush()
        row = _run_pair(ctx, rule, args)
        row['rule'] = rule
        rows.append(row)
    sys.stdout.write('\r' + ' ' * 60 + '\r')
    elapsed = time.perf_counter() - t0

    counts = {k: 0 for k in _SEV_ORDER}
    for r in rows:
        counts[r['grade']] = counts.get(r['grade'], 0) + 1
    min_sev = _SEV_ORDER[args.severity]
    rows_sorted = sorted(rows,
                         key=lambda r: (_SEV_ORDER.get(r['grade'], 9),
                                        -(r.get('max_diff') or 0)))

    print(f"Determinism probe (size={args.size}, steps={args.steps}, "
          f"seed={args.seed}) -- {elapsed:.1f}s")
    print(f"{'SEV':<6} {'RULE':<38} {'MAX_DIFF':>12}  {'N_DIFF':>10}  NOTES")
    print('-' * 96)
    for r in rows_sorted:
        sev = r['grade']
        if _SEV_ORDER.get(sev, 9) > min_sev:
            continue
        md = r.get('max_diff')
        md_s = '   nan' if md != md else f'{md:12.4g}'   # noqa: PLR0124  NaN check
        n_diff = sum(p.get('n_diff_cells', 0) for p in r.get('pairs', []))
        notes = ''
        if r.get('error'):
            notes = r['error'][:60]
        else:
            nan_a = sum(p.get('nan_only_a', 0) for p in r.get('pairs', []))
            nan_b = sum(p.get('nan_only_b', 0) for p in r.get('pairs', []))
            if nan_a or nan_b:
                notes = f'asym-NaN a={nan_a} b={nan_b}'
        print(f"{sev:<6} {r['rule']:<38} {md_s}  {n_diff:>10}  {notes}")
    summary = '  '.join(f'{k}={counts[k]}' for k in ('crit', 'high', 'med', 'ok', 'err')
                        if counts.get(k))
    print(f"\nSummary: {summary or 'all-ok'}  (n={len(rows)})")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'args': vars(args), 'rows': rows, 'elapsed_s': elapsed},
                      f, indent=2, default=str)
        print(f"Wrote {args.json}")
    return 0 if counts.get('crit', 0) == 0 and counts.get('err', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
