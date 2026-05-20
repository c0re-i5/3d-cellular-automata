"""Cross-size sweep of every crystal_* preset.

For each (rule, size), spin up a headless runner, advance N steps, then
report final-state stats for ALL channels (phase φ, supersaturation,
orientation, trapped solute) plus a few mid-run snapshots so we can see
the growth trajectory, not just the endpoint.

Run:  .venv/bin/python scripts/sweep_crystals.py [--steps 1500] [--sizes 64,96,128]

Output is a single table per rule + a global summary that flags rules whose
final stats drift more than ε across grid sizes (= scale-variance bug).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Allow running from repo root or scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from test_harness import create_headless_context, HeadlessRunner  # noqa: E402
from simulator import RULE_PRESETS  # noqa: E402


CRYSTAL_RULES = [r for r in RULE_PRESETS if r.startswith('crystal_')]

# Channel names by shader (what the preset exposes as vis_channels).
CHANNEL_LABELS = {
    'crystal_growth':         ['phi',  'sup',  'orient', 'solute'],
    'dielectric_breakdown':   ['cell', 'phi_e', 'age',   'gradn'],
}


def channel_stats(grid: np.ndarray) -> dict:
    """Per-channel summary stats. Grid shape (Z, Y, X, 4)."""
    out = {}
    for c in range(grid.shape[-1]):
        v = grid[..., c]
        finite = np.isfinite(v)
        n_nan = int((~np.isfinite(v) & np.isnan(v)).sum())
        n_inf = int((~finite & ~np.isnan(v)).sum())
        if finite.all():
            out[c] = {
                'mean': float(v.mean()),
                'std':  float(v.std()),
                'min':  float(v.min()),
                'max':  float(v.max()),
                'nan': 0, 'inf': 0,
            }
        else:
            v2 = v[finite]
            out[c] = {
                'mean': float(v2.mean()) if v2.size else float('nan'),
                'std':  float(v2.std())  if v2.size else float('nan'),
                'min':  float(v2.min())  if v2.size else float('nan'),
                'max':  float(v2.max())  if v2.size else float('nan'),
                'nan': n_nan, 'inf': n_inf,
            }
    return out


def overall_stats(grid: np.ndarray, threshold: float = 0.5) -> dict:
    """Whole-grid descriptors derived from channel 0 (phase / cluster).

    Mirrors the simulator's debug shader so numbers are comparable to
    the JSON dumps in debug_runs/.
    """
    phi = grid[..., 0]
    alive = phi > threshold
    n_alive = int(alive.sum())
    total = int(phi.size)
    out = {
        'active_frac': n_alive / total if total else 0.0,
        'rg':          float('nan'),
        'com':         (float('nan'),) * 3,
        'boundary_frac': 0.0,
    }
    if n_alive == 0:
        return out

    z, y, x = np.nonzero(alive)
    sz_z, sz_y, sz_x = phi.shape
    # Center of mass (normalized to [0, 1])
    com = (x.mean() / sz_x, y.mean() / sz_y, z.mean() / sz_z)
    out['com'] = com
    # Radius of gyration (normalized)
    cx, cy, cz = x.mean(), y.mean(), z.mean()
    rg2 = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2).mean()
    out['rg'] = float(np.sqrt(rg2)) / max(sz_x, sz_y, sz_z)
    # Fraction of ALIVE cells that sit in the 4-voxel shell next to any
    # face. This is what we care about ("is the crystal hitting the wall?")
    # rather than "how big a fraction of total volume is the shell?" (which
    # is purely geometric and trivially nonzero when the grid is full).
    shell = 4
    on_bnd = (
        (x < shell) | (x >= sz_x - shell)
        | (y < shell) | (y >= sz_y - shell)
        | (z < shell) | (z >= sz_z - shell)
    )
    out['boundary_frac'] = float(on_bnd.mean())
    # Saturation flag: ratio of cells alive vs. cells alive in shell —
    # if the alive set is the whole grid this is the geometric shell ratio.
    return out


def fmt_row(label: str, snapshot: dict, ch: dict) -> str:
    com = snapshot['com']
    return (f'    {label:>8} '
            f'act={snapshot["active_frac"]:.4f} '
            f'rg={snapshot["rg"]:.3f} '
            f'com=({com[0]:.2f},{com[1]:.2f},{com[2]:.2f}) '
            f'bnd={snapshot["boundary_frac"]:.4f}  '
            f'phi:m={ch[0]["mean"]:.3f}/s={ch[0]["std"]:.3f}/x={ch[0]["max"]:.3f}  '
            f'sup:m={ch[1]["mean"]:.3f}/s={ch[1]["std"]:.3f}  '
            f'sol:m={ch[3]["mean"]:.3f}/s={ch[3]["std"]:.3f}  '
            f'nan={sum(ch[c]["nan"] for c in ch)} '
            f'inf={sum(ch[c]["inf"] for c in ch)}')


def run_one(ctx, rule: str, size: int, steps: int, seed: int,
            sample_steps=None) -> dict:
    """Run one (rule, size) and return final + intermediate snapshots.

    `sample_steps` is an explicit list of step indices to snapshot at
    (defaults to a log-spaced sequence — early dynamics have lots of
    interesting behaviour that linear sampling misses).
    """
    if sample_steps is None:
        # Log-spaced 6 samples between 25 and `steps`, plus the final step.
        ss = np.unique(np.round(np.geomspace(25, steps, 6)).astype(int))
        sample_steps = sorted(set(ss.tolist()) | {steps})
    t0 = time.time()
    runner = HeadlessRunner(ctx, rule, size=size, seed=seed)
    snapshots = []  # [(step, overall, channel_stats)]
    next_idx = 0
    for i in range(1, steps + 1):
        runner.step()
        if next_idx < len(sample_steps) and i == sample_steps[next_idx]:
            g = runner.read_grid()
            snapshots.append((i, overall_stats(g), channel_stats(g)))
            next_idx += 1
    t_run = time.time() - t0
    runner.release()
    return {'rule': rule, 'size': size, 'steps': steps, 'seed': seed,
            'snapshots': snapshots, 't_run': t_run}


def print_run(result: dict):
    rule, size, steps = result['rule'], result['size'], result['steps']
    print(f'  --- {rule}  size={size}  steps={steps}  ({result["t_run"]:.1f}s) ---')
    for step, overall, ch in result['snapshots']:
        print(fmt_row(f'@{step}', overall, ch))


def diagnose(rule: str, by_size: dict) -> list[str]:
    """Look at final snapshots across sizes and tag obvious issues."""
    flags = []
    finals = {sz: r['snapshots'][-1] for sz, r in by_size.items() if r['snapshots']}
    if len(finals) < 2:
        return flags
    sizes = sorted(finals)

    # 1. Saturation: phi mean ≈ 1 and phi std ≈ 0  ⇒  whole grid is solid
    sat_sizes = [sz for sz in sizes
                 if finals[sz][2][0]['std'] < 1e-3 and finals[sz][2][0]['mean'] > 0.99]
    if sat_sizes:
        flags.append(f'SATURATED at sizes {sat_sizes}: grid 100% solid, no remaining interface')

    # 2. NaN/Inf
    bad = [sz for sz in sizes
           if any(finals[sz][2][c]['nan'] + finals[sz][2][c]['inf']
                  for c in finals[sz][2])]
    if bad:
        flags.append(f'NaN/Inf at sizes {bad}')

    # 3. Active-fraction drift across sizes
    afs = [finals[sz][1]['active_frac'] for sz in sizes]
    if max(afs) > 0:
        spread = max(afs) - min(afs)
        if spread > 0.05:
            af_str = ', '.join(f'{sz}:{af:.3f}' for sz, af in zip(sizes, afs))
            flags.append(f'active_frac scale-drift {spread:.3f}  ({af_str})')

    # 4. Boundary touched (crystal hits the wall) — only meaningful when
    # the crystal hasn't already filled the box. Compare actual shell
    # occupancy to what you'd get if the crystal were uniformly distributed
    # over the whole volume (= same shell fraction by construction). If
    # the actual fraction is BIGGER than expected, the crystal really has
    # grown preferentially towards the walls.
    for sz in sizes:
        af  = finals[sz][1]['active_frac']
        bf  = finals[sz][1]['boundary_frac']
        # Geometric shell volume fraction for a 4-voxel shell on this grid:
        # 1 - (sz-8)**3 / sz**3   (assuming sz > 8)
        shell_geom = 1.0 - max(0, (sz - 8)) ** 3 / sz ** 3 if sz > 8 else 1.0
        if af > 0.001 and af < 0.95 and bf > shell_geom * 1.5:
            flags.append(
                f'size {sz}: crystal preferentially touches wall — '
                f'shell-occupancy {bf:.3f} vs geom {shell_geom:.3f} '
                f'(active={af:.3f})')

    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', type=str, default=','.join(CRYSTAL_RULES),
                    help='Comma-separated rule names (default: all crystal_*)')
    ap.add_argument('--sizes', type=str, default='64,96,128',
                    help='Comma-separated grid sizes')
    ap.add_argument('--steps', type=int, default=400,
                    help='Steps per (rule, size). Crystal growth saturates '
                         'fast at default params; 400 typically captures the '
                         'whole growth-then-freeze trajectory.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rules = [r.strip() for r in args.rules.split(',') if r.strip()]
    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]

    print(f'Sweep: rules={rules}  sizes={sizes}  steps={args.steps}  seed={args.seed}')
    print('=' * 100)

    win, ctx = create_headless_context()
    try:
        all_results = {}
        for rule in rules:
            print(f'\n========== {rule} ==========')
            by_size = {}
            for sz in sizes:
                res = run_one(ctx, rule, sz, args.steps, args.seed)
                by_size[sz] = res
                print_run(res)
            all_results[rule] = by_size
            flags = diagnose(rule, by_size)
            if flags:
                print(f'  ⚠ FLAGS:')
                for f in flags:
                    print(f'     - {f}')
            else:
                print(f'  ✓ clean (no scale-variance / saturation / NaN issues)')

        # Final compact summary
        print('\n' + '=' * 100)
        print('SUMMARY')
        print('=' * 100)
        print(f'{"rule":<22}  ' + ' '.join(f'{sz:>30d}' for sz in sizes))
        sub = '  '.join(f'{"act/std/bnd/sol":>30s}' for _ in sizes)
        print(f'{"":<22}  {sub}')
        for rule in rules:
            br = all_results[rule]
            cells = []
            for sz in sizes:
                snaps = br[sz]['snapshots']
                if not snaps:
                    cells.append(f'{"-":>30s}')
                    continue
                _, ov, ch = snaps[-1]
                cells.append(f'{ov["active_frac"]:.3f}/{ch[0]["std"]:.3f}/'
                             f'{ov["boundary_frac"]:.3f}/{ch[3]["mean"]:.3f}'.rjust(30))
            print(f'{rule:<22}  ' + ' '.join(cells))
    finally:
        try:
            ctx.release()
        except Exception:  # noqa: BLE001  GL resource release, never fatal
            pass


if __name__ == '__main__':
    main()
