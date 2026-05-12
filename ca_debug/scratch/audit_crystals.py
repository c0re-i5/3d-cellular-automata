#!/usr/bin/env python3
"""Thorough geometric audit of every crystal_* preset.

For each rule we run multiple seeds at multiple sizes through the headless
runner, then read back the full grid at log-spaced steps and compute a
suite of *shape* descriptors that distinguish a Wulff blob from a faceted
polyhedron from a branched dendrite. The motivating observation: the
debug-stats summary (active_frac, rg, com) can't tell a sphere from a
cube of the same volume — they both look "compact". We need shape, not
just size.

Metrics per snapshot (all dimensionless):

  active_frac       : N_alive / N_total                          (size)
  sphericity        : rg / rg_of_sphere(N_alive)                 (1.0 = ball)
  bbox_fill         : N_alive / bbox_volume                      (1.0 = solid box; <0.3 = sparse)
  interface_frac    : (cells with phi in (0.05, 0.95)) / N_alive (~N^-1/3 compact, >0.4 branched)
  axis_ratio        : λ_max/λ_min eigenvalue ratio of inertia    (1.0 = isotropic, >2 = plate/needle)
  axis_planarity    : (λ_mid - λ_min) / λ_max                    (high = plate, low = needle/blob)
  envelope_octa     : <100>/<111> growth ratio (axis vs corner)  (>1 = octahedral; <1 = cubic)

Plus: similarity matrix between rules. Two rules count as the "same
shape" if their (sphericity, bbox_fill, interface_frac) tuples agree
within tolerance at matched active_frac.

Run:  .venv/bin/python scripts/audit_crystals.py [--sizes 64,96] [--seeds 42,7,123]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from test_harness import create_headless_context, HeadlessRunner  # noqa: E402
from simulator import RULE_PRESETS  # noqa: E402


CRYSTAL_RULES = [r for r in RULE_PRESETS if r.startswith('crystal_')]


# -------- shape descriptors --------------------------------------------------

def shape_stats(grid: np.ndarray, threshold: float = 0.5) -> dict:
    """All shape descriptors for one snapshot. grid[..., 0] = phi."""
    phi = grid[..., 0]
    sz_z, sz_y, sz_x = phi.shape
    sz = max(sz_x, sz_y, sz_z)
    N_total = phi.size
    alive = phi > threshold
    n = int(alive.sum())
    out = dict(
        active_frac=n / N_total,
        sphericity=float('nan'),
        bbox_fill=float('nan'),
        interface_frac=float('nan'),
        axis_ratio=float('nan'),
        axis_planarity=float('nan'),
        envelope_octa=float('nan'),
        rg=float('nan'),
    )
    if n < 5:
        return out

    z, y, x = np.nonzero(alive)
    cx, cy, cz = x.mean(), y.mean(), z.mean()

    # rg in voxels then normalized to box size
    rg_vox = float(np.sqrt(((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2).mean()))
    out['rg'] = rg_vox / sz

    # SPHERICITY: rg(actual) / rg(sphere of same N)
    R_vox = (3.0 * n / (4.0 * math.pi)) ** (1.0 / 3.0)
    rg_sphere_vox = math.sqrt(3.0 / 5.0) * R_vox
    out['sphericity'] = rg_vox / rg_sphere_vox if rg_sphere_vox > 0 else float('nan')

    # BBOX FILL
    bbox_vol = max(1, (x.max() - x.min() + 1) * (y.max() - y.min() + 1)
                   * (z.max() - z.min() + 1))
    out['bbox_fill'] = n / bbox_vol

    # INTERFACE FRACTION (uses the diffuse phi field, no histogram)
    interface_mask = (phi > 0.05) & (phi < 0.95)
    bulk = (phi >= 0.95).sum()
    iface = int(interface_mask.sum())
    if (iface + bulk) > 0:
        out['interface_frac'] = iface / (iface + bulk)

    # INERTIA TENSOR (mass-weighted): use the full phi field (continuous)
    # so smooth interface contributes proportionally.
    coords_x = (np.arange(sz_x, dtype=np.float32) - cx)
    coords_y = (np.arange(sz_y, dtype=np.float32) - cy)
    coords_z = (np.arange(sz_z, dtype=np.float32) - cz)
    # Mass = phi (clamped to alive region only to keep it cheap)
    m = phi[alive].astype(np.float64)
    dx = coords_x[x]; dy = coords_y[y]; dz = coords_z[z]
    Ixx = float((m * (dy * dy + dz * dz)).sum())
    Iyy = float((m * (dx * dx + dz * dz)).sum())
    Izz = float((m * (dx * dx + dy * dy)).sum())
    Ixy = float(-(m * dx * dy).sum())
    Ixz = float(-(m * dx * dz).sum())
    Iyz = float(-(m * dy * dz).sum())
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]], dtype=np.float64)
    eigs = np.sort(np.linalg.eigvalsh(I))  # ascending
    lam_min, lam_mid, lam_max = float(eigs[0]), float(eigs[1]), float(eigs[2])
    if lam_max > 1e-9:
        out['axis_ratio'] = lam_max / max(lam_min, 1e-9)
        out['axis_planarity'] = (lam_mid - lam_min) / lam_max

    # ENVELOPE_OCTA: ratio of how far the alive set extends along the
    # cube AXES (max(|x|,|y|,|z|)) vs along the cube DIAGONALS (|x|+|y|+|z|/√3).
    # Octahedral crystals reach further along axes (>1), cubic crystals
    # reach further along diagonals (<1), spheres ≈ 1.
    axis_reach = max(
        np.abs(x - cx).max(),
        np.abs(y - cy).max(),
        np.abs(z - cz).max(),
    )
    diag_reach = (np.abs(x - cx) + np.abs(y - cy) + np.abs(z - cz)).max() / math.sqrt(3.0)
    if diag_reach > 0:
        out['envelope_octa'] = axis_reach / diag_reach

    return out


# -------- run + collect ------------------------------------------------------

def run_one(ctx, rule: str, size: int, steps: int, seed: int,
            sample_steps: Sequence[int]):
    runner = HeadlessRunner(ctx, rule, size=size, seed=seed)
    snaps = []
    next_idx = 0
    for i in range(1, steps + 1):
        runner.step()
        if next_idx < len(sample_steps) and i == sample_steps[next_idx]:
            g = runner.read_grid()
            snaps.append((i, shape_stats(g)))
            next_idx += 1
    runner.release()
    return snaps


# -------- output -------------------------------------------------------------

def fmt_stats(s: dict) -> str:
    return (
        f"af={s['active_frac']:.3f} sph={s['sphericity']:5.2f} "
        f"bbf={s['bbox_fill']:.3f} ifr={s['interface_frac']:.3f} "
        f"ar={s['axis_ratio']:5.2f} pl={s['axis_planarity']:.3f} "
        f"oct={s['envelope_octa']:5.2f}"
    )


def find_at_af(snapshots, target_af, tol=0.10):
    """Return the snapshot whose active_frac is closest to target,
    None if nothing within tolerance."""
    best = None
    for _, s in snapshots:
        if not math.isfinite(s.get('active_frac', float('nan'))):
            continue
        if best is None or abs(s['active_frac'] - target_af) < abs(best['active_frac'] - target_af):
            best = s
    if best is None or abs(best['active_frac'] - target_af) > tol:
        return None
    return best


def shape_distance(a: dict, b: dict) -> float:
    """Distance in normalized geometric-feature space.

    Uses (sphericity, bbox_fill, interface_frac, log axis_ratio,
    envelope_octa). Each feature is whitened by a hand-picked typical
    range. Two rules with distance < 0.3 are essentially the same shape.
    """
    if a is None or b is None:
        return float('nan')
    ranges = dict(sphericity=0.5, bbox_fill=0.3, interface_frac=0.3,
                  axis_ratio_log=1.0, envelope_octa=0.5)
    feats_a = (a['sphericity'], a['bbox_fill'], a['interface_frac'],
               math.log(max(a['axis_ratio'], 1e-3)), a['envelope_octa'])
    feats_b = (b['sphericity'], b['bbox_fill'], b['interface_frac'],
               math.log(max(b['axis_ratio'], 1e-3)), b['envelope_octa'])
    keys = ['sphericity', 'bbox_fill', 'interface_frac', 'axis_ratio_log', 'envelope_octa']
    sq = 0.0
    n = 0
    for fa, fb, k in zip(feats_a, feats_b, keys):
        if not (math.isfinite(fa) and math.isfinite(fb)):
            continue
        sq += ((fa - fb) / ranges[k]) ** 2
        n += 1
    if n == 0:
        return float('nan')
    return math.sqrt(sq / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', type=str, default=','.join(CRYSTAL_RULES))
    ap.add_argument('--sizes', type=str, default='64,96')
    ap.add_argument('--seeds', type=str, default='42,7,123',
                    help='Multi-seed = test whether the rule actually has '
                         'stochastic morphology variation or is deterministic.')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--match-af', type=float, default=0.20,
                    help='Active-fraction at which to compare rules.')
    args = ap.parse_args()

    rules = [r.strip() for r in args.rules.split(',') if r.strip()]
    sizes = [int(s) for s in args.sizes.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    steps = args.steps

    sample_steps = sorted(set(np.unique(np.round(np.geomspace(20, steps, 7)).astype(int)).tolist()) | {steps})

    win, ctx = create_headless_context()
    try:
        # results: rule -> size -> seed -> [(step, shape_stats)]
        results = {}
        t_total = time.time()
        for rule in rules:
            results[rule] = {}
            for sz in sizes:
                results[rule][sz] = {}
                for seed in seeds:
                    t0 = time.time()
                    snaps = run_one(ctx, rule, sz, steps, seed, sample_steps)
                    print(f'  {rule:<22} sz={sz} seed={seed}  ({time.time()-t0:.1f}s)')
                    results[rule][sz][seed] = snaps
        print(f'\n  total runtime: {time.time()-t_total:.1f}s\n')

        # ---- Per-rule timeline (size 64, seed 42) ------------------------
        print('=' * 110)
        print(f'PER-RULE TIMELINE  (size={sizes[0]}, seed={seeds[0]})')
        print('=' * 110)
        for rule in rules:
            print(f'\n  {rule}')
            for step, s in results[rule][sizes[0]][seeds[0]]:
                print(f'    step={step:5d}  {fmt_stats(s)}')

        # ---- Multi-seed variance --------------------------------------
        print('\n' + '=' * 110)
        print(f'STOCHASTICITY  (does the rule produce different shapes at '
              f'matched af={args.match_af} across seeds?)')
        print('=' * 110)
        print(f'  {"rule":<22}  {"sphericity_std":>16}  {"bbf_std":>10}  {"ifr_std":>10}  {"oct_std":>10}  diagnosis')
        for rule in rules:
            sph, bbf, ifr, oct_ = [], [], [], []
            for seed in seeds:
                snaps = results[rule][sizes[0]][seed]
                s = find_at_af(snaps, args.match_af)
                if s is None:
                    continue
                if math.isfinite(s['sphericity']): sph.append(s['sphericity'])
                if math.isfinite(s['bbox_fill']):   bbf.append(s['bbox_fill'])
                if math.isfinite(s['interface_frac']): ifr.append(s['interface_frac'])
                if math.isfinite(s['envelope_octa']): oct_.append(s['envelope_octa'])
            def std(v): return float(np.std(v)) if len(v) > 1 else float('nan')
            sigma_sph, sigma_bbf, sigma_ifr, sigma_oct = std(sph), std(bbf), std(ifr), std(oct_)
            verdict = 'stochastic' if (sigma_sph > 0.05 or sigma_oct > 0.10 or sigma_bbf > 0.05) else 'deterministic'
            if not sph:
                verdict = 'never reached target af'
            print(f'  {rule:<22}  {sigma_sph:>16.4f}  {sigma_bbf:>10.4f}  {sigma_ifr:>10.4f}  {sigma_oct:>10.4f}  {verdict}')

        # ---- Cross-rule shape similarity ------------------------------
        print('\n' + '=' * 110)
        print(f'CROSS-RULE SHAPE DISTANCE  (size={sizes[0]}, seed={seeds[0]}, at af≈{args.match_af})')
        print('  small distance (<0.30) means two rules produce the same shape — they are NOT distinct presets')
        print('=' * 110)
        # Header
        widths = max(len(r) for r in rules)
        per_rule_at_af = {r: find_at_af(results[r][sizes[0]][seeds[0]], args.match_af) for r in rules}
        # Print descriptors first
        print(f'\n  Descriptors at af≈{args.match_af}:')
        for r in rules:
            s = per_rule_at_af[r]
            if s is None:
                print(f'    {r:<{widths}}  (no snapshot near target af)')
            else:
                print(f'    {r:<{widths}}  {fmt_stats(s)}')
        # Distance matrix
        print('\n  Pairwise distance matrix:')
        header = '    ' + ' ' * widths + '  ' + '  '.join(f'{r[8:14]:>6}' for r in rules)
        print(header)
        for r1 in rules:
            row = f'    {r1:<{widths}}  '
            cells = []
            for r2 in rules:
                d = shape_distance(per_rule_at_af[r1], per_rule_at_af[r2])
                if not math.isfinite(d):
                    cells.append(f'{"-":>6}')
                else:
                    cells.append(f'{d:>6.2f}')
            print(row + '  '.join(cells))

        # ---- Cluster: which rules collapse to same shape? -------------
        print('\n' + '=' * 110)
        print('SHAPE-EQUIVALENCE CLUSTERS  (rules whose pairwise distance < 0.30)')
        print('=' * 110)
        clusters = []
        unassigned = list(rules)
        while unassigned:
            seed_r = unassigned.pop(0)
            cluster = [seed_r]
            for r in list(unassigned):
                d = shape_distance(per_rule_at_af[seed_r], per_rule_at_af[r])
                if math.isfinite(d) and d < 0.30:
                    cluster.append(r)
                    unassigned.remove(r)
            clusters.append(cluster)
        for i, c in enumerate(clusters):
            if len(c) > 1:
                print(f'  Cluster {i+1} (collapsed):  ' + ', '.join(c))
            else:
                print(f'  Cluster {i+1} (distinct):   ' + c[0])

        # ---- Scale variance --------------------------------------------
        if len(sizes) > 1:
            print('\n' + '=' * 110)
            print(f'SCALE VARIANCE  (does shape change with grid size at matched af={args.match_af}?)')
            print('=' * 110)
            print(f'  {"rule":<22}  ' + '  '.join(f'sz={sz:>3}: sph/bbf/oct' for sz in sizes))
            for rule in rules:
                cells = []
                for sz in sizes:
                    s = find_at_af(results[rule][sz][seeds[0]], args.match_af)
                    if s is None:
                        cells.append(f'{"---":>16}')
                    else:
                        cells.append(f'{s["sphericity"]:.2f}/{s["bbox_fill"]:.2f}/{s["envelope_octa"]:.2f}'.rjust(16))
                print(f'  {rule:<22}  ' + '  '.join(cells))
    finally:
        try:
            ctx.release()
        except Exception:
            pass


if __name__ == '__main__':
    main()
