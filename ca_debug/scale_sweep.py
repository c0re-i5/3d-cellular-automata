"""Scale-sweep probe: run the same rule at a ladder of grid sizes
and print per-step alive-ratio so size-dependent regressions show up
as a single sweep table.

Triggered by the report that 3D Game of Life behaves correctly at
size<=96 but collapses to a static block at size>=128 — exactly the
threshold at which `_sparse_supported()` flips on. This probe lets us
A/B the sparse-dispatch path by toggling
``RULE_PRESETS[rule]['sparse_dispatch']`` between runs.

Usage::

    python -m ca_debug.scale_sweep --rule game_of_life_3d \
        --sizes 64,96,128,160,192 --steps 60
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _alive_ratio(grid: np.ndarray, threshold: float = 0.5) -> float:
    return float((grid[..., 0] > threshold).mean())


def _run_one(ctx, rule: str, *, size: int, steps: int, seed: int,
             sparse: bool | None = None,
             ) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Run rule at size for steps; return (alive trajectory, g0, gN).

    If `sparse` is not None, force-set the preset flag for this run.
    """
    from simulator import RULE_PRESETS
    from test_harness import HeadlessRunner

    saved = RULE_PRESETS[rule].get('sparse_dispatch', False)
    if sparse is not None:
        RULE_PRESETS[rule]['sparse_dispatch'] = sparse
    try:
        r = HeadlessRunner(ctx, rule, size=size, seed=seed)
        traj: list[float] = []
        g0 = r.read_grid().astype(np.float32).copy()
        traj.append(_alive_ratio(g0))
        for _ in range(steps):
            r.step()
            traj.append(_alive_ratio(r.read_grid()))
        gN = r.read_grid().astype(np.float32).copy()
        if hasattr(r, 'release'):
            try: r.release()
            except Exception: pass
        return traj, g0, gN
    finally:
        RULE_PRESETS[rule]['sparse_dispatch'] = saved


def _solid_block_signature(g: np.ndarray, threshold: float = 0.5) -> dict:
    """Quantify how 'solid block'-like the final state is.
    A clean B6-7/S5-7 chaos has alive~5-15% and high boundary entropy.
    A pathological 'cube' state has alive~50%+ in a contiguous region
    and near-zero per-step delta.
    """
    alive = g[..., 0] > threshold
    frac = float(alive.mean())
    # Largest filled axis-aligned box: cheap proxy via per-axis bounds
    if not alive.any():
        return {'alive_frac': 0.0, 'bbox_fill': 0.0, 'bbox_dims': (0, 0, 0)}
    nz = np.argwhere(alive)
    lo = nz.min(0); hi = nz.max(0) + 1
    box_dims = tuple(int(x) for x in (hi - lo))
    box_vol = int(np.prod(box_dims))
    bbox_fill = float(alive.sum()) / max(box_vol, 1)
    return {'alive_frac': frac, 'bbox_fill': bbox_fill, 'bbox_dims': box_dims}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rule', default='game_of_life_3d')
    p.add_argument('--sizes', default='64,96,128,160,192',
                   help='comma-separated grid edges')
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ab-sparse', action='store_true',
                   help='at each size, run with sparse=on AND sparse=off and '
                        'diff the trajectories')
    p.add_argument('--print-traj', action='store_true',
                   help='print full per-step alive ratio')
    args = p.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(',')]

    import moderngl
    ctx = moderngl.create_standalone_context(require=430)

    print(f"\n=== scale_sweep: {args.rule} seed={args.seed} steps={args.steps} ===\n")
    fmt_hdr = (f"{'size':>5}  {'mode':<8}  {'alive[0]':>9}  {'alive[mid]':>10}  "
               f"{'alive[N]':>9}  {'bbox_fill':>9}  {'bbox_dims':<14}  notes")
    print(fmt_hdr)
    print("-" * len(fmt_hdr))

    modes: list[bool | None]
    if args.ab_sparse:
        modes = [False, True]   # dense first, then sparse
    else:
        modes = [None]          # leave preset default

    for size in sizes:
        for mode in modes:
            label = ('dense' if mode is False
                     else 'sparse' if mode is True
                     else 'default')
            try:
                traj, g0, gN = _run_one(
                    ctx, args.rule, size=size, steps=args.steps,
                    seed=args.seed, sparse=mode)
            except Exception as e:
                print(f"{size:>5}  {label:<8}  ERROR: {e!r}")
                continue
            sig = _solid_block_signature(gN)
            mid = traj[len(traj) // 2]
            note = ''
            if sig['bbox_fill'] > 0.85 and sig['alive_frac'] > 0.4:
                note = '<<< SOLID-BLOCK PATHOLOGY'
            elif traj[-1] < 1e-4:
                note = '<<< died out'
            print(f"{size:>5}  {label:<8}  {traj[0]:>9.4f}  {mid:>10.4f}  "
                  f"{traj[-1]:>9.4f}  {sig['bbox_fill']:>9.4f}  "
                  f"{str(sig['bbox_dims']):<14}  {note}")
            if args.print_traj:
                line = ' '.join(f"{v:.3f}" for v in traj)
                print(f"        traj: {line}")
        if args.ab_sparse:
            print()  # blank line between sizes for readability

    return 0


if __name__ == '__main__':
    sys.exit(main())
