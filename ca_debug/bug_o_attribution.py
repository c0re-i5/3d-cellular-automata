"""Probe #25 — Bug O attribution: entity-state race vs paint race.

The base determinism probe reads back the voxel grid only. For arena
rules that have BOTH an entity SSBO and a painted grid, that conflates
two race classes:

    * step-shader races on the entity SSBO itself
      (e.g. atomicCompSwap arrival-order, grazing RMW on shared field)
    * paint races on the voxel grid
      (e.g. SHADER_PAINT non-atomic max-blend on overlapping voxels)

This probe runs each arena/agent rule twice from the same seed and
reports each independently. Verdict:

    state_race  entity SSBO differs across runs (step shader races)
    paint_race  entity SSBO identical but grid differs (paint only)
    both        both differ
    clean       both identical
    n/a         no entity arena (agent-only rules go straight to grid)

Usage:
    python -m ca_debug.bug_o_attribution
    python -m ca_debug.bug_o_attribution --rules wandering_voxels_3d
    python -m ca_debug.bug_o_attribution --size 64 --steps 60 --trials 3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys

import numpy as np


_TARGET_RULES = (
    'wandering_voxels_3d',   # entity_arena, no step-shader writes
    'predator_prey_3d',      # entity_arena, step-shader RMW + atomicCompSwap
    'smugglers_3d',          # agent path, non-atomic add to grid
)


def _run(ctx, rule: str, *, size: int, steps: int, seed: int):
    from test_harness import HeadlessRunner
    r = HeadlessRunner(ctx, rule, size=size, seed=seed)
    try:
        for _ in range(steps):
            r.step()
        grid = np.asarray(r.read_grid()).copy()
        ent_bytes = None
        if hasattr(r, 'arena') and r.arena is not None:
            r.arena.pull_entities()
            ent_bytes = bytes(r.arena.entities.tobytes())
        return grid, ent_bytes
    finally:
        if hasattr(r, 'release'):
            try: r.release()
            except Exception: pass  # noqa: BLE001


def _grid_diff(a, b):
    d = np.abs(a - b)
    return float(d.max()), int((d > 0).sum())


def _classify(g_max, g_ndiff, ent_a, ent_b):
    has_arena = ent_a is not None and ent_b is not None
    grid_diverges = g_max > 0 or g_ndiff > 0
    if not has_arena:
        return ('grid_race' if grid_diverges else 'clean'), False
    state_diverges = ent_a != ent_b
    if state_diverges and grid_diverges:
        return 'both', True
    if state_diverges:
        return 'state_race', True
    if grid_diverges:
        return 'paint_race', False
    return 'clean', False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rules', help='Comma-separated; defaults to target set.')
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--trials', type=int, default=2,
                    help='How many runs to compare (pairwise vs run 0).')
    args = ap.parse_args(argv)

    rules = ([r.strip() for r in args.rules.split(',') if r.strip()]
             if args.rules else list(_TARGET_RULES))

    import moderngl
    ctx = moderngl.create_standalone_context(require=430)

    print(f"\n=== bug_o_attribution — size={args.size} steps={args.steps} "
          f"seed={args.seed} trials={args.trials} ===\n", file=sys.stderr)
    print(f"{'rule':<24}  {'trial':<5}  verdict       grid_max  grid_ndiff  "
          f"ent_bytes_differ")
    print('-' * 84)

    rc = 0
    for rule in rules:
        with contextlib.redirect_stdout(io.StringIO()):
            g0, e0 = _run(ctx, rule, size=args.size,
                          steps=args.steps, seed=args.seed)
        for t in range(1, args.trials):
            with contextlib.redirect_stdout(io.StringIO()):
                g, e = _run(ctx, rule, size=args.size,
                            steps=args.steps, seed=args.seed)
            g_max, g_n = _grid_diff(g0, g)
            verdict, is_race = _classify(g_max, g_n, e0, e)
            ent_diff = '-'
            if e0 is not None and e is not None:
                arr0 = np.frombuffer(e0, dtype=np.uint8)
                arr  = np.frombuffer(e,  dtype=np.uint8)
                ent_diff = str(int((arr0 != arr).sum()))
            print(f"{rule:<24}  {t:<5d}  {verdict:<12}  {g_max:>8.4f}  "
                  f"{g_n:>10d}  {ent_diff}")
            if is_race or verdict in ('grid_race', 'paint_race'):
                rc = 1

    return rc


if __name__ == '__main__':
    sys.exit(main())
