"""Validate block-sparse compute dispatch against dense dispatch.

Runs the SAME rule, SAME seed, SAME init for N steps with
sparse_dispatch_enabled = True, then again with False, and asserts
the field state matches bit-for-bit (allowing tiny FP epsilon).

Usage:
    python validate_sparse.py [--rule game_of_life_3d] [--size 192] [--steps 50]
"""
import argparse
import numpy as np

# Headless EGL/offscreen — set BEFORE importing simulator/moderngl.
import os
os.environ.setdefault('MODERNGL_REQUIRE', 'standalone')

import simulator as S


def run(rule, size, steps, sparse, seed=42):
    sim = S.Simulator(size=size, rule=rule, headless=True)
    # Set the sparse flag BEFORE _reset() and any sim step. Shader was
    # already compiled in __init__ honoring the default flag (=True).
    # If we want to test the dense path, we need to recompile.
    if sim.sparse_dispatch_enabled != sparse:
        sim.sparse_dispatch_enabled = sparse
        sim._compile_compute()
        sim._cache_compute_uniforms()
    sim.seed = seed
    sim._reset()
    for _ in range(steps):
        sim._step_sim()
    src = sim.tex_a if sim.ping == 0 else sim.tex_b
    raw = src.read()
    arr = np.frombuffer(raw, dtype=sim._tex_np_dtype).reshape(size, size, size, 4)
    out = arr.copy()  # detach from buffer
    sim.ctx.release()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rule', default='game_of_life_3d')
    p.add_argument('--size', type=int, default=192)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tol', type=float, default=1e-5)
    args = p.parse_args()

    print(f"Running rule={args.rule} size={args.size} steps={args.steps} seed={args.seed}")
    print("  Dense dispatch ...")
    dense = run(args.rule, args.size, args.steps, sparse=False, seed=args.seed)
    print("  Sparse dispatch ...")
    sparse = run(args.rule, args.size, args.steps, sparse=True, seed=args.seed)

    diff = np.abs(dense.astype(np.float32) - sparse.astype(np.float32))
    max_diff = float(diff.max())
    n_diff = int((diff > args.tol).sum())
    total = diff.size

    print(f"\nDense  alive: {int((dense[..., 0] > 0.5).sum())}")
    print(f"Sparse alive: {int((sparse[..., 0] > 0.5).sum())}")
    print(f"Max abs diff:   {max_diff:.6g}")
    print(f"Cells differing > {args.tol}: {n_diff} / {total} ({100*n_diff/total:.4f}%)")

    if max_diff > args.tol:
        # Show first few differing positions to aid debugging
        idx = np.argwhere(diff > args.tol)[:10]
        print("\nFirst differing cells:")
        for (x, y, z, c) in idx:
            print(f"  ({x},{y},{z}) ch{c}: dense={dense[x,y,z,c]:.4g}  sparse={sparse[x,y,z,c]:.4g}")
        raise SystemExit(1)
    print("\n✓ PASS — sparse and dense dispatch produce identical state.")


if __name__ == '__main__':
    main()
