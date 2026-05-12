#!/usr/bin/env python3
"""Validate that the Margolus 3D rule conserves particle count exactly.

Reversibility implies bit-exact conservation: every step is a permutation
of cells inside disjoint 2x2x2 blocks, so the global sum of channel R is
invariant. Floating-point storage (RGBA32F) can faithfully represent 0.0
and 1.0 without any rounding, so we expect ZERO drift over arbitrary
runs.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_harness import create_headless_context, HeadlessRunner


def main():
    window, ctx = create_headless_context()
    try:
        size = 64
        runner = HeadlessRunner(ctx, "margolus_3d", size=size, seed=123)
        initial = runner.read_grid()
        n0 = float(initial[..., 0].sum())
        density = n0 / (size ** 3)
        print(f"Initial particles: {n0:.0f}  ({density*100:.2f}% density)")

        steps = 500
        check_every = 50
        max_drift = 0
        for s in range(1, steps + 1):
            runner.step()
            if s % check_every == 0:
                grid = runner.read_grid()
                n = float(grid[..., 0].sum())
                drift = abs(n - n0)
                max_drift = max(max_drift, drift)
                # Also check binarity: all values should still be 0 or 1.
                ch = grid[..., 0]
                non_binary = int(((ch != 0.0) & (ch != 1.0)).sum())
                print(f"  step {s:4d}: particles={n:.0f}  drift={drift:.0f}  non_binary={non_binary}")

        print()
        if max_drift == 0:
            print("PASS: particle count conserved exactly across", steps, "steps")
            return 0
        else:
            print(f"FAIL: max drift = {max_drift}")
            return 1
    finally:
        import glfw
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    sys.exit(main())
