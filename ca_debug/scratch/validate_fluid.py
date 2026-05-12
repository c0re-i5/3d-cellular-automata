#!/usr/bin/env python3
"""Validate that the 3D Smoke Plume rule stays numerically stable
(no NaN/Inf, bounded velocity, bounded dye) over a long run with the
default smoke source enabled.

This rule is NOT a true incompressible solver — it relies on viscosity
+ damping to keep divergence-driven energy growth in check. The test
here is therefore stability rather than physical correctness:

  PASS if, after 500 steps:
    - no NaN or Inf anywhere
    - max |velocity| <= 0.5 * grid size (the in-shader clamp)
    - dye stays within [0, 4] (the in-shader clamp)
    - kinetic energy reached a roughly steady state
      (final-vs-mid-window KE ratio between 0.5 and 2.0)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_harness import create_headless_context, HeadlessRunner


def main():
    window, ctx = create_headless_context()
    try:
        size = 48
        runner = HeadlessRunner(ctx, "stable_fluids_3d", size=size, seed=11)
        steps = 500
        check_every = 50
        kes = []
        for s in range(1, steps + 1):
            runner.step()
            if s % check_every == 0:
                grid = runner.read_grid()
                if not np.all(np.isfinite(grid)):
                    print(f"FAIL: non-finite values at step {s}")
                    return 1
                vmax = float(np.max(np.abs(grid[..., :3])))
                dmax = float(np.max(grid[..., 3]))
                ke   = float(np.mean(grid[..., :3] ** 2))
                dye  = float(grid[..., 3].sum())
                kes.append(ke)
                print(f"  step {s:4d}: |v|max={vmax:7.3f}  dye_max={dmax:.3f}  "
                      f"KE={ke:.5f}  dye_total={dye:.1f}")
                if vmax > 0.5 * size + 1e-3:
                    print(f"FAIL: velocity exceeded clamp ({vmax} > {0.5 * size})")
                    return 1
                if dmax > 4.0 + 1e-3:
                    print(f"FAIL: dye exceeded clamp ({dmax} > 4.0)")
                    return 1

        # Steady-state check: compare last-third KE mean to middle-third.
        n = len(kes)
        mid = sum(kes[n//3:2*n//3]) / max(1, len(kes[n//3:2*n//3]))
        end = sum(kes[2*n//3:]) / max(1, len(kes[2*n//3:]))
        ratio = end / mid if mid > 1e-10 else 1.0
        print()
        print(f"Mid-window KE: {mid:.5f}")
        print(f"End-window KE: {end:.5f}")
        print(f"Ratio:         {ratio:.3f}")
        if 0.5 <= ratio <= 2.0:
            print("PASS: smoke plume reached steady state, stayed numerically bounded")
            return 0
        else:
            print("FAIL: KE did not stabilise (ratio outside [0.5, 2.0])")
            return 1
    finally:
        import glfw
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    sys.exit(main())
