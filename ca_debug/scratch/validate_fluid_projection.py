#!/usr/bin/env python3
"""Validate that the multi-pass Stable Fluids solver actually enforces
incompressibility via Helmholtz projection.

Method:
  1. Initialise a turbulent velocity field (high divergence).
  2. Disable the smoke source (param Source=0) so no new energy is injected.
  3. Disable damping (param Damping=0).
  4. Step 50 times.
  5. Measure mean |div(v)| at start and end of the run.

PASS if final divergence is at least 30% lower than initial divergence.
With true projection the typical reduction at default Jacobi-iter count
should be ~70-95%.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_harness import create_headless_context, HeadlessRunner


def divergence(v):
    """Mean absolute divergence of a velocity field v[X,Y,Z,3]."""
    vxp = np.roll(v[..., 0], -1, axis=0)
    vxm = np.roll(v[..., 0],  1, axis=0)
    vyp = np.roll(v[..., 1], -1, axis=1)
    vym = np.roll(v[..., 1],  1, axis=1)
    vzp = np.roll(v[..., 2], -1, axis=2)
    vzm = np.roll(v[..., 2],  1, axis=2)
    div = 0.5 * ((vxp - vxm) + (vyp - vym) + (vzp - vzm))
    return float(np.mean(np.abs(div)))


def main():
    window, ctx = create_headless_context()
    try:
        size = 48
        # Source=0 -> no new energy. Damping=0 -> only projection can reduce
        # divergence. Viscosity small (bg smoothing barely shrinks div at all,
        # but to be safe set tiny).
        runner = HeadlessRunner(
            ctx, "stable_fluids_3d", size=size, seed=7,
            params={"Viscosity": 0.0, "Source": 0.0,
                    "Damping": 0.0, "Vorticity": 0.0},
            init_override="fluid_turbulent",
        )

        grid0 = runner.read_grid()
        d0 = divergence(grid0[..., :3])
        print(f"Initial mean |div(v)| = {d0:.5f}")

        for s in range(50):
            runner.step()

        grid1 = runner.read_grid()
        d1 = divergence(grid1[..., :3])
        print(f"Final   mean |div(v)| = {d1:.5f}")

        if not np.all(np.isfinite(grid1)):
            print("FAIL: non-finite values after run")
            return 1

        if d0 < 1e-6:
            print("INCONCLUSIVE: initial divergence too small to test")
            return 0

        reduction = (d0 - d1) / d0 * 100.0
        print(f"Reduction: {reduction:.1f}%")
        if reduction >= 30.0:
            print(f"PASS: projection reduced divergence by {reduction:.1f}% "
                  f"(target >=30%)")
            return 0
        else:
            print(f"FAIL: projection only reduced divergence by "
                  f"{reduction:.1f}% (target >=30%)")
            return 1
    finally:
        import glfw
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    sys.exit(main())
