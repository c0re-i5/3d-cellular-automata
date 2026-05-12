"""Diagnose why dendritic produces an octahedron with no branches.

Hypothesis: u_init=0.25 reservoir is too small. Crystal halts at af~0.18
within ~90 steps because global u is depleted. Mullins-Sekerka instability
amplifies bumps at rate ~ V·k where V is front velocity; when V→0
quickly, bumps don't grow. We need to keep V positive for thousands of
steps so noise-seeded bumps amplify into branches.

Test:
  1. Monkey-patch init_crystal_seed so u_init can be overridden.
  2. Run dendritic for 6000 steps at varying u_init and U.
  3. Track active_frac, growth_rate, envelope_octa, branchiness over time.
  4. Look for the regime where branchiness rises to ≥ 0.4 (sphere baseline 0.26).
"""

from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import simulator  # noqa: E402
from test_harness import create_headless_context, HeadlessRunner  # noqa: E402
from scripts.batch_debug_audit import snapshot_stats  # noqa: E402


_orig_init = simulator.init_crystal_seed
_U_INIT = 0.25  # mutable knob


def _patched_init(size, rng):
    """Drop-in replacement that uses _U_INIT instead of hard-coded 0.25."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    data[:, :, :, 0] = 0.0
    data[:, :, :, 1] = _U_INIT + simulator._canonical_noise(size, rng, -0.005, 0.005)
    fx = 0.5 + rng.uniform(-0.04, 0.04)
    fy = 0.5 + rng.uniform(-0.04, 0.04)
    fz = 0.5 + rng.uniform(-0.04, 0.04)
    cx, cy, cz = fx * size, fy * size, fz * size
    r = max(2.0, size * 0.025)
    dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
    mask = dist <= r
    data[:, :, :, 0][mask] = 1.0
    data[:, :, :, 1][mask] = 0.0
    return data


simulator.init_crystal_seed = _patched_init
simulator.INIT_FUNCS['crystal_seed'] = _patched_init


def trace_one(u_init, U, D, eps_strength, size=96, steps=6000, seed=42,
              snapshot_every=200):
    """Return time series of (step, af, growth_rate, oct, branchiness, u_mean)."""
    global _U_INIT
    _U_INIT = u_init

    ctx = create_headless_context()
    if isinstance(ctx, tuple):
        _w, ctx = ctx
    runner = HeadlessRunner(ctx, "crystal_dendritic", size=size, seed=seed)
    runner.params["Undercooling"] = U
    runner.params["Diffusion"] = D
    runner.params["Anisotropy strength"] = eps_strength

    series = []
    prev_alive = None
    for i in range(steps + 1):
        if i > 0:
            runner.step()
        if i % snapshot_every == 0:
            g = runner.read_grid()
            s = snapshot_stats(g, i, time.time(), prev_alive)
            prev_alive = s['active_count']
            u_mean = float(np.mean(g[..., 1]))
            series.append({
                'step': i,
                'af': s['active_frac'],
                'gr': s.get('growth_rate', 0),
                'oct': s.get('envelope_octa', float('nan')),
                'br': s.get('branchiness', float('nan')),
                'u_mean': u_mean,
            })
    runner.release()
    return series


def main():
    configs = [
        # (u_init, U,   D,    eps, label)
        (0.25, 0.4, 0.02, 1.0, "DEFAULT"),
        (0.35, 0.4, 0.02, 1.0, "u_init=0.35"),
        (0.50, 0.4, 0.02, 1.0, "u_init=0.50"),
        (0.35, 0.4, 0.02, 2.0, "u_init=0.35 eps=2"),
        (0.35, 0.4, 0.02, 3.0, "u_init=0.35 eps=3"),
        (0.50, 0.4, 0.02, 2.0, "u_init=0.50 eps=2"),
        (0.50, 0.4, 0.02, 3.0, "u_init=0.50 eps=3"),
        (0.40, 0.3, 0.02, 2.0, "u_init=0.40 U=0.3 eps=2 (slower kinetics)"),
        (0.40, 0.5, 0.02, 2.0, "u_init=0.40 U=0.5 eps=2"),
    ]

    for u_init, U, D, eps, label in configs:
        print(f"\n=== {label}: u_init={u_init} U={U} D={D} eps={eps} ===")
        s = trace_one(u_init, U, D, eps, size=96, steps=4000,
                      seed=42, snapshot_every=200)
        print(f"  {'step':>5} {'af':>6} {'gr':>8} {'oct':>5} {'br':>5} {'u_mean':>7}")
        for row in s:
            af = row['af']
            print(f"  {row['step']:>5} {af:>6.3f} {row['gr']:>8.0f} "
                  f"{row['oct']:>5.2f} {row['br']:>5.2f} {row['u_mean']:>7.4f}")


if __name__ == "__main__":
    main()
