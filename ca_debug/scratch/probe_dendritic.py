"""Probe dendritic branching: vary u² coefficient, FBM amplitude, U, D
and measure branchiness directly. Find a config with br ≥ 0.5 (genuine
6-arm dendrite signature) before changing simulator defaults.

Strategy: monkey-patch the GLSL source at runtime to override the
dendritic-mode constants we care about, then run a small sweep.
"""

from __future__ import annotations
import os, sys, math, re, time
from pathlib import Path
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import simulator  # noqa: E402
from test_harness import create_headless_context, HeadlessRunner  # noqa: E402
from scripts.batch_debug_audit import snapshot_stats  # noqa: E402

CRYSTAL_SHADER_BLOCK_KEY = (
    "// DENDRITIC: u^2 positive feedback"
)


def patch_shader(src: str, u2_coef: float, fbm_amp: float,
                 fbm_freq: float) -> str:
    """Replace the magic numbers in the dendritic mode without touching anything else."""
    # u² coefficient line
    src = re.sub(
        r"(driving \+= phi \* \(1\.0 - phi\) \* )12\.0( \* u_field \* u_field;)",
        rf"\g<1>{u2_coef:.3f}\g<2>",
        src, count=1,
    )
    # FBM amplitude and frequency (the dendritic block)
    # `beta += 0.20 * (fnoise - 0.5) * smoothstep(0.005, 0.15, grad_mag);`
    # `float fnoise = fbm3_temporal(vec3(pos), 6.0, 5, 1);`
    src = re.sub(
        r"float fnoise = fbm3_temporal\(vec3\(pos\), 6\.0, 5, 1\);",
        f"float fnoise = fbm3_temporal(vec3(pos), {fbm_freq:.2f}, 5, 1);",
        src, count=1,
    )
    src = re.sub(
        r"(beta \+= )0\.20( \* \(fnoise - 0\.5\) \* smoothstep\(0\.005, 0\.15, grad_mag\);)",
        rf"\g<1>{fbm_amp:.3f}\g<2>",
        src, count=1,
    )
    return src


def run_one(u2_coef, fbm_amp, fbm_freq, U, D, eps_strength=6.0,
            size=96, steps=2500, seeds=(42, 7, 123)):
    """Patch shader source, build a fresh context+runner, sweep across seeds, return matched-af stats."""
    original = simulator.CA_RULES["crystal_growth"]
    simulator.CA_RULES["crystal_growth"] = patch_shader(
        original, u2_coef, fbm_amp, fbm_freq
    )
    try:
        ctx = create_headless_context()
        if isinstance(ctx, tuple):
            _w, ctx = ctx

        # Aggregate across seeds
        per_seed_metrics = []
        for seed in seeds:
            runner = HeadlessRunner(ctx, "crystal_dendritic", size=size, seed=seed)
            runner.params["Undercooling"] = U
            runner.params["Diffusion"] = D
            runner.params["Anisotropy strength"] = eps_strength
            stats_at_af = {}  # af_target → snap

            af_targets = [0.02, 0.05, 0.10, 0.20]
            prev_alive = None
            g = runner.read_grid()
            s = snapshot_stats(g, 0, time.time(), prev_alive)
            prev_alive = s['active_count']
            for i in range(1, steps + 1):
                runner.step()
                if i % 30 == 0 or i == steps:
                    g = runner.read_grid()
                    s = snapshot_stats(g, i, time.time(), prev_alive)
                    prev_alive = s['active_count']
                    for af_t in af_targets:
                        if af_t not in stats_at_af and s['active_frac'] >= af_t:
                            stats_at_af[af_t] = s
                    if all(t in stats_at_af for t in af_targets):
                        break
            runner.release()
            per_seed_metrics.append(stats_at_af)

        # Average envelope/branchiness across seeds at each af
        out = {"u2": u2_coef, "fbm_amp": fbm_amp, "fbm_freq": fbm_freq,
               "U": U, "D": D, "eps": eps_strength}
        for af_t in [0.02, 0.05, 0.10, 0.20]:
            envs = [m[af_t]['envelope_octa'] for m in per_seed_metrics
                    if af_t in m and math.isfinite(m[af_t].get('envelope_octa', float('nan')))]
            brs  = [m[af_t]['branchiness'] for m in per_seed_metrics
                    if af_t in m and math.isfinite(m[af_t].get('branchiness', float('nan')))]
            out[f"oct@{af_t:.2f}"] = float(np.mean(envs)) if envs else float('nan')
            out[f"br@{af_t:.2f}"]  = float(np.mean(brs))  if brs  else float('nan')
        return out
    finally:
        simulator.CA_RULES["crystal_growth"] = original


def main():
    # Probe: low eps_strength so kernel doesn't dominate FBM noise.
    # Vary eps and FBM amplitude. Also try high U where M-S is stronger.
    configs = []
    for eps in (0.5, 1.0, 2.0, 6.0):
        for fbm_amp in (0.20, 0.80, 2.00):
            for U in (0.4, 0.7, 1.0):
                configs.append((30.0, fbm_amp, 6.0, U, 0.02, eps))

    print(f"# Probing {len(configs)} dendritic configs (2 seeds each)…")
    print(f"# {'eps':>4} {'fbm_a':>6} {'U':>5}  "
          f"{'oct@.02':>8} {'oct@.05':>8} {'oct@.10':>8} "
          f"{'br@.02':>7} {'br@.05':>7} {'br@.10':>7} {'br@.20':>7}")
    t0 = time.perf_counter()
    results = []
    for i, (u2, amp, freq, U, D, eps) in enumerate(configs):
        r = run_one(u2, amp, freq, U, D, eps_strength=eps,
                    size=96, steps=2500, seeds=(42, 7))
        results.append(r)
        elapsed = time.perf_counter() - t0
        print(
            f"  {eps:>4.1f} {amp:>6.2f} {U:>5.2f}  "
            f"{r['oct@0.02']:>8.2f} {r['oct@0.05']:>8.2f} {r['oct@0.10']:>8.2f} "
            f"{r['br@0.02']:>7.2f} {r['br@0.05']:>7.2f} "
            f"{r['br@0.10']:>7.2f} {r['br@0.20']:>7.2f}  "
            f"[{i+1}/{len(configs)} {elapsed:.0f}s]"
        )

    def best_br(r):
        return max((r.get(f'br@{a:.2f}', 0) for a in (0.05, 0.10, 0.20)),
                   default=0)
    results.sort(key=best_br, reverse=True)
    print("\n# Top 8 by branchiness:")
    for r in results[:8]:
        print(f"  eps={r['eps']:>4.1f} fbm_amp={r['fbm_amp']:>4.2f} "
              f"U={r['U']:.2f}  "
              f"br = {r['br@0.05']:.2f} / {r['br@0.10']:.2f} / "
              f"{r['br@0.20']:.2f}  oct@.10={r['oct@0.10']:.2f}")


if __name__ == "__main__":
    main()
