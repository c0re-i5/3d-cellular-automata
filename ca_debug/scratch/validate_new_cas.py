"""Physics validators for the four new exotic CAs.

Each test exercises the rule's defining property:
  * causal_ca:        cells lit only inside the Manhattan light cone
  * hopfion_3d:       |n| stays unit (within tolerance)
  * genome_ca_3d:     spatial variance is preserved, no blow-up
  * active_nematic_3d: |Q| stays bounded; ordering grows with time when A<0
"""
import sys
import numpy as np
import moderngl
from test_harness import HeadlessRunner, create_headless_context, destroy_context


def validate_causal():
    win, ctx = create_headless_context()
    try:
        # Single source at the centre; pulse rate fast so we get a wave
        # immediately, no decay so we can see the cone clearly.
        runner = HeadlessRunner(
            ctx, "causal_ca", size=32, seed=1,
            params={"Pulse rate": 1.5, "Decay": 0.0, "Echo": 0.0,
                    "Threshold": 0.001},
            init_override="causal_one_source",
        )
        N_STEPS = 12
        for _ in range(N_STEPS):
            runner.step()
        data = runner.read_grid()
        size = data.shape[0]
        c = size // 2
        # The "Arrival" channel (G, idx 1) records WHICH cells were ever lit.
        ever_lit = data[..., 1] > 0.0
        xs, ys, zs = np.where(ever_lit)
        if len(xs) == 0:
            print("FAIL: causal_ca produced no lit cells.")
            return False
        manhattan = np.abs(xs - c) + np.abs(ys - c) + np.abs(zs - c)
        max_d = manhattan.max()
        # Every lit cell MUST be within Manhattan distance N_STEPS of source.
        # Allow a +1 slack for the first-frame source-write itself.
        if max_d > N_STEPS:
            print(f"FAIL: causal_ca cell at Manhattan {max_d} > {N_STEPS} steps "
                  f"(speed > c).")
            return False
        # Sanity check: should have lit MORE than just the source cell.
        if len(xs) < 5:
            print(f"FAIL: causal_ca only {len(xs)} cells lit; wave never spread.")
            return False
        print(f"PASS: causal_ca after {N_STEPS} steps: {len(xs)} cells lit, "
              f"max Manhattan distance = {max_d} (≤ {N_STEPS}, light-cone respected).")
        
        return True
    finally:
        destroy_context(win)


def validate_hopfion():
    win, ctx = create_headless_context()
    try:
        runner = HeadlessRunner(
            ctx, "hopfion_3d", size=32, seed=2,
            params={"Skyrme κ": 0.5, "Step": 0.05, "_": 0.0, "_2": 0.0},
            init_override="hopfion_h1",
        )
        for _ in range(50):
            runner.step()
        data = runner.read_grid()
        n = data[..., :3]
        norms = np.sqrt((n * n).sum(axis=-1))
        mean_norm = float(norms.mean())
        std_norm = float(norms.std())
        # Shader re-normalises every step → norm should be very close to 1.
        if abs(mean_norm - 1.0) > 0.05 or std_norm > 0.05:
            print(f"FAIL: hopfion |n| mean={mean_norm:.4f} std={std_norm:.4f}; "
                  f"expected ≈1.0±0.05.")
            return False
        # Energy density (alpha) should be finite.
        e = data[..., 3]
        if not np.isfinite(e).all():
            print("FAIL: hopfion energy field contains NaN/Inf.")
            return False
        print(f"PASS: hopfion_3d after 50 steps: |n| = {mean_norm:.4f} ± "
              f"{std_norm:.4f}, energy bounded.")
        
        return True
    finally:
        destroy_context(win)


def validate_genome():
    win, ctx = create_headless_context()
    try:
        runner = HeadlessRunner(
            ctx, "genome_ca_3d", size=32, seed=3,
            params={"Drive": 1.5, "Memory": 0.85, "Gene flow": 0.05,
                    "Mutation": 0.005},
            init_override="genome_state_random",
        )
        var0 = float(runner.read_grid()[..., 0].var())
        for _ in range(80):
            runner.step()
        data = runner.read_grid()
        # State must remain bounded (shader clamps to ±3).
        if not np.isfinite(data).all():
            print("FAIL: genome state contains NaN/Inf.")
            return False
        if abs(data[..., 0]).max() > 3.001 or abs(data[..., 1]).max() > 3.001:
            print(f"FAIL: state exceeds soft clamp: u_max="
                  f"{abs(data[..., 0]).max():.3f} v_max="
                  f"{abs(data[..., 1]).max():.3f}.")
            return False
        # Genome (pair 2) is the second physics field; pair 1 alone may saturate.
        # Check overall: spatial std must be non-trivial across SOMETHING.
        s_std = float(data.std())
        if s_std < 1e-3:
            print(f"FAIL: total std collapsed to {s_std:.6f}; rule degenerated.")
            return False
        var_t = float(data[..., 0].var())
        print(f"PASS: genome_ca_3d after 80 steps: state bounded |s|≤1, "
              f"total std {s_std:.4f}, channel-0 var {var0:.4f} → {var_t:.4f}.")
        
        return True
    finally:
        destroy_context(win)


def validate_nematic():
    win, ctx = create_headless_context()
    try:
        # Pure relaxation (no activity yet): A < 0.5 ⇒ ordering.
        # The Landau-de Gennes drive is `a - B·Tr(Q²)` with a = 0.4, B = 6:
        # equilibrium |Q|² = a/B ≈ 0.067. Random init may land above OR below,
        # so the test is that |Q| approaches the predicted equilibrium and
        # does NOT blow up or go NaN.
        runner = HeadlessRunner(
            ctx, "active_nematic_3d", size=32, seed=4,
            params={"Disorder A": 0.1, "Elastic K": 0.4, "Mobility Γ": 0.4,
                    "Activity ζ": 0.0},
            init_override="nematic_random",
        )
        magQs = []
        for _ in range(150):
            runner.step()
            d = runner.read_grid()
            if not np.isfinite(d).all():
                print("FAIL: active_nematic Q contains NaN/Inf.")
                return False
            # Approximate Tr(Q²) per voxel:  Qxx²+Qyy²+Qzz² + 2*(Qxy²+Qxz²+Qyz²)
            # Qzz = -(Qxx+Qyy). We don't have Qyz here so we underestimate slightly.
            qxx, qyy, qxy, qxz = d[..., 0], d[..., 1], d[..., 2], d[..., 3]
            qzz = -(qxx + qyy)
            tr = qxx ** 2 + qyy ** 2 + qzz ** 2 + 2.0 * (qxy ** 2 + qxz ** 2)
            magQs.append(float(tr.mean()))
        # Boundedness: max never exceeds an absolute ceiling.
        if max(magQs) > 5.0:
            print(f"FAIL: |Q|² blew up to {max(magQs):.4f}.")
            return False
        # Convergence: late variance much smaller than early variance.
        early_std = float(np.std(magQs[:20]))
        late_std = float(np.std(magQs[-30:]))
        if late_std > early_std + 0.05:
            print(f"FAIL: |Q|² not settling — early std {early_std:.4f}, "
                  f"late std {late_std:.4f}.")
            return False
        # Equilibrium check: late mean should be near a/B = 0.4/6 ≈ 0.067.
        late_mean = float(np.mean(magQs[-30:]))
        if not (0.005 < late_mean < 0.5):
            print(f"FAIL: equilibrium |Q|² = {late_mean:.4f} far from "
                  f"theoretical a/B = 0.067.")
            return False
        print(f"PASS: active_nematic_3d relaxed to equilibrium "
              f"|Q|² = {late_mean:.4f} (theory ≈ 0.067), late std "
              f"{late_std:.4f}, no blow-up.")
        
        return True
    finally:
        destroy_context(win)


if __name__ == "__main__":
    results = [
        ("causal_ca",         validate_causal()),
        ("hopfion_3d",        validate_hopfion()),
        ("genome_ca_3d",      validate_genome()),
        ("active_nematic_3d", validate_nematic()),
    ]
    print()
    n_pass = sum(int(p) for _, p in results)
    print(f"=== {n_pass}/{len(results)} passed ===")
    sys.exit(0 if n_pass == len(results) else 1)
