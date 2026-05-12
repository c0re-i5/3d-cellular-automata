"""Smoke-test for the predator_prey_3d preset.

Verifies the ecosystem dynamics actually function: prey graze food,
predators eat prey, both die when starving, both reproduce when fed.

Pass criteria:
  - Initial population matches what on_init spawns.
  - After N steps both populations are still > 0 (no instant collapse).
  - Total prey count CHANGES from initial (deaths + births occurred).
  - Food field is non-trivial (mean > 0, has spatial variance).
  - Composite ch0 has both positive (prey) and negative (predator) values.
  - Spatial hash is populated (predators can sense prey).
"""
from __future__ import annotations

import numpy as np
import moderngl

ctx = moderngl.create_standalone_context(require=430)

from test_harness import HeadlessRunner

SIZE = 128          # match the smallest realistic game size
STEPS = 100

print(f"[validate] creating runner: predator_prey_3d, size={SIZE}")
runner = HeadlessRunner(ctx, "predator_prey_3d", size=SIZE, seed=0)
print(f"[validate]   max_entities={runner.arena.max_entities} "
      f"hash_dim={runner.arena.hash_dim} hash_total={runner.arena.hash_total}")

runner.arena.pull_entities()
kinds0 = runner.arena.entities['ktrf'][:, 0]
prey0 = int(np.count_nonzero(kinds0 == 1))
pred0 = int(np.count_nonzero(kinds0 == 2))
print(f"[validate] initial: prey={prey0} predators={pred0}")
# At size=128 with PREY_DENSITY=0.0024, expect ~5000 prey, ~200 pred.
assert prey0 > 3000, f"expected ~5000 prey, got {prey0}"
assert pred0 > 100, f"expected ~200 predators, got {pred0}"

# Capture initial food state.
grid0 = runner.read_grid()
food0_mean = float(grid0[..., 3].mean())
food0_max = float(grid0[..., 3].max())
print(f"[validate] initial food: mean={food0_mean:.4f} max={food0_max:.4f}")
assert food0_mean > 0.01, "food field starts empty"

print(f"[validate] running {STEPS} steps...")
prey_history = []
pred_history = []
for s in range(STEPS):
    runner.step()
    if s % 10 == 9:
        runner.arena.pull_entities()
        kinds = runner.arena.entities['ktrf'][:, 0]
        prey_n = int(np.count_nonzero(kinds == 1))
        pred_n = int(np.count_nonzero(kinds == 2))
        prey_history.append(prey_n)
        pred_history.append(pred_n)
        print(f"[validate]   step {s+1:3d}: prey={prey_n:5d} pred={pred_n:4d}")

print(f"[validate] population history (every 10 steps):")
print(f"           prey: {prey_history}")
print(f"           pred: {pred_history}")

# Final state.
runner.arena.pull_entities()
kinds = runner.arena.entities['ktrf'][:, 0]
prey_f = int(np.count_nonzero(kinds == 1))
pred_f = int(np.count_nonzero(kinds == 2))
print(f"[validate] final: prey={prey_f} predators={pred_f}")
assert prey_f > 0, "prey went extinct — system unstable"
assert pred_f > 0, "predators went extinct — system unstable"
assert prey_f != prey0, "prey count never changed (no births/deaths happened)"

# Field state.
grid = runner.read_grid()
food = grid[..., 3]
print(f"[validate] final food: mean={food.mean():.4f} max={food.max():.4f} "
      f"std={food.std():.4f}")
assert food.mean() > 0.005, "food field collapsed entirely"
assert food.std() > 0.001, "food field is uniform (no spatial dynamics)"

ch0 = grid[..., 0]
prey_voxels = int((ch0 > 0.20).sum())  # exclude faint food background (~0.15 max)
pred_voxels = int((ch0 < -0.10).sum())
print(f"[validate] composite ch0: range=[{ch0.min():.3f}, {ch0.max():.3f}] "
      f"prey_voxels={prey_voxels} pred_voxels={pred_voxels}")
assert prey_voxels > 0, "no prey painted into composite channel"
assert pred_voxels > 0, "no predators painted into composite channel"

# Hash populated.
hash_count = np.frombuffer(runner.arena.hash_count_ssbo.read(), dtype=np.uint32)
populated = int((hash_count > 0).sum())
total_entries = int(hash_count.sum())
print(f"[validate] hash: {populated}/{hash_count.size} cells populated, "
      f"{total_entries} total entries")
assert total_entries > prey_f * 0.5, "hash missed most entities"

print("[validate] OK — predator-prey ecosystem validated.")
