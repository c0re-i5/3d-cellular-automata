"""Smoke-test the entity_arena substrate end-to-end.

Spins up the wandering_voxels_3d preset in headless mode, runs N steps,
then checks:
  1. compile + dispatch succeeded (no GPU errors)
  2. entities actually moved from their spawn positions
  3. teams' centroids stayed roughly near home (i.e. attraction works)
  4. voxel field has non-zero pixels in both team colours after paint
"""
from __future__ import annotations

import sys
import numpy as np
import moderngl

# Headless context
ctx = moderngl.create_standalone_context(require=430)

from test_harness import HeadlessRunner

SIZE = 64
STEPS = 50

print(f"[validate] creating runner: wandering_voxels_3d, size={SIZE}")
runner = HeadlessRunner(ctx, "wandering_voxels_3d", size=SIZE, seed=0)
print(f"[validate]   max_entities={runner.arena.max_entities} "
      f"hash_dim={runner.arena.hash_dim} hash_total={runner.arena.hash_total}")

# Capture initial positions.
runner.arena.pull_entities()
init_pos = runner.arena.entities['pos_radius'][:, :3].copy()
init_kinds = runner.arena.entities['ktrf'][:, 0].copy()
init_teams = runner.arena.entities['ktrf'][:, 1].copy()
n_alive_init = int(np.count_nonzero(init_kinds))
print(f"[validate] initial alive entities: {n_alive_init}")
assert n_alive_init == 1600, f"expected 1600 entities, got {n_alive_init}"

# Run the sim.
print(f"[validate] running {STEPS} steps...")
for step in range(STEPS):
    runner.step()
print(f"[validate]   done")

# Read back entities.
runner.arena.pull_entities()
final_pos = runner.arena.entities['pos_radius'][:, :3].copy()
final_kinds = runner.arena.entities['ktrf'][:, 0].copy()
final_teams = runner.arena.entities['ktrf'][:, 1].copy()
n_alive_final = int(np.count_nonzero(final_kinds))
print(f"[validate] final alive entities: {n_alive_final}")
assert n_alive_final == n_alive_init, "entity count changed unexpectedly"

# Check entities actually moved.
alive_mask = final_kinds != 0
disp = np.linalg.norm(final_pos[alive_mask] - init_pos[alive_mask], axis=1)
print(f"[validate] displacement stats: "
      f"mean={disp.mean():.2f} max={disp.max():.2f} min={disp.min():.2f}")
assert disp.mean() > 0.5, f"entities barely moved (mean disp {disp.mean()})"

# Check team centroids stayed near home.
for team in (0, 1):
    sel = (final_teams == team) & alive_mask
    centroid = final_pos[sel].mean(axis=0)
    home = runner.arena.teams['spawn_pr'][team][:3]
    # Periodic distance
    d = centroid - home
    d -= SIZE * np.round(d / SIZE)
    dist = np.linalg.norm(d)
    print(f"[validate] team {team}: centroid={centroid.round(2)} "
          f"home={home.round(2)} dist_from_home={dist:.2f}")
    assert dist < SIZE * 0.30, \
        f"team {team} drifted too far from home: {dist:.2f}"

# Check field has been painted. New paint shader writes:
#   ch0: signed (team0 +, team1 -)   ch1: team0 only   ch2: team1 only   ch3: density
grid = runner.read_grid()
ch0 = grid[..., 0]
team0_mass = float(grid[..., 1].sum())
team1_mass = float(grid[..., 2].sum())
density    = float(grid[..., 3].sum())
pos_voxels = int((ch0 > 0.05).sum())
neg_voxels = int((ch0 < -0.05).sum())
print(f"[validate] field: team0_mass(G)={team0_mass:.1f} team1_mass(B)={team1_mass:.1f} "
      f"density(A)={density:.1f}")
print(f"[validate] signed ch0: positive voxels={pos_voxels} negative voxels={neg_voxels} "
      f"(min={ch0.min():.3f} max={ch0.max():.3f})")
assert team0_mass > 10.0, "team 0 (G) not painted"
assert team1_mass > 10.0, "team 1 (B) not painted"
assert pos_voxels > 5,    "no positive (team 0) voxels in signed channel"
assert neg_voxels > 5,    "no negative (team 1) voxels in signed channel"

# Check spatial hash actually populated (read first cell).
hash_count_raw = runner.arena.hash_count_ssbo.read()
hash_counts = np.frombuffer(hash_count_raw, dtype=np.uint32)
total_in_hash = int(hash_counts.sum())
print(f"[validate] spatial hash populated cells: "
      f"{(hash_counts > 0).sum()}/{len(hash_counts)} "
      f"total entries={total_in_hash}")
# Hash should hold roughly (n_alive) entries, give or take overflow drops.
assert total_in_hash >= int(n_alive_final * 0.85), \
    f"hash undercount: {total_in_hash} vs {n_alive_final} alive"

print("[validate] OK — entity_arena substrate validated.")
runner.release()
