"""Smoke test for the predator_prey_lattice_3d CA.

Runs the rule headless for 200 steps and asserts:
  - all four kinds (empty, food, prey, predator) coexist throughout
  - populations actually change over time (it's a real dynamic system)
  - no NaN / out-of-range values appear
  - energy stays in a reasonable range
"""
from __future__ import annotations

import moderngl
import numpy as np

ctx = moderngl.create_standalone_context(require=430)
from test_harness import HeadlessRunner

SIZE = 96
STEPS = 200
SAMPLE_EVERY = 20

print(f"[validate] size={SIZE}, steps={STEPS}")
runner = HeadlessRunner(ctx, "predator_prey_lattice_3d", size=SIZE, seed=0)

def counts(g):
    kind = np.round(g[..., 0] * 4).astype(int)
    c = np.bincount(kind.ravel(), minlength=4)
    return c[0], c[1], c[2], c[3]  # empty, food, prey, pred

g = runner.read_grid()
e0, f0, p0, d0 = counts(g)
print(f"[validate] initial: empty={e0:>7d} food={f0:>6d} prey={p0:>6d} pred={d0:>5d}")
assert f0 > 1000, "no food"
assert p0 > 500, "no prey"
assert d0 > 30,  "no predators"

history = []
for s in range(STEPS):
    runner.step()
    if (s + 1) % SAMPLE_EVERY == 0:
        g = runner.read_grid()
        e, f, p, d = counts(g)
        en = g[..., 1]
        print(f"[validate]   step {s+1:>3d}: empty={e:>7d} food={f:>6d} "
              f"prey={p:>6d} pred={d:>5d}  e_mean={en.mean():.3f}")
        history.append((e, f, p, d))
        # Sanity
        assert e + f + p + d == SIZE ** 3, f"voxels don't sum: {e}+{f}+{p}+{d}"
        assert not np.isnan(en).any(), "energy NaN"
        assert en.min() >= 0.0 and en.max() <= 1.501, \
            f"energy out of range: [{en.min()}, {en.max()}]"

# Final asserts on dynamics.
food_series = [h[1] for h in history]
prey_series = [h[2] for h in history]
pred_series = [h[3] for h in history]

print(f"[validate] food range : [{min(food_series):>6d}, {max(food_series):>6d}]")
print(f"[validate] prey range : [{min(prey_series):>6d}, {max(prey_series):>6d}]")
print(f"[validate] pred range : [{min(pred_series):>6d}, {max(pred_series):>6d}]")

# All three populations must be alive at end.
assert food_series[-1] > 100, "food extinct"
assert prey_series[-1] > 100, "prey extinct"
assert pred_series[-1] > 10,  "predators extinct"

# Populations must NOT be static (the rule must do something).
def changed(series, min_swing=0.10):
    lo, hi = min(series), max(series)
    if lo == 0:
        return hi > 100
    return (hi - lo) / lo >= min_swing

assert changed(food_series), "food count is static"
assert changed(prey_series), "prey count is static"
assert changed(pred_series), "pred count is static"

print("[validate] OK — lattice predator-prey system shows real dynamics.")
