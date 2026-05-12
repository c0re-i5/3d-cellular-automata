#!/usr/bin/env python3
"""Validate Langton's ant 3D — the agent SSBO infrastructure correctness
test.

Method:
  1. Initialise a blank voxel grid (all 0s) with one ant at the centre.
  2. Step N times.
  3. Verify the key invariant: total # of cells flipped between step k
     and step k+1 equals exactly agent_count (1) at every step.
  4. Verify the ant's position is reachable from the start in N moves
     (Manhattan distance <= N).

PASS if both invariants hold over a 200-step run.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_harness import create_headless_context, HeadlessRunner


def main():
    window, ctx = create_headless_context()
    try:
        size = 32
        runner = HeadlessRunner(ctx, "langton_ant_3d", size=size, seed=1)
        steps = 200

        prev = runner.read_grid()[..., 0].astype(np.int32)
        # Sanity: blank grid initially.
        if int(prev.sum()) != 0:
            print(f"FAIL: initial grid not blank (sum={prev.sum()})")
            return 1

        # Total cells ever flipped should grow monotonically while the ant
        # is on previously-untouched cells, then plateau when it revisits
        # old cells (each revisit flips back to 0).
        for s in range(1, steps + 1):
            runner.step()
            cur = runner.read_grid()[..., 0].astype(np.int32)
            diff = np.count_nonzero(cur != prev)
            if diff != 1:
                print(f"FAIL: step {s} changed {diff} cells (expected 1)")
                # Where did it change?
                idx = np.argwhere(cur != prev)
                print(f"  changed indices: {idx[:5].tolist()}")
                return 1
            prev = cur

        # Final grid should have a small connected trail of ~steps cells
        # in {0,1}. Distance from start cannot exceed steps.
        n_set = int(prev.sum())
        print(f"After {steps} steps: {n_set} cells set to 1, "
              f"{(prev == 0).sum() - (size**3 - n_set - n_set)} unflipped pristine.")
        # Read back agent state and verify position is finite.
        if runner.agent_ssbo is None:
            print("FAIL: agent_ssbo not allocated")
            return 1
        rec = np.frombuffer(runner.agent_ssbo.read(), dtype=np.int32).reshape(-1, 8)
        ax, ay, az, ad = rec[0, 0], rec[0, 1], rec[0, 2], rec[0, 3]
        c = size // 2
        manhattan = abs(int(ax) - c) + abs(int(ay) - c) + abs(int(az) - c)
        # Account for wraparound: shortest cyclic distance.
        def cyc(d):
            return min(abs(d), size - abs(d))
        cyc_dist = cyc(int(ax) - c) + cyc(int(ay) - c) + cyc(int(az) - c)
        print(f"Final agent: pos=({ax},{ay},{az}) dir={ad} "
              f"Manhattan={manhattan} Cyclic={cyc_dist}")
        if cyc_dist > steps:
            print(f"FAIL: agent travelled further than possible "
                  f"({cyc_dist} > {steps})")
            return 1
        if not (0 <= ad < 6):
            print(f"FAIL: invalid direction id {ad}")
            return 1

        print(f"\nPASS: 3D Langton's ant flipped exactly 1 cell/step "
              f"for {steps} steps, agent state remained valid.")
        return 0
    finally:
        import glfw
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    sys.exit(main())
