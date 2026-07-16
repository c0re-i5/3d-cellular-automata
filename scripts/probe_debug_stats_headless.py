#!/usr/bin/env python3
"""Bisect the headless debug-stats segfault seen in refine.py pass A.

Reproduces the exact call path:
    Simulator(size, rule, headless=True) -> _reset()
    _debug_enabled = True
    _dispatch_debug_stats()      # PyOpenGL glFenceSync + moderngl SSBO
    _harvest_debug_stats()       # glClientWaitSync + moderngl Buffer.read()

Run each phase behind a fault marker so we can see which native call
segfaults. Usage:
    .venv/bin/python -u scripts/probe_debug_stats_headless.py
"""
import faulthandler
import sys
import time

faulthandler.enable(all_threads=True)


def _mark(msg):
    print(f"[probe] {msg}", flush=True)


def main():
    _mark("import Simulator")
    from simulator import Simulator

    size = 96
    rule = "game_of_life_3d"
    _mark(f"build headless sim size={size} rule={rule}")
    sim = Simulator(size=size, rule=rule, headless=True)
    sim._reset()
    _mark("sim built + reset OK")

    sim._debug_enabled = True
    sim._debug_sample_interval = 1

    # Mimic refine.py pass_a_deep_replay exactly: step, then every
    # `interval` steps drain the previous fence and force a fresh sample.
    # No ctx.finish() — rely on the non-blocking fence path like refine.
    def _drain(s, timeout=2.0):
        t0 = time.time()
        while s._debug_fence is not None and time.time() - t0 < timeout:
            s._harvest_debug_stats()

    def _force_sample(s):
        s._debug_steps_since_sample = s._debug_sample_interval
        s._dispatch_debug_stats()
        _drain(s)

    steps = 1500
    _mark(f"loop {steps} steps, sample every {sim._debug_sample_interval}")
    for i in range(steps):
        sim._step_sim()
        if (i % sim._debug_sample_interval) == 0:
            _drain(sim, 0.5)
            _force_sample(sim)
        if i % 100 == 0:
            _mark(f"step {i} OK (history={len(sim._debug_history)})")

    _mark(f"LOOP COMPLETED {steps} steps without fault")
    return 0


if __name__ == "__main__":
    sys.exit(main())
