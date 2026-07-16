#!/usr/bin/env python3
"""Stress the rule-switch / GL-resource-realloc path to reproduce the
intermittent heap-corruption segfault seen while browsing CAs.

Browsing the discovery list calls _load_discovery -> _change_rule, which
recompiles compute shaders, may resize the grid, and resets. If that path
leaks / double-frees / uses-after-free a moderngl object, the CPU heap gets
corrupted and Python later segfaults at a random line (observed: pure
dict.get and pure float math inside _draw_ui).

Run under a sharp allocator to fault near the corrupting write:
    MPLBACKEND=Agg PYTHONMALLOC=malloc PYTHONFAULTHANDLER=1 PYTHONPATH="$PWD" \
        .venv/bin/python -u scripts/stress_rule_switch.py --iters 4000
"""
import argparse
import faulthandler
import random
import sys
import time

faulthandler.enable(all_threads=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--steps', type=int, default=2,
                    help='steps to run after each switch')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from simulator import Simulator, RULE_PRESETS

    rng = random.Random(args.seed)
    names = list(RULE_PRESETS.keys())
    print(f"[stress] {len(names)} presets, {args.iters} switches, "
          f"size={args.size} steps={args.steps}", flush=True)

    sim = Simulator(size=args.size, rule=names[0], headless=True)
    sim._reset()

    t0 = time.time()
    for i in range(args.iters):
        rule = rng.choice(names)
        sim._change_rule(rule)
        for _ in range(args.steps):
            sim._step_sim()
        if i % 100 == 0:
            sim.ctx.finish()
            dt = time.time() - t0
            print(f"[stress] {i}/{args.iters}  last={rule}  "
                  f"{1000 * dt / max(i, 1):.1f} ms/switch", flush=True)

    sim.ctx.finish()
    print(f"[stress] DONE {args.iters} switches in {time.time() - t0:.1f}s "
          f"— no fault", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
