"""Probe #19 — entity-arena runtime invariants.

For every preset that uses the ``entity_arena`` substrate (currently
``wandering_voxels_3d`` and ``predator_prey_3d``), run a short
``HeadlessRunner`` simulation and verify engine-level invariants on the
GPU-resident entity buffer at multiple checkpoints (init / mid / end).

Motivation: the predator_prey_3d trap (Probe #18) showed that the
entity-arena code path was not systematically validated end-to-end.
This probe sits one level above unit tests — it actually drives the
full simulator pipeline and reads back the SSBO state to confirm:

  S1  slot 0 stays dead (reserved sentinel never overwritten)
  S2  no NaN/Inf in pos_radius / vel_energy / genome
  S3  alive positions in [0, size) per axis (toroidal-wrap respected)
  S4  alive radius > 0
  S5  alive |velocity| bounded by a sane cap
  S6  alive kind ∈ allowed set for this preset
  S7  alive team ∈ [0, team_count)
  S8  free-list partition: every slot with kind==0 (except 0) is on the
      free list; every slot with kind!=0 is not; no duplicates
  S9  alive_count ≤ max_entities  (catches spawn-without-cap leaks)
  S10 (wandering only — no births/deaths) alive_count constant

Exit code 0 iff every preset passes every invariant.

Usage::

    python -m ca_debug.entity_arena_invariants
    python -m ca_debug.entity_arena_invariants --steps 60 --size 48
    python -m ca_debug.entity_arena_invariants --rules predator_prey_3d --json /tmp/eai.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

# Allowed kind values per preset. Kept here (not derived) so the probe
# itself is a documented spec — new arena presets must opt in.
_ALLOWED_KINDS = {
    'wandering_voxels_3d': {1},
    'predator_prey_3d':    {1, 2},  # 1=prey, 2=predator
}

# Generous velocity cap (cells / step). Real values are bounded by
# `genome.x` (~few cells/step); 100 catches genuine blow-ups while
# tolerating short transients from random init.
_VMAX_CELLS_PER_STEP = 100.0


def _candidate_presets():
    """All RULE_PRESETS entries with an entity_arena config."""
    from simulator import RULE_PRESETS, _resolve_composed_preset
    out = []
    for name in RULE_PRESETS:
        p = _resolve_composed_preset(name)
        if 'entity_arena' in p:
            out.append(name)
    return sorted(out)


def _check_arena(arena, preset_name, *, size, label):
    """Pull entity state and check S1..S9. Returns list[str] of violations."""
    arena.pull_entities()
    ents  = arena.entities
    kinds = ents['ktrf'][:, 0]
    teams = ents['ktrf'][:, 1]
    pos   = ents['pos_radius'][:, :3]
    rad   = ents['pos_radius'][:, 3]
    vel   = ents['vel_energy'][:, :3]
    energy = ents['vel_energy'][:, 3]
    genome = ents['genome']

    bad = []
    tag = f"[{label}]"

    # S1: slot 0 reserved.
    if kinds[0] != 0:
        bad.append(f"{tag} S1 slot 0 not dead (kind={int(kinds[0])})")

    alive_mask = kinds != 0
    n_alive = int(alive_mask.sum())

    # S2: no NaN/Inf in any alive entity field.
    for fld_name, arr in (('pos_radius', ents['pos_radius']),
                          ('vel_energy', ents['vel_energy']),
                          ('genome',     genome)):
        slc = arr[alive_mask]
        nbad = int(np.count_nonzero(~np.isfinite(slc)))
        if nbad:
            bad.append(f"{tag} S2 {fld_name}: {nbad} non-finite values among {n_alive} alive")

    # S3: pos in [0, size). Only for alive (dead slots may be garbage).
    if n_alive:
        ap = pos[alive_mask]
        if np.any(ap < 0) or np.any(ap >= size):
            worst_lo = float(ap.min())
            worst_hi = float(ap.max())
            n_out = int(((ap < 0) | (ap >= size)).any(axis=1).sum())
            bad.append(f"{tag} S3 pos out of [0,{size}): {n_out}/{n_alive} entities "
                       f"(min={worst_lo:.3f}, max={worst_hi:.3f})")

    # S4: radius > 0 for alive.
    if n_alive:
        ar = rad[alive_mask]
        n_bad_r = int((ar <= 0).sum())
        if n_bad_r:
            bad.append(f"{tag} S4 non-positive radius: {n_bad_r}/{n_alive}")

    # S5: |vel| bounded.
    if n_alive:
        av = vel[alive_mask]
        speeds = np.linalg.norm(av, axis=1)
        n_fast = int((speeds > _VMAX_CELLS_PER_STEP).sum())
        if n_fast:
            bad.append(f"{tag} S5 |v|>{_VMAX_CELLS_PER_STEP}: {n_fast}/{n_alive} "
                       f"(worst={float(speeds.max()):.2f})")

    # S6: kind in allowed set.
    allowed = _ALLOWED_KINDS.get(preset_name, set(range(1, 256)))
    if n_alive:
        bad_kinds = set(int(k) for k in np.unique(kinds[alive_mask])) - allowed
        if bad_kinds:
            bad.append(f"{tag} S6 unexpected kinds {sorted(bad_kinds)} (allowed={sorted(allowed)})")

    # S7: team index < team_count.
    if n_alive:
        max_team = int(teams[alive_mask].max())
        if max_team >= arena.team_count:
            bad.append(f"{tag} S7 team index {max_team} >= team_count {arena.team_count}")

    # S8: free-list partition.
    fl = set(arena._free_slots)
    if len(fl) != len(arena._free_slots):
        bad.append(f"{tag} S8 free list has duplicates "
                   f"({len(arena._free_slots) - len(fl)} dups)")
    dead_slots = set(int(i) for i in np.where(kinds == 0)[0] if i != 0)
    leaked = dead_slots - fl
    illegal = fl - dead_slots  # slot in free list that is actually alive
    if leaked:
        bad.append(f"{tag} S8 {len(leaked)} dead slots NOT on free list (leak)")
    if illegal:
        bad.append(f"{tag} S8 {len(illegal)} free-list slots are actually alive")
    if 0 in fl:
        bad.append(f"{tag} S8 slot 0 (reserved) is on free list")

    # S9: alive_count ≤ max_entities (trivially true given fixed-size buffer,
    # but cheap sanity).
    if n_alive > arena.max_entities:
        bad.append(f"{tag} S9 alive_count {n_alive} > max_entities {arena.max_entities}")

    _ = energy  # held for future invariants; currently unused beyond NaN check.
    return bad, n_alive


def _run_preset(ctx, name, *, size, steps, seed):
    """Run preset for `steps` and check invariants at init / mid / end.

    Returns (alive_counts, violations) where violations is list[str].
    """
    from test_harness import HeadlessRunner
    runner = HeadlessRunner(ctx, name, size=size, seed=seed)
    try:
        violations = []
        # Checkpoint after init (frame 0, before any GPU pass).
        b0, n0 = _check_arena(runner.arena, name, size=size, label='init')
        violations.extend(b0)

        # Step a quarter / half / full.
        cps = [max(1, steps // 4), max(2, steps // 2), steps]
        alive = [n0]
        done = 0
        for ck_idx, target in enumerate(cps):
            while done < target:
                runner.step()
                done += 1
            label = f'step{done}'
            b, n = _check_arena(runner.arena, name, size=size, label=label)
            violations.extend(b)
            alive.append(n)

        # S10: wandering — no births/deaths in shader, alive must be constant.
        if name == 'wandering_voxels_3d':
            if len(set(alive)) > 1:
                violations.append(f"[final] S10 alive count varied across checkpoints: {alive}")

        return alive, violations
    finally:
        try:
            runner.release()
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rules', default='',
                        help='Comma-separated subset (default: every entity_arena preset).')
    parser.add_argument('--size',  type=int, default=48,
                        help='Grid size (default 48 — predator_prey wants room).')
    parser.add_argument('--steps', type=int, default=40,
                        help='Steps to run before final check (default 40).')
    parser.add_argument('--seed',  type=int, default=1001)
    parser.add_argument('--json',  default=None, help='Write report JSON to this path.')
    args = parser.parse_args(argv)

    import moderngl
    candidates = _candidate_presets()
    if args.rules:
        wanted = [r.strip() for r in args.rules.split(',') if r.strip()]
        unknown = [r for r in wanted if r not in candidates]
        if unknown:
            parser.error(f"unknown / non-entity-arena rules: {unknown}\n"
                         f"available: {candidates}")
        candidates = wanted

    print(f"entity-arena invariants — size={args.size} steps={args.steps} "
          f"seed={args.seed}  (presets: {len(candidates)})")

    ctx = moderngl.create_standalone_context(require=430)
    t0 = time.time()
    results = []
    any_bad = False
    try:
        for name in candidates:
            try:
                alive, viol = _run_preset(ctx, name, size=args.size,
                                          steps=args.steps, seed=args.seed)
            except Exception as e:  # noqa: BLE001
                viol = [f"[!!] runner exception: {type(e).__name__}: {e}"]
                alive = []
            sev = 'ok' if not viol else 'BAD'
            if viol:
                any_bad = True
            results.append({'rule': name, 'alive': alive, 'violations': viol, 'severity': sev})
            print(f"  {sev:3}  {name:28}  alive={alive}")
            for v in viol:
                print(f"        ⚠ {v}")
    finally:
        ctx.release()

    n_ok = sum(1 for r in results if r['severity'] == 'ok')
    print(f"\nSummary: ok={n_ok}  bad={len(results)-n_ok}  "
          f"(n={len(results)}, {time.time()-t0:.1f}s)")

    if args.json:
        with open(args.json, 'w') as fp:
            json.dump({'size': args.size, 'steps': args.steps, 'seed': args.seed,
                       'results': results}, fp, indent=2)
        print(f"wrote {args.json}")

    return 1 if any_bad else 0


if __name__ == '__main__':
    sys.exit(main())
