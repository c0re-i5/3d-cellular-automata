"""Limit-case probe.

Three targeted sanity checks per rule:

  1. DETERMINISM   Run the rule twice with identical (ctx, seed, IC,
                   params, dt).  The two final states must agree to
                   within float round-off.  Failures point to:
                     - reads from uninitialized memory
                     - frame-counter ordering bugs
                     - shader-side time queries that escape the
                       harness (gl_FragCoord/timestamp uniforms)
                     - per-frame RNG seeded from wall clock

  2. EMPTY_IC      Initialize the grid to all zeros and evolve K
                   steps.  Most rules should stay at zero.  A non-
                   trivial response indicates:
                     - an additive bias / source term that fires even
                       with no fuel (the "fire from nothing" class)
                     - boundary / index bug that pulls in stale data
                     - a forced source coded with a literal threshold
                       on a quantity that's exactly zero (sign edge)
                   We allow a small response for rules with explicit
                   stochastic spontaneous-generation (lightning, drop,
                   regrowth -- listed in _SPONTANEOUS_OK).

  3. DAMPING_ZERO  For rules with a clearly named damping/decay/loss
                   parameter, set it to 0 and verify the rule's
                   conservation gets BETTER, not worse.  A damping
                   term that ADDS energy when zeroed indicates a
                   sign error.

Usage:
  python -m ca_debug.limits
  python -m ca_debug.limits --rules wave_3d,fire,em_wave
  python -m ca_debug.limits --probes determinism,empty
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from typing import Any

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4, 'n/a': 5}

# Rules that legitimately respond to an all-zero IC because they have
# explicit spontaneous-generation in the shader (lightning strikes,
# regrowth, particle injection from boundaries, etc.).  An empty-IC
# response is informative but not a bug for these.
_SPONTANEOUS_OK = {
    'forest_fire_3d',     # lightning_f
    'sandpile_3d',        # drop_p (random grain drops)
    'sandbox',            # interactive deposition
    'eden_3d',            # boundary growth from any seed
    # Gray-Scott / Schnakenberg / Brusselator: feed terms F*(1-u) etc
    # are non-zero at u=0 by design (the substrate is continuously
    # replenished).  Not a bug.
    'reaction_diffusion_3d',
    'gray_scott_worms',
    'schnakenberg_3d',
    'brusselator_3d',
    # Rules with non-zero equilibrium fixed points: zero IC is *off-
    # equilibrium* and they relax toward their fixed point, which is
    # the right behavior, not a bug.
    'fitzhugh_nagumo_3d',     # FHN fixed point at (u*,v*)
    'rayleigh_benard_3d',     # T equilibrium between plates
    'predator_prey_lattice_3d',
    'flocking_3d',            # rho has floor
    # BZ: catalyst initial value is a constant (channel default 0.5).
    'bz_spiral_waves', 'bz_turbulence', 'bz_excitable',
    'flagship_glyph_bz', 'flagship_iso_bz',
    # Phase / angle fields where "zero" isn't a valid resting state.
    'kuramoto_3d', 'xy_spin_3d', 'hopfion_3d',
    # Lenia kernel produces non-trivial response at u=0 if growth
    # function maps 0 to non-zero.
    'lenia_3d', 'lenia_multi', 'lenia_geminium',
    # Element CA: vacuum is element id 0 but packed channels are
    # nonzero (mass, energy, ...).  Vacuum encoding is rule-specific.
    'element_ca', 'element_metals', 'element_na_water',
    # Margolus partition: alternating offset means "empty" state
    # depends on phase parity.
    'margolus_3d',
    # Ising: spin field encoded as +/-1, so 0 isn't a state -- we'd
    # need spin=+1 IC to test.  Empty-IC noise here isn't meaningful.
    'ising_3d',
    # Stochastic position-noise rules (Greenberg-Hastings, nucleation,
    # erosion, viscous fingers): seeded at IC time with hash(pos),
    # which is non-zero even with zero starting field.
    'greenberg_hastings_3d', 'nucleation', 'erosion', 'viscous_fingers',
    'volcanic_3d',
}

# Param-name patterns that mean "damping / decay / loss".  Setting
# these to zero should not make the rule LESS conservative.
_DAMPING_RE = re.compile(
    r'\b(damping|damp|decay|loss|dissip|friction|drag|cooling)\b',
    re.IGNORECASE)

# Tolerance: float16 round-trip noise can leave ~1e-3 relative diff
# even for "identical" runs.  Determinism failure is anything above.
DETERMINISM_TOL = 1e-3
# Empty-IC density above this for a non-whitelisted rule: bug suspect.
# We use 0.05 (5%) -- enough to catch real source-from-nothing leaks
# while ignoring single-voxel boundary index errors.
EMPTY_IC_TOL = 0.05


def _evolve(ctx, rule: str, *, size: int, steps: int, seed: int,
            params_override=None, init_zero: bool = False) -> np.ndarray:
    from test_harness import HeadlessRunner
    r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                       params=params_override)
    if init_zero:
        # Wipe both texture pairs.
        zero = np.zeros((size, size, size, 4), dtype=np.float32)
        enc = (zero.astype(np.float32 if r._tex_np_dtype == np.float32
                           else np.float16, copy=False).tobytes())
        r.tex_a.write(enc)
        if hasattr(r, 'tex_a2') and r.tex_a2 is not None:
            r.tex_a2.write(enc)
        r.ping = 0
    for _ in range(steps):
        r.step()
    g = r.read_grid().astype(np.float32)
    if hasattr(r, 'release'):
        try: r.release()
        except Exception: pass  # noqa: BLE001  GL resource release, never fatal
    return g


def _relerr(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a))
    if denom < 1e-9:
        # Both should be near zero -- use absolute scale.
        return float(np.linalg.norm(a - b)) / max(1.0, np.sqrt(a.size))
    return float(np.linalg.norm(a - b)) / denom


def _conservation_sum(g: np.ndarray) -> np.ndarray:
    """Return per-channel sum-of-squares (a generic conserved-energy
    proxy)."""
    return np.array([float((g[..., c] ** 2).sum())
                     for c in range(g.shape[-1])], dtype=np.float64)


def _run_one(ctx, rule: str, args) -> dict[str, Any]:
    from simulator import _resolve_composed_preset
    preset = _resolve_composed_preset(rule)
    params_default = dict(preset.get('params') or {})

    selected = set(p.strip() for p in (args.probes or '').split(',') if p.strip())
    do_det = (not selected) or 'determinism' in selected
    do_empty = (not selected) or 'empty' in selected
    do_damping = (not selected) or 'damping' in selected

    sub: dict[str, dict] = {}
    sev = 'ok'
    flags: list[str] = []

    # 1. Determinism.
    if do_det:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                g1 = _evolve(ctx, rule, size=args.size, steps=args.steps,
                             seed=args.seed)
                g2 = _evolve(ctx, rule, size=args.size, steps=args.steps,
                             seed=args.seed)
            err = _relerr(g1, g2)
            sub['determinism'] = {'err': err}
            if not np.isfinite(err) or err > DETERMINISM_TOL:
                sev = _worst(sev, 'crit' if err > 0.10 else 'high')
                flags.append(f'NONDET={err:.2e}')
        except Exception as e:  # noqa: BLE001  per-rule trial may crash, record error and continue
            sub['determinism'] = {'error': f'{type(e).__name__}: {e}'}
            sev = _worst(sev, 'err')
            flags.append(f'DET_CRASH:{type(e).__name__}')

    # 2. Empty-IC.
    if do_empty:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                g0 = _evolve(ctx, rule, size=args.size, steps=args.steps,
                             seed=args.seed, init_zero=True)
            mag = float(np.abs(g0).sum())
            voxels = float(g0[..., 0].size)
            density = mag / voxels  # average abs value per voxel per channel
            sub['empty_ic'] = {'abs_sum': mag, 'density': density}
            if density > EMPTY_IC_TOL and rule not in _SPONTANEOUS_OK:
                sev = _worst(sev, 'high' if density > 0.1 else 'med')
                flags.append(f'EMPTY_IC_NONZERO={density:.3f}')
        except Exception as e:  # noqa: BLE001  per-rule trial may crash, record error and continue
            sub['empty_ic'] = {'error': f'{type(e).__name__}: {e}'}
            flags.append(f'EMPTY_CRASH:{type(e).__name__}')

    # 3. Damping-zero.
    if do_damping:
        damping_params = [p for p in params_default
                          if _DAMPING_RE.search(p)]
        if damping_params:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    g_base = _evolve(ctx, rule, size=args.size,
                                     steps=args.steps, seed=args.seed)
                    override = dict(params_default)
                    for p in damping_params:
                        override[p] = 0.0
                    g_nodamp = _evolve(ctx, rule, size=args.size,
                                       steps=args.steps, seed=args.seed,
                                       params_override=override)
                # IC energy comparison.
                g_ic = _evolve(ctx, rule, size=args.size, steps=0,
                               seed=args.seed)
                e_ic = _conservation_sum(g_ic).sum()
                e_base = _conservation_sum(g_base).sum()
                e_nodamp = _conservation_sum(g_nodamp).sum()
                # With damping ON the system should lose energy or hold
                # it.  With damping OFF the system should hold or gain.
                # If e_nodamp < e_base * 0.8 then zeroing damping made
                # the rule LOSE more energy -- sign error candidate.
                conserved_better = (e_nodamp >= e_base * 0.8)
                sub['damping'] = {
                    'damping_params': damping_params,
                    'e_ic': float(e_ic),
                    'e_with_damping': float(e_base),
                    'e_zero_damping': float(e_nodamp),
                    'conserved_better': bool(conserved_better),
                }
                if not conserved_better and e_ic > 1e-6:
                    sev = _worst(sev, 'high')
                    ratio = e_nodamp / max(e_base, 1e-9)
                    flags.append(
                        f'DAMP_SIGN?({",".join(damping_params)}):'
                        f'{ratio:.2f}x')
            except Exception as e:  # noqa: BLE001  per-rule trial may crash, record error and continue
                sub['damping'] = {'error': f'{type(e).__name__}: {e}'}
                flags.append(f'DAMP_CRASH:{type(e).__name__}')
        else:
            sub['damping'] = {'skipped': True, 'reason': 'no damping param'}

    return {'rule': rule, 'sub': sub,
            'grade': {'severity': sev, 'flags': flags}}


def _worst(a: str, b: str) -> str:
    return a if _SEV_ORDER[a] < _SEV_ORDER[b] else b


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS, _resolve_composed_preset
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = []
    for r in sorted(RULE_PRESETS.keys()):
        try:
            preset = _resolve_composed_preset(r)
        except Exception:  # noqa: BLE001  preset lookup failure -> caller falls back
            continue
        if preset.get('kind') == 'viewport':
            continue
        if preset.get('agent_count') or 'entity_arena' in preset:
            continue
        if (preset.get('passes') or [{}])[0].get('kind') == 'particle':
            continue
        rules.append(r)
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = set(s.strip() for s in args.skip.split(',') if s.strip())
        rules = [r for r in rules if r not in skip_set]
    return rules


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rules')
    ap.add_argument('--probes', help='Subset of {determinism,empty,damping}.')
    ap.add_argument('--size', type=int, default=24)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=8675309)
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='med')
    ap.add_argument('--json')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    window, ctx = create_headless_context()

    rules = _select_rules(args)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<40}")
        sys.stdout.flush()
        try:
            row = _run_one(ctx, rule, args)
        except Exception as e:  # noqa: BLE001  per-rule trial may crash, record error and continue
            row = {'rule': rule, 'sub': {},
                   'grade': {'severity': 'err',
                             'flags': [f'CRASH:{type(e).__name__}']}}
        rows.append(row)
    sys.stdout.write('\r' + ' ' * 60 + '\r')
    elapsed = time.perf_counter() - t0

    rows.sort(key=lambda r: (_SEV_ORDER[r['grade']['severity']], r['rule']))

    min_sev = _SEV_ORDER[args.severity]
    print(f"\nLimit-case probe (size={args.size}, steps={args.steps}) "
          f"-- {elapsed:.1f}s")
    print(f"{'SEV':<5}  {'RULE':<32}  FLAGS")
    print('-' * 96)
    by_sev: dict[str, int] = {}
    for r in rows:
        sev = r['grade']['severity']
        by_sev[sev] = by_sev.get(sev, 0) + 1
        if _SEV_ORDER[sev] > min_sev:
            continue
        flags = ' | '.join(r['grade']['flags'])
        print(f"{sev:<5}  {r['rule']:<32}  {flags}")

    print()
    print('Summary:', ' '.join(f'{k}={by_sev.get(k, 0)}' for k in
                                ('crit', 'high', 'med', 'ok', 'n/a', 'err')))

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(rows, fh, indent=2, default=str)
        print(f"Wrote {args.json}")

    return 0 if by_sev.get('err', 0) == 0 and by_sev.get('crit', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
