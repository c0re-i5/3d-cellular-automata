"""Analytic-oracle probe (Probe #16, Level 5 of the correctness ladder).

The previous 15 probes verify that the simulator (a) doesn't crash, (b)
renders something, (c) is bit-stable across engine refactors, and (d)
satisfies general physics invariants (conservation, symmetry, etc.).
None of them verifies the engine produces the *quantitatively correct*
answer for a problem with a known closed-form solution.

This probe does.  Each registered rule has an oracle function that:

  1. Sets up a problem the engine can solve and analytic theory can
     predict (typically by overriding params + the initial condition).
  2. Runs the simulator headlessly for a fixed number of steps.
  3. Measures a scalar (or vector) observable from the resulting grid.
  4. Compares it to the closed-form analytic prediction.
  5. Grades by relative error against a tolerance.

Grades:
  err   construction or measurement crashed.
  crit  relative error exceeds ``tol`` (engine is quantitatively wrong).
  high  relative error in [tol, 2·tol) (borderline / drift warning).
  ok    relative error within tol.
  skip  no oracle registered for this rule.

Usage::

    python -m ca_debug.analytic_oracles
    python -m ca_debug.analytic_oracles --rules reaction_diffusion_3d
    python -m ca_debug.analytic_oracles --severity ok --json /tmp/ora.json

Currently registered (2):
  * reaction_diffusion_3d — Gray-Scott reduced to pure 3D diffusion via
    F=k=Dv=0, V≡0; verifies σ²(t) = σ²(0) + 2·D·t for an isotropic
    Gaussian initial condition (Green's function of the heat equation).
  * wave_3d — undriven, undamped standing wave cos(k·x); verifies
    the temporal correlation ⟨u(t)·u(0)⟩/⟨u(0)²⟩ matches cos(ω·t)
    with ω = c·|k| (d'Alembert plane-wave dispersion).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
import traceback

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'ok': 3, 'skip': 4}

# Reference grid size baked into the shader's continuum scaling
# (h_inv = u_size / REF_SIZE).  Must match COMPUTE_HEADER in simulator.py.
_REF_SIZE = 128.0


# ---------------------------------------------------------------------------
# Oracle registry
# ---------------------------------------------------------------------------

ORACLES: dict[str, callable] = {}


def oracle(rule: str):
    """Decorator: register `fn` as the analytic oracle for `rule`."""
    def deco(fn):
        ORACLES[rule] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_field(size: int, sigma_voxels: float, amplitude: float) -> np.ndarray:
    """Isotropic 3D Gaussian centred at the grid centre, in the R (U) channel."""
    c = (size - 1) * 0.5
    ax = np.arange(size, dtype=np.float64) - c
    r2 = ax[:, None, None]**2 + ax[None, :, None]**2 + ax[None, None, :]**2
    g = (amplitude * np.exp(-r2 / (2.0 * sigma_voxels * sigma_voxels))).astype(np.float32)
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    data[:, :, :, 0] = g
    return data


def _measure_isotropic_sigma_sq(field: np.ndarray) -> tuple[float, float]:
    """Return (sigma²-per-axis-in-voxels, total mass) of a positive 3D scalar field."""
    f = np.asarray(field, dtype=np.float64)
    f = np.clip(f, 0.0, None)
    mass = float(f.sum())
    if mass <= 0.0:
        return float('nan'), 0.0
    size = f.shape[0]
    c = (size - 1) * 0.5
    ax = np.arange(size, dtype=np.float64) - c
    r2 = ax[:, None, None]**2 + ax[None, :, None]**2 + ax[None, None, :]**2
    sigma_sq_voxels = float((f * r2).sum() / (3.0 * mass))
    return sigma_sq_voxels, mass


def _voxel_sq_to_physical(sigma_sq_voxels: float, size: int) -> float:
    """Convert per-axis variance from voxel² to reference (physical) units."""
    h = _REF_SIZE / float(size)
    return sigma_sq_voxels * h * h


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------

@oracle('reaction_diffusion_3d')
def _gray_scott_pure_diffusion(ctx, size: int, seed: int, cap: int) -> dict:
    """Gray-Scott reduced to pure heat equation.

    With F=k=0 and V≡0 the Gray-Scott update collapses to
        ∂U/∂t = D · ∇²U
    whose Green's function for an isotropic Gaussian initial condition
    grows the variance linearly:
        σ²(t) = σ²(0) + 2·D·t      (D in physical units, t = Σ dt_eff)

    We seed U as a localised Gaussian, V as zero, integrate, and verify
    the measured σ²(t) matches the analytic prediction within ``tol``.
    """
    from test_harness import HeadlessRunner

    rule = 'reaction_diffusion_3d'
    D = 0.16
    dt = 1.0           # well under CFL: limit ≈ 15 at size=64
    sigma0_voxels = 4.0
    amplitude = 0.5    # keep U(0) ≤ 1 so the shader's clamp never bites
    tol_rel = 0.05     # 5 % relative error budget

    # Override params: F=0, k=0 → no reaction; Dv=0 → V stays inert.
    params = {'Feed rate': 0.0, 'Kill rate': 0.0,
              'U diffusion': D, 'V diffusion': 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # Stomp the engine's default IC with our synthetic Gaussian.
        data = _gaussian_field(size, sigma0_voxels, amplitude)
        current_tex = r.tex_a if r.ping == 0 else r.tex_b
        current_tex.write(data.tobytes())

        # Sanity-check the IC the engine sees.
        g0 = np.asarray(r.read_grid())
        s0_vox, m0 = _measure_isotropic_sigma_sq(g0[..., 0])
        if not np.isfinite(s0_vox) or m0 <= 0.0:
            return {'rule': rule, 'grade': 'err',
                    'oracle': 'gray_scott_pure_diffusion',
                    'reason': f'bad IC: mass={m0}, sigma²={s0_vox}'}

        for _ in range(cap):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'gray_scott_pure_diffusion',
                    'reason': f'NaN/Inf in grid after {cap} steps'}

        s_t_vox, m_t = _measure_isotropic_sigma_sq(g_t[..., 0])
        # V channel should remain zero throughout — sanity check.
        v_max = float(np.abs(g_t[..., 1]).max())

        s0_phys = _voxel_sq_to_physical(s0_vox, size)
        s_t_phys = _voxel_sq_to_physical(s_t_vox, size)
        t_phys = float(cap) * dt
        expected_phys = s0_phys + 2.0 * D * t_phys
        rel_err = abs(s_t_phys - expected_phys) / expected_phys

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'gray_scott_pure_diffusion',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'D': D,
            'sigma0_sq_phys': s0_phys,
            'sigma_t_sq_phys': s_t_phys,
            'expected_sq_phys': expected_phys,
            'rel_err': rel_err,
            'tol': tol_rel,
            'mass_drift_rel': abs(m_t - m0) / max(m0, 1e-12),
            'v_channel_max': v_max,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


@oracle('wave_3d')
def _wave_3d_standing_wave(ctx, size: int, seed: int, cap: int) -> dict:
    """3D wave equation as an undriven, undamped standing wave.

    For the linear wave equation  ∂²u/∂t² = c²·∇²u  with damping=0 and no
    forcing, an eigenmode of the spatial Laplacian evolves as a perfect
    harmonic oscillator:
        u(x, t) = cos(k·x) · cos(ω·t),  ω = c·|k|.
    The temporal autocorrelation of u therefore satisfies
        C(t) = ⟨u(t)·u(0)⟩_x / ⟨u(0)²⟩_x = cos(ω·t).

    We seed an axis-aligned cosine that exactly matches the engine's
    mirror (Neumann) boundary (∂u/∂x = 0 at i=0 and i=L-1), step the
    engine, and verify C(t) matches the analytic prediction.

    Note: the `cap` argument is honoured as an upper bound on integration
    length; the actual step count is chosen so that ω·t ≈ π/3 (a
    well-conditioned phase, away from cos extrema).
    """
    from test_harness import HeadlessRunner

    rule = 'wave_3d'
    c_wave = 0.5        # safely inside CFL at all sizes (c·dt·h_inv ≪ 1/√3)
    dt = 0.15
    amplitude = 0.5     # < 1 keeps us far from the |u|<100 clamp
    n_mode = 4          # 4 half-wavelengths across the box
    tol_rel = 0.03      # 3 % budget covers spatial dispersion (≈ 0.3 %) + slack

    params = {'Wave speed': c_wave, 'Damping': 0.0,
              'Drive freq': 0.0, 'Drive amp': 0.0}

    # Continuum prediction.  h = REF/size; k in voxel units is π·n/(L-1),
    # so k_phys = k_voxel / h.  Mirror BC enforces ∂u/∂x = 0 → cos with
    # arg π·n·i/(L-1) is an exact discrete eigenmode of the boundary.
    h = _REF_SIZE / float(size)
    k_voxel = math.pi * n_mode / float(size - 1)
    k_phys = k_voxel / h
    omega = c_wave * k_phys

    target_phase = math.pi / 3.0
    steps = max(1, min(cap, int(round(target_phase / (omega * dt)))))
    actual_phase = omega * steps * dt
    expected_C = math.cos(actual_phase)

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # Build IC: cosine along axis 0, zero velocity.
        idx = np.arange(size, dtype=np.float64)
        cos_1d = amplitude * np.cos(k_voxel * idx).astype(np.float32)
        data = np.zeros((size, size, size, 4), dtype=np.float32)
        data[:, :, :, 0] = cos_1d[:, None, None]   # u
        # v (channel 1), B, A all zero.
        current_tex = r.tex_a if r.ping == 0 else r.tex_b
        current_tex.write(data.tobytes())

        g0 = np.asarray(r.read_grid())
        u0 = g0[..., 0].astype(np.float64)
        norm0 = float((u0 * u0).sum())
        if norm0 <= 0.0:
            return {'rule': rule, 'grade': 'err',
                    'oracle': 'wave_3d_standing_wave',
                    'reason': f'zero IC norm (norm0={norm0})'}

        for _ in range(steps):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'wave_3d_standing_wave',
                    'reason': f'NaN/Inf in grid after {steps} steps'}

        u_t = g_t[..., 0].astype(np.float64)
        C_t = float((u_t * u0).sum() / norm0)
        rel_err = abs(C_t - expected_C) / max(abs(expected_C), 0.1)

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'wave_3d_standing_wave',
            'grade': grade,
            'size': size,
            'steps': steps,
            'dt': dt,
            'c_wave': c_wave,
            'n_mode': n_mode,
            'k_phys': k_phys,
            'omega': omega,
            'phase_rad': actual_phase,
            'C_measured': C_t,
            'C_expected': expected_C,
            'rel_err': rel_err,
            'tol': tol_rel,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


# ---------------------------------------------------------------------------
# Probe driver
# ---------------------------------------------------------------------------

def _check_rule(ctx, rule: str, size: int, seed: int, cap: int) -> dict:
    fn = ORACLES.get(rule)
    if fn is None:
        return {'rule': rule, 'grade': 'skip',
                'reason': 'no analytic oracle registered'}
    try:
        return fn(ctx, size, seed, cap)
    except Exception as e:  # noqa: BLE001 — surface any oracle crash uniformly
        return {'rule': rule, 'grade': 'err',
                'reason': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc().splitlines()[-3:]}


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    if args.only_registered:
        return sorted(ORACLES.keys())
    rules = sorted(RULE_PRESETS.keys())
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all registered + skips).')
    ap.add_argument('--only-registered', action='store_true',
                    help='Test only rules with a registered oracle (suppresses skip rows).')
    ap.add_argument('--size', type=int, default=64,
                    help='Grid size (default: 64).')
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--cap', type=int, default=100,
                    help='Integration steps for the oracle (default: 100).')
    ap.add_argument('--skip', help='Comma-separated rules to omit entirely.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='high',
                    help='Min severity to print (default: high).')
    ap.add_argument('--json', help='Write per-rule report JSON.')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    _window, ctx = create_headless_context()

    rules = _select_rules(args)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<42}")
        sys.stdout.flush()
        rows.append(_check_rule(ctx, rule, args.size, args.seed, args.cap))
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    counts = {k: 0 for k in _SEV_ORDER}
    for row in rows:
        counts[row['grade']] = counts.get(row['grade'], 0) + 1

    print(f'\nanalytic-oracle probe — {len(rules)} rules in {elapsed:.1f}s')
    print(f'  registered oracles: {len(ORACLES)} '
          f'({", ".join(sorted(ORACLES.keys())) or "none"})')
    for g in ('err', 'crit', 'high', 'ok', 'skip'):
        print(f'    {g:<5}  {counts.get(g, 0):>5}')

    sev_cap = _SEV_ORDER[args.severity]
    flagged = [r for r in rows if _SEV_ORDER.get(r['grade'], 9) <= sev_cap]
    if flagged:
        print(f'\nflagged ({args.severity}+):  {len(flagged)} rules')
        for row in flagged:
            print(f'  [{row["grade"]:<4}] {row["rule"]}  '
                  f'oracle={row.get("oracle", "-")}')
            if row.get('reason'):
                print(f'          {row["reason"]}')
            if 'rel_err' in row:
                # Diagnostic line is oracle-specific; emit whichever
                # measured/expected pair the oracle produced.
                if 'sigma_t_sq_phys' in row:
                    print(f'          σ²(0)={row["sigma0_sq_phys"]:.4f}  '
                          f'σ²(t)={row["sigma_t_sq_phys"]:.4f}  '
                          f'expected={row["expected_sq_phys"]:.4f}  '
                          f'rel_err={row["rel_err"]:.4f}  (tol={row["tol"]:.3f})')
                    print(f'          mass_drift={row["mass_drift_rel"]:.2e}  '
                          f'V_max={row["v_channel_max"]:.2e}')
                elif 'C_measured' in row:
                    print(f'          ω={row["omega"]:.5f}  phase={row["phase_rad"]:.4f}rad  '
                          f'C(t)={row["C_measured"]:.4f}  '
                          f'expected={row["C_expected"]:.4f}  '
                          f'rel_err={row["rel_err"]:.4f}  (tol={row["tol"]:.3f})')
                else:
                    print(f'          rel_err={row["rel_err"]:.4f}  '
                          f'(tol={row.get("tol", "?")})')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'counts': counts, 'rows': rows, 'elapsed_s': elapsed,
                       'size': args.size, 'cap': args.cap,
                       'registered': sorted(ORACLES.keys())},
                      fh, indent=2, default=str)
        print(f'\nwrote {args.json}')

    return 1 if (counts['err'] + counts['crit']) else 0


if __name__ == '__main__':
    sys.exit(main())
