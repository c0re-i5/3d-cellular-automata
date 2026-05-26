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

Currently registered (5):
  * reaction_diffusion_3d — Gray-Scott reduced to pure 3D diffusion via
    F=k=Dv=0, V≡0; verifies σ²(t) = σ²(0) + 2·D·t for an isotropic
    Gaussian initial condition (Green's function of the heat equation).
  * wave_3d — undriven, undamped standing wave cos(k·x); verifies
    the temporal correlation ⟨u(t)·u(0)⟩/⟨u(0)²⟩ matches cos(ω·t)
    with ω = c·|k| (d'Alembert plane-wave dispersion).
  * quantum_wavepacket — Schrödinger free-particle (V=0) Gaussian
    wave packet at rest; verifies the textbook quantum dispersion
    σ²(t) = σ₀² + (α·t/σ₀)² with α = ħ/(2m) measured on the
    probability density |Ψ|² (channel A).
  * sine_gordon_3d — small-amplitude (A≈0.05) Klein-Gordon limit
    sin(u) ≈ u; verifies the massive dispersion ω² = c²·k² + m² via
    the temporal correlation of a periodic standing wave.
  * brusselator_3d — Hopf-spiral linearisation around (U*, V*) = (A, B/A)
    with a spatially homogeneous perturbation; tests the reaction-kinetics
    integrator in isolation (∇² ≡ 0) against the closed-form trajectory
    ε(t) = ε₀·exp(½ t)·(cos, sin) of the complex eigenpair ½ ± i·√3/2.
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


@oracle('quantum_wavepacket')
def _quantum_free_gaussian_spread(ctx, size: int, seed: int, cap: int) -> dict:
    """Free Schrödinger Gaussian wave packet at rest.

    For ∂Ψ/∂t = i·α·∇²Ψ with α = ℏ/(2m) and V = 0, a minimum-uncertainty
    Gaussian ψ_R(x,0) = exp(-r²/(4σ_ψ²)), ψ_I(x,0) = 0 spreads as
        |Ψ|²(x,t) = (2π·σ(t)²)^(-3/2) · exp(-r²/(2σ(t)²))
        σ²(t) = σ_ψ² + (α·t/σ_ψ)²
    (textbook quantum dispersion of a Gaussian wave packet).  The 2nd
    moment of |Ψ|² equals σ_ψ²(t) per axis.

    Engine timestep convention: the Yee/FDTD staggered leapfrog advances
    ψ_I on even frames and ψ_R on odd frames, each by u_dt.  After N
    engine frames the joint (ψ_R, ψ_I) state has advanced by N·dt/2 in
    physical time, so t_phys = N · dt / 2.  We verified this empirically
    against the closed-form before committing.
    """
    from test_harness import HeadlessRunner

    rule = 'quantum_wavepacket'
    hbar_2m = 5.0       # large enough that 100 default steps give visible spread
    dt = 0.1            # max of dt_range; stability: dt < h²/(2α) = 4/10 = 0.4 ✓
    sigma_psi_voxels = 3.0   # ~6 voxel FWHM in |Ψ|²; tails decay to ~1e-4 by box edge
    amplitude = 0.5     # peak |ψ| well under the 1e3 clamp
    tol_rel = 0.05      # 5 % budget: 6-point Laplacian dispersion + leapfrog offset

    params = {'ħ/2m': hbar_2m, 'V strength': 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # IC: ψ_R = A·exp(-r²/(4σ²))  (wavefunction shape, so |Ψ|² has
        # second moment σ² directly).  ψ_I = V = 0; channel A is
        # repopulated by the shader on the first step.
        cc = (size - 1) * 0.5
        ax = np.arange(size, dtype=np.float64) - cc
        r2 = (ax[:, None, None] ** 2
              + ax[None, :, None] ** 2
              + ax[None, None, :] ** 2)
        psi_r = (amplitude * np.exp(-r2 / (4.0 * sigma_psi_voxels ** 2))
                 ).astype(np.float32)
        data = np.zeros((size, size, size, 4), dtype=np.float32)
        data[:, :, :, 0] = psi_r
        current_tex = r.tex_a if r.ping == 0 else r.tex_b
        current_tex.write(data.tobytes())

        for _ in range(cap):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'quantum_free_gaussian_spread',
                    'reason': f'NaN/Inf in grid after {cap} steps'}

        prob = g_t[..., 3].astype(np.float64)
        s_t_vox, m_t = _measure_isotropic_sigma_sq(prob)
        if not np.isfinite(s_t_vox) or m_t <= 0.0:
            return {'rule': rule, 'grade': 'err',
                    'oracle': 'quantum_free_gaussian_spread',
                    'reason': f'bad |Ψ|²: mass={m_t}, sigma²={s_t_vox}'}

        s0_phys = _voxel_sq_to_physical(sigma_psi_voxels ** 2, size)
        s_t_phys = _voxel_sq_to_physical(s_t_vox, size)
        t_phys = float(cap) * dt * 0.5   # Yee leapfrog half-step convention
        expected_phys = s0_phys + (hbar_2m * t_phys) ** 2 / s0_phys
        rel_err = abs(s_t_phys - expected_phys) / expected_phys

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'quantum_free_gaussian_spread',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'hbar_2m': hbar_2m,
            't_phys': t_phys,
            'sigma0_sq_phys_q': s0_phys,
            'sigma_t_sq_phys_q': s_t_phys,
            'expected_sq_phys_q': expected_phys,
            'rel_err': rel_err,
            'tol': tol_rel,
            'prob_mass': float(prob.sum()),
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


@oracle('sine_gordon_3d')
def _sine_gordon_klein_gordon_limit(ctx, size: int, seed: int, cap: int) -> dict:
    """Sine-Gordon in the small-amplitude (Klein-Gordon) limit.

    The shader integrates ∂²u/∂t² = c²·∇²u − m²·sin(u) − γ·v.  For
    |u| ≪ 1 the nonlinearity linearises to sin(u) ≈ u and the equation
    becomes the Klein-Gordon equation
        ∂²u/∂t² = c²·∇²u − m²·u
    with plane-wave dispersion
        ω² = c²·k² + m²   (massive Klein-Gordon).

    Periodic (toroidal) boundary: the discrete eigenmodes are
    cos(2π·n·i/L), so k_voxel = 2π·n/L exactly, and k_phys = k_voxel/h.
    We seed u = A·cos(k·x_0) with v = 0, damping = 0, drive = 0, A=0.05
    (sin(u)/u ≈ 1 − u²/6 ≈ 1 − 4·10⁻⁴, well into the linear regime),
    and verify C(t) = ⟨u(t)·u(0)⟩/⟨u(0)²⟩ matches cos(ω·t).
    """
    from test_harness import HeadlessRunner

    rule = 'sine_gordon_3d'
    c2 = 1.0
    m2 = 0.5            # massive enough that mass term dominates dispersion
    dt = 0.05           # min of dt_range; gives ~29 steps to phase π/3
    amplitude = 0.05    # small-amp linearisation: sin(u) ≈ u within 4·10⁻⁴
    n_mode = 4
    tol_rel = 0.05      # 5 % budget: 6-point Laplacian + symplectic-Euler order

    params = {'c²': c2, 'Damping γ': 0.0, 'Mass m²': m2, 'Drive': 0.0}

    h = _REF_SIZE / float(size)
    # Periodic BC → k_voxel = 2π·n/L (NOT π·n/(L-1) which assumed mirror).
    k_voxel = 2.0 * math.pi * n_mode / float(size)
    k_phys = k_voxel / h
    omega = math.sqrt(c2 * k_phys * k_phys + m2)

    target_phase = math.pi / 3.0
    steps = max(1, min(cap, int(round(target_phase / (omega * dt)))))
    actual_phase = omega * steps * dt
    expected_C = math.cos(actual_phase)

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        idx = np.arange(size, dtype=np.float64)
        cos_1d = (amplitude * np.cos(k_voxel * idx)).astype(np.float32)
        data = np.zeros((size, size, size, 4), dtype=np.float32)
        data[:, :, :, 0] = cos_1d[:, None, None]   # u
        # v (channel 1) and unused channels 2,3 all zero.
        current_tex = r.tex_a if r.ping == 0 else r.tex_b
        current_tex.write(data.tobytes())

        g0 = np.asarray(r.read_grid())
        u0 = g0[..., 0].astype(np.float64)
        norm0 = float((u0 * u0).sum())
        if norm0 <= 0.0:
            return {'rule': rule, 'grade': 'err',
                    'oracle': 'sine_gordon_klein_gordon_limit',
                    'reason': f'zero IC norm (norm0={norm0})'}

        for _ in range(steps):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'sine_gordon_klein_gordon_limit',
                    'reason': f'NaN/Inf in grid after {steps} steps'}

        u_t = g_t[..., 0].astype(np.float64)
        C_t = float((u_t * u0).sum() / norm0)
        amp_t = float(np.abs(u_t).max())
        rel_err = abs(C_t - expected_C) / max(abs(expected_C), 0.1)

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'sine_gordon_klein_gordon_limit',
            'grade': grade,
            'size': size,
            'steps': steps,
            'dt': dt,
            'c2': c2,
            'm2': m2,
            'n_mode': n_mode,
            'k_phys': k_phys,
            'omega': omega,
            'phase_rad': actual_phase,
            'C_measured': C_t,
            'C_expected': expected_C,
            'rel_err': rel_err,
            'tol': tol_rel,
            'amp_max': amp_t,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


@oracle('brusselator_3d')
def _brusselator_hopf_linear_growth(ctx, size: int, seed: int, cap: int) -> dict:
    """Brusselator linearised around the homogeneous fixed point (Hopf spiral).

    The Brusselator reaction equations are
        ∂U/∂t = Dᵤ·∇²U + A − (B+1)·U + U²·V
        ∂V/∂t = Dᵥ·∇²V       + B·U      − U²·V
    with homogeneous fixed point (U*, V*) = (A, B/A).  Linearising
    Ũ = U − A, Ṽ = V − B/A around the fixed point gives the Jacobian
        J = [[ B − 1,   A²  ],
             [ −B,     −A²  ]].
    For default A=1, B=3:  J = [[2, 1], [−3, −1]],  tr = 1,  det = 1,
    so the eigenvalues are λ = ½ ± i·√3/2 — a Hopf spiral with growth
    rate σ = ½ and angular frequency Ω = √3/2.

    With a *spatially homogeneous* perturbation the Laplacian is exactly
    zero everywhere, so diffusion drops out and the dynamics reduce to
    pure ODE integration of the reaction terms — a clean, independent
    test of the engine's reaction kinetics (complementary to
    `reaction_diffusion_3d` which tests pure diffusion).

    We seed (U, V) = (A + ε, B/A − 1.5·ε) — the real part of the right
    eigenvector for λ_+ = ½ + i·√3/2 — and verify that after t = N·dt
    both (Ũ, Ṽ) match the closed-form trajectory
        Ũ(t) = ε·exp(σt)·cos(Ωt)
        Ṽ(t) = ε·exp(σt)·[−1.5·cos(Ωt) + Ω/σ·sin(Ωt)·σ/Ω... ]
    Explicitly, from Re[c·v_+·exp(λ_+·t)] with c=1, v_+=(1, −1.5+i·√3/2):
        Ũ(t) = ε·exp(σt)·cos(Ωt)
        Ṽ(t) = ε·exp(σt)·[−1.5·cos(Ωt) − (√3/2)·sin(Ωt)·(−1)]
             = ε·exp(σt)·[−1.5·cos(Ωt) + (√3/2)·sin(Ωt)]
    (cross-checked: at t=0 gives (ε, −1.5ε) ✓).
    """
    from test_harness import HeadlessRunner

    rule = 'brusselator_3d'
    A_param = 1.0
    B_param = 3.0
    Du = 2.0
    Dv = 8.0
    # The shader integrates with forward Euler (symplectic on a spiral
    # over-rotates the growth eigenvalue).  Per-step ratio
    # |1+λ·dt|/exp(σ·dt) at dt=0.05 leaks ~6 % over 100 steps, which
    # blows the 5 % continuum-vs-discrete budget.  dt=0.01 (5× tighter)
    # brings FE within <1 % of the continuum trajectory.
    dt = 0.01
    epsilon = 1.0e-3    # quadratic terms ~ε²/A ≈ 1e-6 stay 1000× below linear
    tol_rel = 0.05      # 5 % budget: explicit-Euler O(dt²) phase + amplitude drift

    params = {'A': A_param, 'B': B_param, 'Du': Du, 'Dv': Dv}

    # Closed-form linear trajectory (J = [[B-1, A²], [-B, -A²]] at fixed pt).
    U_star = A_param
    V_star = B_param / A_param
    trJ = (B_param - 1.0) - A_param * A_param          # = 1 at defaults
    detJ = -A_param * A_param * (B_param - 1.0) + A_param * A_param * B_param  # = A²
    sigma = trJ / 2.0
    disc = trJ * trJ - 4.0 * detJ
    if disc >= 0.0:
        return {'rule': rule, 'grade': 'err',
                'oracle': 'brusselator_hopf_linear_growth',
                'reason': f'not in Hopf regime (disc={disc})'}
    Omega = math.sqrt(-disc) / 2.0                     # = √3/2 at defaults

    # IC: U = A + ε, V = B/A + ε·v_y where v_y = -1.5 (= (J11-σ)/J12·something)
    # is the real part of the right eigenvector for λ_+.
    # General formula: Re-part of v_+ has (1, (σ - J11)/J12) for J12 ≠ 0.
    J11 = B_param - 1.0
    J12 = A_param * A_param
    v_y = (sigma - J11) / J12                           # = -1.5 at defaults

    t_phys = float(cap) * dt
    growth = math.exp(sigma * t_phys)
    c_arg = Omega * t_phys
    cos_c = math.cos(c_arg)
    sin_c = math.sin(c_arg)
    # Closed-form (derived from Re[v_+ · exp(λ_+ t)] with c=1, v_+=(1, v_y + i·Ω/J12)):
    expected_dU = epsilon * growth * cos_c
    expected_dV = epsilon * growth * (v_y * cos_c - (Omega / J12) * sin_c)

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # Homogeneous IC ensures ∇²U = ∇²V ≡ 0 — diffusion drops out
        # entirely so this is a pure-ODE test of the reaction kinetics.
        data = np.zeros((size, size, size, 4), dtype=np.float32)
        data[:, :, :, 0] = np.float32(U_star + epsilon)
        data[:, :, :, 1] = np.float32(V_star + epsilon * v_y)
        current_tex = r.tex_a if r.ping == 0 else r.tex_b
        current_tex.write(data.tobytes())

        for _ in range(cap):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'brusselator_hopf_linear_growth',
                    'reason': f'NaN/Inf in grid after {cap} steps'}

        # Homogeneous dynamics: every voxel must hold the same value
        # (modulo fp32 noise).  Measure deviation from the fixed point
        # via mean, and homogeneity loss via spatial std.
        U_t_field = g_t[..., 0].astype(np.float64)
        V_t_field = g_t[..., 1].astype(np.float64)
        U_t_mean = float(U_t_field.mean())
        V_t_mean = float(V_t_field.mean())
        U_t_std = float(U_t_field.std())
        V_t_std = float(V_t_field.std())
        dU = U_t_mean - U_star
        dV = V_t_mean - V_star

        # Relative error vs the predicted (Ũ, Ṽ) trajectory, normalised
        # by ε·exp(σt) so growth doesn't artificially shrink small denoms.
        denom = epsilon * growth
        err_U = abs(dU - expected_dU) / denom
        err_V = abs(dV - expected_dV) / denom
        rel_err = max(err_U, err_V)

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'brusselator_hopf_linear_growth',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'A': A_param,
            'B': B_param,
            'sigma_hopf': sigma,
            'Omega_hopf': Omega,
            't_phys': t_phys,
            'growth_efold': growth,
            'epsilon': epsilon,
            'dU_measured': dU,
            'dU_expected': expected_dU,
            'dV_measured': dV,
            'dV_expected': expected_dV,
            'rel_err': rel_err,
            'tol': tol_rel,
            'homogeneity_std_U': U_t_std,
            'homogeneity_std_V': V_t_std,
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
                elif 'sigma_t_sq_phys_q' in row:
                    print(f'          σ²(0)={row["sigma0_sq_phys_q"]:.4f}  '
                          f'σ²(t)={row["sigma_t_sq_phys_q"]:.4f}  '
                          f'expected={row["expected_sq_phys_q"]:.4f}  '
                          f'rel_err={row["rel_err"]:.4f}  (tol={row["tol"]:.3f})')
                    print(f'          α=ℏ/2m={row["hbar_2m"]:.3f}  '
                          f't={row["t_phys"]:.3f}  prob_mass={row["prob_mass"]:.4e}')
                elif 'dU_measured' in row:
                    print(f'          σ={row["sigma_hopf"]:.4f}  Ω={row["Omega_hopf"]:.4f}  '
                          f't={row["t_phys"]:.3f}  growth={row["growth_efold"]:.4f}×  '
                          f'rel_err={row["rel_err"]:.4f}  (tol={row["tol"]:.3f})')
                    print(f'          ΔU={row["dU_measured"]:+.5e} vs {row["dU_expected"]:+.5e}  '
                          f'ΔV={row["dV_measured"]:+.5e} vs {row["dV_expected"]:+.5e}  '
                          f'hom_std=(U:{row["homogeneity_std_U"]:.2e}, V:{row["homogeneity_std_V"]:.2e})')
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
