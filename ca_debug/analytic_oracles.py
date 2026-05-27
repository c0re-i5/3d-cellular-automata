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

Currently registered (8):
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
  * compressible_euler_3d — canonical Sod shock tube; verifies the
    right-moving shock travels at the exact Riemann speed S ≈ 1.75216
    (extruded along z, planar-averaged density crosses half-jump
    threshold).  Tests the LF flux + ideal-gas EOS in the nonlinear
    hyperbolic regime.
  * phase_separation — Cahn-Hilliard linear spinodal regime; seeds a
    low-mode plane wave c(x)=a₀·cos(k·x) and verifies amplitude growth
    against the discrete IMEX recurrence g²−g−β=0 with
    β = M·dt·k²(1−ε²k²).  Tests the biharmonic ∇⁴ pathway and the
    lagged-μ time stepping.
  * schnakenberg_3d — activator-substrate Turing instability; seeds a
    single plane-wave perturbation along the unstable eigenvector of
    M(k) = J − diag(Du,Dv)·k² and checks (a) per-step amplification
    (1+σ₊·dt)^N and (b) eigenvector preservation δV/δU = v_e/u_e.
    Tests linear coupling of two diffusing species in the Turing band.
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


@oracle('schnakenberg_3d')
def _schnakenberg_turing_growth(ctx, size: int, seed: int, cap: int) -> dict:
    """Schnakenberg activator-substrate Turing instability — verify the
    linear dispersion σ(k) of an unstable Turing mode.

    The reaction kinetics
        ∂U/∂t = Du ∇²U + a − U + U²V
        ∂V/∂t = Dv ∇²V + b     − U²V
    has the homogeneous fixed point  (U*, V*) = (a+b, b/(a+b)²).
    Linearising around it with Jacobian
        J = [[−1 + 2 U*V*,   U*²  ],   [[ 0.8, 1.0 ],
             [   −2 U*V*  , −U*²  ]] =  [−1.8, −1.0 ]]   (defaults)
    and adding the diffusion symbol −diag(Du, Dv)·k² gives a
    wavenumber-dependent stability matrix
        M(k) = J − diag(Du, Dv)·k².
    The system is Turing-unstable for k² in the band where det M < 0;
    for default (a=0.1, b=0.9, Du=0.05, Dv=1) this band is
    [1.48, 13.52] with σ_max = 0.219 at k² = 7.5.

    We seed a single plane-wave perturbation along x at exactly the
    right-eigenvector of M(k) for the positive eigenvalue σ₊(k):
        (U, V) = (U* + ε·u_e·cos kx,  V* + ε·v_e·cos kx)
    The forward-Euler integrator then evolves this purely along the
    eigenvector with per-step amplification (1 + σ₊·dt_eff), so two
    things should hold to floating-point precision:
      (a) the projected amplitude δU(N)/ε = (1 + σ₊·dt_eff)^N
      (b) the projected ratio δV/δU stays at v_e/u_e for all time
          (eigenvector preservation — no spurious mode mixing).

    Discrete-spectrum corrections captured in the prediction:
      * 19-point Laplacian → k²_eff = 2(1−cos k_voxel)·h_sq
      * shader caps dt at dt_eff = 0.9/(6·D_max·h_sq); we choose
        dt below the cap so no silent capping occurs.

    NOTE on grid size: the Turing band [1.48, 13.52] only intersects
    the reachable 19-pt Laplacian spectrum (max k²_eff = 4·h_sq) for
    h_sq ≥ ~0.4 — i.e. size ≥ ~80.  We therefore force size ≥ 128
    inside this oracle and report the effective size used.
    """
    from test_harness import HeadlessRunner

    rule = 'schnakenberg_3d'
    # Up-rank size so the Turing band is reachable (see NOTE above).
    size_eff = max(int(size), 128)

    a, b = 0.1, 0.9
    Du, Dv = 0.05, 1.0
    n_mode = 32        # at size=128 → k_voxel = π/2 → k_eff_sq = 2·h_sq = 2
    eps0 = 1.0e-3      # |δU| stays ≪ U*=1 even after the full 100-step run
    dt = 0.10          # below shader's dt cap of 0.15 at size=128
    tol_rel = 0.05     # 5 % budget

    params = {'a': a, 'b': b, 'Du': Du, 'Dv': Dv}

    # Fixed point and Jacobian.
    U_star = a + b
    V_star = b / (U_star * U_star)
    J11 = -1.0 + 2.0 * U_star * V_star      # = 0.8 at defaults
    J12 =  U_star * U_star                  # = 1.0
    J21 = -2.0 * U_star * V_star            # = -1.8
    J22 = -U_star * U_star                  # = -1.0

    # 19-point Laplacian eigenvalue on a plane wave along one axis.
    h_inv = float(size_eff) / _REF_SIZE
    h_sq = h_inv * h_inv
    k_voxel = 2.0 * math.pi * n_mode / float(size_eff)
    k_eff_sq = 2.0 * (1.0 - math.cos(k_voxel)) * h_sq

    # M(k) eigenvalues — verify mode is Turing-unstable, pick σ₊ + eigenvector.
    A = J11 - Du * k_eff_sq
    D = J22 - Dv * k_eff_sq
    tr_M = A + D
    det_M = A * D - J12 * J21
    disc = tr_M * tr_M - 4.0 * det_M
    if det_M >= 0.0:
        return {'rule': rule, 'grade': 'err',
                'oracle': 'schnakenberg_turing_growth',
                'reason': f'k²_eff={k_eff_sq:.3f} not in Turing band '
                          f'(det M={det_M:.3e}); choose a different n_mode'}
    if disc < 0.0:
        return {'rule': rule, 'grade': 'err',
                'oracle': 'schnakenberg_turing_growth',
                'reason': f'M(k) eigenvalues complex (disc={disc:.3e})'}
    sigma_plus = 0.5 * (tr_M + math.sqrt(disc))
    # Right eigenvector for σ₊: (A−σ₊)·u + J12·v = 0  ⇒  v = (σ₊−A)·u / J12.
    u_e = 1.0
    v_e = (sigma_plus - A) * u_e / J12

    # Shader's effective dt (dt_eff = min(u_dt, 0.9 / 6 / (D_max·h_sq)))
    dt_cap = 0.9 / 6.0 / max(max(Du, Dv) * h_sq, 1e-12)
    dt_eff = min(dt, dt_cap)
    # Forward-Euler amplification per step along the eigenvector.
    amp_per_step = 1.0 + sigma_plus * dt_eff
    amp_predicted = amp_per_step ** cap

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size_eff, seed=seed,
                           params=params, dt=dt)
    try:
        # Build IC: homogeneous (U*, V*) + ε·v_+·cos(k·i_x), planar in y,z.
        idx = np.arange(size_eff, dtype=np.float64)
        cos_1d = np.cos(k_voxel * idx).astype(np.float64)
        data = np.zeros((size_eff, size_eff, size_eff, 4), dtype=np.float32)
        # Broadcast cos_1d along x; uniform in y, z.
        cos_3d = np.broadcast_to(
            cos_1d[:, None, None],
            (size_eff, size_eff, size_eff)).astype(np.float32)
        data[..., 0] = np.float32(U_star) + np.float32(eps0 * u_e) * cos_3d
        data[..., 1] = np.float32(V_star) + np.float32(eps0 * v_e) * cos_3d

        current = r.tex_a if r.ping == 0 else r.tex_b
        current.write(data.tobytes())

        for _ in range(cap):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'schnakenberg_turing_growth',
                    'reason': f'NaN/Inf in grid after {cap} steps'}

        U_t = g_t[..., 0].astype(np.float64)
        V_t = g_t[..., 1].astype(np.float64)
        # Project (U - U*) and (V - V*) onto cos(k·i_x) basis.
        cos_norm_sq = float((cos_1d * cos_1d).sum())
        dU_x = (U_t - U_star).mean(axis=(1, 2))    # average over y, z
        dV_x = (V_t - V_star).mean(axis=(1, 2))
        dU_amp = float((dU_x * cos_1d).sum() / cos_norm_sq)
        dV_amp = float((dV_x * cos_1d).sum() / cos_norm_sq)

        amp_U_ratio_measured = dU_amp / (eps0 * u_e)
        # Eigenvector-preservation check: δV/δU should equal v_e/u_e.
        evec_ratio_measured = dV_amp / dU_amp if abs(dU_amp) > 0.0 else float('nan')
        evec_ratio_expected = v_e / u_e

        err_amp = abs(amp_U_ratio_measured - amp_predicted) / amp_predicted
        err_evec = abs(evec_ratio_measured - evec_ratio_expected) / abs(evec_ratio_expected)
        rel_err = max(err_amp, err_evec)

        # Planar-uniformity (y, z) check: should be at fp32 noise level.
        planar_std = float(max(U_t.std(axis=(1, 2)).max(),
                               V_t.std(axis=(1, 2)).max()))

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'schnakenberg_turing_growth',
            'grade': grade,
            'size': size_eff,
            'steps': cap,
            'dt': dt,
            'dt_eff': dt_eff,
            'n_mode': n_mode,
            'k_eff_sq': k_eff_sq,
            'sigma_plus': sigma_plus,
            't_phys': float(cap) * dt_eff,
            'amp_U_measured': amp_U_ratio_measured,
            'amp_U_predicted': amp_predicted,
            'amp_U_continuum': math.exp(sigma_plus * cap * dt_eff),
            'evec_ratio_measured': evec_ratio_measured,
            'evec_ratio_expected': evec_ratio_expected,
            'rel_err': rel_err,
            'err_amp': err_amp,
            'err_evec': err_evec,
            'tol': tol_rel,
            'planar_std_max': planar_std,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


@oracle('phase_separation')
def _cahn_hilliard_spinodal_growth(ctx, size: int, seed: int, cap: int) -> dict:
    """Cahn-Hilliard linear spinodal growth rate.

    The Cahn-Hilliard equation
        ∂c/∂t = M ∇² (c³ − c − ε²∇²c)
    linearises around the symmetric mixed state c = 0 to
        ∂c/∂t = −M ∇²c + M ε²∇⁴c
    so a Fourier eigenmode c(x,t) = a(t)·cos(k·x) evolves with
        a(t) = a₀ · exp(σ·t),     σ(k) = M·k²·(1 − ε²·k²)
    — positive (spinodal-unstable) for k² < 1/ε², peaking at
    k_max² = 1/(2ε²) with σ_max = M/(4ε²).

    The shader uses an IMEX integrator: μ is computed from the
    current c each frame, but the c-update uses ∇²μ from the *previous*
    frame's μ (a one-step lag).  Linearising that two-step recurrence
    c_{n+2} = c_{n+1} + β·c_n,  β = M·dt·k²_eff·(1 − ε²_eff·k²_eff)
    gives a per-step amplification factor
        g = ½·(1 + √(1 + 4β))
    valid whenever β > 0 (unstable band).  Comparing to the continuum
    exp(σ·t) leaves a O(β) phase-error budget; comparing to the discrete
    recurrence directly is exact for the linearised scheme.

    We seed a low-mode plane wave c(x) = a₀·cos(k·i_x) (n = 4 along x,
    uniform in y,z) with small amplitude a₀ = 0.01 so c³ stays
    1e-4 below the linear term, project the resulting field onto
    cos(k·i_x), and compare to the discrete prediction.

    Discretisation details captured in the prediction:
      * 19-point Laplacian → effective k²_eff = 2(1−cos k)·h_sq
      * shader floors ε² to 1/(8·h_sq) and caps it via the biharmonic
        CFL — we choose ε² safely above the floor.
      * noise = 0, asymmetry = 0 → deterministic, c=0-symmetric.
    """
    from test_harness import HeadlessRunner

    rule = 'phase_separation'
    M = 1.0
    eps2 = 1.0          # well above the size=64 floor (0.5), below ceiling
    dt = 0.05
    n_mode = 4          # k_voxel = 2π·n/L, small enough for 19-pt Lap to be exact
    a0 = 0.01           # c³/c ratio = a₀² = 1e-4 ⇒ linear regime to 0.01 %
    tol_rel = 0.05      # 5 % budget covers fp32 noise + sub-leading nonlinear bias

    params = {'Mobility': M, 'Epsilon²': eps2, 'Noise': 0.0, 'Asymmetry': 0.0}

    # Discrete spectrum of the 19-point Laplacian on a plane wave
    # along x (uniform in y,z) — exact, not a Taylor truncation.
    h_inv = float(size) / _REF_SIZE
    h_sq = h_inv * h_inv
    k_voxel = 2.0 * math.pi * n_mode / float(size)   # periodic wrap matches
    k_eff_sq = 2.0 * (1.0 - math.cos(k_voxel)) * h_sq

    # The shader floors ε² to 1/(8·h_sq); double-check our choice clears it.
    eps2_floor = 1.0 / (8.0 * h_sq)
    eps2_ceiling = 0.025 / (M * dt * h_sq * h_sq)
    eps2_eff = max(eps2, eps2_floor)
    eps2_eff = min(eps2_eff, max(eps2_floor, eps2_ceiling))

    beta = M * dt * k_eff_sq * (1.0 - eps2_eff * k_eff_sq)
    if beta <= 0.0:
        return {'rule': rule, 'grade': 'err',
                'oracle': 'cahn_hilliard_spinodal_growth',
                'reason': f'mode is stable (β={beta:.3e}); '
                          f'choose n_mode with k²·ε² < 1'}
    # Continuum growth rate (for diagnostics) and the exact discrete
    # 2-step recurrence amplitude prediction.
    sigma = M * k_eff_sq * (1.0 - eps2_eff * k_eff_sq)
    a_prev, a_curr = 1.0, 1.0 + beta              # a_0, a_1 (in units of a0)
    for _ in range(cap - 1):
        a_prev, a_curr = a_curr, a_curr + beta * a_prev
    amp_ratio_predicted = a_curr                   # a_N / a_0

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # Build IC: c = a0·cos(k·i_x), μ = -(1-ε²k²)·c (linearised
        # equilibrium that the shader would compute on step 0 — keeps
        # the first-frame lag transient at machine epsilon).
        idx = np.arange(size, dtype=np.float64)
        cos_1d = np.cos(k_voxel * idx).astype(np.float32)
        c_field = (a0 * cos_1d)[:, None, None] * np.ones((1, size, size), np.float32)
        mu_field = (-(1.0 - eps2_eff * k_eff_sq) * c_field).astype(np.float32)

        data = np.zeros((size, size, size, 4), dtype=np.float32)
        data[:, :, :, 0] = c_field
        data[:, :, :, 1] = mu_field
        current = r.tex_a if r.ping == 0 else r.tex_b
        current.write(data.tobytes())

        for _ in range(cap):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'cahn_hilliard_spinodal_growth',
                    'reason': f'NaN/Inf in grid after {cap} steps'}

        c_t = g_t[..., 0].astype(np.float64)
        # Project c(t) onto cos(k·i_x); IC has c projection = a0 (since
        # planar in y,z and ⟨cos²⟩ over one full period = 1/2 per axis).
        cos_basis = np.cos(k_voxel * idx)          # shape (size,)
        cos_norm_sq = float((cos_basis * cos_basis).sum())
        # ⟨c·cos⟩ over the volume, with cos depending only on x.
        c_x_profile = c_t.mean(axis=(1, 2))        # average over y, z
        amp_measured = float((c_x_profile * cos_basis).sum() / cos_norm_sq)
        amp_ratio_measured = amp_measured / a0

        rel_err = abs(amp_ratio_measured - amp_ratio_predicted) / amp_ratio_predicted

        # Planar homogeneity in (y, z): IC is uniform in those axes, so any
        # deviation indicates spurious mode coupling. Should be ~fp32 noise.
        planar_std = float(c_t.std(axis=(1, 2)).max())

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'cahn_hilliard_spinodal_growth',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'M': M,
            'eps2_eff': eps2_eff,
            'k_voxel': k_voxel,
            'k_eff_sq': k_eff_sq,
            'beta': beta,
            'sigma_continuum': sigma,
            't_phys': float(cap) * dt,
            'amp_ratio_measured': amp_ratio_measured,
            'amp_ratio_predicted': amp_ratio_predicted,
            'amp_ratio_continuum': math.exp(sigma * cap * dt),
            'rel_err': rel_err,
            'tol': tol_rel,
            'planar_std_max': planar_std,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


# Exact star-region values for the canonical 3-D Sod shock tube
# (ρ_L, u_L, p_L) = (1.0, 0.0, 1.0),  (ρ_R, u_R, p_R) = (0.125, 0.0, 0.1),
# γ = 1.4.  Tabulated to 13 sig-figs from Toro's exact Riemann solver
# (cross-checked against the reference values in Toro 2009, §4.3.3).
_SOD_PSTAR     = 0.30313017805064697
_SOD_USTAR     = 0.92745262004897489
_SOD_RHO_LSTAR = 0.42631942817849519   # post-rarefaction density
_SOD_RHO_RSTAR = 0.26557371170530727   # post-shock density
_SOD_SHOCK_SPEED = 1.7521557320301779  # right-moving shock speed (voxels / time)


@oracle('compressible_euler_3d')
def _euler_sod_shocktube(ctx, size: int, seed: int, cap: int) -> dict:
    """Sod shock tube — verify the right-moving shock speed against
    the exact Riemann solution.

    The canonical 1-D Sod problem
        (ρ_L, u_L, p_L) = (1.0,   0.0, 1.0)
        (ρ_R, u_R, p_R) = (0.125, 0.0, 0.1)         γ = 1.4
    has a closed-form solution consisting of (left → right) a
    rarefaction fan, a contact discontinuity, and a right-moving shock.
    Tabulated values: p* ≈ 0.30313, u* ≈ 0.92745, ρ*_L ≈ 0.42632,
    ρ*_R ≈ 0.26557 and shock speed S ≈ 1.75216 (in cell-units / time —
    the Euler shader uses Δx = 1 cell, no REF_SIZE scaling).

    We extrude the 1-D problem along z (uniform in x, y), integrate
    `cap` steps with mirror BC (so the unperturbed L/R states repeat
    at the +z / -z edges instead of wrapping), and locate the shock
    front as the rightmost z where the planar-averaged density crosses
    the midpoint between ρ*_R and ρ_R.  Sub-cell interpolation lets us
    measure displacement to ≪ 1 voxel.

    The Lax-Friedrichs scheme is 1st-order and smears the shock over
    ~5-10 cells, but Rankine-Hugoniot is preserved exactly in the
    inviscid limit, so the shock *speed* is robust even when the
    profile is diffused.  Tolerance is set at 5 % on shock displacement
    — enough slack to absorb half-cell measurement quantisation at
    a moderate run length, tight enough to catch a wrong flux or EOS.
    """
    from test_harness import HeadlessRunner

    rule = 'compressible_euler_3d'
    gamma = 1.4
    dt = 0.05            # CFL ≈ 0.5 with peak |u|+c ≈ 2.7 post-shock
    tol_rel = 0.05       # 5 % shock-displacement budget

    # Left / right primitive states.
    rho_L, u_L, p_L = 1.0,   0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    E_L = p_L / (gamma - 1.0) + 0.5 * rho_L * u_L * u_L
    E_R = p_R / (gamma - 1.0) + 0.5 * rho_R * u_R * u_R

    # Disable the centre-source injector and gravity; keep a small slug
    # of artificial viscosity to discourage post-shock ringing without
    # smearing the shock measurably further than LF already does.
    params = {'Gamma': gamma, 'Art. visc.': 0.05,
              'Gravity': 0.0, 'Source': 0.0}

    # Place the discontinuity at the cell-face midway through the grid.
    z_split = size // 2

    # Analytic shock displacement after `cap` steps.
    t_phys = float(cap) * dt
    expected_shock_pos = float(z_split) + _SOD_SHOCK_SPEED * t_phys
    threshold = 0.5 * (_SOD_RHO_RSTAR + rho_R)   # half-jump on the shock face

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # Override the wrap boundary with mirror so the undisturbed
        # L/R states repeat past the grid edges instead of leaking the
        # opposite state in via wrap. Shallow-copy preset to avoid
        # mutating the shared RULE_PRESETS entry.
        r.preset = {**r.preset, 'boundary': 'mirror'}

        # Build the discontinuous IC.
        # pair-1: ρ, ρu_x, ρu_y, ρu_z   (u = 0 everywhere → momentum = 0)
        data1 = np.zeros((size, size, size, 4), dtype=np.float32)
        # pair-2: E, p, |u|, helicity
        data2 = np.zeros((size, size, size, 4), dtype=np.float32)

        data1[:, :, :z_split, 0] = np.float32(rho_L)
        data1[:, :, z_split:, 0] = np.float32(rho_R)
        data2[:, :, :z_split, 0] = np.float32(E_L)
        data2[:, :, z_split:, 0] = np.float32(E_R)
        data2[:, :, :z_split, 1] = np.float32(p_L)
        data2[:, :, z_split:, 1] = np.float32(p_R)

        current1 = r.tex_a if r.ping == 0 else r.tex_b
        current1.write(data1.tobytes())
        if getattr(r, 'tex_a2', None) is None:
            return {'rule': rule, 'grade': 'err',
                    'oracle': 'euler_sod_shocktube',
                    'reason': 'pair-2 textures missing — '
                              'compressible_euler_3d expects extra_fields=1'}
        current2 = r.tex_a2 if r.ping2 == 0 else r.tex_b2
        current2.write(data2.tobytes())

        for _ in range(cap):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'euler_sod_shocktube',
                    'reason': f'NaN/Inf in grid after {cap} steps'}

        rho_field = g_t[..., 0].astype(np.float64)
        # Project to 1-D along z by averaging over the (x, y) plane.
        rho_z = rho_field.mean(axis=(0, 1))

        # Locate the shock front: rightmost crossing of `threshold` as
        # we scan from high z (ρ ≈ ρ_R) toward low z (ρ ≈ ρ*_R > thresh).
        shock_pos = None
        for z in range(size - 2, z_split, -1):
            lo, hi = rho_z[z + 1], rho_z[z]
            if lo < threshold <= hi:
                # Linear interp: z* where ρ_z(z*) = threshold,
                # between cell centres z (hi) and z+1 (lo).
                frac = (threshold - hi) / (lo - hi)
                shock_pos = float(z) + frac
                break

        if shock_pos is None:
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'euler_sod_shocktube',
                    'reason': (f'shock front not found '
                               f'(rho_z range [{rho_z.min():.3f}, '
                               f'{rho_z.max():.3f}], threshold={threshold:.3f})')}

        measured_shock_speed = (shock_pos - z_split) / t_phys
        rel_err = abs(measured_shock_speed - _SOD_SHOCK_SPEED) / _SOD_SHOCK_SPEED

        # Planar-homogeneity check: x,y std of ρ should stay ~ fp32 noise
        # since the IC is uniform along x, y and so are all fluxes.
        planar_std = float(rho_field.std(axis=(0, 1)).max())

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'euler_sod_shocktube',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'gamma': gamma,
            't_phys': t_phys,
            'shock_pos_measured': shock_pos,
            'shock_pos_expected': expected_shock_pos,
            'shock_speed_measured': measured_shock_speed,
            'shock_speed_expected': _SOD_SHOCK_SPEED,
            'rel_err': rel_err,
            'tol': tol_rel,
            'planar_std_max': planar_std,
            'rho_min': float(rho_z.min()),
            'rho_max': float(rho_z.max()),
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
                elif 'amp_U_measured' in row:
                    print(f'          δU(t)/ε meas={row["amp_U_measured"]:.5f}  '
                          f'pred(disc)={row["amp_U_predicted"]:.5f}  '
                          f'cont={row["amp_U_continuum"]:.5f}')
                    print(f'          σ₊={row["sigma_plus"]:.5f}  k_eff²={row["k_eff_sq"]:.4f}  '
                          f'dt_eff={row["dt_eff"]:.4f}  t={row["t_phys"]:.3f}  '
                          f'evec δV/δU=(meas {row["evec_ratio_measured"]:+.4f}, '
                          f'exp {row["evec_ratio_expected"]:+.4f})')
                    print(f'          err_amp={row["err_amp"]:.4f}  '
                          f'err_evec={row["err_evec"]:.4f}  '
                          f'rel_err={row["rel_err"]:.4f}  (tol={row["tol"]:.3f})  '
                          f'planar_std={row["planar_std_max"]:.2e}')
                elif 'amp_ratio_measured' in row:
                    print(f'          A(t)/A(0) meas={row["amp_ratio_measured"]:.5f}  '
                          f'pred(discrete)={row["amp_ratio_predicted"]:.5f}  '
                          f'cont={row["amp_ratio_continuum"]:.5f}  '
                          f'rel_err={row["rel_err"]:.4f}  (tol={row["tol"]:.3f})')
                    print(f'          σ={row["sigma_continuum"]:.5f}  β={row["beta"]:.5f}  '
                          f'k_eff²={row["k_eff_sq"]:.5f}  ε²_eff={row["eps2_eff"]:.3f}  '
                          f't={row["t_phys"]:.3f}  planar_std={row["planar_std_max"]:.2e}')
                elif 'shock_speed_measured' in row:
                    print(f'          S_meas={row["shock_speed_measured"]:.5f}  '
                          f'S_exact={row["shock_speed_expected"]:.5f}  '
                          f't={row["t_phys"]:.3f}  '
                          f'rel_err={row["rel_err"]:.4f}  (tol={row["tol"]:.3f})')
                    print(f'          x_shock=(meas {row["shock_pos_measured"]:.3f}, '
                          f'exp {row["shock_pos_expected"]:.3f})  '
                          f'planar_std={row["planar_std_max"]:.2e}  '
                          f'ρ_z∈[{row["rho_min"]:.3f}, {row["rho_max"]:.3f}]')
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
