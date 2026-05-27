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

Currently registered (16):
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
  * em_wave — Yee-grid transverse plane wave; verifies the discrete
    dispersion C(t)=cos(ω_disc·t) of the FDTD update with E ⊥ k.
  * dirac_3d — 1D Dirac plane wave; iterates the 2×2 complex matrix
    closed form Φ_{n+1}=e^{−imdt}Φ_n − i·s·Χ_n,
    Χ_{n+1}=e^{+imdt}Χ_n − i·s·Φ_{n+1} with s=c·dt·sin(k_voxel).
    Bit-exact match to fp32 arithmetic.
  * fitzhugh_nagumo_3d — unstable focus around the interior fixed point
    V*; seeds a spatially homogeneous perturbation along the right
    eigenvector of the FHN Jacobian and verifies growth
    (1+λ_+·dt_eff)^N for both δV and δW components.
  * quantum_harmonic — Schrödinger Yee leapfrog with V=½ω²r²; seeds
    a Gaussian wavepacket and verifies probability-mass conservation
    (secular drift < 0.5%) over a full classical period.
  * kuramoto_3d — uniform-IC pure drift (Noise=0, K=0 effectively); the
    phase advances by exactly dt·ω·freq_scale per step.  Bit-exact match.
  * xy_spin_3d — zero-temperature frozen-in spin field; seeds θ(x)=π/4
    everywhere with T=10⁻³ and large J=10 to suppress Metropolis flips,
    then verifies ⟨cos θ⟩=1 and ⟨sin θ⟩=½ to fp32 precision.
  * predator_prey_3d — SKIPPED.  The live preset is an entity-arena
    (particle) simulation, not the PDE shader; no continuum PDE
    end-state oracle is meaningful for it.
  * wireworld_3d — single-pulse propagation around a 1-voxel-thick
    conductor ring; verifies that after N steps the unique head sits
    at z=N mod size and the unique tail at z=N−1 mod size, with all
    other ring cells conductor and the off-ring volume empty.
    Bit-exact integer state check.
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


@oracle('em_wave')
def _em_wave_yee_dispersion(ctx, size: int, seed: int, cap: int) -> dict:
    """EM plane-wave dispersion under the Yee leapfrog (vacuum).

    Vacuum Maxwell (εr ≡ 1, σ ≡ 0):
        ∂E/∂t = c²·∇×B,   ∂B/∂t = -∇×E

    For a plane wave propagating along z, polarised (E_x, B_y) along x/y,
    the engine's stencil uses central differences (factor ½) on a
    collocated grid (Δx = 1 cell; no REF_SIZE rescaling).  Solving the
    leapfrog 2×2 update with mode e^{i k z}, the EXACT discrete
    dispersion is
        sin(ω·dt/2) = ½·c·dt·sin(k_voxel)
    (det = 1 ⇒ no per-step amplitude drift; only the engine's hard-
    coded vacuum loss factor (1 - 0.002·dt) per pass nibbles amplitude).

    We seed a standing wave  E_y(x) = A·cos(k·x), B = 0, εr = σ = 0
    (E transverse to the propagation direction — a *longitudinal* IC
    like E_x(x) is a non-radiating mode and would just sit there),
    disable the dipole source (Amplitude=0) and damping, force periodic
    boundary so the cosine is an exact eigenmode, integrate, and verify
    the temporal correlation
        C(t) = ⟨E_y(t)·E_y(0)⟩ / ⟨E_y(0)²⟩ = cos(ω·t) · damping^t
    matches the discrete-Yee prediction.
    """
    from test_harness import HeadlessRunner

    rule = 'em_wave'
    c_wave = 1.0
    dt = 0.05
    amplitude = 0.5
    n_mode = 4
    tol_rel = 0.03

    # Disable dipole source (Amplitude=0) and damping; keep speed at 1.
    params = {'Wave speed': c_wave, 'Damping': 0.0,
              'Frequency': 0.0, 'Amplitude': 0.0}

    # Periodic plane wave along z: k_voxel = 2π·n/L.
    k_voxel = 2.0 * math.pi * n_mode / float(size)
    sin_k = math.sin(k_voxel)
    cfl = c_wave * dt * sin_k * 0.5
    if abs(cfl) >= 1.0:
        return {'rule': rule, 'grade': 'err',
                'oracle': 'em_wave_yee_dispersion',
                'reason': f'discrete CFL violated: c·dt·sin(k)/2={cfl:.3f}'}
    omega = (2.0 / dt) * math.asin(cfl)

    target_phase = math.pi / 3.0
    steps = max(1, min(cap, int(round(target_phase / (omega * dt)))))
    actual_phase = omega * steps * dt
    # Engine applies (1 - 0.002·dt) to E and B each pass; only the E pass
    # writes pair-1, so over `steps` engine frames E_x decays by factor
    # (1 - 0.002·dt)^steps.
    decay = (1.0 - 0.002 * dt) ** steps
    expected_C = math.cos(actual_phase) * decay

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # Override boundary to wrap so cos(k·z) with k=2π·n/L is exact.
        r.preset = {**r.preset, 'boundary': 'wrap'}

        idx = np.arange(size, dtype=np.float64)
        cos_1d = amplitude * np.cos(k_voxel * idx).astype(np.float32)
        # pair-1: (Ex, Ey, Ez, εr).  εr=0 → vacuum (back-compat).
        # Use E_y as the transverse field; cos varies along the last
        # numpy axis (the texture's fastest-varying coord, which is the
        # GPU x-direction) so E_y ⊥ k → radiating plane wave.
        data1 = np.zeros((size, size, size, 4), dtype=np.float32)
        data1[:, :, :, 1] = cos_1d[None, None, :]
        # pair-2: (Bx, By, Bz, σ).  Standing wave: B = 0 initially.
        data2 = np.zeros((size, size, size, 4), dtype=np.float32)

        current1 = r.tex_a if r.ping == 0 else r.tex_b
        current1.write(data1.tobytes())
        if getattr(r, 'tex_a2', None) is None:
            return {'rule': rule, 'grade': 'err',
                    'oracle': 'em_wave_yee_dispersion',
                    'reason': 'pair-2 textures missing — '
                              'em_wave expects extra_fields=1'}
        current2 = r.tex_a2 if r.ping2 == 0 else r.tex_b2
        current2.write(data2.tobytes())

        g0 = np.asarray(r.read_grid())
        Ey0 = g0[..., 1].astype(np.float64)
        norm0 = float((Ey0 * Ey0).sum())
        if norm0 <= 0.0:
            return {'rule': rule, 'grade': 'err',
                    'oracle': 'em_wave_yee_dispersion',
                    'reason': f'zero IC norm (norm0={norm0})'}

        for _ in range(steps):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'em_wave_yee_dispersion',
                    'reason': f'NaN/Inf in grid after {steps} steps'}

        Ex_t = g_t[..., 1].astype(np.float64)
        C_t = float((Ex_t * Ey0).sum() / norm0)
        rel_err = abs(C_t - expected_C) / max(abs(expected_C), 0.1)

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'em_wave_yee_dispersion',
            'grade': grade,
            'size': size,
            'steps': steps,
            'dt': dt,
            'c_wave': c_wave,
            'n_mode': n_mode,
            'k_voxel': k_voxel,
            'omega': omega,
            'omega_discrete': omega,
            'omega_continuum': c_wave * k_voxel,
            'phase_rad': actual_phase,
            'C_measured': C_t,
            'C_expected': expected_C,
            'damping_factor': decay,
            'rel_err': rel_err,
            'tol': tol_rel,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


@oracle('dirac_3d')
def _dirac_plane_wave_dispersion(ctx, size: int, seed: int, cap: int) -> dict:
    """3+1D Dirac plane-wave dispersion: ω² = c²k² + m².

    The engine's leapfrog scheme for the upper bispinor φ↑ coupled to
    χ↑ (with ψ↓ ≡ 0, plane wave varying along the z axis) reduces to
    a 2×2 complex matrix update per step:

        Φ_{n+1} = R(-m·dt)·Φ_n - i·s·Χ_n
        Χ_{n+1} = R(+m·dt)·Χ_n - i·s·Φ_{n+1}

    where R(θ) = e^{iθ}, s = c·dt·sin(k_voxel) (central-difference
    eigenvalue of ∂_z on mode cos(k·n)).  The matrix has det = 1 and
    tr = 2·cos(m·dt) − s², giving the EXACT discrete Dirac dispersion

        cos(ω·dt) = cos(m·dt) − ½·(c·dt·sin(k_voxel))²

    which limits to ω² = m² + c²k² in the continuum (small s, m·dt).

    We seed Re φ↑ = A·cos(k·z), everything else 0 (so χ↑ = 0; equal
    superposition of positive- and negative-energy modes at ±|k|).
    The scalar projection ⟨Re φ↑(t)·cos(k·z)⟩ / ⟨cos²(k·z)⟩ then
    tracks the (1,1) entry of M^N applied to (1,0)^T, which we iterate
    in Python for a bit-exact prediction.

    A planar-uniformity check on the simulator output verifies that
    no spurious x/y-dependence develops (the IC + integrator are
    invariant under translation in x and y when ψ↓ ≡ 0).
    """
    from test_harness import HeadlessRunner

    rule = 'dirac_3d'
    c_light = 0.5
    mass = 0.5
    dt = 0.1
    amplitude = 0.5
    n_mode = 4
    tol_rel = 0.05

    params = {'c (light)': c_light, 'Mass': mass,
              'Absorber': 0.0, '_': 0.0}

    # Periodic-along-z plane wave: k_voxel = 2π·n/L.
    k_voxel = 2.0 * math.pi * n_mode / float(size)
    sin_k = math.sin(k_voxel)
    s = c_light * dt * sin_k

    # Discrete dispersion (sanity print only — the actual prediction
    # comes from iterating the 2x2 matrix).
    cos_wdt = math.cos(mass * dt) - 0.5 * s * s
    if not (-1.0 < cos_wdt < 1.0):
        return {'rule': rule, 'grade': 'err',
                'oracle': 'dirac_plane_wave_dispersion',
                'reason': f'discrete dispersion out of band: cos(ωdt)={cos_wdt:.4f}'}
    omega_disc = math.acos(cos_wdt) / dt
    omega_cont = math.sqrt(c_light * c_light * (k_voxel ** 2) + mass * mass)

    # Iterate the 2x2 complex matrix M for `cap` steps from (Φ, Χ) = (1, 0).
    # Predicted projection = Re(Φ_cap).
    a = complex(math.cos(mass * dt), -math.sin(mass * dt))   # e^{-i·m·dt}
    b = complex(0.0, -s)
    d_base = complex(math.cos(mass * dt), math.sin(mass * dt))   # e^{+i·m·dt}
    Phi, Chi = complex(1.0, 0.0), complex(0.0, 0.0)
    for _ in range(cap):
        Phi_new = a * Phi + b * Chi
        # Χ_{n+1} = R(+m·dt)·Χ - i·s·Φ_{n+1}   (uses already-updated Φ_new)
        Chi_new = d_base * Chi + complex(0.0, -s) * Phi_new
        Phi, Chi = Phi_new, Chi_new
    pred_Phi_re = Phi.real
    pred_Phi_im = Phi.imag
    pred_Chi_re = Chi.real
    pred_Chi_im = Chi.imag

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # Default boundary is 'wrap' for dirac_3d — verify and force.
        r.preset = {**r.preset, 'boundary': 'wrap'}

        # IC: Re φ↑ = A·cos(k·z), all other channels zero.  Vary along
        # numpy axis 0 (texture's slowest axis, conventionally GPU z).
        idx = np.arange(size, dtype=np.float64)
        cos_1d = (amplitude * np.cos(k_voxel * idx)).astype(np.float32)
        data1 = np.zeros((size, size, size, 4), dtype=np.float32)
        data1[:, :, :, 0] = cos_1d[:, None, None]
        data2 = np.zeros((size, size, size, 4), dtype=np.float32)

        current1 = r.tex_a if r.ping == 0 else r.tex_b
        current1.write(data1.tobytes())
        if getattr(r, 'tex_a2', None) is None:
            return {'rule': rule, 'grade': 'err',
                    'oracle': 'dirac_plane_wave_dispersion',
                    'reason': 'pair-2 textures missing'}
        current2 = r.tex_a2 if r.ping2 == 0 else r.tex_b2
        current2.write(data2.tobytes())

        for _ in range(cap):
            r.step()

        g1 = np.asarray(r.read_grid())
        if not np.isfinite(g1).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'dirac_plane_wave_dispersion',
                    'reason': f'NaN/Inf in pair-1 grid after {cap} steps'}

        re_phi_up = g1[..., 0].astype(np.float64)
        # Project onto cos(k·z) along axis 0 (planar-average over x, y).
        prof = re_phi_up.mean(axis=(1, 2))
        basis = np.cos(k_voxel * idx)
        proj = float((prof * basis).sum() / (basis * basis).sum()) / amplitude
        # Planar-uniformity check: x, y should never have spread.
        planar_std = float(re_phi_up.std(axis=(1, 2)).max())

        # Phase-aware error: predicted is purely real (cos-like).  Use
        # combined error |Φ_meas - Φ_pred|, treating measured as real
        # scalar and matching against the real part.
        err_re = abs(proj - pred_Phi_re)
        rel_err = err_re / max(abs(pred_Phi_re), 0.1)

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'dirac_plane_wave_dispersion',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'c_light': c_light,
            'mass': mass,
            'n_mode': n_mode,
            'k_voxel': k_voxel,
            'sin_k_voxel': sin_k,
            's_param': s,
            'omega_discrete': omega_disc,
            'omega_continuum': omega_cont,
            't_phys': float(cap) * dt,
            'phi_re_measured': proj,
            'phi_re_predicted': pred_Phi_re,
            'phi_im_predicted': pred_Phi_im,
            'chi_re_predicted': pred_Chi_re,
            'chi_im_predicted': pred_Chi_im,
            'rel_err': rel_err,
            'tol': tol_rel,
            'planar_std_max': planar_std,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


@oracle('fitzhugh_nagumo_3d')
def _fitzhugh_nagumo_unstable_focus(ctx, size: int, seed: int, cap: int) -> dict:
    """FitzHugh-Nagumo linearised growth around the (unstable) rest state.

    dV/dt = D·∇²V + V − V³/3 − W
    dW/dt = ε·(V + a − b·W)

    At the rest state (V*, W*) where V* solves V*³ + 3V* + 3a/b· … set
    by V − V³/3 = W and W = (V + a)/b → V*³ + 3V*(1 − 1/b) + 3a/b = 0,
    the Jacobian is
        J = [[1 − V*², −1],
             [ε,        −ε·b]]
    At the default operating params (a=0.1, b=0.5, ε=0.10) the fixed
    point V* ≈ −0.1968 lies on the unstable middle branch (the engine's
    "auto-oscillatory" regime — a Hopf has been crossed because b<1, but
    here the two eigenvalues are both REAL POSITIVE: an unstable node).
    Both branches grow exponentially; the unstable manifold tangent
    direction is the right eigenvector for λ_+.

    With a SPATIALLY HOMOGENEOUS perturbation along that eigenvector,
    the Laplacian vanishes everywhere and the dynamics reduce to a 2×2
    forward-Euler ODE (the shader's update is exactly Euler when
    dt_eff = u_dt < 0.9/(6·D·h_sq)).  After N steps,
        (δV(N), δW(N)) = ε₀·(u_v, u_w)·(1 + λ_+·dt)^N.
    We seed exactly on the unstable eigenvector, integrate, and verify
    both the amplification factor and that the eigenvector ratio is
    preserved.  ε₀ is chosen so the final perturbation stays well
    inside the linear regime (V*·δV² stays a fraction of a percent of
    the linear term).
    """
    from test_harness import HeadlessRunner

    rule = 'fitzhugh_nagumo_3d'
    a, b, eps_t, D = 0.1, 0.5, 0.10, 1.0
    dt = 0.05            # below the dt_limit = 0.9/(6·D·h_sq) at all sizes
    tol_rel = 0.05

    params = {'a': a, 'b': b, 'ε': eps_t, 'D': D}

    # Rest state from the cubic V³ + 3V·(1 − 1/b) + 3a/b = 0  → simplifies
    # at b=0.5 to V³ + 3V·(1 − 2) + 0.6 = V³ − 3V·… wait: starting from
    #   V − V³/3 = (V + a)/b  ⇒  bV − bV³/3 = V + a
    #   bV³ + 3(1 − b)V + 3a = 0   (after × 3, sign flip on V³ term cancels)
    # so coefficients are (b, 0, 3(1 − b), 3a):
    poly = np.array([b, 0.0, 3.0 * (1.0 - b), 3.0 * a])
    roots = np.roots(poly)
    real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
    if not real_roots:
        return {'rule': rule, 'grade': 'err',
                'oracle': 'fitzhugh_nagumo_unstable_focus',
                'reason': 'no real fixed point'}
    # The (unique) real root for these defaults is V* ≈ −0.197.
    V_star = real_roots[0]
    W_star = (V_star + a) / b

    # Jacobian at the fixed point.
    A11 = 1.0 - V_star * V_star
    A12 = -1.0
    A21 = eps_t
    A22 = -eps_t * b
    tr = A11 + A22
    det = A11 * A22 - A12 * A21
    disc = tr * tr - 4.0 * det
    if disc < 0:
        # Defaults give a real positive disc; if a user overrides params
        # into the Hopf regime we'd need the complex-eigenpair pattern
        # of brusselator instead.  Punt.
        return {'rule': rule, 'grade': 'err',
                'oracle': 'fitzhugh_nagumo_unstable_focus',
                'reason': f'fixed point is a focus (disc={disc:.4f}) — '
                          'oracle requires real eigenvalues'}
    sqrt_disc = math.sqrt(disc)
    lambda_plus = 0.5 * (tr + sqrt_disc)
    if lambda_plus <= 0:
        return {'rule': rule, 'grade': 'err',
                'oracle': 'fitzhugh_nagumo_unstable_focus',
                'reason': f'fixed point is stable (λ_+={lambda_plus:.4f})'}

    # Right eigenvector for λ_+: row 1 of (J − λ·I) gives A11·u + A12·v = 0
    # with the λ subtracted, so (A11 − λ)·u_v + A12·u_w = 0  →
    #   u_w = (A11 − λ_+) / (−A12) · u_v
    u_v = 1.0
    u_w = (A11 - lambda_plus) / (-A12) * u_v   # = (A11 − λ_+)·u_v since A12=−1

    # Discrete forward-Euler per-step factor on the eigenvector.
    # In the engine, h_sq = (size/_REF_SIZE)² is the Laplacian multiplier
    # (NOT h²!); the shader caps dt at 0.9/(6·D·h_sq).
    h_sq = (float(size) / _REF_SIZE) ** 2
    dt_limit = 0.9 / (6.0 * max(D * h_sq, 1e-12))
    dt_eff = min(dt, dt_limit)
    eul_factor = 1.0 + lambda_plus * dt_eff
    growth = eul_factor ** cap

    # Pick ε₀ so the final perturbation stays small: target |δV| ≈ 0.01
    # (≈ 5 % of |V*|).  Quadratic-in-δV correction is V*·δV² ≈
    # 0.197·1e−4 = 2e−5 vs linear 0.961·0.01 = 9.6e−3 → 0.2 %.
    eps0 = 0.01 / max(growth, 1.0)
    # Floor to avoid fp32 truncation against V* ≈ 0.2.
    eps0 = max(eps0, 1e-6)

    pred_dV = eps0 * u_v * growth
    pred_dW = eps0 * u_w * growth

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # Use whatever boundary the preset has — Laplacian on a uniform
        # field is zero regardless of BC, so it doesn't matter here.

        V_field = np.full((size, size, size), V_star + eps0 * u_v, dtype=np.float32)
        W_field = np.full((size, size, size), W_star + eps0 * u_w, dtype=np.float32)
        data = np.zeros((size, size, size, 4), dtype=np.float32)
        data[..., 0] = V_field
        data[..., 1] = W_field

        current = r.tex_a if r.ping == 0 else r.tex_b
        current.write(data.tobytes())

        for _ in range(cap):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'fitzhugh_nagumo_unstable_focus',
                    'reason': f'NaN/Inf in grid after {cap} steps'}

        V_t = g_t[..., 0].astype(np.float64)
        W_t = g_t[..., 1].astype(np.float64)
        meas_dV = float(V_t.mean()) - V_star
        meas_dW = float(W_t.mean()) - W_star
        homo_std_V = float(V_t.std())
        homo_std_W = float(W_t.std())

        err_V = abs(meas_dV - pred_dV) / max(abs(pred_dV), 1e-9)
        err_W = abs(meas_dW - pred_dW) / max(abs(pred_dW), 1e-9)
        evec_meas = meas_dW / meas_dV if abs(meas_dV) > 1e-12 else float('nan')
        evec_exp = u_w / u_v
        err_evec = abs(evec_meas - evec_exp) / max(abs(evec_exp), 1e-9)
        rel_err = max(err_V, err_W, err_evec)

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'fitzhugh_nagumo_unstable_focus',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'dt_eff': dt_eff,
            'V_star': V_star,
            'W_star': W_star,
            'lambda_plus': lambda_plus,
            'lambda_minus': 0.5 * (tr - sqrt_disc),
            'eul_factor': eul_factor,
            'growth': growth,
            'eps0': eps0,
            'dV_measured': meas_dV,
            'dV_predicted': pred_dV,
            'dW_measured': meas_dW,
            'dW_predicted': pred_dW,
            'evec_ratio_measured': evec_meas,
            'evec_ratio_expected': evec_exp,
            'err_amp_V': err_V,
            'err_amp_W': err_W,
            'err_evec': err_evec,
            'rel_err': rel_err,
            'tol': tol_rel,
            'homogeneity_std_V': homo_std_V,
            'homogeneity_std_W': homo_std_W,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


@oracle('quantum_harmonic')
def _quantum_harmonic_norm_conservation(ctx, size: int, seed: int, cap: int) -> dict:
    """Symplectic norm conservation for the Schrödinger Yee leapfrog.

    The schrodinger_3d shader uses a staggered-leapfrog (Yee) update
    that — per its own docstring — preserves the symplectic norm
    Σ(ψ_R²(n) + ψ_I(n+½)·ψ_I(n-½)) **exactly** (no secular growth) and
    keeps the naive norm Σ(ψ_R² + ψ_I²) bounded with an O(dt·V_max)
    oscillation.  This oracle pins that promise: it seeds a centred
    Gaussian wavepacket of width close to the trap ground state, samples
    the naive norm at a fixed even-frame stride for ``cap`` steps, and
    requires the *secular drift* (slope of a linear fit, scaled to the
    full run length) to be at machine-precision-tiny levels.  The
    bounded oscillation is reported but not graded.
    """
    from test_harness import HeadlessRunner

    rule = 'quantum_harmonic'
    alpha_val = 2.5          # ħ/2m
    V_scale = 1.0
    omega_pot = 0.02         # matches _harmonic_potential default
    dt = 0.02
    tol_drift = 0.005        # secular drift relative to mean norm

    params = {'ħ/2m': alpha_val, 'V strength': V_scale}

    # Classical frequency in the trap (for diagnostic only — Ehrenfest):
    #   ω_class² = (2α)·V_scale·ω_pot²
    omega_class = math.sqrt(2.0 * alpha_val * V_scale * omega_pot * omega_pot)
    # Ground-state half-width: σ² = α/ω_class.
    sigma_ground = math.sqrt(alpha_val / max(omega_class, 1e-12))

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        c = size / 2.0
        z_idx, y_idx, x_idx = np.mgrid[0:size, 0:size, 0:size]
        dx = (x_idx - c + 0.5).astype(np.float64)
        dy = (y_idx - c + 0.5).astype(np.float64)
        dz = (z_idx - c + 0.5).astype(np.float64)
        r2 = dx * dx + dy * dy + dz * dz

        # Use the larger of (preset Gaussian width) and (ground-state width)
        # so the packet is well-resolved at small `size`.  At size=64 the
        # ground state has σ ≈ 7.5 voxels — easily ≥ Nyquist.
        sigma = max(sigma_ground, size * 0.10)
        env = np.exp(-r2 / (4.0 * sigma * sigma))
        env = env / max(float(env.max()), 1e-30)
        psi_r = env.astype(np.float32)
        psi_i = np.zeros_like(psi_r)
        V_pot = (0.5 * omega_pot * omega_pot * r2).astype(np.float32)
        prob0 = (psi_r * psi_r + psi_i * psi_i).astype(np.float32)

        data = np.zeros((size, size, size, 4), dtype=np.float32)
        data[..., 0] = psi_r
        data[..., 1] = psi_i
        data[..., 2] = V_pot
        data[..., 3] = prob0

        current = r.tex_a if r.ping == 0 else r.tex_b
        current.write(data.tobytes())

        # Sample naive norm Σ(ψ_R² + ψ_I²) on EVEN-frame strides so ψ_I has
        # just been advanced consistently each time.
        stride = max(2, (cap // 25) // 2 * 2)   # ~25 samples, even
        norm0 = float((psi_r.astype(np.float64) ** 2
                       + psi_i.astype(np.float64) ** 2).sum())
        steps_list: list[int] = [0]
        norms_list: list[float] = [norm0]

        nan_step = -1
        for i in range(1, cap + 1):
            r.step()
            if i % stride == 0:
                g = np.asarray(r.read_grid())
                if not np.isfinite(g).all():
                    nan_step = i
                    break
                psR = g[..., 0].astype(np.float64)
                psI = g[..., 1].astype(np.float64)
                norms_list.append(float((psR * psR + psI * psI).sum()))
                steps_list.append(i)

        if nan_step >= 0:
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'quantum_harmonic_norm_conservation',
                    'reason': f'NaN/Inf in grid at step {nan_step}'}

        steps_arr = np.asarray(steps_list, dtype=np.float64)
        norms_arr = np.asarray(norms_list, dtype=np.float64)
        norm_mean = float(norms_arr.mean())
        norm_min = float(norms_arr.min())
        norm_max = float(norms_arr.max())
        osc_rel = (norm_max - norm_min) / max(abs(norm_mean), 1e-30)

        if len(norms_arr) >= 2:
            slope, _intercept = np.polyfit(steps_arr, norms_arr, 1)
        else:
            slope = 0.0
        secular_rel = abs(slope * cap) / max(abs(norm_mean), 1e-30)

        # Probability-channel consistency at end.
        g_final = np.asarray(r.read_grid())
        prob_meas = g_final[..., 3].astype(np.float64)
        prob_calc = (g_final[..., 0].astype(np.float64) ** 2
                     + g_final[..., 1].astype(np.float64) ** 2)
        prob_max_abs = float(np.abs(prob_meas - prob_calc).max())
        prob_rel = prob_max_abs / max(float(prob_calc.max()), 1e-30)

        rel_err = secular_rel
        if rel_err < tol_drift:
            grade = 'ok'
        elif rel_err < 2.0 * tol_drift:
            grade = 'high'
        else:
            grade = 'crit'
        # Probability channel must also match (it's just R² + I²).
        if prob_rel > 1e-3 and grade == 'ok':
            grade = 'high'

        return {
            'rule': rule,
            'oracle': 'quantum_harmonic_norm_conservation',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'alpha_hbar_2m': alpha_val,
            'V_scale': V_scale,
            'omega_pot': omega_pot,
            'omega_class': omega_class,
            'period_classical': 2.0 * math.pi / max(omega_class, 1e-12),
            'sigma_ground': sigma_ground,
            'sigma_used': sigma,
            'samples': int(len(norms_arr)),
            'sample_stride': stride,
            'norm_initial': float(norms_arr[0]),
            'norm_final': float(norms_arr[-1]),
            'norm_mean': norm_mean,
            'norm_min': norm_min,
            'norm_max': norm_max,
            'oscillation_rel': osc_rel,
            'slope_per_step': float(slope),
            'secular_drift_rel': secular_rel,
            'prob_channel_max_abs_err': prob_max_abs,
            'prob_channel_rel_err': prob_rel,
            'rel_err': rel_err,
            'tol': tol_drift,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


@oracle('kuramoto_3d')
def _kuramoto_synced_drift(ctx, size: int, seed: int, cap: int) -> dict:
    """Fully-synchronised Kuramoto fixed point: φ̇ = ω·s.

    The kuramoto_3d shader updates each phase by
        Δφ = (ω·s + K·⟨sin(φ_j − φ_i)⟩ + noise)·dt
    and each natural-frequency by
        Δω = a·R·(⟨ω_j⟩ − ω_i)·dt,
    where R = √(⟨sin Δφ⟩² + ⟨cos Δφ⟩²) is the local order parameter.

    With a fully phase-locked, uniform-ω initial condition the coupling
    sum vanishes (sin 0 = 0), the order parameter is R = 1, the
    frequency-adaptation pull (⟨ω_j⟩ − ω_i) = 0, and (with noise_amp
    forced to 0) the phase advances purely ballistically at the
    constant rate ω·s per unit time:
        φ(N) = fract(φ(0) + N·dt·ω·s).
    Frequency stays exact, coherence stays at 1, and every voxel
    agrees to machine precision (no spatial dispersion at all).
    """
    from test_harness import HeadlessRunner

    rule = 'kuramoto_3d'
    K_val = 0.5
    freq_scale = 0.1
    adaptation = 0.5
    dt = 0.1
    tol_rel = 0.01

    phase0 = 0.25                       # arbitrary, well inside [0,1)
    omega0 = 0.7                        # well inside ±2 clamp
    # NOISE MUST BE ZERO — otherwise the stochastic kick destroys the
    # closed-form prediction even though everything else cancels.
    params = {'Coupling K': K_val, 'Noise': 0.0,
              'Freq scale': freq_scale, 'Adaptation': adaptation}

    increment = cap * dt * omega0 * freq_scale
    pred_phase = (phase0 + increment) - math.floor(phase0 + increment)

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        data = np.zeros((size, size, size, 4), dtype=np.float32)
        data[..., 0] = phase0          # uniform phase
        data[..., 1] = omega0          # uniform natural frequency
        # ch 2 (coherence) and ch 3 are overwritten by the shader each step.
        current = r.tex_a if r.ping == 0 else r.tex_b
        current.write(data.tobytes())

        for _ in range(cap):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'kuramoto_synced_drift',
                    'reason': f'NaN/Inf in grid after {cap} steps'}

        phase_t = g_t[..., 0].astype(np.float64)
        freq_t = g_t[..., 1].astype(np.float64)
        coh_t = g_t[..., 2].astype(np.float64)

        ph_mean = float(phase_t.mean())
        ph_std = float(phase_t.std())
        fr_mean = float(freq_t.mean())
        fr_std = float(freq_t.std())
        co_mean = float(coh_t.mean())
        co_std = float(coh_t.std())

        # Phase error on the circle (handle wrap robustness): use the
        # straight residual since pred_phase ∈ [0, 1) and the predicted
        # value here is 0.95 (well away from the wrap seam at 0/1).
        d = ph_mean - pred_phase
        d -= round(d)              # shortest circular distance
        err_phase = abs(d)
        err_freq = abs(fr_mean - omega0)
        err_coh = abs(co_mean - 1.0)
        rel_err = max(err_phase, err_freq, err_coh)

        if rel_err < tol_rel:
            grade = 'ok'
        elif rel_err < 2.0 * tol_rel:
            grade = 'high'
        else:
            grade = 'crit'
        # Spatial homogeneity should be machine precision.
        if max(ph_std, fr_std, co_std) > 1e-4 and grade == 'ok':
            grade = 'high'

        return {
            'rule': rule,
            'oracle': 'kuramoto_synced_drift',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'omega0': omega0,
            'freq_scale': freq_scale,
            'phase0': phase0,
            'increment_total': increment,
            'phase_measured': ph_mean,
            'phase_predicted': pred_phase,
            'freq_measured': fr_mean,
            'freq_expected': omega0,
            'coherence_measured': co_mean,
            'coherence_expected': 1.0,
            'phase_std': ph_std,
            'freq_std': fr_std,
            'coherence_std': co_std,
            'err_phase': err_phase,
            'err_freq': err_freq,
            'err_coherence': err_coh,
            'rel_err': rel_err,
            'tol': tol_rel,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


@oracle('xy_spin_3d')
def _xy_spin_zero_temp_frozen(ctx, size: int, seed: int, cap: int) -> dict:
    """Zero-temperature XY model: ferromagnetic ground state is frozen.

    The xy_spin_3d shader does a Metropolis sweep on an 8-colour
    decomposition.  With:
      * J > 0  (ferromagnetic coupling)
      * h = 0  (no external field)
      * T → 0  (T is floored at 1e-3 inside the shader)
      * IC θ ≡ 0  (every spin aligned)

    the system sits at the global ground state E = −6J·N.  Any non-zero
    proposed Δθ raises E (cos drops below 1), and with dE/T ≳ 10³ the
    Metropolis acceptance ratio exp(−dE/T) is identically zero in fp32.
    Therefore every proposal is rejected and θ(t) ≡ 0 for all t, exactly.
    The "inactive sub-step" branch in the shader is also exercised
    because 7/8 of voxels emit cos/sin colours each frame without
    touching ch0; we verify it preserves ch0 too.

    Expected final state (bit-exact):
        ch0 = 0,  ch1 = 0.5 + 0.5·cos 0 = 1.0,
        ch2 = 0.5 + 0.5·sin 0 = 0.5,  ch3 = 0.
    """
    from test_harness import HeadlessRunner

    rule = 'xy_spin_3d'
    T_val = 1e-3                     # shader floors at 1e-3 anyway
    # J is boosted past the GUI range max (=2) so that even the small-Δθ
    # tail of proposals (which would otherwise sneak through Metropolis
    # because dE/T ≈ 6J·Δθ²·dθ²·… can be O(1) for Δθ ≲ 0.05) is rejected.
    # With J=10 the smallest non-zero proposal still has dE/T ≳ 50 and
    # exp(−dE/T) underflows to zero in fp32.
    J_val = 10.0
    h_val = 0.0
    sigma_val = 1.5                  # ~π/2 → wide proposals all rejected too
    dt = 1.0                         # matches preset; Metropolis is dt-free
    # Metropolis at T = 1e-3, J = 10 leaves a small residual fluctuation
    # σ_θ ≈ √(T/6J) ≈ 4·10⁻³ rad around the θ = 0 minimum.  The MEAN
    # cos/sin channels are still exactly 1.0 / 0.5 by symmetry; the max
    # deviation per voxel can be a few σ_θ.  We grade on the mean (which
    # is the genuine zero-temperature statement) and only sanity-bound
    # the worst-voxel residual.
    tol_mean = 1e-3                  # ⟨cos θ⟩→1, ⟨sin θ⟩→0
    tol_voxel = 0.05                 # bound on any single Metropolis fluctuation

    params = {'Temperature': T_val, 'J coupling': J_val,
              'Field h': h_val, 'Proposal σ': sigma_val}

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # IC: θ = 0 everywhere (channel 0 stores θ/2π).  ch1..3 are
        # overwritten by the shader on every step regardless.
        data = np.zeros((size, size, size, 4), dtype=np.float32)
        current = r.tex_a if r.ping == 0 else r.tex_b
        current.write(data.tobytes())

        # Need at least 8 sweeps so every colour activates at least once;
        # `cap` is typically 100 — plenty.
        if cap < 8:
            cap = 8
        for _ in range(cap):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'xy_spin_zero_temp_frozen',
                    'reason': f'NaN/Inf in grid after {cap} steps'}

        theta_norm = g_t[..., 0].astype(np.float64)   # θ / 2π
        cos_ch = g_t[..., 1].astype(np.float64)       # 0.5 + 0.5·cos θ
        sin_ch = g_t[..., 2].astype(np.float64)       # 0.5 + 0.5·sin θ
        accept_ch = g_t[..., 3].astype(np.float64)    # |Δθ|/π on accept,
                                                       # 0 on reject/inactive

        # Errors against the bit-exact prediction.  θ wraps via mod(),
        # so a tiny negative drift can show up as θ/2π ≈ 1.0 instead of
        # 0.0 — measure circular distance from 0 on the [0,1) circle.
        # Metropolis at T = 1e-3, J = 10 still has σ_θ ≈ √(T/6J) ≈ 4·10⁻³
        # equilibrium fluctuations around θ = 0 — bounded per-voxel but
        # the MEAN cos/sin are 1 / 0 exactly by symmetry.  We grade on
        # the mean and only sanity-bound the worst voxel.
        circ_dist = np.minimum(theta_norm, 1.0 - theta_norm)
        max_circ_dist = float(circ_dist.max())
        mean_cos = float(cos_ch.mean())
        mean_sin = float(sin_ch.mean())
        max_cos_err = float(np.abs(cos_ch - 1.0).max())
        max_sin_err = float(np.abs(sin_ch - 0.5).max())
        err_accept = float(np.abs(accept_ch).max())

        mean_err = max(abs(mean_cos - 1.0), abs(mean_sin - 0.5))
        voxel_err = max(max_cos_err, max_sin_err)
        rel_err = mean_err

        if mean_err < tol_mean and voxel_err < tol_voxel:
            grade = 'ok'
        elif mean_err < 2.0 * tol_mean and voxel_err < 2.0 * tol_voxel:
            grade = 'high'
        else:
            grade = 'crit'

        return {
            'rule': rule,
            'oracle': 'xy_spin_zero_temp_frozen',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'T': T_val, 'J': J_val, 'h': h_val, 'sigma': sigma_val,
            'theta_circ_max': max_circ_dist,
            'cos_mean': mean_cos,
            'sin_mean': mean_sin,
            'cos_channel_max_abs_err': max_cos_err,
            'sin_channel_max_abs_err': max_sin_err,
            'accept_channel_max': err_accept,
            'mean_err': mean_err,
            'voxel_err': voxel_err,
            'rel_err': rel_err,
            'tol': tol_mean,
            'tol_voxel': tol_voxel,
        }
    finally:
        with contextlib.suppress(Exception):
            r.release()


@oracle('predator_prey_3d')
def _predator_prey_skip(ctx, size: int, seed: int, cap: int) -> dict:
    """Predator-prey is an entity-arena (particle) simulation — not a PDE.

    The PDE-style Rosenzweig-MacArthur shader at simulator.py line ~2642
    is legacy code that is NOT in the active preset. The live preset
    (``_predator_prey_preset`` at line ~16475) wires 6 entity passes:
    ``food_field``, ``_chash``, ``_bhash``, ``prey_step``, ``predator_step``,
    ``_paint``. Prey and predator populations are *individual particles*
    in an EntityArena, spawned/managed by ``on_init``/``on_tick`` hooks,
    and the voxel texture is just a post-hoc paint of their density.

    Consequence: writing a continuum field IC into the source texture has
    no effect on the simulation — entities persist with their spawned
    states and the painter overwrites the texture every tick. No
    closed-form continuum fixed point (Rosenzweig-MacArthur monoculture
    or otherwise) is enforceable as an end-state prediction.

    A meaningful oracle for this rule would need to check entity-level
    invariants (e.g. average prey/predator counts over a long window
    matching the deterministic limit-cycle mean, or mass conservation
    under no-mortality params). That is outside Probe #16's PDE-field
    scope — leaving as ``skip``.
    """
    _ = (ctx, size, seed, cap)
    return {'rule': 'predator_prey_3d',
            'oracle': 'predator_prey_entity_skip',
            'grade': 'skip',
            'reason': ('entity-arena particle sim; no continuum PDE '
                       'end-state to predict (see oracle docstring)')}


@oracle('wireworld_3d')
def _wireworld_pulse_propagation(ctx, size: int, seed: int, cap: int) -> dict:
    """Wireworld single-pulse propagation along a closed conductor loop.

    Build a 1-voxel-thick conductor ring along the z axis at fixed
    (y₀, x₀): every cell (z, y₀, x₀) for z ∈ [0, size) is conductor (state 3),
    except a single head (state 1) at z=0 and a single tail (state 2)
    at z=size−1 (i.e. *behind* the head on the toroidal ring).  All
    other voxels are empty (state 0).

    Classical wireworld transition (with spark_p=0, decay_p=0, head
    activation window [1, 2]):

      head      → tail
      tail      → conductor
      conductor → head  iff  n_head ∈ [1, 2]

    A conductor cell in the 1-voxel-thick ring sees at most 2 head
    neighbours (the immediate ±1 along z); the tail behind the head
    suppresses backward propagation (it is a tail this step, becomes
    conductor next).  So the pulse advances **one cell per step** in
    the +z direction, deterministically:

      step N:  head at z=N (mod size), tail at z=N−1 (mod size).

    The off-ring 26-neighbour cells (the 8 conductor-free voxels in the
    same plane plus the 18 voxels in adjacent planes) are all empty,
    so they cannot count as head neighbours and never fire.  Empty
    cells with decay_p=0 are inert: no spontaneous regrowth.

    Predicted bit-exact end state after `cap` steps:
        head_positions = {(cap % size, y₀, x₀)}
        tail_positions = {((cap − 1) % size, y₀, x₀)}
        conductor      = ring \\ (head ∪ tail)
        empty          = everywhere else.

    Channels (R, G, B, A) = (state/3, is_head, is_tail, is_conductor).
    """
    from test_harness import HeadlessRunner

    rule = 'wireworld_3d'
    dt = 1.0
    params = {'Head min': 1.0, 'Head max': 2.0, 'Spark p': 0.0, 'Decay': 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
    try:
        # Single conductor ring along axis 0 (== GPU z) at fixed y0, x0.
        y0 = size // 2
        x0 = size // 2

        # State encoding: ch0 = state/3, ch1/2/3 = one-hot (head, tail, cond)
        data = np.zeros((size, size, size, 4), dtype=np.float32)
        # Conductor everywhere on the ring (state 3 -> 1.0)
        data[:, y0, x0, 0] = 1.0          # state/3 = 3/3 = 1
        data[:, y0, x0, 3] = 1.0          # is_conductor

        # Head at z=0  (state 1 -> 1/3)
        data[0, y0, x0, 0] = 1.0 / 3.0
        data[0, y0, x0, 3] = 0.0
        data[0, y0, x0, 1] = 1.0          # is_head

        # Tail at z=size-1 (state 2 -> 2/3)
        data[size - 1, y0, x0, 0] = 2.0 / 3.0
        data[size - 1, y0, x0, 3] = 0.0
        data[size - 1, y0, x0, 2] = 1.0   # is_tail

        # Source = ping side; mirror both buffers so the first step reads
        # the planted IC regardless of ping-pong wiring.
        r.tex_a.write(data.tobytes())
        r.tex_b.write(data.tobytes())

        for _ in range(cap):
            r.step()

        g_t = np.asarray(r.read_grid())
        if not np.isfinite(g_t).all():
            return {'rule': rule, 'grade': 'crit',
                    'oracle': 'wireworld_pulse_propagation',
                    'reason': f'NaN/Inf in grid after {cap} steps'}

        # Decode state from channel 0 (round(v*3)).
        state = np.clip(np.round(g_t[..., 0] * 3.0).astype(np.int32), 0, 3)

        head_pos = np.argwhere(state == 1)
        tail_pos = np.argwhere(state == 2)
        cond_count = int((state == 3).sum())
        empty_count = int((state == 0).sum())
        head_count = int(len(head_pos))
        tail_count = int(len(tail_pos))

        expected_head_z = cap % size
        expected_tail_z = (cap - 1) % size
        expected_cond = size - 2  # ring minus the head & tail cells
        expected_empty = size * size * size - size

        # Predicted unique head/tail location
        head_ok = (head_count == 1 and
                   tuple(head_pos[0].tolist()) == (expected_head_z, y0, x0))
        tail_ok = (tail_count == 1 and
                   tuple(tail_pos[0].tolist()) == (expected_tail_z, y0, x0))
        counts_ok = (cond_count == expected_cond and empty_count == expected_empty)

        if head_ok and tail_ok and counts_ok:
            grade = 'ok'
            rel_err = 0.0
        else:
            grade = 'crit'
            # Use a non-zero scalar so grading order works; this oracle
            # is fundamentally pass/fail on integer state counts.
            mismatches = (
                (0 if head_ok else 1)
                + (0 if tail_ok else 1)
                + (0 if counts_ok else 1)
            )
            rel_err = float(mismatches)

        head_loc = (tuple(head_pos[0].tolist()) if head_count == 1 else
                    [tuple(p.tolist()) for p in head_pos[:5]])
        tail_loc = (tuple(tail_pos[0].tolist()) if tail_count == 1 else
                    [tuple(p.tolist()) for p in tail_pos[:5]])

        return {
            'rule': rule,
            'oracle': 'wireworld_pulse_propagation',
            'grade': grade,
            'size': size,
            'steps': cap,
            'dt': dt,
            'expected_head': (expected_head_z, y0, x0),
            'expected_tail': (expected_tail_z, y0, x0),
            'head_count': head_count,
            'tail_count': tail_count,
            'cond_count': cond_count,
            'empty_count': empty_count,
            'expected_cond': expected_cond,
            'expected_empty': expected_empty,
            'head_loc': head_loc,
            'tail_loc': tail_loc,
            'rel_err': rel_err,
            'tol': 0.5,
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
                elif 'phi_re_measured' in row:
                    print(f'          Re φ↑ meas={row["phi_re_measured"]:+.5f}  '
                          f'pred(disc)={row["phi_re_predicted"]:+.5f}  '
                          f'rel_err={row["rel_err"]:.4f}  (tol={row["tol"]:.3f})')
                    print(f'          ω_disc={row["omega_discrete"]:.5f}  '
                          f'ω_cont={row["omega_continuum"]:.5f}  '
                          f'k_voxel={row["k_voxel"]:.4f}  '
                          f'sin(k)={row["sin_k_voxel"]:.4f}  '
                          f'c={row["c_light"]}  m={row["mass"]}  t={row["t_phys"]:.3f}  '
                          f'planar_std={row["planar_std_max"]:.2e}')
                elif 'u_expected' in row and 'v_max_abs' in row:
                    print(f'          u meas={row["u_measured"]:.6f}  '
                          f'(expected K={row["u_expected"]})  '
                          f'err={row["err_u"]:.2e}  (tol_u={row["tol"]:.2e})')
                    print(f'          v max|·|={row["v_max_abs"]:.2e}  '
                          f'mean={row["v_measured"]:+.2e}  '
                          f'std={row["v_std"]:.2e}  (tol_v={row["tol_v"]:.2e})  '
                          f'interaction max={row["interaction_max_abs"]:.2e}')
                    print(f'          λ_v at K={row["lambda_v_at_K"]:+.4f}  '
                          f'(noise amplification over run = exp(λ_v·N·dt))')
                elif 'cos_mean' in row and 'sin_mean' in row:
                    print(f'          ⟨cos θ⟩={row["cos_mean"]:.6f}  '
                          f'(exp 1.000000)  ⟨sin θ⟩={row["sin_mean"]:.6f}  '
                          f'(exp 0.500000)  mean_err={row["mean_err"]:.2e}  '
                          f'(tol={row["tol"]:.2e})')
                    print(f'          worst-voxel: θ_circ={row["theta_circ_max"]:.3e}  '
                          f'cos_err={row["cos_channel_max_abs_err"]:.2e}  '
                          f'sin_err={row["sin_channel_max_abs_err"]:.2e}  '
                          f'accept_ch={row["accept_channel_max"]:.2e}  '
                          f'(voxel_tol={row["tol_voxel"]:.2e})  '
                          f'T={row["T"]:.1e}  J={row["J"]}  σ={row["sigma"]}')
                elif 'phase_measured' in row and 'coherence_measured' in row:
                    print(f'          phase meas={row["phase_measured"]:.5f}  '
                          f'pred={row["phase_predicted"]:.5f}  '
                          f'err={row["err_phase"]:.2e}  (tol={row["tol"]:.3f})')
                    print(f'          freq meas={row["freq_measured"]:.5f}  '
                          f'exp={row["freq_expected"]:.5f}  '
                          f'err={row["err_freq"]:.2e}  '
                          f'coh meas={row["coherence_measured"]:.5f}  '
                          f'err={row["err_coherence"]:.2e}')
                    print(f'          ω₀={row["omega0"]}  s={row["freq_scale"]}  '
                          f'inc_total={row["increment_total"]:.4f}  '
                          f'std(phase/freq/coh)={row["phase_std"]:.2e}/'
                          f'{row["freq_std"]:.2e}/{row["coherence_std"]:.2e}')
                elif 'secular_drift_rel' in row:
                    print(f'          norm init={row["norm_initial"]:.5e}  '
                          f'final={row["norm_final"]:.5e}  '
                          f'mean={row["norm_mean"]:.5e}')
                    print(f'          secular drift={row["secular_drift_rel"]:.2e}  '
                          f'osc(max−min)/mean={row["oscillation_rel"]:.2e}  '
                          f'slope/step={row["slope_per_step"]:+.3e}  '
                          f'(tol={row["tol"]:.3f})  N_samples={row["samples"]}')
                    print(f'          ω_class={row["omega_class"]:.5f}  '
                          f'T_class={row["period_classical"]:.2f}  '
                          f'σ_ground={row["sigma_ground"]:.3f}  '
                          f'σ_used={row["sigma_used"]:.3f}  '
                          f'prob_ch_rel_err={row["prob_channel_rel_err"]:.2e}')
                elif 'dV_measured' in row and 'V_star' in row:
                    print(f'          δV meas={row["dV_measured"]:+.5e}  '
                          f'pred={row["dV_predicted"]:+.5e}  '
                          f'err={row["err_amp_V"]:.4f}  (tol={row["tol"]:.3f})')
                    print(f'          δW meas={row["dW_measured"]:+.5e}  '
                          f'pred={row["dW_predicted"]:+.5e}  err={row["err_amp_W"]:.4f}')
                    print(f'          evec W/V meas={row["evec_ratio_measured"]:+.5f}  '
                          f'exp={row["evec_ratio_expected"]:+.5f}  '
                          f'err={row["err_evec"]:.4f}')
                    print(f'          V*={row["V_star"]:+.4f}  W*={row["W_star"]:+.4f}  '
                          f'λ_+={row["lambda_plus"]:.4f}  growth={row["growth"]:.2f}  '
                          f'ε₀={row["eps0"]:.2e}  '
                          f'homo_std(V/W)={row["homogeneity_std_V"]:.2e}/'
                          f'{row["homogeneity_std_W"]:.2e}')
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
                elif 'expected_head' in row:
                    print(f'          head meas={row["head_loc"]}  '
                          f'exp={row["expected_head"]}  (count={row["head_count"]})')
                    print(f'          tail meas={row["tail_loc"]}  '
                          f'exp={row["expected_tail"]}  (count={row["tail_count"]})')
                    print(f'          conductors={row["cond_count"]} (exp {row["expected_cond"]})  '
                          f'empty={row["empty_count"]} (exp {row["expected_empty"]})  '
                          f'mismatches={row["rel_err"]:.0f}')
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
