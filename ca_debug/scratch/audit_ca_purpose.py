"""Purpose audit: does each CA actually produce structure matching its label,
or is it just a 64^3 cube of incoherent noise?

For every preset, on the *visible* channel(s), measure:

  active_frac          fraction of voxels with |v| > eps
  spatial_corr_len     1/e autocorrelation length in voxels
                       (≈1.0 = white noise, ≈feature_scale = patterned)
  temporal_corr        Pearson corr between snapshots 50 steps apart at t=1500
                       (≈1.0 = static/persistent, ≈0 = noise/turbulent)
  spectral_peak        location of dominant radial FFT peak (0 = no peak)
  spectral_slope       log-log slope of radial power spectrum
                       (≈0 = white noise, < -2 = structured/red noise)
  drift_per_step       L2(g[1500] - g[1450]) / sqrt(N) — magnitude of change
  monochrome           True if visible channel collapsed to <0.01 std
  saturation_frac      fraction of voxels at |v| > 0.99·max(|v|)
  bbox_fill            volume of axis-aligned bbox of active voxels / box vol
  com_drift            distance the centre-of-mass moved between t=1500 & t=1700
                       (waves/flocking should be > 0; static crystals = 0)
  conservation_drift   |mean(t=1500) - mean(t=200)| / max(|mean|, 1e-6)
                       (low for diffusion/quantum; high for growth)

Plus a verdict per CA: STRUCTURED / NOISE_CUBE / FROZEN / EXTINCT / SATURATED.
"""

from __future__ import annotations

import os
import sys
import time
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from test_harness import create_headless_context, HeadlessRunner  # noqa: E402
from simulator import RULE_PRESETS  # noqa: E402


def _radial_power_spectrum(field: np.ndarray) -> np.ndarray:
    """Return 1D radially-averaged power spectrum of a 3D real field."""
    f = np.fft.fftn(field - field.mean())
    p = (f * np.conj(f)).real
    sz = field.shape[0]
    kx = np.fft.fftfreq(sz) * sz
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    kr = np.sqrt(KX*KX + KY*KY + KZ*KZ)
    nbins = sz // 2
    bins = np.linspace(0, nbins, nbins + 1)
    idx = np.digitize(kr.ravel(), bins) - 1
    radial = np.zeros(nbins)
    counts = np.zeros(nbins)
    pflat = p.ravel()
    for i, b in enumerate(idx):
        if 0 <= b < nbins:
            radial[b] += pflat[i]
            counts[b] += 1
    counts[counts == 0] = 1
    return radial / counts


def _spatial_corr_len(field: np.ndarray, max_lag: int = 16) -> float:
    """Estimate isotropic 1/e correlation length via X/Y/Z averaged autocorr."""
    f = field - field.mean()
    var = (f * f).mean()
    if var < 1e-12:
        return 0.0
    lengths = []
    for axis in range(3):
        # Average correlation along axis
        for lag in range(1, max_lag + 1):
            shifted = np.roll(f, lag, axis=axis)
            c = float((f * shifted).mean() / var)
            if c < 1.0 / math.e:
                lengths.append(lag - 1 + (lengths_prev := 1.0))  # approx
                break
        else:
            lengths.append(float(max_lag))
    return float(np.mean(lengths))


def _com(field: np.ndarray) -> tuple:
    a = np.abs(field)
    s = a.sum()
    if s < 1e-9:
        return (0.0, 0.0, 0.0)
    sz = field.shape[0]
    coords = np.arange(sz)
    cx = float((a.sum(axis=(1, 2)) * coords).sum() / s)
    cy = float((a.sum(axis=(0, 2)) * coords).sum() / s)
    cz = float((a.sum(axis=(0, 1)) * coords).sum() / s)
    return (cx, cy, cz)


def _bbox_fill(field: np.ndarray, thresh: float) -> float:
    mask = np.abs(field) > thresh
    if not mask.any():
        return 0.0
    xs, ys, zs = np.where(mask)
    vol = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) * (zs.max() - zs.min() + 1)
    return float(vol) / float(field.size)


def analyze(rule: str, ctx, size: int = 64, seed: int = 42,
            warmup: int = 500, measure_at: int = 1500) -> dict:
    """Run one preset and harvest structural metrics on its visible channel."""
    preset = RULE_PRESETS[rule]
    vis_default = preset.get('vis_default', 0)
    runner = HeadlessRunner(ctx, rule, size=size, seed=seed)
    actual_size = runner.read_grid().shape[0]
    # Warmup
    for _ in range(warmup):
        runner.step()
    g0 = runner.read_grid().copy()
    # Step to measure_at
    for _ in range(measure_at - warmup):
        runner.step()
    g1 = runner.read_grid().copy()
    # Step further for temporal correlation and com drift
    for _ in range(50):
        runner.step()
    g2 = runner.read_grid().copy()
    for _ in range(150):
        runner.step()
    g3 = runner.read_grid().copy()
    runner.release()

    field1 = g1[..., vis_default].astype(np.float32)
    field2 = g2[..., vis_default].astype(np.float32)
    field3 = g3[..., vis_default].astype(np.float32)
    field0 = g0[..., vis_default].astype(np.float32)

    if not np.isfinite(field1).all():
        return dict(rule=rule, verdict='NaN', size=actual_size,
                    nan_count=int((~np.isfinite(field1)).sum()))

    eps = max(1e-3, 0.02 * float(np.abs(field1).max()))
    active_frac = float((np.abs(field1) > eps).mean())
    fmax = float(np.abs(field1).max())
    fstd = float(field1.std())
    fmean = float(field1.mean())

    # Temporal correlation
    f1c = field1 - field1.mean()
    f2c = field2 - field2.mean()
    denom = math.sqrt(float((f1c*f1c).sum()) * float((f2c*f2c).sum()))
    temporal_corr = float((f1c * f2c).sum() / denom) if denom > 1e-9 else 1.0

    # Drift magnitude
    drift = float(np.sqrt(((field2 - field1)**2).mean()))

    # Spectral
    spec = _radial_power_spectrum(field1)
    if spec[1:].max() > 0:
        peak_k = int(np.argmax(spec[1:]) + 1)
        # log-log slope of spectrum from k=1..N/4
        ks = np.arange(1, max(2, len(spec)//4))
        ps = spec[1:max(2, len(spec)//4)]
        valid = ps > 1e-12
        if valid.sum() >= 3:
            slope = float(np.polyfit(np.log(ks[valid]), np.log(ps[valid]), 1)[0])
        else:
            slope = 0.0
    else:
        peak_k = 0
        slope = 0.0

    # Spatial corr length
    corr_len = _spatial_corr_len(field1)

    # COM drift
    com1 = _com(field1)
    com3 = _com(field3)
    com_drift = math.sqrt(sum((a-b)**2 for a, b in zip(com1, com3)))

    # Saturation / monochrome
    monochrome = fstd < 0.01 * (abs(fmean) + 1.0)
    sat_frac = float((np.abs(field1) > 0.99 * fmax).mean()) if fmax > 1e-6 else 0.0

    # Bbox fill
    bbox = _bbox_fill(field1, eps)

    # Conservation drift (use grid mass on visible channel)
    m0 = float(np.abs(field0).mean())
    m1 = float(np.abs(field1).mean())
    cons = abs(m1 - m0) / max(abs(m0) + abs(m1), 1e-6) * 2

    # Verdict heuristic.
    #
    # The original rules false-positived on several real CAs at size=64
    # because their characteristic features happen to be ~1 voxel wide
    # at that resolution (Life-like attractors, Lenia, BZ wavefronts,
    # element chemistry). The refined rules below require multiple
    # weak signals to converge before declaring a failure mode.
    if monochrome and active_frac < 0.01:
        verdict = 'EXTINCT'
    elif sat_frac > 0.9:
        verdict = 'SATURATED'
    elif drift < 1e-5 and temporal_corr > 0.999:
        verdict = 'FROZEN'
    elif (corr_len < 1.5 and abs(temporal_corr) < 0.3
          and peak_k <= 2 and active_frac > 0.5):
        # Truly random-looking: tiny spatial features, no temporal
        # coherence, no spectral peak, AND fills most of the volume
        # (random-fill noise saturates active_frac).
        verdict = 'NOISE_CUBE'
    elif (corr_len < 1.5 and slope > -0.5
          and peak_k <= 2 and abs(temporal_corr) < 0.3):
        # White spectrum + tiny features + no temporal coherence +
        # no spectral peak: looks like dynamic noise.
        verdict = 'NOISE_CUBE'
    else:
        verdict = 'STRUCTURED'

    return dict(
        rule=rule, size=actual_size, verdict=verdict,
        active_frac=round(active_frac, 4),
        corr_len=round(corr_len, 2),
        temporal_corr=round(temporal_corr, 3),
        drift=round(drift, 4),
        peak_k=peak_k,
        spec_slope=round(slope, 2),
        com_drift=round(com_drift, 2),
        bbox_fill=round(bbox, 3),
        sat_frac=round(sat_frac, 3),
        cons_drift=round(cons, 3),
        fmax=round(fmax, 3),
        fstd=round(fstd, 4),
    )


def main():
    ctx = create_headless_context()
    if isinstance(ctx, tuple):
        _, ctx = ctx
    rules = sorted(RULE_PRESETS.keys())
    print(f'\nRunning purpose audit on {len(rules)} CAs (size=64, 1700 steps each)...\n')
    results = []
    for r in rules:
        t0 = time.time()
        try:
            res = analyze(r, ctx)
            elapsed = time.time() - t0
            res['t'] = round(elapsed, 1)
            results.append(res)
            print(f'  {r:<26} {res["verdict"]:<11} af={res["active_frac"]:.3f} '
                  f'cl={res["corr_len"]:.1f} tc={res["temporal_corr"]:+.2f} '
                  f'k*={res["peak_k"]:>2} slope={res["spec_slope"]:+.2f} '
                  f'comΔ={res["com_drift"]:.1f} bbox={res["bbox_fill"]:.2f} '
                  f'sat={res["sat_frac"]:.2f} ({elapsed:.1f}s)')
        except Exception as e:
            print(f'  {r:<26} ERROR: {str(e)[:120]}')
            results.append(dict(rule=r, verdict='ERROR', error=str(e)[:200]))

    # Group by verdict
    print('\n=== SUMMARY ===\n')
    by_verdict = {}
    for r in results:
        by_verdict.setdefault(r['verdict'], []).append(r['rule'])
    for v in sorted(by_verdict):
        print(f'{v} ({len(by_verdict[v])}): {", ".join(by_verdict[v])}')

    # Save JSON
    import json
    out = os.path.join(ROOT, 'debug_runs', 'purpose_audit.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
