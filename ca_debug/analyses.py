"""Pure-numpy analyses on voxel grids and metric histories.

All functions here are stateless: they take a `grid` (W,H,D,C float array)
or a sequence of grids / metric dicts and return a small dict of named
metrics. They were extracted from `test_harness.py` so the GUI overlay
and replay tools can call them without importing the harness's heavy
CLI/discovery machinery.

Conventions:

* `grid` is a single snapshot (W,H,D,C). `channel` defaults to 0.
* `grid_snapshots` is a list of grids ordered in time.
* `metric_history` is a list of dicts, each with at least ``alive_ratio``.
* All "scores" are normalized to [0, 1] where higher = more interesting.

Top-level entry points:

* :func:`analyze_structure` — structural metrics on a single snapshot.
* :func:`analyze_dynamics`  — dynamic metrics across a snapshot sequence.

The individual detectors (``detect_period``, ``detect_translation``,
``detect_growth``, ``analyze_clusters``, ``measure_symmetry``) are also
exported for piecemeal use in the GUI.
"""
from __future__ import annotations

import numpy as np


# ── Slice / projection helpers ────────────────────────────────────────

def _binary_slice(grid_3d, axis, index, threshold=0.5):
    """Extract a 2D binary slice from a 3D grid along the given axis."""
    if axis == 0:
        s = grid_3d[index, :, :]
    elif axis == 1:
        s = grid_3d[:, index, :]
    else:
        s = grid_3d[:, :, index]
    return (s > threshold).astype(np.float32)


def slice_gol_coherence(grid, channel=0, axis=2, threshold=0.5):
    """Measure how well consecutive Z-slices follow 2D GoL rules.

    Returns a value in [0, 1] where 1.0 means each slice is the exact
    GoL successor of the previous slice. This detects whether the 3D
    structure embeds a 2D GoL spacetime.
    """
    vol = grid[:, :, :, channel]
    size = vol.shape[axis]
    if size < 2:
        return 0.0

    # Binarize once and move the stacking axis to front so b[i] is the
    # i-th slice. Vectorized GoL step across ALL slices in a single pass.
    b = (vol > threshold).astype(np.int8)
    b = np.moveaxis(b, axis, 0)  # (size, H, W)
    rp1 = np.roll(b, 1, axis=1); rm1 = np.roll(b, -1, axis=1)
    cp1 = np.roll(b, 1, axis=2); cm1 = np.roll(b, -1, axis=2)
    n = (rp1 + rm1 + cp1 + cm1 +
         np.roll(rp1, 1, axis=2) + np.roll(rp1, -1, axis=2) +
         np.roll(rm1, 1, axis=2) + np.roll(rm1, -1, axis=2))
    predicted = ((b == 0) & (n == 3)) | ((b == 1) & ((n == 2) | (n == 3)))
    pred_cur = predicted[:-1]
    actual_next = b[1:].astype(bool)
    alive_cur  = b[:-1].mean(axis=(1, 2))
    alive_next = b[1:].mean(axis=(1, 2))
    trivial = ((alive_cur < 0.01) & (alive_next < 0.01)) | \
              ((alive_cur > 0.99) & (alive_next > 0.99))
    agreement = (pred_cur == actual_next).mean(axis=(1, 2))
    valid = ~trivial
    return float(agreement[valid].mean()) if valid.any() else 0.0


def projection_entropy(grid, channel=0, threshold=0.01):
    """Compute Shannon entropy of max-projection along each axis.

    Returns dict with 'entropy_x', 'entropy_y', 'entropy_z' (each 0-1
    normalized) and 'projection_complexity' (mean entropy).
    """
    vol = grid[:, :, :, channel]
    vol = np.abs(vol)  # handle wave-type CAs

    entropies = {}
    for ax, name in enumerate(['x', 'y', 'z']):
        proj = np.max(vol, axis=ax)
        pmax = proj.max()
        if pmax > threshold:
            proj = proj / pmax
        else:
            entropies[f'entropy_{name}'] = 0.0
            continue
        bins = 16
        hist, _ = np.histogram(proj.ravel(), bins=bins, range=(0, 1))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        ent = -np.sum(hist * np.log2(hist))
        entropies[f'entropy_{name}'] = float(ent / np.log2(bins))

    entropies['projection_complexity'] = float(np.mean([
        entropies['entropy_x'], entropies['entropy_y'], entropies['entropy_z']
    ]))
    return entropies


def projection_structure(grid, channel=0):
    """Measure spatial structure in projections (edge density).

    Returns dict with 'structure_x/y/z' and 'projection_structure' mean.
    Higher values = more internal spatial pattern (edges, boundaries).
    """
    vol = np.abs(grid[:, :, :, channel])

    structures = {}
    for ax, name in enumerate(['x', 'y', 'z']):
        proj = np.max(vol, axis=ax)
        pmax = proj.max()
        if pmax < 0.01:
            structures[f'structure_{name}'] = 0.0
            continue
        proj = proj / pmax
        dx = np.abs(np.diff(proj, axis=0))
        dy = np.abs(np.diff(proj, axis=1))
        edge_density = (dx.mean() + dy.mean()) / 2.0
        structures[f'structure_{name}'] = float(edge_density)

    structures['projection_structure'] = float(np.mean([
        structures['structure_x'], structures['structure_y'], structures['structure_z']
    ]))
    return structures


def slice_mutual_info(grid, channel=0, axis=2, n_samples=8, threshold=0.5):
    """Mutual information between evenly-spaced slices along an axis.

    High MI = slices are related (3D structure has depth coherence).
    Low MI  = slices are independent (random 3D noise).
    Returns value in [0, 1].
    """
    vol = grid[:, :, :, channel]
    size = vol.shape[axis]
    if size < n_samples:
        n_samples = size

    indices = np.linspace(0, size - 1, n_samples, dtype=int)
    slices = [_binary_slice(vol, axis, i, threshold).ravel() for i in indices]

    mis = []
    bins = 2  # binary slices
    for i in range(len(slices)):
        for j in range(i + 1, len(slices)):
            joint = slices[i] * bins + slices[j]
            hist = np.bincount(joint.astype(int), minlength=bins * bins).astype(float)
            hist /= hist.sum()
            hist = hist.reshape(bins, bins)
            px = hist.sum(axis=1)
            py = hist.sum(axis=0)
            mi = 0.0
            for xi in range(bins):
                for yi in range(bins):
                    if hist[xi, yi] > 0 and px[xi] > 0 and py[yi] > 0:
                        mi += hist[xi, yi] * np.log2(hist[xi, yi] / (px[xi] * py[yi]))
            mis.append(mi)

    return float(np.mean(mis)) if mis else 0.0


def spatial_variation(grid, channel=0, n_blocks=8):
    """Spatial heterogeneity via block-level coefficient of variation.

    Returns [0, 1]: 0 = spatially uniform, 1 = highly heterogeneous.
    """
    vol = np.abs(grid[:, :, :, channel])
    sz = vol.shape[0]
    bsz = max(1, sz // n_blocks)
    if sz == bsz * n_blocks:
        block_means = vol.reshape(n_blocks, bsz, n_blocks, bsz, n_blocks, bsz)\
                         .mean(axis=(1, 3, 5)).ravel()
    else:
        block_means = []
        for ix in range(0, sz, bsz):
            for iy in range(0, sz, bsz):
                for iz in range(0, sz, bsz):
                    block = vol[ix:ix+bsz, iy:iy+bsz, iz:iz+bsz]
                    block_means.append(block.mean())
        block_means = np.array(block_means)
    mean_val = block_means.mean()
    if mean_val < 1e-6:
        return 0.0
    cv = float(block_means.std() / mean_val)
    return min(cv, 1.0)


def analyze_structure(grid, channel=0):
    """Run all structural analysis on a single grid snapshot."""
    result = {}
    result['gol_coherence_z'] = slice_gol_coherence(grid, channel, axis=2)
    result['gol_coherence_y'] = slice_gol_coherence(grid, channel, axis=1)
    result['gol_coherence_x'] = slice_gol_coherence(grid, channel, axis=0)
    result['gol_coherence_max'] = max(
        result['gol_coherence_z'], result['gol_coherence_y'], result['gol_coherence_x']
    )
    result.update(projection_entropy(grid, channel))
    result.update(projection_structure(grid, channel))
    result['slice_mi_z'] = slice_mutual_info(grid, channel, axis=2)
    result['slice_mi_y'] = slice_mutual_info(grid, channel, axis=1)
    result['slice_mi_x'] = slice_mutual_info(grid, channel, axis=0)
    result['slice_mi_max'] = max(
        result['slice_mi_z'], result['slice_mi_y'], result['slice_mi_x']
    )
    result['spatial_variation'] = spatial_variation(grid, channel)
    return result


# ── Advanced dynamics metrics (period, gliders, growth, symmetry) ─────

def _grid_hash(binary_grid):
    """Fast hash of a binary grid for period detection."""
    return hash(binary_grid.tobytes())


def detect_period(grid_snapshots, channel=0, threshold=0.5):
    """Detect exact periodicity in a sequence of grid snapshots.

    Returns dict with:
      period: cycle length (0 = no period found)
      period_start: step index where cycle starts
      period_score: 0-1, how clean the period is (1 = perfect cycle)
    """
    hashes = []
    for g in grid_snapshots:
        binary = (g[:, :, :, channel] > threshold).astype(np.uint8)
        hashes.append(_grid_hash(binary))

    n = len(hashes)
    best_period = 0
    best_start = 0
    best_confirmations = 0

    max_period = min(n // 3, 200)
    for p in range(1, max_period + 1):
        for start in range(n - 2 * p, max(0, n - 4 * p) - 1, -1):
            confirmations = 0
            valid = True
            for k in range(start, n - p, p):
                if hashes[k] == hashes[k + p]:
                    confirmations += 1
                else:
                    valid = False
                    break
            if valid and confirmations >= 2 and confirmations > best_confirmations:
                best_period = p
                best_start = start
                best_confirmations = confirmations

    period_score = 0.0
    if best_period > 0:
        period_score = min(1.0, best_confirmations / 5.0) * min(1.0, 20.0 / best_period)

        # Devalue trivial period-2 global oscillation.
        if best_period <= 2 and len(grid_snapshots) >= 2:
            last = grid_snapshots[-1]
            alive_frac = (last[:, :, :, channel] > threshold).mean()
            if alive_frac > 0.2:
                prev = grid_snapshots[-2]
                changed = ((last[:, :, :, channel] > threshold) !=
                          (prev[:, :, :, channel] > threshold)).mean()
                if changed > 0.3:
                    period_score *= 0.1

    return {
        'period': best_period,
        'period_start': best_start,
        'period_score': float(period_score),
    }


def detect_translation(grid_snapshots, channel=0, threshold=0.5):
    """Detect translating structures (gliders/spaceships) via FFT phase correlation.

    Returns dict with:
      translation_score: 0-1 (1 = perfect glider-like translation)
      translation_speed: cells/step of detected translation
      translation_dir: (dx, dy, dz) unit direction
    """
    if len(grid_snapshots) < 10:
        return {'translation_score': 0.0, 'translation_speed': 0.0, 'translation_dir': (0, 0, 0)}

    half = len(grid_snapshots) // 2
    snaps = grid_snapshots[half:]
    size = snaps[0].shape[0]

    bins = [(g[:, :, :, channel] > threshold).astype(np.float32) for g in snaps]

    alive = bins[-1].mean()
    if alive < 0.005 or alive > 0.5:
        return {'translation_score': 0.0, 'translation_speed': 0.0, 'translation_dir': (0, 0, 0)}

    best_score = 0.0
    best_shift = (0, 0, 0)
    best_dt = 1

    for dt_steps in [2, 4, 8]:
        if dt_steps >= len(bins):
            continue

        pairs = list(range(0, len(bins) - dt_steps, max(1, dt_steps)))
        if not pairs:
            continue

        accum = None
        for i in pairs:
            fa = np.fft.rfftn(bins[i])
            fb = np.fft.rfftn(bins[i + dt_steps])
            cross = fa * np.conj(fb)
            mag = np.abs(cross)
            mag[mag < 1e-12] = 1e-12
            if accum is None:
                accum = cross / mag
            else:
                accum += cross / mag

        corr = np.fft.irfftn(accum, s=(size, size, size))
        corr[0, 0, 0] = 0.0

        peak_val = 0.0
        peak_shift = (0, 0, 0)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-3, 4):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    val = corr[dx % size, dy % size, dz % size]
                    if val > peak_val:
                        peak_val = val
                        peak_shift = (dx, dy, dz)

        norm_score = peak_val / max(len(pairs), 1)

        if norm_score > 0.1:
            overlaps = []
            check_pairs = pairs[:min(4, len(pairs))]
            dx, dy, dz = peak_shift
            for i in check_pairs:
                a = bins[i]
                b = bins[i + dt_steps]
                b_shifted = np.roll(np.roll(np.roll(b, -dx, axis=0), -dy, axis=1), -dz, axis=2)
                union_sum = np.maximum(a, b_shifted).sum()
                if union_sum > 10:
                    overlaps.append(float((a * b_shifted).sum() / union_sum))
            if overlaps:
                iou_score = np.mean(overlaps)
                if iou_score > best_score:
                    best_score = iou_score
                    best_shift = peak_shift
                    best_dt = dt_steps

    speed = np.sqrt(best_shift[0]**2 + best_shift[1]**2 + best_shift[2]**2) / best_dt if best_dt > 0 else 0

    return {
        'translation_score': float(best_score),
        'translation_speed': float(speed),
        'translation_dir': best_shift,
    }


def detect_growth(metric_history):
    """Detect monotonic growth patterns (guns, replicators).

    Expects ``metric_history`` items to have key ``alive_ratio``.

    Returns dict with:
      growth_score: 0-1 (1 = steady monotonic growth from sparse start)
      growth_rate: alive cells gained per step
      growth_type: 'none' | 'linear' | 'accelerating' | 'decelerating'
    """
    if len(metric_history) < 10:
        return {'growth_score': 0.0, 'growth_rate': 0.0, 'growth_type': 'none'}

    alive = np.array([m['alive_ratio'] for m in metric_history])

    if alive[0] > 0.3:
        return {'growth_score': 0.0, 'growth_rate': 0.0, 'growth_type': 'none'}

    n = len(alive)
    quarters = [alive[i*n//4:(i+1)*n//4].mean() for i in range(4)]

    monotonic_quarters = sum(1 for i in range(3) if quarters[i+1] > quarters[i] * 1.02)
    if monotonic_quarters < 2:
        return {'growth_score': 0.0, 'growth_rate': 0.0, 'growth_type': 'none'}

    growth = alive[-1] - alive[0]
    if growth < 0.01:
        return {'growth_score': 0.0, 'growth_rate': 0.0, 'growth_type': 'none'}

    mid = alive[n//2]
    expected_linear_mid = (alive[0] + alive[-1]) / 2
    if mid > expected_linear_mid * 1.1:
        growth_type = 'accelerating'
    elif mid < expected_linear_mid * 0.9:
        growth_type = 'decelerating'
    else:
        growth_type = 'linear'

    if alive[-1] > 0.9:
        score = 0.2
    elif alive[-1] > 0.5:
        score = 0.5
    else:
        score = 0.8

    diffs = np.diff(alive)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) > 5:
        cv = np.std(positive_diffs) / (np.mean(positive_diffs) + 1e-10)
        if cv < 0.3:
            score = min(1.0, score + 0.2)

    rate = growth / len(alive)

    return {
        'growth_score': float(score),
        'growth_rate': float(rate),
        'growth_type': growth_type,
    }


def analyze_clusters(grid, channel=0, threshold=0.5):
    """Connected-component analysis of a single grid snapshot.

    Returns dict with:
      n_clusters: number of connected components (6-connectivity)
      cluster_score: 0-1 (high = multiple well-separated structures)
      largest_cluster_frac: fraction of alive cells in largest cluster
      mean_cluster_size: average cluster size in cells
    """
    from scipy import ndimage

    vol = (grid[:, :, :, channel] > threshold).astype(np.int32)
    alive = vol.sum()
    if alive < 5:
        return {'n_clusters': 0, 'cluster_score': 0.0,
                'largest_cluster_frac': 0.0, 'mean_cluster_size': 0}

    structure = np.zeros((3, 3, 3), dtype=np.int32)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1
    labels, n_clusters = ndimage.label(vol, structure=structure)

    if n_clusters == 0:
        return {'n_clusters': 0, 'cluster_score': 0.0,
                'largest_cluster_frac': 0.0, 'mean_cluster_size': 0}

    sizes = ndimage.sum(vol, labels, range(1, n_clusters + 1))
    sizes = np.array(sizes, dtype=float)
    largest = sizes.max()
    mean_size = sizes.mean()

    alive_frac = alive / max(1, grid.shape[0] * grid.shape[1] * grid.shape[2])
    if n_clusters == 1:
        score = 0.1
    elif n_clusters > 1000:
        score = 0.05
    elif n_clusters > 50 and alive_frac > 0.2:
        score = 0.05
    else:
        size_variety = 1.0 - (largest / alive)
        count_score = min(1.0, n_clusters / 10.0) * min(1.0, 50.0 / max(n_clusters, 1))
        score = 0.3 * count_score + 0.4 * size_variety + 0.3 * min(1.0, mean_size / 100.0)

    return {
        'n_clusters': int(n_clusters),
        'cluster_score': float(min(1.0, score)),
        'largest_cluster_frac': float(largest / alive),
        'mean_cluster_size': float(mean_size),
    }


def measure_symmetry(grid, channel=0, threshold=0.5):
    """Reflective + rotational symmetry of a single grid snapshot.

    Returns dict with:
      symmetry_score: 0-1 (1 = perfectly symmetric under all transforms)
      reflection_score: avg reflective symmetry (x, y, z mirrors)
      rotation_score:   avg rotational symmetry (90° rotations)
    """
    vol = (grid[:, :, :, channel] > threshold).astype(np.float32)
    alive = vol.sum()
    if alive < 5:
        return {'symmetry_score': 0.0, 'reflection_score': 0.0, 'rotation_score': 0.0}

    total = vol.size

    ref_scores = []
    for ax in range(3):
        flipped = np.flip(vol, axis=ax)
        agreement = np.sum(vol == flipped) / total
        ref_scores.append(agreement)

    rot_scores = []
    rotated = np.rot90(vol, k=1, axes=(0, 1))
    rot_scores.append(np.sum(vol == rotated) / total)
    rotated = np.rot90(vol, k=1, axes=(0, 2))
    rot_scores.append(np.sum(vol == rotated) / total)
    rotated = np.rot90(vol, k=1, axes=(1, 2))
    rot_scores.append(np.sum(vol == rotated) / total)

    ref_mean = float(np.mean(ref_scores))
    rot_mean = float(np.mean(rot_scores))

    alive_frac = alive / total
    baseline = (1 - alive_frac)**2 + alive_frac**2
    sym_score = max(0.0, (ref_mean + rot_mean) / 2.0 - baseline) / max(0.01, 1.0 - baseline)

    return {
        'symmetry_score': float(min(1.0, sym_score)),
        'reflection_score': float(max(0, ref_mean - baseline) / max(0.01, 1.0 - baseline)),
        'rotation_score': float(max(0, rot_mean - baseline) / max(0.01, 1.0 - baseline)),
    }


def analyze_dynamics(grid_snapshots, metric_history, channel=0, threshold=0.5):
    """Run all advanced dynamics analyses on a snapshot sequence.

    Returns a dict with period, translation, growth, cluster, and
    symmetry metrics combined.
    """
    result = {}
    result.update(detect_period(grid_snapshots, channel, threshold))
    result.update(detect_translation(grid_snapshots, channel, threshold))
    result.update(detect_growth(metric_history))
    result.update(analyze_clusters(grid_snapshots[-1], channel, threshold))
    result.update(measure_symmetry(grid_snapshots[-1], channel, threshold))
    return result


__all__ = [
    'slice_gol_coherence', 'projection_entropy', 'projection_structure',
    'slice_mutual_info', 'spatial_variation', 'analyze_structure',
    'detect_period', 'detect_translation', 'detect_growth',
    'analyze_clusters', 'measure_symmetry', 'analyze_dynamics',
]
