#!/usr/bin/env python3
"""
ghost_cube_audit.py
═══════════════════

Headlessly renders each volumetric CA preset and detects the
"ghost cube" / opaque-uniform-cube rendering artifact.

Symptoms detected:
  1. Interior pixels of the rendered image have very low colour variance
     (the cube projection is filled with one near-uniform colour).
  2. A high fraction of foreground (non-background) pixels are saturated
     near the same RGB triplet.
  3. Edge pixels (silhouette) and interior pixels have nearly identical
     colour — a real volumetric render shows depth-modulated structure.

Usage:
    python ghost_cube_audit.py                  # all volumetric presets
    python ghost_cube_audit.py sandpile_3d      # single rule
    python ghost_cube_audit.py --save-images    # also write PNGs
    python ghost_cube_audit.py --steps 200      # custom warmup
"""

from __future__ import annotations
import argparse, os, sys, time
import numpy as np

os.environ.setdefault('CA_DISABLE_PRESET_OVERRIDES', '1')

from simulator import Simulator, RULE_PRESETS


# ── Background colour the renderer clears with (matches simulator.py) ──
BG_RGB = np.array([0.02, 0.02, 0.04])  # near-black blue


def render_frame(sim: Simulator, warmup_steps: int) -> np.ndarray:
    """Run the sim for warmup_steps, then render one frame and return
    the framebuffer as a (H, W, 4) uint8 array."""
    for _ in range(warmup_steps):
        sim._step_sim()
    # Force the renderer to produce a fresh frame.
    sim._render()
    sim.ctx.finish()
    # Voxel path renders into the default framebuffer; volumetric (compute
    # raymarch) path writes to _cr_output_tex. Pick the right source.
    if sim.renderer_mode == 'volumetric' and \
            hasattr(sim, '_cr_output_tex') and sim._cr_output_tex is not None:
        raw = sim._cr_output_tex.read()
        w, h = sim.width, sim.height
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
    raw = sim.ctx.screen.read(components=4, dtype='f1')
    w, h = sim.width, sim.height
    img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
    return img[::-1].copy()


def analyse_frame(img: np.ndarray) -> dict:
    """Quantify ghost-cube symptoms.

    Returns a dict of metrics. Decisive ones:
      foreground_frac   -- pixels that are not background
      interior_std      -- std-dev of luminance in foreground pixels
      saturation_frac   -- foreground pixels with luminance > 0.85
      edge_vs_interior  -- |mean(edge) - mean(interior)| / max(mean(interior), eps)
                           Low value = silhouette and interior look the same =
                           ghost-cube. High = real depth structure.
      verdict           -- 'OK' or 'GHOST_CUBE' or 'EMPTY'
    """
    h, w, _ = img.shape
    rgb = img[:, :, :3].astype(np.float32) / 255.0
    lum = rgb.mean(axis=-1)

    # Background mask: pixels close to BG_RGB.
    bg_dist = np.linalg.norm(rgb - BG_RGB, axis=-1)
    fg_mask = bg_dist > 0.02
    fg_frac = float(fg_mask.mean())

    out = {
        'shape': (h, w),
        'foreground_frac': fg_frac,
        'interior_std': 0.0,
        'saturation_frac': 0.0,
        'edge_vs_interior': 0.0,
        'mean_lum_fg': 0.0,
        'verdict': 'EMPTY',
    }
    if fg_frac < 0.001:
        return out

    fg_lum = lum[fg_mask]
    fg_rgb = rgb[fg_mask]
    out['mean_lum_fg'] = float(fg_lum.mean())
    out['interior_std'] = float(fg_lum.std())
    out['saturation_frac'] = float((fg_lum > 0.85).mean())

    # Per-channel std (a uniform-coloured cube has tiny std on every channel).
    out['rgb_std'] = tuple(float(s) for s in fg_rgb.std(axis=0))

    # Edge vs interior. Erode the foreground mask by a few pixels; the
    # difference is the "edge" band. Compare mean colour of that band to
    # the mean colour of the eroded interior.
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(fg_mask, iterations=4)
    edge_band = fg_mask & ~eroded
    if eroded.sum() > 100 and edge_band.sum() > 100:
        edge_mean = rgb[edge_band].mean(axis=0)
        interior_mean = rgb[eroded].mean(axis=0)
        diff = np.linalg.norm(edge_mean - interior_mean)
        denom = max(np.linalg.norm(interior_mean), 1e-3)
        out['edge_vs_interior'] = float(diff / denom)
        out['edge_mean_rgb'] = tuple(float(x) for x in edge_mean)
        out['interior_mean_rgb'] = tuple(float(x) for x in interior_mean)
    else:
        out['edge_vs_interior'] = -1.0  # not measurable

    # Verdict heuristic. The genuine "ghost cube" symptom is that the
    # foreground region tightly fills its own axis-aligned bounding box
    # — i.e. you can see the projected box silhouette as a solid blob,
    # not as something with internal structure (holes, gaps, thin
    # filaments). We measure SOLIDITY = fg_pixels / bbox_area. A real
    # 3D structure (clusters, tubes, walls) projects with lots of holes
    # so solidity stays well below 1. A filled cube saturates near 1
    # regardless of whether the surface is uniformly coloured or
    # speckled with multiple colours.
    ys, xs = np.where(fg_mask)
    if len(xs) > 100:
        bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        out['bbox_area_frac'] = bbox_area / (h * w)
        out['solidity'] = float(fg_mask.sum()) / max(bbox_area, 1.0)
    else:
        out['bbox_area_frac'] = 0.0
        out['solidity'] = 0.0

    is_ghost_uniform = (
        fg_frac > 0.20 and
        out['interior_std'] < 0.05 and
        out['edge_vs_interior'] >= 0 and out['edge_vs_interior'] < 0.10
    )
    is_saturated_uniform = (
        fg_frac > 0.15 and
        out['saturation_frac'] > 0.6 and
        out['interior_std'] < 0.20
    )
    # Solid-projection: foreground covers a large fraction of its own
    # bounding box. A fully-filled cube viewed from a corner projects as
    # a hexagon with bbox-solidity ≈ 0.75; real patterns with internal
    # structure rarely exceed 0.5. Combined with a foreground big enough
    # to plausibly be the simulation cube (>= 5% of frame).
    # Empirically, true ghost cubes (every cell rendered) hit
    # foreground_frac >= 0.30 because the projected hex covers ~30-40%
    # of the frame at default camera distance. Real organic clusters
    # (DLA, fire, mycelium, GoL) sit at 5-25% foreground regardless of
    # how solid their bbox-projection looks. Combining the two filters
    # eliminates that class of false positive.
    is_solid_projection = (
        out['solidity'] > 0.62 and
        out['bbox_area_frac'] > 0.10 and
        out['foreground_frac'] > 0.25
    )
    if is_ghost_uniform or is_saturated_uniform or is_solid_projection:
        out['verdict'] = 'GHOST_CUBE'
    else:
        out['verdict'] = 'OK'
    return out


def diagnose_view_tex(sim: Simulator) -> dict:
    """Read the view tex and report what's actually being fed to the
    raymarcher. Confirms whether vis_range remap landed.

    Caller is responsible for ensuring the view tex is current (typically
    by calling sim._render() first, which builds the accel textures)."""
    sim.ctx.finish()
    if not hasattr(sim, '_view_tex') or sim._view_tex is None:
        return {'view_tex': 'absent'}
    sz = sim.size
    raw = sim._view_tex.read()
    arr = np.frombuffer(raw, dtype=np.float16).reshape(sz, sz, sz).astype(np.float32)
    return {
        'view_tex_min': float(arr.min()),
        'view_tex_max': float(arr.max()),
        'view_tex_mean': float(arr.mean()),
        'view_frac_above_0.01': float((arr > 0.01).mean()),
        'view_frac_above_0.5': float((arr > 0.5).mean()),
        'view_frac_at_baseline': float((arr < 0.01).mean()),
    }


def audit_rule(rule: str, size: int, warmup: int, save_dir: str | None,
               renderer: str = 'volumetric'):
    label = RULE_PRESETS[rule].get('label', rule)
    vis_range = RULE_PRESETS[rule].get('vis_range', (0.0, 1.0))
    print(f'\n── {rule}  ({label})  [{renderer}]')
    print(f'   vis_range = {vis_range}')

    t0 = time.time()
    try:
        sim = Simulator(size=size, rule=rule, headless=True)
        sim.renderer_mode = renderer
    except Exception as e:  # noqa: BLE001  sim construction may fail on bad params
        print(f'   BUILD ERROR: {type(e).__name__}: {e}')
        return None
    try:
        img = render_frame(sim, warmup)
        view = diagnose_view_tex(sim) if renderer == 'volumetric' else {
            'view_tex_min': float('nan'), 'view_tex_max': float('nan'),
            'view_tex_mean': float('nan'), 'view_frac_above_0.01': float('nan'),
            'view_frac_above_0.5': float('nan'), 'view_frac_at_baseline': float('nan'),
        }
        metrics = analyse_frame(img)
        metrics.update(view)
        metrics['rule'] = rule
        metrics['renderer'] = renderer
        metrics['build_render_s'] = round(time.time() - t0, 2)
        print(f'   foreground={metrics["foreground_frac"]:.3f}  '
              f'solidity={metrics.get("solidity", 0):.3f}  '
              f'interior_std={metrics["interior_std"]:.3f}  '
              f'sat={metrics["saturation_frac"]:.3f}')
        if renderer == 'volumetric':
            print(f'   view_tex   range=[{view["view_tex_min"]:.3f}, {view["view_tex_max"]:.3f}]  '
                  f'mean={view["view_tex_mean"]:.3f}  baseline_frac={view["view_frac_at_baseline"]:.2f}')
        else:
            print(f'   voxel_threshold={sim.voxel_threshold:.4f}')
        print(f'   VERDICT: {metrics["verdict"]}')
        if save_dir:
            try:
                from PIL import Image
                os.makedirs(save_dir, exist_ok=True)
                tag = metrics['verdict']
                path = os.path.join(save_dir, f'{tag}_{renderer}_{rule}.png')
                Image.fromarray(img[:, :, :3]).save(path)
                print(f'   -> saved {path}')
            except Exception as e:  # noqa: BLE001  best-effort mkdir
                print(f'   image save failed: {e}')
        return metrics
    finally:
        try:
            sim.close()
        except Exception:  # noqa: BLE001  teardown, never fatal
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rules', nargs='*',
                    help='Specific rules to audit (default: all volumetric)')
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--steps', type=int, default=120,
                    help='Warmup steps before rendering')
    ap.add_argument('--save-images', action='store_true')
    ap.add_argument('--out-dir', default='ghost_cube_renders')
    ap.add_argument('--renderer', choices=['volumetric', 'voxel', 'both'],
                    default='volumetric',
                    help='Which renderer to test (default: volumetric)')
    args = ap.parse_args()

    if args.rules:
        rules = args.rules
    else:
        # In voxel/both mode, audit BOTH volumetric- and voxel-default
        # presets — many voxel-default rules also fill the cube.
        if args.renderer == 'volumetric':
            rules = [n for n, p in RULE_PRESETS.items()
                     if p.get('render_mode') == 'volumetric']
        else:
            rules = [n for n, p in RULE_PRESETS.items()
                     if p.get('render_mode') in ('volumetric', 'voxel')]
    save_dir = args.out_dir if args.save_images else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    renderers = ['volumetric', 'voxel'] if args.renderer == 'both' else [args.renderer]
    results = []
    for renderer in renderers:
        for rule in rules:
            if rule not in RULE_PRESETS:
                print(f'\n── {rule}: not in RULE_PRESETS')
                continue
            m = audit_rule(rule, args.size, args.steps, save_dir, renderer=renderer)
            if m is not None:
                results.append(m)

    print('\n' + '═' * 78)
    print(' SUMMARY')
    print('═' * 78)
    ghost = [r for r in results if r['verdict'] == 'GHOST_CUBE']
    empty = [r for r in results if r['verdict'] == 'EMPTY']
    ok    = [r for r in results if r['verdict'] == 'OK']
    print(f' OK         : {len(ok)}')
    print(f' EMPTY      : {len(empty)}  -> {[r["rule"] for r in empty]}')
    print(f' GHOST_CUBE : {len(ghost)}')
    for r in ghost:
        print(f'   - {r["rule"]:30s}  [{r["renderer"]:10s}]  '
              f'fg={r["foreground_frac"]:.2f}  solidity={r.get("solidity",0):.2f}  '
              f'sat={r["saturation_frac"]:.2f}  std={r["interior_std"]:.3f}')
    return 0 if not ghost else 1


if __name__ == '__main__':
    sys.exit(main())
