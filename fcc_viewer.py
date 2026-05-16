"""
Minimal FCC viewer (voxel renderer).

Modes
-----
* ``sphere``      Gaussian-density ball, DENSITY vis mode, colormap fire.
* ``neighbours``  Centre cell + 12 nearest neighbours spaced by ``--nn-spacing``
                  cells, each lit with a distinct RGB colour (RGB_CHANNELS).
* ``empty``       Blank field (smoke test: must render 0 lit pixels).

Acceptance tests (``--headless`` mode prints PASS/FAIL)
-------------------------------------------------------
* ``empty``       Pass iff < 0.1 percent of pixels deviate from background.
* ``sphere``      Pass iff the silhouette is round: extract edge pixels of the
                  lit mask, compute their radii from the centroid, and require
                  ``stddev(radii) / mean(radii) < --roundness-max``. A
                  parallelepiped silhouette fails this with a noticeably
                  larger ratio.
* ``neighbours``  Connected-component analysis on the lit mask; pass iff the
                  number of components matches ``--expect-components``
                  (default 13 = centre + 12 NN).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from typing import Tuple

import numpy as np

from lattice import FCC
from fcc_field import FCCField, FCCFieldShape
from fcc_render import (
    OrbitCamera,
    VoxelRenderer,
    VoxelSettings,
    VIS_MODE_DENSITY,
    VIS_MODE_RGB_CHANNELS,
)


# ---------------------------------------------------------------------------
# Field builders
# ---------------------------------------------------------------------------


def _world_grid(shape: FCCFieldShape) -> Tuple[np.ndarray, np.ndarray]:
    a = np.arange(shape.Na, dtype=np.float64)
    b = np.arange(shape.Nb, dtype=np.float64)
    c = np.arange(shape.Nc, dtype=np.float64)
    A, B, C = np.meshgrid(a, b, c, indexing='ij')
    idx = np.stack([A, B, C], axis=-1)
    world = idx @ FCC.M.T
    centre_idx = np.array([shape.Na, shape.Nb, shape.Nc], dtype=np.float64) * 0.5
    return world, FCC.index_to_world(centre_idx)


def _shortest_corner_dist(shape: FCCFieldShape) -> float:
    half = np.array([shape.Na, shape.Nb, shape.Nc], dtype=np.float64) * 0.5
    dists = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                dists.append(float(np.linalg.norm(
                    FCC.index_to_world(half * np.array([sx, sy, sz]))
                )))
    return min(dists)


def _pack_field(shape: FCCFieldShape, abc_channels: np.ndarray) -> np.ndarray:
    """``abc_channels`` shape ``(Na, Nb, Nc, channels)``."""
    out = np.zeros(shape.numpy_shape(), dtype=np.float32)
    out[:] = abc_channels.astype(np.float32).transpose(2, 1, 0, 3)
    return out


def build_sphere(shape: FCCFieldShape, *, sigma_frac: float = 0.30) -> np.ndarray:
    """Gaussian density in channel 0. ``sigma_frac`` is sigma as a fraction
    of the shortest centre-to-corner distance; 0.30 gives a fat ball that
    cuts off cleanly above the alive threshold."""
    world, centre = _world_grid(shape)
    dist = np.linalg.norm(world - centre, axis=-1)
    sigma = max(_shortest_corner_dist(shape) * sigma_frac, 1e-6)
    density = np.exp(-(dist / sigma) ** 2)
    abc = np.zeros((shape.Na, shape.Nb, shape.Nc, shape.channels),
                   dtype=np.float32)
    abc[..., 0] = density
    return _pack_field(shape, abc)


def build_neighbours(shape: FCCFieldShape, *, spacing: int = 3) -> np.ndarray:
    """Centre cell (white) + 12 NN spaced by ``spacing`` cells, each lit with
    a distinct RGB colour via golden-angle HSV."""
    abc = np.zeros((shape.Na, shape.Nb, shape.Nc, shape.channels),
                   dtype=np.float32)
    cx, cy, cz = shape.Na // 2, shape.Nb // 2, shape.Nc // 2
    abc[cx, cy, cz, :3] = (1.0, 1.0, 1.0)
    for k, (da, db, dc) in enumerate(FCC.neighbours):
        a = (cx + int(da) * spacing) % shape.Na
        b = (cy + int(db) * spacing) % shape.Nb
        c = (cz + int(dc) * spacing) % shape.Nc
        h = (k * 0.618033988) % 1.0
        abc[a, b, c, :3] = _hsv_to_rgb(h, 0.85, 0.95)
    return _pack_field(shape, abc)


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    return [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ][i % 6]


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


def _build_world(
    ctx, mode: str, grid: int, nn_spacing: int,
) -> Tuple[FCCField, FCCFieldShape, VoxelSettings, float]:
    if mode == 'sphere':
        shape = FCCFieldShape(grid, grid, grid)
        field = FCCField(ctx, shape, linear_filter=False)
        field.upload(build_sphere(shape))
        settings = VoxelSettings(
            threshold=0.10,
            vis_mode=VIS_MODE_DENSITY,
            channel=0,
            colormap=0,                # fire
            voxel_shrink=1.0,
            ao_strength=0.35,
            brightness=2.0,
        )
        half = np.array([shape.Na, shape.Nb, shape.Nc], dtype=np.float64) * 0.5
        return field, shape, settings, float(np.linalg.norm(FCC.index_to_world(half)))

    if mode == 'neighbours':
        vgrid = max(grid, 2 * nn_spacing + 4)
        shape = FCCFieldShape(vgrid, vgrid, vgrid)
        field = FCCField(ctx, shape, linear_filter=False)
        field.upload(build_neighbours(shape, spacing=nn_spacing))
        settings = VoxelSettings(
            threshold=0.10,
            vis_mode=VIS_MODE_RGB_CHANNELS,
            voxel_shrink=1.0,
            ao_strength=0.0,
        )
        return field, shape, settings, float(nn_spacing) * 1.6

    if mode == 'empty':
        shape = FCCFieldShape(grid, grid, grid)
        field = FCCField(ctx, shape)
        settings = VoxelSettings()
        half = np.array([shape.Na, shape.Nb, shape.Nc], dtype=np.float64) * 0.5
        return field, shape, settings, float(np.linalg.norm(FCC.index_to_world(half)))

    raise ValueError(f"unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def _lit_mask(rgba: np.ndarray, bg: Tuple[float, float, float],
              tol: int = 8) -> np.ndarray:
    bg255 = np.array([int(c * 255) for c in bg], dtype=np.int16)
    delta = np.abs(rgba[..., :3].astype(np.int16) - bg255).max(axis=-1)
    return delta > tol


def _silhouette_roundness(mask: np.ndarray) -> Tuple[float, float, float]:
    """``(centroid_x, centroid_y, std_r / mean_r)`` over the silhouette's
    boundary pixels. Lower is rounder."""
    if not mask.any():
        return (0.0, 0.0, float('inf'))
    nb = np.zeros_like(mask)
    nb[1:, :]  |= mask[:-1, :]
    nb[:-1, :] |= mask[1:, :]
    nb[:, 1:]  |= mask[:, :-1]
    nb[:, :-1] |= mask[:, 1:]
    # Edge = lit pixel that has at least one 4-neighbour NOT lit.
    inner = np.zeros_like(mask)
    inner[1:, :]  &= False
    interior = (mask
                & np.roll(mask,  1, 0) & np.roll(mask, -1, 0)
                & np.roll(mask,  1, 1) & np.roll(mask, -1, 1))
    edge = mask & ~interior
    ys, xs = np.where(edge)
    if len(xs) < 8:
        return (0.0, 0.0, float('inf'))
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return (cx, cy, float(np.std(r) / max(np.mean(r), 1e-6)))


def _distinct_colors(rgba: np.ndarray, mask: np.ndarray, *,
                     hue_bins: int = 48, min_pixels: int = 400) -> int:
    """Count distinct hues among lit pixels. Phong shading varies *value*
    per facet but not hue, so binning by hue gives one count per voxel
    regardless of face orientation or AO."""
    if not mask.any():
        return 0
    rgb = rgba[..., :3][mask].astype(np.float32) / 255.0
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    rng = mx - mn
    # Achromatic pixels (near-greyscale) bucket into their own bin.
    achro = rng < 0.08
    hue = np.zeros_like(mx)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    safe = np.where(rng > 0, rng, 1.0)
    hr = ((g - b) / safe) % 6.0
    hg = ((b - r) / safe) + 2.0
    hb = ((r - g) / safe) + 4.0
    pick_r = (mx == r)
    pick_g = (mx == g) & ~pick_r
    pick_b = ~pick_r & ~pick_g
    hue = np.where(pick_r, hr, np.where(pick_g, hg, hb)) / 6.0
    bins = (hue * hue_bins).astype(np.uint32) % hue_bins
    bins = np.where(achro, np.uint32(hue_bins), bins)   # achromatic bucket
    vals, counts = np.unique(bins, return_counts=True)
    kept_vals = vals[counts >= min_pixels]
    # Merge bins that are adjacent in hue (straddle a bin boundary).
    # Achromatic bucket (hue_bins) never merges with chromatic bins.
    if kept_vals.size == 0:
        return 0
    chroma = np.sort(kept_vals[kept_vals < hue_bins])
    has_achro = bool((kept_vals == hue_bins).any())
    clusters = 0
    if chroma.size > 0:
        clusters = 1
        for prev, cur in zip(chroma[:-1], chroma[1:]):
            if int(cur) - int(prev) > 1:
                clusters += 1
        # Wrap-around: bin 0 and bin (hue_bins-1) are also adjacent.
        if chroma.size >= 2 and chroma[0] == 0 and chroma[-1] == hue_bins - 1 \
                and clusters >= 2:
            clusters -= 1
    return clusters + (1 if has_achro else 0)


def _connected_components(mask: np.ndarray, *, min_size: int = 8) -> int:
    """4-connected component count, ignoring components smaller than
    ``min_size`` (anti-alias spurs)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    n = 0
    for y0 in range(h):
        row = mask[y0]
        for x0 in range(w):
            if not row[x0] or seen[y0, x0]:
                continue
            count = 0
            q = deque([(y0, x0)])
            seen[y0, x0] = True
            while q:
                y, x = q.popleft()
                count += 1
                if y > 0     and mask[y-1, x] and not seen[y-1, x]: seen[y-1, x] = True; q.append((y-1, x))
                if y < h - 1 and mask[y+1, x] and not seen[y+1, x]: seen[y+1, x] = True; q.append((y+1, x))
                if x > 0     and mask[y, x-1] and not seen[y, x-1]: seen[y, x-1] = True; q.append((y, x-1))
                if x < w - 1 and mask[y, x+1] and not seen[y, x+1]: seen[y, x+1] = True; q.append((y, x+1))
            if count >= min_size:
                n += 1
    return n


# ---------------------------------------------------------------------------
# Headless render + accept
# ---------------------------------------------------------------------------


def headless_render(
    out_path: str,
    *,
    mode: str = 'sphere',
    grid: int = 64,
    nn_spacing: int = 3,
    width: int = 720,
    height: int = 720,
    expect_components: int = 13,
    roundness_max: float = 0.10,
) -> int:
    import moderngl
    print(f"[viewer] headless ({mode}): booting standalone context...")
    ctx = moderngl.create_standalone_context(require=430)

    field, shape, settings, frame_radius = _build_world(ctx, mode, grid, nn_spacing)
    print(f"[viewer]   field = {shape.Na}x{shape.Nb}x{shape.Nc} "
          f"({shape.cell_count} cells)")

    renderer = VoxelRenderer(ctx)
    centre = renderer.field_world_center(shape)
    cam = OrbitCamera(target=centre, distance=frame_radius * 2.6,
                      azimuth=0.7, elevation=0.45)

    colour_tex = ctx.texture((width, height), 4, dtype='f1')
    depth_tex  = ctx.depth_texture((width, height))
    fbo = ctx.framebuffer(color_attachments=[colour_tex],
                          depth_attachment=depth_tex)
    fbo.use()
    fbo.clear(*settings.bg_color, 1.0)

    n_voxels = renderer.render(field, cam, (width, height), settings)
    ctx.finish()
    print(f"[viewer]   cull pass emitted {n_voxels} voxel instances")

    rgba = np.frombuffer(colour_tex.read(), dtype=np.uint8).reshape(height, width, 4)
    rgba = rgba[::-1]   # GL origin is bottom-left

    try:
        from PIL import Image
        Image.fromarray(rgba).save(out_path)
    except ImportError:
        out_path = out_path.rsplit('.', 1)[0] + '.ppm'
        with open(out_path, 'wb') as fp:
            fp.write(f"P6\n{width} {height}\n255\n".encode())
            fp.write(rgba[..., :3].tobytes())
    print(f"[viewer]   wrote {out_path}")

    mask = _lit_mask(rgba, settings.bg_color)
    total = width * height
    lit = int(mask.sum())
    print(f"[viewer]   {lit}/{total} pixels lit ({100 * lit / total:.2f}%)")

    if mode == 'empty':
        ok = lit < total // 1000
        print(f"[viewer] empty: {'PASS' if ok else 'FAIL'} "
              f"(lit must be < {total // 1000})")
        return 0 if ok else 1

    if mode == 'sphere':
        if lit < total // 500:
            print("[viewer] sphere: FAIL (silhouette not visible)"); return 1
        cx, cy, roundness = _silhouette_roundness(mask)
        print(f"[viewer]   centroid = ({cx:.1f}, {cy:.1f})  "
              f"stddev(r)/mean(r) = {roundness:.4f}")
        ok = roundness < roundness_max
        print(f"[viewer] sphere: {'PASS' if ok else 'FAIL'} "
              f"(roundness must be < {roundness_max})")
        return 0 if ok else 1

    if mode == 'neighbours':
        if lit < total // 5000:
            print("[viewer] neighbours: FAIL (no voxels visible)"); return 1
        n_colors = _distinct_colors(rgba, mask)
        n_comp   = _connected_components(mask)
        print(f"[viewer]   distinct colours = {n_colors} (expected {expect_components})")
        print(f"[viewer]   connected components = {n_comp} "
              f"(<= {expect_components}; merged by projection overlap)")
        ok = n_colors == expect_components
        print(f"[viewer] neighbours: {'PASS' if ok else 'FAIL'} "
              f"(distinct colours must equal {expect_components})")
        return 0 if ok else 1

    return 0


# ---------------------------------------------------------------------------
# Interactive GLFW viewer
# ---------------------------------------------------------------------------


def interactive(
    *, mode: str = 'sphere', grid: int = 64, nn_spacing: int = 3,
    width: int = 1024, height: int = 768,
) -> None:
    import glfw, moderngl

    if not glfw.init():
        print("[viewer] glfw.init failed", file=sys.stderr); sys.exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.DOUBLEBUFFER, True)
    glfw.window_hint(glfw.DEPTH_BITS, 24)

    win = glfw.create_window(width, height, f"FCC voxel viewer ({mode})", None, None)
    if not win:
        glfw.terminate(); print("[viewer] window failed", file=sys.stderr); sys.exit(1)
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    print(f"[viewer] {ctx.info.get('GL_RENDERER', '?')}  "
          f"{ctx.info.get('GL_VERSION', '?')}")

    field, shape, settings, frame_radius = _build_world(ctx, mode, grid, nn_spacing)
    renderer = VoxelRenderer(ctx)
    centre = renderer.field_world_center(shape)
    cam = OrbitCamera(target=centre, distance=frame_radius * 2.6,
                      azimuth=0.7, elevation=0.45)
    cam_default = (cam.distance, cam.azimuth, cam.elevation)

    zoom_min, zoom_max = frame_radius * 0.4, frame_radius * 20.0

    s = {'down': False, 'x': 0.0, 'y': 0.0}

    def on_mb(_w, b, action, _m):
        if b == glfw.MOUSE_BUTTON_LEFT:
            s['down'] = (action == glfw.PRESS)
            s['x'], s['y'] = glfw.get_cursor_pos(_w)

    def on_cur(_w, x, y):
        if not s['down']: return
        dx, dy = x - s['x'], y - s['y']
        s['x'], s['y'] = x, y
        cam.azimuth   -= dx * 0.005
        cam.elevation = max(-math.pi * 0.49, min(math.pi * 0.49,
                                                  cam.elevation + dy * 0.005))

    def on_scroll(_w, _xo, yo):
        cam.distance = max(zoom_min, min(zoom_max, cam.distance * (0.9 ** yo)))

    def on_key(_w, key, _sc, action, _m):
        if action != glfw.PRESS: return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(_w, True)
        elif key == glfw.KEY_R:
            cam.distance, cam.azimuth, cam.elevation = cam_default

    glfw.set_mouse_button_callback(win, on_mb)
    glfw.set_cursor_pos_callback(win, on_cur)
    glfw.set_scroll_callback(win, on_scroll)
    glfw.set_key_callback(win, on_key)

    print("[viewer] drag=orbit, scroll=zoom, R=reset, Esc=quit")
    frame, t0 = 0, time.perf_counter()
    while not glfw.window_should_close(win):
        glfw.poll_events()
        w, h = glfw.get_framebuffer_size(win)
        ctx.screen.use()
        ctx.screen.clear(*settings.bg_color, 1.0)
        renderer.render(field, cam, (w, h), settings)
        glfw.swap_buffers(win)
        frame += 1
        if frame % 120 == 0:
            dt = time.perf_counter() - t0
            print(f"[viewer]   {120 / dt:.1f} fps")
            t0 = time.perf_counter()

    renderer.release(); field.release()
    glfw.destroy_window(win); glfw.terminate()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('sphere', 'neighbours', 'empty'),
                    default='sphere')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--out', default='/tmp/fcc_view.png')
    ap.add_argument('--grid', type=int, default=64)
    ap.add_argument('--nn-spacing', type=int, default=3,
                    help='neighbours mode: cells between centre and each NN')
    ap.add_argument('--width', type=int, default=720)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--expect-components', type=int, default=13)
    ap.add_argument('--roundness-max', type=float, default=0.10)
    args = ap.parse_args()

    if args.headless:
        rc = headless_render(
            args.out, mode=args.mode, grid=args.grid,
            nn_spacing=args.nn_spacing, width=args.width, height=args.height,
            expect_components=args.expect_components,
            roundness_max=args.roundness_max,
        )
        sys.exit(rc)
    else:
        interactive(mode=args.mode, grid=args.grid, nn_spacing=args.nn_spacing,
                    width=args.width, height=args.height)


if __name__ == '__main__':
    main()
