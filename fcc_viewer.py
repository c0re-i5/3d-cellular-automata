"""
Minimal FCC viewer.

Two modes:

* **Interactive** (default): opens a GLFW window with an orbit camera.
  Drag left mouse to orbit, scroll to zoom, ``R`` to reset, ``Esc`` to quit.

* **Headless** (``--headless``): renders one frame to a PNG without opening
  a window. Used as the Phase A3 smoke test.

In both modes the field is a hand-built **world-space sphere** rasterised
onto the FCC lattice: density 1.0 inside the sphere, smooth falloff at the
edge. If the raymarcher, camera, lattice transform, and ping-pong texture
all agree, the result is a clean smooth ball.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Tuple

import numpy as np

from lattice import FCC
from fcc_field import FCCField, FCCFieldShape
from fcc_render import OrbitCamera, Raymarcher, RaymarchSettings


# ---------------------------------------------------------------------------
# Test field: world-space sphere rasterised onto the FCC lattice
# ---------------------------------------------------------------------------


def make_sphere_field(
    shape: FCCFieldShape,
    *,
    radius_frac: float = 0.35,
    edge_frac: float = 0.06,
) -> np.ndarray:
    """Build a numpy field of FCC cell values: 1.0 inside a centred sphere,
    smooth falloff over ``edge_frac * world_diameter`` at the boundary.

    Returned shape matches :meth:`FCCFieldShape.numpy_shape`.
    """
    Na, Nb, Nc, ch = shape.numpy_shape()
    # Index-space coordinate grids (a, b, c) per cell.
    a = np.arange(shape.Na, dtype=np.float64)
    b = np.arange(shape.Nb, dtype=np.float64)
    c = np.arange(shape.Nc, dtype=np.float64)
    A, B, C = np.meshgrid(a, b, c, indexing='ij')      # shape (Na, Nb, Nc)
    idx = np.stack([A, B, C], axis=-1)                  # (Na, Nb, Nc, 3)

    # Convert to world space cell-by-cell. M is (3, 3); idx is (..., 3).
    world = idx @ FCC.M.T                               # (Na, Nb, Nc, 3)

    # Centre of the field in world space.
    centre_idx = np.array(
        [shape.Na, shape.Nb, shape.Nc], dtype=np.float64,
    ) * 0.5
    centre_world = FCC.index_to_world(centre_idx)

    # World-space distance to centre.
    dist = np.linalg.norm(world - centre_world, axis=-1)

    # Bounding-sphere radius (largest world distance from centre to a corner).
    half = centre_idx
    corners_world = np.array(
        [FCC.index_to_world(half * np.array(s)) for s in
         [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
          ( 1, -1, -1), ( 1, -1, 1), ( 1, 1, -1), ( 1, 1, 1)]],
    )
    r_max = float(np.linalg.norm(corners_world, axis=-1).max())

    r_in  = r_max * radius_frac
    r_out = r_in + r_max * edge_frac

    # Smooth shell: 1.0 inside r_in, 0.0 outside r_out, smoothstep between.
    t = np.clip((r_out - dist) / max(r_out - r_in, 1e-9), 0.0, 1.0)
    density = t * t * (3.0 - 2.0 * t)                   # smoothstep
    density = density.astype(np.float32)

    # Write density into channel 0; leave the rest zero. numpy layout is
    # (Nc, Nb, Na, ch), so transpose accordingly.
    field = np.zeros(shape.numpy_shape(), dtype=np.float32)
    # density is (Na, Nb, Nc); we need (Nc, Nb, Na).
    field[..., 0] = density.transpose(2, 1, 0)
    return field


# ---------------------------------------------------------------------------
# Headless one-shot render
# ---------------------------------------------------------------------------


def headless_render(
    out_path: str,
    *,
    grid: int = 64,
    width: int = 720,
    height: int = 720,
) -> None:
    import moderngl
    print(f"[viewer] headless: booting standalone context...")
    ctx = moderngl.create_standalone_context(require=430)

    shape = FCCFieldShape(grid, grid, grid, channels=4)
    field = FCCField(ctx, shape)
    print(f"[viewer]   building world-space sphere ({grid}^3 cells)...")
    field.upload(make_sphere_field(shape))

    rm = Raymarcher(ctx)

    # Frame the field: place camera at distance ~ 2.5 * bounding-sphere radius,
    # aimed at the field centre.
    centre = rm.field_world_center(shape)
    radius = rm.field_world_radius(shape)
    cam = OrbitCamera(
        target=centre,
        distance=radius * 2.5,
        azimuth=0.7,
        elevation=0.45,
    )

    # Render-to-texture: colour attachment, no depth (raymarcher writes 1.0 alpha).
    colour_tex = ctx.texture((width, height), 4, dtype='f1')
    fbo = ctx.framebuffer(color_attachments=[colour_tex])
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)

    rm.render(field, cam, (width, height))
    ctx.finish()

    rgba = np.frombuffer(colour_tex.read(), dtype=np.uint8).reshape(height, width, 4)
    # OpenGL framebuffer has origin at bottom-left; flip for image convention.
    rgba = rgba[::-1]

    try:
        from PIL import Image
    except ImportError:
        # Fall back to bare PPM if Pillow is missing.
        ppm_path = out_path.rsplit('.', 1)[0] + '.ppm'
        with open(ppm_path, 'wb') as fp:
            fp.write(f"P6\n{width} {height}\n255\n".encode())
            fp.write(rgba[..., :3].tobytes())
        print(f"[viewer]   wrote {ppm_path} (Pillow not installed)")
        return

    Image.fromarray(rgba).save(out_path)
    print(f"[viewer]   wrote {out_path}")

    nonzero_pixels = int(np.count_nonzero(rgba[..., :3].sum(axis=-1)))
    total = width * height
    # A pixel is "lit" if any channel is well above the bg colour (~10/255).
    lit = int(np.count_nonzero(rgba[..., :3].max(axis=-1) > 32))
    print(f"[viewer]   {nonzero_pixels}/{total} non-zero pixels "
          f"({100 * nonzero_pixels / total:.1f}%), "
          f"{lit} lit ({100 * lit / total:.1f}%)")
    if lit < total // 200:
        print("[viewer] WARNING: very few lit pixels; sphere may not be visible")
    else:
        print("[viewer] sphere visible: OK")


# ---------------------------------------------------------------------------
# Interactive GLFW viewer
# ---------------------------------------------------------------------------


def interactive(grid: int = 64, width: int = 1024, height: int = 768) -> None:
    import glfw
    import moderngl

    if not glfw.init():
        print("[viewer] glfw.init failed", file=sys.stderr)
        sys.exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.DOUBLEBUFFER, True)

    window = glfw.create_window(width, height, "FCC viewer (Phase A3)", None, None)
    if not window:
        glfw.terminate()
        print("[viewer] window creation failed", file=sys.stderr)
        sys.exit(1)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    print(f"[viewer] {ctx.info.get('GL_RENDERER', '?')}  "
          f"{ctx.info.get('GL_VERSION', '?')}")

    shape = FCCFieldShape(grid, grid, grid, channels=4)
    field = FCCField(ctx, shape)
    field.upload(make_sphere_field(shape))
    rm = Raymarcher(ctx)

    centre = rm.field_world_center(shape)
    radius = rm.field_world_radius(shape)
    cam = OrbitCamera(
        target=centre,
        distance=radius * 2.5,
        azimuth=0.7,
        elevation=0.45,
    )
    cam_default = (cam.distance, cam.azimuth, cam.elevation)

    # ---- Input state ----
    state = {
        'mouse_down': False,
        'last_x':     0.0,
        'last_y':     0.0,
    }

    def on_mouse_button(_w, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            state['mouse_down'] = (action == glfw.PRESS)
            state['last_x'], state['last_y'] = glfw.get_cursor_pos(_w)

    def on_cursor(_w, x, y):
        if not state['mouse_down']:
            return
        dx = x - state['last_x']
        dy = y - state['last_y']
        state['last_x'], state['last_y'] = x, y
        cam.azimuth   -= dx * 0.005
        cam.elevation += dy * 0.005
        cam.elevation = max(-math.pi * 0.49, min(math.pi * 0.49, cam.elevation))

    def on_scroll(_w, _xoff, yoff):
        cam.distance *= (0.9 ** yoff)
        cam.distance = max(radius * 1.05, min(radius * 20.0, cam.distance))

    def on_key(_w, key, _sc, action, _mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(_w, True)
        elif key == glfw.KEY_R:
            cam.distance, cam.azimuth, cam.elevation = cam_default

    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    print("[viewer] drag to orbit, scroll to zoom, R to reset, Esc to quit")

    frame = 0
    t0 = time.perf_counter()
    while not glfw.window_should_close(window):
        glfw.poll_events()
        w, h = glfw.get_framebuffer_size(window)
        ctx.screen.use()
        ctx.screen.clear(0.04, 0.04, 0.06, 1.0)
        rm.render(field, cam, (w, h))
        glfw.swap_buffers(window)
        frame += 1
        if frame % 120 == 0:
            dt = time.perf_counter() - t0
            print(f"[viewer]   {120 / dt:.1f} fps")
            t0 = time.perf_counter()

    rm.release()
    field.release()
    glfw.destroy_window(window)
    glfw.terminate()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--headless', action='store_true',
                    help='render one frame to PNG and exit')
    ap.add_argument('--out', default='/tmp/fcc_sphere.png',
                    help='output path for --headless')
    ap.add_argument('--grid', type=int, default=64)
    ap.add_argument('--width', type=int, default=720)
    ap.add_argument('--height', type=int, default=720)
    args = ap.parse_args()

    if args.headless:
        headless_render(args.out, grid=args.grid,
                        width=args.width, height=args.height)
    else:
        interactive(grid=args.grid, width=args.width, height=args.height)


if __name__ == '__main__':
    main()
