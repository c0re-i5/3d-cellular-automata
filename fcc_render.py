"""
FCC raymarcher.

Renders an :class:`fcc_field.FCCField` by casting one ray per fragment, in
world space, transformed into index space for AABB intersection against the
field's bounding parallelepiped. Hardware trilinear sampling does the
interpolation between FCC cells. The lattice's primitive-cell matrix
``LATTICE_M_INV`` (injected via :func:`lattice.FCCSpec.glsl_header`) maps
world-space sample positions back to index space for the texture lookup.

This module owns no field, no camera, and no window. It is a pure renderer:
construct once with a context, then call :meth:`Raymarcher.render` per frame
with the field, the camera matrices, and any tunables you want to vary.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Tuple

import numpy as np
import moderngl

from lattice import FCC


# ---------------------------------------------------------------------------
# Camera helper (right-handed, OpenGL convention)
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenGL-style right-handed view matrix (column-major in result)."""
    f = _normalize(target - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def perspective(fovy_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(fovy_rad * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


@dataclass
class OrbitCamera:
    """Right-handed orbit camera around ``target``."""

    target: np.ndarray = dc_field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    distance: float = 5.0
    azimuth: float = 0.7              # radians, around world up (Y)
    elevation: float = 0.5            # radians, above XZ plane
    fovy_deg: float = 45.0
    near: float = 0.1
    far: float = 1000.0
    up: np.ndarray = dc_field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))

    def eye(self) -> np.ndarray:
        cx = np.cos(self.elevation) * np.sin(self.azimuth)
        cy = np.sin(self.elevation)
        cz = np.cos(self.elevation) * np.cos(self.azimuth)
        return self.target + self.distance * np.array([cx, cy, cz], dtype=np.float64)

    def view_matrix(self) -> np.ndarray:
        return look_at(self.eye(), self.target, self.up)

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return perspective(np.radians(self.fovy_deg), aspect, self.near, self.far)


# ---------------------------------------------------------------------------
# Shader sources
# ---------------------------------------------------------------------------

_VS = """\
#version 430
in vec2 in_pos;
out vec2 v_ndc;
void main() {
    v_ndc = in_pos;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FS_TEMPLATE = """\
#version 430
in  vec2 v_ndc;
out vec4 frag_color;

uniform sampler3D u_field;
uniform mat4  u_view_inv;
uniform mat4  u_proj_inv;
uniform vec3  u_cam_pos;
uniform vec3  u_field_size;     // (Na, Nb, Nc) as float
uniform float u_step_size;      // world units per ray step
uniform int   u_max_steps;
uniform vec4  u_channel_weights;
uniform vec3  u_bg_color;
uniform float u_density_gain;
uniform float u_alpha_max;
uniform vec3  u_warm;
uniform vec3  u_cool;

/*__LATTICE_HEADER__*/

// Ray-AABB intersection in index space (AABB = [0, u_field_size]).
// Returns vec2(t_enter, t_exit) as ray parameters in the input ray's units.
// Empty interval if t_exit <= t_enter.
vec2 intersect_field_aabb(vec3 ro_w, vec3 rd_w) {
    vec3 ro_i = LATTICE_M_INV * ro_w;
    vec3 rd_i = LATTICE_M_INV * rd_w;
    vec3 inv_d = 1.0 / rd_i;
    vec3 t1 = (vec3(0.0)        - ro_i) * inv_d;
    vec3 t2 = (u_field_size     - ro_i) * inv_d;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    float t_enter = max(max(tmin.x, tmin.y), tmin.z);
    float t_exit  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(max(t_enter, 0.0), t_exit);
}

void main() {
    // Reconstruct world-space ray from NDC.
    // Take a point on the near plane in clip space, run it through proj_inv
    // to get a view-space point, perspective-divide, then transform direction
    // to world space.
    vec4 near_clip = vec4(v_ndc, -1.0, 1.0);
    vec4 near_view = u_proj_inv * near_clip;
    near_view /= near_view.w;
    vec3 dir_view = normalize(near_view.xyz);     // view origin -> near-plane pixel
    vec3 rd_w     = normalize((u_view_inv * vec4(dir_view, 0.0)).xyz);
    vec3 ro_w     = u_cam_pos;

    vec2 ts = intersect_field_aabb(ro_w, rd_w);
    if (ts.y <= ts.x) {
        frag_color = vec4(u_bg_color, 1.0);
        return;
    }

    vec3  accum = vec3(0.0);
    float alpha = 0.0;
    int   n_steps = min(int((ts.y - ts.x) / u_step_size) + 1, u_max_steps);
    for (int i = 0; i < n_steps; ++i) {
        float t = ts.x + (float(i) + 0.5) * u_step_size;
        vec3 pos_w = ro_w + t * rd_w;
        vec3 idx   = LATTICE_M_INV * pos_w;
        // Texture coords in [0, 1]^3 mapped from index space [0, size].
        vec3 uvw = idx / u_field_size;
        vec4 samp = texture(u_field, uvw);
        float dens = clamp(dot(samp, u_channel_weights) * u_density_gain, 0.0, 1.0);

        // Front-to-back compositing.
        float a = (1.0 - alpha) * dens;
        vec3  col = mix(u_cool, u_warm, dens);
        accum += a * col;
        alpha += a;
        if (alpha >= u_alpha_max) break;
    }

    vec3 final_rgb = accum + (1.0 - alpha) * u_bg_color;
    frag_color = vec4(final_rgb, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Raymarcher
# ---------------------------------------------------------------------------


@dataclass
class RaymarchSettings:
    """Knobs passed every frame. Defaults are sane for first viewing."""

    step_size: float = 0.5                    # world units; nn_distance = 1.0
    max_steps: int = 512
    channel_weights: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    bg_color: Tuple[float, float, float] = (0.04, 0.04, 0.06)
    density_gain: float = 2.0
    alpha_max: float = 0.98
    warm: Tuple[float, float, float] = (1.00, 0.85, 0.55)   # high-density colour
    cool: Tuple[float, float, float] = (0.20, 0.40, 0.85)   # low-density colour


class Raymarcher:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        fs = _FS_TEMPLATE.replace('/*__LATTICE_HEADER__*/', FCC.glsl_header())
        self.prog = ctx.program(vertex_shader=_VS, fragment_shader=fs)
        # Fullscreen covering triangle (no diagonal seam).
        verts = np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype='f4')
        self.vbo = ctx.buffer(verts.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, '2f', 'in_pos')])

    # ------------------------------------------------------------------

    def field_world_center(self, field_shape) -> np.ndarray:
        """World-space centre of a field with index-space dims (Na, Nb, Nc)."""
        idx_centre = np.array(
            [field_shape.Na, field_shape.Nb, field_shape.Nc],
            dtype=np.float64,
        ) * 0.5
        return FCC.index_to_world(idx_centre)

    def field_world_radius(self, field_shape) -> float:
        """Bounding-sphere radius of the world-space field parallelepiped."""
        # Worst-case corner offset from centre.
        half = np.array(
            [field_shape.Na, field_shape.Nb, field_shape.Nc],
            dtype=np.float64,
        ) * 0.5
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(FCC.index_to_world(half * np.array([sx, sy, sz])))
        return float(max(np.linalg.norm(c) for c in corners))

    # ------------------------------------------------------------------

    def render(
        self,
        field,                       # fcc_field.FCCField
        camera: OrbitCamera,
        viewport: Tuple[int, int],
        settings: RaymarchSettings | None = None,
    ) -> None:
        """Render the field's current state into the bound framebuffer."""
        if settings is None:
            settings = RaymarchSettings()

        w, h = viewport
        self.ctx.viewport = (0, 0, w, h)
        aspect = w / max(h, 1)

        view  = camera.view_matrix()
        proj  = camera.projection_matrix(aspect)
        view_inv = np.linalg.inv(view).astype('f4')
        proj_inv = np.linalg.inv(proj).astype('f4')

        # Texture binding (sampler unit 0).
        field.current.use(location=0)
        self.prog['u_field'].value = 0
        self.prog['u_view_inv'].write(view_inv.T.tobytes())   # GL column-major
        self.prog['u_proj_inv'].write(proj_inv.T.tobytes())
        self.prog['u_cam_pos'].value = tuple(float(x) for x in camera.eye())
        self.prog['u_field_size'].value = (
            float(field.shape.Na),
            float(field.shape.Nb),
            float(field.shape.Nc),
        )
        self.prog['u_step_size'].value     = settings.step_size
        self.prog['u_max_steps'].value     = settings.max_steps
        self.prog['u_channel_weights'].value = settings.channel_weights
        self.prog['u_bg_color'].value      = settings.bg_color
        self.prog['u_density_gain'].value  = settings.density_gain
        self.prog['u_alpha_max'].value     = settings.alpha_max
        self.prog['u_warm'].value          = settings.warm
        self.prog['u_cool'].value          = settings.cool

        self.vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        self.vao.release()
        self.vbo.release()
        self.prog.release()


# ---------------------------------------------------------------------------
# Stand-alone compile check
# ---------------------------------------------------------------------------


def _self_check() -> None:
    print("[fcc_render] booting standalone context...")
    ctx = moderngl.create_standalone_context(require=430)
    rm = Raymarcher(ctx)
    print(f"[fcc_render] shader programs compiled OK")
    rm.release()
    print("[fcc_render] self-check: OK")


if __name__ == "__main__":
    _self_check()
