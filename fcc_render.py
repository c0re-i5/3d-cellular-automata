"""
FCC voxel renderer.

Two-pass GPU pipeline:

1.  **Cull / pack compute pass** (`_CULL_CS`). Walks the field texture, picks
    cells whose value is above ``u_threshold``, counts alive axial-FCC
    neighbours for an AO shade hint, and packs the triple ``(a, b, c, shade)``
    into one ``uint`` written to an SSBO. An ``atomicAdd`` on the indirect
    draw command's ``instance_count`` field eliminates the CPU round-trip:
    the draw call's instance count is set entirely on the GPU.

2.  **Instanced draw pass** (`_VOXEL_VS`/`_VOXEL_FS`). One instance per
    surviving cell, 36 verts each (12 triangles, 6 rhombic faces).
    The FCC primitive cell is a rhombohedron, so the unit-cube vertex
    table is interpreted in *index space* and transformed to world space
    by ``LATTICE_M``. The 6 face normals in world space come from
    ``transpose(LATTICE_M_INV) * face_dir_index``.

Bit packing
-----------
``9 + 9 + 9 + 5 = 32`` bits per voxel: 9 bits per axis (max grid 512) and
5 bits of AO shade (32 levels). For grids above 512 widen the packing to
``uvec2`` — not a v0 concern.

Vis modes
---------
* ``DENSITY`` (0): ``value = sample[u_channel]``, run through a colormap
  in the fragment shader (fire / cool / discrete).
* ``RGB_CHANNELS`` (1): colour is taken straight from ``sample.rgb``;
  density used for the alive test is ``length(sample.rgb) / sqrt(3)``.

Other legacy vis modes (HSV_PHASE, BIPOLAR, RGBA_BLEND) will land when the
first rule that needs them arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Tuple

import numpy as np
import moderngl

from lattice import FCC


# ---------------------------------------------------------------------------
# Camera (right-handed, GL convention)
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
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
    target: np.ndarray = dc_field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    distance: float = 5.0
    azimuth: float = 0.7
    elevation: float = 0.5
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
# Packing constants (kept in sync between Python and GLSL)
# ---------------------------------------------------------------------------

BITS_POS = 9
BITS_SHADE = 5
MAX_DIM = 1 << BITS_POS                          # 512
SHADE_LEVELS = 1 << BITS_SHADE                   # 32

VIS_MODE_DENSITY = 0
VIS_MODE_RGB_CHANNELS = 1


# ---------------------------------------------------------------------------
# Shader sources
# ---------------------------------------------------------------------------

_COMMON_HEADER_TEMPLATE = """\
#define BITS_POS    {bits_pos}
#define BITS_SHADE  {bits_shade}
#define MASK_POS    {mask_pos}u
#define MASK_SHADE  {mask_shade}u
#define SHADE_MAX   {shade_max}
""".format(
    bits_pos=BITS_POS,
    bits_shade=BITS_SHADE,
    mask_pos=(1 << BITS_POS) - 1,
    mask_shade=(1 << BITS_SHADE) - 1,
    shade_max=(1 << BITS_SHADE) - 1,
)


_CULL_CS = (
    """\
#version 430
layout(local_size_x=4, local_size_y=4, local_size_z=4) in;

/*__LATTICE_HEADER__*/
/*__PACK_HEADER__*/

uniform sampler3D u_field;
uniform ivec3     u_dims;
uniform int       u_channel;
uniform float     u_threshold;
uniform int       u_vis_mode;

layout(std430, binding=0) buffer VoxelBuf  { uint voxels[]; };
layout(std430, binding=1) buffer DrawCmd   {
    uint vert_count;        // 36, set on CPU
    uint instance_count;    // atomicAdd'd here
    uint first;             // 0
    uint base_instance;     // 0
};

float density(vec4 s) {
    if (u_vis_mode == 1) {
        return length(s.rgb) * 0.5773502691896258;   // 1/sqrt(3)
    }
    return s[u_channel];
}

bool alive_at(ivec3 p) {
    // Periodic wrap so neighbour reads at the boundary stay defined.
    ivec3 q = ((p % u_dims) + u_dims) % u_dims;
    return density(texelFetch(u_field, q, 0)) > u_threshold;
}

void main() {
    ivec3 p = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(p, u_dims))) return;
    if (!alive_at(p)) return;

    // AO shade hint: count alive 12-NN neighbours (0..12), scale to 0..SHADE_MAX.
    int cnt = 0;
    for (int k = 0; k < LATTICE_N_NEIGHBOURS; ++k) {
        if (alive_at(p + LATTICE_NEIGHBOURS[k])) cnt++;
    }
    uint shade = uint(clamp(cnt * SHADE_MAX / LATTICE_N_NEIGHBOURS, 0, SHADE_MAX));

    uint packed_data = (uint(p.x) & MASK_POS)
                     | ((uint(p.y) & MASK_POS) << uint(BITS_POS))
                     | ((uint(p.z) & MASK_POS) << uint(2 * BITS_POS))
                     | ((shade    & MASK_SHADE) << uint(3 * BITS_POS));

    uint idx = atomicAdd(instance_count, 1u);
    voxels[idx] = packed_data;
}
"""
)


_VOXEL_VS = (
    """\
#version 430

layout(std430, binding=0) buffer VoxelBuf { uint voxels[]; };

/*__LATTICE_HEADER__*/
/*__PACK_HEADER__*/

uniform sampler3D u_field;
uniform ivec3     u_dims;
uniform int       u_channel;
uniform float     u_threshold;
uniform int       u_vis_mode;
uniform float     u_voxel_shrink;     // 1.0 = touching cells, 0.92 = small visible gap
uniform mat4      u_view_proj;

// 36 verts (6 faces x 2 tris x 3 verts) in INDEX space.
// Order: -c, +c, -a, +a, -b, +b.  Winding is CCW when looking from
// outside the cell, given a right-handed basis (det(LATTICE_M) > 0).
const vec3 cube_verts[36] = vec3[36](
    vec3(0,0,0), vec3(1,0,0), vec3(1,1,0),  vec3(0,0,0), vec3(1,1,0), vec3(0,1,0),
    vec3(0,0,1), vec3(1,1,1), vec3(1,0,1),  vec3(0,0,1), vec3(0,1,1), vec3(1,1,1),
    vec3(0,0,0), vec3(0,1,0), vec3(0,1,1),  vec3(0,0,0), vec3(0,1,1), vec3(0,0,1),
    vec3(1,0,0), vec3(1,1,1), vec3(1,1,0),  vec3(1,0,0), vec3(1,0,1), vec3(1,1,1),
    vec3(0,0,0), vec3(0,0,1), vec3(1,0,1),  vec3(0,0,0), vec3(1,0,1), vec3(1,0,0),
    vec3(0,1,0), vec3(1,1,0), vec3(1,1,1),  vec3(0,1,0), vec3(1,1,1), vec3(0,1,1)
);

const ivec3 face_dirs_idx[6] = ivec3[6](
    ivec3(0,0,-1), ivec3(0,0,1),
    ivec3(-1,0,0), ivec3(1,0,0),
    ivec3(0,-1,0), ivec3(0,1,0)
);

out vec3  v_world_normal;
out vec3  v_world_pos;
out vec3  v_color;
out float v_value;
out float v_shade;

float density(vec4 s) {
    if (u_vis_mode == 1) return length(s.rgb) * 0.5773502691896258;
    return s[u_channel];
}

bool alive_at(ivec3 p) {
    ivec3 q = ((p % u_dims) + u_dims) % u_dims;
    return density(texelFetch(u_field, q, 0)) > u_threshold;
}

void main() {
    uint pdata = voxels[gl_InstanceID];
    ivec3 ipos = ivec3(int(pdata          & MASK_POS),
                       int((pdata >> uint(BITS_POS))     & MASK_POS),
                       int((pdata >> uint(2 * BITS_POS)) & MASK_POS));
    uint shade_hint = (pdata >> uint(3 * BITS_POS)) & MASK_SHADE;

    int face_id = gl_VertexID / 6;

    // Hidden-face cull: if the axial neighbour across this face is alive,
    // emit a degenerate triangle so the rasteriser drops it.
    if (alive_at(ipos + face_dirs_idx[face_id])) {
        gl_Position    = vec4(0.0);
        v_world_normal = vec3(0.0);
        v_world_pos    = vec3(0.0);
        v_color        = vec3(0.0);
        v_value        = 0.0;
        v_shade        = 0.0;
        return;
    }

    vec3 local       = cube_verts[gl_VertexID];
    vec3 local_shrunk = (local - 0.5) * u_voxel_shrink + 0.5;
    vec3 idx_pos     = vec3(ipos) + local_shrunk;
    vec3 world_pos   = LATTICE_M * idx_pos;

    gl_Position = u_view_proj * vec4(world_pos, 1.0);

    // World-space face normal: covariant transform of the index-space normal.
    vec3 n_idx = vec3(face_dirs_idx[face_id]);
    v_world_normal = normalize(transpose(LATTICE_M_INV) * n_idx);
    v_world_pos    = world_pos;

    vec4 cell = texelFetch(u_field, ipos, 0);
    if (u_vis_mode == 1) {
        v_color = clamp(cell.rgb, 0.0, 1.0);
        v_value = clamp(length(cell.rgb) * 0.5773502691896258, 0.0, 1.0);
    } else {
        float v = cell[u_channel];
        v_value = clamp(v, 0.0, 1.0);
        v_color = vec3(0.0);     // colormap applied in FS
    }
    v_shade = float(shade_hint) / float(SHADE_MAX);
}
"""
)


_VOXEL_FS = """\
#version 430

in  vec3  v_world_normal;
in  vec3  v_world_pos;
in  vec3  v_color;
in  float v_value;
in  float v_shade;
out vec4  frag_color;

uniform vec3  u_camera_pos;
uniform vec3  u_light_dir;
uniform int   u_vis_mode;
uniform int   u_colormap;        // 0=fire, 1=cool, 2=discrete
uniform float u_brightness;
uniform float u_ao_strength;     // 0..1, fraction of brightness lost in deepest interior

vec3 colormap_fire(float t) {
    return vec3(clamp(t * 3.0, 0.0, 1.0),
                clamp(t * 3.0 - 1.0, 0.0, 1.0),
                clamp(t * 3.0 - 2.0, 0.0, 1.0));
}
vec3 colormap_cool(float t) {
    return vec3(sin(t * 1.5708) * 0.3, t * 0.8, 0.5 + t * 0.5);
}
vec3 colormap_discrete(float t) {
    int idx = clamp(int(floor(t * 16.0)), 0, 15);
    float hue = fract(float(idx) * 0.618033988);
    float s = 0.75, v = 0.95, c = v * s;
    float h = hue * 6.0;
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    vec3 rgb;
    if      (h < 1.0) rgb = vec3(c, x, 0);
    else if (h < 2.0) rgb = vec3(x, c, 0);
    else if (h < 3.0) rgb = vec3(0, c, x);
    else if (h < 4.0) rgb = vec3(0, x, c);
    else if (h < 5.0) rgb = vec3(x, 0, c);
    else              rgb = vec3(c, 0, x);
    return rgb + vec3(v - c);
}

void main() {
    vec3 base;
    if (u_vis_mode == 1) {
        base = v_color;
    } else {
        float t = clamp(v_value, 0.0, 1.0);
        if      (u_colormap == 1) base = colormap_cool(t);
        else if (u_colormap == 2) base = colormap_discrete(t);
        else                      base = colormap_fire(t);
    }

    vec3 n = normalize(v_world_normal);
    vec3 l = normalize(u_light_dir);
    float diff    = max(dot(n, l), 0.0);
    float ambient = 0.25;
    vec3  v       = normalize(u_camera_pos - v_world_pos);
    float spec    = pow(max(dot(reflect(-l, n), v), 0.0), 24.0);

    float ao = mix(1.0, 1.0 - u_ao_strength, v_shade);

    vec3 lit = base * (ambient + diff * 0.7) * ao * u_brightness
             + vec3(spec * 0.15);
    frag_color = vec4(lit, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class VoxelSettings:
    threshold:     float = 0.5
    vis_mode:      int = VIS_MODE_DENSITY
    channel:       int = 0
    colormap:      int = 0                  # fire
    brightness:    float = 1.0
    voxel_shrink:  float = 1.0
    light_dir:     Tuple[float, float, float] = (0.45, 1.0, 0.30)
    ao_strength:   float = 0.35
    bg_color:      Tuple[float, float, float] = (0.04, 0.04, 0.06)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def _inject(src: str) -> str:
    return (
        src.replace('/*__LATTICE_HEADER__*/', FCC.glsl_header())
           .replace('/*__PACK_HEADER__*/',    _COMMON_HEADER_TEMPLATE)
    )


class VoxelRenderer:
    """FCC voxel renderer with GPU-side compute-cull + indirect draw."""

    # Indirect draw command struct: (vert_count, instance_count, first, base_instance).
    # vert_count is constant (36); compute shader atomicAdd's to instance_count.
    _CMD_DTYPE = np.dtype(np.uint32)

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.cull = ctx.compute_shader(_inject(_CULL_CS))
        self.prog = ctx.program(
            vertex_shader=_inject(_VOXEL_VS),
            fragment_shader=_VOXEL_FS,
        )

        # Indirect command buffer (read-write by GPU; rewritten each frame).
        self._cmd = ctx.buffer(reserve=16)              # 4 uints
        # Voxel SSBO grown on demand.
        self._voxels: moderngl.Buffer | None = None
        self._voxel_cap: int = 0

        # VAO that pulls everything from gl_VertexID / gl_InstanceID.
        self._vao = ctx.vertex_array(self.prog, [])

    # ------------------------------------------------------------------

    def field_world_center(self, shape) -> np.ndarray:
        idx_centre = np.array([shape.Na, shape.Nb, shape.Nc], dtype=np.float64) * 0.5
        return FCC.index_to_world(idx_centre)

    def field_world_radius(self, shape) -> float:
        half = np.array([shape.Na, shape.Nb, shape.Nc], dtype=np.float64) * 0.5
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(FCC.index_to_world(half * np.array([sx, sy, sz])))
        return float(max(np.linalg.norm(c) for c in corners))

    # ------------------------------------------------------------------

    def _ensure_voxel_buf(self, capacity: int) -> None:
        if self._voxels is not None and capacity <= self._voxel_cap:
            return
        if self._voxels is not None:
            self._voxels.release()
        # 4 bytes per packed voxel.
        self._voxels = self.ctx.buffer(reserve=capacity * 4)
        self._voxel_cap = capacity

    def render(
        self,
        field,
        camera: OrbitCamera,
        viewport: Tuple[int, int],
        settings: VoxelSettings | None = None,
    ) -> int:
        """Render ``field`` to the bound framebuffer. Returns the number of
        voxels emitted by the cull pass (read back from the indirect buffer
        for diagnostics; not free, ~100 us, skip in hot inner loop)."""
        if settings is None:
            settings = VoxelSettings()

        shape = field.shape
        if max(shape.Na, shape.Nb, shape.Nc) > MAX_DIM:
            raise ValueError(
                f"grid edge > {MAX_DIM} not supported with {BITS_POS}-bit packing"
            )
        self._ensure_voxel_buf(shape.cell_count)

        # Reset indirect cmd: vert_count=36, instance_count=0, first=0, base_instance=0
        self._cmd.write(np.array([36, 0, 0, 0], dtype=np.uint32).tobytes())

        # ---- Cull / pack pass ----
        field.current.use(location=0)
        self.cull['u_field'].value     = 0
        self.cull['u_dims'].value      = (shape.Na, shape.Nb, shape.Nc)
        self.cull['u_channel'].value   = settings.channel
        self.cull['u_threshold'].value = settings.threshold
        self.cull['u_vis_mode'].value  = settings.vis_mode

        self._voxels.bind_to_storage_buffer(0)
        self._cmd.bind_to_storage_buffer(1)

        gx = (shape.Na + 3) // 4
        gy = (shape.Nb + 3) // 4
        gz = (shape.Nc + 3) // 4
        self.cull.run(gx, gy, gz)
        self.ctx.memory_barrier(
            moderngl.SHADER_STORAGE_BARRIER_BIT
            | moderngl.COMMAND_BARRIER_BIT
        )

        # ---- Draw pass ----
        w, h = viewport
        self.ctx.viewport = (0, 0, w, h)
        aspect = w / max(h, 1)
        view = camera.view_matrix()
        proj = camera.projection_matrix(aspect)
        vp = (proj @ view).astype('f4')

        field.current.use(location=0)
        self.prog['u_field'].value         = 0
        self.prog['u_dims'].value          = (shape.Na, shape.Nb, shape.Nc)
        self.prog['u_channel'].value       = settings.channel
        self.prog['u_threshold'].value     = settings.threshold
        self.prog['u_vis_mode'].value      = settings.vis_mode
        self.prog['u_voxel_shrink'].value  = settings.voxel_shrink
        self.prog['u_view_proj'].write(vp.T.tobytes())
        self.prog['u_camera_pos'].value    = tuple(float(x) for x in camera.eye())
        self.prog['u_colormap'].value      = settings.colormap
        self.prog['u_brightness'].value    = settings.brightness
        self.prog['u_light_dir'].value     = settings.light_dir
        self.prog['u_ao_strength'].value   = settings.ao_strength

        self._voxels.bind_to_storage_buffer(0)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.front_face = 'ccw'
        self.ctx.cull_face = 'back'

        # Indirect-draw doesn't dispatch in moderngl 5.10 on this driver
        # (compute shader writes the indirect cmd buffer atomically, but the
        # subsequent `render_indirect` call produces 0 instances even with a
        # COMMAND_BARRIER_BIT memory barrier). 16-byte CPU readback is cheap.
        cmd = np.frombuffer(self._cmd.read(), dtype=np.uint32)
        n_inst = int(cmd[1])
        if n_inst > 0:
            self._vao.render(
                mode=moderngl.TRIANGLES, vertices=36, instances=n_inst
            )

        return n_inst

    def release(self) -> None:
        self._vao.release()
        if self._voxels is not None:
            self._voxels.release()
        self._cmd.release()
        self.prog.release()
        self.cull.release()


# ---------------------------------------------------------------------------
# Stand-alone compile check
# ---------------------------------------------------------------------------


def _self_check() -> None:
    print("[fcc_render] booting standalone context...")
    ctx = moderngl.create_standalone_context(require=430)
    r = VoxelRenderer(ctx)
    print("[fcc_render] cull + draw programs compiled OK")
    print(f"[fcc_render] BITS_POS={BITS_POS} (max grid {MAX_DIM}), "
          f"BITS_SHADE={BITS_SHADE} ({SHADE_LEVELS} levels)")
    r.release()
    print("[fcc_render] self-check: OK")


if __name__ == "__main__":
    _self_check()
