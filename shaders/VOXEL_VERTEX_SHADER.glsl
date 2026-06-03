
#version 430

// Per-instance data from packed SSBO (1 uint per voxel — position only)
layout(std430, binding=3) buffer VoxelBuffer {
    uint voxels[];
};

// Element property table (for element CA color lookup)
layout(std430, binding=2) buffer ElementTable {
    float elements[];  // 120 * 16 floats
};

uniform sampler3D u_volume_tex;  // 3D volume texture for color sampling
uniform int u_size;
uniform ivec3 u_dims;            // (W, H, D) — per-axis bounds for nb_alive
uniform mat4 u_view_proj;
uniform float u_voxel_gap;  // gap between voxels (0 = touching, 0.1 = 10% gap)
uniform int u_is_element_ca;
uniform int u_channel;      // which channel to read for color
uniform int u_use_abs;      // 1 = use abs(value) for wave mode
uniform float u_threshold;  // visibility threshold for value normalization
// Multi-channel vis_mode (mirrors VIEW_TEX_BUILD_SHADER):
//   0 = DENSITY (legacy: u_channel + apply_colormap in fragment)
//   1 = RGB_CHANNELS (per-voxel rgb shown directly; e.g. ant team colors)
//   4 = RGBA_BLEND   (rgb=colour, density=.a)
// Modes 2/3 keep the legacy single-channel path.
uniform int u_vis_mode;
// Per-channel value range, same semantics as VIEW_TEX_BUILD_SHADER:
// rgb (or .a) values are remapped through (v - vis_lo) / (vis_hi - vis_lo)
// and clamped to [0,1]. Default (0,1) is a no-op.
uniform float u_vis_lo;
uniform float u_vis_hi;

// ── Lattice (FCC) index->world transform ──────────────────────────────
// When u_lattice_fcc == 1, cell index coordinates are mapped to world
// space through u_lattice_M (the FCC primitive-cell basis) so cells render
// as sheared rhombic cells at their true isotropic positions, recentred on
// the unit-cube centre. When u_lattice_fcc == 0 (cubic, the default) this
// block is skipped entirely and the legacy axis-aligned mapping is used,
// keeping cubic rules byte-for-byte unchanged.
uniform int  u_lattice_fcc;
uniform mat3 u_lattice_M;
uniform mat3 u_lattice_M_inv;
const vec3 cube_verts[36] = vec3[36](
    // -Z face
    vec3(0,0,0), vec3(1,0,0), vec3(1,1,0),  vec3(0,0,0), vec3(1,1,0), vec3(0,1,0),
    // +Z face
    vec3(0,0,1), vec3(1,1,1), vec3(1,0,1),  vec3(0,0,1), vec3(0,1,1), vec3(1,1,1),
    // -X face
    vec3(0,0,0), vec3(0,1,0), vec3(0,1,1),  vec3(0,0,0), vec3(0,1,1), vec3(0,0,1),
    // +X face
    vec3(1,0,0), vec3(1,1,1), vec3(1,1,0),  vec3(1,0,0), vec3(1,0,1), vec3(1,1,1),
    // -Y face
    vec3(0,0,0), vec3(0,0,1), vec3(1,0,1),  vec3(0,0,0), vec3(1,0,1), vec3(1,0,0),
    // +Y face
    vec3(0,1,0), vec3(1,1,0), vec3(1,1,1),  vec3(0,1,0), vec3(1,1,1), vec3(0,1,1)
);

const vec3 cube_normals[6] = vec3[6](
    vec3(0,0,-1), vec3(0,0,1), vec3(-1,0,0), vec3(1,0,0), vec3(0,-1,0), vec3(0,1,0)
);

// Face normal directions as ivec3 for neighbor lookup (matches cube_normals order)
const ivec3 face_dirs[6] = ivec3[6](
    ivec3(0,0,-1), ivec3(0,0,1), ivec3(-1,0,0), ivec3(1,0,0), ivec3(0,-1,0), ivec3(0,1,0)
);

// ── Rhombic-dodecahedron geometry (FCC Wigner-Seitz / Voronoi cell) ──
// When rendering the FCC lattice each cell is drawn as its true Voronoi
// cell: a rhombic dodecahedron whose 12 faces are the perpendicular
// bisectors of the 12 nearest-neighbour bonds. Adjacent cells share whole
// faces, so the cells visibly interlock with no gaps (the actual "FCC packs
// space perfectly" property). Canonical vertices (faces at distance sqrt(2)
// from centre): 8 cube-type (+-1,+-1,+-1) and 6 axis-type (+-2,0,0)... .
// In our world frame the 12 NN bonds lie along (+-1,+-1,0)-type directions,
// so the polyhedron is axis-aligned in world space — no shear of the cell
// shape itself, only of the lattice of cell centres.
// 12 faces x 2 triangles x 3 verts = 72 vertices, face_id = vert_id / 6.
const vec3 rd_verts[72] = vec3[72](
    // face 0  n=(+1,+1,0)
    vec3( 2,0,0), vec3( 1, 1, 1), vec3(0, 2,0),  vec3( 2,0,0), vec3(0, 2,0), vec3( 1, 1,-1),
    // face 1  n=(-1,-1,0)
    vec3(-2,0,0), vec3(-1,-1, 1), vec3(0,-2,0),  vec3(-2,0,0), vec3(0,-2,0), vec3(-1,-1,-1),
    // face 2  n=(0,+1,+1)
    vec3(0, 2,0), vec3( 1, 1, 1), vec3(0,0, 2),  vec3(0, 2,0), vec3(0,0, 2), vec3(-1, 1, 1),
    // face 3  n=(0,-1,-1)
    vec3(0,-2,0), vec3( 1,-1,-1), vec3(0,0,-2),  vec3(0,-2,0), vec3(0,0,-2), vec3(-1,-1,-1),
    // face 4  n=(+1,0,+1)
    vec3( 2,0,0), vec3( 1, 1, 1), vec3(0,0, 2),  vec3( 2,0,0), vec3(0,0, 2), vec3( 1,-1, 1),
    // face 5  n=(-1,0,-1)
    vec3(-2,0,0), vec3(-1, 1,-1), vec3(0,0,-2),  vec3(-2,0,0), vec3(0,0,-2), vec3(-1,-1,-1),
    // face 6  n=(+1,0,-1)
    vec3( 2,0,0), vec3( 1, 1,-1), vec3(0,0,-2),  vec3( 2,0,0), vec3(0,0,-2), vec3( 1,-1,-1),
    // face 7  n=(-1,0,+1)
    vec3(-2,0,0), vec3(-1, 1, 1), vec3(0,0, 2),  vec3(-2,0,0), vec3(0,0, 2), vec3(-1,-1, 1),
    // face 8  n=(-1,+1,0)
    vec3(-2,0,0), vec3(-1, 1, 1), vec3(0, 2,0),  vec3(-2,0,0), vec3(0, 2,0), vec3(-1, 1,-1),
    // face 9  n=(+1,-1,0)
    vec3( 2,0,0), vec3( 1,-1, 1), vec3(0,-2,0),  vec3( 2,0,0), vec3(0,-2,0), vec3( 1,-1,-1),
    // face 10 n=(0,+1,-1)
    vec3(0, 2,0), vec3( 1, 1,-1), vec3(0,0,-2),  vec3(0, 2,0), vec3(0,0,-2), vec3(-1, 1,-1),
    // face 11 n=(0,-1,+1)
    vec3(0,-2,0), vec3( 1,-1, 1), vec3(0,0, 2),  vec3(0,-2,0), vec3(0,0, 2), vec3(-1,-1, 1)
);

// Outward world-space normals for the 12 rhombic faces (unit length).
const vec3 rd_normals[12] = vec3[12](
    vec3( 0.70710678,  0.70710678, 0.0       ), vec3(-0.70710678, -0.70710678, 0.0       ),
    vec3( 0.0,         0.70710678, 0.70710678 ), vec3( 0.0,       -0.70710678,-0.70710678 ),
    vec3( 0.70710678,  0.0,        0.70710678 ), vec3(-0.70710678,  0.0,       -0.70710678 ),
    vec3( 0.70710678,  0.0,       -0.70710678 ), vec3(-0.70710678,  0.0,        0.70710678 ),
    vec3(-0.70710678,  0.70710678, 0.0       ), vec3( 0.70710678, -0.70710678, 0.0       ),
    vec3( 0.0,         0.70710678,-0.70710678 ), vec3( 0.0,       -0.70710678, 0.70710678 )
);

// Index-space neighbour offset behind each rhombic face (for hidden-face
// culling). Matches the FCC 12-NN order used by the compute stencil.
const ivec3 rd_neighbors[12] = ivec3[12](
    ivec3( 1, 0, 0), ivec3(-1, 0, 0), ivec3( 0, 1, 0), ivec3( 0,-1, 0),
    ivec3( 0, 0, 1), ivec3( 0, 0,-1), ivec3( 1,-1, 0), ivec3(-1, 1, 0),
    ivec3( 0, 1,-1), ivec3( 0,-1, 1), ivec3( 1, 0,-1), ivec3(-1, 0, 1)
);

out vec3 v_normal;
out vec3 v_world_pos;
out vec3 v_color;
out float v_value;

bool nb_alive(ivec3 p) {
    if (any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, u_dims))) return false;
    vec4 nb = texelFetch(u_volume_tex, p, 0);
    if (u_is_element_ca == 1) return int(round(nb.r)) > 0;
    if (u_vis_mode == 1) return max(max(nb.r, nb.g), nb.b) > u_threshold;
    if (u_vis_mode == 4) return nb.a > u_threshold;
    float v = nb[u_channel];
    if (u_use_abs == 1) v = abs(v);
    else if (u_use_abs == 2) v = abs(clamp(v, -1.0, 1.0));  // signed: alive if either sign
    return v > u_threshold;
}

void main() {
    int instance_id = gl_InstanceID;
    int vert_id = gl_VertexID;

    // Unpack position from 1 uint (4 bytes) — __PB__ bits per axis + __SB__-bit shading
    uint pdata = voxels[instance_id];
    uint shade_hint = (pdata >> (3u * __PB__)) & __SMASK__;

    // Sample color from 3D texture at this voxel's grid coordinate (exact texel)
    ivec3 ipos = ivec3(int(pdata & __PMASK__),
                        int((pdata >>  __PB__) & __PMASK__),
                        int((pdata >> (2u * __PB__)) & __PMASK__));

    // Per-face visibility: degenerate triangles for faces hidden by a neighbor.
    // Cubic cells have 6 faces (face_id 0..5); FCC rhombic-dodecahedron cells
    // have 12 (face_id 0..11). The hidden-face test uses the neighbour behind
    // each face — the 6 axial dirs for the cube, the 12 FCC bonds for the RD.
    int face_id = vert_id / 6;
    ivec3 cull_dir = (u_lattice_fcc == 1) ? rd_neighbors[face_id] : face_dirs[face_id];
    if (nb_alive(ipos + cull_dir)) {
        gl_Position = vec4(0.0);
        v_normal = vec3(0.0);
        v_world_pos = vec3(0.0);
        v_color = vec3(0.0);
        v_value = 0.0;
        return;
    }

    vec3 cell_pos = vec3(ipos);
    vec4 cell = texelFetch(u_volume_tex, ipos, 0);

    float value;
    vec3 color;

    if (u_is_element_ca == 1) {
        int z = int(round(cell.r));
        value = cell.r;
        // Color from element table
        color = vec3(
            elements[z * 16 + 8],
            elements[z * 16 + 9],
            elements[z * 16 + 10]
        );
        // Tint by temperature
        float temp = cell.g;
        float heat = clamp(temp / 2000.0, 0.0, 1.0);
        color = mix(color, vec3(1.0, 0.3, 0.1), heat * 0.5);
    } else {
        if (u_vis_mode == 1 || u_vis_mode == 4) {
            // Per-voxel RGB pre-baked by the simulation (team colours, R/G/B
            // channel composites, etc.). Apply the same (vis_lo, vis_hi)
            // remap that the volumetric raymarcher uses so voxel and ray
            // paths show identical hues.
            float scale = (u_vis_hi > u_vis_lo) ? (1.0 / (u_vis_hi - u_vis_lo)) : 1.0;
            color = clamp((cell.rgb - vec3(u_vis_lo)) * scale, 0.0, 1.0);
            // 'value' picks an alpha-style scalar used downstream by Phong
            // brightness. RGB_CHANNELS: max channel (so a pure-green voxel
            // still looks lit); RGBA_BLEND: explicit .a.
            value = (u_vis_mode == 4) ? clamp(cell.a, 0.0, 1.0)
                                      : clamp(max(max(color.r, color.g), color.b), 0.0, 1.0);
            // Preserve the AO darkening from the legacy path: deeply
            // embedded cells (high shade_hint) get dimmed without losing
            // hue. Same response curve as the colormap branch below.
            float sn = float(shade_hint) / float(__SHADE_MAX__);
            float ao = mix(1.0, 0.35, pow(sn, 0.7));
            color *= ao;
        } else {
            value = cell[u_channel];
            if (u_use_abs == 1) value = abs(value);
            else if (u_use_abs == 2) value = 0.5 + 0.5 * clamp(value, -1.0, 1.0);  // signed bipolar
            // Normalize value to [0,1] relative to threshold
            value = clamp((value - u_threshold) / max(1.0 - u_threshold, 0.001), 0.0, 1.0);

            // For binary CAs (value ~= 1.0), use the AO-weighted shading hint
            // (5 bits, 0..31) to give 32 smooth darkening levels instead of the
            // previous 4. This eliminates visible posterization on iso-surfaces.
            if (value > 0.99) {
                float sn = float(shade_hint) / float(__SHADE_MAX__);  // shade-bit hint, normalised to [0,1]
                // Map so isolated cells (sn ~ 0) are bright (0.85) and deeply
                // embedded cells (sn ~ 1) are dim (0.30). The exponent shapes
                // the response curve to emphasize edges.
                value = mix(0.85, 0.30, pow(sn, 0.7));
            }
            color = vec3(value);
        }
    }

    // Cell geometry
    float cell_size = 1.0 / float(u_size);
    float shrink = 1.0 - u_voxel_gap;
    vec3 world_pos;
    vec3 normal;
    if (u_lattice_fcc == 1) {
        // FCC: draw the cell's rhombic-dodecahedron Voronoi cell. The cell
        // CENTRES form the sheared FCC lattice (centre = M * index), but the
        // cell SHAPE is the axis-aligned RD, so neighbouring cells share whole
        // faces and interlock with no gaps. Recentre the block on the unit
        // cube so the existing camera framing (target ~ (0.5,0.5,0.5)) sees it.
        vec3 center = u_lattice_M * (cell_pos - vec3(float(u_size) * 0.5)) * cell_size
                      + vec3(0.5);
        // Canonical RD faces sit at distance sqrt(2); scale so they sit at
        // half the nearest-neighbour spacing (0.5 * cell_size) and apply the
        // voxel gap via `shrink`.
        const float RD_SCALE = 0.35355339;  // 0.5 / sqrt(2)
        world_pos = center + rd_verts[vert_id] * (cell_size * RD_SCALE * shrink);
        normal = rd_normals[face_id];
    } else {
        // Cubic: unit cube mapped into the [0,1]^3 box (legacy path, unchanged).
        vec3 local_pos = cube_verts[vert_id];
        normal = cube_normals[face_id];
        vec3 offset = cell_pos * cell_size + cell_size * 0.5 * (1.0 - shrink);
        world_pos = offset + local_pos * cell_size * shrink;
    }

    gl_Position = u_view_proj * vec4(world_pos, 1.0);
    v_normal = normal;
    v_world_pos = world_pos;
    v_color = color;
    v_value = value;
}
