
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(rgba32f, binding=0) uniform image3D u_src;
layout(rgba32f, binding=1) uniform image3D u_dst;

// Element property table: 120 elements x 16 floats each (0=vacuum, 1-118=elements, 119=wall)
// Layout per element:
//   [0] atomic_number  [1] mass      [2] electronegativity  [3] valence_electrons
//   [4] melting_point  [5] boiling_pt [6] density           [7] thermal_cond
//   [8] color_r        [9] color_g    [10] color_b          [11] phase_at_25C
//   [12] group         [13] period    [14] category_id      [15] reserved
layout(std430, binding=2) buffer ElementTable {
    float elements[];  // 120 * 16 floats
};

uniform int u_size;
uniform float u_dt;
uniform float u_param0;  // ambient temperature
uniform float u_param1;  // gravity strength
uniform float u_param2;  // reaction rate multiplier
uniform float u_param3;  // unused
uniform int u_boundary;  // 0 = toroidal (wrap), 1 = clamped (Dirichlet, zero outside), 2 = mirror (Neumann)

// Cell channels:
// R = element ID (0=vacuum, 1-118=element), encoded as float
// G = temperature (°C)
// B = phase (0=solid, 1=liquid, 2=gas)
// A = velocity_y (for gravity/buoyancy)

vec4 fetch(ivec3 p) {
    if (u_boundary == 1) {
        if (any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, ivec3(u_size))))
            return vec4(0.0);
        return imageLoad(u_src, p);
    }
    if (u_boundary == 2) {
        // Mirror (Neumann zero-flux): treat the wall as identical to its inner neighbor.
        p = clamp(p, ivec3(0), ivec3(u_size - 1));
        return imageLoad(u_src, p);
    }
    p = (p + u_size) % u_size;
    return imageLoad(u_src, p);
}

// Get element property by atomic number and property index
float elem_prop(int z, int prop) {
    if (z < 0 || z > 118) return 0.0;
    return elements[z * 16 + prop];
}

float get_mass(int z)       { return elem_prop(z, 1); }
float get_eneg(int z)       { return elem_prop(z, 2); }
float get_valence(int z)    { return elem_prop(z, 3); }
float get_mp(int z)         { return elem_prop(z, 4); }
float get_bp(int z)         { return elem_prop(z, 5); }
float get_density(int z)    { return elem_prop(z, 6); }
float get_thermal(int z)    { return elem_prop(z, 7); }
float get_category(int z)   { return elem_prop(z, 14); }

// Determine phase from temperature and element properties
float compute_phase(int z, float temp) {
    float mp = get_mp(z);
    float bp = get_bp(z);
    if (temp < mp) return 0.0;       // solid
    if (temp < bp) return 1.0;       // liquid
    return 2.0;                       // gas
}
