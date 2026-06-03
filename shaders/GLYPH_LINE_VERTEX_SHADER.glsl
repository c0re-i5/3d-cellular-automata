
#version 430
uniform sampler3D u_field_tex;
uniform mat4  u_view_proj;
uniform int   u_size;
uniform vec3  u_world_per_cell; // box_dims/size — grid cell → world coord
uniform ivec3 u_grid_dim;     // glyphs along each axis
uniform vec3  u_origin;       // grid origin in cell coords
uniform vec3  u_step;         // spacing in cell coords
uniform float u_scale;        // length of unit vector in grid cells
uniform int   u_source;       // 0 = rgb directly, 1 = grad of R, 2 = curl of rgb
uniform float u_threshold;    // skip glyphs with |v| below this (avoids forest of noise)
uniform float u_mag_scale;    // scale factor when colouring by magnitude

out vec3 v_color;
out float v_alpha;

float fs;

vec4 sampleField(vec3 p) {
    p = clamp(p, vec3(0.0), vec3(fs - 1.0));
    return texture(u_field_tex, (p + 0.5) / fs);
}

vec3 grad_r(vec3 p) {
    float h = 1.0;
    return 0.5 * vec3(
        sampleField(p + vec3(h,0,0)).r - sampleField(p - vec3(h,0,0)).r,
        sampleField(p + vec3(0,h,0)).r - sampleField(p - vec3(0,h,0)).r,
        sampleField(p + vec3(0,0,h)).r - sampleField(p - vec3(0,0,h)).r);
}

vec3 curl_rgb(vec3 p) {
    float h = 1.0;
    vec3 xp = sampleField(p + vec3(h,0,0)).rgb;
    vec3 xm = sampleField(p - vec3(h,0,0)).rgb;
    vec3 yp = sampleField(p + vec3(0,h,0)).rgb;
    vec3 ym = sampleField(p - vec3(0,h,0)).rgb;
    vec3 zp = sampleField(p + vec3(0,0,h)).rgb;
    vec3 zm = sampleField(p - vec3(0,0,h)).rgb;
    vec3 dFdx = 0.5 * (xp - xm);
    vec3 dFdy = 0.5 * (yp - ym);
    vec3 dFdz = 0.5 * (zp - zm);
    return vec3(dFdy.z - dFdz.y, dFdz.x - dFdx.z, dFdx.y - dFdy.x);
}

void main() {
    fs = float(u_size);
    int gid = gl_VertexID / 2;
    int tip = gl_VertexID & 1;
    int gx = gid % u_grid_dim.x;
    int gy = (gid / u_grid_dim.x) % u_grid_dim.y;
    int gz = gid / (u_grid_dim.x * u_grid_dim.y);
    vec3 base = u_origin + vec3(gx, gy, gz) * u_step;

    vec3 v;
    if (u_source == 1)      v = grad_r(base);
    else if (u_source == 2) v = curl_rgb(base);
    else                    v = sampleField(base).rgb - 0.5;

    float mag = length(v);
    vec3 dir = mag > 1e-5 ? v / mag : vec3(0.0);
    vec3 pos = base + dir * (mag * u_scale * float(tip));

    // Cull below-threshold glyphs by collapsing them to a point off-screen.
    float vis = step(u_threshold, mag);
    pos = mix(vec3(-1e6), pos, vis);

    // Convert from grid cell coords [0..size] to world coords [0..box_dims].
    pos = pos * u_world_per_cell;

    gl_Position = u_view_proj * vec4(pos, 1.0);

    // Colour: cool→hot mapped from magnitude. Tips brighter than bases.
    float t = clamp(mag * u_mag_scale, 0.0, 1.0);
    vec3 c_low  = vec3(0.15, 0.4, 0.95);
    vec3 c_high = vec3(1.0, 0.75, 0.15);
    v_color = mix(c_low, c_high, t);
    v_alpha = mix(0.45, 0.95, float(tip));   // tip vertex more opaque
}
