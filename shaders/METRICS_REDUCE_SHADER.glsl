
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(rgba32f, binding=0) readonly uniform image3D u_current;
layout(rgba32f, binding=1) readonly uniform image3D u_prev;

layout(std430, binding=5) buffer Metrics {
    uint alive_count;
    uint change_count;
    uint surface_count;
    uint nan_count;
};

uniform int u_size;
uniform float u_threshold;
uniform int u_channel;
uniform int u_mode;        // 0=discrete, 1=continuous, 2=wave, 3=element
uniform float u_change_thr;
uniform int u_has_prev;    // 0 = no previous snapshot, skip activity
uniform int u_boundary;    // 0 = toroidal, 1 = clamped (Dirichlet), 2 = mirror (Neumann)

shared uint s_alive;
shared uint s_change;
shared uint s_surface;
shared uint s_nan;

bool is_alive(vec4 cell) {
    float v = cell[u_channel];
    if (u_mode == 3) return abs(v) > 0.5 && abs(v - 119.0) > 0.5;
    if (u_mode == 2) return abs(v) > u_threshold;
    return v > u_threshold;
}

vec4 safe_load(ivec3 p) {
    if (u_boundary == 1) {
        // Clamped: out-of-bounds reads as dead (zero)
        if (any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, ivec3(u_size))))
            return vec4(0.0);
        return imageLoad(u_current, p);
    } else if (u_boundary == 2) {
        // Mirror: out-of-bounds reflects to its inner neighbor (zero-flux).
        return imageLoad(u_current, clamp(p, ivec3(0), ivec3(u_size - 1)));
    } else {
        // Toroidal: wrap
        return imageLoad(u_current, (p + ivec3(u_size)) % ivec3(u_size));
    }
}

void main() {
    uint lid = gl_LocalInvocationIndex;

    // Clear shared counters (one thread per workgroup)
    if (lid == 0u) {
        s_alive = 0u; s_change = 0u; s_surface = 0u; s_nan = 0u;
    }
    barrier();

    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (all(lessThan(pos, ivec3(u_size)))) {
        vec4 cur = imageLoad(u_current, pos);
        float val = cur[u_channel];

        // NaN / Inf check
        if (isnan(val) || isinf(val)) {
            atomicAdd(s_nan, 1u);
        } else {
            bool alive = is_alive(cur);

            if (alive) {
                atomicAdd(s_alive, 1u);

                // Surface check: alive cell with < 6 alive von-Neumann neighbours
                int solid = 0;
                if (is_alive(safe_load(pos + ivec3(-1, 0, 0)))) solid++;
                if (is_alive(safe_load(pos + ivec3( 1, 0, 0)))) solid++;
                if (is_alive(safe_load(pos + ivec3( 0,-1, 0)))) solid++;
                if (is_alive(safe_load(pos + ivec3( 0, 1, 0)))) solid++;
                if (is_alive(safe_load(pos + ivec3( 0, 0,-1)))) solid++;
                if (is_alive(safe_load(pos + ivec3( 0, 0, 1)))) solid++;
                if (solid < 6) atomicAdd(s_surface, 1u);
            }

            // Activity: compare with previous snapshot
            if (u_has_prev != 0) {
                vec4 prv = imageLoad(u_prev, pos);
                float prev_val = prv[u_channel];
                float diff = abs(val - prev_val);
                bool changed = diff > u_change_thr;
                if (u_mode == 3) {
                    float temp_diff = abs(cur[1] - prv[1]);
                    changed = changed || temp_diff > 1.0;
                }
                if (changed) atomicAdd(s_change, 1u);
            }
        }
    }

    // Flush workgroup totals to global memory (one atomic per counter)
    barrier();
    if (lid == 0u) {
        if (s_alive > 0u)   atomicAdd(alive_count, s_alive);
        if (s_change > 0u)  atomicAdd(change_count, s_change);
        if (s_surface > 0u) atomicAdd(surface_count, s_surface);
        if (s_nan > 0u)     atomicAdd(nan_count, s_nan);
    }
}
