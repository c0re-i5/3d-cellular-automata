
#version 430
layout(local_size_x=4, local_size_y=4, local_size_z=4) in;

// RGBA16F view texture:
//   .r       = scalar density used for occupancy/min-max accel + iso/MIP
//   .gba     = pre-mapped colour (or zero when u_vis_mode==0)
// Downstream raymarchers read .r for density and (when u_use_baked_color
// is set) .gba directly as the per-voxel colour, bypassing apply_colormap.
layout(rgba16f, binding=0) writeonly uniform image3D u_view;
uniform sampler3D u_volume;
uniform int u_size;
uniform int u_channel;     // primary channel for density (legacy)
uniform int u_use_abs;     // 0=raw, 1=abs, 2=signed→[0,1]
// Optional value-range remap so rules whose channel range exceeds [0,1]
// (e.g. sandpile grain count 0..12) don't render as a fully-opaque
// "ghost cube". After the abs/sign step we map (v - lo)/(hi - lo) and
// clamp to [0,1]. Default lo=0, hi=1 is a no-op (legacy behaviour).
uniform float u_vis_lo;
uniform float u_vis_hi;
// Multi-channel transfer-function mode:
//   0 = DENSITY      (legacy: only u_channel; .gba=0; raymarcher uses apply_colormap)
//   1 = RGB_CHANNELS (R=ch0, G=ch1, B=ch2; density=length / sqrt(3))
//   2 = HSV_PHASE    (hue=atan2(ch1,ch0)/2π, sat=1, val=length(ch0,ch1); density=val)
//   3 = BIPOLAR      (signed ch0 → diverging blue/white/red; density=|ch0|)
//   4 = RGBA_BLEND   (R=ch0,G=ch1,B=ch2; density=ch3)
uniform int u_vis_mode;
// Per-mode auxiliary range used for normalising vector magnitudes / phase values
// (defaults set from preset; falls back to (vis_lo, vis_hi) when not specified).
uniform float u_aux_lo;
uniform float u_aux_hi;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 diverging_bwr(float d) {
    // d in [-1, 1] -> blue (cold) ↔ white ↔ red (warm)
    float a = clamp(abs(d), 0.0, 1.0);
    vec3 cold = vec3(0.10, 0.30, 0.95);
    vec3 warm = vec3(0.95, 0.20, 0.20);
    vec3 hue  = (d >= 0.0) ? warm : cold;
    return mix(vec3(0.95, 0.95, 0.95), hue, a);
}

void main() {
    ivec3 p = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(p, ivec3(u_size)))) return;
    vec4 raw = texelFetch(u_volume, p, 0);

    float density = 0.0;
    vec3  colour  = vec3(0.0);
    float scale = (u_vis_hi > u_vis_lo) ? (1.0 / (u_vis_hi - u_vis_lo)) : 1.0;
    float aux_scale = (u_aux_hi > u_aux_lo) ? (1.0 / (u_aux_hi - u_aux_lo)) : 1.0;

    if (u_vis_mode == 1) {
        // RGB_CHANNELS: each channel mapped through (vis_lo, vis_hi)
        vec3 rgb = clamp((raw.rgb - vec3(u_vis_lo)) * scale, 0.0, 1.0);
        density = clamp(length(rgb) * (1.0 / 1.7320508), 0.0, 1.0);
        colour  = rgb;
    } else if (u_vis_mode == 2) {
        // HSV_PHASE: (ch0, ch1) treated as a 2-vector
        float a = raw.r, b = raw.g;
        float mag = length(vec2(a, b));
        // Inverted aux range (lo > hi) flips amplitude→opacity so that
        // *defect cores* (low |A| in an oscillator otherwise saturated
        // at |A|≈1) render bright and the bulk vanishes — essential
        // for CGLE/BZ where the spiral filaments live at |A|<<1.
        float val;
        if (u_aux_hi >= u_aux_lo) {
            val = clamp((mag - u_aux_lo) * aux_scale, 0.0, 1.0);
        } else {
            float inv_scale = 1.0 / (u_aux_lo - u_aux_hi);
            val = clamp(1.0 - (mag - u_aux_hi) * inv_scale, 0.0, 1.0);
        }
        float hue = atan(b, a) / 6.2831853 + 0.5;  // wrap to [0,1]
        colour  = hsv2rgb(vec3(hue, 1.0, val));
        density = val;
    } else if (u_vis_mode == 3) {
        // BIPOLAR: signed ch0
        float v = raw[u_channel];
        float d = clamp(v * aux_scale, -1.0, 1.0);   // aux_lo unused; aux_hi normalises
        if (u_aux_hi <= u_aux_lo) d = clamp(v, -1.0, 1.0);
        density = abs(d);
        colour  = diverging_bwr(d);
    } else if (u_vis_mode == 4) {
        // RGBA_BLEND: explicit per-channel colour, density from ch3
        vec3 rgb = clamp((raw.rgb - vec3(u_vis_lo)) * scale, 0.0, 1.0);
        density  = clamp((raw.a - u_aux_lo) * aux_scale, 0.0, 1.0);
        colour   = rgb;
    } else {
        // 0 = DENSITY (legacy single-channel, colormap applied in raymarcher)
        float v = raw[u_channel];
        if (u_use_abs == 1)      v = abs(v);
        else if (u_use_abs == 2) v = v * 0.5 + 0.5;
        v = clamp((v - u_vis_lo) * scale, 0.0, 1.0);
        density = v;
        colour  = vec3(0.0);  // raymarcher will recolour via apply_colormap
    }

    imageStore(u_view, p, vec4(density, colour));
}
