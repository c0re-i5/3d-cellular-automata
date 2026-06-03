
#version 430
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_half_res;    // half-resolution volume render
uniform vec2 u_texel_size;       // 1.0 / half_res_dimensions

void main() {
    // Joint bilateral upsample (5-tap: center + 4 diagonal neighbors at the
    // adjacent half-res texel centers).
    //
    // PRIOR BUG: the 4 corner samples used offsets of ±0.5 * u_texel_size,
    // which sample BETWEEN half-res texels. With LINEAR filtering enabled
    // those reads collapse to bilinear-interpolated values that already
    // include the center, so the bilateral weights had nothing distinct to
    // compare against — the filter degenerated to a plain box blur.
    //
    // The correct neighbor offset on the half-res grid is ±1.0 texel.
    vec2 uv = v_uv;
    vec4 center = texture(u_half_res, uv);

    vec4 s00 = texture(u_half_res, uv + vec2(-1.0, -1.0) * u_texel_size);
    vec4 s10 = texture(u_half_res, uv + vec2( 1.0, -1.0) * u_texel_size);
    vec4 s01 = texture(u_half_res, uv + vec2(-1.0,  1.0) * u_texel_size);
    vec4 s11 = texture(u_half_res, uv + vec2( 1.0,  1.0) * u_texel_size);

    // Range (color) sigma: how perceptually different a sample must be before
    // its weight collapses. 0.08 keeps the filter sharp on iso-surfaces while
    // still smoothing within smooth gradients.
    const float sigma_color = 0.08;
    const float inv_2sig2   = 1.0 / (2.0 * sigma_color * sigma_color);

    // Squared color distance from center (Euclidean in linear RGB).
    vec3 d00 = s00.rgb - center.rgb;
    vec3 d10 = s10.rgb - center.rgb;
    vec3 d01 = s01.rgb - center.rgb;
    vec3 d11 = s11.rgb - center.rgb;

    float w00 = exp(-dot(d00, d00) * inv_2sig2);
    float w10 = exp(-dot(d10, d10) * inv_2sig2);
    float w01 = exp(-dot(d01, d01) * inv_2sig2);
    float w11 = exp(-dot(d11, d11) * inv_2sig2);
    // Center always carries a weight of 1 — guarantees a nonzero denominator
    // and keeps the result anchored on the original sample.
    float w_sum = 1.0 + w00 + w10 + w01 + w11;

    vec3 result = (center.rgb
                 + s00.rgb * w00
                 + s10.rgb * w10
                 + s01.rgb * w01
                 + s11.rgb * w11) / w_sum;

    fragColor = vec4(result, 1.0);
}
