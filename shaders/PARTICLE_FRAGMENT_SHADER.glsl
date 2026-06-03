
#version 430
in vec2 v_uv;
in vec3 v_color;
in float v_life;
out vec4 fragColor;

uniform float u_alpha;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;
    // Soft Gaussian-ish falloff (exp(-3r²) is fine and cheap).
    float fall = exp(-3.0 * r2);
    float a = u_alpha * fall * clamp(v_life, 0.0, 1.0);
    // Pre-multiplied additive: (rgb*a, a)
    fragColor = vec4(v_color * a, a);
}
