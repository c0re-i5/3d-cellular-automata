
#version 430
in vec3 v_normal;
in vec3 v_world;
out vec4 fragColor;

uniform vec3  u_camera_pos;
uniform vec3  u_color;
uniform float u_alpha;
uniform int   u_double_sided;
uniform int   u_size;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_camera_pos - v_world);   // headlight
    float cosNL = dot(N, L);
    if (u_double_sided != 0) cosNL = abs(cosNL);
    else                     cosNL = max(cosNL, 0.0);
    // Hemispheric ambient (sky/ground tint) + Lambert.
    vec3 sky    = vec3(0.55, 0.62, 0.75);
    vec3 ground = vec3(0.12, 0.10, 0.08);
    float h = 0.5 + 0.5 * N.y;
    vec3 ambient = mix(ground, sky, h) * 0.35;
    vec3 col = u_color * (ambient + cosNL * 0.85);
    // Distance-based subtle desaturation
    float d = length(u_camera_pos - v_world) / float(u_size);
    col = mix(col, vec3(dot(col, vec3(0.299, 0.587, 0.114))), clamp(d * 0.05, 0.0, 0.3));
    fragColor = vec4(col, u_alpha);
}
