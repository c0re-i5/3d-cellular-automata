
// Multi-element CA with real physical properties
// Movement uses matched thresholds: a cell vacates itself at the same threshold
// that a vacuum cell uses to accept an incoming atom, preventing duplication/loss.
void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    int self_id = int(round(self_val.r));
    float self_temp = self_val.g;
    float self_phase = self_val.b;
    float self_vy = self_val.a;

    // Wall element (id=119): indestructible, no physics, just stay put
    if (self_id == 119) {
        imageStore(u_dst, pos, self_val);
        return;
    }

    // Movement threshold — must be the same for sender and receiver
    const float MOVE_THRESHOLD = 0.3;

    // Vacuum stays vacuum (but can be filled by moving atoms)
    if (self_id == 0) {
        // Check if a neighbor wants to move here (gravity, buoyancy, diffusion)

        // Liquid/solid falling from above?
        vec4 above = fetch(pos + ivec3(0, 1, 0));
        int above_id = int(round(above.r));
        float above_phase = above.b;
        float above_vy = above.a;
        if (above_id > 0 && above_phase < 1.5 && above_vy < -MOVE_THRESHOLD) {
            imageStore(u_dst, pos, vec4(float(above_id), above.g, above_phase, above_vy * 0.8));
            return;
        }

        // Gas rising from below?
        vec4 below = fetch(pos + ivec3(0, -1, 0));
        int below_id = int(round(below.r));
        float below_phase = below.b;
        float below_vy = below.a;
        if (below_id > 0 && below_phase > 1.5 && below_vy > MOVE_THRESHOLD) {
            imageStore(u_dst, pos, vec4(float(below_id), below.g, below_phase, below_vy * 0.8));
            return;
        }

        // Liquid spreading sideways: check 4 horizontal neighbors for liquid wanting to spread
        for (int axis = 0; axis < 3; axis += 2) {  // X and Z axes only
            for (int dir = -1; dir <= 1; dir += 2) {
                ivec3 offset = ivec3(0);
                offset[axis] = dir;
                vec4 nb = fetch(pos + offset);
                int nb_id = int(round(nb.r));
                if (nb_id > 0 && nb.b > 0.5 && nb.b < 1.5) {
                    // Liquid neighbor — check if there's liquid above it (hydrostatic pressure)
                    vec4 nb_above = fetch(pos + offset + ivec3(0, 1, 0));
                    if (int(round(nb_above.r)) > 0) {
                        float hash = fract(sin(float(pos.x * 374761 + pos.y * 668265 + pos.z * 928114 + axis * 13) * 0.0001) * 43758.5453);
                        if (hash < 0.2 * u_dt) {
                            imageStore(u_dst, pos, vec4(float(nb_id), nb.g, nb.b, 0.0));
                            return;
                        }
                    }
                }
            }
        }

        // Gas diffusion: check all 6 neighbors for gas-phase atoms
        for (int axis = 0; axis < 3; axis++) {
            for (int dir = -1; dir <= 1; dir += 2) {
                ivec3 offset = ivec3(0);
                offset[axis] = dir;
                vec4 nb = fetch(pos + offset);
                int nb_id = int(round(nb.r));
                if (nb_id > 0 && nb.b > 1.5) {
                    float mass = get_mass(nb_id);
                    float diffuse_rate = 1.0 / max(sqrt(mass), 1.0);
                    float hash = fract(sin(float(pos.x * 374761 + pos.y * 668265 + pos.z * 928114 + axis * 13) * 0.0001) * 43758.5453);
                    if (hash < diffuse_rate * u_dt * 0.3) {
                        imageStore(u_dst, pos, vec4(float(nb_id), nb.g, nb.b, 0.0));
                        return;
                    }
                }
            }
        }

        // Stay vacuum, but conduct ambient temperature
        float ambient = u_param0;
        imageStore(u_dst, pos, vec4(0.0, mix(self_temp, ambient, 0.1 * u_dt), 0.0, 0.0));
        return;
    }

    // Non-vacuum cell: apply physics

    // 1. Thermal conduction — average temperature with neighbors, weighted by thermal conductivity
    float temp_sum = 0.0;
    float weight_sum = 0.0;
    for (int axis = 0; axis < 3; axis++) {
        for (int dir = -1; dir <= 1; dir += 2) {
            ivec3 offset = ivec3(0);
            offset[axis] = dir;
            vec4 nb = fetch(pos + offset);
            int nb_id = int(round(nb.r));
            float k_self = get_thermal(self_id);
            float k_nb = nb_id > 0 ? get_thermal(nb_id) : 0.01;
            float k_avg = (k_self + k_nb) * 0.5;
            // Normalize conductivity to reasonable rate
            float rate = k_avg * 0.001;
            temp_sum += nb.g * rate;
            weight_sum += rate;
        }
    }
    float new_temp = self_temp;
    if (weight_sum > 0.0) {
        new_temp = mix(self_temp, temp_sum / weight_sum, min(u_dt * 0.5, 0.4));
    }

    // 2. Phase transitions
    float new_phase = compute_phase(self_id, new_temp);

    // 3. Gravity and movement
    float new_vy = self_vy;
    float density = get_density(self_id);
    float gravity = u_param1;

    if (new_phase > 0.5) {
        // Liquid or gas: affected by gravity/buoyancy
        if (new_phase > 1.5) {
            // Gas: buoyancy upward
            new_vy += gravity * 0.5 * u_dt;
        } else {
            // Liquid: falls with gravity
            new_vy -= gravity * density * 0.1 * u_dt;
        }
        new_vy = clamp(new_vy, -5.0, 5.0);
        new_vy *= (1.0 - 0.1 * u_dt); // drag

        // Liquid falling: vacate if destination (below) is vacuum
        vec4 below = fetch(pos + ivec3(0, -1, 0));
        int below_id = int(round(below.r));
        if (new_vy < -MOVE_THRESHOLD && below_id == 0 && new_phase < 1.5) {
            imageStore(u_dst, pos, vec4(0.0, new_temp, 0.0, 0.0));
            return;
        }

        // Gas rising: vacate if destination (above) is vacuum
        vec4 above = fetch(pos + ivec3(0, 1, 0));
        int above_id = int(round(above.r));
        if (new_vy > MOVE_THRESHOLD && above_id == 0 && new_phase > 1.5) {
            imageStore(u_dst, pos, vec4(0.0, new_temp, 0.0, 0.0));
            return;
        }

        // Liquid spreading: vacate sideways if pressured from above
        if (new_phase > 0.5 && new_phase < 1.5) {
            vec4 my_above = fetch(pos + ivec3(0, 1, 0));
            if (int(round(my_above.r)) > 0) {
                // Under pressure — try to spread sideways into vacuum
                for (int axis = 0; axis < 3; axis += 2) {
                    for (int dir = -1; dir <= 1; dir += 2) {
                        ivec3 offset = ivec3(0);
                        offset[axis] = dir;
                        vec4 side = fetch(pos + offset);
                        if (int(round(side.r)) == 0) {
                            float hash = fract(sin(float(pos.x * 571 + pos.y * 887 + pos.z * 233 + axis * 17) * 0.0001) * 43758.5453);
                            if (hash < 0.2 * u_dt) {
                                imageStore(u_dst, pos, vec4(0.0, new_temp, 0.0, 0.0));
                                return;
                            }
                        }
                    }
                }
            }
        }
    } else {
        new_vy = 0.0; // Solids don't move
    }

    // 4. Chemical reactions with neighbors
    float react_mult = u_param2;
    for (int axis = 0; axis < 3; axis++) {
        for (int dir = -1; dir <= 1; dir += 2) {
            ivec3 offset = ivec3(0);
            offset[axis] = dir;
            vec4 nb = fetch(pos + offset);
            int nb_id = int(round(nb.r));
            if (nb_id == 0 || nb_id == self_id) continue;

            // Electronegativity difference drives reaction probability
            float en_self = get_eneg(self_id);
            float en_nb = get_eneg(nb_id);
            float en_diff = abs(en_self - en_nb);

            // Higher EN difference = more likely to react
            // Also need sufficient temperature (activation energy ~ mass-weighted)
            float activation = (get_mass(self_id) + get_mass(nb_id)) * 2.0;
            if (en_diff > 0.8 && new_temp > activation * 0.1) {
                float hash = fract(sin(float(pos.x * 127 + pos.y * 311 + pos.z * 523 + axis * 7) * 0.0001) * 43758.5453);
                float react_prob = en_diff * 0.02 * react_mult * u_dt;
                if (hash < react_prob) {
                    // Reaction! Release energy (exothermic if large EN diff)
                    float energy = en_diff * 200.0;
                    new_temp += energy;
                    // Switch phase (reaction heat may cause phase change)
                    new_phase = compute_phase(self_id, new_temp);
                }
            }
        }
    }

    // 5. Radiative cooling toward ambient temperature.
    // Without this the chemical reactions deposit en_diff*200 K per
    // event with no sink, so temperature accumulates indefinitely and
    // pegs at the 10000 K cap (audit: max=10000 forever). Newton's
    // law of cooling with a small linear coefficient lets the system
    // dissipate accumulated reaction heat to the environment, plus a
    // quadratic term provides extra cooling at the high end so
    // blackbody-hot cells return to manageable temperatures.
    float T_amb = u_param0;  // ambient temperature slider
    float dT = new_temp - T_amb;
    new_temp -= 0.05 * dT * u_dt;             // linear (~Newton)
    if (dT > 0.0) {
        new_temp -= 1e-5 * dT * dT * u_dt;    // quadratic boost at high T
    }

    new_temp = clamp(new_temp, -273.15, 10000.0);
    imageStore(u_dst, pos, vec4(float(self_id), new_temp, new_phase, new_vy));
}
