"""Gray-Scott reaction-diffusion on the FCC lattice.

Two coupled species stored in the R (U, substrate) and G (V, catalyst)
channels of an ``FCCField``:

    dU/dt = Du * lap(U) - U*V^2 + F*(1-U)
    dV/dt = Dv * lap(V) + U*V^2 - (F+k)*V

The discrete Laplacian uses the 12 nearest FCC neighbours. For unit
nearest-neighbour distance and a cubic-symmetric lattice,
``sum_i d_i (x) d_i = 4 I``, so

    lap(f) ~= 0.5 * (sum_{nn} f_nn - 12 * f)

(See `lattice.py` for the NN list.)

The compute shader is wrapped by ``GrayScottFCC``; one ``step()`` call
dispatches one Euler step and swaps the field's ping-pong textures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import moderngl

from lattice import FCC
from fcc_field import FCCField, FCCFieldShape


# ---------------------------------------------------------------------------
# Compute shader
# ---------------------------------------------------------------------------

_COMPUTE_HEADER = """\
#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

/*__LATTICE_HEADER__*/

layout(binding = 0)              uniform sampler3D u_src;
layout(rgba32f, binding = 1)     uniform image3D  u_dst;

uniform ivec3 u_dims;
uniform float u_F;
uniform float u_k;
uniform float u_Du;
uniform float u_Dv;
uniform float u_dt;

// Periodic wrap, since the cull / render path already assumes periodic.
ivec3 wrap(ivec3 p) {
    return ((p % u_dims) + u_dims) % u_dims;
}

vec2 sample_uv(ivec3 p) {
    return texelFetch(u_src, wrap(p), 0).rg;
}

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(pos, u_dims))) return;

    vec2 c = sample_uv(pos);
    float U = c.r;
    float V = c.g;

    // 12-NN sum.
    vec2 nn_sum = vec2(0.0);
    for (int i = 0; i < LATTICE_N_NEIGHBOURS; ++i) {
        nn_sum += sample_uv(pos + LATTICE_NEIGHBOURS[i]);
    }

    // FCC 12-NN Laplacian: 0.5 * (sum - 12*center).
    vec2 lap = 0.5 * (nn_sum - float(LATTICE_N_NEIGHBOURS) * c);

    // CFL-safe dt clamp (explicit Euler on the 12-NN stencil; same
    // 1/(2*Z) bound as a 6-NN cubic stencil since the lattice sum of
    // d_i (x) d_i has the same isotropic norm).
    float D_max    = max(u_Du, u_Dv);
    float dt_limit = 0.9 / (0.5 * float(LATTICE_N_NEIGHBOURS) * max(D_max, 1e-8));
    float dt_eff   = min(u_dt, dt_limit);

    float uvv = U * V * V;
    float dU  = u_Du * lap.x - uvv + u_F * (1.0 - U);
    float dV  = u_Dv * lap.y + uvv - (u_F + u_k) * V;

    float new_U = clamp(U + dU * dt_eff, 0.0, 1.0);
    float new_V = clamp(V + dV * dt_eff, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(new_U, new_V, 0.0, 1.0));
}
"""


# ---------------------------------------------------------------------------
# Parameters (Pearson's classic regimes for the cubic stencil; tuned
# slightly for the 12-NN FCC stencil but the same neighbourhood as a
# cubic 6-NN, so the parameter map is very similar).
# ---------------------------------------------------------------------------

@dataclass
class GrayScottParams:
    F:  float = 0.054
    k:  float = 0.063
    Du: float = 0.16
    Dv: float = 0.08
    dt: float = 1.0


# Curated regimes (F, k) (Du and Dv unchanged unless noted).
REGIMES = {
    "spots":   GrayScottParams(F=0.030, k=0.062),    # static round spots
    "mitosis": GrayScottParams(F=0.054, k=0.063),    # dividing spots
    "worms":   GrayScottParams(F=0.046, k=0.063),    # labyrinthine
    "holes":   GrayScottParams(F=0.039, k=0.058),    # negative spots
}


# ---------------------------------------------------------------------------
# Rule wrapper
# ---------------------------------------------------------------------------

class GrayScottFCC:
    """One compute pass per ``step()``. Ping-pongs an external ``FCCField``."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        src = _COMPUTE_HEADER.replace("/*__LATTICE_HEADER__*/", FCC.glsl_header())
        self.prog = ctx.compute_shader(src)

    def step(self, field: FCCField, params: GrayScottParams) -> None:
        shape = field.shape
        self.prog['u_dims'].value = (shape.Na, shape.Nb, shape.Nc)
        self.prog['u_F'].value    = params.F
        self.prog['u_k'].value    = params.k
        self.prog['u_Du'].value   = params.Du
        self.prog['u_Dv'].value   = params.Dv
        self.prog['u_dt'].value   = params.dt

        field.current.use(location=0)
        field.other.bind_to_image(1, read=False, write=True)
        self.prog['u_src'].value = 0

        gx = (shape.Na + 3) // 4
        gy = (shape.Nb + 3) // 4
        gz = (shape.Nc + 3) // 4
        self.prog.run(gx, gy, gz)
        self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT
                                | moderngl.TEXTURE_FETCH_BARRIER_BIT)

        field.swap()

    def release(self) -> None:
        self.prog.release()


# ---------------------------------------------------------------------------
# Field seeding
# ---------------------------------------------------------------------------

def seed_random_clusters(field: FCCField, *,
                         n_seeds: int = 24,
                         seed_radius: int = 3,
                         rng_seed: int = 0) -> None:
    """Initialise U=1 everywhere; sprinkle V=1 in a few small random
    clusters. This is the classic Gray-Scott initial condition that
    breaks the U=1, V=0 trivial fixed point."""
    shape = field.shape
    arr = np.zeros(shape.numpy_shape(), dtype=np.float32)
    arr[..., 0] = 1.0    # U = 1 everywhere
    arr[..., 3] = 1.0    # alpha (for RGB_CHANNELS rendering)

    rng = np.random.default_rng(rng_seed)
    for _ in range(n_seeds):
        a0 = int(rng.integers(0, shape.Na))
        b0 = int(rng.integers(0, shape.Nb))
        c0 = int(rng.integers(0, shape.Nc))
        for da in range(-seed_radius, seed_radius + 1):
            for db in range(-seed_radius, seed_radius + 1):
                for dc in range(-seed_radius, seed_radius + 1):
                    if da * da + db * db + dc * dc > seed_radius * seed_radius:
                        continue
                    a = (a0 + da) % shape.Na
                    b = (b0 + db) % shape.Nb
                    c = (c0 + dc) % shape.Nc
                    arr[c, b, a, 1] = 1.0    # V = 1
                    arr[c, b, a, 0] = 0.5    # consume some U

    field.upload(arr)


# ---------------------------------------------------------------------------
# Stand-alone check: GLSL must compile and a few steps must change V.
# ---------------------------------------------------------------------------

def _self_check() -> None:
    print("[fcc_rule_gray_scott] booting standalone context...")
    ctx = moderngl.create_standalone_context(require=430)

    shape = FCCFieldShape(32, 32, 32, channels=4)
    field = FCCField(ctx, shape, dtype=np.dtype('float32'), linear_filter=False)
    seed_random_clusters(field, n_seeds=8, seed_radius=3, rng_seed=1)

    rule = GrayScottFCC(ctx)
    before = np.frombuffer(field.current.read(), dtype=np.float32) \
               .reshape(shape.numpy_shape())
    print(f"  initial: U_mean={before[..., 0].mean():.4f}  "
          f"V_mean={before[..., 1].mean():.4f}  "
          f"V_max={before[..., 1].max():.4f}")

    params = REGIMES["mitosis"]
    for _ in range(200):
        rule.step(field, params)

    after = np.frombuffer(field.current.read(), dtype=np.float32) \
              .reshape(shape.numpy_shape())
    print(f"  after 200 steps: U_mean={after[..., 0].mean():.4f}  "
          f"V_mean={after[..., 1].mean():.4f}  "
          f"V_max={after[..., 1].max():.4f}")

    # Sanity: V must have spread but not exploded.
    v_mean = float(after[..., 1].mean())
    v_max  = float(after[..., 1].max())
    assert 0.005 < v_mean < 0.6, f"V mean out of range: {v_mean}"
    assert 0.2   < v_max  <= 1.0, f"V max out of range: {v_max}"
    # And U must have been consumed somewhere but not driven to zero
    # everywhere.
    u_mean = float(after[..., 0].mean())
    assert 0.2 < u_mean < 0.99, f"U mean out of range: {u_mean}"

    print("[fcc_rule_gray_scott] PASS")


if __name__ == "__main__":
    _self_check()
