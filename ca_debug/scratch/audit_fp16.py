"""FP16 precision audit.

Runs each candidate rule for N steps under both FP32 and FP16 storage
and reports relative drift (mean abs diff and max abs diff). Results
inform which rules deserve a `precision: 'fp32'` opt-out in their preset.
"""
import os, time
os.environ.setdefault('MODERNGL_REQUIRE', 'standalone')
import numpy as np
import simulator as S
from OpenGL import GL


CANDIDATES = [
    # (rule, size, steps)  — keep size small to bench many rules
    ('game_of_life_3d',        128, 80),
    ('445_rule',               128, 80),
    ('smoothlife_3d',          128, 80),
    ('reaction_diffusion_3d',  128, 80),
    ('lenia_3d',               128, 80),
    ('wave_3d',                128, 80),
    ('gray_scott_3d',          128, 80),
    ('bz_3d',                  128, 80),
    ('crystal_growth',         128, 80),
    ('em_yee_E_3d',            128, 80),
    ('schrodinger_3d',          96, 50),
]


def run(rule, size, steps, force_precision):
    try:
        sim = S.Simulator(size=size, rule=rule, headless=True,
                          force_precision=force_precision)
    except KeyError:
        return None  # rule not in this build
    sim.seed = 42
    sim._reset()
    for _ in range(steps):
        sim._step_sim()
    src = sim.tex_a if sim.ping == 0 else sim.tex_b
    arr = np.frombuffer(src.read(), dtype=sim._tex_np_dtype).reshape(size, size, size, 4)
    out = arr.astype(np.float32).copy()  # promote for diff math
    sim.ctx.release()
    return out


print(f"{'rule':<25} {'size':>5} {'steps':>5}  {'mean Δ':>10}  {'max Δ':>10}  {'mean rel':>10}  verdict")
print('-' * 100)
for rule, size, steps in CANDIDATES:
    a32 = run(rule, size, steps, 'fp32')
    a16 = run(rule, size, steps, 'fp16')
    if a32 is None or a16 is None:
        print(f"{rule:<25} (skipped — not in build)")
        continue
    diff = np.abs(a32 - a16)
    mean_d = float(diff.mean())
    max_d = float(diff.max())
    # Relative diff on cells with non-zero magnitude
    mag32 = np.abs(a32)
    rel = diff / (mag32 + 1e-6)
    mean_rel = float(rel[mag32 > 1e-3].mean()) if (mag32 > 1e-3).any() else 0.0
    verdict = 'OK' if max_d < 0.05 else ('WARN' if max_d < 0.5 else 'BAD')
    print(f"{rule:<25} {size:>5} {steps:>5}  {mean_d:10.4g}  {max_d:10.4g}  {mean_rel:10.4g}  {verdict}")
