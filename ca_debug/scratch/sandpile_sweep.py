#!/usr/bin/env python3
"""Sweep sandpile vis_range to find a setting that shows structure
   instead of a uniform cube. Renders one frame per setting."""
import os
os.environ.setdefault('CA_DISABLE_PRESET_OVERRIDES', '1')
import numpy as np
from PIL import Image
from simulator import Simulator, RULE_PRESETS

settings = [(0,12), (4,12), (5,12), (6,12), (6,10), (7,12)]
out = 'ghost_cube_renders'
os.makedirs(out, exist_ok=True)

for lo, hi in settings:
    RULE_PRESETS['sandpile_3d']['vis_range'] = (float(lo), float(hi))
    sim = Simulator(size=64, rule='sandpile_3d', headless=True)
    for _ in range(150): sim._step_sim()
    sim._render(); sim.ctx.finish()
    raw = sim._cr_output_tex.read()
    img = np.frombuffer(raw, dtype=np.uint8).reshape(sim.height, sim.width, 4)
    rgb = img[:, :, :3].astype(np.float32) / 255.0
    lum = rgb.mean(axis=-1)
    fg = lum > 0.05
    fg_lum = lum[fg]
    print(f'vis_range=({lo:2d},{hi:2d})  '
          f'fg_frac={fg.mean():.3f}  '
          f'fg_lum_mean={fg_lum.mean() if fg.any() else 0:.3f}  '
          f'fg_lum_std={fg_lum.std() if fg.any() else 0:.3f}')
    Image.fromarray(img[:, :, :3]).save(f'{out}/sandpile_lo{lo}_hi{hi}.png')
    sim.close()
print('saved to', out)
