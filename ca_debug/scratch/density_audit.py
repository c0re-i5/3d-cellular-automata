"""Audit all rules' density distribution under their declared vis_mode.

Mirrors the GPU shader's density computation in volume_view.glsl:
  mode 0 (DENSITY)     : d = clip((ch[k] - lo)/(hi-lo), 0, 1)  (or |ch|, ch*0.5+0.5)
  mode 1 (RGB_CHANNELS): d = length(clip((rgb-lo)/(hi-lo))) / sqrt(3)
  mode 2 (HSV_PHASE)   : d = clip((|ch0,ch1| - aux_lo)/(aux_hi-aux_lo))   (or inverted)
  mode 3 (BIPOLAR)     : d = |clip(ch[k]*aux_scale, -1, 1)|
  mode 4 (RGBA_BLEND)  : d = clip((ch3 - aux_lo)/(aux_hi-aux_lo))

Flags rules where:
  - SAT  : density mean > 0.55 AND fraction>0.5 > 60%  → solid cube
  - DARK : density max < 0.05  → entirely invisible
  - FLAT : std(density) < 0.04 AND mean > 0.15  → uniform haze
"""
import os; os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import OpenGL.GL as GL; GL.glMemoryBarrier = lambda *a, **k: None
import numpy as np, simulator as S
import traceback

VIS_MODE_MAP = {None: 0, 'density':0, 'rgb_channels':1, 'rgb':1,
                'hsv_phase':2, 'phase':2, 'bipolar':3, 'signed':3,
                'rgba_blend':4, 'rgba':4}

def density_for(field, preset):
    """Replicate volume_view.glsl density computation."""
    mode_name = preset.get('vis_mode')
    if isinstance(mode_name, int):
        mode = mode_name
    else:
        mode = VIS_MODE_MAP.get(mode_name, 0)
    vis_lo, vis_hi = preset.get('vis_range', (0.0, 1.0))
    aux_lo, aux_hi = preset.get('vis_aux_range', preset.get('aux_range', (0.0, 1.0)))
    ch = int(preset.get('vis_default', 0))
    use_abs = preset.get('vis_abs', False)
    scale = 1.0/(vis_hi-vis_lo) if vis_hi>vis_lo else 1.0
    aux_scale = 1.0/(aux_hi-aux_lo) if aux_hi>aux_lo else 1.0
    if mode == 1:
        rgb = np.clip((field[..., :3] - vis_lo) * scale, 0, 1)
        d = np.clip(np.linalg.norm(rgb, axis=-1) / np.sqrt(3), 0, 1)
    elif mode == 2:
        a, b = field[..., 0], field[..., 1]
        mag = np.sqrt(a*a + b*b)
        if aux_hi >= aux_lo:
            d = np.clip((mag - aux_lo) * aux_scale, 0, 1)
        else:
            inv = 1.0/(aux_lo - aux_hi)
            d = np.clip(1.0 - (mag - aux_hi) * inv, 0, 1)
    elif mode == 3:
        v = field[..., ch]
        if aux_hi > aux_lo:
            d = np.abs(np.clip(v * aux_scale, -1, 1))
        else:
            d = np.abs(np.clip(v, -1, 1))
    elif mode == 4:
        d = np.clip((field[..., 3] - aux_lo) * aux_scale, 0, 1)
    else:  # mode 0 density
        v = field[..., ch].copy()
        if use_abs == 1 or use_abs is True: v = np.abs(v)
        elif use_abs == 2: v = v*0.5 + 0.5
        d = np.clip((v - vis_lo) * scale, 0, 1)
    return d

def audit_rule(rule, size=48, steps=400):
    sim = S.Simulator(rule=rule, size=size, headless=True)
    p = sim.preset
    if p.get('audit_skip'): return None
    while sim.step_count < steps:
        sim._step_sim()
    src = sim.tex_a if sim.ping == 0 else sim.tex_b
    W, H, D = sim.W, sim.H, sim.D
    f = np.frombuffer(src.read(), dtype=np.float32).reshape(D, H, W, 4)
    d = density_for(f, p)
    return {
        'mean': float(d.mean()), 'std': float(d.std()),
        'min': float(d.min()), 'max': float(d.max()),
        'gt05': float((d>0.5).mean()),
        'gt01': float((d>0.1).mean()),
        'lt005': float((d<0.05).mean()),
        'mode': p.get('vis_mode', 'density'),
        'vis_range': p.get('vis_range', (0,1)),
        'vis_default': p.get('vis_default', 0),
        'aux_range': p.get('vis_aux_range', None),
    }

def classify(s):
    flags = []
    if s['mean'] > 0.55 and s['gt05'] > 0.6: flags.append('SAT')
    if s['max'] < 0.05: flags.append('DARK')
    if s['std'] < 0.04 and s['mean'] > 0.15: flags.append('FLAT')
    if s['gt05'] > 0.85: flags.append('SOLID')
    return flags

# Audit all real rules (skip aliases / metadata-less)
rules = [r for r in S.RULE_PRESETS if isinstance(S.RULE_PRESETS[r], dict)
         and 'shader' in S.RULE_PRESETS[r] or 'passes' in S.RULE_PRESETS[r]]
rules = sorted(set(rules))
print(f"auditing {len(rules)} rules at size=48, t=400")
results = {}
for r in rules:
    try:
        s = audit_rule(r)
        if s is None: continue
        results[r] = s
    except Exception as e:
        print(f"  {r}: ERROR {e}")

print("\n=== FLAGGED ===")
for r, s in sorted(results.items()):
    fl = classify(s)
    if fl:
        print(f"  {r:38s} {fl}  mode={s['mode']!r:18s} "
              f"vr={s['vis_range']} ch={s['vis_default']}  "
              f"d:mean={s['mean']:.2f} std={s['std']:.2f} "
              f"gt05={s['gt05']*100:.0f}%")

print("\n=== HEALTHY (sample) ===")
healthy = [(r,s) for r,s in results.items() if not classify(s)]
for r, s in sorted(healthy)[:6]:
    print(f"  {r:38s} mode={s['mode']!r:18s} d:mean={s['mean']:.2f} std={s['std']:.2f}")
print(f"... +{max(0,len(healthy)-6)} more healthy")

print(f"\nTotal: {len(results)} rules, {len([1 for r,s in results.items() if classify(s)])} flagged")
