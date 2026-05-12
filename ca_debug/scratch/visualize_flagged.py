"""For each flagged CA, dump a 2D slice as ASCII art + key metrics
to quickly tell apart real-but-mislabeled vs really-broken."""

import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from test_harness import create_headless_context, HeadlessRunner
from simulator import RULE_PRESETS

FLAGGED = {
    'EXTINCT': ['crystal_dla', 'fire', 'fracture', 'lichen'],
    'FROZEN': ['mycelium'],
    'NOISE_CUBE': ['445_rule', 'bz_excitable', 'bz_turbulence', 'element_ca',
                   'game_of_life_3d', 'lenia_3d', 'nucleation',
                   'phase_separation', 'smoothlife_3d'],
}

def ascii_slice(field, w=32):
    """Render a 32x32 ASCII slice of the central XY plane."""
    sz = field.shape[0]
    sl = field[sz//2]
    # Downsample to w
    step = max(1, sz // w)
    sl = sl[::step, ::step]
    fmin, fmax = float(sl.min()), float(sl.max())
    rng = fmax - fmin if fmax > fmin else 1.0
    chars = ' .:-=+*#%@'
    rows = []
    for r in sl:
        row = ''.join(chars[min(len(chars)-1, max(0, int((v-fmin)/rng*(len(chars)-1))))] for v in r)
        rows.append(row)
    return rows, fmin, fmax

def ax_corrs(field):
    return [float(np.corrcoef(field.ravel(), np.roll(field, 1, axis=a).ravel())[0,1]) for a in range(3)]

def main():
    ctx = create_headless_context()
    if isinstance(ctx, tuple): _, ctx = ctx
    for verdict, rules in FLAGGED.items():
        print(f'\n{"="*72}\n  {verdict}\n{"="*72}')
        for r in rules:
            preset = RULE_PRESETS[r]
            vis = preset.get('vis_default', 0)
            print(f'\n--- {r}  ({preset.get("label","?")}) ---')
            print(f'  Description: {preset.get("description","")[:110]}')
            print(f'  Default params: {preset.get("params", {})}')
            runner = HeadlessRunner(ctx, r, size=64, seed=42)
            for _ in range(1500): runner.step()
            g = runner.read_grid()
            field = g[..., vis].astype(np.float32)
            slc, fmin, fmax = ascii_slice(field, w=32)
            print(f'  vis_channel={vis}  range=[{fmin:.3f}, {fmax:.3f}]  std={field.std():.4f}')
            corrs = ax_corrs(field)
            print(f'  adj-correlations (per axis): {[round(c,2) for c in corrs]}')
            # All channel summary
            for c in range(4):
                v = g[..., c]
                print(f'    ch{c}: mean={float(v.mean()):+.3f} std={float(v.std()):.3f} min={float(v.min()):+.2f} max={float(v.max()):+.2f}')
            print('  central XY slice:')
            for row in slc:
                print(f'    {row}')
            runner.release()

if __name__ == '__main__':
    main()
