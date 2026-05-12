"""Test single-preset variations to see which boundary/undercooling combo
gives the longest-lasting morphology phase before the crystal saturates.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from test_harness import create_headless_context, HeadlessRunner
from simulator import RULE_PRESETS

import math

def shape(grid, threshold=0.5):
    phi = grid[..., 0]
    sz = max(phi.shape)
    n = int((phi > threshold).sum())
    if n < 5:
        return dict(af=0.0, sph=float('nan'), bbf=float('nan'),
                    ifr=float('nan'), oct=float('nan'), umean=float(grid[...,1].mean()))
    z, y, x = np.nonzero(phi > threshold)
    cx, cy, cz = x.mean(), y.mean(), z.mean()
    rg = float(np.sqrt(((x-cx)**2+(y-cy)**2+(z-cz)**2).mean()))
    R = (3.0*n/(4.0*math.pi))**(1/3)
    sph = rg / (math.sqrt(3/5)*R)
    bbox = max(1,(x.max()-x.min()+1)*(y.max()-y.min()+1)*(z.max()-z.min()+1))
    bbf = n / bbox
    iface = int(((phi>0.05) & (phi<0.95)).sum())
    bulk = int((phi>=0.95).sum())
    ifr = iface/(iface+bulk) if (iface+bulk)>0 else float('nan')
    axis = max(np.abs(x-cx).max(), np.abs(y-cy).max(), np.abs(z-cz).max())
    diag = (np.abs(x-cx)+np.abs(y-cy)+np.abs(z-cz)).max()/math.sqrt(3.0)
    oct_ = axis/diag if diag>0 else float('nan')
    return dict(af=n/phi.size, sph=sph, bbf=bbf, ifr=ifr, oct=oct_,
                umean=float(grid[...,1].mean()))


def run(ctx, rule, size, steps, seed, *, override=None, sample=None):
    """Run a preset with optional override of params and boundary."""
    if sample is None:
        sample = sorted(set(np.unique(np.round(np.geomspace(20, steps, 8)).astype(int)).tolist()) | {steps})

    # Patch RULE_PRESETS in place for the duration of the run.
    import simulator
    orig = simulator.RULE_PRESETS[rule].copy()
    if override:
        if 'boundary' in override:
            simulator.RULE_PRESETS[rule]['boundary'] = override['boundary']
        if 'params' in override:
            simulator.RULE_PRESETS[rule] = simulator.RULE_PRESETS[rule].copy()
            simulator.RULE_PRESETS[rule]['params'] = {**simulator.RULE_PRESETS[rule]['params'], **override['params']}
    try:
        runner = HeadlessRunner(ctx, rule, size=size, seed=seed)
        snaps = []
        for i in range(1, steps+1):
            runner.step()
            if i in sample:
                g = runner.read_grid()
                snaps.append((i, shape(g)))
        runner.release()
        return snaps
    finally:
        simulator.RULE_PRESETS[rule] = orig


def main():
    win, ctx = create_headless_context()
    try:
        rule = 'crystal_cubic'
        sz = 64
        steps = 600
        seed = 42

        configs = [
            ('cubic (default)',  None),
        ]
        for srule in ['crystal_octahedral','crystal_dendritic','crystal_snowflake','crystal_hopper','crystal_morphology','crystal_polycrystal']:
            configs.append((f'{srule}', {'rule': srule}))
        for label, ovr in configs:
            print(f'\n=== {label} ===')
            t0 = time.time()
            run_rule = ovr.pop('rule') if ovr and 'rule' in ovr else rule
            snaps = run(ctx, run_rule, sz, steps, seed, override=ovr)
            print(f'    ({time.time()-t0:.1f}s)')
            for step, s in snaps:
                marker = ''
                if s['af'] > 0.99: marker = '  <-- SATURATED'
                if 0.05 < s['af'] < 0.30 and s['sph'] > 1.05: marker = '  ** morphology'
                print(f'    step={step:5d}  af={s["af"]:.3f}  sph={s["sph"]:.2f}  '
                      f'bbf={s["bbf"]:.3f}  ifr={s["ifr"]:.3f}  oct={s["oct"]:.2f}  '
                      f'u={s["umean"]:.3f}{marker}')
    finally:
        try: ctx.release()
        except: pass

if __name__ == '__main__':
    main()
