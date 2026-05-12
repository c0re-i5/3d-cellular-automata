"""Deep-debug probe for particle-using rules."""
import os; os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import OpenGL.GL as GL; GL.glMemoryBarrier = lambda *a, **k: None
import numpy as np
import simulator as S

def read_particles(sim):
    """Read the SSBO; layout is (count, 8) float32: pos.xyz life vel.xyz mass."""
    raw = sim.particle_ssbo.read()
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, 8)
    return arr  # rows: [px,py,pz, life, vx,vy,vz, mass]

def read_field(sim):
    src = sim.tex_a if sim.ping == 0 else sim.tex_b
    N = sim.size
    return np.frombuffer(src.read(), dtype=np.float32).reshape(N, N, N, 4)

def read_deposit(sim):
    """Read the deposit SSBO (r32f, W*H*D scalars)."""
    if getattr(sim, 'deposit_tex', None) is None:
        return None
    raw = sim.deposit_tex.read()
    return np.frombuffer(raw, dtype=np.float32).reshape(sim.D, sim.H, sim.W)

def stats(name, arr):
    a = arr.astype(np.float64)
    return (f"  {name}: shape={arr.shape} min={a.min():.4g} "
            f"max={a.max():.4g} mean={a.mean():.4g} std={a.std():.4g}")

def probe(rule, size=64, steps=(0, 1, 50, 200, 500)):
    print(f"\n=== {rule} ===")
    sim = S.Simulator(rule=rule, size=size, headless=True)
    p = sim.preset
    print(f"  preset force={p.get('particle_force')!r} init={p.get('particle_init')!r}"
          f" count={p.get('particle_count')} dt={p.get('particle_dt')}"
          f" drag={p.get('particle_drag')} life_decay={p.get('particle_life_decay')}"
          f" color_mode={p.get('particle_color_mode')!r}")
    print(f"  deposit: strength={p.get('particle_deposit_strength', 0)}"
          f" amount={p.get('particle_deposit_amount', 0)}"
          f" radius={p.get('particle_deposit_radius', 0)}"
          f" channel={p.get('particle_deposit_channel', None)}")
    print(f"  validated color_mode → mode_id={S.PARTICLE_COLOR_MODES.get(p.get('particle_color_mode','speed'),'MISSING')}")
    print(f"  validated force → mode_id={S.PARTICLE_FORCE_MODES.get(p.get('particle_force','none'),'MISSING')}")
    last_t = 0
    for t in steps:
        while sim.step_count < t:
            sim._step_sim()
        last_t = t
        parts = read_particles(sim)
        pos, life, vel = parts[:, 0:3], parts[:, 3], parts[:, 4:7]
        speed = np.linalg.norm(vel, axis=1)
        alive = life > 0
        n_alive = int(alive.sum())
        out_of_box = ((pos < 0) | (pos >= sim.size)).any(axis=1).sum()
        # Cluster metric: stddev of pos (lower = clustered, higher = spread)
        spread = pos[alive].std(axis=0).mean() if n_alive > 0 else 0
        print(f"  t={t:4d}: alive={n_alive}/{len(parts)} OOB={out_of_box} "
              f"|v|=[{speed.min():.3g},{speed.max():.3g}] mean={speed.mean():.3g}  "
              f"spread={spread:.2f}  life=[{life.min():.3g},{life.max():.3g}]")
        # Field state
        field = read_field(sim)
        ch0 = field[..., 0]
        print(f"    field ch0: [{ch0.min():.3g}, {ch0.max():.3g}] mean={ch0.mean():.3g}")
        # Deposit field
        dep = read_deposit(sim)
        if dep is not None:
            n_nonzero = int((dep != 0).sum())
            print(f"    deposit:   [{dep.min():.3g}, {dep.max():.3g}] mean={dep.mean():.3g} "
                  f"nonzero_voxels={n_nonzero}/{dep.size} "
                  f"({100*n_nonzero/dep.size:.1f}%)")

for r in ['particle_lenia', 'flagship_neural_swarm',
          'flagship_unison_flock', 'flagship_galaxy_dust']:
    try:
        probe(r)
    except Exception as e:
        import traceback; traceback.print_exc()
