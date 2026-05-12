import sys, os, numpy as np
from PIL import Image
from simulator import Simulator
n = sys.argv[1]
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 80
CHECKPOINTS = [10, 40, 100, 200, 400]
sim = Simulator(size=SIZE, rule=n, headless=True)
frames = []
last = 0
for t in CHECKPOINTS:
    for _ in range(t - last):
        sim._step_sim()
    last = t
    sim._render(); sim.ctx.finish()
    from OpenGL import GL as _GL
    _GL.glReadBuffer(_GL.GL_BACK)
    raw = _GL.glReadPixels(0, 0, sim.width, sim.height, _GL.GL_RGB, _GL.GL_UNSIGNED_BYTE)
    img = np.frombuffer(raw, dtype=np.uint8).reshape(sim.height, sim.width, 3)[::-1].copy()
    frames.append(img)
strip = np.concatenate(frames, axis=1)
Image.fromarray(strip).save(f'flagship_strips/{n}.png')
nb = [int((f.sum(axis=2)>20).sum()) for f in frames]
print(f'  {n:35s} nonblack@steps={nb}', flush=True)
