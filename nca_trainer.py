"""Offline trainer for the 3D Neural Cellular Automaton preset.

This script trains the same MLP architecture that runs inside the
`nca_3d` GPU shader (see simulator.py CA_RULES['nca_3d']) and exports
its weights as a `.npz` blob the simulator can load at runtime via
`preset['weights_path']` or the `--nca-weights` CLI flag.

Architecture (must stay byte-identical with the shader):
  • State        : 4 channels per voxel (channel 0 is the visible "alpha")
  • Perception   : 16 features = identity + central-diff ∂x/∂y/∂z (×4 ch)
  • MLP          : 16 → 32 (ReLU) → 4 (linear)
  • Update       : new = clip(state + dt * fire * delta, ±clip)
  • Fire mask    : per-cell Bernoulli(fire_rate) — applied each step
  • Boundary     : clamped (matches preset["boundary"] = "clamped")

Training regime (Mordvintsev et al., "Growing Neural Cellular Automata",
Distill 2020 — extended to 3D):
  • Pool of N initial states; each iteration, sample a batch, run T
    randomised steps, MSE-loss on channel 0 vs the binary target shape,
    backprop, re-seed the worst sample to keep the pool from collapsing.
  • A fraction of samples are "damaged" by zeroing a random sphere —
    encourages regenerative attractors.

Usage:
  python nca_trainer.py --target sphere --size 32 --steps 6000 \
      --out trained_nca/sphere_32.npz

Run `python nca_trainer.py --help` for all options.
"""

from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Architecture constants — MUST match simulator.py CA_RULES['nca_3d'] ──
NCA_IN  = 16
NCA_HID = 32
NCA_OUT = 4
NCA_CH  = 4   # state channels per voxel


# =====================================================================
# Target shapes
# =====================================================================

def make_target(name: str, size: int, device) -> torch.Tensor:
    """Return a (size, size, size) float tensor in [0, 1] — the desired
    channel-0 occupancy for the trained NCA's stable state."""
    g = torch.arange(size, device=device, dtype=torch.float32)
    z, y, x = torch.meshgrid(g, g, g, indexing='ij')
    cx = cy = cz = (size - 1) / 2.0
    rx, ry, rz = x - cx, y - cy, z - cz
    r = torch.sqrt(rx*rx + ry*ry + rz*rz)
    R = size * 0.30  # nominal "outer" radius

    if name == 'sphere':
        # Soft-edge ball: 1 inside, 0 outside, ~1-voxel transition.
        return torch.sigmoid((R - r) * 2.0)
    if name == 'cube':
        d = torch.maximum(torch.maximum(rx.abs(), ry.abs()), rz.abs())
        return torch.sigmoid((R - d) * 2.0)
    if name == 'torus':
        # Torus in the xy-plane: major radius R, minor r0.
        r0 = size * 0.10
        ring = torch.sqrt(rx*rx + ry*ry) - R
        d = torch.sqrt(ring*ring + rz*rz)
        return torch.sigmoid((r0 - d) * 2.0)
    if name == 'octahedron':
        d = rx.abs() + ry.abs() + rz.abs()
        return torch.sigmoid((R * 1.3 - d) * 2.0)
    if name == 'shell':
        # Hollow sphere — tests that the network can stop growing inwards.
        inner = R * 0.6
        outer = R
        return torch.sigmoid((outer - r) * 2.0) * torch.sigmoid((r - inner) * 2.0)
    raise ValueError(f"unknown target: {name}")


# =====================================================================
# NCA model
# =====================================================================

class NCA3D(torch.nn.Module):
    """Per-cell MLP applied as 3D convolutions for parallelism.

    Equivalence with the shader:
      • Identity feature      ↔ a 1x1x1 conv reading the cell value.
      • Sobel-like ∂x feature ↔ central-difference filter [-0.5, 0, 0.5].
      The 16 features are concatenated channel-wise, then a 1x1x1 conv
      stack (= per-cell MLP) emits the 4-channel delta.

    All convs are implemented as F.conv3d for speed on GPU; the
    parameters that matter (W1, b1, W2, b2) are stored as Linear layers
    so the export step can reshape them to the exact shader layout.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(NCA_IN, NCA_HID)
        self.fc2 = torch.nn.Linear(NCA_HID, NCA_OUT)
        # Match the shader's "0.1×Xavier" init on the output layer so the
        # initial network is near-identity (no exploding deltas) and the
        # optimiser gets a calm starting point.  W1 is left at PyTorch's
        # default Kaiming-uniform which is reasonable for ReLU.
        with torch.no_grad():
            self.fc2.weight.mul_(0.1)
            self.fc2.bias.zero_()

        # Build the central-difference kernels once and register as
        # buffers (move with .to(device), don't get gradients).
        # Each filter is a depth-wise 3x3x3 conv applied to the 4-channel
        # state — output has 4 channels (one ∂axis per input channel).
        kx = torch.zeros(1, 1, 3, 3, 3)
        kx[0, 0, 1, 1, 0] = -0.5
        kx[0, 0, 1, 1, 2] =  0.5
        ky = torch.zeros(1, 1, 3, 3, 3)
        ky[0, 0, 1, 0, 1] = -0.5
        ky[0, 0, 1, 2, 1] =  0.5
        kz = torch.zeros(1, 1, 3, 3, 3)
        kz[0, 0, 0, 1, 1] = -0.5
        kz[0, 0, 2, 1, 1] =  0.5
        # Repeat per channel for depthwise conv (groups=NCA_CH).
        kx = kx.repeat(NCA_CH, 1, 1, 1, 1)
        ky = ky.repeat(NCA_CH, 1, 1, 1, 1)
        kz = kz.repeat(NCA_CH, 1, 1, 1, 1)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        self.register_buffer('kz', kz)

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, D, H, W).  Returns (B, 16, D, H, W) — identity + 3 grads."""
        # Replicate-padding ≈ "clamped" boundary in the shader (the
        # simulator's clamped reads return the edge value, which is
        # exactly what F.pad mode='replicate' does).
        xp = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')
        dx = F.conv3d(xp, self.kx, groups=NCA_CH)
        dy = F.conv3d(xp, self.ky, groups=NCA_CH)
        dz = F.conv3d(xp, self.kz, groups=NCA_CH)
        return torch.cat([x, dx, dy, dz], dim=1)

    def forward(self, x: torch.Tensor, fire_rate: float = 0.5) -> torch.Tensor:
        """One asynchronous update step.  x: (B, 4, D, H, W)."""
        p = self.perceive(x)                        # (B, 16, D, H, W)
        B, _, D, H, W = p.shape
        # MLP: reshape (B, 16, V) → linear → reshape back.
        p_flat = p.permute(0, 2, 3, 4, 1).reshape(-1, NCA_IN)
        h = F.relu(self.fc1(p_flat))
        delta = self.fc2(h).reshape(B, D, H, W, NCA_OUT).permute(0, 4, 1, 2, 3)
        # Per-cell stochastic firing — same Bernoulli mask shape as shader.
        fire = (torch.rand(B, 1, D, H, W, device=x.device) < fire_rate).float()
        return x + delta * fire


# =====================================================================
# Training utilities
# =====================================================================

def make_seed(size: int, batch: int, device) -> torch.Tensor:
    """Single bright voxel at the centre, all 4 channels = 1.0 — matches
    `init_nca_seed` in simulator.py."""
    s = torch.zeros(batch, NCA_CH, size, size, size, device=device)
    c = size // 2
    s[:, :, c, c, c] = 1.0
    return s


def damage_batch(x: torch.Tensor, frac: float = 0.0) -> torch.Tensor:
    """Zero a random sphere in `frac` of the batch — encourages
    regeneration. No-op when frac<=0."""
    if frac <= 0:
        return x
    B, _, D, H, W = x.shape
    n = max(1, int(B * frac))
    for b in range(B - n, B):
        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()
        cz = torch.randint(0, D, (1,)).item()
        r = max(2, min(D, H, W) // 6)
        gz = torch.arange(D, device=x.device) - cz
        gy = torch.arange(H, device=x.device) - cy
        gx = torch.arange(W, device=x.device) - cx
        zz, yy, xx = torch.meshgrid(gz, gy, gx, indexing='ij')
        mask = (xx*xx + yy*yy + zz*zz) > r*r
        x[b] = x[b] * mask.float().unsqueeze(0)
    return x


def loss_fn(state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between channel 0 and the target, plus a small "overflow"
    penalty that discourages hidden channels from drifting beyond the
    same range used by the shader's soft clamp."""
    pred = state[:, 0]
    main = F.mse_loss(pred, target.unsqueeze(0).expand_as(pred))
    overflow = (state.abs() - 4.0).clamp(min=0).pow(2).mean()
    return main + 0.01 * overflow


def export_npz(model: NCA3D, path: Path):
    """Save weights in the exact byte layout the shader expects:
        W1 [IN × HID]  row-major  →  flat[0 : IN*HID]
        b1 [HID]                  →  next HID
        W2 [HID × OUT] row-major  →  next HID*OUT
        b2 [OUT]                  →  final OUT
    PyTorch's nn.Linear stores weight as (out_features, in_features) so
    we transpose to put it into the shader's (in, out) row-major form.
    """
    sd = model.state_dict()
    W1 = sd['fc1.weight'].cpu().numpy().T.astype(np.float32)  # (IN, HID)
    b1 = sd['fc1.bias'].cpu().numpy().astype(np.float32)
    W2 = sd['fc2.weight'].cpu().numpy().T.astype(np.float32)  # (HID, OUT)
    b2 = sd['fc2.bias'].cpu().numpy().astype(np.float32)
    assert W1.shape == (NCA_IN, NCA_HID)
    assert W2.shape == (NCA_HID, NCA_OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"  → saved {path}  ({W1.size + b1.size + W2.size + b2.size} floats)")


# =====================================================================
# Main training loop
# =====================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}  target: {args.target}  size: {args.size}")
    if device.type == 'cuda':
        print(f"        {torch.cuda.get_device_name(0)}")

    target = make_target(args.target, args.size, device)
    model = NCA3D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)

    # Sample pool: most start as the seed, gets cycled so the network
    # learns to maintain (not just grow) the target shape.
    pool = make_seed(args.size, args.pool, device)

    t0 = time.time()
    best_loss = float('inf')
    for it in range(1, args.steps + 1):
        # Sample a batch from the pool (with replacement is fine).
        idx = torch.randint(0, args.pool, (args.batch,), device=device)
        x = pool[idx].clone()
        # Always re-seed the lowest-index sample so the pool can't
        # forget what "start from a seed" looks like.
        x[0] = make_seed(args.size, 1, device)[0]
        x = damage_batch(x, args.damage)

        # Run a random number of steps so the network learns a stable
        # attractor (not just a fixed-T trajectory).
        n_steps = int(torch.randint(args.min_steps, args.max_steps + 1, (1,)).item())
        for _ in range(n_steps):
            x = model(x, fire_rate=args.fire_rate)
            # Soft clamp matches the shader — keeps gradients well-behaved.
            x = x.clamp(-args.clip, args.clip)

        loss = loss_fn(x, target)
        opt.zero_grad()
        loss.backward()
        # Per-tensor gradient normalisation (Mordvintsev's published TF
        # code: grads = [g/(norm(g)+1e-8) for g in grads]).  Without
        # this the loss spikes occasionally and the network forgets.
        # Adam's running-variance estimator then re-introduces sensible
        # per-parameter scaling on top of the unit-norm directions.
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(p.grad.norm() + 1e-8)
        opt.step()
        sched.step()

        # Replace highest-loss sample in batch with seed (pool curation).
        with torch.no_grad():
            per_sample = ((x[:, 0] - target.unsqueeze(0))**2).mean(dim=(1, 2, 3))
            worst = per_sample.argmax().item()
            x[worst] = make_seed(args.size, 1, device)[0]
            pool[idx] = x

        cur = loss.item()
        if cur < best_loss:
            best_loss = cur
        if it == 1 or it % args.log_every == 0 or it == args.steps:
            elapsed = time.time() - t0
            it_per_s = it / max(elapsed, 1e-9)
            print(f"  iter {it:5d}/{args.steps}  loss={cur:.5f}  "
                  f"best={best_loss:.5f}  ({it_per_s:.1f} it/s)")

    print(f"done in {time.time()-t0:.1f}s, best loss {best_loss:.5f}")
    export_npz(model, Path(args.out))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', default='sphere',
                   choices=['sphere', 'cube', 'torus', 'octahedron', 'shell'])
    p.add_argument('--size', type=int, default=32, help="cube edge length")
    p.add_argument('--steps', type=int, default=4000, help="training iterations")
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--pool',  type=int, default=64)
    p.add_argument('--min-steps', type=int, default=48,
                   help="min NCA steps per training iteration")
    p.add_argument('--max-steps', type=int, default=80)
    p.add_argument('--fire-rate', type=float, default=0.5)
    p.add_argument('--clip', type=float, default=4.0)
    p.add_argument('--damage', type=float, default=0.0,
                   help="fraction of batch to spherically damage (0..1)")
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--out', required=True, help="output .npz path")
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
