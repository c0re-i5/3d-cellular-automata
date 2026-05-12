"""Validate the 3D Ising shader against the known critical temperature
T_c ≈ 4.5115 (J=1, h=0) for the 3D simple-cubic Ising model.

We use the most reliable qualitative test for a phase transition with a
finite, fairly small lattice: SEED FROM A FULLY ORDERED STATE (all spins
up) and check whether the magnetisation survives. Below T_c it's stable;
above T_c it decays toward zero. A random init at low T fragments into
metastable domains and gives misleading "low |m|" readings — that's a
sampling artefact, not a shader bug, so we sidestep it.

  - For T well below T_c (~2):     |m| should stay > 0.9
  - For T near T_c (~4.5):         |m| should fall into [0.3, 0.95]
  - For T well above T_c (~7):     |m| should drop below 0.2

We don't try to extract the critical exponent β ≈ 0.326 — that requires
finite-size scaling on multiple grid sizes and is a much bigger run.

Run:  .venv/bin/python validate_ising.py
"""
from __future__ import annotations

import sys
import numpy as np

from test_harness import HeadlessRunner, create_headless_context, destroy_context
from simulator import INIT_FUNCS


def _all_up_init(size, rng):
    """Override init: every spin = +1. Channel R = 1.0 everywhere."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    data[:, :, :, 0] = 1.0
    return data


def measure_magnetisation(runner: HeadlessRunner) -> float:
    """Return |<s>| where s = +1 if cell.r > 0.5 else -1."""
    grid = runner.read_grid()[..., 0]
    spins = np.where(grid > 0.5, 1.0, -1.0)
    return float(abs(spins.mean()))


def run_temperature(ctx, T: float, size: int, equil_sweeps: int,
                    measure_sweeps: int, seed: int) -> tuple[float, float]:
    """Run at fixed T from an all-up init; report mean and std of |m|."""
    # Monkey-patch the init for this trial
    saved = INIT_FUNCS.get('random_spins')
    INIT_FUNCS['random_spins'] = _all_up_init
    try:
        runner = HeadlessRunner(
            ctx, 'ising_3d',
            size=size,
            seed=seed,
            params={'Temperature': T, 'Field h': 0.0,
                    'J coupling': 1.0, 'Update prob': 1.0},
            dt=1.0,
        )
    finally:
        INIT_FUNCS['random_spins'] = saved

    # One Metropolis "sweep" = 2 ping-pong steps (checkerboard parity
    # alternates with frame number). So 2 GL steps per sweep.
    for _ in range(equil_sweeps * 2):
        runner.step()

    samples: list[float] = []
    for s in range(measure_sweeps):
        for _ in range(2):
            runner.step()
        # Drop first 25% of measurement window as additional burn-in
        if s >= measure_sweeps // 4:
            samples.append(measure_magnetisation(runner))
    runner.release()
    return float(np.mean(samples)), float(np.std(samples))


def main() -> int:
    size = 32
    equil = 200
    measure = 150
    seed = 12345

    temperatures = [2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.5]

    window, ctx = create_headless_context()
    try:
        print(f"Lattice {size}^3, {equil} equilibration sweeps, "
              f"{measure} measurement sweeps. Init: all spins up.\n")
        print(f"{'T':>6}  {'|m|':>8}  {'sigma':>8}  phase")
        print('-' * 50)
        results = []
        for T in temperatures:
            m, sigma = run_temperature(ctx, T, size, equil, measure, seed)
            if m > 0.7:
                phase = 'ordered (ferromagnet)'
            elif m > 0.3:
                phase = 'critical / crossover'
            else:
                phase = 'disordered (paramagnet)'
            print(f'{T:6.2f}  {m:8.4f}  {sigma:8.4f}  {phase}')
            results.append((T, m))
    finally:
        destroy_context(window)

    # Pass/fail criteria — qualitative only at this grid size.
    cold = [m for T, m in results if T <= 3.0]
    near_crit = [m for T, m in results if 4.0 <= T <= 5.0]
    hot = [m for T, m in results if T >= 6.0]

    cold_ok = all(m > 0.9 for m in cold)
    crit_ok = any(0.30 < m < 0.95 for m in near_crit)
    hot_ok = all(m < 0.2 for m in hot)

    print()
    print(f'cold (T<=3) all > 0.9?            {cold_ok}')
    print(f'critical (4<=T<=5) crossover?     {crit_ok}')
    print(f'hot (T>=6) all < 0.2?             {hot_ok}')

    if cold_ok and crit_ok and hot_ok:
        print('\nPASS: 3D Ising shader exhibits the expected phase transition.')
        return 0
    print('\nFAIL: Phase-transition signature NOT found.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
