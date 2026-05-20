"""Conservation probe.

For each rule we compute a panel of conserved-quantity candidates at
frame 0 and after K steps, then flag:

  DISCOVERED   a channel (or pair-sum) that drifts by < EPSILON over
               K steps.  These are auto-discovered conservation laws.
               Useful as documentation -- and as a regression baseline.

  VIOLATED     a quantity the rule *should* conserve (per the table
               below) but doesn't.  This is the bug class.

  DRIFT        a channel that drifts monotonically (not just step-to-
               step noise) but is not declared conserved.  Often
               legitimate (energy is dissipated by damping, fuel is
               consumed) but flags rules whose visualization will
               eventually wash out.

The expected-conservation table encodes physics literature:

    quantum_*           normalization (sum of |psi|^2) == 1
    cahn_hilliard       sum of phase field (channel 0)
    margolus_*          particle count (sum of channel 0)
    sandpile_3d         total grains (sum of channel 0) -- modulo
                        boundary outflow if not toroidal
    ising_3d            total spins are NOT conserved (Glauber);
                        but for Kawasaki dynamics they would be.
    brusselator,        catalysts X+Y conserved against substrate.
    schnakenberg
    predator_prey       no exact conservation (Lotka-Volterra has
                        a constant of motion but it's nonlinear).
    em_*                charge density should be conserved if Maxwell
                        is consistent.

Usage:
  python -m ca_debug.conservation
  python -m ca_debug.conservation --rules quantum_orbital,cahn_hilliard_3d
  python -m ca_debug.conservation --steps 100 --size 32
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from typing import Any

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4, 'n/a': 5}

# Drift below this (relative to t=0 magnitude) is considered "conserved".
DISCOVERY_EPS = 1e-3
# Drift above this for a should-conserve quantity is a bug.
VIOLATION_EPS = 1e-2

# Expected conservation laws.  Each entry is one of:
#   'channel:N'         -- sum of channel N is conserved
#   'sum:I,J,...'       -- sum of channels I+J+... is conserved
#   'norm2:N'           -- sum of channel N squared (e.g. |psi|^2)
#   'norm2_pair:I,J'    -- sum of I^2 + J^2 (Re/Im pairs)
EXPECTED: dict[str, list[str]] = {
    # Quantum: |psi|^2 = Re^2 + Im^2 should integrate to a constant.
    'quantum_wavepacket':   ['norm2_pair:0,1'],
    'quantum_double_slit':  ['norm2_pair:0,1'],
    'quantum_orbital':      ['norm2_pair:0,1'],
    'quantum_hydrogen':     ['norm2_pair:0,1'],
    'quantum_harmonic':     ['norm2_pair:0,1'],
    'quantum_tunneling':    ['norm2_pair:0,1'],
    'quantum_molecule':     ['norm2_pair:0,1'],
    'quantum_antibonding':  ['norm2_pair:0,1'],
    'quantum_element':      ['norm2_pair:0,1'],
    'quantum_selfinteract': ['norm2_pair:0,1'],
    # Margolus partition: lattice gas conserves particles.
    'margolus_3d':          ['channel:0'],
    # Sandpile and Cahn-Hilliard variants are toroidal-only conservers.
    # The default presets use clamped/non-toroidal BCs which legitimately
    # leak grains/mass.  We don't tag them: their drift is BC-driven,
    # not a math bug.  Re-add here once a strictly-toroidal variant
    # exists if you want regression coverage.
}


def _measure(g: np.ndarray) -> dict[str, float]:
    """Return dict of named scalar quantities for this grid state."""
    out: dict[str, float] = {}
    n_chan = g.shape[-1]
    for c in range(n_chan):
        out[f'channel:{c}'] = float(g[..., c].sum())
        out[f'norm2:{c}'] = float((g[..., c] ** 2).sum())
        out[f'absum:{c}'] = float(np.abs(g[..., c]).sum())
    # Pair sums and norm2 pairs (for Re/Im).
    for i in range(n_chan):
        for j in range(i + 1, n_chan):
            out[f'sum:{i},{j}'] = out[f'channel:{i}'] + out[f'channel:{j}']
            out[f'norm2_pair:{i},{j}'] = out[f'norm2:{i}'] + out[f'norm2:{j}']
    # Total mass + total |state|.
    out['total'] = float(g.sum())
    out['total_abs'] = float(np.abs(g).sum())
    return out


def _evolve(ctx, rule: str, *, size: int, steps: int, seed: int):
    from test_harness import HeadlessRunner
    r = HeadlessRunner(ctx, rule, size=size, seed=seed)
    g0 = r.read_grid().astype(np.float32).copy()
    # Sample at three points so we can detect monotonic drift vs noise.
    samples = [_measure(g0)]
    half = max(1, steps // 2)
    for _ in range(half):
        r.step()
    samples.append(_measure(r.read_grid().astype(np.float32)))
    for _ in range(steps - half):
        r.step()
    g_final = r.read_grid().astype(np.float32)
    samples.append(_measure(g_final))
    if hasattr(r, 'release'):
        try: r.release()
        except Exception: pass  # noqa: BLE001  GL resource release, never fatal
    return samples, g0, g_final


def _drift(s0: dict[str, float], sf: dict[str, float],
           name: str, voxels: int = 1) -> tuple[float, float]:
    """Return (relative_drift, absolute_drift) for the named quantity.

    The denominator is floored at voxels * 1e-3 so a quantity whose
    baseline mean is near zero (Cahn-Hilliard order parameter centered
    at c=0, signed wavefunctions, etc.) doesn't produce spuriously
    large relative drift from float-round-off-scale absolute drift.
    """
    a = s0.get(name, 0.0)
    b = sf.get(name, 0.0)
    abs_d = abs(b - a)
    floor = max(1.0, float(voxels)) * 1e-3
    rel_d = abs_d / max(abs(a), abs(b), floor)
    return rel_d, abs_d


def _classify_run(samples: list[dict], expected: list[str],
                  voxels: int) -> dict[str, Any]:
    s0, smid, sf = samples
    # Filter out trivially-zero quantities (channel sums that are zero
    # at t=0 and stay zero are useless to report).
    quantities = [k for k in s0.keys()
                  if abs(s0[k]) > 1e-6 or abs(sf[k]) > 1e-6]

    discovered: list[tuple[str, float]] = []
    drifted: list[tuple[str, float]] = []
    for q in quantities:
        rel_d, _ = _drift(s0, sf, q, voxels=voxels)
        if rel_d < DISCOVERY_EPS:
            discovered.append((q, rel_d))
        elif rel_d > 0.05:  # >5% drift is notable
            # Check monotonicity vs midpoint to distinguish drift from
            # oscillation.
            rel_mid, _ = _drift(s0, smid, q, voxels=voxels)
            monotonic = (rel_mid < rel_d * 0.9)  # halfway is < 90% of full
            if monotonic:
                drifted.append((q, rel_d))

    violations: list[dict] = []
    for q in expected:
        rel_d, abs_d = _drift(s0, sf, q, voxels=voxels)
        if rel_d > VIOLATION_EPS:
            violations.append({'quantity': q, 'rel_drift': rel_d,
                               'abs_drift': abs_d,
                               't0': s0.get(q, 0.0), 'tf': sf.get(q, 0.0)})

    # Severity:
    #   crit   any violation > 10%
    #   high   any violation > 1%
    #   med    drift > 50% on a top-3 channel (likely visualization burnout)
    #   ok     otherwise
    sev = 'ok'
    flags: list[str] = []
    for v in violations:
        if v['rel_drift'] > 0.10:
            sev = _worst(sev, 'crit')
        else:
            sev = _worst(sev, 'high')
        flags.append(f"VIOLATED {v['quantity']}=" 
                     f"{v['rel_drift']*100:.1f}%")
    # Surface notable channel drifts (top 3 channels only, since pair
    # quantities just duplicate).
    for q, d in drifted:
        if q.startswith('channel:') and d > 0.5:
            sev = _worst(sev, 'med')
            flags.append(f'DRIFT {q}={d*100:.0f}%')

    # Filter discovered to the most useful ones (skip "absum" duplicates
    # of sign-definite channels, and pair sums that just reflect single
    # channel conservation).
    interesting = [(q, d) for q, d in discovered
                   if not q.startswith('absum:')
                   and not q.startswith('sum:')]

    return {'severity': sev, 'flags': flags,
            'discovered': interesting, 'violations': violations,
            'n_discovered': len(interesting), 'n_violations': len(violations)}


def _worst(a: str, b: str) -> str:
    return a if _SEV_ORDER[a] < _SEV_ORDER[b] else b


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS, _resolve_composed_preset
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = []
    for r in sorted(RULE_PRESETS.keys()):
        try:
            preset = _resolve_composed_preset(r)
        except Exception:  # noqa: BLE001  preset lookup failure -> caller falls back
            continue
        if preset.get('kind') == 'viewport':
            continue
        if preset.get('agent_count') or 'entity_arena' in preset:
            continue
        if (preset.get('passes') or [{}])[0].get('kind') == 'particle':
            continue
        rules.append(r)
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = set(s.strip() for s in args.skip.split(',') if s.strip())
        rules = [r for r in rules if r not in skip_set]
    return rules


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rules')
    ap.add_argument('--size', type=int, default=32)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--seed', type=int, default=314)
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='med')
    ap.add_argument('--show-discoveries', action='store_true',
                    help='Print discovered conservation laws even for ok rules.')
    ap.add_argument('--json')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    window, ctx = create_headless_context()

    rules = _select_rules(args)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<40}")
        sys.stdout.flush()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                samples, g0, _gf = _evolve(ctx, rule, size=args.size,
                                           steps=args.steps, seed=args.seed)
            voxels = int(np.prod(g0.shape[:-1]))
            grade = _classify_run(samples, EXPECTED.get(rule, []), voxels)
            row = {'rule': rule, 'grade': grade}
        except Exception as e:  # noqa: BLE001  per-rule trial may crash, record error and continue
            row = {'rule': rule,
                   'grade': {'severity': 'err',
                             'flags': [f'CRASH:{type(e).__name__}:{e}'],
                             'discovered': [], 'violations': [],
                             'n_discovered': 0, 'n_violations': 0}}
        rows.append(row)
    sys.stdout.write('\r' + ' ' * 60 + '\r')
    elapsed = time.perf_counter() - t0

    rows.sort(key=lambda r: (_SEV_ORDER[r['grade']['severity']],
                             -r['grade']['n_violations'], r['rule']))

    min_sev = _SEV_ORDER[args.severity]
    print(f"\nConservation probe (size={args.size}, steps={args.steps}) "
          f"-- {elapsed:.1f}s")
    print(f"{'SEV':<5}  {'RULE':<32}  V  D  FLAGS")
    print('-' * 96)
    by_sev: dict[str, int] = {}
    for r in rows:
        sev = r['grade']['severity']
        by_sev[sev] = by_sev.get(sev, 0) + 1
        if _SEV_ORDER[sev] > min_sev:
            continue
        flags = ' | '.join(r['grade']['flags'])
        print(f"{sev:<5}  {r['rule']:<32}  "
              f"{r['grade']['n_violations']:>1}  "
              f"{r['grade']['n_discovered']:>1}  {flags}")

    if args.show_discoveries:
        print('\nDiscovered conservation laws (drift < 0.1%):')
        for r in rows:
            disc = r['grade']['discovered']
            if disc:
                qs = ', '.join(f'{q}({d*100:.2g}%)' for q, d in disc)
                print(f"  {r['rule']:<32}  {qs}")

    print()
    print('Summary:', ' '.join(f'{k}={by_sev.get(k, 0)}' for k in
                                ('crit', 'high', 'med', 'ok', 'n/a', 'err')))

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(rows, fh, indent=2, default=str)
        print(f"Wrote {args.json}")

    return 0 if by_sev.get('err', 0) == 0 and by_sev.get('crit', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
