"""Param coupling probe.

For each rule we run a baseline trial, then perturb each tunable
parameter by +/-10% and measure how each output channel responds.
This produces a (n_params x n_channels) coupling matrix.

Bug patterns this surfaces:

  DEAD_PARAM       a tunable that has zero measurable effect on every
                   channel.  Almost always means the param's slot in
                   `pass_params` is wrong, the shader reads the wrong
                   uniform, or the value is being overwritten before
                   it reaches the math.  This is exactly the shape of
                   the original fire `T_ign` desync we found via
                   shader_lint -- but lint only catches it when the
                   literal happens to equal the default; coupling
                   catches it when the slot is plain wrong.

  EXPLOSIVE_PARAM  a 10% perturbation makes the output > 100x larger
                   or produces NaN/Inf.  Indicates a missing clamp,
                   division by a near-zero quantity, or unstable
                   coefficient.

  ASYMMETRIC       +10% and -10% give wildly different |response|
                   (>10x ratio).  Often legitimate (e.g. coupling
                   constant near zero) but flags coefficients that
                   sit on a bifurcation.

Usage:
  python -m ca_debug.coupling                         # all rules
  python -m ca_debug.coupling --rules fire,wave_3d
  python -m ca_debug.coupling --steps 30 --size 32
  python -m ca_debug.coupling --json /tmp/coupling.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
import traceback
from typing import Any

import re

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4, 'n/a': 5}

# Channel-level response below this is considered "no signal".  Set
# generously: float16 round-trip noise + first-step transients can
# easily produce ~1e-4 relative diffs even for unrelated params.
DEAD_THRESHOLD = 5e-3
EXPLOSIVE_THRESHOLD = 1e2

# Param names that are explicit placeholders -- the preset author has
# already declared these slots unused.  We skip them entirely so the
# probe doesn't pollute output with hundreds of expected-dead rows.
_PLACEHOLDER_RE = re.compile(
    r'^(_+\d*|\(unused\)|unused(_\d+)?|reserved|n/?a)$',
    re.IGNORECASE)

# Param names that look like discrete mode selectors -- a +/-10%
# perturbation never crosses an integer boundary so the response is
# legitimately zero.  Flagged as MODE_PARAM (info), not DEAD (bug).
_MODE_NAMES = {
    'shape', 'mode', 'moore', 'region', 'variant', 'kind',
    'head min', 'head max', 'birth min', 'birth max', 'survive min',
    'survive max', 'radius r', 'birth low', 'birth high', 'survive ±',
}

# Params whose value is baked into the initial-condition geometry or a
# precomputed kernel -- changing them mid-run does nothing because the
# convolution kernel / IC was already built.  This is a design fact,
# not a bug.  Flagged as INIT_TIME (info).
_INIT_TIME_NAMES = {
    'kernel radius', 'ring position', 'a_kernel radius', 'b_kernel radius',
}


def _evolve(ctx, rule: str, *, size: int, steps: int, seed: int,
            params_override: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Build a runner, set params, evolve K steps, return (initial, final) grids."""
    from test_harness import HeadlessRunner
    r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                       params=params_override)
    g0 = r.read_grid().astype(np.float32).copy()
    for _ in range(steps):
        r.step()
    g = r.read_grid().astype(np.float32)
    if hasattr(r, 'release'):
        try: r.release()
        except Exception: pass
    return g0, g


def _per_channel_response(g_base: np.ndarray, g_pert: np.ndarray,
                          delta_base: np.ndarray) -> np.ndarray:
    """Per-channel coupling response, normalized against the baseline's
    own evolution: how much did the perturbation displace the trajectory
    relative to how far it travelled?

        resp[c] = ||g_base[c] - g_pert[c]|| / max(||delta_base[c]||, eps)

    Where delta_base = g_base - g_initial.  This makes the probe robust
    to rules whose IC dominates the absolute magnitude (e.g. localized
    wave pulse) but whose dynamics are still being measured.
    """
    out = np.zeros(g_base.shape[-1], dtype=np.float32)
    for c in range(g_base.shape[-1]):
        diff = float(np.linalg.norm(g_base[..., c] - g_pert[..., c]))
        denom = float(np.linalg.norm(delta_base[..., c]))
        if denom < 1e-6:
            # Baseline didn't evolve in this channel.  Fall back to
            # absolute-magnitude normalization, but the result is only
            # meaningful if the diff is also non-trivially large.
            mag = float(np.linalg.norm(g_base[..., c]))
            if mag < 1e-6:
                out[c] = 0.0
            else:
                out[c] = diff / mag
        else:
            out[c] = diff / denom
    return out


def _grade_rule(coupling: dict, delta_norms: np.ndarray) -> dict[str, Any]:
    """Coupling[param] = {'+': per_chan_relerr, '-': per_chan_relerr,
                          'value': default, 'nan': bool, 'kind': str}.
    """
    flags: list[str] = []
    sev = 'ok'
    dead: list[str] = []
    explosive: list[str] = []
    asym: list[str] = []
    mode_params: list[str] = []
    init_params: list[str] = []
    for pname, info in coupling.items():
        kind = info.get('kind', 'normal')
        if info.get('nan_pos') or info.get('nan_neg'):
            explosive.append(f'{pname}(NaN)')
            sev = _worst(sev, 'crit')
            continue
        plus = info['+']
        minus = info['-']
        max_resp = float(max(plus.max(), minus.max()))
        if max_resp < DEAD_THRESHOLD:
            if kind == 'mode':
                mode_params.append(pname)
            elif kind == 'init_time':
                init_params.append(pname)
            else:
                dead.append(pname)
                sev = _worst(sev, 'med')
            continue
        if max_resp > EXPLOSIVE_THRESHOLD:
            explosive.append(f'{pname}({max_resp:.0f}x)')
            sev = _worst(sev, 'high')
            continue
        # Asymmetric response: take ratio of summed responses.
        # Require BOTH directions to clear DEAD_THRESHOLD on at least
        # one channel before flagging -- otherwise discrete-threshold
        # rules (Game of Life etc.) where a small float perturbation
        # never crosses the integer boundary will register asym noise.
        sp = float(plus.sum())
        sm = float(minus.sum())
        if (plus.max() > DEAD_THRESHOLD and minus.max() > DEAD_THRESHOLD
                and sp + sm > 1e-3):
            ratio = max(sp, sm) / max(min(sp, sm), 1e-6)
            if ratio > 50.0:
                asym.append(f'{pname}({ratio:.0f}:1)')
                sev = _worst(sev, 'med')
    if dead:
        # If EVERY non-mode/non-init param is dead, the run is more
        # likely too short to express any param's effect than every
        # single param being broken simultaneously.  Downgrade to
        # informational ("UNRESPONSIVE") rather than flagging each
        # param as a bug.
        all_normal = [p for p, info in coupling.items()
                      if info.get('kind', 'normal') == 'normal'
                      and not (info.get('nan_pos') or info.get('nan_neg'))]
        if all_normal and len(dead) == len(all_normal):
            return {'severity': 'n/a',
                    'flags': [f'UNRESPONSIVE (all {len(dead)} params'
                              f' below threshold; rule may need more steps)'],
                    'n_dead': len(dead), 'n_explosive': 0,
                    'n_asym': 0,
                    'n_mode': len(mode_params), 'n_init': len(init_params)}
        flags.append('DEAD=' + ','.join(dead))
    if explosive:
        flags.append('EXP=' + ','.join(explosive))
    if asym:
        flags.append('ASYM=' + ','.join(asym))
    if mode_params:
        flags.append('MODE=' + ','.join(mode_params))
    if init_params:
        flags.append('INIT=' + ','.join(init_params))
    return {'severity': sev, 'flags': flags,
            'n_dead': len(dead), 'n_explosive': len(explosive),
            'n_asym': len(asym),
            'n_mode': len(mode_params), 'n_init': len(init_params)}


def _worst(a: str, b: str) -> str:
    return a if _SEV_ORDER[a] < _SEV_ORDER[b] else b


def _perturb(value: float, frac: float) -> tuple[float, float]:
    """Return (high, low) perturbed values.  For values near zero we
    use an additive perturbation so the probe doesn't degenerate."""
    if abs(value) < 1e-3:
        return value + 0.1, value - 0.1
    return value * (1.0 + frac), value * (1.0 - frac)


def _run_one(ctx, rule: str, args) -> dict[str, Any]:
    from simulator import _resolve_composed_preset
    preset = _resolve_composed_preset(rule)
    params_default = dict(preset.get('params') or {})
    if not params_default:
        return {'rule': rule, 'coupling': {},
                'grade': {'severity': 'n/a', 'flags': ['no params'],
                          'n_dead': 0, 'n_explosive': 0, 'n_asym': 0}}

    # Baseline.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            g_init, g_base = _evolve(ctx, rule, size=args.size, steps=args.steps,
                                     seed=args.seed, params_override=None)
    except Exception as e:
        return {'rule': rule, 'coupling': {},
                'grade': {'severity': 'err',
                          'flags': [f'BASELINE_CRASH:{type(e).__name__}'],
                          'n_dead': 0, 'n_explosive': 0, 'n_asym': 0}}

    delta_base = g_base - g_init
    # If the baseline barely evolved at all (every channel's L2 delta is
    # below noise), the probe can't distinguish param effects from noise.
    delta_norms = np.array([np.linalg.norm(delta_base[..., c])
                            for c in range(g_base.shape[-1])], dtype=np.float32)
    base_mag = float(np.linalg.norm(g_base))
    if float(delta_norms.max()) < 1e-4 * max(base_mag, 1.0):
        return {'rule': rule, 'coupling': {},
                'grade': {'severity': 'n/a',
                          'flags': [f'STATIC (||Δ||={float(delta_norms.max()):.2e})'],
                          'n_dead': 0, 'n_explosive': 0, 'n_asym': 0}}

    coupling: dict = {}
    pname_list = list(params_default.keys())
    # Drop placeholders entirely -- not worth a row.
    pname_list = [p for p in pname_list if not _PLACEHOLDER_RE.match(p.strip())]
    if args.max_params:
        pname_list = pname_list[:args.max_params]
    for pname in pname_list:
        v = float(params_default[pname])
        v_hi, v_lo = _perturb(v, args.perturb)
        # Classify the param so dead-flag triage is meaningful.  Only
        # explicit name matches qualify as MODE -- the "integer default"
        # heuristic is unreliable because many continuous params (e.g.
        # Momentum, Gravity) legitimately default to 0.0.
        pn_lower = pname.strip().lower()
        if pn_lower in _MODE_NAMES:
            kind = 'mode'
        elif pn_lower in _INIT_TIME_NAMES:
            kind = 'init_time'
        else:
            kind = 'normal'
        info = {'value': v, 'hi': v_hi, 'lo': v_lo, 'kind': kind}
        for tag, vp in (('+', v_hi), ('-', v_lo)):
            override = {pname: vp}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    _, g = _evolve(ctx, rule, size=args.size, steps=args.steps,
                                   seed=args.seed, params_override=override)
            except Exception as e:
                info[tag] = np.full(g_base.shape[-1], float('nan'),
                                    dtype=np.float32)
                info[f'nan_{ "pos" if tag=="+" else "neg" }'] = True
                info[f'err_{tag}'] = f'{type(e).__name__}: {e}'
                continue
            if not np.isfinite(g).all():
                info[tag] = np.full(g_base.shape[-1], float('nan'),
                                    dtype=np.float32)
                info[f'nan_{ "pos" if tag=="+" else "neg" }'] = True
                continue
            info[tag] = _per_channel_response(g_base, g, delta_base)
        coupling[pname] = info

    grade = _grade_rule(coupling, delta_norms)
    # Promote dead-headline-param to 'high' (only for non-mode params).
    headline = pname_list[:4]
    headline_dead = [p for p in headline
                     if p in coupling
                     and coupling[p].get('kind') == 'normal'
                     and coupling[p].get('+') is not None
                     and coupling[p].get('-') is not None
                     and not coupling[p].get('nan_pos')
                     and not coupling[p].get('nan_neg')
                     and float(max(coupling[p]['+'].max(),
                                   coupling[p]['-'].max())) < DEAD_THRESHOLD]
    if headline_dead:
        grade['severity'] = _worst(grade['severity'], 'high')
        grade['flags'].insert(0, 'HEADLINE_DEAD=' + ','.join(headline_dead))

    return {'rule': rule, 'coupling': coupling, 'grade': grade}


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS, _resolve_composed_preset
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = []
    for r in sorted(RULE_PRESETS.keys()):
        try:
            preset = _resolve_composed_preset(r)
        except Exception:
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
    ap.add_argument('--size', type=int, default=24)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--perturb', type=float, default=0.10,
                    help='Fractional perturbation (default 0.10 = +/-10%).')
    ap.add_argument('--max-params', type=int, default=8,
                    help='Cap params per rule to keep runtime bounded.')
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='med')
    ap.add_argument('--json')
    ap.add_argument('--matrix-for', help='Print full coupling matrix for this rule.')
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
            row = _run_one(ctx, rule, args)
        except Exception as e:
            row = {'rule': rule, 'coupling': {},
                   'grade': {'severity': 'err', 'flags': [f'CRASH:{e}'],
                             'n_dead': 0, 'n_explosive': 0, 'n_asym': 0}}
        rows.append(row)
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    rows.sort(key=lambda r: (_SEV_ORDER[r['grade']['severity']],
                             -r['grade']['n_dead'] - r['grade']['n_explosive'],
                             r['rule']))

    min_sev = _SEV_ORDER[args.severity]
    print(f"\nCoupling probe (size={args.size}, steps={args.steps}, "
          f"perturb=+/-{args.perturb*100:.0f}%) -- {elapsed:.1f}s")
    print(f"{'SEV':<5}  {'RULE':<32}  D  E  A  FLAGS")
    print('-' * 96)
    by_sev: dict[str, int] = {}
    for r in rows:
        sev = r['grade']['severity']
        by_sev[sev] = by_sev.get(sev, 0) + 1
        if _SEV_ORDER[sev] > min_sev:
            continue
        flags = ' | '.join(r['grade']['flags'])
        print(f"{sev:<5}  {r['rule']:<32}  "
              f"{r['grade']['n_dead']:>1}  {r['grade']['n_explosive']:>1}  "
              f"{r['grade']['n_asym']:>1}  {flags}")

    print()
    print('Summary:', ' '.join(f'{k}={by_sev.get(k, 0)}' for k in
                                ('crit', 'high', 'med', 'ok', 'n/a', 'err')))

    # Optional: print coupling matrix for one rule.
    if args.matrix_for:
        target = next((r for r in rows if r['rule'] == args.matrix_for), None)
        if target and target['coupling']:
            print(f"\nCoupling matrix for {args.matrix_for}:")
            params = list(target['coupling'].keys())
            n_chan = len(next(iter(target['coupling'].values()))['+'])
            head = '  PARAM' + ' ' * 24 + ''.join(f'  C{c}+   C{c}- ' for c in range(n_chan))
            print(head)
            for p in params:
                info = target['coupling'][p]
                row = f'  {p:<28}'
                for c in range(n_chan):
                    row += f'  {info["+"][c]:5.2f}  {info["-"][c]:5.2f} '
                print(row)

    if args.json:
        # Convert numpy arrays for JSON.
        def _conv(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating,)):
                return float(o)
            return str(o)
        with open(args.json, 'w') as fh:
            json.dump(rows, fh, indent=2, default=_conv)
        print(f"Wrote {args.json}")

    return 0 if by_sev.get('err', 0) == 0 and by_sev.get('crit', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
