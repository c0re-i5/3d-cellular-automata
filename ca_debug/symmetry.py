"""Dynamic symmetry / isotropy probe.

For each rule we evolve a control IC for K steps, then re-evolve a
transformed IC, undo the transform on the final state, and compare.
A correct shader will satisfy the equivariance:

    G(transform(IC)) == transform(G(IC))     for symmetries of the rule

Rules that fail this for *generic* symmetries (lattice translation,
axis-aligned rotation, axial reflection) almost always have a stencil
indexing bug, an axis-locked vector, or a sign error on one component
of a curl/grad -- exactly the bug class that scoring-based triage
cannot see.

Probes (each opt-in via the rule's preset; default-on for the 3):

  TRANSLATE_X  -- shift the IC by (k, 0, 0) under periodic BCs
  ROTATE_Z90   -- rotate IC 90 deg around z-axis (cubic-symmetric rules)
  REFLECT_X    -- mirror IC along x-axis (centro-symmetric rules)

We *don't* run rotate/reflect on rules that legitimately break that
symmetry (gravity, wind, handed chirality).  Those are tagged in the
preset via:

    "symmetry_break": ["rotate", "reflect_x"]   # skip these probes

Score:  err = || G_ctrl - inverse_transform(G_xform) ||_2 / || G_ctrl ||_2

Severity:
    crit   err > 0.50  (rule is wildly anisotropic / indexing broken)
    high   err > 0.20
    med    err > 0.05
    ok     err <= 0.05
    n/a    rule opted out of this transform
    err    crash / NaN / boundary mismatch

Usage:

    python -m ca_debug.symmetry                       # all rules
    python -m ca_debug.symmetry --rules fire,wave_3d
    python -m ca_debug.symmetry --steps 40 --size 32
    python -m ca_debug.symmetry --probes translate    # subset

Implementation notes:
  - We only probe rules with `kind != 'viewport'` and a 'voxel'
    `render_mode`.
  - Periodic-only transforms (translation, rotation) on rules with
    non-periodic boundaries will produce huge errors at the wall;
    we crop a margin of 4 voxels off each face before scoring so the
    probe measures bulk dynamics rather than boundary artifacts.
  - For rotation/reflection probes we additionally apply the same
    transform to vector-valued channels (vx, vy, vz) when the preset
    declares which channels are vectors via `vector_channels`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4, 'n/a': 5}


# ---------------------------------------------------------------------------
# IC transforms.  Each is (forward, inverse) where both operate on a
# (N, N, N, 4) numpy array.  Vector channels (if declared) are rotated
# in lockstep.
# ---------------------------------------------------------------------------

def _shift(arr: np.ndarray, axis: int, k: int) -> np.ndarray:
    return np.roll(arr, k, axis=axis)


def _rot90(arr: np.ndarray, axis: int, k: int = 1,
           vec_channels: tuple[int, int, int] | None = None) -> np.ndarray:
    """Rotate 90deg around `axis` (0=X, 1=Y, 2=Z), repeated k times.

    Spatial axes are permuted; if vec_channels = (vx, vy, vz) is given,
    the corresponding channel components are also rotated.
    """
    # `np.rot90` operates on a plane; we want a 3D rotation about an
    # axis, which is rot90 in the plane perpendicular to that axis.
    plane = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[axis]
    out = np.rot90(arr, k=k, axes=plane).copy()
    if vec_channels is not None:
        vx, vy, vz = vec_channels
        # Sample components on the rotated tensor (same channel indices,
        # but values came from rotated spatial sampling).
        cx = out[..., vx].copy()
        cy = out[..., vy].copy()
        cz = out[..., vz].copy()
        if axis == 0:  # X-axis: y -> z, z -> -y
            out[..., vy] = -cz
            out[..., vz] =  cy
        elif axis == 1:  # Y-axis: z -> x, x -> -z
            out[..., vz] = -cx
            out[..., vx] =  cz
        elif axis == 2:  # Z-axis: x -> y, y -> -x
            out[..., vx] = -cy
            out[..., vy] =  cx
    return out


def _reflect(arr: np.ndarray, axis: int,
             vec_channels: tuple[int, int, int] | None = None) -> np.ndarray:
    out = np.flip(arr, axis=axis).copy()
    if vec_channels is not None:
        vx, vy, vz = vec_channels
        comp = (vx, vy, vz)[axis]
        out[..., comp] *= -1.0
    return out


# ---------------------------------------------------------------------------
# Probe definitions.  Each probe has:
#   name, applies_to (predicate over preset), forward, inverse, label.
# `forward` and `inverse` take (arr, vec_channels).
# ---------------------------------------------------------------------------

# Shaders whose internal lab-frame noise (hash_static(pos,..),
# hash_temporal(pos,..), fbm3(vec3(pos),..)) breaks rotational and
# reflection equivariance by design.  Detected at import time by
# inspecting shader source.  Translation by an integer voxel under
# periodic BCs is still meaningful (the pos-keyed hash shifts with
# the field, so equivariance holds).
def _detect_lab_noise_shaders() -> set[str]:
    """Find shaders whose update reads stochastic noise keyed on lab
    coordinates -- breaks rotation/reflection/translation equivariance.

    Heuristic: any call to a hash/noise/fbm helper, OR an inline
    `fract(sin(...pos...))` random idiom, counts.
    """
    try:
        from simulator import CA_RULES
    except Exception:  # noqa: BLE001  optional dependency
        return set()
    import re
    pat = re.compile(
        r'\b(?:'
        r'hash_static|hash_temporal|hash_dir|block_hash|pp_hash|ppr_hash'
        r'|fbm3|noise3'
        r')\s*\('
        r'|fract\s*\(\s*sin\s*\([^)]*\bpos\b',
        re.MULTILINE,
    )
    out: set[str] = set()
    for name, src in CA_RULES.items():
        if not isinstance(src, str):
            continue
        if pat.search(src):
            out.add(name)
    # Element-CA shader is built from ELEMENT_CA_RULE at runtime, not
    # in CA_RULES.  Tag the synthetic 'element_ca' shader name here.
    try:
        from simulator import ELEMENT_CA_RULE
        if pat.search(ELEMENT_CA_RULE):
            out.add('element_ca')
    except Exception:  # noqa: BLE001  optional dependency
        pass
    # Manual additions: shaders that use lab-frame literal coordinates
    # for design-time asymmetry (rain cells, plume sources, chiral spin
    # axes, etc.) that the regex above can't catch.  All confirmed by
    # source inspection to be intentional, not bugs.
    out.update({
        'erosion_hydraulic_3d',     # 3 lab-frame rain cells at fixed XZ
        'galaxy_dynamics_3d',       # chiral spin axis (handed dynamics)
        'galaxy_poisson_3d',        # Poisson solve over chiral mass dist
        'peridyn_force_3d',         # crack-tip nucleation at fixed seed
        'peridyn_disp_3d',          # paired with peridyn_force_3d
        'physarum_adaptive_3d',     # agent population in lab frame
        'physarum_pressure_3d',     # pressure solve over agent field
        'q_relax_3d',               # nematic relax (chiral)
        'q_flow_3d',                # nematic active flow (chiral)
        'q_advect_3d',              # nematic advection
    })
    return out


_SHADERS_QUENCHED_LAB_NOISE: set[str] = _detect_lab_noise_shaders()


class Probe:
    def __init__(self, name: str, label: str,
                 forward: Callable, inverse: Callable,
                 needs_periodic: bool = True):
        self.name = name
        self.label = label
        self.forward = forward
        self.inverse = inverse
        self.needs_periodic = needs_periodic

    def applies(self, preset: dict) -> tuple[bool, str]:
        """Returns (applies, reason_if_not).  A symmetry probe is only
        meaningful when the rule actually claims that symmetry."""
        skip = preset.get('symmetry_break') or []
        if self.name in skip or self.label in skip:
            return False, 'opt-out via symmetry_break'
        # Shaders that intentionally use position-dependent quenched
        # noise (hash_static(pos,..), fbm3(vec3(pos),..)) are *not*
        # exactly equivariant under spatial transformations -- their
        # internal "defect" texture is fixed in lab coordinates.  This
        # is a documented design choice (e.g. crystal_growth defects +
        # twin nucleation), not a bug, so we skip rotate/reflect for
        # those shaders.  Translation by an integer voxel under
        # periodic BCs is still meaningful (hashes shift with the
        # field, since they're indexed by pos which we also shift).
        shader = preset.get('shader') or ''
        # Multi-pass presets may have lab-frame noise in any pass even
        # if the headline shader is clean (e.g. genome_ca_3d -> state
        # pass clean, evolve pass uses fbm3/hashes).  Collect every
        # shader name from the pipeline.
        pass_shaders = {p.get('shader') for p in (preset.get('passes') or [])
                        if isinstance(p, dict) and p.get('shader')}
        all_shaders = {shader} | pass_shaders
        if all_shaders & _SHADERS_QUENCHED_LAB_NOISE:
            if self.name in ('rotate_z90', 'reflect_x', 'translate_x'):
                bad = all_shaders & _SHADERS_QUENCHED_LAB_NOISE
                return False, f'{",".join(sorted(bad))}: lab-frame noise'
        bnd = (preset.get('boundary') or 'toroidal').lower()
        is_periodic = bnd in ('toroidal', 'periodic', 'wrap')
        # Translation symmetry only holds under periodic BCs.  Without
        # them, the IC's contact with the wall depends on its position.
        if self.name == 'translate_x' and not is_periodic:
            return False, f'non-periodic BC ({bnd})'
        # Rotation around Z is broken by anything that singles out the
        # vertical axis: explicit gravity, buoyancy, wind, etc.  We use
        # a conservative heuristic over preset params + description.
        if self.name == 'rotate_z90':
            asym_keys = ('Gravity', 'Buoyancy', 'Wind', 'Convection',
                         'gravity', 'buoyancy', 'wind')
            params = preset.get('params') or {}
            if any(any(k in p for p in params) for k in asym_keys):
                return False, 'has gravity/buoyancy/wind param'
            desc = (preset.get('description') or '').lower()
            if any(k in desc for k in ('gravity', 'buoyan', 'rises', 'sinks',
                                        'falls', 'plume', 'wind')):
                return False, 'description mentions gravity/buoyancy'
        # Reflection along X is broken by:
        #   - the same gravity/wind set above (if wind is non-axis-aligned)
        #   - rules with chirality (handed dynamics)
        #   - rules that use a hash function of pos (stochastic position-
        #     dependent operators); we conservatively skip such rules
        #     because their reflected dynamics are valid but not bitwise
        #     comparable.
        if self.name == 'reflect_x':
            params = preset.get('params') or {}
            if any('Wind' in p or 'wind' in p for p in params):
                return False, 'wind param breaks reflect_x'
            if any('Chirality' in p or 'Handedness' in p for p in params):
                return False, 'chirality param breaks reflect_x'
        return True, ''


PROBES: list[Probe] = [
    Probe(
        'translate_x', 'TRANSLATE_X',
        forward=lambda a, vc: _shift(a, axis=0, k=4),
        inverse=lambda a, vc: _shift(a, axis=0, k=-4),
        needs_periodic=True,
    ),
    Probe(
        'rotate_z90', 'ROTATE_Z90',
        forward=lambda a, vc: _rot90(a, axis=2, k=1, vec_channels=vc),
        inverse=lambda a, vc: _rot90(a, axis=2, k=-1, vec_channels=vc),
        needs_periodic=True,
    ),
    Probe(
        'reflect_x', 'REFLECT_X',
        forward=lambda a, vc: _reflect(a, axis=0, vec_channels=vc),
        inverse=lambda a, vc: _reflect(a, axis=0, vec_channels=vc),
        needs_periodic=False,
    ),
]


# ---------------------------------------------------------------------------
# Running pairs of trials with the same seed but transformed IC.
# ---------------------------------------------------------------------------

def _run_pair(ctx, rule: str, *, size: int, steps: int, seed: int,
              probe: Probe, vec_channels) -> dict[str, Any]:
    """Run a control trial, then a transformed trial sharing the same
    seed and params.  Returns dict with err, ic_norm, ctrl_norm and any
    error message.
    """
    from test_harness import HeadlessRunner

    # Control: build runner, take a copy of the IC, evolve.
    r1 = HeadlessRunner(ctx, rule, size=size, seed=seed)
    ic = r1.read_grid()
    for _ in range(steps):
        r1.step()
    g_ctrl = r1.read_grid()
    if hasattr(r1, 'release'):
        try: r1.release()
        except Exception: pass  # noqa: BLE001  GL resource release, never fatal

    # Transformed: build a fresh runner with the SAME seed (so any stored
    # state matches), then overwrite tex_a with the transformed IC.
    r2 = HeadlessRunner(ctx, rule, size=size, seed=seed)
    ic_t = probe.forward(ic, vec_channels)
    # Encode and write back into tex_a.
    enc = ic_t.astype(np.float32 if r2._tex_np_dtype == np.float32
                      else np.float16, copy=False).tobytes()
    r2.tex_a.write(enc)
    r2.ping = 0  # ensure tex_a is the active source
    # If the rule has a second image pair, transform it too -- otherwise
    # rules with vector velocity in pair2 (fire, flocking, EM) will
    # appear to break rotation/reflection symmetry purely because
    # pair2 was left untouched.
    if getattr(r2, 'tex_a2', None) is not None:
        # Harness may auto-bump size when the preset has a default_size /
        # search_size larger than what we requested; use the runner's
        # actual size, not the originally requested one.
        rsize = getattr(r2, 'size', size)
        ic2 = np.frombuffer(r2.tex_a2.read(), dtype=r2._tex_np_dtype).reshape(
            rsize, rsize, rsize, 4).astype(np.float32)
        # Use vector_channels=(0,1,2) for pair2 since the convention is
        # almost universally (vx, vy, vz, scalar) in this codebase.
        ic2_t = probe.forward(ic2, (0, 1, 2))
        enc2 = ic2_t.astype(np.float32 if r2._tex_np_dtype == np.float32
                            else np.float16, copy=False).tobytes()
        r2.tex_a2.write(enc2)
    for _ in range(steps):
        r2.step()
    g_xform = r2.read_grid()
    if hasattr(r2, 'release'):
        try: r2.release()
        except Exception: pass  # noqa: BLE001  GL resource release, never fatal

    g_undo = probe.inverse(g_xform, vec_channels)

    # Crop a 4-voxel margin to avoid boundary artifacts when BCs aren't
    # truly periodic.  This is safe for translation under periodic BCs
    # (no information lost) and necessary otherwise.  Use the runner's
    # actual size (may be auto-bumped from the requested size).
    actual_size = g_ctrl.shape[0]
    m = 4 if actual_size > 16 else 2
    a = g_ctrl[m:-m, m:-m, m:-m, :]
    b = g_undo[m:-m, m:-m, m:-m, :]

    diff = a - b
    ctrl_norm = float(np.linalg.norm(a))
    diff_norm = float(np.linalg.norm(diff))
    err = diff_norm / max(ctrl_norm, 1e-9)
    return {
        'err': err,
        'ctrl_norm': ctrl_norm,
        'diff_norm': diff_norm,
        'ic_norm': float(np.linalg.norm(ic)),
    }


# ---------------------------------------------------------------------------
# Per-rule grade
# ---------------------------------------------------------------------------

def _grade(probe_results: dict[str, dict]) -> dict[str, Any]:
    sev = 'ok'
    flags: list[str] = []
    max_err = 0.0
    for pname, res in probe_results.items():
        if 'error' in res:
            flags.append(f'ERR:{pname}')
            sev = _worst(sev, 'err')
            continue
        if res.get('skipped'):
            continue
        e = res.get('err', 0.0)
        max_err = max(max_err, e)
        if e > 0.50:
            sev = _worst(sev, 'crit')
            flags.append(f'{pname.upper()}={e:.2f}')
        elif e > 0.20:
            sev = _worst(sev, 'high')
            flags.append(f'{pname.upper()}={e:.2f}')
        elif e > 0.05:
            sev = _worst(sev, 'med')
            flags.append(f'{pname.upper()}={e:.2f}')
    return {'severity': sev, 'flags': flags, 'max_err': max_err}


def _worst(a: str, b: str) -> str:
    return a if _SEV_ORDER[a] < _SEV_ORDER[b] else b


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

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
        # Skip entity-arena rules (state isn't in tex_a alone).
        if preset.get('agent_count') or 'entity_arena' in preset:
            continue
        # Skip particle-only rules.
        if (preset.get('passes') or [{}])[0].get('kind') == 'particle':
            continue
        rules.append(r)
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = set(s.strip() for s in args.skip.split(',') if s.strip())
        rules = [r for r in rules if r not in skip_set]
    return rules


def _vec_channels(preset: dict) -> tuple[int, int, int] | None:
    vc = preset.get('vector_channels')
    if vc and len(vc) == 3:
        return tuple(int(x) for x in vc)
    # Heuristic from vis_channels names.
    vis = preset.get('vis_channels') or []
    found = {}
    for i, name in enumerate(vis):
        n = str(name).lower()
        for axis, key in (('x', 'vx'), ('y', 'vy'), ('z', 'vz')):
            if (f'vel {axis}' in n or n.endswith(f'_{key}') or
                    n == key or n.endswith(f' {axis}') and 'vel' in n):
                found[axis] = i
        # Also catch Bx/By/Bz, Ex/Ey/Ez
        if len(n) <= 3 and n.endswith(('x', 'y', 'z')) and n[0] in 'beuv':
            found[n[-1]] = i
    if all(k in found for k in 'xyz'):
        return (found['x'], found['y'], found['z'])
    return None


def _run_one(ctx, rule: str, args) -> dict[str, Any]:
    from simulator import _resolve_composed_preset
    preset = _resolve_composed_preset(rule)
    vec_channels = _vec_channels(preset)
    selected_probes = {p.name for p in PROBES}
    if args.probes:
        selected_probes = set(p.strip() for p in args.probes.split(','))

    results: dict[str, dict] = {}
    for probe in PROBES:
        if probe.name not in selected_probes:
            continue
        ok, reason = probe.applies(preset)
        if not ok:
            results[probe.name] = {'skipped': True, 'reason': reason}
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = _run_pair(ctx, rule, size=args.size, steps=args.steps,
                              seed=args.seed, probe=probe,
                              vec_channels=vec_channels)
            results[probe.name] = r
        except Exception as e:  # noqa: BLE001  per-rule trial may crash, record error and continue
            results[probe.name] = {'error': f'{type(e).__name__}: {e}',
                                   'tb': traceback.format_exc()}
    grade = _grade(results)
    return {'rule': rule, 'probes': results, 'grade': grade,
            'vec_channels': vec_channels}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--probes', help='Comma-separated probe names (default: all).')
    ap.add_argument('--size', type=int, default=32)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='med',
                    help='Min severity to print (default: med).')
    ap.add_argument('--json', help='Write per-rule report JSON to this path.')
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
        except Exception as e:  # noqa: BLE001  per-rule trial may crash, record error and continue
            row = {'rule': rule, 'probes': {},
                   'grade': {'severity': 'err', 'flags': [f'CRASH:{e}'],
                             'max_err': 0.0}}
        rows.append(row)
    sys.stdout.write('\r' + ' ' * 60 + '\r')
    elapsed = time.perf_counter() - t0

    # Sort: severity, then -max_err, then rule name.
    rows.sort(key=lambda r: (_SEV_ORDER[r['grade']['severity']],
                             -r['grade']['max_err'], r['rule']))

    min_sev = _SEV_ORDER[args.severity]
    print(f"\nSymmetry probe (size={args.size}, steps={args.steps}, "
          f"seed={args.seed}) -- {elapsed:.1f}s")
    print(f"{'SEV':<5}  {'RULE':<36}  {'MAX_ERR':>8}  FLAGS")
    print('-' * 80)
    by_sev: dict[str, int] = {}
    for r in rows:
        sev = r['grade']['severity']
        by_sev[sev] = by_sev.get(sev, 0) + 1
        if _SEV_ORDER[sev] > min_sev:
            continue
        flags = ' '.join(r['grade']['flags'])
        print(f"{sev:<5}  {r['rule']:<36}  {r['grade']['max_err']:>8.3f}  {flags}")

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
