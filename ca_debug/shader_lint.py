"""Static lint pass over the GLSL compute-shader strings in CA_RULES.

Catches bug *patterns* before any GPU work runs.  Designed around the
fire bug class we hit in ef68de5:

  1. Hardcoded literal numerics inside the shader body that match a
     preset default value -- almost always a stale copy of a now-tunable
     parameter (the `T_ign = 0.20` desync bug).

  2. Axis-locked vectors near curl/cross/grad blocks
     (`vec3(x, 0.0, 0.0)` patterns inside vorticity / gradient
     computations).  The fire bug was exactly this: N forced onto X.

  3. `pass_params` slot mismatches: a pass references `u_paramN` but the
     preset's `pass_params[shader]` has fewer than N+1 entries.

  4. Shaders that fetch from `u_src*` but never imageStore to `u_dst*`
     (or vice versa).

  5. Single-pass rules where the shader body uses `imageLoad(u_src2)`
     or writes to `u_dst2` without the preset declaring `extra_fields`.

  6. Asymmetric stencils in laplacian/gradient code (e.g. uses +1 on
     three axes but +2/-0 on one -- the kind of typo that makes a rule
     anisotropic).  Detected by counting +k/-k offset literals per axis
     in any `lap`/`grad`/`omega` block.

Usage:
    python -m ca_debug.shader_lint                      # all shaders
    python -m ca_debug.shader_lint --rules fire,physarum
    python -m ca_debug.shader_lint --json lint.json
    python -m ca_debug.shader_lint --severity warn      # warn+ only

Severity tiers:
    err   - definite bug (slot mismatch, missing imageStore, ...)
    warn  - very likely bug (axis-locked vec3 in curl, hardcoded
            literal matching a tunable param default)
    info  - suspicious but may be intentional (literal that *could*
            be a default, asymmetric stencil offsets)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Loading shader source
# ---------------------------------------------------------------------------

def _load_shader_dict() -> dict[str, str]:
    """Pull every compute-shader source dict from simulator + sibling
    modules.  Different rule families live in different dicts:
      - simulator.CA_RULES         : voxel-grid rules
      - simulator.AGENT_RULES      : agent rules (langton ant, smugglers)
      - element_data.SHADERS / etc : per-rule modules
    Plus per-preset `entity_shaders` dicts injected at runtime.
    """
    import simulator
    out: dict[str, str] = {}

    def _scan_module(mod):
        for attr in dir(mod):
            if attr.startswith('_'):
                continue
            obj = getattr(mod, attr, None)
            if not isinstance(obj, dict) or not obj:
                continue
            sample = next(iter(obj.values()))
            if not isinstance(sample, str):
                continue
            if 'void main' in sample or 'imageStore' in sample or 'layout(' in sample:
                for k, v in obj.items():
                    if isinstance(k, str) and isinstance(v, str):
                        out.setdefault(k, v)

    _scan_module(simulator)
    # Sibling modules that define their own shader strings (best-effort).
    for sibling in ('element_data', 'entity_arena'):
        try:
            mod = __import__(sibling)
            _scan_module(mod)
        except Exception:
            pass
    # Pull per-preset `entity_shaders` and `pass_shaders` dicts as well.
    for preset in simulator.RULE_PRESETS.values():
        if not isinstance(preset, dict):
            continue
        for key in ('entity_shaders', 'pass_shaders', 'extra_shaders'):
            extra = preset.get(key)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if isinstance(k, str) and isinstance(v, str):
                        out.setdefault(k, v)
    return out


def _load_presets() -> dict[str, dict]:
    from simulator import RULE_PRESETS
    return dict(RULE_PRESETS)


def _resolve(rule: str) -> dict:
    from simulator import _resolve_composed_preset
    return _resolve_composed_preset(rule)


# ---------------------------------------------------------------------------
# Shader-level static checks
# ---------------------------------------------------------------------------

# Match a float literal: 0.20, .5, 1., 1e-3, etc.  Avoids matching the
# `0` in `vec3(0)` because we require a decimal point or exponent.
_FLOAT_RE = re.compile(r'(?<![\w.])([+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d+[eE][+-]?\d+))(?![\w.])')

# Detect statements like:  float T_ign = 0.20;  vec3 N = vec3(eps, 0.0, 0.0);
_ASSIGN_LITERAL_RE = re.compile(
    r'\b(?:float|int|vec[234]|ivec[234])\s+(\w+)\s*=\s*([^;]+);'
)

# `vec3(<expr>, 0.0, 0.0)` etc -- axis-locked construction patterns.
_AXIS_LOCKED_VEC3 = [
    re.compile(r'vec3\s*\(\s*([^,()]+?)\s*,\s*0\.?0?\s*,\s*0\.?0?\s*\)'),  # X-locked
    re.compile(r'vec3\s*\(\s*0\.?0?\s*,\s*([^,()]+?)\s*,\s*0\.?0?\s*\)'),  # Y-locked
    re.compile(r'vec3\s*\(\s*0\.?0?\s*,\s*0\.?0?\s*,\s*([^,()]+?)\s*\)'),  # Z-locked
]

_CURL_CONTEXT_HINTS = ('omega', 'curl', 'vortic', 'grad', 'normal', 'N =', 'N=')


def _strip_comments(src: str) -> str:
    """Remove // line comments and /* */ block comments so literal scans
    don't pick up parameter docs / equations / examples in comments."""
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.DOTALL)
    src = re.sub(r'//[^\n]*', '', src)
    return src


def _line_of(src: str, idx: int) -> int:
    return src.count('\n', 0, idx) + 1


# ---------------------------------------------------------------------------
# Per-shader checks
# ---------------------------------------------------------------------------

def _check_uparam_usage(shader_name: str, src: str) -> list[dict]:
    """Find which u_paramN slots a shader actually reads."""
    used = set()
    for m in re.finditer(r'\bu_param(\d+)\b', src):
        used.add(int(m.group(1)))
    return sorted(used)


def _check_image_io(shader_name: str, src: str, preset_kind: str = 'voxel') -> list[dict]:
    """A pass should both read (fetch/imageLoad) and write (imageStore)
    -- except for viewport SDF rules whose 'shader' is a ray-marcher
    that writes to the framebuffer rather than imageStore-ing voxels."""
    findings = []
    has_load = bool(re.search(r'\bimageLoad\s*\(', src) or
                    re.search(r'\bfetch\s*\(', src))
    has_store = bool(re.search(r'\bimageStore\s*\(', src))
    if preset_kind == 'viewport':
        # Viewport rules render directly; not having imageLoad/Store
        # is the correct shape for them.
        return findings
    if not has_store:
        findings.append({
            'severity': 'err',
            'code': 'NO_IMAGESTORE',
            'msg': f"{shader_name}: shader has no imageStore() call",
        })
    if not has_load:
        findings.append({
            'severity': 'warn',
            'code': 'NO_IMAGELOAD',
            'msg': f"{shader_name}: shader has no imageLoad/fetch call",
        })
    return findings


def _check_pair2_usage(shader_name: str, src: str) -> list[dict]:
    """Reads/writes to u_src2/u_dst2 imply extra_fields >= 1 in the preset."""
    findings = []
    if re.search(r'\b(u_src2|u_dst2)\b', src):
        findings.append({
            'severity': 'info',
            'code': 'USES_PAIR2',
            'msg': f"{shader_name}: references second image pair",
            'detail': {'pair2': True},
        })
    return findings


def _check_axis_locked_curl(shader_name: str, src: str) -> list[dict]:
    """Flag vec3(x, 0, 0)-style constructions in curl/grad contexts.

    This is the static signature of the fire vorticity bug: an axis-
    locked N = vec3(grad_x, 0, 0) makes cross(N, omega) collapse into
    a 2D plane and silently kills 3D isotropy.
    """
    findings = []
    src_clean = _strip_comments(src)
    for axis_idx, pat in enumerate(_AXIS_LOCKED_VEC3):
        axis = 'XYZ'[axis_idx]
        for m in pat.finditer(src_clean):
            # only flag if the surrounding ~80 chars mention curl / grad / omega / N
            start, end = m.span()
            ctx = src_clean[max(0, start - 80):min(len(src_clean), end + 40)].lower()
            if any(h.lower() in ctx for h in _CURL_CONTEXT_HINTS):
                expr = m.group(1).strip()
                # Skip stencil basis vectors and other literal-only
                # constructions: vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0.5, 0, 0).
                # We only want to flag REAL expressions like
                # vec3(wind_x, 0, 0) or vec3(grad_x - om_self, 0, 0).
                if re.fullmatch(r'[+-]?\d+(?:\.\d*)?', expr):
                    continue
                if re.fullmatch(_FLOAT_RE, expr):
                    continue
                findings.append({
                    'severity': 'warn',
                    'code': 'AXIS_LOCKED_VEC3',
                    'msg': (f"{shader_name}: axis-locked vec3 (only {axis} non-zero) "
                            f"near curl/grad context -- check for missing components"),
                    'detail': {'expr': expr, 'line': _line_of(src, start), 'axis': axis},
                })
    return findings


def _check_asymmetric_stencil(shader_name: str, src: str) -> list[dict]:
    """In any block that mentions lap/grad/omega, count axial offsets per axis.
    A healthy 3D stencil uses (+1,-1) on each of x/y/z.  Mismatches like
    (+2,0) on one axis are usually typos that produce anisotropic dynamics.
    """
    findings = []
    src_clean = _strip_comments(src)
    # Find blocks of code introduced by `lap`, `grad`, `omega`, `curl` keywords.
    for kw_match in re.finditer(r'\b(lap|grad|omega|curl|vortic)\w*\s*=\s*', src_clean):
        block_start = kw_match.start()
        # block is the next ~600 chars (covers a typical stencil sum) or
        # until the next `;` that isn't inside parens.
        chunk = src_clean[block_start:block_start + 600]
        offsets_per_axis = defaultdict(set)  # axis -> set of int offsets
        # Match ivec3( a , b , c ) literals for the x/y/z offsets.
        for off in re.finditer(
                r'ivec3\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', chunk):
            ox, oy, oz = (int(off.group(i)) for i in (1, 2, 3))
            if ox: offsets_per_axis['x'].add(ox)
            if oy: offsets_per_axis['y'].add(oy)
            if oz: offsets_per_axis['z'].add(oz)
        if len(offsets_per_axis) < 2:
            continue  # not a 3D stencil block -- skip
        # All axes that are touched should have the same offset set
        # (typically {-1, +1}).
        sets = {axis: tuple(sorted(s)) for axis, s in offsets_per_axis.items()}
        unique = set(sets.values())
        if len(unique) > 1:
            findings.append({
                'severity': 'info',
                'code': 'ASYMMETRIC_STENCIL',
                'msg': (f"{shader_name}: stencil offsets differ per axis "
                        f"({sets}) inside `{kw_match.group(0).strip()}` block"),
                'detail': {'offsets': sets, 'line': _line_of(src, block_start)},
            })
    return findings


# ---------------------------------------------------------------------------
# Cross-checks against the preset
# ---------------------------------------------------------------------------

def _check_pass_params_slots(rule: str, preset: dict, shaders: dict[str, str]) -> list[dict]:
    """For multi-pass rules: every u_paramN read by a pass must have a
    matching entry in pass_params[<shader>]."""
    findings = []
    passes = preset.get('passes')
    if not passes:
        return findings
    pp = preset.get('pass_params') or {}
    for p in passes:
        sh = p.get('shader')
        src = shaders.get(sh)
        if not src:
            continue
        used = _check_uparam_usage(sh, src)
        if not used:
            continue
        slot_count = len(pp.get(sh, [])) if sh in pp else 4  # default uniform path = 4 slots
        max_used = max(used)
        if sh in pp and max_used >= slot_count:
            findings.append({
                'severity': 'err',
                'code': 'PASS_PARAMS_SLOT_MISSING',
                'msg': (f"[{rule}] pass `{sh}` reads u_param{max_used} but "
                        f"pass_params[{sh!r}] has only {slot_count} entries"),
                'detail': {'used_slots': used, 'declared': pp.get(sh)},
            })
    return findings


def _check_pair2_vs_extra_fields(rule: str, preset: dict, shaders: dict[str, str]) -> list[dict]:
    findings = []
    passes = preset.get('passes') or [{'shader': preset.get('shader')}]
    extra_fields = int(preset.get('extra_fields', 0) or 0)
    has_field2_init = bool(preset.get('field2_init'))
    # The framework's _needs_field2 hardcodes a back-compat exemption
    # for the historical crystal_growth shader.
    legacy_pair2_shaders = {'crystal_growth'}
    base_shader = preset.get('shader')
    framework_alloc = (extra_fields >= 1 or has_field2_init
                       or base_shader in legacy_pair2_shaders)
    uses_pair2 = False
    for p in passes:
        sh = p.get('shader')
        src = shaders.get(sh)
        if src and re.search(r'\b(u_src2|u_dst2)\b', src):
            uses_pair2 = True
            break
        if p.get('writes') and 'p2' in (p.get('writes') or []):
            uses_pair2 = True
            break
    if uses_pair2 and not framework_alloc:
        findings.append({
            'severity': 'err',
            'code': 'PAIR2_NO_EXTRA_FIELDS',
            'msg': (f"[{rule}] shader uses pair2 (u_src2/u_dst2 or writes:p2) "
                    f"but preset has no extra_fields/field2_init"),
        })
    elif uses_pair2 and extra_fields == 0 and not has_field2_init \
            and base_shader not in legacy_pair2_shaders:
        # Shouldn't happen (above branch covers), here for completeness.
        pass
    elif uses_pair2 and extra_fields == 0 and not has_field2_init:
        # Pair2 used purely via the legacy crystal_growth back-compat shim
        # — not a bug per se but a preset/framework inconsistency worth
        # surfacing as info.
        findings.append({
            'severity': 'info',
            'code': 'PAIR2_VIA_LEGACY_SHIM',
            'msg': (f"[{rule}] uses pair2 only because the framework has a "
                    f"hardcoded exemption for shader=`{base_shader}`"),
        })
    return findings


def _hardcoded_literal_matches_default(
        rule: str, preset: dict, shaders: dict[str, str]) -> list[dict]:
    """Hardcoded float literal inside a shader that exactly matches a
    preset default param value -- the fire `T_ign = 0.20` bug class.

    Heuristics:
      - Only consider assignments to a *named* local var (not arithmetic
        constants like `* 4.0` or `0.5 * vec3(...)`).
      - The value must equal a current preset default to within 1e-6
        AND that default must be tunable (in `param_ranges`).
      - Skip locals whose name appears in any pass_params slot for this
        shader (those are correctly-routed params).
    """
    findings = []
    params = preset.get('params') or {}
    ranges = preset.get('param_ranges') or {}
    if not params:
        return findings
    # Build value -> param-name map for tunable params only.
    val_to_name = defaultdict(list)
    for k, v in params.items():
        if k in ranges and isinstance(v, (int, float)):
            val_to_name[round(float(v), 6)].append(k)
    if not val_to_name:
        return findings

    pp = preset.get('pass_params') or {}
    passes = preset.get('passes') or [{'shader': preset.get('shader')}]
    for p in passes:
        sh = p.get('shader')
        src = shaders.get(sh)
        if not src:
            continue
        legit_routed = set(pp.get(sh, []))
        src_clean = _strip_comments(src)
        for m in _ASSIGN_LITERAL_RE.finditer(src_clean):
            name, rhs = m.group(1), m.group(2).strip()
            # Only flag when the RHS is *just* a float literal (not an
            # expression with multiple terms).
            lit_match = re.fullmatch(_FLOAT_RE, rhs)
            if not lit_match:
                continue
            try:
                val = round(float(lit_match.group(1)), 6)
            except ValueError:
                continue
            # Pure-zero initialisations are almost always accumulator
            # zeroing (`float E = 0.0;`, `int sum = 0;`) and not a
            # stale param copy.  Skip them to avoid the dominant
            # source of false positives.
            if val == 0.0:
                continue
            # Single/two-character locals are usually math symbols
            # (E, T, J, dx) -- too short for a meaningful name match.
            if len(name) < 3:
                continue
            matches = val_to_name.get(val, [])
            # Strip shader-local scoping: a local called `T_ign` matching
            # a preset param `T-ignition` is the suspicious case.  Use a
            # loose alpha-only equality.
            def _norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())
            local_norm = _norm(name)
            for pname in matches:
                pnorm = _norm(pname)
                # Stronger similarity: require either substring
                # containment (>=4 chars) or 5-char prefix match.
                if not ((len(local_norm) >= 4 and local_norm in pnorm)
                        or (len(pnorm) >= 4 and pnorm in local_norm)
                        or (len(local_norm) >= 5 and len(pnorm) >= 5
                            and local_norm[:5] == pnorm[:5])):
                    continue
                if pname in legit_routed:
                    continue
                findings.append({
                    'severity': 'warn',
                    'code': 'HARDCODED_LITERAL_MATCHES_PARAM',
                    'msg': (f"[{rule}] pass `{sh}`: local `{name} = {val}` "
                            f"matches preset default of tunable param "
                            f"`{pname}` (not routed via pass_params)"),
                    'detail': {'local': name, 'value': val,
                               'param': pname, 'line': _line_of(src, m.start())},
                })
    return findings


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_SEV_ORDER = {'err': 0, 'warn': 1, 'info': 2}


def lint_rule(rule: str, shaders: dict[str, str]) -> list[dict]:
    findings: list[dict] = []
    try:
        preset = _resolve(rule)
    except Exception as e:
        return [{'severity': 'err', 'code': 'PRESET_RESOLVE_FAILED',
                 'msg': f"[{rule}] {e}"}]

    # Per-shader checks for every pass this rule uses.
    passes = preset.get('passes') or [{'shader': preset.get('shader')}]
    seen_shaders = set()
    for p in passes:
        sh = p.get('shader')
        if not sh or sh in seen_shaders:
            continue
        seen_shaders.add(sh)
        # Skip passes that are resolved by a non-voxel framework
        # (entity-arena hash/paint/step kernels live in entity_arena.py
        # and don't follow the standard CA_RULES contract).
        kind = (p.get('kind') or 'voxel').lower()
        if kind != 'voxel':
            continue
        # `noop` is the framework's explicit "don't dispatch" sentinel.
        # `element_ca` is a special case: its shader is constructed
        # dynamically via a separate compute header and is not in the
        # CA_RULES dict by design (see `is_element_ca` branch in
        # simulator._build_compute_program).
        if sh in ('noop', 'element_ca'):
            continue
        src = shaders.get(sh)
        if src is None:
            findings.append({'severity': 'err', 'code': 'SHADER_MISSING',
                             'msg': f"[{rule}] pass shader `{sh}` not in CA_RULES"})
            continue
        findings.extend(_check_image_io(sh, src, preset.get('kind', 'voxel')))
        findings.extend(_check_axis_locked_curl(sh, src))
        findings.extend(_check_asymmetric_stencil(sh, src))

    findings.extend(_check_pass_params_slots(rule, preset, shaders))
    findings.extend(_check_pair2_vs_extra_fields(rule, preset, shaders))
    findings.extend(_hardcoded_literal_matches_default(rule, preset, shaders))

    # Tag findings with rule.
    for f in findings:
        f['rule'] = rule
    return findings


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--severity', choices=['err', 'warn', 'info'], default='info',
                    help='Minimum severity to report (default: info = all).')
    ap.add_argument('--json', help='Write findings JSON to this path.')
    ap.add_argument('--codes', help='Comma-separated list of finding codes to keep.')
    args = ap.parse_args(argv)

    shaders = _load_shader_dict()
    presets = _load_presets()

    if args.rules:
        rules = [r.strip() for r in args.rules.split(',') if r.strip()]
    else:
        rules = sorted(presets.keys())

    min_sev = _SEV_ORDER[args.severity]
    keep_codes = set(c.strip() for c in args.codes.split(',')) if args.codes else None

    all_findings: list[dict] = []
    for rule in rules:
        for f in lint_rule(rule, shaders):
            if _SEV_ORDER.get(f['severity'], 99) > min_sev:
                continue
            if keep_codes and f.get('code') not in keep_codes:
                continue
            all_findings.append(f)

    # Sort: severity, then rule, then code
    all_findings.sort(key=lambda f: (_SEV_ORDER.get(f['severity'], 99),
                                     f.get('rule', ''), f.get('code', '')))

    # Print table.
    by_sev = defaultdict(int)
    for f in all_findings:
        by_sev[f['severity']] += 1
        print(f"  {f['severity']:<4}  {f.get('code',''):<32}  {f.get('msg','')}")

    print()
    print(f"Summary: " + " ".join(f"{k}={by_sev[k]}" for k in ('err', 'warn', 'info')))
    unique_shaders: set[str] = set()
    for r in rules:
        try:
            preset = _resolve(r)
        except Exception:
            continue
        for p in (preset.get('passes') or [{'shader': preset.get('shader')}]):
            sh = p.get('shader')
            if sh:
                unique_shaders.add(sh)
    print(f"Scanned {len(rules)} rules, {len(unique_shaders)} unique shader passes")

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(all_findings, fh, indent=2)
        print(f"Wrote {args.json}")

    return 0 if by_sev['err'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
