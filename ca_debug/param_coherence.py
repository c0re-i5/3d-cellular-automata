"""param-spec ↔ shader-uniform coherence audit.

Probe #10 — every rule preset declares a ``params`` dict whose keys
appear as named sliders in the GUI. Each pass binds the first 4 entries
of that dict to ``u_param0..u_param3`` (overridable per-pass via
``preset['pass_params'][shader]`` or a pass entry's ``param_names``).

Two classes of bug this catches:

  - **err** — a shader reads ``u_param`` at a slot index that is NOT
    bound for that pass (out-of-range read).  The uniform is still
    declared by the COMPUTE_HEADER so it links cleanly, but its value
    is whatever stale write the previous program left — usually 0.
    The user sees a slider that silently controls nothing.

  - **warn** — the preset declares a param whose slot is never read by
    ANY pass of the rule.  The GUI exposes a slider that does nothing.

Both surface the same user-visible symptom (slider doesn't change the
simulation) but have very different cures: ``err`` means the shader was
written assuming a slot it doesn't get; ``warn`` means the preset has
an extra param entry.

Audit scope (conservative — engine has many param-binding back-doors
that this probe does NOT model):

  - Skipped: ``kind == 'viewport'`` rules — fractal raymarchers that
    bind named params (``Box scale``, ``Folds``, ``Min radius``,
    ``Julia c{x,y,z}``, ``Zoom rate``) directly to dedicated
    ``u_aux_a/u_aux_b/u_aux3/u_origin/u_zoom`` uniforms.
  - Skipped: ``particle_count``-bearing rules — SSBO-driven particle
    compute shaders not exposed via ``preset['passes']``.
  - Skipped: ``'entity_arena' in preset`` rules — entity-step bodies
    live in ``preset['entity_shaders']`` and read params via a
    different uniform layout.
  - Audited: voxel passes (shader from ``CA_RULES``) and agent passes
    (shader from ``AGENT_RULES``).
  - Param names beginning with ``_`` are skipped from the warn check
    (placeholder convention used for reserved/unused slider slots).

Usage::

    python -m ca_debug.param_coherence
    python -m ca_debug.param_coherence --rules nca_3d,gray_scott_3d
    python -m ca_debug.param_coherence --severity warn
    python -m ca_debug.param_coherence --json /tmp/pcoh.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time


_SEV_ORDER = {'err': 0, 'warn': 1, 'ok': 2, 'n/a': 3, 'skip': 4}

_UPARAM_RE = re.compile(r'\bu_param([0-9]+)\b')


def _shader_for(kind: str, key: str, ca_rules: dict, agent_rules: dict):
    """Return shader source for a (kind, key) pass, or None if not auditable."""
    if kind == 'voxel':
        return ca_rules.get(key)
    if kind == 'agent':
        return agent_rules.get(key)
    return None  # entity_*, particle, etc.


def _slots_read(shader_src: str) -> set[int]:
    """Return the set of u_paramK indices the shader source reads."""
    return {int(m.group(1)) for m in _UPARAM_RE.finditer(shader_src)
            # ignore the COMPUTE_HEADER declaration lines themselves;
            # those appear in headers but the slot is "available" only
            # if the shader body uses it.  Filter out 'uniform float
            # u_paramK;' lines by checking for a leading 'uniform'.
            if not _is_uniform_decl(m, shader_src)}


def _is_uniform_decl(m, src: str) -> bool:
    """True if the match is inside a 'uniform float u_paramK;' decl."""
    line_start = src.rfind('\n', 0, m.start()) + 1
    line_end = src.find('\n', m.end())
    if line_end < 0:
        line_end = len(src)
    line = src[line_start:line_end]
    return line.lstrip().startswith('uniform ')


def _bound_slots_for_pass(preset: dict, pass_entry,
                          fallback_param_names: list[str]) -> list[str]:
    """Return the ordered list of param-name bindings for a single pass.

    The list is length-≤4 (the engine binds u_param0..u_param3).
    Element K is the param name bound to ``u_paramK`` for this pass,
    or ``None`` if that slot is unbound.
    """
    # Per-pass overrides:
    if isinstance(pass_entry, dict):
        shader_name = pass_entry.get('shader')
        names = pass_entry.get('param_names')
        if names is not None:
            return list(names)[:4] + [None] * max(0, 4 - len(names))
    else:
        shader_name = pass_entry
    pp = preset.get('pass_params') or {}
    if shader_name in pp:
        names = pp[shader_name]
        return list(names)[:4] + [None] * max(0, 4 - len(names))
    # Default binding: first 4 entries of preset['params'].
    out = list(fallback_param_names[:4])
    out += [None] * max(0, 4 - len(out))
    return out


def _audit_rule(rule: str, ca_rules: dict, agent_rules: dict,
                resolve_composed) -> dict:
    try:
        preset = resolve_composed(rule)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'reason': f'preset resolve: {type(e).__name__}: {e}'}

    # Skip non-auditable rule classes.
    if preset.get('kind') == 'viewport':
        return {'rule': rule, 'grade': 'skip',
                'reason': 'viewport kind — params bound by name to u_aux_*'}
    if preset.get('particle_count'):
        return {'rule': rule, 'grade': 'skip',
                'reason': 'particle rule — params consumed by SSBO compute'}
    if 'entity_arena' in preset:
        return {'rule': rule, 'grade': 'skip',
                'reason': 'entity-arena rule — params bound in entity_step'}

    params = preset.get('params') or {}
    if not params:
        return {'rule': rule, 'grade': 'n/a', 'reason': 'no params declared'}
    param_names = list(params.keys())

    # Enumerate passes.
    if 'passes' in preset:
        passes = list(preset['passes'])
    else:
        passes = [preset.get('shader')]

    per_pass: list[dict] = []
    union_read_param_names: set[str] = set()
    err_flags: list[str] = []
    any_audited = False

    for idx, entry in enumerate(passes):
        if isinstance(entry, dict):
            kind = entry.get('kind', 'voxel')
            key = entry.get('shader')
        else:
            kind = 'voxel'
            key = entry
        src = _shader_for(kind, key, ca_rules, agent_rules)
        if src is None:
            per_pass.append({'idx': idx, 'kind': kind, 'shader': key,
                             'audited': False,
                             'reason': f'kind={kind} not audited'})
            continue
        any_audited = True
        bound = _bound_slots_for_pass(preset, entry, param_names)
        reads = _slots_read(src)
        # err: shader reads a slot beyond what was bound.
        oob = sorted(k for k in reads if k >= 4 or bound[k] is None)
        for k in oob:
            err_flags.append(
                f'pass {idx} ({kind}/{key}) reads u_param{k} but slot is unbound')
        # Track which preset-named params are actually consumed.
        for k in reads:
            if k < 4 and bound[k] is not None:
                union_read_param_names.add(bound[k])
        per_pass.append({'idx': idx, 'kind': kind, 'shader': key,
                         'audited': True,
                         'bound': bound,
                         'reads': sorted(reads)})

    if not any_audited:
        return {'rule': rule, 'grade': 'skip',
                'reason': 'no passes have a shader we can audit',
                'passes': per_pass}

    # warn: any first-4 param name (non-underscore) that no pass reads.
    auditable_param_names = [n for n in param_names[:4]
                             if not (n or '').startswith('_')]
    orphans = [n for n in auditable_param_names
               if n not in union_read_param_names]

    out = {
        'rule': rule,
        'params':  param_names,
        'passes':  per_pass,
        'read_params': sorted(union_read_param_names),
        'orphans': orphans,
        'errors':  err_flags,
    }
    if err_flags:
        out['grade'] = 'err'
        out['reason'] = err_flags[0]
    elif orphans:
        out['grade'] = 'warn'
        out['reason'] = f'preset params never read by any pass: {orphans}'
    else:
        out['grade'] = 'ok'
    return out


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = sorted(RULE_PRESETS.keys())
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='warn',
                    help='Min severity to print (default: warn).')
    ap.add_argument('--json', help='Write per-rule report JSON to this path.')
    args = ap.parse_args(argv)

    from simulator import (
        AGENT_RULES, CA_RULES, _resolve_composed_preset,
    )

    rules = _select_rules(args)
    t0 = time.perf_counter()
    rows = [_audit_rule(r, CA_RULES, AGENT_RULES, _resolve_composed_preset)
            for r in rules]
    elapsed = time.perf_counter() - t0

    counts = {k: 0 for k in _SEV_ORDER}
    for row in rows:
        counts[row['grade']] = counts.get(row['grade'], 0) + 1

    print(f'\nparam-coherence probe — {len(rules)} rules in {elapsed*1000:.0f}ms')
    print('  by rule:')
    for g in ('err', 'warn', 'ok', 'n/a', 'skip'):
        print(f'    {g:<5}  {counts.get(g, 0):>4}')

    sev_cap = _SEV_ORDER[args.severity]
    flagged = [r for r in rows if _SEV_ORDER.get(r['grade'], 9) <= sev_cap]
    if flagged:
        print(f'\nflagged ({args.severity}+):')
        for row in flagged:
            print(f'  [{row["grade"]:<4}] {row["rule"]}')
            if row.get('reason'):
                print(f'          {row["reason"]}')
            for e in (row.get('errors') or [])[:6]:
                if e != row.get('reason'):
                    print(f'          ! {e}')
            if row.get('orphans'):
                print(f'          params: {row.get("params")}')
                print(f'          read:   {row.get("read_params")}')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({
                'elapsed_s': elapsed,
                'counts': counts,
                'rows': rows,
            }, fh, indent=2, default=str)
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
