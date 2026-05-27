"""Probe #18 — dead-shader / legacy-PDE audit.

Walks every entry in ``RULE_PRESETS``, collects the shader keys each
preset will actually dispatch (top-level ``preset['shader']`` only when
no ``passes`` list overrides it, then every key inside ``passes`` of
``kind=='voxel'``, plus everything in ``entity_shaders``), and reports
``CA_RULES`` entries that no preset references.

Motivation:  while writing the ``predator_prey_3d`` analytic oracle I
discovered ``CA_RULES['predator_prey_3d']`` is a legacy PDE shader; the
live preset substitutes an entity-arena particle simulation and never
dispatches the named shader.  That mismatch wasted hours of debugging
and produced an oracle that could never have agreed with reality.  This
probe surfaces every similar trap so future oracle authors (and code
readers) aren't fooled.

Two kinds of finding:

  orphan       CA_RULES key not referenced by *any* preset.
  shadowed     A preset declares ``preset['shader']=X`` but the same
               preset also defines ``passes`` (or ``entity_arena``) so
               that X is never actually dispatched.

Exit code 0 if both lists are empty.

Usage::

    python -m ca_debug.dead_shader_audit
    python -m ca_debug.dead_shader_audit --json /tmp/dead.json
"""
from __future__ import annotations

import argparse
import json
import sys


def _collect_dispatched(preset: dict) -> set[str]:
    """Return shader keys that would actually be dispatched for this preset.

    Mirrors the precedence used by ``HeadlessRunner.__init__``: if the
    preset declares ``passes``, the top-level ``shader`` field is *not*
    dispatched (only the passes are).
    """
    keys: set[str] = set()
    raw_passes = preset.get('passes')
    if raw_passes:
        for entry in raw_passes:
            if isinstance(entry, str):
                keys.add(entry)
            elif isinstance(entry, dict) and entry.get('shader'):
                keys.add(entry['shader'])
    else:
        if preset.get('shader'):
            keys.add(preset['shader'])
    # entity-arena and entity_field shaders live in a separate dict.
    for k in (preset.get('entity_shaders') or {}):
        keys.add(k)
    # particle/agent paint shader override.
    if preset.get('entity_paint_shader'):
        # not a CA_RULES key (raw GLSL body), ignore.
        pass
    return keys


def _shadowed_field(preset: dict) -> str | None:
    """If preset.shader is set but `passes` overrides it, return the key."""
    if preset.get('passes') and preset.get('shader'):
        return preset['shader']
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', help='Write findings JSON.')
    args = ap.parse_args(argv)

    from simulator import CA_RULES, RULE_PRESETS, _resolve_composed_preset

    referenced: dict[str, set[str]] = {}   # key -> set of presets that dispatch it
    shadowed: list[tuple[str, str]] = []   # (preset_name, shadowed_key)
    unresolved: list[tuple[str, str]] = [] # (preset_name, error)

    for name in sorted(RULE_PRESETS.keys()):
        try:
            preset = _resolve_composed_preset(name)
        except Exception as e:  # noqa: BLE001  per-preset compose failure recorded, not fatal
            unresolved.append((name, f'{type(e).__name__}: {e}'))
            continue
        for k in _collect_dispatched(preset):
            referenced.setdefault(k, set()).add(name)
        sh = _shadowed_field(preset)
        if sh is not None:
            shadowed.append((name, sh))

    all_ca_keys = set(CA_RULES.keys())
    orphans = sorted(all_ca_keys - set(referenced.keys()))

    print(f"dead-shader audit — {len(CA_RULES)} CA_RULES, "
          f"{len(RULE_PRESETS)} presets")
    print(f"  referenced keys: {len(referenced)}")
    print(f"  orphan keys:     {len(orphans)}")
    print(f"  shadowed fields: {len(shadowed)}")
    if unresolved:
        print(f"  unresolved presets: {len(unresolved)}")

    if orphans:
        print(f"\nORPHAN CA_RULES keys (defined but no preset dispatches them):")
        for k in orphans:
            body_len = len(CA_RULES[k])
            print(f"  {k:<40} ({body_len} chars)")

    if shadowed:
        print(f"\nSHADOWED preset.shader fields "
              f"(declared but masked by preset['passes']):")
        for preset_name, sh_key in shadowed:
            # Whether the shadowed key is *also* dispatched by another preset.
            also_used = referenced.get(sh_key, set()) - {preset_name}
            tag = '' if also_used else '  ← key never dispatched anywhere'
            print(f"  preset={preset_name:<32} shader={sh_key}{tag}")

    if unresolved:
        print(f"\nUNRESOLVED presets:")
        for name, err in unresolved:
            print(f"  {name:<32} {err}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({
                'orphans': orphans,
                'shadowed': [[n, k] for n, k in shadowed],
                'unresolved': [[n, e] for n, e in unresolved],
                'referenced': {k: sorted(v) for k, v in referenced.items()},
            }, f, indent=2)
        print(f"\nWrote {args.json}")

    return 0 if not orphans and not shadowed else 1


if __name__ == '__main__':
    sys.exit(main())
