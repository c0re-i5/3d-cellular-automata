"""Discovery-entry schema and strict/lenient field access.

A single source of truth for the on-disk discovery dict layout. Whenever
we bump ``DISCOVERY_SCHEMA_VERSION`` (in ``test_harness.py``), the
required-field set for that version lives here.

Why ``get_field`` instead of plain ``entry.get(...)``:

- Legacy entries (no ``schema_version``) silently return the supplied
  default — exactly like the old ``.get()`` calls. No breakage.
- v1+ entries are contract-checked: a missing required field raises
  ``KeyError`` instead of silently substituting ``0`` / ``'?'`` and
  corrupting downstream sorts / aggregates / displays.

Use this only on dicts that are *meant* to be discovery entries (the
ones written by ``_make_discovery`` and friends). Worker status payloads,
preview dicts, dashboard caches, and other ad-hoc structures keep using
plain ``dict.get`` — they have their own schemas.
"""

from __future__ import annotations

from typing import Any, Mapping

# Fields guaranteed present on every v1 discovery entry. Keep in sync with
# ``_make_discovery`` in test_harness.py and ``_save_current_to_discoveries``
# in simulator.py.
REQUIRED_V1: frozenset[str] = frozenset({
    'schema_version',
    'rule',
    'params',
    'score',
    'seed',
    'size',
    'steps',
    'rule_code_hash',
})


class SchemaViolation(KeyError):
    """A v1+ entry is missing a field its schema version requires."""


def get_field(
    entry: Mapping[str, Any],
    name: str,
    default: Any = None,
    *,
    strict_v1: bool = True,
) -> Any:
    """Discovery-aware field access.

    - If ``name`` is present, returns it.
    - Otherwise, if ``strict_v1`` and the entry advertises
      ``schema_version >= 1`` and ``name`` is in ``REQUIRED_V1``,
      raises ``SchemaViolation``.
    - Otherwise returns ``default`` (legacy / non-required field).
    """
    if name in entry:
        return entry[name]
    if strict_v1:
        sv = entry.get('schema_version', 0)
        if isinstance(sv, int) and sv >= 1 and name in REQUIRED_V1:
            raise SchemaViolation(
                f"v{sv} discovery entry missing required field {name!r} "
                f"(hash={entry.get('hash', '?')!r}, "
                f"rule={entry.get('rule', '?')!r})"
            )
    return default
