#!/usr/bin/env python3
"""Tier 4c: legacy backfill of size/steps on pre-v1 discoveries.

Pre-v1 entries in `discoveries.json` (schema_version absent) record
parameters and score but no grid geometry. The audit's pass-3 replay
already assumed historical defaults (size=48, steps=200) for these;
this script makes that assumption explicit in the data itself so
consumers can read size/steps uniformly across the corpus without
branching on schema version.

The backfill:
  * adds `size: 48` and `steps: 200` to entries missing them
  * adds `_legacy_backfill: {'size': 'default', 'steps': 'default',
                              'script': 'backfill_legacy_discoveries.py'}`
    so the provenance is queryable
  * leaves `schema_version` ABSENT (these entries still lack a
    rule_code_hash, so bit-exact replay is NOT guaranteed — v1
    semantics require the hash)
  * leaves v1+ entries untouched

A timestamped backup is written before any modification.

Usage:
    python scripts/backfill_legacy_discoveries.py [path-to-discoveries.json]
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

LEGACY_SIZE = 48      # mirrors audit.py pass3 fallback
LEGACY_STEPS = 200    # mirrors audit.py pass3 fallback

PROVENANCE = {
    'size': 'default',
    'steps': 'default',
    'script': 'backfill_legacy_discoveries.py',
}


def backfill(path: Path) -> dict:
    entries = json.loads(path.read_text())
    counts = {
        'total': len(entries),
        'v1_kept': 0,
        'legacy_backfilled': 0,
        'legacy_already_backfilled': 0,
        'partial_already_present': 0,
    }
    for e in entries:
        if e.get('schema_version', 0) >= 1:
            counts['v1_kept'] += 1
            continue
        if '_legacy_backfill' in e:
            counts['legacy_already_backfilled'] += 1
            continue
        has_size = 'size' in e
        has_steps = 'steps' in e
        if has_size and has_steps:
            # Already has both for some reason — record provenance only.
            counts['partial_already_present'] += 1
            e['_legacy_backfill'] = {
                **PROVENANCE,
                'size': 'preexisting',
                'steps': 'preexisting',
            }
            continue
        if not has_size:
            e['size'] = LEGACY_SIZE
        if not has_steps:
            e['steps'] = LEGACY_STEPS
        e['_legacy_backfill'] = {
            'size': 'default' if not has_size else 'preexisting',
            'steps': 'default' if not has_steps else 'preexisting',
            'script': 'backfill_legacy_discoveries.py',
        }
        counts['legacy_backfilled'] += 1
    return counts, entries


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else 'discoveries.json')
    if not src.exists():
        sys.exit(f'no such file: {src}')

    ts = time.strftime('%Y%m%d_%H%M%S')
    backup = src.with_name(f'{src.stem}_backup_pre_backfill_{ts}.json')
    shutil.copy2(src, backup)
    print(f'backup: {backup}  ({backup.stat().st_size:,} bytes)')

    counts, entries = backfill(src)
    print('counts:')
    for k, v in counts.items():
        print(f'  {k:30s} {v:>7}')

    if counts['legacy_backfilled'] == 0 and counts['partial_already_present'] == 0:
        print('nothing to write')
        return

    # Atomic write: tmp + replace.
    tmp = src.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.replace(src)
    print(f'wrote: {src}  ({src.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
