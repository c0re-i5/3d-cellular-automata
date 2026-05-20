#!/usr/bin/env python3
"""
audit.py - Read-only health check for the CA discovery corpus.

Four passes (run all by default, or pick individually with --pass N):

  1  Schema audit       - per-entry field presence, missing fields,
                          orphan/typo keys, value sanity (NaN, ranges).
  2  Cross-reference    - derived_from parents resolve, refinement
                          sidecars match entries, disk usage per rule.
  3  Replay sample      - GPU; pick K entries, replay headless, compare
                          recorded score to fresh score. SLOW. Off unless
                          --replay K is given (default 0 = skip).
  4  Code surface       - grep counts for entry.get(), except Exception,
                          subprocess.Popen, moderngl context lifecycle.

Outputs audit_report.md.  Never writes to discoveries.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ─── paths / config ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DISCOVERIES = ROOT / 'discoveries.json'
REFINEMENTS = ROOT / 'refinements'
STATUS_DIR = REFINEMENTS / '.status'
REPORT_PATH = ROOT / 'audit_report.md'

# Fields the writer always stamps (verified against current corpus).
CORE_FIELDS = {
    'rule', 'params', 'dt', 'score', 'seed',
    'final_alive', 'median_alive', 'final_activity', 'mean_activity',
    'final_surface', 'gol_coherence', 'projection_complexity',
    'projection_structure', 'slice_mi', 'spatial_variation',
}
# Fields added with capture_dynamics.
DYNAMICS_FIELDS = {
    'period', 'period_score', 'translation_score', 'translation_speed',
    'growth_score', 'growth_rate', 'growth_type',
    'n_clusters', 'cluster_score', 'symmetry_score', 'translation_dir',
}
# Fields added later in the project lifecycle.
LATER_OPTIONAL = {
    'init_variant', 'init_density', 'derived_from',
    'marked', 'marked_at', 'refinement',
}
# Schema v1 fields (2026-05 audit, Tier 1+2). Always present on new writes
# from schema_version >= 1; absent on historical entries.
V1_FIELDS = {'schema_version', 'size', 'steps', 'rule_code_hash'}
KNOWN_FIELDS = CORE_FIELDS | DYNAMICS_FIELDS | LATER_OPTIONAL | V1_FIELDS

# Fields that *should* be on every entry but historical entries lack
# (kept for the per-entry coverage stat in the report).
PROPOSED_MISSING = ['size', 'steps', 'rule_code_hash', 'schema_version']

# Score should be in [0, 1] for the project's quality metric.
SCORE_RANGE = (0.0, 1.0)


# ─── helpers ────────────────────────────────────────────────────────
def _short_hash(entry: dict) -> str:
    """Replicate the project's entry-hash convention. Must match
    batch_refine._short_hash / refine.short_hash / simulator._refine_short_hash:
    sha1[:10] of json.dumps([rule, sorted((str(k),float(v)) for k,v in params), seed]).
    """
    import hashlib
    rule = entry.get('rule', '')
    params = sorted((str(k), float(v))
                    for k, v in (entry.get('params') or {}).items())
    seed = int(entry.get('seed', 0))
    key = json.dumps([rule, params, seed], sort_keys=True)
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]


def _is_bad_number(v: Any) -> bool:
    return isinstance(v, float) and (math.isnan(v) or math.isinf(v))


def _human_bytes(n: int) -> str:
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.1f} {u}'
        n /= 1024
    return f'{n:.1f} TB'


# ─── pass 1: schema audit ───────────────────────────────────────────
def pass1_schema(entries: list[dict]) -> dict:
    out: dict = {'name': 'Schema audit'}
    key_counts: Counter = Counter()
    for e in entries:
        for k in e.keys():
            key_counts[k] += 1
    out['field_counts'] = dict(key_counts.most_common())

    n = len(entries)
    out['n'] = n

    # Missing core fields → real bugs.
    missing_core = defaultdict(list)
    for i, e in enumerate(entries):
        for f in CORE_FIELDS:
            if f not in e:
                missing_core[f].append(i)
    out['missing_core'] = {f: len(v) for f, v in missing_core.items() if v}
    out['missing_core_examples'] = {
        f: v[:5] for f, v in missing_core.items() if v}

    # Orphan/typo keys → keys appearing on < 0.5% of entries that aren't
    # in KNOWN_FIELDS.
    threshold = max(5, int(0.005 * n))
    orphans = {k: c for k, c in key_counts.items()
               if k not in KNOWN_FIELDS and c < threshold}
    out['orphan_keys'] = orphans

    # Schema shapes per rule: count distinct "key-set signatures" per rule.
    shape_per_rule: dict[str, Counter] = defaultdict(Counter)
    for e in entries:
        sig = frozenset(e.keys())
        shape_per_rule[e.get('rule', '?')][sig] += 1
    drift = {}
    for rule, shapes in shape_per_rule.items():
        if len(shapes) > 1:
            drift[rule] = {
                'distinct_shapes': len(shapes),
                'breakdown': [(c, sorted(sig)) for sig, c in
                              shapes.most_common(3)],
            }
    out['shape_drift_top_rules'] = dict(
        sorted(drift.items(), key=lambda kv: -kv[1]['distinct_shapes'])[:10])
    out['rules_with_schema_drift'] = len(drift)

    # Value sanity.
    bad_score_range = []
    bad_score_nan = []
    bad_seed = []
    bad_params_type = []
    bad_marked_consistency = []
    for i, e in enumerate(entries):
        s = e.get('score')
        if _is_bad_number(s):
            bad_score_nan.append(i)
        elif isinstance(s, (int, float)) and not (
                SCORE_RANGE[0] <= s <= SCORE_RANGE[1]):
            bad_score_range.append((i, s))
        if not isinstance(e.get('seed'), int):
            bad_seed.append(i)
        if not isinstance(e.get('params'), dict):
            bad_params_type.append(i)
        # marked without marked_at, or vice versa
        if e.get('marked') and 'marked_at' not in e:
            bad_marked_consistency.append(('marked_no_timestamp', i))
        if 'marked_at' in e and not e.get('marked'):
            bad_marked_consistency.append(('timestamp_no_mark', i))
    out['bad_score_nan'] = len(bad_score_nan)
    out['bad_score_range'] = bad_score_range[:10]
    out['bad_score_range_count'] = len(bad_score_range)
    out['bad_seed'] = len(bad_seed)
    out['bad_params_type'] = len(bad_params_type)
    out['bad_marked'] = Counter(t for t, _ in bad_marked_consistency)
    out['bad_marked_examples'] = bad_marked_consistency[:10]

    # Proposed-but-absent fields.
    out['proposed_missing'] = PROPOSED_MISSING

    # Schema v1 coverage (Tier 1+2 of 2026-05 audit). Per-field counts of
    # how many entries carry it, plus per-rule_code_hash drift detection
    # against the live shader source.
    v1_counts = {f: sum(1 for e in entries if f in e) for f in V1_FIELDS}
    out['v1_field_coverage'] = v1_counts

    versioned = [e for e in entries if e.get('schema_version') is not None]
    out['v1_entries'] = len(versioned)

    # rule_code_hash drift: for every v1+ entry, compare the stored hash
    # to the current source hash for the same rule. A mismatch means the
    # rule's GLSL has been edited since this discovery was scored — the
    # discovery may no longer reproduce.
    try:
        from simulator import rule_code_hash as _rch
    except Exception:
        _rch = None
    drift_by_rule: Counter = Counter()
    drift_examples = []
    if _rch is not None:
        for i, e in enumerate(versioned):
            stored = e.get('rule_code_hash')
            if not stored:
                continue
            current = _rch(e.get('rule', ''))
            if current is None:
                continue
            if stored != current:
                drift_by_rule[e.get('rule', '?')] += 1
                if len(drift_examples) < 5:
                    drift_examples.append(
                        (i, e.get('rule'), stored, current))
    out['rule_code_drift_by_rule'] = dict(drift_by_rule)
    out['rule_code_drift_examples'] = drift_examples
    out['rule_code_drift_total'] = sum(drift_by_rule.values())

    # Per-rule entry counts (for context).
    out['rule_counts'] = dict(
        Counter(e.get('rule', '?') for e in entries).most_common())

    return out


# ─── pass 2: cross-reference audit ──────────────────────────────────
def pass2_xref(entries: list[dict]) -> dict:
    out: dict = {'name': 'Cross-reference audit'}

    # Index entries by hash (recomputed from the canonical convention).
    by_hash: dict[str, int] = {}
    hash_collisions = 0
    for i, e in enumerate(entries):
        h = _short_hash(e)
        if h in by_hash:
            hash_collisions += 1
        else:
            by_hash[h] = i
    out['n_entries'] = len(entries)
    out['n_unique_hashes'] = len(by_hash)
    out['hash_collisions'] = hash_collisions

    # derived_from.parent_hash resolution.
    derived = [(i, e) for i, e in enumerate(entries) if 'derived_from' in e]
    unresolved = []
    for i, e in derived:
        ph = e['derived_from'].get('parent_hash')
        if not ph:
            unresolved.append((i, 'no_parent_hash'))
        elif ph not in by_hash:
            unresolved.append((i, f'missing_parent:{ph}'))
    out['derived_total'] = len(derived)
    out['derived_unresolved'] = len(unresolved)
    out['derived_unresolved_examples'] = unresolved[:10]

    # refinement entries ↔ refinements/ dirs ↔ status sidecars.
    refined_entries = [(i, e) for i, e in enumerate(entries)
                       if 'refinement' in e]
    out['refined_entries'] = len(refined_entries)

    if REFINEMENTS.exists():
        dirs = {d.name for d in REFINEMENTS.iterdir()
                if d.is_dir() and not d.name.startswith('.')}
    else:
        dirs = set()
    out['refinement_dirs'] = len(dirs)

    # Expected dir name = f"{rule}_{hash}"
    expected_dirs = set()
    refined_entry_hashes = set()
    for i, e in refined_entries:
        h = _short_hash(e)
        refined_entry_hashes.add(h)
        expected_dirs.add(f"{e['rule']}_{h}")

    dirs_without_entry = sorted(dirs - expected_dirs)
    entries_without_dir = sorted(expected_dirs - dirs)
    out['refinement_dirs_orphaned'] = dirs_without_entry
    out['refinement_entries_missing_dir'] = entries_without_dir

    # Status sidecars.
    if STATUS_DIR.exists():
        status_jsons = {f.stem for f in STATUS_DIR.glob('*.json')}
        status_logs = {f.stem for f in STATUS_DIR.glob('*.log')}
        # explore_<hash>.log lives here too; strip prefix for matching
        status_logs_explore = {s for s in status_logs
                               if s.startswith('explore_')}
        status_logs_plain = status_logs - status_logs_explore
    else:
        status_jsons = set()
        status_logs_plain = set()
        status_logs_explore = set()
    out['status_jsons'] = len(status_jsons)
    out['status_logs_plain'] = len(status_logs_plain)
    out['status_logs_explore'] = len(status_logs_explore)
    out['status_logs_without_json'] = sorted(
        status_logs_plain - status_jsons)[:10]
    out['status_jsons_without_entry'] = sorted(
        status_jsons - refined_entry_hashes)[:10]

    # Disk usage per rule prefix.
    per_rule_bytes: Counter = Counter()
    total = 0
    for d in REFINEMENTS.glob('*'):
        if d.is_dir() and not d.name.startswith('.'):
            sz = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            total += sz
            # Strip trailing _<hash> to get rule
            m = re.match(r'^(.*?)_([0-9a-f]{10})$', d.name)
            rule = m.group(1) if m else d.name
            per_rule_bytes[rule] += sz
    out['refinements_total_bytes'] = total
    out['refinements_per_rule'] = dict(per_rule_bytes.most_common())

    return out


# ─── pass 4: code surface audit ─────────────────────────────────────
def pass4_codesurface() -> dict:
    out: dict = {'name': 'Code surface audit'}

    py_files = [p for p in ROOT.glob('*.py') if p.name != 'audit.py']

    patterns = {
        'entry.get_calls': re.compile(r'\bentry\.get\(|disc\.get\(|d\.get\(|e\.get\('),
        # Migration candidates: any dict.get('FIELD', ...) where FIELD is a
        # v1-required discovery field. Receiver-agnostic so we also catch
        # `row.get('score', ...)`, `x.get('rule', ...)` etc.
        'entry.get_v1_field': re.compile(
            r"\.get\(\s*['\"](?:schema_version|rule|params|score|seed|size|steps|rule_code_hash)['\"]"
        ),
        # Migrated sites: anything calling the schema helper.
        'schema_get_field': re.compile(r'\bget_field\s*\('),
        # Any `except Exception:` site, INCLUDING those marked acknowledged
        # via a trailing `# noqa: BLE001` comment. The split between
        # acknowledged and unannotated is computed below.
        'except_exception_bare': re.compile(
            r'except\s+Exception(\s+as\s+\w+)?\s*:'),
        # Subset: bare-excepts explicitly marked as intentional. The
        # `# noqa: BLE001  <reason>` marker mirrors the ruff lint code
        # for blind-except so a future ruff pass agrees with the audit.
        'except_exception_acknowledged': re.compile(
            r'except\s+Exception(\s+as\s+\w+)?\s*:.*#\s*noqa:\s*BLE001'),
        'subprocess_popen': re.compile(r'subprocess\.Popen\b'),
        # Count CALL sites only — strip leading whitespace then skip
        # `def `, `import`, comment lines, and string-prefixed lines.
        'mgl_create_context': re.compile(
            r'(create_headless_context|moderngl\.create_(standalone_)?context)\s*\('),
        'mgl_destroy': re.compile(r'\bdestroy_context\s*\(|\bctx\.release\s*\(|\bglfw\.destroy_window\s*\('),
        'todo_fixme': re.compile(r'#\s*(TODO|FIXME|XXX|HACK)\b'),
    }
    # Per-pattern line filter: regex match against the *stripped* line
    # that, if it matches, excludes the hit. Used to suppress def/import
    # lines from the create/destroy counts.
    skip_patterns = {
        'mgl_create_context': re.compile(r'^\s*(def\s|from\s|import\s|#)'),
        'mgl_destroy':        re.compile(r'^\s*(def\s|from\s|import\s|#)'),
        # Don't count the helper's own definition or its docstring example.
        'schema_get_field':   re.compile(r'^\s*(def\s|#)'),
        # Don't count the regex that defines this pattern (audit.py itself
        # is excluded from py_files, but other meta-references would slip).
        'entry.get_v1_field': re.compile(r'^\s*#'),
    }

    hits: dict[str, list[tuple[str, int, str]]] = {k: [] for k in patterns}
    for p in py_files:
        try:
            text = p.read_text(errors='replace').splitlines()
        except Exception:
            continue
        for lineno, line in enumerate(text, 1):
            for name, rx in patterns.items():
                if rx.search(line):
                    skip_rx = skip_patterns.get(name)
                    if skip_rx is not None and skip_rx.match(line):
                        continue
                    hits[name].append((p.name, lineno, line.strip()[:120]))

    # Summarise.
    out['hit_counts'] = {k: len(v) for k, v in hits.items()}
    out['hits_per_file'] = {
        k: dict(Counter(f for f, _, _ in v).most_common())
        for k, v in hits.items()
    }
    out['sample_hits'] = {
        k: v[:8] for k, v in hits.items()
    }
    # Context lifecycle asymmetry per file.
    #
    # A context is owned by whichever code calls moderngl.create_*; the
    # destroying file can legitimately be different (e.g. snapshot_3d.py
    # consumes a Simulator instance and releases its borrowed `sim.ctx`).
    # So per-file (create != destroy) is NOT in itself a leak signal.
    # We classify each file:
    #   - producer-only (create>0, destroy<create): potential leak
    #   - consumer-only (create==0, destroy>0): borrowed ctx, fine
    #   - mixed (create>0, destroy>=create): fine
    # We also report the project-wide totals so a true net leak surfaces.
    per_file = defaultdict(lambda: {'create': 0, 'destroy': 0})
    for f, _, _ in hits['mgl_create_context']:
        per_file[f]['create'] += 1
    for f, _, _ in hits['mgl_destroy']:
        per_file[f]['destroy'] += 1
    total_create = sum(v['create'] for v in per_file.values())
    total_destroy = sum(v['destroy'] for v in per_file.values())
    suspicious = {
        f: v for f, v in per_file.items()
        if v['create'] > 0 and v['destroy'] < v['create']
    }
    out['context_lifecycle_per_file'] = suspicious
    out['context_lifecycle_totals'] = {
        'create': total_create, 'destroy': total_destroy,
        'net_outstanding': total_create - total_destroy,
    }
    # Bare-except triage per file: total vs explicitly acknowledged
    # (`# noqa: BLE001  <reason>`). Unannotated = total − acknowledged
    # is the "suspicious" residual that should keep shrinking over time.
    bare = defaultdict(lambda: {'total': 0, 'acknowledged': 0})
    for f, _, _ in hits['except_exception_bare']:
        bare[f]['total'] += 1
    for f, _, _ in hits['except_exception_acknowledged']:
        bare[f]['acknowledged'] += 1
    out['bare_except_per_file'] = {
        f: {**v, 'unannotated': v['total'] - v['acknowledged']}
        for f, v in sorted(bare.items())
        if v['total']
    }
    out['bare_except_totals'] = {
        'total': sum(v['total'] for v in bare.values()),
        'acknowledged': sum(v['acknowledged'] for v in bare.values()),
        'unannotated': sum(v['total'] - v['acknowledged']
                           for v in bare.values()),
    }
    # Schema-migration progress per file. The helper's own file is the
    # source of truth and is excluded — its `.get('schema_version', 0)`
    # etc. are the canonical implementation, not migration candidates.
    migr = defaultdict(lambda: {'candidates': 0, 'migrated': 0})
    for f, _, _ in hits['entry.get_v1_field']:
        if f == 'schema.py':
            continue
        migr[f]['candidates'] += 1
    for f, _, _ in hits['schema_get_field']:
        if f == 'schema.py':
            continue
        migr[f]['migrated'] += 1
    out['schema_migration_per_file'] = {
        f: v for f, v in sorted(migr.items())
        if v['candidates'] or v['migrated']
    }
    return out


# ─── pass 3: replay sample (optional, GPU) ──────────────────────────
def pass3_replay(entries: list[dict], k: int, tolerance: float) -> dict:
    out: dict = {'name': f'Replay sample (k={k}, tol={tolerance})'}
    if k <= 0:
        out['skipped'] = True
        return out

    # Stratified sample: try to pick from many rules.
    import random
    rng = random.Random(0xCA0DD)
    per_rule = defaultdict(list)
    for i, e in enumerate(entries):
        per_rule[e.get('rule', '?')].append(i)
    rules = list(per_rule.keys())
    rng.shuffle(rules)
    sample_idx = []
    while len(sample_idx) < k and rules:
        for r in list(rules):
            if not per_rule[r]:
                rules.remove(r)
                continue
            sample_idx.append(per_rule[r].pop(
                rng.randint(0, len(per_rule[r]) - 1)))
            if len(sample_idx) >= k:
                break

    print(f'[audit/pass3] sampling {len(sample_idx)} entries '
          f'across {len(set(entries[i].get("rule") for i in sample_idx))} '
          f'rules', file=sys.stderr)

    # Import the heavy bits only here.
    try:
        from test_harness import (
            create_headless_context, destroy_context, run_trial,
        )
    except Exception as e:
        out['error'] = f'cannot import test_harness: {e}'
        return out

    # Default replay parameters for *legacy* entries that don't record
    # size/steps. v1+ entries carry these fields and we use them verbatim,
    # which is the whole point of the v1 schema: reproducible replay.
    LEGACY_SIZE = 48
    LEGACY_STEPS = 200

    window, ctx = create_headless_context()
    results = {
        'match': 0,           # within tolerance
        'within_5pct': 0,
        'wildly_different': 0,
        'crashed': 0,
        'unloadable': 0,
    }
    legacy_replays = 0
    v1_replays = 0
    deltas = []
    crashes: list[tuple[int, str, str]] = []
    wild: list[tuple[int, str, float, float]] = []
    try:
        for j, idx in enumerate(sample_idx):
            e = entries[idx]
            rule = e.get('rule')
            # v1+ entries record size/steps — use them so the replay
            # reproduces the original conditions. Legacy entries fall
            # back to the small fixed pair (replay is approximate).
            is_v1 = bool(e.get('schema_version', 0))
            size_ = int(e['size']) if is_v1 and 'size' in e else LEGACY_SIZE
            steps_ = int(e['steps']) if is_v1 and 'steps' in e else LEGACY_STEPS
            if is_v1:
                v1_replays += 1
            else:
                legacy_replays += 1
            try:
                fresh = run_trial(
                    ctx, rule, size=size_, steps=steps_,
                    seed=e.get('seed', 0),
                    params=e.get('params', {}),
                    dt=e.get('dt'),
                    init_density=e.get('init_density'),
                    init_override=e.get('init_variant'),
                    verbose=False,
                    capture_dynamics=False,
                )
            except Exception as ex:
                results['crashed'] += 1
                crashes.append((idx, rule, f'{type(ex).__name__}: {ex}'[:120]))
                continue
            recorded = float(e.get('score', 0.0))
            actual = float(fresh.get('score', 0.0))
            delta = abs(recorded - actual)
            rel = delta / max(1e-6, abs(recorded))
            deltas.append((idx, rule, recorded, actual, delta))
            if delta <= tolerance:
                results['match'] += 1
            elif rel <= 0.05:
                results['within_5pct'] += 1
            else:
                results['wildly_different'] += 1
                wild.append((idx, rule, recorded, actual))
            if (j + 1) % 10 == 0:
                print(f'  [{j+1}/{len(sample_idx)}] '
                      f'match={results["match"]} '
                      f'wild={results["wildly_different"]} '
                      f'crash={results["crashed"]}',
                      file=sys.stderr)
    finally:
        destroy_context(window)

    out['sampled'] = len(sample_idx)
    out['results'] = results
    out['v1_replays'] = v1_replays
    out['legacy_replays'] = legacy_replays
    out['note'] = (
        f'Replayed {v1_replays} v1+ entries at their recorded size/steps '
        f'and {legacy_replays} legacy entries at fallback '
        f'size={LEGACY_SIZE}, steps={LEGACY_STEPS} (legacy corpus does '
        f'not record these, so replay is approximate).'
    )
    out['wildly_different_examples'] = wild[:15]
    out['crashes_examples'] = crashes[:15]
    if deltas:
        ds = sorted(d[4] for d in deltas)
        out['delta_median'] = ds[len(ds) // 2]
        out['delta_p95'] = ds[int(len(ds) * 0.95)]
        out['delta_max'] = ds[-1]
    return out


# ─── report renderer ────────────────────────────────────────────────
def render(passes: list[dict]) -> str:
    lines = [
        '# CA Engine Audit Report',
        '',
        f'_Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}_',
        '',
        'Read-only audit of discoveries.json + refinements/. '
        'See `audit.py` for methodology.',
        '',
    ]
    for p in passes:
        lines.append(f'## {p["name"]}')
        lines.append('')
        if p.get('skipped'):
            lines.append('_skipped_')
            lines.append('')
            continue
        if 'error' in p:
            lines.append(f'**ERROR:** {p["error"]}')
            lines.append('')
            continue
        # Pass-specific rendering.
        if p['name'].startswith('Schema'):
            _render_schema(lines, p)
        elif p['name'].startswith('Cross'):
            _render_xref(lines, p)
        elif p['name'].startswith('Code'):
            _render_code(lines, p)
        elif p['name'].startswith('Replay'):
            _render_replay(lines, p)
        lines.append('')
    return '\n'.join(lines)


def _render_schema(lines, p):
    n = p['n']
    lines.append(f'- Entries: **{n:,}**')
    lines.append('')
    lines.append('### Field presence')
    lines.append('')
    lines.append('| field | count | % |')
    lines.append('|---|---:|---:|')
    for f, c in p['field_counts'].items():
        pct = 100 * c / n
        mark = '' if f in KNOWN_FIELDS else '  ⚠ unknown'
        lines.append(f'| {f}{mark} | {c:,} | {pct:.1f}% |')
    lines.append('')
    if p['missing_core']:
        lines.append('### ⚠ Missing CORE fields')
        lines.append('')
        for f, c in p['missing_core'].items():
            ex = p['missing_core_examples'].get(f, [])
            lines.append(f'- `{f}`: missing on {c} entries '
                         f'(examples: {ex})')
        lines.append('')
    else:
        lines.append('All core fields present on every entry.')
        lines.append('')
    if p['orphan_keys']:
        lines.append('### ⚠ Orphan / possible-typo keys '
                     '(< 0.5% of entries, not in known schema)')
        lines.append('')
        for k, c in sorted(p['orphan_keys'].items(), key=lambda kv: -kv[1]):
            lines.append(f'- `{k}`: {c} entries')
        lines.append('')
    lines.append('### Proposed-but-absent fields '
                 '(blockers for full replay verification)')
    lines.append('')
    n_total = p['n']
    for f in p['proposed_missing']:
        have = p['v1_field_coverage'].get(f, 0)
        pct = (100.0 * have / n_total) if n_total else 0.0
        if have == 0:
            lines.append(f'- `{f}` — never recorded')
        elif have == n_total:
            lines.append(f'- `{f}` — ✓ on all {have:,} entries')
        else:
            lines.append(f'- `{f}` — on {have:,}/{n_total:,} '
                         f'entries ({pct:.1f}%, schema v1+ only)')
    lines.append('')

    lines.append('### Schema v1 coverage')
    lines.append('')
    lines.append(f'- v1+ entries (have `schema_version`): '
                 f'**{p["v1_entries"]:,}** of {n_total:,}')
    lines.append('')

    lines.append('### Rule code drift (v1+ entries)')
    lines.append('')
    if p['v1_entries'] == 0:
        lines.append('_No v1+ entries yet — drift check is a no-op until '
                     'new discoveries are written._')
    elif p['rule_code_drift_total'] == 0:
        lines.append(f'✓ All {p["v1_entries"]:,} v1+ entries match current '
                     'shader source.')
    else:
        lines.append(f'⚠ **{p["rule_code_drift_total"]:,}** v1+ entries '
                     'reference a rule whose GLSL has been edited since the '
                     'discovery was scored. Replay may not reproduce.')
        for rule, c in sorted(p['rule_code_drift_by_rule'].items(),
                              key=lambda kv: -kv[1]):
            lines.append(f'- `{rule}`: {c} entries')
        if p['rule_code_drift_examples']:
            lines.append('')
            lines.append('Examples (idx, rule, stored → current):')
            for i, rule, stored, current in p['rule_code_drift_examples']:
                lines.append(f'    - #{i} `{rule}`: '
                             f'`{stored}` → `{current}`')
    lines.append('')
    lines.append('### Schema drift per rule')
    lines.append('')
    lines.append(f'{p["rules_with_schema_drift"]} of '
                 f'{len(p["rule_counts"])} rules have entries with '
                 'multiple distinct key-set shapes.')
    lines.append('')
    if p['shape_drift_top_rules']:
        lines.append('Top 10 most-drifted rules:')
        lines.append('')
        for rule, info in p['shape_drift_top_rules'].items():
            lines.append(f'- **{rule}** ({info["distinct_shapes"]} shapes)')
            for cnt, sig in info['breakdown']:
                missing = sorted(set(KNOWN_FIELDS) - set(sig))[:6]
                extra = sorted(set(sig) - KNOWN_FIELDS)[:6]
                lines.append(f'    - {cnt} entries  '
                             f'missing={missing}  extra={extra}')
        lines.append('')
    lines.append('### Value sanity')
    lines.append('')
    lines.append(f'- NaN/inf scores: {p["bad_score_nan"]}')
    lines.append(f'- Scores outside [0,1]: {p["bad_score_range_count"]}  '
                 f'examples: {p["bad_score_range"]}')
    lines.append(f'- Non-int seeds: {p["bad_seed"]}')
    lines.append(f'- Non-dict params: {p["bad_params_type"]}')
    lines.append(f'- Marked inconsistencies: {dict(p["bad_marked"])}  '
                 f'examples: {p["bad_marked_examples"]}')


def _render_xref(lines, p):
    lines.append(f'- Entries: {p["n_entries"]:,}')
    lines.append(f'- Unique hashes: {p["n_unique_hashes"]:,}')
    lines.append(f'- Hash collisions: '
                 f'{p["hash_collisions"]}  '
                 f'_(non-zero means two entries share rule+params+seed)_')
    lines.append('')
    lines.append('### `derived_from` chain integrity')
    lines.append('')
    lines.append(f'- {p["derived_total"]:,} entries reference a parent')
    lines.append(f'- {p["derived_unresolved"]} cannot resolve their parent_hash')
    if p['derived_unresolved_examples']:
        lines.append('')
        lines.append('Examples (entry_idx, reason):')
        for ex in p['derived_unresolved_examples']:
            lines.append(f'  - {ex}')
    lines.append('')
    lines.append('### Refinement ↔ filesystem')
    lines.append('')
    lines.append(f'- entries with `refinement` field: '
                 f'{p["refined_entries"]}')
    lines.append(f'- refinements/ subdirs:           '
                 f'{p["refinement_dirs"]}')
    lines.append(f'- status JSON sidecars:           '
                 f'{p["status_jsons"]}')
    lines.append(f'- status .log files (refine):     '
                 f'{p["status_logs_plain"]}')
    lines.append(f'- status .log files (explore):    '
                 f'{p["status_logs_explore"]}')
    if p['refinement_dirs_orphaned']:
        lines.append('')
        lines.append('**Orphaned refinement dirs '
                     '(no matching entry):**')
        for d in p['refinement_dirs_orphaned']:
            lines.append(f'  - `{d}`')
    if p['refinement_entries_missing_dir']:
        lines.append('')
        lines.append('**Entries claim a refinement but dir is missing:**')
        for d in p['refinement_entries_missing_dir']:
            lines.append(f'  - `{d}`')
    if p['status_logs_without_json']:
        lines.append('')
        lines.append('**Status logs without corresponding sidecar JSON:**')
        for d in p['status_logs_without_json']:
            lines.append(f'  - `{d}`')
    if p['status_jsons_without_entry']:
        lines.append('')
        lines.append('**Sidecar JSONs without an entry having `refinement`:**')
        for d in p['status_jsons_without_entry']:
            lines.append(f'  - `{d}`')
    lines.append('')
    lines.append('### Disk usage')
    lines.append('')
    lines.append(f'Total in `refinements/`: '
                 f'**{_human_bytes(p["refinements_total_bytes"])}**')
    lines.append('')
    lines.append('| rule | bytes |')
    lines.append('|---|---:|')
    for r, b in p['refinements_per_rule'].items():
        lines.append(f'| {r} | {_human_bytes(b)} |')


def _render_code(lines, p):
    lines.append('### Counts')
    lines.append('')
    lines.append('| pattern | hits |')
    lines.append('|---|---:|')
    for k, c in p['hit_counts'].items():
        lines.append(f'| {k} | {c} |')
    lines.append('')
    lines.append('### Per-file breakdown')
    lines.append('')
    for k, per_file in p['hits_per_file'].items():
        if not per_file:
            continue
        lines.append(f'**{k}**')
        for f, c in per_file.items():
            lines.append(f'  - {f}: {c}')
    lines.append('')
    totals = p.get('context_lifecycle_totals') or {}
    if totals:
        net = totals.get('net_outstanding', 0)
        marker = '⚠ ' if net > 0 else ''
        lines.append(f'### {marker}GPU context lifecycle totals')
        lines.append('')
        lines.append(f'- creates: **{totals["create"]}**  releases: **{totals["destroy"]}**  '
                     f'net outstanding: **{net}** '
                     f'({"likely leak" if net > 0 else "balanced or consumer-heavy"})')
        lines.append('')
    if p['context_lifecycle_per_file']:
        lines.append('### ⚠ Producer files with unreleased contexts')
        lines.append('')
        lines.append('_Files that call `moderngl.create_*` more times than they '
                     'release. Consumer-only files (release a borrowed `sim.ctx`) '
                     'are excluded._')
        lines.append('')
        for f, v in p['context_lifecycle_per_file'].items():
            lines.append(f'- {f}: create={v["create"]} destroy={v["destroy"]}')
        lines.append('')
    if p.get('bare_except_per_file'):
        tot = p['bare_except_totals']
        marker = '⚠ ' if tot['unannotated'] else '✓ '
        lines.append(f'### {marker}Bare-except triage')
        lines.append('')
        lines.append('_Every `except Exception:` is either narrowed to a '
                     'specific exception type, or annotated with '
                     '`# noqa: BLE001  <reason>` to mark it as intentional '
                     'defensive cleanup. Unannotated sites are the '
                     'residual hygiene debt._')
        lines.append('')
        lines.append(f'- total bare-except sites: **{tot["total"]}**')
        lines.append(f'- acknowledged (`# noqa: BLE001`): **{tot["acknowledged"]}**')
        lines.append(f'- **unannotated: {tot["unannotated"]}**')
        lines.append('')
        lines.append('| file | total | acknowledged | unannotated |')
        lines.append('|---|---:|---:|---:|')
        for f, v in p['bare_except_per_file'].items():
            lines.append(f'| {f} | {v["total"]} | {v["acknowledged"]} | '
                         f'{v["unannotated"]} |')
        lines.append('')
    if p.get('schema_migration_per_file'):
        lines.append('### Schema-aware field access migration')
        lines.append('')
        lines.append('_Per file: legacy `.get("FIELD", default)` candidates '
                     'vs migrated `get_field(...)` call sites. '
                     '100% means all v1-field accesses go through the helper. '
                     'Note: the regex cannot distinguish discovery entries '
                     'from worker-status / preview dicts that happen to '
                     'share field names (`rule`, `score`, ...), so a small '
                     'irreducible noise floor remains per file._')
        lines.append('')
        lines.append('| file | candidates | migrated | progress |')
        lines.append('|---|---:|---:|---:|')
        total_c = total_m = 0
        for f, v in p['schema_migration_per_file'].items():
            c, m = v['candidates'], v['migrated']
            total_c += c
            total_m += m
            denom = c + m
            pct = f'{100 * m / denom:.0f}%' if denom else '—'
            lines.append(f'| {f} | {c} | {m} | {pct} |')
        denom = total_c + total_m
        pct = f'{100 * total_m / denom:.0f}%' if denom else '—'
        lines.append(f'| **TOTAL** | **{total_c}** | **{total_m}** | **{pct}** |')
        lines.append('')
    lines.append('### Sample hits (first 8 per pattern)')
    lines.append('')
    for k, samples in p['sample_hits'].items():
        if not samples:
            continue
        lines.append(f'**{k}**')
        for f, ln, txt in samples:
            lines.append(f'  - `{f}:{ln}`  {txt}')
        lines.append('')


def _render_replay(lines, p):
    if 'error' in p:
        return
    lines.append(p.get('note', ''))
    lines.append('')
    lines.append(f'Sampled: {p["sampled"]}')
    lines.append('')
    r = p['results']
    lines.append(f'- exact match (within tol): **{r["match"]}**')
    lines.append(f'- within 5%:                {r["within_5pct"]}')
    lines.append(f'- wildly different:         **{r["wildly_different"]}**')
    lines.append(f'- crashed:                  **{r["crashed"]}**')
    lines.append('')
    if 'delta_median' in p:
        lines.append(f'Δscore median={p["delta_median"]:.4f}  '
                     f'p95={p["delta_p95"]:.4f}  '
                     f'max={p["delta_max"]:.4f}')
        lines.append('')
    if p['wildly_different_examples']:
        lines.append('**Wildly different examples (idx, rule, recorded, actual):**')
        for ex in p['wildly_different_examples']:
            lines.append(f'  - {ex}')
        lines.append('')
    if p['crashes_examples']:
        lines.append('**Crashes (idx, rule, error):**')
        for ex in p['crashes_examples']:
            lines.append(f'  - {ex}')
        lines.append('')


# ─── main ───────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--passes', default='1,2,4',
                    help='Comma-separated pass numbers to run '
                         '(default: 1,2,4; "all" runs 1,2,3,4).')
    ap.add_argument('--replay', type=int, default=0,
                    help='Replay sample size for pass 3 '
                         '(implies --passes includes 3). '
                         'Default 0 = skip.')
    ap.add_argument('--replay-tolerance', type=float, default=0.01,
                    help='Absolute Δscore considered a match '
                         '(default 0.01).')
    ap.add_argument('--out', default=str(REPORT_PATH),
                    help=f'Report path (default {REPORT_PATH}).')
    ap.add_argument('--input', default=str(DISCOVERIES),
                    help=f'Discoveries JSON to audit '
                         f'(default {DISCOVERIES.name}). Useful for '
                         f'validating fresh sweep output before merge.')
    args = ap.parse_args()

    if args.passes.lower() == 'all':
        wanted = {1, 2, 3, 4}
    else:
        wanted = {int(x) for x in args.passes.split(',') if x.strip()}
    if args.replay > 0:
        wanted.add(3)

    discoveries_path = Path(args.input)
    if not discoveries_path.exists():
        print(f'[audit] FATAL: {discoveries_path} not found', file=sys.stderr)
        return 1
    t0 = time.time()
    print(f'[audit] loading {discoveries_path} ...', file=sys.stderr)
    with open(discoveries_path) as f:
        entries = json.load(f)
    print(f'[audit] loaded {len(entries):,} entries '
          f'in {time.time()-t0:.1f}s', file=sys.stderr)

    results = []
    if 1 in wanted:
        print('[audit] pass 1: schema ...', file=sys.stderr)
        results.append(pass1_schema(entries))
    if 2 in wanted:
        print('[audit] pass 2: cross-reference ...', file=sys.stderr)
        results.append(pass2_xref(entries))
    if 4 in wanted:
        print('[audit] pass 4: code surface ...', file=sys.stderr)
        results.append(pass4_codesurface())
    if 3 in wanted:
        print('[audit] pass 3: replay sample (GPU, slow) ...',
              file=sys.stderr)
        results.append(pass3_replay(entries, args.replay,
                                    args.replay_tolerance))

    report = render(results)
    Path(args.out).write_text(report)
    print(f'[audit] wrote {args.out}  '
          f'({len(report):,} bytes, {time.time()-t0:.1f}s total)',
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
