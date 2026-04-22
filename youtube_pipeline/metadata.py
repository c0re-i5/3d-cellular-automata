"""Build YouTube title / description / tags from a recording sidecar JSON."""
from __future__ import annotations

import json
from pathlib import Path

REPO_URL = 'https://github.com/c0re-i5/3d-cellular-automata'

# YouTube hard limits.
TITLE_MAX = 100
DESCRIPTION_MAX = 5000
TAG_MAX_TOTAL = 500   # combined character count across all tags


def _is_shorts(resolution: list[int]) -> bool:
    """Vertical aspect ratio (9:16-ish) → Shorts."""
    if not resolution or len(resolution) != 2:
        return False
    w, h = resolution
    return h > w


def build_metadata(sidecar_path: Path) -> dict:
    """Return a dict with ``title``, ``description``, ``tags``, ``shorts``.

    Honours an optional ``<basename>_overrides.json`` next to the sidecar
    so any field can be hand-tuned without editing this module.
    """
    meta = json.loads(sidecar_path.read_text())
    label = meta.get('label', meta.get('rule', 'Cellular Automaton'))
    desc = meta.get('description', '')
    rule = meta.get('rule', '')
    size = meta.get('size', 0)
    seed = meta.get('seed', 0)
    params = meta.get('params', {})
    score = meta.get('discovery_score')
    resolution = meta.get('resolution', [0, 0])
    shorts = _is_shorts(resolution)

    # ── Title ─────────────────────────────────────────────────────────
    if shorts:
        title = f'{label} — 3D Cellular Automata #Shorts'
    else:
        title = f'{label} — 3D Cellular Automata Simulation'
    title = title[:TITLE_MAX]

    # ── Description ───────────────────────────────────────────────────
    lines = []
    if desc:
        lines.append(desc)
        lines.append('')
    lines.append(f'Rule: {rule}')
    lines.append(f'Grid: {size}³  •  Seed: {seed}')
    if score is not None:
        lines.append(f'Discovery score: {score:.2f}')
    if params:
        lines.append('')
        lines.append('Parameters:')
        for k, v in params.items():
            lines.append(f'  {k} = {v:.4g}' if isinstance(v, float)
                         else f'  {k} = {v}')
    lines.append('')
    lines.append(f'Source code: {REPO_URL}')
    lines.append('')
    if shorts:
        lines.append('#Shorts #CellularAutomata #GenerativeArt #Simulation')
    else:
        lines.append('#CellularAutomata #GenerativeArt #Simulation '
                     '#EmergentBehavior #GPU')
    description = '\n'.join(lines)[:DESCRIPTION_MAX]

    # ── Tags ──────────────────────────────────────────────────────────
    raw_tags = [
        'cellular automata', '3d cellular automata', 'generative art',
        'simulation', 'emergent behavior', 'gpu compute',
        'volumetric rendering', label.lower(),
        rule.replace('_', ' ') if rule else '',
    ]
    # Dedup case-insensitively, preserve order, drop empties.
    seen: set[str] = set()
    tags: list[str] = []
    for t in raw_tags:
        key = t.lower()
        if not t or key in seen:
            continue
        seen.add(key)
        tags.append(t)
    # Trim to total character budget.
    tags = _trim_tags(tags)

    result = {
        'title': title,
        'description': description,
        'tags': tags,
        'shorts': shorts,
        'category_id': '28',   # Science & Technology
    }

    # Optional per-recording overrides.
    overrides_path = sidecar_path.with_name(
        sidecar_path.stem + '_overrides.json')
    if overrides_path.exists():
        result.update(json.loads(overrides_path.read_text()))

    return result


def _trim_tags(tags: list[str]) -> list[str]:
    """Trim tag list so combined char count stays under YouTube's limit."""
    out: list[str] = []
    total = 0
    for t in tags:
        # YouTube counts the tag length plus 2 for the surrounding quotes.
        cost = len(t) + 2
        if total + cost > TAG_MAX_TOTAL:
            break
        out.append(t)
        total += cost
    return out
