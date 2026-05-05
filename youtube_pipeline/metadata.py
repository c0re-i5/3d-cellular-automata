"""Build YouTube title / description / tags from a recording sidecar JSON."""
from __future__ import annotations

import json
from pathlib import Path

REPO_URL = 'https://github.com/c0re-i5/3d-cellular-automata'

# YouTube hard limits.
TITLE_MAX = 100
DESCRIPTION_MAX = 5000
TAG_MAX_TOTAL = 500   # combined character count across all tags


def _sanitize(text: str) -> str:
    """Strip characters YouTube rejects in title / description / tags.

    The Data API rejects ``<`` and ``>`` in the snippet metadata as a
    blanket anti-injection measure (HttpError 400 ``invalidDescription``
    / ``invalidTitle``).  Crystal preset descriptions use Miller-index
    notation like ``<100>-facet`` which trips this filter, so we replace
    angle brackets with their nearest typographic equivalents.
    """
    if not text:
        return text
    # U+27E8 / U+27E9 are mathematical angle brackets -- visually almost
    # identical to ASCII < > and accepted by YouTube.
    return text.replace('<', '\u27e8').replace('>', '\u27e9')


def _is_shorts(resolution: list[int]) -> bool:
    """Vertical aspect ratio (9:16-ish) → Shorts."""
    if not resolution or len(resolution) != 2:
        return False
    w, h = resolution
    return h > w


# Shorts only show the first ~100 chars of the description before
# truncating with a "more" link, so a 400-char paragraph just produces
# an unreadable wall of text on the watch screen.
SHORTS_DESC_MAX = 140


def _shorten_for_shorts(desc: str) -> str:
    """Return a Shorts-friendly blurb derived from the full description.

    Strategy: take the first sentence (split on ". " — preserves
    abbreviations like "B6/S5" and "vs."), then if it's too long try
    keeping the first em-dash clause if that gives a substantive head
    (≥ 60 chars), otherwise hard-truncate at a word boundary.
    """
    if not desc:
        return desc
    desc = desc.replace('\n', ' ').strip()
    # First sentence — split on ". " (with trailing space) so decimals
    # like "0.5" don't break it.
    sentence = desc.split('. ', 1)[0].rstrip('.').strip()
    if len(sentence) <= SHORTS_DESC_MAX:
        return sentence + '.'
    # Drop a trailing em-dash explanation if present, the head fits
    # AND the head is long enough on its own to be a useful blurb.
    if ' — ' in sentence:
        head = sentence.split(' — ', 1)[0].strip()
        if 60 <= len(head) <= SHORTS_DESC_MAX:
            return head + '.'
    # Hard-truncate at a word boundary.
    truncated = sentence[:SHORTS_DESC_MAX].rsplit(' ', 1)[0]
    return truncated + '…'


def build_metadata(sidecar_path: Path) -> dict:
    """Return a dict with ``title``, ``description``, ``tags``, ``shorts``.

    Honours an optional ``<basename>_overrides.json`` next to the sidecar
    so any field can be hand-tuned without editing this module.
    """
    meta = json.loads(sidecar_path.read_text())
    label = meta.get('label', meta.get('rule', 'Cellular Automaton'))
    desc = meta.get('description', '')
    short_desc = meta.get('short_description', '') or _shorten_for_shorts(desc)
    rule = meta.get('rule', '')
    size = meta.get('size', 0)
    seed = meta.get('seed', 0)
    params = meta.get('params', {})
    score = meta.get('discovery_score')
    resolution = meta.get('resolution', [0, 0])
    shorts = _is_shorts(resolution)

    # ── Title ─────────────────────────────────────────────────────────
    # Strip the in-app "Flagship: " curation prefix — viewers searching
    # YouTube don't care about our internal preset tier, they care what's
    # actually on screen ("Coral Reef", not "Flagship: Coral Reef").
    title_label = label
    if title_label.startswith('Flagship: '):
        title_label = title_label[len('Flagship: '):]
    if shorts:
        title = f'{title_label} — 3D Cellular Automata #Shorts'
    else:
        title = f'{title_label} — 3D Cellular Automata Simulation'
    title = title[:TITLE_MAX]

    # ── Description ───────────────────────────────────────────────────
    # Shorts use a single-sentence blurb (the watch screen truncates
    # to ~100 chars before "more"); long-form keeps the rich paragraph.
    lines = []
    body_desc = short_desc if shorts else desc
    if body_desc:
        lines.append(body_desc)
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
        'volumetric rendering', title_label.lower(),
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
        'title': _sanitize(title),
        'description': _sanitize(description),
        'tags': [_sanitize(t) for t in tags],
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
