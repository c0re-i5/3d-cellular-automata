"""Build a Reddit submission (title + markdown body) from a recording
sidecar JSON and its YouTube URL.

Reuses the rule-categorisation and hook-extraction helpers from
``youtube_pipeline.metadata`` so titles/categories stay consistent
across the two surfaces.
"""
from __future__ import annotations

import json
from pathlib import Path

from youtube_pipeline.metadata import (
    PROJECT_BLURB,
    REPO_URL,
    _category_for,
    _format_duration,
    _hook_from_description,
)

YOUTUBE_CHANNEL_URL = 'https://www.youtube.com/@3dCellularAutomata'

# Reddit hard limits.
TITLE_MAX = 300        # subreddit-side cap is also 300 by default
BODY_MAX = 40000       # Reddit selftext cap

# Default flair templates (set at submit time if subreddit has flair
# configured). Mapped from category → flair text. Falls back to None.
FLAIR_FOR_CATEGORY: dict[str, str] = {
    'Flagship':                'Flagship',
    'Reaction-Diffusion':      'Reaction-Diffusion',
    'Continuous CA':           'Continuous CA',
    'Life-Like CA':            'Life-Family',
    'Crystal Growth':          'Crystal Growth',
    'Excitable Media':         'Excitable Media',
    'Active Matter':           'Active Matter',
    'Spin System':             'Spin System',
    'Coupled Oscillators':     'Coupled Oscillators',
    'Wave Equation':           'Waves',
    'Electromagnetism':        'Electromagnetism',
    'Quantum Mechanics':       'Quantum',
    'Fluid Dynamics':          'Fluid Dynamics',
    'Population Dynamics':     'Population Dynamics',
    'Biological Network':      'Biological',
    'Phase Separation':        'Phase Separation',
    'Combustion':              'Combustion',
    'Soliton Dynamics':        'Solitons',
    'Topological Field':       'Topological',
    'Self-Organised Criticality': 'Self-Organised Criticality',
    'Discrete Logic':          'Discrete Logic',
    'Block CA':                'Discrete CA',
    'Turing Machine':          'Discrete CA',
    'Growth Process':          'Growth',
    'Game Theory':             'Game Theory',
    'Geomorphology':           'Geomorphology',
    'Solid Mechanics':         'Solid Mechanics',
}


def _strip_flagship_prefix(label: str) -> str:
    """Drop the in-app "Flagship: " curation prefix for outward-facing copy."""
    if label.startswith('Flagship: '):
        return label[len('Flagship: '):]
    return label


def build_submission(sidecar_path: Path, youtube_url: str) -> dict:
    """Return a dict with ``title``, ``body``, ``flair``, ``shorts``.

    ``youtube_url`` is the URL of the already-uploaded video (link
    submission target). The body is markdown with the rule's
    description, reproduction parameters, project blurb, and links to
    the source repo and YouTube channel.

    Honours an optional ``<basename>_reddit_overrides.json`` next to
    the sidecar so any field can be hand-tuned.
    """
    meta = json.loads(sidecar_path.read_text())
    label = _strip_flagship_prefix(
        meta.get('label', meta.get('rule', 'Cellular Automaton')))
    desc = meta.get('description', '').strip()
    rule = meta.get('rule', '')
    size = meta.get('size', 0)
    seed = meta.get('seed', 0)
    params = meta.get('params', {}) or {}
    init_variant = meta.get('init_variant')
    score = meta.get('discovery_score')
    resolution = meta.get('resolution', [0, 0]) or [0, 0]
    fps = meta.get('fps', 0)
    frames = meta.get('frames', 0)
    duration_sec = meta.get('duration_sec', 0)
    dt = meta.get('dt')
    renderer_mode = meta.get('renderer_mode', '')
    shorts = bool(resolution and len(resolution) == 2
                  and resolution[1] > resolution[0])
    category = _category_for(rule)
    hook = _hook_from_description(desc)

    # ── Title ─────────────────────────────────────────────────────────
    # Audience here already knows it's a CA simulation, so skip the
    # "3D Cellular Automata" anchor that the YouTube title needs.
    # Lead with the visual subject (label), then a hook clause if we
    # have one and it adds info beyond the label, then the category as
    # a context tag in [brackets]. Suppress the hook when it's just
    # the label re-stated (e.g. "Crystal (Dendritic): Dendritic crystal")
    # which happens for descriptions that lead with the rule's own name.
    label_words = {w.lower().strip('()') for w in label.split()}
    hook_adds_info = bool(hook) and not all(
        w.lower() in label_words for w in hook.split() if len(w) > 2)
    if hook and hook_adds_info:
        candidate = f'{label}: {hook}  [{category}]'
        if len(candidate) <= TITLE_MAX:
            title = candidate
        else:
            title = f'{label}  [{category}]'
    else:
        title = f'{label}  [{category}]'
    title = title[:TITLE_MAX]

    # ── Body ──────────────────────────────────────────────────────────
    parts: list[str] = []

    # Lead paragraph: rule description if present, else a generic
    # one-liner. Keep it short — readers came for the video.
    if desc:
        parts.append(desc)
    else:
        parts.append(
            f'A 3D cellular automaton from the `{rule}` rule, '
            f'simulated on a {size}³ voxel grid.')

    # Watch link — this is the main payload of the post even though
    # the submission is also a link post. Visible inline matters.
    parts.append('')
    parts.append(f'🎬 **[Watch the full recording]({youtube_url})**')

    # Reproduction block. The whole point of this community is that
    # discoveries are reproducible, so this gets prominent placement.
    parts.append('')
    parts.append('---')
    parts.append('')
    parts.append('### Reproduction')
    parts.append('')
    parts.append(f'- **Rule**: `{rule}`')
    parts.append(f'- **Grid**: {size}³ voxels')
    parts.append(f'- **Seed**: `{seed}`')
    if init_variant:
        parts.append(f'- **Init**: `{init_variant}`')
    if dt is not None:
        parts.append(f'- **dt**: `{dt}`')
    if params:
        parts.append('- **Parameters**:')
        for k, v in params.items():
            val = f'{v:.4g}' if isinstance(v, float) else str(v)
            parts.append(f'    - `{k}` = `{val}`')
    if score is not None:
        parts.append(f'- **Discovery score**: `{score:.3f}`')

    # Capture details — useful for "what hardware do I need" / "how
    # long is this". Skipped on Shorts (already short).
    if not shorts and (duration_sec or frames or fps or renderer_mode):
        parts.append('')
        parts.append('### Recording')
        if duration_sec:
            parts.append(
                f'- **Duration**: {_format_duration(duration_sec)}')
        if frames and fps:
            parts.append(f'- **Frames**: {frames:,} @ {fps} fps')
        if resolution and len(resolution) == 2 and all(resolution):
            parts.append(
                f'- **Resolution**: {resolution[0]}×{resolution[1]}')
        if renderer_mode:
            parts.append(f'- **Renderer**: {renderer_mode}')

    # Project tail — kept brief because every post repeats it. Two
    # links and a one-line pitch. Skipped on Shorts to keep the post
    # tight (Shorts get drive-by viewers, less context tolerance).
    parts.append('')
    parts.append('---')
    parts.append('')
    if shorts:
        parts.append(
            f'_From the [3D Cellular Automata YouTube channel]'
            f'({YOUTUBE_CHANNEL_URL}). '
            f'Open-source simulator: [GitHub]({REPO_URL})._')
    else:
        parts.append(f'### About this project')
        parts.append('')
        parts.append(PROJECT_BLURB)
        parts.append('')
        parts.append(f'- 💻 **Source**: [{REPO_URL}]({REPO_URL})')
        parts.append(f'- 📺 **YouTube**: [{YOUTUBE_CHANNEL_URL}]'
                     f'({YOUTUBE_CHANNEL_URL})')

    body = '\n'.join(parts)[:BODY_MAX]
    flair = FLAIR_FOR_CATEGORY.get(category)

    result = {
        'title': title,
        'body': body,
        'flair': flair,
        'shorts': shorts,
        'category': category,
    }

    overrides_path = sidecar_path.with_name(
        sidecar_path.stem + '_reddit_overrides.json')
    if overrides_path.exists():
        result.update(json.loads(overrides_path.read_text()))

    return result
