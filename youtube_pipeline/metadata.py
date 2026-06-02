"""Build YouTube title / description / tags from a recording sidecar JSON."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_URL = 'https://github.com/c0re-i5/3d-cellular-automata'

# YouTube hard limits.
TITLE_MAX = 100
DESCRIPTION_MAX = 5000
TAG_MAX_TOTAL = 500   # combined character count across all tags

# Standing copy used in every long-form description and the channel
# About text.  Keep this in sync with README.md so the project pitch is
# consistent across surfaces.
PROJECT_BLURB = (
    "A real-time GPU simulator for 3D cellular automata: 97 hand-tuned "
    "presets spanning Game of Life, Lenia, reaction-diffusion "
    "(Gray-Scott, Belousov-Zhabotinsky), Lattice Boltzmann fluids, "
    "crystal growth, peridynamic fracture, electromagnetism, "
    "Schrödinger-equation quantum mechanics, slime mold, predator-prey "
    "lattices, and active matter. Every preset was discovered by an "
    "automated parameter search and rendered with volumetric "
    "ray-marching at up to 4K."
)

# Map rule-name prefixes / explicit rule names to a category tag used in
# titles and descriptions.  Order matters: the first matching prefix or
# exact key wins.
_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ('flagship_', 'Flagship'),
    ('quantum_', 'Quantum Mechanics'),
    ('crystal_', 'Crystal Growth'),
    ('lenia', 'Continuous CA'),
    ('bz_', 'Reaction-Diffusion'),
    ('gray_scott', 'Reaction-Diffusion'),
    ('reaction_diffusion', 'Reaction-Diffusion'),
    ('schnakenberg', 'Reaction-Diffusion'),
    ('brusselator', 'Reaction-Diffusion'),
    ('fitzhugh', 'Reaction-Diffusion'),
    ('morphogen', 'Reaction-Diffusion'),
    ('greenberg_hastings', 'Excitable Media'),
    ('cyclic_ca', 'Excitable Media'),
    ('hodgepodge', 'Excitable Media'),
    ('flocking', 'Active Matter'),
    ('active_nematic', 'Active Matter'),
    ('xy_spin', 'Spin System'),
    ('ising', 'Spin System'),
    ('kuramoto', 'Coupled Oscillators'),
    ('sine_gordon', 'Soliton Dynamics'),
    ('wave', 'Wave Equation'),
    ('em_wave', 'Electromagnetism'),
    ('dirac', 'Quantum Mechanics'),
    ('hopfion', 'Topological Field'),
    ('stable_fluids', 'Fluid Dynamics'),
    ('smoke_wind', 'Fluid Dynamics'),
    ('rayleigh_benard', 'Fluid Dynamics'),
    ('compressible_euler', 'Fluid Dynamics'),
    ('viscous_fingers', 'Fluid Dynamics'),
    ('volcanic', 'Fluid Dynamics'),
    ('predator_prey', 'Population Dynamics'),
    ('prisoners_dilemma', 'Game Theory'),
    ('physarum', 'Biological Network'),
    ('mycelium', 'Biological Network'),
    ('lichen', 'Biological Network'),
    ('eden', 'Growth Process'),
    ('forest_fire', 'Excitable Media'),
    ('fire', 'Combustion'),
    ('erosion', 'Geomorphology'),
    ('fracture', 'Solid Mechanics'),
    ('phase_separation', 'Phase Separation'),
    ('nucleation', 'Phase Separation'),
    ('sandpile', 'Self-Organised Criticality'),
    ('langton_ant', 'Turing Machine'),
    ('wireworld', 'Discrete Logic'),
    ('margolus', 'Block CA'),
    ('larger_than_life', 'Life-Like CA'),
    ('445_rule', 'Life-Like CA'),
    ('445', 'Life-Like CA'),
    ('game_of_life', 'Life-Like CA'),
    ('smoothlife', 'Life-Like CA'),
    ('life', 'Life-Like CA'),
    ('smallworld_ca', 'Network CA'),
    ('genome_ca', 'Evolutionary CA'),
    ('causal_ca', 'Causal CA'),
    ('element_ca', 'Particle Chemistry'),
    ('element_metals', 'Particle Chemistry'),
    ('element_na_water', 'Particle Chemistry'),
    ('galaxy', 'Astrophysics'),
    ('mandelbulb', 'Fractal'),
    ('mandelbox', 'Fractal'),
    ('juliabulb', 'Fractal'),
    ('menger', 'Fractal'),
    ('predator_prey_lattice', 'Population Dynamics'),
    ('smugglers', 'Agent-Based'),
    ('wandering_voxels', 'Agent-Based'),
    ('noop', 'Diagnostic'),
)


def _normalize_rule(rule: str) -> str:
    """Strip a lattice-variant prefix so FCC ports share the cubic rule's
    category and hook (e.g. ``fcc_forest_fire`` -> ``forest_fire``)."""
    if rule.startswith('fcc_'):
        return rule[len('fcc_'):]
    return rule


def _category_for(rule: str) -> str:
    """Return a short human-readable category label for ``rule``."""
    rule = _normalize_rule(rule)
    for prefix, cat in _CATEGORY_RULES:
        if rule.startswith(prefix):
            return cat
    return 'Cellular Automaton'


# Human "hooks" for Shorts titles.  These translate the technical preset
# name into a concrete, swipe-stopping phrase: an active verb + a
# physical metaphor a non-specialist can picture in half a second.  The
# searchable proper noun still lives in the science category that
# follows the em-dash (e.g. "3D Spin System"), so we keep algorithm
# reach while giving a human a reason to stop.
#
# Order matters: the first matching prefix wins (same as _CATEGORY_RULES).
# Rules whose label is ALREADY human and high-search (Game of Life,
# SmoothLife, Crystal, 4/4/5) are intentionally absent so their proven
# label passes through unchanged.
_HOOK_RULES: tuple[tuple[str, str], ...] = (
    ('bz_', 'Chemical Spirals That Never Stop'),
    ('hodgepodge', 'A Chemical Clock Comes Alive'),
    ('cyclic_ca', 'Colors Chasing Each Other Forever'),
    ('greenberg_hastings', 'Self-Healing Waves Race Around'),
    ('gray_scott', 'Patterns That Copy Themselves'),
    ('reaction_diffusion', 'Patterns That Copy Themselves'),
    ('schnakenberg', 'Spots Growing Into Stripes'),
    ('brusselator', 'A Chemical Clock Ticking in 3D'),
    ('morphogen', 'How an Animal Gets Its Spots'),
    ('fitzhugh', 'Heartbeat Waves Pulsing in 3D'),
    ('xy_spin', 'A Magnet Making Up Its Mind'),
    ('ising', 'A Magnet Making Up Its Mind'),
    ('kuramoto', 'Thousands of Clocks Snapping Into Sync'),
    ('flocking', 'A Swarm That Thinks as One'),
    ('active_nematic', 'A Fluid That Stirs Itself'),
    ('lenia', 'Digital Lifeforms Evolving'),
    ('eden', 'A Crystal That Builds Itself'),
    ('forest_fire', 'Fires That Never Burn Out'),
    ('sandpile', 'The Pile That Collapses Itself'),
    ('predator_prey', 'Hunters and Prey Locked in Balance'),
    ('prisoners_dilemma', 'Cooperation vs Betrayal, Spreading'),
    ('physarum', 'Slime Mold Finding the Shortest Path'),
    ('mycelium', 'A Fungal Network Creeping Outward'),
    ('lichen', 'Lichen Crawling Across Stone'),
    ('sine_gordon', 'Solitons Passing Through Each Other'),
    ('em_wave', 'Light Waves You Can Actually See'),
    ('dirac', 'Antimatter Rippling Through Space'),
    ('quantum_', 'A Quantum Particle Spreading Out'),
    ('wave', 'Ripples Racing Through a 3D Lattice'),
    ('hopfion', 'A Knot of Energy That Holds Its Shape'),
    ('phase_separation', 'Oil and Water Pulling Apart'),
    ('nucleation', 'Droplets Condensing Out of Nothing'),
    ('fracture', 'Watch a Solid Shatter From Within'),
    ('erosion', 'A Landscape Carving Itself'),
    ('wireworld', 'Electricity Computing on a Grid'),
    ('langton_ant', 'One Ant Builds a Highway'),
)


def _hook_for(rule: str) -> str:
    """Return a human Shorts hook for ``rule``, or '' if none is defined."""
    rule = _normalize_rule(rule)
    for prefix, hook in _HOOK_RULES:
        if rule.startswith(prefix):
            return hook
    return ''


def _humanize_label(label: str) -> str:
    """Clean a preset label for use in a title's hook segment.

    The channel analytics showed parenthetical qualifiers (e.g.
    "Crystal (Faceted)") correlate with lower views, so fold a trailing
    "(Variant)" into a natural adjective prefix: "Crystal (Faceted)" ->
    "Faceted Crystal", "Gray-Scott (coral)" -> "Coral Gray-Scott".
    """
    if not label:
        return label
    m = re.match(r'^(.*?)\s*\(([^)]+)\)\s*$', label)
    if m:
        base, variant = m.group(1).strip(), m.group(2).strip()
        if base and variant:
            # Capitalise the variant's first letter without lowercasing
            # acronyms/Miller indices already inside it.
            variant = variant[:1].upper() + variant[1:]
            return f'{variant} {base}'
    return label


def _format_duration(seconds: float) -> str:
    """Pretty duration: '31s', '1m 24s', '2m 03s'."""
    if not seconds or seconds < 0:
        return ''
    s = int(round(seconds))
    if s < 60:
        return f'{s}s'
    return f'{s // 60}m {s % 60:02d}s'


def _hook_from_description(desc: str) -> str:
    """Pick a short subject phrase from a rule's full description.

    Strategy: take the first em-dash clause if present and short enough
    (≤ 50 chars), otherwise the first sentence head.  Returns '' if no
    sensible hook can be extracted.
    """
    if not desc:
        return ''
    desc = desc.strip().rstrip('.')
    if ' — ' in desc:
        head = desc.split(' — ', 1)[0].strip()
        if 5 <= len(head) <= 50:
            return head
    # Fallback: first sentence, truncated.
    sent = desc.split('. ', 1)[0].strip()
    if len(sent) <= 50:
        return sent
    return ''


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
    fps = meta.get('fps', 0)
    frames = meta.get('frames', 0)
    duration_sec = meta.get('duration_sec', 0)
    dt = meta.get('dt')
    renderer_mode = meta.get('renderer_mode', '')
    shorts = _is_shorts(resolution)
    category = _category_for(rule)
    hook = _hook_from_description(desc)

    # ── Title ─────────────────────────────────────────────────────────
    # Strip the in-app "Flagship: " curation prefix — viewers searching
    # YouTube don't care about our internal preset tier, they care what's
    # actually on screen ("Coral Reef", not "Flagship: Coral Reef").
    title_label = label
    if title_label.startswith('Flagship: '):
        title_label = title_label[len('Flagship: '):]
    if shorts:
        # Shorts: lead with a HUMAN hook (active verb + metaphor) so a
        # mid-scroll viewer has a reason to stop, then anchor with the
        # science category ("3D Spin System") which carries the
        # searchable keyword for the algorithm.  Jargon-heavy presets get
        # a hand-written hook; presets whose label is already human and
        # high-search (Game of Life, SmoothLife, 4/4/5, Crystal) fall
        # back to their cleaned label.
        hook_seg = _hook_for(rule) or _humanize_label(title_label)
        title = f'{hook_seg} — 3D {category} #Shorts'
        if len(title) > TITLE_MAX:
            title = f'{hook_seg} #Shorts'
        if len(title) > TITLE_MAX:
            title = f'{_humanize_label(title_label)} #Shorts'
    else:
        # Long-form: lead with the label, then a hook phrase from the
        # rule description, then the project anchor.  Falls back to a
        # bare project anchor if hook + label is already too long.
        anchor = '3D Cellular Automata'
        if hook:
            candidate = f'{title_label}: {hook} | {anchor}'
            if len(candidate) <= TITLE_MAX:
                title = candidate
            else:
                title = f'{title_label} | {anchor}'
        else:
            title = f'{title_label} | {anchor}'
    title = title[:TITLE_MAX]

    # ── Description ───────────────────────────────────────────────────
    # Shorts: single-sentence hook (the watch-screen UI truncates after
    # ~100 chars to a "more" link, so a wall of text just looks bad).
    # Long-form: rich description with viewer-facing context, technical
    # details, project blurb, repo link, hashtags.
    if shorts:
        lines = [body for body in (short_desc,) if body]
        lines.append('')
        lines.append(f'Source: {REPO_URL}')
        lines.append('')
        lines.append('#Shorts #CellularAutomata #GenerativeArt '
                     '#Simulation #3D')
    else:
        lines: list[str] = []
        if desc:
            lines.append(desc)
            lines.append('')
        lines.append('▸ What you\'re seeing')
        what = (
            f'The "{title_label}" preset of a 3D cellular automaton, '
            f'simulated on a {size}³ voxel grid'
        )
        if duration_sec:
            what += f' for {_format_duration(duration_sec)}'
        if frames and fps:
            what += f' ({frames:,} frames at {fps} fps)'
        what += '. Every voxel is updated in parallel each step on the '
        what += 'GPU using OpenGL 4.3 compute shaders, then rendered '
        what += 'with volumetric ray-marching.'
        lines.append(what)
        lines.append('')
        lines.append('▸ Parameters')
        if params:
            for k, v in params.items():
                lines.append(f'    {k} = {v:.4g}' if isinstance(v, float)
                             else f'    {k} = {v}')
        else:
            lines.append('    (no tunable parameters — pure rule lookup)')
        lines.append('')
        lines.append('▸ Run details')
        lines.append(f'    Rule shader     : {rule}')
        lines.append(f'    Category        : {category}')
        if renderer_mode:
            lines.append(f'    Renderer        : {renderer_mode}')
        if resolution and len(resolution) == 2 and all(resolution):
            lines.append(f'    Resolution      : '
                         f'{resolution[0]}×{resolution[1]} @ {fps or "?"}fps')
        lines.append(f'    RNG seed        : {seed}')
        if dt is not None:
            lines.append(f'    Time step (dt)  : {dt}')
        if score is not None:
            lines.append(f'    Discovery score : {score:.3f}')
        lines.append('')
        lines.append('▸ About this project')
        lines.append(PROJECT_BLURB)
        lines.append('')
        lines.append(f'▸ Source code')
        lines.append(f'    {REPO_URL}')
        lines.append('')
        lines.append('#CellularAutomata #GenerativeArt #Simulation '
                     '#EmergentBehavior #ComputeShader #GPU '
                     '#ComplexSystems #ProceduralGeneration')
    description = '\n'.join(lines)[:DESCRIPTION_MAX]

    # ── Tags ──────────────────────────────────────────────────────────
    raw_tags = [
        'cellular automata', '3d cellular automata', 'generative art',
        'simulation', 'emergent behavior', 'gpu compute',
        'volumetric rendering', 'compute shader', 'complex systems',
        'procedural generation', 'opengl',
        category.lower(),
        title_label.lower(),
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


# ── Channel description ──────────────────────────────────────────────
# YouTube channel "About" description (max 1000 chars).  Used by
# ``python -m youtube_pipeline --print-channel-description``; not
# uploaded automatically — paste into YouTube Studio yourself so you
# stay in control of the channel page.
CHANNEL_DESCRIPTION_MAX = 1000

CHANNEL_DESCRIPTION = f"""3D cellular automata, simulated in real time on the GPU.

Every video is a single run of an open-source volumetric simulator
with 97 hand-tuned presets, including:

• Life-like      — Game of Life, SmoothLife, Larger-than-Life
• Continuous     — Lenia (single & multi-channel)
• Reaction-diff  — Gray-Scott, Belousov-Zhabotinsky, Schnakenberg
• Crystal growth — dendritic, faceted, snowflake, DLA
• Fluids         — Navier-Stokes, Rayleigh-Bénard, Lattice Boltzmann
• Quantum        — Schrödinger orbitals, tunneling, Dirac equation
• Active matter  — Vicsek flocking, active nematics, predator-prey
• Bio networks   — slime mold (Physarum), mycelium, lichen
• Excitable      — BZ scroll waves, Greenberg-Hastings, FitzHugh-N.
• Plus peridynamic fracture, Cahn-Hilliard, Ising spins, Kuramoto
  oscillators, sandpile criticality, and Element chemistry.

Every preset was discovered by an automated parameter search.

Source: {REPO_URL}
"""


def channel_description() -> str:
    """Return the channel About text. Errors if it exceeds YouTube's cap."""
    text = CHANNEL_DESCRIPTION.strip()
    if len(text) > CHANNEL_DESCRIPTION_MAX:
        raise ValueError(
            f'CHANNEL_DESCRIPTION is {len(text)} chars, exceeds '
            f'YouTube limit of {CHANNEL_DESCRIPTION_MAX}. Trim it in '
            f'metadata.py rather than letting the print get cut off.')
    return text
