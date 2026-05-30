"""Probe #26 — colormap GLSL <-> Python parity.

The renderer exposes 12 colormaps (Fire, Cool, Grayscale, Neon, Discrete,
Spectral, Diverging, Viridis, Magma, Plasma, Turbo, Twilight) selected by the
``u_colormap`` integer uniform.  The *same* dispatch is duplicated in FOUR
places: three GLSL ``vec3 apply_colormap(float t)`` sites
(``COMPUTE_RAYMARCH_SHADER``, ``FRAGMENT_SHADER``,
``SDF_VIEWPORT_FRAGMENT_SHADER``) plus the Python mirror
``Simulator._colormap_eval`` that paints the UI legend swatches.

Nothing guarded that this hand-maintained quartet stays in sync.  Earlier this
session a magma-polynomial divergence shipped a magenta midpoint because the
Python mirror and the GLSL stops drifted apart.  This probe closes the gap.

What it does
------------
* Extracts the LIVE colormap source (every ``vec3 colormap_*`` function plus
  ``vec3 apply_colormap``) straight out of each shader *string* in
  ``simulator`` — so it always tests the real source, never a copy.
* Wraps the extracted functions in a minimal ``#version 430`` compute shader
  and evaluates ``apply_colormap(t)`` on the GPU for every colormap ID across a
  dense ``t`` grid.
* Compares each GLSL site against (a) the Python mirror and (b) the other GLSL
  sites, grading by the worst per-channel absolute error.

Known intentional divergence
-----------------------------
Colormap ID 5 (Spectral) is a *documented approximation* on the Python side
(a cheap hue sweep) versus the full Gaussian CIE colour-matching fit in GLSL.
Its Python-vs-GLSL error is therefore reported as informational (``approx``)
and never fails the probe.  The GLSL sites must still agree with each other on
Spectral, and that cross-site check is enforced.

Exit code 0 when every enforced comparison is within tolerance.

Usage::

    python -m ca_debug.colormap_parity
    python -m ca_debug.colormap_parity --samples 256 --json /tmp/cmap.json
    python -m ca_debug.colormap_parity --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

# Allow direct execution (python ca_debug/colormap_parity.py) as well as
# module execution (python -m ca_debug.colormap_parity) from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Severity buckets.  Order matters: lower index = more severe.  Matches the
# convention used by determinism.py / golden_snapshots.py.
_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'approx': 3, 'ok': 4}

# The three module-level shader strings that define a `vec3 apply_colormap`.
_SITE_ATTRS = (
    'COMPUTE_RAYMARCH_SHADER',
    'FRAGMENT_SHADER',
    'SDF_VIEWPORT_FRAGMENT_SHADER',
)

# Colormap IDs whose Python mirror is an intentional approximation of the
# GLSL implementation (see module docstring).  Reported, never enforced.
_APPROX_PY = frozenset({5})

# Compute-shader scaffold wrapped around the extracted colormap functions.
_HEADER = """#version 430
layout(local_size_x=64) in;
uniform int u_colormap;
uniform int u_n;
layout(std430, binding=0) buffer Out { vec4 data[]; };
"""

_FOOTER = """
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_n)) return;
    float t = (u_n > 1) ? float(i) / float(u_n - 1) : 0.0;
    data[i] = vec4(apply_colormap(t), 1.0);
}
"""


def _extract_colormap_funcs(src: str) -> str:
    """Pull every `vec3 colormap_*` / `vec3 apply_colormap` body from `src`.

    Returns the concatenated function source in declaration order.  Only the
    colormap helpers and the dispatcher are taken (by name), so unrelated
    shader helpers that reference other uniforms are never dragged in.
    """
    funcs: list[str] = []
    seen: set[str] = set()
    pat = re.compile(r'vec3\s+(colormap_\w+|apply_colormap)\s*\(\s*float\s+\w+\s*\)')
    for m in pat.finditer(src):
        name = m.group(1)
        if name in seen:
            continue
        brace = src.find('{', m.end())
        if brace < 0:
            continue
        depth = 0
        i = brace
        end = -1
        while i < len(src):
            ch = src[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            i += 1
        if end < 0:
            continue
        funcs.append(src[m.start():end])
        seen.add(name)
    if 'apply_colormap' not in seen:
        raise ValueError("no `vec3 apply_colormap(float t)` found in shader source")
    if not any(n.startswith('colormap_') for n in seen):
        raise ValueError("no `vec3 colormap_*` functions found in shader source")
    return '\n'.join(funcs)


def _glsl_curves(ctx, block: str, n: int, n_cmaps: int) -> dict[int, np.ndarray]:
    """Evaluate apply_colormap(t) on the GPU for every colormap ID.

    Returns {cmap_id: ndarray(n, 3) float64} of RGB values.
    """
    shell = _HEADER + block + _FOOTER
    prog = ctx.compute_shader(shell)
    buf = ctx.buffer(reserve=n * 16)  # vec4 == 16 bytes
    buf.bind_to_storage_buffer(0)
    prog['u_n'] = n
    groups = (n + 63) // 64
    out: dict[int, np.ndarray] = {}
    for cm in range(n_cmaps):
        prog['u_colormap'] = cm
        prog.run(group_x=groups)
        ctx.finish()
        raw = np.frombuffer(buf.read(), dtype=np.float32).reshape(n, 4)
        out[cm] = raw[:, :3].astype(np.float64).copy()
    buf.release()
    prog.release()
    return out


def _python_curves(n: int, n_cmaps: int, ts: np.ndarray) -> dict[int, np.ndarray]:
    """Evaluate Simulator._colormap_eval(t) for every colormap ID off-GPU."""
    import simulator

    class _Stub:
        __slots__ = ('colormap',)

    stub = _Stub()
    out: dict[int, np.ndarray] = {}
    for cm in range(n_cmaps):
        stub.colormap = cm
        rows = [simulator.Simulator._colormap_eval(stub, float(t)) for t in ts]
        out[cm] = np.asarray(rows, dtype=np.float64)
    return out


def _grade(max_diff: float, tol: float, crit_tol: float) -> str:
    if max_diff <= tol:
        return 'ok'
    if max_diff <= crit_tol:
        return 'high'
    return 'crit'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', type=int, default=128,
                    help='Number of t samples in [0,1] (default: 128).')
    ap.add_argument('--tol', type=float, default=2e-3,
                    help='Max abs RGB error treated as OK (default: 2e-3).')
    ap.add_argument('--crit-tol', type=float, default=1e-2,
                    help='Abs RGB error above which a finding is crit (default: 1e-2).')
    ap.add_argument('--verbose', action='store_true',
                    help='Print a row for every colormap, not just failures.')
    ap.add_argument('--json', help='Write the full per-colormap report to this path.')
    args = ap.parse_args(argv)

    import simulator
    from test_harness import create_headless_context, destroy_context

    names = list(getattr(simulator, 'COLORMAP_NAMES', []))
    n_cmaps = len(names) if names else 12
    if not names:
        names = [f'cmap{i}' for i in range(n_cmaps)]

    ts = np.linspace(0.0, 1.0, args.samples)

    # ── Extract live GLSL colormap source from every dispatcher site ──────
    site_blocks: dict[str, str] = {}
    extract_errors: list[tuple[str, str]] = []
    for attr in _SITE_ATTRS:
        src = getattr(simulator, attr, None)
        if src is None:
            extract_errors.append((attr, 'module attribute missing'))
            continue
        try:
            site_blocks[attr] = _extract_colormap_funcs(src)
        except ValueError as exc:
            extract_errors.append((attr, str(exc)))

    if not site_blocks:
        print("colormap_parity: FAILED to extract colormap source from any site")
        for attr, err in extract_errors:
            print(f"  {attr}: {err}")
        return 1

    # ── Python reference curves (no GPU needed) ───────────────────────────
    py = _python_curves(args.samples, n_cmaps, ts)

    # ── GPU evaluation of each GLSL site ──────────────────────────────────
    window, ctx = create_headless_context()
    site_curves: dict[str, dict[int, np.ndarray]] = {}
    compile_errors: list[tuple[str, str]] = []
    try:
        for attr, block in site_blocks.items():
            try:
                site_curves[attr] = _glsl_curves(ctx, block, args.samples, n_cmaps)
            except Exception as exc:  # noqa: BLE001 — surface shader compile errors
                compile_errors.append((attr, str(exc).strip().splitlines()[0] if str(exc) else repr(exc)))
    finally:
        destroy_context(window)

    if not site_curves:
        print("colormap_parity: every GLSL site failed to compile")
        for attr, err in compile_errors:
            print(f"  {attr}: {err}")
        return 1

    # ── Compare ───────────────────────────────────────────────────────────
    report: list[dict] = []
    worst_sev = 'ok'
    primary = _SITE_ATTRS[0] if _SITE_ATTRS[0] in site_curves else next(iter(site_curves))

    for cm in range(n_cmaps):
        # Python-vs-GLSL: measured against every site, worst case.
        py_diff = 0.0
        for attr, curves in site_curves.items():
            d = float(np.max(np.abs(curves[cm] - py[cm])))
            py_diff = max(py_diff, d)
        # Cross-site: GLSL sites against each other.
        attrs = list(site_curves.keys())
        cross_diff = 0.0
        for a in range(len(attrs)):
            for b in range(a + 1, len(attrs)):
                d = float(np.max(np.abs(site_curves[attrs[a]][cm] - site_curves[attrs[b]][cm])))
                cross_diff = max(cross_diff, d)

        is_approx = cm in _APPROX_PY
        py_sev = 'approx' if is_approx else _grade(py_diff, args.tol, args.crit_tol)
        cross_sev = _grade(cross_diff, args.tol, args.crit_tol)
        # Enforced severity ignores the documented-approx Python divergence.
        enforced = cross_sev if is_approx else (
            py_sev if _SEV_ORDER[py_sev] < _SEV_ORDER[cross_sev] else cross_sev
        )
        if _SEV_ORDER[enforced] < _SEV_ORDER[worst_sev]:
            worst_sev = enforced

        report.append({
            'id': cm,
            'name': names[cm] if cm < len(names) else f'cmap{cm}',
            'py_diff': py_diff,
            'py_sev': py_sev,
            'cross_diff': cross_diff,
            'cross_sev': cross_sev,
            'enforced': enforced,
        })

    # ── Print ─────────────────────────────────────────────────────────────
    print(f"colormap parity — {len(site_curves)}/{len(_SITE_ATTRS)} GLSL sites, "
          f"{n_cmaps} colormaps, {args.samples} samples, tol={args.tol:g}")
    print(f"  sites: {', '.join(site_curves.keys())}")
    if extract_errors:
        for attr, err in extract_errors:
            print(f"  WARN extract {attr}: {err}")
    if compile_errors:
        for attr, err in compile_errors:
            print(f"  WARN compile {attr}: {err}")

    header = f"  {'id':>2} {'name':<12} {'py_diff':>10} {'cross_diff':>11}  verdict"
    printed_header = False
    for row in report:
        show = args.verbose or row['enforced'] not in ('ok', 'approx') or row['py_sev'] == 'approx'
        if not show:
            continue
        if not printed_header:
            print(header)
            printed_header = True
        tag = row['enforced'].upper()
        if row['py_sev'] == 'approx':
            tag += ' (py=approx)'
        print(f"  {row['id']:>2} {row['name']:<12} {row['py_diff']:>10.2e} "
              f"{row['cross_diff']:>11.2e}  {tag}")

    counts = {k: 0 for k in _SEV_ORDER}
    for row in report:
        counts[row['enforced']] += 1
    summary = ' '.join(f"{k}={counts[k]}" for k in ('ok', 'high', 'crit', 'err') if counts[k])
    approx_n = sum(1 for r in report if r['py_sev'] == 'approx')
    extra = f" approx_py={approx_n}" if approx_n else ''
    print(f"\nworst={worst_sev}  {summary}{extra}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({
                'samples': args.samples,
                'tol': args.tol,
                'crit_tol': args.crit_tol,
                'sites': list(site_curves.keys()),
                'extract_errors': [[a, e] for a, e in extract_errors],
                'compile_errors': [[a, e] for a, e in compile_errors],
                'worst': worst_sev,
                'colormaps': report,
            }, f, indent=2)
        print(f"Wrote {args.json}")

    # Fail on any enforced high/crit/err, or if a declared site never compiled.
    bad = counts['high'] + counts['crit'] + counts['err']
    if bad or compile_errors or extract_errors:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
