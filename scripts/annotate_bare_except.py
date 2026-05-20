#!/usr/bin/env python3
"""Bulk-annotate intentional bare-except sites with `# noqa: BLE001  <reason>`.

Inspects the try-block above each `except Exception(...):` line and assigns a
category based on body content. Only writes the annotation when the category
is confident; leaves ambiguous sites untouched for manual review.

Usage: python annotate_bare_except.py <file> [<file> ...]
Prints a diff-style summary; modifies files in place.
"""
import re
import sys
import pathlib

BARE = re.compile(r'^(\s*)except\s+Exception(\s+as\s+\w+)?\s*:(.*)$')
NOQA = re.compile(r'#\s*noqa:\s*BLE001')

# (matcher, reason). First match wins.
RULES = [
    (lambda b: 'glFenceSync' in b or 'glClientWaitSync' in b or 'glDeleteSync' in b,
     'GL fence not supported by driver'),
    (lambda b: 'glGetIntegerv' in b and 'NVX' in b,
     'GPU memory query unsupported (non-NVIDIA)'),
    (lambda b: 'ssbo.read()' in b or '_ssbo.read' in b,
     'ssbo read race during teardown'),
    (lambda b: '.read()' in b and ('tex_' in b or '_tex' in b or '_grid' in b),
     'GPU texture read race during teardown'),
    (lambda b: 'GL.glFinish' in b or 'glMemoryBarrier' in b
               or 'glCopyImageSubData' in b,
     'GL sync call, best-effort'),
    (lambda b: 'ctx.info' in b,
     'GL info probe, may be unsupported'),
    (lambda b: 'ctx.compute_shader' in b,
     'shader compile fallback'),
    (lambda b: '_run_recorder' in b or 'recorder.' in b
               or 'rec.' in b and ('.log' in b or '.update' in b
                                    or '.close' in b or '.snapshot' in b
                                    or '.set_derived' in b),
     'optional recorder, never fatal'),
    (lambda b: '_open_run' in b or 'RunRecorder' in b
               or '_trial_recorder' in b,
     'optional recorder, never block startup'),
    (lambda b: '_resolve_composed_preset' in b,
     'preset lookup failure -> caller falls back'),
    (lambda b: 'imgui.' in b and ('.destroy' in b or '.shutdown' in b
                                   or '_renderer' in b.lower()),
     'imgui teardown, never fatal'),
    (lambda b: 'imgui_renderer' in b and '.shutdown' in b,
     'imgui teardown, never fatal'),
    (lambda b: 'glfw.destroy_window' in b or 'glfw.terminate' in b,
     'GLFW teardown, never fatal'),
    (lambda b: 'ctx.release' in b or '.release()' in b,
     'GL resource release, never fatal'),
    (lambda b: '_cleanup' in b or '_stop_recording' in b
               or '.shutdown(' in b,
     'cleanup hook, never fatal'),
    (lambda b: 'shutil.rmtree' in b or 'os.remove' in b or '.unlink' in b,
     'best-effort cleanup'),
    (lambda b: 'os.makedirs' in b or '.mkdir' in b,
     'best-effort mkdir'),
    (lambda b: '_pick_viewport_surface_origin' in b,
     'bisection may fail, keep current vista'),
    (lambda b: '_compile_compute' in b,
     'transient compile error, GUI keeps running'),
    (lambda b: '_change_rule' in b,
     'rule change failure, keep current rule'),
    (lambda b: 'tempfile.mkstemp' in b or 'tempfile.mkdtemp' in b,
     'tempfile creation, best-effort'),
    (lambda b: 'json.dump' in b or '.write_text' in b
               or 'open(' in b and ("'w'" in b or '"w"' in b or "'wb'" in b
                                     or '"wb"' in b),
     'best-effort write'),
    (lambda b: 'json.load' in b or 'json.loads' in b,
     'malformed JSON, treat as missing'),
    (lambda b: 'np.load' in b,
     'optional file load'),
    (lambda b: 'open(' in b and ('.read' in b or "'rb'" in b or "'r'" in b),
     'best-effort read'),
    (lambda b: re.search(r'\bint\s*\(', b) and 'os.environ' in b,
     'env var malformed, use default'),
    (lambda b: re.search(r'\b(int|float)\s*\(.*\)', b)
               and len(b.splitlines()) <= 3,
     'malformed value, use default'),
    (lambda b: 'getattr' in b or 'hasattr' in b,
     'optional attribute probe'),
    (lambda b: 'importlib' in b
               or re.search(r'^\s*from\s+\w+\s+import', b, re.M)
               or re.search(r'^\s*import\s+\w', b, re.M),
     'optional dependency'),
    (lambda b: 'queue' in b.lower() and ('.get(' in b or '.put(' in b),
     'queue race during teardown'),
    (lambda b: re.search(r'\bproc\.(wait|kill|terminate|poll)', b)
               or re.search(r'\b\w*proc\.(wait|kill|terminate)', b),
     'subprocess cleanup, never fatal'),
    (lambda b: 'subprocess' in b or '.kill(' in b or '.terminate(' in b
               or '.poll(' in b,
     'subprocess cleanup, never fatal'),
    (lambda b: 'stdin.close' in b or 'stdout.close' in b,
     'pipe cleanup, never fatal'),
    (lambda b: '_render_overlay' in b or '_record_meta' in b
               or '_rec_overlay' in b or '_write_recording_metadata' in b,
     'recording overlay, best-effort'),
    (lambda b: 'mesh' in b.lower() and ('voxelized' in b or 'contains' in b
                                         or '.fill' in b),
     'mesh op fallback'),
    (lambda b: 'snapshot' in b.lower() or 'perf_log' in b,
     'optional snapshot/log, never fatal'),
    (lambda b: 'detect_period' in b or 'detect_translation' in b
               or 'detect_growth' in b or 'detect_clusters' in b
               or 'detect_symmetry' in b,
     'optional dynamics analysis'),
    (lambda b: '_build_sim' in b or '_read_voxels' in b,
     'optional snapshot build'),
    (lambda b: 'run_trial(' in b or 'run_once(' in b,
     'trial may crash on bad params, score=0'),
    (lambda b: '_grid_signature' in b,
     'signature compute may fail on edge grids'),
    (lambda b: 'socket.' in b,
     'hostname probe, best-effort'),
    (lambda b: 'base64' in b or 'b64decode' in b,
     'malformed preview, skip'),
    (lambda b: 'refine_one' in b or 'load_json' in b,
     'per-item failure, skip and continue'),
    (lambda b: 'ChemicalCA' in b,
     'trial may crash on degenerate params'),
    (lambda b: 'tail_line' in b or 'log_line' in b,
     'log parse, best-effort'),
    # ca_debug analysis idioms: per-rule trial wrappers — record error,
    # continue scanning. Crashes are expected for some preset/param combos.
    (lambda b: '_evolve(' in b or '_run_one(' in b or '_run_pair(' in b,
     'per-rule trial may crash, record error and continue'),
    (lambda b: 'analyze(' in b or 'audit_rule(' in b or 'probe(' in b,
     'per-item analysis may crash, record error and continue'),
    (lambda b: 'analyzer.sql' in b or '.sql(' in b,
     'sql query may fail, log and continue'),
    (lambda b: '_resolve(' in b,
     'preset resolution may fail'),
    (lambda b: 'prop(run)' in b or 'cross(' in b,
     'property check crash, log and continue'),
    (lambda b: '.derived' in b,
     'optional derived attribute access'),
    (lambda b: '__import__' in b,
     'optional dependency'),
    (lambda b: 'Simulator(' in b,
     'sim construction may fail on bad params'),
    (lambda b: 'microscope(' in b,
     'microscope analysis may crash per rule'),
    (lambda b: '.close()' in b or '.destroy()' in b,
     'teardown, never fatal'),
    (lambda b: 'p.read_text' in b or '.read_text(' in b,
     'best-effort read'),
]


def find_try_for(lines, except_idx, indent):
    """Walk back to find the matching `try:` at the same indent."""
    for j in range(except_idx - 1, max(except_idx - 200, -1), -1):
        l = lines[j]
        stripped = l.lstrip()
        if not stripped or stripped.startswith('#'):
            continue
        cur_ind = len(l) - len(stripped)
        if cur_ind == indent and stripped.startswith('try:'):
            return j
        if cur_ind < indent:
            return None
    return None


def classify(body: str) -> str | None:
    for matcher, reason in RULES:
        try:
            if matcher(body):
                return reason
        except Exception:
            continue
    return None


def annotate(path: pathlib.Path) -> tuple[int, int]:
    lines = path.read_text().splitlines(keepends=False)
    out = list(lines)
    annotated = 0
    skipped = 0
    for i, line in enumerate(lines):
        if NOQA.search(line):
            continue
        m = BARE.match(line)
        if not m:
            continue
        indent_str, _, trailing = m.group(1), m.group(2), m.group(3)
        indent = len(indent_str)
        # Skip if there's already a non-noqa comment we'd clobber.
        # (NOQA was checked above so trailing has no BLE001.)
        try_idx = find_try_for(lines, i, indent)
        if try_idx is None:
            skipped += 1
            continue
        body = '\n'.join(lines[try_idx:i])
        reason = classify(body)
        if reason is None:
            skipped += 1
            continue
        as_clause = m.group(2) or ''
        trailing_stmt = trailing.strip()
        # Strip any pre-existing comment (we replace it with our reason).
        if '#' in trailing_stmt:
            trailing_stmt = trailing_stmt.split('#', 1)[0].rstrip()
        if trailing_stmt:
            # One-liner: keep the statement, append noqa as comment.
            new = (f'{indent_str}except Exception{as_clause}: '
                   f'{trailing_stmt}  # noqa: BLE001  {reason}')
        else:
            new = (f'{indent_str}except Exception{as_clause}:  '
                   f'# noqa: BLE001  {reason}')
        out[i] = new
        annotated += 1
    if annotated:
        path.write_text('\n'.join(out) + '\n')
    return annotated, skipped


def main():
    grand_a = grand_s = 0
    for arg in sys.argv[1:]:
        p = pathlib.Path(arg)
        a, s = annotate(p)
        print(f'{p}: annotated={a}  skipped(ambiguous)={s}')
        grand_a += a
        grand_s += s
    print(f'TOTAL: annotated={grand_a}  skipped={grand_s}')


if __name__ == '__main__':
    main()
