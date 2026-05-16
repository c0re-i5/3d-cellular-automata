"""
GPU validation of the FCC lattice GLSL header.

Standalone diagnostic that:
  1. Boots a headless moderngl context (standalone, requires GL 4.3).
  2. Compiles a minimal compute shader using the generated lattice.py header.
  3. Dispatches a "delta-function Laplacian": writes 1.0 at a single cell,
     runs a shader that sums the 12 nearest neighbours at every cell, and
     checks the result.

Pass criteria:
  - Shader compiles (i.e. the generated GLSL header is syntactically valid
    and the LATTICE_NEIGHBOURS array is consumable from a for-loop).
  - Exactly 12 output cells are non-zero.
  - Those 12 cells are at the centre plus each LATTICE_NEIGHBOURS offset
    (modulo the grid size — i.e. periodic boundary wrap works).
  - Each of those cells has value exactly 1.0.
  - Sum of output equals 12.0.

This is foundation work: until this passes, no FCC rule can be trusted.
Run after editing lattice.py:

    .venv/bin/python validate_lattice_glsl.py
"""

from __future__ import annotations

import sys

import numpy as np
import moderngl

from lattice import FCC


# Use a small grid so we can eyeball any failure. 8^3 = 512 cells, single
# delta at the centre, periodic boundaries.
GRID = 8
CENTRE = (GRID // 2, GRID // 2, GRID // 2)


COMPUTE_SHADER_TEMPLATE = """\
#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(r32f, binding = 0) readonly  uniform image3D field_in;
layout(r32f, binding = 1) writeonly uniform image3D field_out;

{lattice_header}

void main() {{
    ivec3 sz  = imageSize(field_in);
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(pos, sz))) return;

    float acc = 0.0;
    for (int k = 0; k < LATTICE_N_NEIGHBOURS; ++k) {{
        // Periodic wrap. (+sz) before % handles negative offsets cleanly.
        ivec3 npos = (pos + LATTICE_NEIGHBOURS[k] + sz) % sz;
        acc += imageLoad(field_in, npos).r;
    }}
    imageStore(field_out, pos, vec4(acc, 0.0, 0.0, 0.0));
}}
"""


def main() -> int:
    print("[validate] booting moderngl standalone context (require=430)...")
    ctx = moderngl.create_standalone_context(require=430)
    print(f"[validate]   GL vendor   = {ctx.info.get('GL_VENDOR', '?')}")
    print(f"[validate]   GL renderer = {ctx.info.get('GL_RENDERER', '?')}")
    print(f"[validate]   GL version  = {ctx.info.get('GL_VERSION', '?')}")

    # ----- Compile shader ---------------------------------------------------
    src = COMPUTE_SHADER_TEMPLATE.format(lattice_header=FCC.glsl_header())
    print(f"[validate] compiling compute shader ({len(src)} chars)...")
    try:
        prog = ctx.compute_shader(src)
    except moderngl.Error as e:
        print("[validate] SHADER COMPILE FAILED")
        print(str(e))
        print("--- shader source ---")
        for i, line in enumerate(src.splitlines(), 1):
            print(f"{i:4d}  {line}")
        return 1
    print("[validate]   ok")

    # ----- Allocate textures -----------------------------------------------
    # Delta input: 1.0 at the centre, 0 elsewhere. Channel order is (W, H, D)
    # in moderngl's texture3d, and numpy is laid out (D, H, W) for the upload
    # buffer. Stick to a symmetric grid so the distinction is invisible here.
    field = np.zeros((GRID, GRID, GRID), dtype=np.float32)
    field[CENTRE] = 1.0

    tex_in = ctx.texture3d((GRID, GRID, GRID), 1, field.tobytes(), dtype='f4')
    tex_in.filter = (moderngl.NEAREST, moderngl.NEAREST)

    tex_out = ctx.texture3d((GRID, GRID, GRID), 1, b'\x00' * (GRID ** 3 * 4),
                            dtype='f4')
    tex_out.filter = (moderngl.NEAREST, moderngl.NEAREST)

    # ----- Dispatch --------------------------------------------------------
    tex_in.bind_to_image(0, read=True,  write=False)
    tex_out.bind_to_image(1, read=False, write=True)
    groups = GRID // 4
    prog.run(groups, groups, groups)
    ctx.finish()
    print(f"[validate] dispatched ({groups}, {groups}, {groups}) workgroups")

    # ----- Read back -------------------------------------------------------
    raw = tex_out.read()
    out = np.frombuffer(raw, dtype=np.float32).reshape((GRID, GRID, GRID))

    # ----- Expected --------------------------------------------------------
    # For a delta input at CENTRE, the output should be 1.0 at exactly the
    # 12 neighbour positions of CENTRE (with wrap), and 0 everywhere else.
    expected = np.zeros_like(out)
    cx, cy, cz = CENTRE
    expected_positions = set()
    for off in FCC.neighbours:
        nx = (cx + int(off[0])) % GRID
        ny = (cy + int(off[1])) % GRID
        nz = (cz + int(off[2])) % GRID
        expected[nx, ny, nz] = 1.0
        expected_positions.add((nx, ny, nz))

    # ----- Check -----------------------------------------------------------
    failures = []

    nz = int(np.count_nonzero(out))
    if nz != 12:
        failures.append(f"expected exactly 12 non-zero cells, got {nz}")

    s = float(out.sum())
    if not np.isclose(s, 12.0):
        failures.append(f"expected sum == 12.0, got {s}")

    if not np.allclose(out, expected):
        diff = np.where(out != expected)
        n_diff = len(diff[0])
        sample = list(zip(*diff))[:6]
        failures.append(
            f"output != expected at {n_diff} positions; first few:\n"
            + "\n".join(
                f"    pos={p}  out={out[p]!r}  expected={expected[p]!r}"
                for p in sample
            )
        )

    actual_positions = {
        tuple(int(x) for x in pos) for pos in zip(*np.where(out > 0))
    }
    missing = expected_positions - actual_positions
    extra = actual_positions - expected_positions
    if missing:
        failures.append(f"missing positions: {sorted(missing)}")
    if extra:
        failures.append(f"unexpected positions: {sorted(extra)}")

    # ----- Report ----------------------------------------------------------
    if failures:
        print("\n[validate] FAILED")
        for f in failures:
            print("  -", f)
        return 1

    print(f"[validate]   12 neighbours found at correct positions")
    print(f"[validate]   sum = {s:.1f}  (expected 12.0)")
    print(f"[validate]   max = {out.max():.6f}  min = {out.min():.6f}")
    print("[validate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
