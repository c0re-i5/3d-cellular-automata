"""
FCC field storage on the GPU.

An :class:`FCCField` is a pair of 3D textures (``tex_a``/``tex_b``) holding
``channels`` floats per lattice cell, in dense FCC primitive-cell coordinates.
The pair supports ping-pong updates: the active read texture is ``current``,
the active write target is ``other``; :meth:`swap` flips them after a step.

Texture shape is ``(Na, Nb, Nc)`` in index space (rebuild :mod:`lattice` for
the basis vectors). All cells are real; there is no parity bit or validity
test.

Channel count
-------------
The legacy simulator uses ``rgba32f`` (4 channels) to give every rule four
fields to play with. FCC stays with the same default for parity with how
existing rule logic is written, but rules that need fewer channels can
allocate with ``channels=1`` or ``channels=2``.

Boundary mode
-------------
Defaults to periodic wrap (``repeat_x/y/z = True``). Rules and the raymarcher
can rely on hardware wrap for trilinear sampling at boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import moderngl


@dataclass(frozen=True)
class FCCFieldShape:
    """Index-space dimensions of an FCC field."""

    Na: int
    Nb: int
    Nc: int
    channels: int = 4

    @property
    def cell_count(self) -> int:
        return self.Na * self.Nb * self.Nc

    @property
    def texture_size(self) -> Tuple[int, int, int]:
        # moderngl wants (W, H, D); we treat a,b,c as W,H,D respectively.
        return (self.Na, self.Nb, self.Nc)

    def numpy_shape(self) -> Tuple[int, int, int, int]:
        # numpy lays out as (Nc, Nb, Na, channels) so that .tobytes() yields
        # the same byte order as moderngl's texture3d expects (W fastest,
        # then H, then D).
        return (self.Nc, self.Nb, self.Na, self.channels)


_DTYPE_TO_MGL = {
    np.dtype('float32'): 'f4',
    np.dtype('float16'): 'f2',
}


class FCCField:
    """Ping-pong pair of FCC 3D textures.

    Lifecycle: construct with a context and shape, then for each step do
    ``current.bind_to_image(0, read=True, write=False)``,
    ``other.bind_to_image(1, read=False, write=True)``, dispatch the rule
    program, then call :meth:`swap`.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        shape: FCCFieldShape,
        *,
        dtype: np.dtype = np.dtype('float32'),
        linear_filter: bool = True,
    ) -> None:
        if dtype not in _DTYPE_TO_MGL:
            raise ValueError(
                f"unsupported dtype {dtype}; expected float32 or float16"
            )
        self.ctx = ctx
        self.shape = shape
        self.dtype = dtype
        self._mgl_dtype = _DTYPE_TO_MGL[dtype]

        # Allocate two zeroed textures.
        bytes_per_cell = np.dtype(dtype).itemsize * shape.channels
        zero = b'\x00' * (shape.cell_count * bytes_per_cell)
        self.tex_a = ctx.texture3d(
            shape.texture_size, shape.channels, zero, dtype=self._mgl_dtype,
        )
        self.tex_b = ctx.texture3d(
            shape.texture_size, shape.channels, zero, dtype=self._mgl_dtype,
        )
        filt = moderngl.LINEAR if linear_filter else moderngl.NEAREST
        for tex in (self.tex_a, self.tex_b):
            tex.filter = (filt, filt)
            tex.repeat_x = True
            tex.repeat_y = True
            tex.repeat_z = True

        # Ping-pong state: read from current, write to other.
        self._read_is_a = True

    # ------------------------------------------------------------------
    # Ping-pong
    # ------------------------------------------------------------------

    @property
    def current(self) -> moderngl.Texture3D:
        """The texture holding the most recent state."""
        return self.tex_a if self._read_is_a else self.tex_b

    @property
    def other(self) -> moderngl.Texture3D:
        """The texture to write the next state into."""
        return self.tex_b if self._read_is_a else self.tex_a

    def swap(self) -> None:
        """Promote ``other`` to ``current`` after a successful dispatch."""
        self._read_is_a = not self._read_is_a

    # ------------------------------------------------------------------
    # CPU <-> GPU transfer
    # ------------------------------------------------------------------

    def upload(self, data: np.ndarray) -> None:
        """Write a numpy array into the current read texture.

        Shape must match :meth:`FCCFieldShape.numpy_shape` (``(Nc, Nb, Na, ch)``)
        and dtype must match the field's dtype.
        """
        expected = self.shape.numpy_shape()
        if data.shape != expected:
            raise ValueError(
                f"upload shape mismatch: got {data.shape}, expected {expected}"
            )
        if data.dtype != self.dtype:
            raise ValueError(
                f"upload dtype mismatch: got {data.dtype}, expected {self.dtype}"
            )
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        self.current.write(data.tobytes())

    def download(self) -> np.ndarray:
        """Read the current texture back as a numpy array.

        Returns shape ``(Nc, Nb, Na, channels)`` to match :meth:`upload`.
        """
        raw = self.current.read()
        arr = np.frombuffer(raw, dtype=self.dtype)
        return arr.reshape(self.shape.numpy_shape())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        for tex in (self.tex_a, self.tex_b):
            tex.release()


# ---------------------------------------------------------------------------
# Stand-alone self-check
# ---------------------------------------------------------------------------


def _self_check() -> None:
    print("[fcc_field] booting standalone context...")
    ctx = moderngl.create_standalone_context(require=430)
    try:
        shape = FCCFieldShape(Na=8, Nb=8, Nc=8, channels=4)
        field = FCCField(ctx, shape)

        # Round-trip: upload random data, download, expect bit-identity.
        rng = np.random.default_rng(0)
        data = rng.standard_normal(shape.numpy_shape()).astype(np.float32)
        field.upload(data)
        got = field.download()
        assert got.shape == data.shape
        assert np.array_equal(got, data), "round-trip mismatch"

        # Ping-pong sanity: swap returns to original after two swaps.
        a_before = field.current
        field.swap()
        assert field.current is not a_before
        field.swap()
        assert field.current is a_before

        field.release()
        print("[fcc_field] self-check: OK")
        print(f"  shape = {shape}")
        print(f"  cell_count = {shape.cell_count}")
    finally:
        ctx.release()


if __name__ == "__main__":
    _self_check()
