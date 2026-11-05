"""Geometry identity for caching and restart."""

import hashlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def geometry_hash(coords_bohr: npt.ArrayLike, atomic_numbers: Sequence[int], settings: str) -> str:
    """SHA-256 over atomic numbers, bit-exact float64 coordinates and settings.

    Bit-exact on purpose: matched stencil points differ by tiny steps and must
    never collide, and any rounding would silently merge distinct geometries.
    """
    h = hashlib.sha256()
    h.update(b"saddle-geometry-v1\0")
    h.update(np.asarray(atomic_numbers, dtype="<i4").tobytes())
    h.update(b"\0")
    h.update(np.ascontiguousarray(coords_bohr, dtype="<f8").tobytes())
    h.update(b"\0")
    h.update(settings.encode())
    return h.hexdigest()
