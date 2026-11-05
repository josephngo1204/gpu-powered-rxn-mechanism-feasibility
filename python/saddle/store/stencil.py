"""Restartable finite-difference Hessian execution (spec §6, §9, §14).

Each stencil point is stored as soon as its gradient is available, keyed by
geometry hash and settings. A rerun after an interruption reuses finished
points and computes only the rest.
"""

import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from saddle import _core
from saddle.store.geometry import geometry_hash

FloatArray = npt.NDArray[np.float64]
Evaluate = Callable[[FloatArray], tuple[float, FloatArray]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stencil_point (
    key TEXT PRIMARY KEY,
    energy REAL NOT NULL,
    gradient BLOB NOT NULL,
    n INTEGER NOT NULL,
    created REAL NOT NULL
)
"""


class StencilStore:
    """SQLite store of evaluated stencil points."""

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> tuple[float, FloatArray] | None:
        row = self._conn.execute(
            "SELECT energy, gradient, n FROM stencil_point WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        energy, blob, n = row
        gradient = np.frombuffer(blob, dtype="<f8").copy()
        if gradient.size != n:
            raise ValueError(f"stored gradient for {key} is corrupt")
        return float(energy), gradient

    def put(self, key: str, energy: float, gradient: FloatArray) -> None:
        gradient = np.ascontiguousarray(gradient, dtype="<f8")
        existing = self.get(key)
        if existing is not None:
            if existing[0] == energy and np.array_equal(existing[1], gradient):
                return
            raise ValueError(f"conflicting result for stencil point {key}")
        self._conn.execute(
            "INSERT INTO stencil_point (key, energy, gradient, n, created) VALUES (?, ?, ?, ?, ?)",
            (key, energy, gradient.tobytes(), gradient.size, time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


@dataclass(frozen=True)
class HessianRun:
    raw: FloatArray
    hessian: FloatArray
    asymmetry: float
    step: float
    computed: int
    reused: int


def run_central_hessian(
    evaluate: Evaluate,
    coords_bohr: FloatArray,
    atomic_numbers: Sequence[int],
    step: float,
    store: StencilStore,
    settings: str,
) -> HessianRun:
    """Central-difference Hessian with per-point persistence.

    `settings` must fingerprint everything that changes the gradient (level
    of theory, grid, thresholds, math mode); points are reused only on an
    exact match.
    """
    points = _core.central_stencil(np.asarray(coords_bohr, dtype=np.float64), step)
    gradients: list[FloatArray] = []
    computed = reused = 0
    for _, _, displaced in points:
        key = geometry_hash(displaced, atomic_numbers, settings)
        cached = store.get(key)
        if cached is None:
            energy, gradient = evaluate(displaced)
            gradient = np.asarray(gradient, dtype=np.float64)
            store.put(key, float(energy), gradient)
            computed += 1
        else:
            gradient = cached[1]
            reused += 1
        gradients.append(gradient)
    raw, hessian, asymmetry = _core.assemble_central(gradients, step)
    return HessianRun(raw, hessian, float(asymmetry), step, computed, reused)
