from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from saddle import _core
from saddle.store.geometry import geometry_hash
from saddle.store.stencil import StencilStore, run_central_hessian

K = np.array([[3.0, 0.2, -0.1], [0.2, 1.0, 0.4], [-0.1, 0.4, 2.0]])
CENTER = np.array([0.5, -0.2, 1.0])
X0 = np.array([0.1, 0.2, 0.3])
Z: tuple[int, ...] = ()
SETTINGS = "fake-harmonic/v1"


class Counter:
    """Evaluate a harmonic surface, optionally failing after n calls."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.model = _core.Harmonic(CENTER, K)
        self.calls = 0
        self.fail_after = fail_after

    def __call__(self, coords: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated worker crash")
        self.calls += 1
        energy, gradient = self.model.evaluate(coords)
        return energy, gradient


def test_geometry_hash_is_bit_exact_and_includes_atoms() -> None:
    a = geometry_hash(X0, (1, 1, 1), "s")
    assert a == geometry_hash(X0.copy(), (1, 1, 1), "s")
    assert a != geometry_hash(X0 + 1e-15, (1, 1, 1), "s")
    assert a != geometry_hash(X0, (1, 1, 6), "s")
    assert a != geometry_hash(X0, (1, 1, 1), "t")


def test_full_run_matches_exact_hessian(tmp_path: Path) -> None:
    store = StencilStore(tmp_path / "s.sqlite")
    result = run_central_hessian(Counter(), X0, Z, 1e-3, store, SETTINGS)
    np.testing.assert_allclose(result.hessian, K, atol=1e-9)
    assert result.computed == 6
    assert result.reused == 0
    assert result.asymmetry < 1e-12


def test_interrupted_run_resumes_without_recomputing(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    crashing = Counter(fail_after=4)
    with pytest.raises(RuntimeError, match="crash"):
        run_central_hessian(crashing, X0, Z, 1e-3, StencilStore(path), SETTINGS)
    assert crashing.calls == 4

    resumed = Counter()
    result = run_central_hessian(resumed, X0, Z, 1e-3, StencilStore(path), SETTINGS)
    assert resumed.calls == 2
    assert result.reused == 4
    assert result.computed == 2
    np.testing.assert_allclose(result.hessian, K, atol=1e-9)


def test_different_settings_do_not_reuse(tmp_path: Path) -> None:
    store = StencilStore(tmp_path / "s.sqlite")
    run_central_hessian(Counter(), X0, Z, 1e-3, store, SETTINGS)
    other = Counter()
    run_central_hessian(other, X0, Z, 1e-3, store, "fake-harmonic/v2")
    assert other.calls == 6


def test_store_rejects_conflicting_rewrite(tmp_path: Path) -> None:
    store = StencilStore(tmp_path / "s.sqlite")
    store.put("k", 1.0, np.zeros(3))
    store.put("k", 1.0, np.zeros(3))  # identical rewrite is fine
    with pytest.raises(ValueError, match="conflict"):
        store.put("k", 2.0, np.zeros(3))


def test_core_stencil_and_assembly_bindings() -> None:
    points = _core.central_stencil(X0, 1e-2)
    assert len(points) == 6
    coordinate, sign, coords = points[1]
    assert (coordinate, sign) == (0, -1)
    np.testing.assert_allclose(coords, X0 - [1e-2, 0, 0])
    grads = [K @ (c - CENTER) for _, _, c in points]
    _raw, sym, asym = _core.assemble_central(grads, 1e-2)
    np.testing.assert_allclose(sym, K, atol=1e-12)
    assert asym < 1e-12
