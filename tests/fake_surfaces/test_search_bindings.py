from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pytest

from saddle import _core

FloatArray = npt.NDArray[np.float64]
Z = [6, 1, 6]
MASSES = np.array([12.0, 1.00782503223, 12.0])
AC = 6.5


def h_transfer() -> _core.PairModel:
    return _core.PairModel([(0, 2, 1.0, AC)], [(0, 1, 0.15, 1.0, 1.8), (1, 2, 0.15, 1.0, 1.8)])


def geom(h_x: float, h_y: float = 0.0) -> FloatArray:
    return np.array([0.0, 0, 0, h_x, h_y, 0, AC, 0, 0])


class PyHarmonic:
    """A pure-Python model, to show C++ search runs on Python-side surfaces."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, coords: FloatArray) -> tuple[float, FloatArray]:
        self.calls += 1
        k = np.array([2.0, 5.0])
        return float(0.5 * np.sum(k * (coords - 1.0) ** 2)), k * (coords - 1.0)


def test_minimize_python_model() -> None:
    model = PyHarmonic()
    r = _core.minimize(model, np.array([3.0, -2.0]), [])
    assert r["converged"]
    np.testing.assert_allclose(r["coords"], [1.0, 1.0], atol=1e-5)
    assert r["n_gradients"] == model.calls


def test_saddle_search_and_irc_on_double_well() -> None:
    model = h_transfer()
    ts = _core.saddle_search(model, geom(3.1, 0.05), Z)
    assert ts["converged"]
    x = ts["coords"]
    assert x[3] - x[0] == pytest.approx(0.5 * (x[6] - x[0]), abs=1e-4)
    points = _core.central_stencil(x, 1e-4)
    _, hess, _ = _core.assemble_central([model.evaluate(c)[1] for _, _, c in points], 1e-4)
    path = _core.irc(model, x, Z, hess, MASSES)
    for branch in ("forward", "reverse"):
        assert path[branch]["endpoint"]["converged"]
        e = path[branch]["energies"]
        assert all(b < a for a, b in pairwise(e))
    assert path["transition_eigenvalue"] < 0


def test_probe_noise_and_optimal_step() -> None:
    noisy = _core.NoisyMullerBrown(1e-6, 3)
    eps, n = _core.probe_noise(noisy, np.array([0.1, 0.4]), np.array([1.0, 1.0]))
    assert 5e-7 < eps < 2e-6 and n == 16
    assert _core.optimal_step(1e-6, 2.0) == pytest.approx(0.01145, rel=1e-3)


def test_bad_python_model_gradient_is_rejected() -> None:
    class Bad:
        def evaluate(self, coords: FloatArray) -> tuple[float, FloatArray]:
            return 0.0, np.zeros(coords.size + 1)

    with pytest.raises(ValueError, match="gradient length"):
        _core.minimize(Bad(), np.zeros(2), [])
