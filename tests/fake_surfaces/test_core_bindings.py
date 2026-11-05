import numpy as np
import pytest

from saddle import _core


def test_muller_brown_from_python() -> None:
    mb = _core.MullerBrown()
    energy, gradient = mb.evaluate(np.array([-0.558224, 1.441726]))
    assert energy == pytest.approx(-146.69951720, abs=1e-5)
    assert gradient.shape == (2,)
    assert np.linalg.norm(gradient) < 1e-2


def test_morse_from_python() -> None:
    morse = _core.Morse(0.15, 1.0, 1.4)
    energy, gradient = morse.evaluate(np.array([0, 0, 0, 0, 0, 1.4], dtype=float))
    assert energy == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-12)


def test_harmonic_from_python() -> None:
    k = np.array([[2.0, 0.5], [0.5, 1.0]])
    h = _core.Harmonic(np.array([1.0, -1.0]), k)
    energy, gradient = h.evaluate(np.array([2.0, 0.0]))
    assert energy == pytest.approx(2.0)
    np.testing.assert_allclose(gradient, [2.5, 1.5])


def test_noisy_muller_brown_is_deterministic() -> None:
    noisy = _core.NoisyMullerBrown(1e-6, 7)
    x = np.array([0.1, 0.4])
    _, g1 = noisy.evaluate(x)
    _, g2 = noisy.evaluate(x)
    np.testing.assert_array_equal(g1, g2)


def test_shape_errors_become_value_errors() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        _core.MullerBrown().evaluate(np.zeros(3))
