import numpy as np
import numpy.typing as npt
import pytest

from saddle import _core
from saddle.analysis.kie import IsotopologuePair, isotope_effect

FloatArray = npt.NDArray[np.float64]
T = 298.15
H, D, C12, C13 = 1.00782503223, 2.01410177812, 12.0, 13.00335483507

# Triangle C (0), H (1), acceptor B (2), coordinates in bohr.
COORDS = np.array([0.0, 0.0, 0.0, 2.06, 0.0, 0.0, 1.6, 2.6, 0.0])


def _r(i: int, j: int) -> float:
    return float(np.linalg.norm(COORDS[3 * i : 3 * i + 3] - COORDS[3 * j : 3 * j + 3]))


def _hessian(k_ch: float, k_hb: float, k_cb: float) -> FloatArray:
    model = _core.Springs([(0, 1, k_ch, _r(0, 1)), (1, 2, k_hb, _r(1, 2)), (0, 2, k_cb, _r(0, 2))])
    points = _core.central_stencil(COORDS, 1e-4)
    _, sym, _ = _core.assemble_central([model.evaluate(c)[1] for _, _, c in points], 1e-4)
    return np.asarray(sym)


REACTANT = _hessian(0.33, 0.02, 0.30)  # stiff C-H stretch near 3000 cm^-1
TS = _hessian(-0.08, 0.02, 0.30)  # C-H stretch becomes the reaction coordinate


def _pair(hess: FloatArray, light: list[float], heavy: list[float]) -> IsotopologuePair:
    return IsotopologuePair(hess, COORDS, np.array(light), np.array(heavy))


def test_identical_masses_give_unity() -> None:
    m = [C12, H, C12]
    res = isotope_effect([_pair(REACTANT, m, m)], _pair(TS, m, m), T)
    assert res.kie == pytest.approx(1.0, abs=1e-12)


def test_primary_deuterium_effect_is_large_and_normal() -> None:
    res = isotope_effect(
        [_pair(REACTANT, [C12, H, C12], [C12, D, C12])], _pair(TS, [C12, H, C12], [C12, D, C12]), T
    )
    assert 2.0 < res.kie < 8.0
    assert res.ts_imaginary_light_cm < 0 and res.ts_imaginary_heavy_cm < 0
    assert abs(res.ts_imaginary_heavy_cm) < abs(res.ts_imaginary_light_cm)
    assert res.kie_with_wigner > res.kie


def test_heavy_atom_effect_is_small() -> None:
    res = isotope_effect(
        [_pair(REACTANT, [C12, H, C12], [C12, H, C13])], _pair(TS, [C12, H, C12], [C12, H, C13]), T
    )
    assert 0.95 < res.kie < 1.1


def test_ts_must_have_exactly_one_imaginary_mode() -> None:
    m = [C12, H, C12]
    with pytest.raises(ValueError, match="imaginary"):
        isotope_effect([_pair(REACTANT, m, m)], _pair(REACTANT, m, m), T)


def test_reactant_must_be_a_minimum() -> None:
    m = [C12, H, C12]
    with pytest.raises(ValueError, match="imaginary"):
        isotope_effect([_pair(TS, m, m)], _pair(TS, m, m), T)


def test_mass_arrays_must_match() -> None:
    with pytest.raises(ValueError):
        isotope_effect(
            [_pair(REACTANT, [C12, H, C12], [C12, D])], _pair(TS, [C12, H, C12], [C12, D, C12]), T
        )
