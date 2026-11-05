import math

import pytest

from saddle import units
from saddle.analysis import tunneling

T = 298.15


def test_wigner_formula() -> None:
    u = 1000.0 * units.PLANCK_J_S * units.SPEED_OF_LIGHT_M_S * 100 / (units.BOLTZMANN_J_K * T)
    assert tunneling.wigner(1000.0, T) == pytest.approx(1.0 + u * u / 24.0)
    assert tunneling.wigner(1000.0, T) == pytest.approx(1.970, abs=1e-3)


def test_wigner_accepts_signed_imaginary_frequency() -> None:
    assert tunneling.wigner(-1000.0, T) == tunneling.wigner(1000.0, T)


def test_eckart_matches_wigner_for_shallow_tunneling() -> None:
    v = units.kcal_mol_to_hartree(20.0)
    k_e = tunneling.eckart(100.0, v, v, T)
    k_w = tunneling.wigner(100.0, T)
    assert k_e == pytest.approx(k_w, abs=2e-3)


def test_eckart_exceeds_wigner_for_deep_tunneling() -> None:
    v = units.kcal_mol_to_hartree(10.0)
    assert tunneling.eckart(2000.0, v, v, T) > 2.0 * tunneling.wigner(2000.0, T)


def test_eckart_approaches_one_at_high_temperature() -> None:
    v = units.kcal_mol_to_hartree(15.0)
    assert tunneling.eckart(1200.0, v, units.kcal_mol_to_hartree(25.0), 5000.0) == pytest.approx(
        1.0, abs=0.02
    )


def test_eckart_is_independent_of_direction() -> None:
    v1 = units.kcal_mol_to_hartree(12.0)
    v2 = units.kcal_mol_to_hartree(30.0)
    assert tunneling.eckart(1500.0, v1, v2, T) == pytest.approx(
        tunneling.eckart(1500.0, v2, v1, T), rel=1e-6
    )


def test_eckart_transmission_is_half_at_symmetric_barrier_top_in_classical_limit() -> None:
    v = units.kcal_mol_to_hartree(50.0)
    p = tunneling.eckart_transmission(v, 1.0, v, v)
    # Asymptotic limit; the finite-alpha correction here is ~6e-6.
    assert p == pytest.approx(0.5, abs=1e-4)


def test_eckart_transmission_limits() -> None:
    v = units.kcal_mol_to_hartree(10.0)
    assert tunneling.eckart_transmission(0.0, 1500.0, v, v) == 0.0
    assert tunneling.eckart_transmission(10 * v, 1500.0, v, v) == pytest.approx(1.0, abs=1e-6)


def test_rejects_bad_barriers() -> None:
    with pytest.raises(ValueError):
        tunneling.eckart(1000.0, -0.01, 0.02, T)
    with pytest.raises(ValueError):
        tunneling.eckart(0.0, 0.01, 0.02, T)


def test_result_bundle_reports_both_separately() -> None:
    v = units.kcal_mol_to_hartree(15.0)
    res = tunneling.corrections(-1500.0, v, v, T)
    assert res.wigner == tunneling.wigner(1500.0, T)
    assert res.eckart == tunneling.eckart(1500.0, v, v, T)
    assert math.isfinite(res.eckart_delta_g_kcal_mol)
    assert res.eckart_delta_g_kcal_mol < 0
