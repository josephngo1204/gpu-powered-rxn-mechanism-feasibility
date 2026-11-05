import math

from hypothesis import given
from hypothesis import strategies as st

from saddle import units

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)


def test_codata_2018_values() -> None:
    assert units.BOHR_TO_ANGSTROM == 0.529177210903
    assert math.isclose(units.HARTREE_TO_KCAL_MOL, 627.5094740631, rel_tol=1e-12)
    assert math.isclose(units.HARTREE_TO_KJ_MOL, 2625.4996394799, rel_tol=1e-12)
    assert math.isclose(units.HARTREE_TO_WAVENUMBER, 219474.6313632, rel_tol=1e-12)
    assert math.isclose(units.AMU_TO_ELECTRON_MASS, 1822.888486209, rel_tol=1e-12)


@given(finite)
def test_length_round_trip(x: float) -> None:
    assert math.isclose(units.bohr_to_angstrom(units.angstrom_to_bohr(x)), x, abs_tol=1e-9)


@given(finite)
def test_energy_round_trip(x: float) -> None:
    assert math.isclose(units.kcal_mol_to_hartree(units.hartree_to_kcal_mol(x)), x, abs_tol=1e-9)
    assert math.isclose(units.kj_mol_to_hartree(units.hartree_to_kj_mol(x)), x, abs_tol=1e-9)


def test_kcal_to_kj_is_thermochemical_calorie() -> None:
    one_kcal_in_hartree = units.kcal_mol_to_hartree(1.0)
    assert math.isclose(units.hartree_to_kj_mol(one_kcal_in_hartree), 4.184, rel_tol=1e-12)


def test_gas_constant_consistency() -> None:
    assert math.isclose(
        units.BOLTZMANN_J_K * units.AVOGADRO, units.GAS_CONSTANT_J_MOL_K, rel_tol=1e-12
    )
