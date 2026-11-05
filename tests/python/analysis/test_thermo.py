import math

import numpy as np
import pytest

from saddle import elements, units
from saddle.analysis import thermo

T = 298.15
J_MOL_K_PER_HARTREE_K = units.HARTREE_TO_J * units.AVOGADRO


def s_j_mol_k(entropy_hartree_k: float) -> float:
    return entropy_hartree_k * J_MOL_K_PER_HARTREE_K


def water() -> tuple[np.ndarray, np.ndarray]:
    r, half = 0.9572, math.radians(104.52) / 2
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [r * math.sin(half), r * math.cos(half), 0.0],
            [-r * math.sin(half), r * math.cos(half), 0.0],
        ]
    )
    masses = np.array(
        [elements.isotope_mass("O"), elements.isotope_mass("H"), elements.isotope_mass("H")]
    )
    return xyz.ravel() / units.BOHR_TO_ANGSTROM, masses


def test_isotope_masses() -> None:
    assert elements.isotope_mass("H") == 1.00782503223
    assert elements.isotope_mass("C") == 12.0
    assert elements.isotope_mass("Cl") == 34.968852682
    with pytest.raises(KeyError):
        elements.isotope_mass("Og")


def test_argon_translational_entropy_matches_sackur_tetrode() -> None:
    res = thermo.rrho(
        coords_bohr=np.zeros(3),
        masses_amu=np.array([39.9623831237]),
        frequencies_cm=np.array([]),
        temperature_k=T,
        symmetry_number=1,
        multiplicity=1,
    )
    # NIST S = 154.85 J/mol/K is at 1 bar; at 1 atm subtract R ln(1.01325) = 0.109.
    assert s_j_mol_k(res.entropy_translational) == pytest.approx(154.85 - 0.109, abs=0.01)
    assert res.entropy_rotational == 0.0
    assert res.zpe == 0.0


def test_water_entropy_matches_nist() -> None:
    coords, masses = water()
    res = thermo.rrho(
        coords_bohr=coords,
        masses_amu=masses,
        frequencies_cm=np.array([1595.0, 3657.0, 3756.0]),
        temperature_k=T,
        symmetry_number=2,
        multiplicity=1,
    )
    assert s_j_mol_k(res.entropy_rotational) == pytest.approx(43.9, abs=0.2)
    # NIST S = 188.84 J/mol/K at 1 bar -> 188.73 at 1 atm.
    assert s_j_mol_k(res.entropy) == pytest.approx(188.73, abs=0.2)
    assert units.hartree_to_kcal_mol(res.zpe) == pytest.approx(
        0.5 * (1595 + 3657 + 3756) / units.HARTREE_TO_WAVENUMBER * units.HARTREE_TO_KCAL_MOL
    )
    assert units.hartree_to_kcal_mol(res.zpe) == pytest.approx(12.88, abs=0.01)


def test_symmetry_number_lowers_entropy_by_r_ln_sigma() -> None:
    coords, masses = water()
    kw = dict(
        coords_bohr=coords,
        masses_amu=masses,
        frequencies_cm=np.array([1595.0, 3657.0, 3756.0]),
        temperature_k=T,
        multiplicity=1,
    )
    s1 = thermo.rrho(symmetry_number=1, **kw).entropy  # type: ignore[arg-type]
    s2 = thermo.rrho(symmetry_number=2, **kw).entropy  # type: ignore[arg-type]
    assert s_j_mol_k(s1 - s2) == pytest.approx(units.GAS_CONSTANT_J_MOL_K * math.log(2), rel=1e-10)


def test_linear_molecule_rotational_entropy() -> None:
    # CO, r = 1.128 A, sigma = 1: S_rot ~ 47.2 J/mol/K
    coords = np.array([0, 0, 0, 0, 0, 1.128]) / units.BOHR_TO_ANGSTROM
    res = thermo.rrho(
        coords_bohr=coords,
        masses_amu=np.array([12.0, 15.99491461957]),
        frequencies_cm=np.array([2170.0]),
        temperature_k=T,
        symmetry_number=1,
        multiplicity=1,
    )
    assert res.linear
    assert s_j_mol_k(res.entropy_rotational) == pytest.approx(47.2, abs=0.2)


def test_electronic_entropy_from_multiplicity() -> None:
    res = thermo.rrho(
        coords_bohr=np.zeros(3),
        masses_amu=np.array([1.00782503223]),
        frequencies_cm=np.array([]),
        temperature_k=T,
        symmetry_number=1,
        multiplicity=2,
    )
    assert s_j_mol_k(res.entropy_electronic) == pytest.approx(
        units.GAS_CONSTANT_J_MOL_K * math.log(2)
    )


def test_rejects_imaginary_frequencies() -> None:
    with pytest.raises(ValueError, match="imaginary"):
        thermo.rrho(
            coords_bohr=np.zeros(3),
            masses_amu=np.array([1.0]),
            frequencies_cm=np.array([-500.0]),
            temperature_k=T,
            symmetry_number=1,
            multiplicity=1,
        )


def test_standard_state_correction_is_1_89_kcal_per_species() -> None:
    assert units.hartree_to_kcal_mol(thermo.standard_state_correction(T)) == pytest.approx(
        1.89, abs=0.005
    )


def test_bimolecular_barrier_drops_by_1_89() -> None:
    # TS is one species, reactants are two: net -1 correction on the barrier.
    corr = thermo.standard_state_correction(T)
    delta = corr - 2 * corr
    assert units.hartree_to_kcal_mol(delta) == pytest.approx(-1.89, abs=0.005)


def test_quasi_harmonic_weight() -> None:
    assert thermo.qrrho_weight(100.0) == pytest.approx(0.5)
    assert thermo.qrrho_weight(1000.0) == pytest.approx(1.0, abs=1e-4)
    assert thermo.qrrho_weight(10.0) == pytest.approx(0.0, abs=1e-4)


def test_quasi_harmonic_tames_low_modes_and_matches_rrho_for_stiff_modes() -> None:
    coords, masses = water()
    kw = dict(
        coords_bohr=coords, masses_amu=masses, temperature_k=T, symmetry_number=2, multiplicity=1
    )
    stiff = np.array([1595.0, 3657.0, 3756.0])
    soft = np.array([12.0, 3657.0, 3756.0])
    h_stiff = thermo.rrho(frequencies_cm=stiff, **kw)  # type: ignore[arg-type]
    q_stiff = thermo.rrho(frequencies_cm=stiff, quasi_harmonic=True, **kw)  # type: ignore[arg-type]
    assert q_stiff.entropy_vibrational == pytest.approx(h_stiff.entropy_vibrational, rel=1e-3)
    h_soft = thermo.rrho(frequencies_cm=soft, **kw)  # type: ignore[arg-type]
    q_soft = thermo.rrho(frequencies_cm=soft, quasi_harmonic=True, **kw)  # type: ignore[arg-type]
    assert q_soft.entropy_vibrational < h_soft.entropy_vibrational


def test_free_energy_assembly() -> None:
    coords, masses = water()
    res = thermo.rrho(
        coords_bohr=coords,
        masses_amu=masses,
        frequencies_cm=np.array([1595.0, 3657.0, 3756.0]),
        temperature_k=T,
        symmetry_number=2,
        multiplicity=1,
        standard_state="1M",
    )
    expected = (
        res.zpe + res.thermal_enthalpy - T * res.entropy + thermo.standard_state_correction(T)
    )
    assert res.g_correction == pytest.approx(expected, rel=1e-14)
    gas = thermo.rrho(
        coords_bohr=coords,
        masses_amu=masses,
        frequencies_cm=np.array([1595.0, 3657.0, 3756.0]),
        temperature_k=T,
        symmetry_number=2,
        multiplicity=1,
        standard_state="1atm",
    )
    assert res.g_correction - gas.g_correction == pytest.approx(thermo.standard_state_correction(T))
