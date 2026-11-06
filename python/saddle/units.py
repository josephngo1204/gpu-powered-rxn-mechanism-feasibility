"""Physical constants and unit conversions (CODATA 2018).

Saddle works internally in bohr and hartree (Eh). Convert only at input and
output boundaries.
"""

from typing import Final

PLANCK_J_S: Final = 6.62607015e-34
BOLTZMANN_J_K: Final = 1.380649e-23
AVOGADRO: Final = 6.02214076e23
SPEED_OF_LIGHT_M_S: Final = 299792458.0
GAS_CONSTANT_J_MOL_K: Final = BOLTZMANN_J_K * AVOGADRO

BOHR_TO_ANGSTROM: Final = 0.529177210903
HARTREE_TO_J: Final = 4.3597447222071e-18
HARTREE_TO_KJ_MOL: Final = HARTREE_TO_J * AVOGADRO / 1000.0
CAL_TO_J: Final = 4.184
HARTREE_TO_KCAL_MOL: Final = HARTREE_TO_KJ_MOL / CAL_TO_J
HARTREE_TO_WAVENUMBER: Final = 219474.6313632
AMU_TO_ELECTRON_MASS: Final = 1822.888486209
AMU_TO_KG: Final = 1.66053906660e-27

ATM_TO_PA: Final = 101325.0


def angstrom_to_bohr(x: float) -> float:
    return x / BOHR_TO_ANGSTROM


def bohr_to_angstrom(x: float) -> float:
    return x * BOHR_TO_ANGSTROM


def hartree_to_kcal_mol(e: float) -> float:
    return e * HARTREE_TO_KCAL_MOL


def kcal_mol_to_hartree(e: float) -> float:
    return e / HARTREE_TO_KCAL_MOL


def hartree_to_kj_mol(e: float) -> float:
    return e * HARTREE_TO_KJ_MOL


def kj_mol_to_hartree(e: float) -> float:
    return e / HARTREE_TO_KJ_MOL


def kcal_mol_to_j_mol(e: float) -> float:
    return e * CAL_TO_J * 1000.0


def j_mol_to_kcal_mol(e: float) -> float:
    return e / (CAL_TO_J * 1000.0)
