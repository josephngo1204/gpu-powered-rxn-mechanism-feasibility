"""Element symbols and atomic numbers."""

from typing import Final

_SYMBOLS: Final = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()

ATOMIC_NUMBER: Final[dict[str, int]] = {s: z for z, s in enumerate(_SYMBOLS, start=1)}


def atomic_number(symbol: str) -> int:
    """Atomic number for an element symbol; raises KeyError if unknown."""
    return ATOMIC_NUMBER[symbol.capitalize()]


# Most abundant isotope masses (amu), AME2020. Covers the chemistry in scope;
# extend with a citation before adding elements.
ISOTOPE_MASS: Final[dict[str, float]] = {
    "H": 1.00782503223,
    "D": 2.01410177812,
    "He": 4.00260325413,
    "Li": 7.0160034366,
    "B": 11.00930536,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Ne": 19.9924401762,
    "Na": 22.989769282,
    "Si": 27.97692653465,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Ar": 39.9623831237,
    "Br": 78.9183376,
    "I": 126.9044719,
}


def isotope_mass(symbol: str) -> float:
    """Mass of the most abundant isotope in amu ("D" for deuterium)."""
    return ISOTOPE_MASS[symbol if symbol == "D" else symbol.capitalize()]
