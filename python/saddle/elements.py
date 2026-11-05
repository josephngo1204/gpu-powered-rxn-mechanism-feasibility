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
