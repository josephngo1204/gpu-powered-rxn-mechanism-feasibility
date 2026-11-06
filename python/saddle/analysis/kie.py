"""Kinetic isotope effects from isotopic rediagonalization.

The same Cartesian Hessian is rediagonalized with substituted masses; no new
electronic-structure work is needed. The ratio uses full RRHO partition
functions, which under the harmonic approximation is equivalent to the
Bigeleisen-Mayer form (the explicit moment-of-inertia ratio replaces the
Teller-Redlich imaginary-frequency ratio). Frequencies are unscaled.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from saddle import _core, units
from saddle.analysis import thermo, tunneling

FloatArray = npt.NDArray[np.float64]

ZERO_MODE_CM = 1.0


@dataclass(frozen=True)
class IsotopologuePair:
    hessian: FloatArray
    coords_bohr: FloatArray
    masses_light: FloatArray
    masses_heavy: FloatArray
    symmetry_light: int = 1
    symmetry_heavy: int = 1
    multiplicity: int = 1


@dataclass(frozen=True)
class IsotopeEffect:
    kie: float
    kie_with_wigner: float
    ts_imaginary_light_cm: float
    ts_imaginary_heavy_cm: float


def _corrected_free_energy(
    s: IsotopologuePair, masses: FloatArray, symmetry: int, temperature_k: float, is_ts: bool
) -> tuple[float, float]:
    if masses.shape != (s.coords_bohr.size // 3,):
        raise ValueError("isotopologue masses do not match the atom count")
    vib = _core.analyze_vibrations(s.hessian, s.coords_bohr, masses)
    freqs = np.asarray(vib.frequencies_cm)
    imaginary = freqs[freqs < -ZERO_MODE_CM]
    expected = 1 if is_ts else 0
    if imaginary.size != expected:
        kind = "transition state" if is_ts else "minimum"
        raise ValueError(f"{kind} has {imaginary.size} imaginary modes, expected {expected}")
    real = freqs[freqs > ZERO_MODE_CM]
    if real.size + imaginary.size != freqs.size:
        raise ValueError("near-zero vibrational modes; check the projection and geometry")
    res = thermo.rrho(
        coords_bohr=s.coords_bohr,
        masses_amu=masses,
        frequencies_cm=real,
        temperature_k=temperature_k,
        symmetry_number=symmetry,
        multiplicity=s.multiplicity,
    )
    imag = float(imaginary[0]) if is_ts else 0.0
    return res.g_correction, imag


def isotope_effect(
    reactants: Sequence[IsotopologuePair], ts: IsotopologuePair, temperature_k: float
) -> IsotopeEffect:
    """KIE = k_light / k_heavy. Electronic energies cancel between isotopologues."""
    dg = {}
    imag = {}
    for label in ("light", "heavy"):
        g_ts, nu = _corrected_free_energy(
            ts,
            getattr(ts, f"masses_{label}"),
            getattr(ts, f"symmetry_{label}"),
            temperature_k,
            True,
        )
        g_r = sum(
            _corrected_free_energy(
                r,
                getattr(r, f"masses_{label}"),
                getattr(r, f"symmetry_{label}"),
                temperature_k,
                False,
            )[0]
            for r in reactants
        )
        dg[label] = g_ts - g_r
        imag[label] = nu
    kt = units.BOLTZMANN_J_K * temperature_k / units.HARTREE_TO_J
    kie = math.exp(-(dg["light"] - dg["heavy"]) / kt)
    wigner_ratio = tunneling.wigner(imag["light"], temperature_k) / tunneling.wigner(
        imag["heavy"], temperature_k
    )
    return IsotopeEffect(kie, kie * wigner_ratio, imag["light"], imag["heavy"])
