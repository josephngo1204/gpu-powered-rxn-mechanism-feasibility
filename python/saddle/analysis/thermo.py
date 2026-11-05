"""Ideal-gas RRHO thermochemistry with quasi-harmonic entropy (spec §4).

G = E_high + ZPE + H_thermal - T*S + dG_standard_state

All outputs are in hartree (energies) and hartree/K (entropies). The
quasi-harmonic entropy follows Grimme, Chem. Eur. J. 18, 9955 (2012):
modes are interpolated toward free rotors with w = 1 / (1 + (w0/w)^4),
w0 = 100 cm^-1. Enthalpy stays harmonic.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from saddle import units

FloatArray = npt.NDArray[np.float64]

QRRHO_CUTOFF_CM = 100.0
QRRHO_AVERAGE_MOMENT_KG_M2 = 1e-44  # Grimme's B_av
LINEAR_TOLERANCE = 1e-8

_K = units.BOLTZMANN_J_K
_H = units.PLANCK_J_S
_C_CM = units.SPEED_OF_LIGHT_M_S * 100.0
_R = units.GAS_CONSTANT_J_MOL_K
_J_MOL_TO_HARTREE = 1.0 / (units.HARTREE_TO_J * units.AVOGADRO)
_BOHR_M = units.BOHR_TO_ANGSTROM * 1e-10


@dataclass(frozen=True)
class ThermoResult:
    temperature_k: float
    zpe: float
    thermal_enthalpy: float  # H(T) - E - ZPE, including the PV = RT term
    entropy_translational: float
    entropy_rotational: float
    entropy_vibrational: float
    entropy_electronic: float
    standard_state: float
    linear: bool
    symmetry_number: int
    quasi_harmonic: bool

    @property
    def entropy(self) -> float:
        return (
            self.entropy_translational
            + self.entropy_rotational
            + self.entropy_vibrational
            + self.entropy_electronic
        )

    @property
    def g_correction(self) -> float:
        """ZPE + H_thermal - T*S + standard-state term; add to E_high."""
        return (
            self.zpe
            + self.thermal_enthalpy
            - self.temperature_k * self.entropy
            + self.standard_state
        )


def standard_state_correction(temperature_k: float) -> float:
    """Per-species free-energy change from 1 atm to 1 mol/L: RT ln(c° R T / P°)."""
    c0 = 1000.0  # mol/m^3
    return (
        _R * temperature_k * math.log(c0 * _R * temperature_k / units.ATM_TO_PA) * _J_MOL_TO_HARTREE
    )


def qrrho_weight(frequency_cm: float) -> float:
    return 1.0 / (1.0 + (QRRHO_CUTOFF_CM / frequency_cm) ** 4)


def _principal_moments(coords_bohr: FloatArray, masses_amu: FloatArray) -> FloatArray:
    xyz = coords_bohr.reshape(-1, 3) * _BOHR_M
    m = masses_amu * units.AMU_TO_KG
    com = (m[:, None] * xyz).sum(axis=0) / m.sum()
    r = xyz - com
    inertia = np.zeros((3, 3))
    for mi, ri in zip(m, r, strict=True):
        inertia += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    return np.sort(np.linalg.eigvalsh(inertia))


def _vib_entropy_harmonic(nu_cm: float, temperature_k: float) -> float:
    x = _H * _C_CM * nu_cm / (_K * temperature_k)
    return _R * (x / math.expm1(x) - math.log(-math.expm1(-x)))


def _vib_entropy_free_rotor(nu_cm: float, temperature_k: float) -> float:
    mu = _H / (8.0 * math.pi**2 * _C_CM * nu_cm)
    mu_eff = mu * QRRHO_AVERAGE_MOMENT_KG_M2 / (mu + QRRHO_AVERAGE_MOMENT_KG_M2)
    return _R * (0.5 + math.log(math.sqrt(8.0 * math.pi**3 * mu_eff * _K * temperature_k / _H**2)))


def rrho(
    *,
    coords_bohr: npt.ArrayLike,
    masses_amu: npt.ArrayLike,
    frequencies_cm: npt.ArrayLike,
    temperature_k: float,
    symmetry_number: int,
    multiplicity: int,
    quasi_harmonic: bool = False,
    standard_state: Literal["1M", "1atm"] = "1atm",
) -> ThermoResult:
    """Thermal corrections for one species.

    `frequencies_cm` must contain only real vibrational modes. For a TS,
    remove the reaction-coordinate mode first; any negative value raises.
    """
    coords = np.asarray(coords_bohr, dtype=np.float64)
    masses = np.asarray(masses_amu, dtype=np.float64)
    freqs = np.asarray(frequencies_cm, dtype=np.float64)
    if not temperature_k > 0.0:
        raise ValueError("temperature must be positive")
    if symmetry_number < 1 or multiplicity < 1:
        raise ValueError("symmetry number and multiplicity must be >= 1")
    if coords.size != 3 * masses.size:
        raise ValueError("coords and masses disagree on atom count")
    if np.any(freqs <= 0.0):
        raise ValueError("imaginary or zero frequencies must be removed before RRHO")
    t = temperature_k
    rt = _R * t

    # Translation (ideal gas at 1 atm).
    m_kg = masses.sum() * units.AMU_TO_KG
    q_trans = (2.0 * math.pi * m_kg * _K * t / _H**2) ** 1.5 * _K * t / units.ATM_TO_PA
    s_trans = _R * (math.log(q_trans) + 2.5)
    h_thermal = 2.5 * rt

    # Rotation.
    s_rot = 0.0
    linear = False
    if masses.size > 1:
        moments = _principal_moments(coords, masses)
        linear = bool(moments[0] < LINEAR_TOLERANCE * moments[2])
        if linear:
            q_rot = 8.0 * math.pi**2 * moments[2] * _K * t / (symmetry_number * _H**2)
            s_rot = _R * (math.log(q_rot) + 1.0)
            h_thermal += rt
        else:
            q_rot = (
                math.sqrt(math.pi)
                / symmetry_number
                * math.prod(math.sqrt(8.0 * math.pi**2 * i * _K * t / _H**2) for i in moments)
            )
            s_rot = _R * (math.log(q_rot) + 1.5)
            h_thermal += 1.5 * rt

    # Vibration.
    zpe = 0.0
    s_vib = 0.0
    for nu in freqs:
        theta = _H * _C_CM * nu / _K
        zpe += 0.5 * _R * theta
        h_thermal += _R * theta / math.expm1(theta / t)
        s_h = _vib_entropy_harmonic(nu, t)
        if quasi_harmonic:
            w = qrrho_weight(nu)
            s_vib += w * s_h + (1.0 - w) * _vib_entropy_free_rotor(nu, t)
        else:
            s_vib += s_h

    s_el = _R * math.log(multiplicity)
    to_h = _J_MOL_TO_HARTREE
    return ThermoResult(
        temperature_k=t,
        zpe=zpe * to_h,
        thermal_enthalpy=h_thermal * to_h,
        entropy_translational=s_trans * to_h,
        entropy_rotational=s_rot * to_h,
        entropy_vibrational=s_vib * to_h,
        entropy_electronic=s_el * to_h,
        standard_state=standard_state_correction(t) if standard_state == "1M" else 0.0,
        linear=linear,
        symmetry_number=symmetry_number,
        quasi_harmonic=quasi_harmonic,
    )
