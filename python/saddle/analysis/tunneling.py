"""Tunneling corrections, reported separately and never folded in silently (spec §4).

Wigner: kappa = 1 + (h nu / kT)^2 / 24.

Eckart: transmission through an unsymmetric Eckart barrier fitted to the
imaginary frequency and the forward and reverse barrier heights (ZPE-corrected),
Johnston and Heicklen, J. Phys. Chem. 66, 532 (1962). The thermal average is
taken from the higher of the two wells, so kappa is the same in both
directions.
"""

import math
from dataclasses import dataclass

from scipy import integrate

from saddle import units

_KB_HARTREE = units.BOLTZMANN_J_K / units.HARTREE_TO_J
_INTEGRATION_SPAN_KT = 60.0


def _h_nu_hartree(frequency_cm: float) -> float:
    nu = abs(frequency_cm)
    if not nu > 0.0:
        raise ValueError("imaginary frequency magnitude must be positive")
    return nu / units.HARTREE_TO_WAVENUMBER


def wigner(frequency_cm: float, temperature_k: float) -> float:
    if not temperature_k > 0.0:
        raise ValueError("temperature must be positive")
    u = _h_nu_hartree(frequency_cm) / (_KB_HARTREE * temperature_k)
    return 1.0 + u * u / 24.0


def _ordered(v_forward: float, v_reverse: float) -> tuple[float, float]:
    if not (v_forward > 0.0 and v_reverse > 0.0):
        raise ValueError("Eckart needs positive forward and reverse barriers")
    return min(v_forward, v_reverse), max(v_forward, v_reverse)


def _log_eckart_transmission(
    energy: float, frequency_cm: float, v_forward: float, v_reverse: float
) -> float:
    """log P(E), computed in log space so deep-tunneling energies never overflow."""
    v1, v2 = _ordered(v_forward, v_reverse)
    if energy <= 0.0:
        return -math.inf
    h_nu = _h_nu_hartree(frequency_cm)
    a1 = 2.0 * math.pi * v1 / h_nu
    a2 = 2.0 * math.pi * v2 / h_nu
    xi = energy / v1
    denom = 1.0 / math.sqrt(a1) + 1.0 / math.sqrt(a2)
    big_a = 2.0 * math.sqrt(a1 * xi) / denom
    big_b = 2.0 * math.sqrt((xi - 1.0) * a1 + a2) / denom
    s = big_a + big_b
    # P = 2 sinh A sinh B / (cosh(A+B) + cosh D); divide through by e^(A+B)/2.
    log_num = math.log(-math.expm1(-2.0 * big_a)) + math.log(-math.expm1(-2.0 * big_b))
    d_sq = a1 * a2 - math.pi**2 / 4.0
    if d_sq >= 0.0:
        big_d = 2.0 * math.sqrt(d_sq)
        exponents = [0.0, -2.0 * s, big_d - s, -big_d - s]
        top = max(exponents)
        log_den = top + math.log(sum(math.exp(x - top) for x in exponents))
    else:
        log_den = math.log(
            1.0 + math.exp(-2.0 * s) + 2.0 * math.cos(2.0 * math.sqrt(-d_sq)) * math.exp(-s)
        )
    return log_num - log_den


def eckart_transmission(
    energy: float, frequency_cm: float, v_forward: float, v_reverse: float
) -> float:
    """Transmission probability at `energy` (Eh) above the higher well."""
    return math.exp(_log_eckart_transmission(energy, frequency_cm, v_forward, v_reverse))


def eckart(frequency_cm: float, v_forward: float, v_reverse: float, temperature_k: float) -> float:
    """Thermally averaged Eckart transmission coefficient kappa(T)."""
    if not temperature_k > 0.0:
        raise ValueError("temperature must be positive")
    v1, _ = _ordered(v_forward, v_reverse)
    _h_nu_hartree(frequency_cm)
    kt = _KB_HARTREE * temperature_k

    def integrand(e: float) -> float:
        log_p = _log_eckart_transmission(e, frequency_cm, v_forward, v_reverse)
        return math.exp(log_p + (v1 - e) / kt)

    upper = v1 + _INTEGRATION_SPAN_KT * kt
    value, error = integrate.quad(integrand, 0.0, upper, points=[v1], limit=500, epsrel=1e-10)
    if not math.isfinite(value) or error > 1e-6 * max(value, 1.0):
        raise ArithmeticError(f"Eckart integration did not converge (value={value}, err={error})")
    return float(value) / kt


@dataclass(frozen=True)
class TunnelingCorrections:
    frequency_cm: float
    temperature_k: float
    wigner: float
    eckart: float

    @property
    def wigner_delta_g_kcal_mol(self) -> float:
        return _delta_g(self.wigner, self.temperature_k)

    @property
    def eckart_delta_g_kcal_mol(self) -> float:
        return _delta_g(self.eckart, self.temperature_k)


def _delta_g(kappa: float, temperature_k: float) -> float:
    rt = units.j_mol_to_kcal_mol(units.GAS_CONSTANT_J_MOL_K * temperature_k)
    return -rt * math.log(kappa)


def corrections(
    frequency_cm: float, v_forward: float, v_reverse: float, temperature_k: float
) -> TunnelingCorrections:
    """Both corrections, side by side, as the report shows them."""
    return TunnelingCorrections(
        frequency_cm=frequency_cm,
        temperature_k=temperature_k,
        wigner=wigner(frequency_cm, temperature_k),
        eckart=eckart(frequency_cm, v_forward, v_reverse, temperature_k),
    )
