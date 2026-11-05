"""Transition-state-theory kinetics and the feasibility limit (spec §1).

Energies here are free energies in kcal/mol, which are a reporting boundary;
the electronic-structure core stays in hartree.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from saddle import units


def _rt_kcal_mol(temperature_k: float) -> float:
    if not temperature_k > 0.0:
        raise ValueError(f"temperature must be positive, got {temperature_k}")
    return units.j_mol_to_kcal_mol(units.GAS_CONSTANT_J_MOL_K * temperature_k)


def eyring_rate(barrier_kcal_mol: float, temperature_k: float) -> float:
    """k = (kB*T/h) * exp(-dG‡/RT), in s^-1 (transmission coefficient 1)."""
    rt = _rt_kcal_mol(temperature_k)
    prefactor = units.BOLTZMANN_J_K * temperature_k / units.PLANCK_J_S
    return prefactor * math.exp(-barrier_kcal_mol / rt)


def half_life_s(barrier_kcal_mol: float, temperature_k: float) -> float:
    """First-order half-life ln2 / k."""
    return math.log(2.0) / eyring_rate(barrier_kcal_mol, temperature_k)


def barrier_limit_kcal_mol(temperature_k: float, half_life_target_s: float) -> float:
    """dG‡_limit = RT * ln((kB*T/h) * t½ / ln 2)."""
    if not half_life_target_s > 0.0:
        raise ValueError(f"half-life must be positive, got {half_life_target_s}")
    rt = _rt_kcal_mol(temperature_k)
    prefactor = units.BOLTZMANN_J_K * temperature_k / units.PLANCK_J_S
    return rt * math.log(prefactor * half_life_target_s / math.log(2.0))


def rate_ratio(delta_delta_g_kcal_mol: float, temperature_k: float) -> float:
    """Rate (or selectivity) ratio implied by a free-energy gap: exp(ddG/RT)."""
    return math.exp(delta_delta_g_kcal_mol / _rt_kcal_mol(temperature_k))


@dataclass(frozen=True)
class State:
    """One stationary point along a pathway, in pathway order."""

    name: str
    free_energy_kcal_mol: float
    is_ts: bool


@dataclass(frozen=True)
class EnergeticSpan:
    value: float
    tdts: str
    tdi: str


def energetic_span(states: Sequence[State], reaction_energy: float | None = None) -> EnergeticSpan:
    """Energetic span of a multistep pathway (Kozuch and Shaik).

    With ``reaction_energy=None`` the pathway is linear: only TS/intermediate
    pairs where the TS follows the intermediate count. With a reaction energy
    the pathway is a catalytic cycle, and pairs where the TS precedes the
    intermediate add the reaction energy.
    """
    best: EnergeticSpan | None = None
    for i, ts in enumerate(states):
        if not ts.is_ts:
            continue
        for j, inter in enumerate(states):
            if inter.is_ts:
                continue
            span = ts.free_energy_kcal_mol - inter.free_energy_kcal_mol
            if i < j:
                if reaction_energy is None:
                    continue
                span += reaction_energy
            if best is None or span > best.value:
                best = EnergeticSpan(span, ts.name, inter.name)
    if best is None:
        raise ValueError("energetic span needs at least one TS after an intermediate")
    return best
