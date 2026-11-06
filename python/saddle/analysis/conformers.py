"""Conformer ensembles: Boltzmann weights and ensemble free energy."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from saddle import units

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Ensemble:
    weights: FloatArray
    free_energy: float
    lowest_index: int


def boltzmann_ensemble(
    free_energies_hartree: npt.ArrayLike,
    temperature_k: float,
    degeneracy: npt.ArrayLike | None = None,
) -> Ensemble:
    """Weights and ensemble free energy of distinct conformers.

    Duplicate conformers must be removed (or counted with `degeneracy`),
    otherwise they are double counted.
    """
    g = np.asarray(free_energies_hartree, dtype=np.float64)
    if g.size == 0:
        raise ValueError("conformer ensemble is empty")
    if not temperature_k > 0.0:
        raise ValueError("temperature must be positive")
    d = np.ones_like(g) if degeneracy is None else np.asarray(degeneracy, dtype=np.float64)
    if d.shape != g.shape or np.any(d < 1):
        raise ValueError("degeneracy must be >= 1 and match the energies")
    rt = units.GAS_CONSTANT_J_MOL_K * temperature_k / (units.HARTREE_TO_J * units.AVOGADRO)
    g_min = float(g.min())
    terms = d * np.exp(-(g - g_min) / rt)
    total = float(terms.sum())
    return Ensemble(terms / total, g_min - rt * float(np.log(total)), int(np.argmin(g)))
