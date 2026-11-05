import math

import numpy as np
import pytest

from saddle import units
from saddle.analysis.conformers import boltzmann_ensemble

T = 298.15
RT = units.GAS_CONSTANT_J_MOL_K * T / (units.HARTREE_TO_J * units.AVOGADRO)


def test_two_degenerate_conformers() -> None:
    ens = boltzmann_ensemble(np.array([-100.0, -100.0]), T)
    np.testing.assert_allclose(ens.weights, [0.5, 0.5])
    assert ens.free_energy == pytest.approx(-100.0 - RT * math.log(2), abs=1e-12)
    assert ens.lowest_index == 0


def test_two_state_analytic() -> None:
    gap = units.kcal_mol_to_hartree(1.0)
    ens = boltzmann_ensemble(np.array([0.0, gap]), T)
    ratio = math.exp(-gap / RT)
    np.testing.assert_allclose(ens.weights, [1 / (1 + ratio), ratio / (1 + ratio)])
    assert ens.free_energy == pytest.approx(-RT * math.log(1 + ratio))


def test_ensemble_is_never_above_lowest_conformer() -> None:
    g = np.array([-50.0, -49.99, -49.9])
    assert boltzmann_ensemble(g, T).free_energy <= g.min()


def test_numerically_stable_for_large_absolute_energies() -> None:
    ens = boltzmann_ensemble(np.array([-1500.123456, -1500.123]), T)
    assert np.isfinite(ens.free_energy)
    assert ens.weights.sum() == pytest.approx(1.0)


def test_degeneracy_counts_like_duplicates() -> None:
    a = boltzmann_ensemble(np.array([0.0, 0.001, 0.001]), T)
    b = boltzmann_ensemble(np.array([0.0, 0.001]), T, degeneracy=np.array([1, 2]))
    assert a.free_energy == pytest.approx(b.free_energy)


def test_rejects_empty_and_bad_degeneracy() -> None:
    with pytest.raises(ValueError):
        boltzmann_ensemble(np.array([]), T)
    with pytest.raises(ValueError):
        boltzmann_ensemble(np.array([0.0]), T, degeneracy=np.array([0]))
