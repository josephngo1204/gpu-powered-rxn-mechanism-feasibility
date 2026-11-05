import numpy as np
import pytest

from saddle import units
from saddle.graphs import (
    GraphError,
    bond_graph,
    compare_graphs,
    expected_products,
)
from saddle.hypothesis import Pathway

# Atom maps: 1 = Cl (nucleophile), 2 = C, 3-5 = H, 6 = Cl (leaving group).
SYMBOLS = ("Cl", "C", "H", "H", "H", "Cl")
MAP = (1, 2, 3, 4, 5, 6)


def _geom(c_cl_in: float, c_cl_out: float, h_x: float) -> np.ndarray:
    """Linear Cl...C...Cl along x with an umbrella of H atoms (angstrom -> bohr)."""
    xyz = [
        [-c_cl_in, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [h_x, 1.03, 0.0],
        [h_x, -0.515, 0.892],
        [h_x, -0.515, -0.892],
        [c_cl_out, 0.0, 0.0],
    ]
    return np.asarray(xyz).ravel() / units.BOHR_TO_ANGSTROM


REACTANT_COMPLEX = _geom(3.2, 1.80, -0.36)  # Cl- ... CH3Cl
PRODUCT_COMPLEX = _geom(1.80, 3.2, 0.36)  # ClCH3 ... Cl-
SN2 = Pathway("sn2", breaking=((2, 6),), forming=((1, 2),))


def test_bond_graph_of_ion_dipole_complex() -> None:
    g = bond_graph(SYMBOLS, REACTANT_COMPLEX, MAP)
    assert g == {frozenset(b) for b in [(2, 3), (2, 4), (2, 5), (2, 6)]}


def test_expected_products_apply_bond_changes() -> None:
    r = bond_graph(SYMBOLS, REACTANT_COMPLEX, MAP)
    p = expected_products(r, SN2)
    assert p == {frozenset(b) for b in [(2, 3), (2, 4), (2, 5), (1, 2)]}


def test_matching_product_endpoint() -> None:
    r = bond_graph(SYMBOLS, REACTANT_COMPLEX, MAP)
    observed = bond_graph(SYMBOLS, PRODUCT_COMPLEX, MAP)
    result = compare_graphs(observed, expected_products(r, SN2))
    assert result.matches
    assert result.missing == () and result.extra == ()


def test_wrong_endpoint_reports_missing_and_extra_bonds() -> None:
    r = bond_graph(SYMBOLS, REACTANT_COMPLEX, MAP)
    result = compare_graphs(r, expected_products(r, SN2))
    assert not result.matches
    assert result.missing == ((1, 2),)
    assert result.extra == ((2, 6),)


def test_breaking_a_nonexistent_bond_is_an_error() -> None:
    r = bond_graph(SYMBOLS, REACTANT_COMPLEX, MAP)
    with pytest.raises(GraphError, match="not present"):
        expected_products(r, Pathway("bad", breaking=((1, 3),), forming=()))


def test_forming_an_existing_bond_is_an_error() -> None:
    r = bond_graph(SYMBOLS, REACTANT_COMPLEX, MAP)
    with pytest.raises(GraphError, match="already"):
        expected_products(r, Pathway("bad", breaking=(), forming=((2, 3),)))


def test_unknown_radius_and_shape_errors() -> None:
    with pytest.raises(GraphError):
        bond_graph(("Xe",), np.zeros(3), (1,))
    with pytest.raises(GraphError):
        bond_graph(SYMBOLS, REACTANT_COMPLEX[:-3], MAP)
