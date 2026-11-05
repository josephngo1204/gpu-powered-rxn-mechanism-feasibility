"""Molecular graphs keyed by atom map, for gates G5 and G6 (spec §5, §7).

Bonds are detected from covalent radii (Cordero et al., Dalton Trans. 2008,
2832) scaled by BOND_SCALE. Atom maps are fixed through a search and IRC, so
graphs are compared index by index; no loose isomorphism is attempted,
because that could hide a wrong product.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from saddle import units
from saddle.hypothesis import Pathway

BOND_SCALE = 1.2

# Covalent radii in angstrom (Cordero 2008; sp3 carbon).
COVALENT_RADIUS_ANGSTROM: dict[str, float] = {
    "H": 0.31,
    "Li": 1.28,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Na": 1.66,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}

Graph = set[frozenset[int]]


class GraphError(ValueError):
    pass


def bond_graph(
    symbols: Sequence[str], coords_bohr: npt.ArrayLike, atom_map: Sequence[int]
) -> Graph:
    coords = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3) * units.BOHR_TO_ANGSTROM
    if not (len(symbols) == len(atom_map) == coords.shape[0]):
        raise GraphError("symbols, coordinates and atom map disagree on atom count")
    try:
        radii = np.array([COVALENT_RADIUS_ANGSTROM[s] for s in symbols])
    except KeyError as exc:
        raise GraphError(f"no covalent radius for element {exc.args[0]}") from exc
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    limit = BOND_SCALE * (radii[:, None] + radii[None, :])
    bonds: Graph = set()
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < limit[i, j]:
                bonds.add(frozenset((atom_map[i], atom_map[j])))
    return bonds


def expected_products(reactant: Graph, pathway: Pathway) -> Graph:
    product = set(reactant)
    for a, b in pathway.breaking:
        bond = frozenset((a, b))
        if bond not in product:
            raise GraphError(
                f"pathway '{pathway.name}' breaks bond {a}-{b}, not present in reactants"
            )
        product.remove(bond)
    for a, b in pathway.forming:
        bond = frozenset((a, b))
        if bond in product:
            raise GraphError(f"pathway '{pathway.name}' forms bond {a}-{b}, which already exists")
        product.add(bond)
    return product


@dataclass(frozen=True)
class GraphComparison:
    matches: bool
    missing: tuple[tuple[int, int], ...]  # expected but not observed
    extra: tuple[tuple[int, int], ...]  # observed but not expected


def _sorted(bonds: Graph) -> tuple[tuple[int, int], ...]:
    pairs = [tuple(sorted(b)) for b in bonds]
    return tuple(sorted((p[0], p[1]) for p in pairs))


def compare_graphs(observed: Graph, expected: Graph) -> GraphComparison:
    missing = _sorted(expected - observed)
    extra = _sorted(observed - expected)
    return GraphComparison(not missing and not extra, missing, extra)
