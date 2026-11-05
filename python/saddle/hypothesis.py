"""Mechanism hypothesis input: versioned YAML, validated at the boundary (spec §7, §12).

Coordinates are read in angstrom and stored in bohr. Atom maps are mandatory
because gates G5 and G6 depend on them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from saddle import elements, units

SCHEMA_VERSION = 1
MIN_INTERATOMIC_DISTANCE_ANGSTROM = 0.4

Bond = tuple[int, int]


class HypothesisError(ValueError):
    """Raised when a hypothesis file is invalid."""


@dataclass(frozen=True)
class Species:
    name: str
    symbols: tuple[str, ...]
    coords_bohr: npt.NDArray[np.float64]  # shape (n_atoms, 3)
    atom_map: tuple[int, ...]
    charge: int


@dataclass(frozen=True)
class Pathway:
    name: str
    breaking: tuple[Bond, ...]
    forming: tuple[Bond, ...]


@dataclass(frozen=True)
class Hypothesis:
    id: str
    temperature_k: float
    target_half_life_s: float
    solvent: str | None
    charge: int
    multiplicity: int
    reactants: tuple[Species, ...]
    pathways: tuple[Pathway, ...]
    compare: tuple[str, ...]

    @property
    def n_atoms(self) -> int:
        return sum(len(s.symbols) for s in self.reactants)

    @property
    def electron_count(self) -> int:
        nuclear = sum(elements.atomic_number(sym) for s in self.reactants for sym in s.symbols)
        return nuclear - self.charge


def load(path: str | Path) -> Hypothesis:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        raise HypothesisError(f"{path}: top level must be a mapping")
    return from_dict(raw, base_dir=path.parent)


def _require(raw: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in raw:
        raise HypothesisError(f"{where}: missing required field '{key}'")
    return raw[key]


def _read_xyz(path: Path, where: str) -> tuple[tuple[str, ...], npt.NDArray[np.float64]]:
    try:
        lines = path.read_text().splitlines()
        n = int(lines[0].split()[0])
        body = [ln.split() for ln in lines[2 : 2 + n]]
    except (OSError, ValueError, IndexError) as exc:
        raise HypothesisError(f"{where}: cannot read xyz file {path}: {exc}") from exc
    if len(body) != n or any(len(f) < 4 for f in body):
        raise HypothesisError(f"{where}: {path} declares {n} atoms but has malformed lines")
    symbols: list[str] = []
    for fields in body:
        sym = fields[0].capitalize()
        if sym not in elements.ATOMIC_NUMBER:
            raise HypothesisError(f"{where}: unknown element '{fields[0]}' in {path}")
        symbols.append(sym)
    coords = np.array([[float(x) for x in f[1:4]] for f in body], dtype=np.float64)
    return tuple(symbols), coords / units.BOHR_TO_ANGSTROM


def _check_geometry(name: str, coords_bohr: npt.NDArray[np.float64]) -> None:
    limit = MIN_INTERATOMIC_DISTANCE_ANGSTROM / units.BOHR_TO_ANGSTROM
    diff = coords_bohr[:, None, :] - coords_bohr[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    if dist.size and float(dist.min()) < limit:
        i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
        raise HypothesisError(
            f"species '{name}': atoms {i + 1} and {j + 1} are too close "
            f"({float(dist[i, j]) * units.BOHR_TO_ANGSTROM:.3f} Å)"
        )


def _parse_species(raw: Mapping[str, Any], base_dir: Path) -> Species:
    name = str(_require(raw, "name", "reactant"))
    where = f"reactant '{name}'"
    symbols, coords = _read_xyz(base_dir / str(_require(raw, "xyz", where)), where)
    if "atom_map" not in raw:
        raise HypothesisError(f"{where}: atom_map is mandatory")
    atom_map = tuple(int(i) for i in raw["atom_map"])
    if len(atom_map) != len(symbols):
        raise HypothesisError(
            f"{where}: atom_map has {len(atom_map)} entries but xyz has {len(symbols)} atoms"
        )
    _check_geometry(name, coords)
    return Species(name, symbols, coords, atom_map, int(raw.get("charge", 0)))


def _parse_bonds(raw: Sequence[Sequence[int]], where: str, valid: set[int]) -> tuple[Bond, ...]:
    bonds: list[Bond] = []
    for pair in raw:
        if len(pair) != 2 or pair[0] == pair[1]:
            raise HypothesisError(f"{where}: bond {list(pair)} must join two distinct atoms")
        a, b = int(pair[0]), int(pair[1])
        for atom in (a, b):
            if atom not in valid:
                raise HypothesisError(f"{where}: bond refers to unmapped atom {atom}")
        bonds.append((a, b))
    return tuple(bonds)


def from_dict(raw: Mapping[str, Any], base_dir: str | Path) -> Hypothesis:
    base_dir = Path(base_dir)
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise HypothesisError(f"schema_version must be {SCHEMA_VERSION}, got {version!r}")

    hid = str(_require(raw, "id", "hypothesis"))
    temperature = float(_require(raw, "temperature_K", hid))
    if not temperature > 0.0:
        raise HypothesisError(f"{hid}: temperature_K must be positive")
    half_life = float(_require(raw, "target_half_life_s", hid))
    if not half_life > 0.0:
        raise HypothesisError(f"{hid}: target_half_life_s must be positive")
    charge = int(_require(raw, "charge", hid))
    multiplicity = int(_require(raw, "multiplicity", hid))
    if multiplicity < 1:
        raise HypothesisError(f"{hid}: multiplicity must be at least 1")
    solvent = raw.get("solvent")

    reactants = tuple(_parse_species(r, base_dir) for r in _require(raw, "reactants", hid))
    if not reactants:
        raise HypothesisError(f"{hid}: at least one reactant is required")

    all_maps = [i for s in reactants for i in s.atom_map]
    if len(set(all_maps)) != len(all_maps):
        raise HypothesisError(f"{hid}: atom_map indices must be unique across reactants")
    if sorted(all_maps) != list(range(1, len(all_maps) + 1)):
        raise HypothesisError(f"{hid}: atom_map indices must be contiguous from 1")

    fragment_charge = sum(s.charge for s in reactants)
    if fragment_charge != charge:
        raise HypothesisError(
            f"{hid}: fragment charges sum to {fragment_charge} but total charge is {charge}"
        )

    valid = set(all_maps)
    pathways: list[Pathway] = []
    for p in _require(raw, "pathways", hid):
        pname = str(_require(p, "name", "pathway"))
        where = f"pathway '{pname}'"
        breaking = _parse_bonds(p.get("breaking", []), where, valid)
        forming = _parse_bonds(p.get("forming", []), where, valid)
        if not breaking and not forming:
            raise HypothesisError(f"{where}: needs at least one bond change")
        pathways.append(Pathway(pname, breaking, forming))
    names = [p.name for p in pathways]
    if len(set(names)) != len(names):
        raise HypothesisError(f"{hid}: pathway names must be unique")

    compare = tuple(str(c) for c in raw.get("compare", names))
    for c in compare:
        if c not in names:
            raise HypothesisError(f"{hid}: compare names undeclared pathway '{c}'")

    h = Hypothesis(
        id=hid,
        temperature_k=temperature,
        target_half_life_s=half_life,
        solvent=None if solvent is None else str(solvent),
        charge=charge,
        multiplicity=multiplicity,
        reactants=reactants,
        pathways=tuple(pathways),
        compare=compare,
    )
    electrons = h.electron_count
    if electrons < 0:
        raise HypothesisError(f"{hid}: charge {charge} leaves a negative electron count")
    unpaired = multiplicity - 1
    if unpaired > electrons or (electrons - unpaired) % 2 != 0:
        raise HypothesisError(
            f"{hid}: multiplicity {multiplicity} is inconsistent with {electrons} electrons"
        )
    return h
