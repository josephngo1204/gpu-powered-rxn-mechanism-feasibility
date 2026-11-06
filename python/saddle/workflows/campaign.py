"""Campaign input: the YAML that pairs a hypothesis with a model and TS guesses.

A campaign file names the hypothesis (species, atom maps, pathways), the level
of theory, one TS guess per pathway, symmetry numbers, stated uncertainty
terms and the robustness variants. ``load_campaign`` validates it and returns
an immutable ``Campaign``; nothing here evaluates a model.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from saddle import elements
from saddle.analysis.uncertainty import Term
from saddle.hypothesis import Hypothesis, HypothesisError, load, read_xyz
from saddle.workflows.models import SystemInfo

FloatArray = npt.NDArray[np.float64]

CAMPAIGN_SCHEMA_VERSION = 1
MEASURED_COMPONENTS = ("numerical", "thermo")


class CampaignError(ValueError):
    pass


@dataclass(frozen=True)
class Atoms:
    """A subset of the system (the whole system or one reactant species)."""

    symbols: tuple[str, ...]
    atom_map: tuple[int, ...]
    multiplicity: int

    @property
    def masses(self) -> FloatArray:
        return np.array([elements.isotope_mass(s) for s in self.symbols])

    @property
    def atomic_numbers(self) -> list[int]:
        return [elements.atomic_number(s) for s in self.symbols]


@dataclass(frozen=True)
class Campaign:
    path: Path
    hypothesis: Hypothesis
    model_spec: Mapping[str, Any]
    ts_guesses: Mapping[str, FloatArray]
    symmetry_reactant: int
    symmetry_ts: int
    symmetry_by_name: Mapping[str, int]
    quasi_harmonic: bool
    uncertainty: Mapping[str, Term]
    robustness: Mapping[str, Mapping[str, Any]]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(s for sp in self.hypothesis.reactants for s in sp.symbols)

    @property
    def atom_map(self) -> tuple[int, ...]:
        return tuple(i for sp in self.hypothesis.reactants for i in sp.atom_map)

    @property
    def system(self) -> Atoms:
        """The whole reacting system, at the hypothesis multiplicity."""
        return Atoms(self.symbols, self.atom_map, self.hypothesis.multiplicity)

    @property
    def masses(self) -> FloatArray:
        return self.system.masses

    @property
    def atomic_numbers(self) -> list[int]:
        return self.system.atomic_numbers

    @property
    def reactant_coords(self) -> FloatArray:
        return np.concatenate([sp.coords_bohr.ravel() for sp in self.hypothesis.reactants])

    @property
    def symmetry_species(self) -> tuple[int, ...]:
        return tuple(
            self.symmetry_by_name.get(sp.name, self.symmetry_reactant)
            for sp in self.hypothesis.reactants
        )

    def species(self, index: int) -> Atoms:
        """One reactant species; closed shell unless its electron count is odd."""
        sp = self.hypothesis.reactants[index]
        electrons = sum(elements.atomic_number(x) for x in sp.symbols) - sp.charge
        return Atoms(tuple(sp.symbols), tuple(sp.atom_map), 1 if electrons % 2 == 0 else 2)

    def species_slice(self, index: int) -> slice:
        """Atom range of one species within the concatenated reactant atoms."""
        start = sum(len(sp.symbols) for sp in self.hypothesis.reactants[:index])
        return slice(start, start + len(self.hypothesis.reactants[index].symbols))

    def species_coords(self, coords: FloatArray, index: int) -> FloatArray:
        """The flat (3N) coordinates of one species cut out of ``coords``."""
        atoms = self.species_slice(index)
        return coords[3 * atoms.start : 3 * atoms.stop]

    def system_info(self) -> SystemInfo:
        return SystemInfo(
            symbols=self.symbols,
            species_symbols=tuple(tuple(sp.symbols) for sp in self.hypothesis.reactants),
            species_charges=tuple(sp.charge for sp in self.hypothesis.reactants),
            charge=self.hypothesis.charge,
            solvent=self.hypothesis.solvent,
        )


def _read_guess(path: Path, symbols: tuple[str, ...]) -> FloatArray:
    """A TS guess XYZ file as flat bohr coordinates, in the hypothesis atom order."""
    try:
        guess_symbols, coords = read_xyz(path, "TS guess")
    except HypothesisError as exc:
        raise CampaignError(str(exc)) from exc
    if len(guess_symbols) != len(symbols):
        raise CampaignError(
            f"{path}: TS guess has {len(guess_symbols)} atoms, hypothesis has {len(symbols)}"
        )
    if guess_symbols != symbols:
        raise CampaignError(
            f"{path}: TS guess atom order {list(guess_symbols)} does not match the "
            f"hypothesis atom order {list(symbols)}"
        )
    return coords.ravel()


def _stated_uncertainty(raw: Mapping[str, Any]) -> dict[str, Term]:
    terms: dict[str, Term] = {}
    for name, t in raw.items():
        if name in MEASURED_COMPONENTS:
            raise CampaignError(f"uncertainty '{name}' is measured by the pipeline; do not set it")
        value = t.get("value")
        terms[name] = Term(
            name, None if value is None else float(value), t["source"], t.get("note", "")
        )
    return terms


def load_campaign(path: str | Path) -> Campaign:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if raw.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise CampaignError(f"campaign schema_version must be {CAMPAIGN_SCHEMA_VERSION}")
    base = path.parent
    hyp = load(base / raw["hypothesis"])
    if len(hyp.reactants) != 1 and str(raw["model"].get("type", "")).startswith("fake:"):
        raise CampaignError(
            "fake-surface campaigns support one reactant species; bimolecular reactants need "
            "a model that evaluates separated fragments (type: cuest)"
        )
    names = {p.name for p in hyp.pathways}
    guesses_raw = raw.get("pathways", {})
    if set(guesses_raw) != names:
        raise CampaignError(
            f"campaign pathways {sorted(guesses_raw)} != hypothesis {sorted(names)}"
        )
    hyp_symbols = tuple(s for sp in hyp.reactants for s in sp.symbols)
    guesses = {n: _read_guess(base / g["ts_guess"], hyp_symbols) for n, g in guesses_raw.items()}
    sym = raw.get("symmetry_numbers", {})
    return Campaign(
        path=path,
        hypothesis=hyp,
        model_spec=raw["model"],
        ts_guesses=guesses,
        symmetry_reactant=int(sym.get("reactant", 1)),
        symmetry_ts=int(sym.get("ts", 1)),
        symmetry_by_name={str(k): int(v) for k, v in sym.items() if k not in ("reactant", "ts")},
        quasi_harmonic=bool(raw.get("quasi_harmonic", True)),
        uncertainty=_stated_uncertainty(raw.get("uncertainty") or {}),
        robustness=raw.get("robustness") or {},
    )
