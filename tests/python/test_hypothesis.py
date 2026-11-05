from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from saddle import hypothesis as hyp
from saddle import units

DATA = Path(__file__).parent / "data"
EXAMPLE = DATA / "c3_f_etf.yaml"


def load_raw() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text())
    return raw


def parse(raw: dict[str, Any]) -> hyp.Hypothesis:
    return hyp.from_dict(raw, base_dir=DATA)


def test_spec_example_parses() -> None:
    h = hyp.load(EXAMPLE)
    assert h.id == "c3-f-ethyl-fluoride-gas"
    assert h.temperature_k == 298.15
    assert h.charge == -1
    assert h.multiplicity == 1
    assert [p.name for p in h.pathways] == ["sn2", "anti_e2"]
    assert h.pathways[1].breaking == ((2, 9), (3, 6))
    assert h.compare == ("sn2", "anti_e2")
    assert h.n_atoms == 9
    assert h.electron_count == 9 + 6 + 6 + 5 * 1 + 9 + 1


def test_coordinates_are_converted_to_bohr_at_boundary() -> None:
    h = hyp.load(EXAMPLE)
    etf = h.reactants[1]
    assert etf.symbols[1] == "C"
    np.testing.assert_allclose(etf.coords_bohr[1, 0], units.angstrom_to_bohr(1.52))


def test_atom_map_is_mandatory() -> None:
    raw = load_raw()
    del raw["reactants"][0]["atom_map"]
    with pytest.raises(hyp.HypothesisError, match="atom_map"):
        parse(raw)


def test_atom_map_length_must_match_xyz() -> None:
    raw = load_raw()
    raw["reactants"][1]["atom_map"] = [2, 3, 4]
    with pytest.raises(hyp.HypothesisError, match="atom_map"):
        parse(raw)


def test_atom_map_must_be_unique_and_contiguous() -> None:
    raw = load_raw()
    raw["reactants"][1]["atom_map"] = [1, 3, 4, 5, 6, 7, 8, 9]
    with pytest.raises(hyp.HypothesisError, match="unique"):
        parse(raw)
    raw["reactants"][1]["atom_map"] = [2, 3, 4, 5, 6, 7, 8, 12]
    with pytest.raises(hyp.HypothesisError, match="contiguous"):
        parse(raw)


def test_pathway_bonds_must_reference_mapped_atoms() -> None:
    raw = load_raw()
    raw["pathways"][0]["forming"] = [[1, 42]]
    with pytest.raises(hyp.HypothesisError, match="42"):
        parse(raw)


def test_pathway_needs_a_bond_change() -> None:
    raw = load_raw()
    raw["pathways"][0]["breaking"] = []
    raw["pathways"][0]["forming"] = []
    with pytest.raises(hyp.HypothesisError, match="bond change"):
        parse(raw)


def test_multiplicity_must_match_electron_parity() -> None:
    raw = load_raw()
    raw["multiplicity"] = 2
    with pytest.raises(hyp.HypothesisError, match="multiplicity"):
        parse(raw)


def test_fragment_charges_must_sum_to_total() -> None:
    raw = load_raw()
    raw["charge"] = 0
    with pytest.raises(hyp.HypothesisError, match="charge"):
        parse(raw)


def test_compare_must_name_declared_pathways() -> None:
    raw = load_raw()
    raw["compare"] = ["sn2", "syn_e2"]
    with pytest.raises(hyp.HypothesisError, match="syn_e2"):
        parse(raw)


def test_temperature_and_half_life_positive() -> None:
    raw = load_raw()
    raw["temperature_K"] = -1
    with pytest.raises(hyp.HypothesisError, match="temperature"):
        parse(raw)


def test_schema_version_is_required() -> None:
    raw = load_raw()
    raw["schema_version"] = 99
    with pytest.raises(hyp.HypothesisError, match="schema_version"):
        parse(raw)


def test_unknown_element_rejected(tmp_path: Path) -> None:
    (tmp_path / "x.xyz").write_text("1\n\nXx 0 0 0\n")
    raw = load_raw()
    raw["reactants"] = [{"name": "x", "xyz": "x.xyz", "atom_map": [1]}]
    with pytest.raises(hyp.HypothesisError, match="Xx"):
        hyp.from_dict(raw, base_dir=tmp_path)


def test_overlapping_atoms_rejected(tmp_path: Path) -> None:
    (tmp_path / "h2.xyz").write_text("2\n\nH 0 0 0\nH 0 0 0.05\n")
    raw = load_raw()
    raw["charge"] = 0
    raw["reactants"] = [{"name": "h2", "xyz": "h2.xyz", "atom_map": [1, 2]}]
    raw["pathways"] = [{"name": "p", "breaking": [[1, 2]], "forming": []}]
    raw["compare"] = ["p"]
    with pytest.raises(hyp.HypothesisError, match="close"):
        hyp.from_dict(raw, base_dir=tmp_path)
