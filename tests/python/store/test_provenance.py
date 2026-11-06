import dataclasses
import re

import pytest

from saddle.store.provenance import (
    NOT_APPLICABLE,
    SPEC_FIELDS,
    Provenance,
    ProvenanceError,
    collect_environment,
)


def record(**override: object) -> Provenance:
    fields: dict[str, object] = dict(
        geometry_hash="ab" * 32,
        atom_order=("C", "H", "H", "H", "Cl"),
        atom_map=(2, 3, 4, 5, 6),
        charge=0,
        multiplicity=1,
        model="cuest",
        basis="def2-SVPD",
        auxiliary_basis="def2-universal-jkfit",
        grid="(99,590)",
        functional="wB97X-V",
        thresholds={"thresholdPQ": 1e-14, "orbital_gradient": 1e-8},
        math_mode="native-fp64",
        cuest_version="0.2.2",
        cuda_version="12.8",
        gpu_model="NVIDIA H200 NVL MIG 1g.18gb",
        compute_capability="9.0",
        git_commit="0" * 40,
        saddle_version="0.0.1",
    )
    fields.update(override)
    return Provenance(**fields)  # type: ignore[arg-type]


def test_record_covers_every_provenance_field() -> None:
    names = {f.name for f in dataclasses.fields(Provenance)}
    assert set(SPEC_FIELDS) <= names
    for required in (
        "geometry_hash",
        "atom_order",
        "charge",
        "multiplicity",
        "basis",
        "grid",
        "functional",
        "thresholds",
        "math_mode",
        "cuest_version",
        "cuda_version",
        "gpu_model",
        "compute_capability",
        "git_commit",
    ):
        assert required in SPEC_FIELDS


def test_round_trip() -> None:
    r = record()
    assert Provenance.from_dict(r.to_dict()) == r


@pytest.mark.parametrize("field", ["basis", "cuest_version", "git_commit"])
def test_blank_fields_are_rejected(field: str) -> None:
    with pytest.raises(ProvenanceError, match=field):
        record(**{field: ""})


def test_not_applicable_must_be_explicit() -> None:
    r = record(model="fake:muller-brown", basis=NOT_APPLICABLE, cuest_version=NOT_APPLICABLE)
    assert r.to_dict()["basis"] == "n/a"


def test_atom_order_and_map_lengths_must_match() -> None:
    with pytest.raises(ProvenanceError, match="atom"):
        record(atom_map=(1, 2))


def test_settings_fingerprint_ignores_environment_and_key_order() -> None:
    a = record(thresholds={"thresholdPQ": 1e-14, "orbital_gradient": 1e-8})
    b = record(thresholds={"orbital_gradient": 1e-8, "thresholdPQ": 1e-14}, git_commit="1" * 40)
    assert a.settings_fingerprint() == b.settings_fingerprint()
    assert a.record_hash() != b.record_hash()


def test_settings_fingerprint_tracks_numerics() -> None:
    base = record().settings_fingerprint()
    assert record(math_mode="default").settings_fingerprint() != base
    assert record(grid="(75,302)").settings_fingerprint() != base
    assert record(cuest_version="0.2.3").settings_fingerprint() != base


def test_collect_environment_reports_this_checkout() -> None:
    env = collect_environment()
    assert re.fullmatch(r"[0-9a-f]{40}", env["git_commit"])
    assert env["git_dirty"] in ("true", "false")
    assert env["saddle_version"] == "0.0.1"
    for key in ("cuest_version", "cuda_version", "gpu_model", "compute_capability"):
        assert env[key]
