import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from saddle.store.provenance import NOT_APPLICABLE, Provenance
from saddle.store.results import SCHEMA_VERSION, ResultError, read_result, write_result


def prov() -> Provenance:
    return Provenance(
        geometry_hash="cd" * 32,
        atom_order=("H", "H"),
        atom_map=(1, 2),
        charge=0,
        multiplicity=1,
        model="fake:morse",
        basis=NOT_APPLICABLE,
        auxiliary_basis=NOT_APPLICABLE,
        grid=NOT_APPLICABLE,
        functional=NOT_APPLICABLE,
        thresholds={"gtol_rms": 1e-5},
        math_mode=NOT_APPLICABLE,
        cuest_version=NOT_APPLICABLE,
        cuda_version=NOT_APPLICABLE,
        gpu_model=NOT_APPLICABLE,
        compute_capability=NOT_APPLICABLE,
        git_commit="0" * 40,
        saddle_version="0.0.1",
    )


ARRAYS = {
    "coords": np.arange(6, dtype=np.float64),
    "hessian": np.eye(6),
    "frequencies": np.array([-500.0, 1200.0]),
}
UNITS = {"coords": "bohr", "hessian": "Eh/bohr^2", "frequencies": "cm^-1"}
SUMMARY = {"verdict": "SUPPORTED", "barrier_kcal_mol": 18.2, "gates": {"G1": True}}


def test_round_trip(tmp_path: Path) -> None:
    path = write_result(tmp_path / "ts.h5", prov(), ARRAYS, UNITS, SUMMARY)
    res = read_result(path)
    assert res.provenance == prov()
    assert res.summary == SUMMARY
    for k, v in ARRAYS.items():
        np.testing.assert_array_equal(res.arrays[k], v)
    assert res.units == UNITS


def test_json_summary_is_readable_without_hdf5(tmp_path: Path) -> None:
    write_result(tmp_path / "ts.h5", prov(), ARRAYS, UNITS, SUMMARY)
    doc = json.loads((tmp_path / "ts.json").read_text())
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["summary"]["verdict"] == "SUPPORTED"
    assert doc["provenance"]["model"] == "fake:morse"
    assert doc["arrays"]["hessian"] == {"shape": [6, 6], "dtype": "float64", "units": "Eh/bohr^2"}
    assert doc["record_hash"] == prov().record_hash()


def test_existing_results_are_not_overwritten(tmp_path: Path) -> None:
    write_result(tmp_path / "ts.h5", prov(), ARRAYS, UNITS, SUMMARY)
    with pytest.raises(ResultError, match="exists"):
        write_result(tmp_path / "ts.h5", prov(), ARRAYS, UNITS, SUMMARY)
    write_result(tmp_path / "ts.h5", prov(), ARRAYS, UNITS, {"verdict": "X"}, overwrite=True)
    assert read_result(tmp_path / "ts.h5").summary == {"verdict": "X"}


def test_every_array_needs_units(tmp_path: Path) -> None:
    with pytest.raises(ResultError, match="units"):
        write_result(tmp_path / "ts.h5", prov(), ARRAYS, {"coords": "bohr"}, SUMMARY)


def test_summary_must_be_strict_json(tmp_path: Path) -> None:
    with pytest.raises(ResultError):
        write_result(tmp_path / "ts.h5", prov(), ARRAYS, UNITS, {"barrier": float("nan")})
    assert not (tmp_path / "ts.h5").exists()


def test_schema_version_checked_on_read(tmp_path: Path) -> None:
    path = write_result(tmp_path / "ts.h5", prov(), ARRAYS, UNITS, SUMMARY)
    with h5py.File(path, "r+") as f:
        f.attrs["schema_version"] = 99
    with pytest.raises(ResultError, match="schema"):
        read_result(path)
