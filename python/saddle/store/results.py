"""HDF5 results with a JSON summary (spec §7 "Data rules").

<name>.h5   provenance and summary as JSON attributes; arrays under /data,
            each with a 'units' attribute
<name>.json provenance, summary and an array manifest (no array data)

Writes go to a temporary file and are renamed into place. Existing results
are never overwritten unless asked, so failed runs are not hidden.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from saddle.store.provenance import Provenance

SCHEMA_VERSION = 1


class ResultError(ValueError):
    pass


@dataclass(frozen=True)
class StoredResult:
    provenance: Provenance
    summary: dict[str, Any]
    arrays: dict[str, npt.NDArray[Any]]
    units: dict[str, str]


def _strict_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResultError(f"summary is not strict JSON: {exc}") from exc


def write_result(
    path: str | Path,
    provenance: Provenance,
    arrays: Mapping[str, npt.ArrayLike],
    units: Mapping[str, str],
    summary: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    path = Path(path)
    json_path = path.with_suffix(".json")
    if not overwrite and (path.exists() or json_path.exists()):
        raise ResultError(f"result {path} exists; refusing to overwrite")
    missing = sorted(set(arrays) - set(units))
    if missing:
        raise ResultError(f"arrays without units: {', '.join(missing)}")
    summary_text = _strict_json(dict(summary))
    prov_text = _strict_json(provenance.to_dict())

    manifest: dict[str, Any] = {}
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["provenance"] = prov_text
            f.attrs["summary"] = summary_text
            group = f.create_group("data")
            for name, value in arrays.items():
                arr = np.asarray(value)
                ds = group.create_dataset(name, data=arr)
                ds.attrs["units"] = units[name]
                manifest[name] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "units": units[name],
                }
        doc = {
            "schema_version": SCHEMA_VERSION,
            "record_hash": provenance.record_hash(),
            "provenance": provenance.to_dict(),
            "summary": dict(summary),
            "arrays": manifest,
        }
        json_tmp = json_path.with_suffix(".json.tmp")
        json_tmp.write_text(json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n")
        os.replace(tmp, path)
        os.replace(json_tmp, json_path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_result(path: str | Path) -> StoredResult:
    path = Path(path)
    with h5py.File(path, "r") as f:
        version = int(f.attrs["schema_version"])
        if version != SCHEMA_VERSION:
            raise ResultError(f"unsupported result schema {version}, expected {SCHEMA_VERSION}")
        provenance = Provenance.from_dict(json.loads(str(f.attrs["provenance"])))
        summary = json.loads(str(f.attrs["summary"]))
        arrays: dict[str, npt.NDArray[Any]] = {}
        units: dict[str, str] = {}
        for name, ds in f["data"].items():
            arrays[name] = np.asarray(ds[()])
            units[name] = str(ds.attrs["units"])
    return StoredResult(provenance, summary, arrays, units)
