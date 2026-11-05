"""Result provenance (spec §7 "Data rules").

Every result stores geometry hash, atom order, charge, spin, basis data,
grids, functional, thresholds, math mode, cuEST version, CUDA version, GPU
model, compute capability and git commit. A field that does not apply must
say so explicitly with NOT_APPLICABLE; blanks are rejected.
"""

import dataclasses
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import saddle

NOT_APPLICABLE = "n/a"
UNKNOWN = "unknown"

SPEC_FIELDS = (
    "geometry_hash",
    "atom_order",
    "atom_map",
    "charge",
    "multiplicity",
    "basis",
    "auxiliary_basis",
    "grid",
    "functional",
    "thresholds",
    "math_mode",
    "cuest_version",
    "cuda_version",
    "gpu_model",
    "compute_capability",
    "git_commit",
)

# Fields that change a computed gradient; they key cached stencil points.
SETTINGS_FIELDS = (
    "model",
    "basis",
    "auxiliary_basis",
    "grid",
    "functional",
    "thresholds",
    "math_mode",
    "cuest_version",
    "charge",
    "multiplicity",
)


class ProvenanceError(ValueError):
    pass


@dataclass(frozen=True)
class Provenance:
    geometry_hash: str
    atom_order: tuple[str, ...]
    atom_map: tuple[int, ...]
    charge: int
    multiplicity: int
    model: str
    basis: str
    auxiliary_basis: str
    grid: str
    functional: str
    thresholds: Mapping[str, float]
    math_mode: str
    cuest_version: str
    cuda_version: str
    gpu_model: str
    compute_capability: str
    git_commit: str
    saddle_version: str
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, str) and not value.strip():
                raise ProvenanceError(
                    f"provenance field '{f.name}' is blank; use '{NOT_APPLICABLE}'"
                )
        if len(self.atom_order) != len(self.atom_map):
            raise ProvenanceError("atom_order and atom_map lengths differ")
        if self.multiplicity < 1:
            raise ProvenanceError("multiplicity must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["atom_order"] = list(self.atom_order)
        d["atom_map"] = list(self.atom_map)
        d["thresholds"] = dict(self.thresholds)
        d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Provenance":
        kw = dict(d)
        kw["atom_order"] = tuple(kw["atom_order"])
        kw["atom_map"] = tuple(int(i) for i in kw["atom_map"])
        kw["thresholds"] = {str(k): float(v) for k, v in kw["thresholds"].items()}
        kw["extra"] = dict(kw.get("extra", {}))
        return cls(**kw)

    def settings_fingerprint(self) -> str:
        d = self.to_dict()
        return _digest({k: d[k] for k in SETTINGS_FIELDS})

    def record_hash(self) -> str:
        return _digest(self.to_dict())


def _digest(obj: Mapping[str, Any]) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def collect_environment() -> dict[str, str]:
    """Code and hardware identity. Unavailable items are reported, never omitted."""
    repo = Path(saddle.__file__).resolve().parents[2]
    git = ["git", "-c", f"safe.directory={repo}"]
    commit = _run([*git, "rev-parse", "HEAD"], cwd=repo) or UNKNOWN
    status = _run([*git, "status", "--porcelain", "--untracked-files=no"], cwd=repo)
    gpu = _run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"]
    )
    name, cap, driver = (
        (s.strip() for s in gpu.splitlines()[0].split(",")) if gpu else (UNKNOWN,) * 3
    )
    try:
        import cuest  # type: ignore[import-not-found]

        cuest_version = str(getattr(cuest, "__version__", UNKNOWN))
    except ImportError:
        cuest_version = "not_installed"
    return {
        "git_commit": commit,
        "git_dirty": UNKNOWN if status is None else ("true" if status else "false"),
        "saddle_version": saddle.__version__,
        "cuest_version": cuest_version,
        "cuda_version": UNKNOWN if not gpu else f"driver {driver}",
        "gpu_model": name,
        "compute_capability": cap,
    }
