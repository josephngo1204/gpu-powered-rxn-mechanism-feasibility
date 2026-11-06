"""Electronic-structure model factory for campaigns.

Fake surfaces come from the C++ core. Real cuEST models plug in through the same
evaluate(coords) -> (energy, gradient) protocol.
"""

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from saddle import _core
from saddle.errors import PCMNotConverged, SCFError, StabilityNotConverged

FloatArray = npt.NDArray[np.float64]


class Model(Protocol):
    def evaluate(self, coords: FloatArray) -> tuple[float, FloatArray]: ...


class ModelError(ValueError):
    pass


class NoisyModel:
    """Adds reproducible per-geometry Gaussian gradient noise (a precision stand-in)."""

    def __init__(self, inner: Model, amplitude: float, seed: int = 0) -> None:
        self.inner = inner
        self.amplitude = amplitude
        self.seed = seed

    def evaluate(self, coords: FloatArray) -> tuple[float, FloatArray]:
        energy, gradient = self.inner.evaluate(coords)
        digest = hashlib.sha256(
            np.ascontiguousarray(coords, dtype="<f8").tobytes() + self.seed.to_bytes(8, "little")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        noise = self.amplitude * rng.standard_normal(np.asarray(gradient).shape)
        return energy, np.asarray(gradient) + noise


def build_model(spec: Mapping[str, Any], variant: Mapping[str, Any] | None = None) -> Model:
    """Build a model from a campaign 'model' block, optionally with a robustness variant."""
    kind = spec.get("type")
    variant = variant or {}
    scale = dict(variant.get("model_scale", {}))
    unknown_scale = set(scale) - {"morse_depth", "spring_k"}
    if unknown_scale:
        raise ModelError(f"unknown model_scale keys: {sorted(unknown_scale)}")
    model: Model
    if kind == "fake:pair_model":
        springs = [
            (int(i), int(j), float(k) * float(scale.get("spring_k", 1.0)), float(r0))
            for i, j, k, r0 in spec.get("springs", [])
        ]
        morse = [
            (int(i), int(j), float(d) * float(scale.get("morse_depth", 1.0)), float(a), float(re))
            for i, j, d, a, re in spec.get("morse", [])
        ]
        model = _core.PairModel(springs, morse)
    elif kind == "fake:springs":
        if scale:
            raise ModelError("model_scale is only supported for fake:pair_model")
        model = _core.Springs(
            [(int(i), int(j), float(k), float(r0)) for i, j, k, r0 in spec["springs"]]
        )
    elif kind == "cuest":
        raise ModelError("cuest models are built by build_provider(), which knows the system")
    else:
        raise ModelError(f"unknown model type {kind!r}")
    noise = variant.get("gradient_noise")
    if noise is not None:
        model = NoisyModel(model, float(noise), seed=int(variant.get("seed", 0)))
    return model


@dataclasses.dataclass(frozen=True)
class SystemInfo:
    """What a provider needs to know about the hypothesis."""

    symbols: tuple[str, ...]
    species_symbols: tuple[tuple[str, ...], ...]
    species_charges: tuple[int, ...]
    charge: int
    solvent: str | None


class ModelProvider:
    """Builds models for the whole system and for each reactant species."""

    kind = "abstract"

    def system(self) -> Model:
        raise NotImplementedError

    def fragment(self, index: int) -> Model:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        raise NotImplementedError

    def g2_evidence(self, model: Model, coords: FloatArray) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class FakeProvider(ModelProvider):
    kind = "fake"

    def __init__(
        self, spec: Mapping[str, Any], variant: Mapping[str, Any] | None, info: SystemInfo
    ) -> None:
        self.spec = spec
        self.variant = variant or {}
        self.info = info
        self._model = build_model(spec, variant)

    def system(self) -> Model:
        return self._model

    def fragment(self, index: int) -> Model:
        if len(self.info.species_symbols) != 1:
            raise ModelError(
                "fake surfaces describe the whole system only; bimolecular reactants need a "
                "model that can evaluate separated fragments (use a cuest model)"
            )
        return self._model

    def describe(self) -> dict[str, Any]:
        return {"type": str(self.spec.get("type")), "variant": dict(self.variant)}

    def g2_evidence(self, model: Model, coords: FloatArray) -> dict[str, Any]:
        return {
            "scf_converged": True,
            "orbitals_stable": True,
            "uses_pcm": False,
            "pcm_converged": None,
            "note": "fake surface: no SCF",
        }


class CuestProvider(ModelProvider):
    """Real DFT/HF through saddle.scf.model.CuestModel."""

    kind = "cuest"

    def __init__(
        self, spec: Mapping[str, Any], variant: Mapping[str, Any] | None, info: SystemInfo
    ) -> None:
        from saddle.cuest import MathMode
        from saddle.scf.model import LevelOfTheory
        from saddle.solvation import solvent_model

        merged = dict(spec)
        merged.update(variant or {})
        unknown = set(merged) - {
            "type",
            "functional",
            "basis",
            "auxiliary_basis",
            "grid",
            "nonlocal_grid",
            "math_mode",
            "threshold_pq",
            "orbital_gradient",
            "pcm",
        }
        if unknown:
            raise ModelError(f"unknown cuest model keys: {sorted(unknown)}")
        functional = merged.get("functional")
        if functional is not None and str(functional).upper() in ("HF", "RHF"):
            functional = None
        pcm_options = dict(merged.get("pcm", {}))
        self.level = LevelOfTheory(
            functional=functional,
            basis=str(merged["basis"]),
            auxiliary_basis=str(merged.get("auxiliary_basis", "def2-universal-jkfit")),
            grid=merged.get("grid", "(99,590)"),
            nonlocal_grid=merged.get("nonlocal_grid", "(50,194)"),
            threshold_pq=float(merged.get("threshold_pq", 1e-14)),
            orbital_gradient=float(merged.get("orbital_gradient", 1e-10)),
            math_mode=MathMode(merged.get("math_mode", "native-fp64")),
            solvent=solvent_model(info.solvent, **pcm_options),
        )
        self.info = info
        self.variant = dict(variant or {})
        self._models: dict[Any, Any] = {}

    def _model(self, key: Any, symbols: Sequence[str], charge: int) -> Any:
        from saddle.scf.model import CuestModel

        if key not in self._models:
            self._models[key] = CuestModel(self.level, list(symbols), charge=charge)
        return self._models[key]

    def system(self) -> Model:
        return self._model("system", self.info.symbols, self.info.charge)  # type: ignore[no-any-return]

    def fragment(self, index: int) -> Model:
        return self._model(  # type: ignore[no-any-return]
            ("fragment", index), self.info.species_symbols[index], self.info.species_charges[index]
        )

    def describe(self) -> dict[str, Any]:
        d = {"type": "cuest", **self.level.describe(), "variant": self.variant}
        return d

    def g2_evidence(self, model: Model, coords: FloatArray) -> dict[str, Any]:
        """Rerun the SCF at the TS with an internal stability analysis.

        Failures become G2 evidence: an unconverged SCF fails G2, and a
        stability solver that does not converge leaves stability unknown,
        which G2 treats as missing evidence.
        """
        uses_pcm = self.level.solvent is not None
        pcm = {"uses_pcm": uses_pcm, "pcm_converged": True if uses_pcm else None}
        try:
            r = model.run(coords, gradient=False, stability=True)  # type: ignore[attr-defined]
        except PCMNotConverged as exc:
            return {
                "scf_converged": False,
                "orbitals_stable": False,
                "uses_pcm": True,
                "pcm_converged": False,
                "note": f"PCM not converged at the TS: {exc}",
            }
        except StabilityNotConverged as exc:
            return {
                **pcm,
                "scf_converged": True,
                "orbitals_stable": None,
                "note": f"stability analysis failed at the TS: {exc}",
            }
        except SCFError as exc:
            return {
                **pcm,
                "scf_converged": False,
                "orbitals_stable": None,
                "note": f"SCF failed at the TS: {exc}",
            }
        return {
            **pcm,
            "scf_converged": bool(r.converged),
            "orbitals_stable": r.orbitals_stable,
            "note": (
                f"internal RHF->RHF stability, lowest A+B eigenvalue "
                f"{float(r.stability_eigenvalues[0]):.4f} Eh; RHF->UHF not tested"
            ),
        }

    def close(self) -> None:
        for m in self._models.values():
            m.close()
        self._models.clear()


def build_provider(
    spec: Mapping[str, Any], variant: Mapping[str, Any] | None, info: SystemInfo
) -> ModelProvider:
    kind = str(spec.get("type", ""))
    if kind == "cuest":
        return CuestProvider(spec, variant, info)
    if kind.startswith("fake:"):
        return FakeProvider(spec, variant, info)
    raise ModelError(f"unknown model type {kind!r}")
