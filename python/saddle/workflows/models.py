"""Electronic-structure model factory for campaigns.

Fake surfaces come from the C++ core. Real cuEST models arrive in Phase F
and plug in through the same evaluate(coords) -> (energy, gradient) protocol.
"""

import hashlib
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from saddle import _core

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
        raise ModelError("cuEST models are not available until Phase F")
    else:
        raise ModelError(f"unknown model type {kind!r}")
    noise = variant.get("gradient_noise")
    if noise is not None:
        model = NoisyModel(model, float(noise), seed=int(variant.get("seed", 0)))
    return model
