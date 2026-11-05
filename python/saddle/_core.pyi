"""Saddle C++ core"""

from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray


class MullerBrown:
    def __init__(self) -> None: ...

    def evaluate(self, coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]]:
        """
        Return (energy in Eh, gradient in Eh/bohr) at flattened coordinates in bohr.
        """

class Morse:
    def __init__(self, depth: float, alpha: float, r_eq: float) -> None: ...

    def evaluate(self, coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]]:
        """
        Return (energy in Eh, gradient in Eh/bohr) at flattened coordinates in bohr.
        """

class Harmonic:
    def __init__(self, center: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], force_constants: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]) -> None: ...

    def evaluate(self, coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]]:
        """
        Return (energy in Eh, gradient in Eh/bohr) at flattened coordinates in bohr.
        """

class NoisyMullerBrown:
    def __init__(self, amplitude: float, seed: int) -> None: ...

    def evaluate(self, coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]]:
        """
        Return (energy in Eh, gradient in Eh/bohr) at flattened coordinates in bohr.
        """

class Springs:
    def __init__(self, springs: Sequence[tuple[int, int, float, float]]) -> None:
        """Pairwise harmonic springs given as (i, j, k, r0) tuples."""

    def evaluate(self, coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]]:
        """
        Return (energy in Eh, gradient in Eh/bohr) at flattened coordinates in bohr.
        """

class VibrationalAnalysis:
    @property
    def frequencies_cm(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    @property
    def modes_mass_weighted(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @property
    def modes_cartesian(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]: ...

    @property
    def n_external(self) -> int: ...

    @property
    def linear(self) -> bool: ...

    @property
    def max_external_residual_cm(self) -> float: ...

    @property
    def translation_sum_rule(self) -> float: ...

    def n_imaginary(self, threshold_cm: float) -> int: ...

class PairModel:
    def __init__(self, springs: Sequence[tuple[int, int, float, float]], morse: Sequence[tuple[int, int, float, float, float]]) -> None:
        """Springs (i, j, k, r0) plus Morse bonds (i, j, depth, alpha, r_eq)."""

    def evaluate(self, coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> tuple[float, Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]]:
        """
        Return (energy in Eh, gradient in Eh/bohr) at flattened coordinates in bohr.
        """

def minimize(model: object, coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], atomic_numbers: Sequence[int], max_iterations: int = 200, trust_radius: float = 0.1, exact_hessian: bool = True, fd_step: float = 0.0001, gtol_rms: float = 1e-05, gtol_max: float = 1.5e-05) -> dict:
    """RS-RFO minimization to the G1 thresholds. Returns a dict of results."""

def saddle_search(model: object, coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], atomic_numbers: Sequence[int], follow_direction: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None = None, max_iterations: int = 200, trust_radius: float = 0.1, fd_step: float = 0.0001, gtol_rms: float = 1e-05, gtol_max: float = 1.5e-05) -> dict:
    """RS-P-RFO first-order saddle search with Bofill updates."""

def irc(model: object, coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], atomic_numbers: Sequence[int], hessian: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], masses: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], step: float = 0.05, max_steps: int = 300, switch_gradient: float = 0.001) -> dict:
    """
    IRC in both directions from a first-order saddle, with endpoint optimization.
    """

def probe_noise(model: object, coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], direction: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], delta: float = 0.001, order: int = 6, n_samples: int = 16) -> tuple[float, int]:
    """
    Line-scan gradient noise estimate; returns (epsilon Eh/bohr, gradients used).
    """

def optimal_step(epsilon: float, fourth_derivative: float) -> float:
    """Optimal central-difference step (3 eps / |fourth derivative|)^(1/3)."""

def analyze_vibrations(hessian: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], masses_amu: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> VibrationalAnalysis:
    """
    Harmonic analysis with translation/rotation projection. Frequencies in cm^-1, imaginary modes negative.
    """

def central_stencil(coords: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], step: float) -> list[tuple[int, int, Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]]]:
    """
    Central-difference stencil as (coordinate, sign, displaced coords), in order.
    """

def assemble_central(gradients: Sequence[Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]], step: float) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], float]:
    """
    Assemble (raw, symmetrized, relative asymmetry) from stencil-ordered gradients.
    """
