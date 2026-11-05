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
