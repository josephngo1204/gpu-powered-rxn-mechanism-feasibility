"""CuestModel: the real ElectronicModel (spec §7) behind evaluate(coords).

It owns one cuEST handle (one worker process per GPU, spec §9), runs RHF or
RKS with an analytic gradient, warm-starts from the previous orbitals, and
records SCF diagnostics for gate G2 and the timing measurements of spec §9.
Plans are rebuilt for every geometry because pair lists, DF plans and grids
capture coordinates (spec §2, §16 Q3).
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from saddle.basis import BasisSet, load_basis
from saddle.cuest import Handle, MathMode
from saddle.cuest.bindings import CUEST_VERSION
from saddle.grids import GridSpec, parse_grid
from saddle.scf.rhf import SCFError, SCFResult, rks
from saddle.solvation import PCMModel

FloatArray = npt.NDArray[np.float64]

# Distance guard (spec §8: cuEST warns against far-from-equilibrium geometries
# with severe roundoff). Searches that collapse atoms are stopped here.
MIN_INTERATOMIC_BOHR = 0.5


def check_geometry(coords_bohr: npt.ArrayLike) -> None:
    x = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3)
    if len(x) < 2:
        return
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    if float(d.min()) < MIN_INTERATOMIC_BOHR:
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        raise SCFError(
            f"distance guard: atoms {i + 1} and {j + 1} are {float(d[i, j]):.3f} bohr apart "
            f"(minimum {MIN_INTERATOMIC_BOHR})"
        )


@dataclass(frozen=True)
class LevelOfTheory:
    """One spec §4 profile: method, basis, auxiliary basis, grid, thresholds, math mode."""

    functional: str | None  # None = HF
    basis: str
    auxiliary_basis: str = "def2-universal-jkfit"
    grid: str | None = "(99,590)"
    # VV10 grid. PROPOSED default ~SG-1 size (spec gives none; ADR 0002):
    # wB97X-V was parameterized with VV10 on SG-1, and VV10 cost grows with
    # the square of the grid (12 s vs 2 s per gradient for HCN at (99,590)).
    nonlocal_grid: str | None = "(50,194)"
    threshold_pq: float = 1e-14
    orbital_gradient: float = 1e-8
    math_mode: MathMode = MathMode.NATIVE_FP64
    solvent: PCMModel | None = None

    @property
    def grid_spec(self) -> GridSpec | None:
        return None if self.functional is None or self.grid is None else parse_grid(self.grid)

    @property
    def nonlocal_grid_spec(self) -> GridSpec | None:
        if self.functional is None or self.nonlocal_grid is None:
            return None
        return parse_grid(self.nonlocal_grid)

    def describe(self) -> dict[str, Any]:
        return {
            "functional": self.functional or "HF",
            "basis": self.basis,
            "auxiliary_basis": self.auxiliary_basis,
            "grid": self.grid if self.functional else "n/a",
            "nonlocal_grid": self.nonlocal_grid if self.functional else "n/a",
            "thresholdPQ": self.threshold_pq,
            "orbital_gradient": self.orbital_gradient,
            "math_mode": self.math_mode.value,
            "solvent": "gas"
            if self.solvent is None
            else f"{self.solvent.method} eps={self.solvent.epsilon}",
            "cuest_version": CUEST_VERSION,
        }


@dataclass
class EvaluationRecord:
    energy: float
    converged: bool
    iterations: int
    orbital_gradient: float
    warm_start: bool
    seconds: float
    solvation_energy: float


@dataclass
class CuestModel:
    level: LevelOfTheory
    symbols: Sequence[str]
    charge: int = 0
    warm_start: bool = True
    history: list[EvaluationRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._handle = Handle(self.level.math_mode)
        self._basis: BasisSet = load_basis(self.level.basis, self.symbols)
        self._aux: BasisSet = load_basis(self.level.auxiliary_basis, self.symbols)
        self._previous: Any = None

    def run(
        self,
        coords_bohr: npt.ArrayLike,
        *,
        gradient: bool = True,
        warm: bool | None = None,
        stability: bool = False,
    ) -> SCFResult:
        check_geometry(coords_bohr)
        use_warm = self.warm_start if warm is None else warm
        guess = self._previous if use_warm else None
        t0 = time.perf_counter()
        result = rks(
            self._handle,
            self._basis,
            self._aux,
            self.symbols,
            coords_bohr,
            functional=self.level.functional,
            grid=self.level.grid_spec,
            nonlocal_grid=self.level.nonlocal_grid_spec,
            solvent=self.level.solvent,
            gradient=gradient,
            guess=guess,
            stability=stability,
            charge=self.charge,
            threshold_pq=self.level.threshold_pq,
            orbital_gradient_tol=self.level.orbital_gradient,
        )
        if not result.converged:
            raise SCFError(f"SCF did not converge in {result.iterations} iterations")
        self._previous = result.coefficients
        self.history.append(
            EvaluationRecord(
                result.energy,
                result.converged,
                result.iterations,
                result.orbital_gradient,
                guess is not None,
                time.perf_counter() - t0,
                result.solvation_energy,
            )
        )
        return result

    def evaluate(self, coords: FloatArray) -> tuple[float, FloatArray]:
        """ElectronicModel protocol: (energy Eh, flattened gradient Eh/bohr)."""
        r = self.run(coords, gradient=True)
        assert r.gradient is not None
        return r.energy, np.asarray(r.gradient, dtype=np.float64).ravel()

    def diagnostics(self) -> Mapping[str, Any]:
        warm = [h.iterations for h in self.history if h.warm_start]
        cold = [h.iterations for h in self.history if not h.warm_start]
        return {
            "evaluations": len(self.history),
            "all_converged": all(h.converged for h in self.history),
            "mean_iterations_warm": float(np.mean(warm)) if warm else None,
            "mean_iterations_cold": float(np.mean(cold)) if cold else None,
            "seconds": float(sum(h.seconds for h in self.history)),
        }

    def close(self) -> None:
        self._handle.close()
