"""Restricted Hartree-Fock with DIIS on the GPU.

F = h + J[D] - K[C_occ], with D = 2 C_occ C_occ^T. Convergence is judged on
the orbital gradient ||X^T (F D S - S D F) X||_max, and the run reports
non-convergence instead of returning an unconverged energy silently.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cupy
import numpy as np
import numpy.typing as npt

from saddle import elements
from saddle.basis import BasisSet
from saddle.cuest.basis import AOBasis, PairList
from saddle.cuest.core import Handle
from saddle.cuest.df import DFIntPlan
from saddle.cuest.oei import OEIntPlan

# Canonical orthogonalization drops overlap eigenvalues below this (rank guard, spec §6).
OVERLAP_CUTOFF = 1e-8


class SCFError(RuntimeError):
    pass


@dataclass
class SCFResult:
    energy: float
    electronic_energy: float
    nuclear_repulsion: float
    converged: bool
    iterations: int
    orbital_gradient: float
    orbital_energies: npt.NDArray[np.float64]
    coefficients: Any  # cupy (nao, nmo), columns are MOs
    density: Any  # cupy (nao, nao), total density
    retained_rank: int
    history: list[float] = field(default_factory=list)


def nuclear_repulsion(coords_bohr: npt.ArrayLike, charges: Sequence[float]) -> float:
    x = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3)
    z = np.asarray(charges, dtype=np.float64)
    e = 0.0
    for i in range(len(z)):
        for j in range(i):
            e += z[i] * z[j] / float(np.linalg.norm(x[i] - x[j]))
    return e


class DIIS:
    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.focks: list[Any] = []
        self.errors: list[Any] = []

    def extrapolate(self, fock: Any, error: Any) -> Any:
        self.focks.append(fock)
        self.errors.append(error)
        if len(self.focks) > self.size:
            self.focks.pop(0)
            self.errors.pop(0)
        n = len(self.focks)
        if n < 2:
            return fock
        b = cupy.empty((n + 1, n + 1))
        b[-1, :] = -1.0
        b[:, -1] = -1.0
        b[-1, -1] = 0.0
        for i in range(n):
            for j in range(i + 1):
                b[i, j] = b[j, i] = cupy.vdot(self.errors[i], self.errors[j])
        rhs = cupy.zeros(n + 1)
        rhs[-1] = -1.0
        try:
            coef = cupy.linalg.solve(b, rhs)
        except cupy.linalg.LinAlgError:
            self.focks, self.errors = self.focks[-1:], self.errors[-1:]
            return fock
        return sum(c * f for c, f in zip(coef[:n], self.focks, strict=True))


def rhf(
    handle: Handle,
    basis: BasisSet,
    auxiliary: BasisSet,
    symbols: Sequence[str],
    coords_bohr: npt.ArrayLike,
    *,
    charge: int = 0,
    threshold_pq: float = 1e-14,
    orbital_gradient_tol: float = 1e-8,
    energy_tol: float = 1e-10,
    max_iterations: int = 100,
) -> SCFResult:
    z = [elements.atomic_number(s) for s in symbols]
    n_electrons = sum(z) - charge
    if n_electrons <= 0 or n_electrons % 2:
        raise SCFError(f"RHF needs an even, positive electron count, got {n_electrons}")
    n_occ = n_electrons // 2
    xyz = np.asarray(coords_bohr, dtype=np.float64).ravel()

    ao = AOBasis(handle, basis, symbols)
    aux = AOBasis(handle, auxiliary, symbols)
    pairs = PairList(handle, ao, xyz, threshold_pq=threshold_pq)
    oei = OEIntPlan(handle, ao, pairs)
    df = DFIntPlan(handle, ao, aux, pairs, exchange_fraction=1.0)
    try:
        s = oei.overlap()
        h = oei.kinetic() + oei.nuclear_attraction(xyz, z)
        w, v = cupy.linalg.eigh(s)
        keep = w > OVERLAP_CUTOFF
        x = v[:, keep] / cupy.sqrt(w[keep])
        rank = int(keep.sum())
        if rank < n_occ:
            raise SCFError("basis too near-dependent for the electron count")

        def diagonalize(f: Any) -> tuple[Any, Any]:
            e, c_ortho = cupy.linalg.eigh(x.T @ f @ x)
            return e, x @ c_ortho

        eps, c = diagonalize(h)  # core guess
        e_nuc = nuclear_repulsion(xyz, z)
        diis = DIIS()
        history: list[float] = []
        e_old = 0.0
        grad = float("inf")
        converged = False
        iterations = 0
        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            c_occ = c[:, :n_occ]
            d = 2.0 * c_occ @ c_occ.T
            fock = h + df.coulomb(d) - df.exchange(cupy.ascontiguousarray(c_occ.T))
            e_elec = float(0.5 * cupy.vdot(d, h + fock))
            err = fock @ d @ s - s @ d @ fock
            err_o = x.T @ err @ x
            grad = float(cupy.abs(err_o).max())
            history.append(e_elec + e_nuc)
            if grad < orbital_gradient_tol and abs(e_elec - e_old) < energy_tol:
                converged = True
                break
            e_old = e_elec
            eps, c = diagonalize(diis.extrapolate(fock, err_o))
        # Orbitals consistent with the final Fock matrix.
        eps, c = diagonalize(fock)
        return SCFResult(
            energy=e_elec + e_nuc,
            electronic_energy=e_elec,
            nuclear_repulsion=e_nuc,
            converged=converged,
            iterations=iterations,
            orbital_gradient=grad,
            orbital_energies=cupy.asnumpy(eps),
            coefficients=c,
            density=d,
            retained_rank=rank,
            history=history,
        )
    finally:
        df.close()
        oei.close()
        pairs.close()
        aux.close()
        ao.close()
