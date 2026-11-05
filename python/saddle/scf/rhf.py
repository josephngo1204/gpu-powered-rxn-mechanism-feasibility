"""Closed-shell SCF (RHF and RKS) with DIIS on the GPU.

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
from saddle.cuest.grid import MolecularGrid
from saddle.cuest.oei import OEIntPlan
from saddle.cuest.pcm import PCMIntPlan
from saddle.cuest.xc import XCIntPlan
from saddle.grids import GridSpec
from saddle.solvation import PCMModel

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
    xc_energy: float = 0.0
    nonlocal_energy: float = 0.0
    functional: str | None = None
    grid: str | None = None
    solvation_energy: float = 0.0
    solvent: str | None = None
    gradient: npt.NDArray[np.float64] | None = None  # (natom, 3), Eh/bohr
    gradient_terms: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    # Lowest eigenvalues of the real singlet stability matrix A + B (Eh); None if not run.
    # Covers RHF -> RHF (internal) only; RHF -> UHF needs UKS (spec Stage 6).
    stability_eigenvalues: npt.NDArray[np.float64] | None = None
    orbitals_stable: bool | None = None


def nuclear_repulsion(coords_bohr: npt.ArrayLike, charges: Sequence[float]) -> float:
    x = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3)
    z = np.asarray(charges, dtype=np.float64)
    e = 0.0
    for i in range(len(z)):
        for j in range(i):
            e += z[i] * z[j] / float(np.linalg.norm(x[i] - x[j]))
    return e


# Internal (real RHF -> RHF) stability: orbital-rotation step for the
# finite-difference orbital-Hessian product, and the eigenvalue below which a
# solution is called unstable (eigenvalues of A + B, Eh).
STABILITY_STEP = 1e-4
STABILITY_THRESHOLD = -1e-5

# J/K gradient scales for the closed-shell (RKS) formulation, spec §8:
# E_J = 2 E_J^bare(P), E_K = -E_K^bare(C_occ), P = C_occ C_occ^T.
RKS_DENSITY_SCALE = 2.0
RKS_COEFFICIENT_SCALE = -1.0


def nuclear_repulsion_gradient(
    coords_bohr: npt.ArrayLike, charges: Sequence[float]
) -> npt.NDArray[np.float64]:
    x = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3)
    z = np.asarray(charges, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(len(z)):
        for j in range(len(z)):
            if i != j:
                d = x[i] - x[j]
                g[i] -= z[i] * z[j] * d / float(np.linalg.norm(d)) ** 3
    return g


def orthonormalize_guess(c_prev: Any, s: Any, n_occ: int) -> Any:
    """Warm start (spec §6): Loewdin-orthonormalize previous MOs in the new overlap metric.

    Only the occupied block matters for the first density, so the occupied
    columns are orthonormalized: C' = C (C^T S C)^(-1/2).
    """
    if c_prev.ndim != 2 or c_prev.shape[0] != s.shape[0] or c_prev.shape[1] < n_occ:
        raise SCFError(
            f"guess orbitals {c_prev.shape} do not fit {s.shape[0]} AOs and {n_occ} occupied"
        )
    c_occ = c_prev[:, :n_occ]
    m = c_occ.T @ s @ c_occ
    w, v = cupy.linalg.eigh(m)
    if float(w.min()) < 1e-8:
        raise SCFError("guess orbitals are linearly dependent in the new metric")
    return c_occ @ (v / cupy.sqrt(w)) @ v.T


def internal_stability(
    c: Any,
    eps: Any,
    n_occ: int,
    rank: int,
    s: Any,
    fock_of: Any,
    *,
    n_roots: int = 3,
    step: float = STABILITY_STEP,
) -> npt.NDArray[np.float64]:
    """Lowest eigenvalues of the real singlet orbital Hessian A + B.

    (A + B) u = (e_a - e_i) u + C_v^T [F(C_o + d C_v u) - F(C_o - d C_v u)] C_o / 2d,
    with the rotated occupied orbitals Loewdin-orthonormalized in the S metric.
    The difference captures every response term (J, K, XC kernel, PCM), so the
    same code serves HF and every functional.
    """
    from saddle import _core

    c_o = c[:, :n_occ]
    c_v = c[:, n_occ:rank]
    nv = int(c_v.shape[1])
    if nv == 0:
        return np.array([np.inf])
    d_eps = eps[n_occ:rank][:, None] - eps[:n_occ][None, :]

    def rotated(u: Any, sign: float) -> Any:
        co = c_o + sign * step * (c_v @ u)
        w, v = cupy.linalg.eigh(co.T @ s @ co)
        return co @ (v / cupy.sqrt(w)) @ v.T

    def matvec(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u = cupy.asarray(vec).reshape(nv, n_occ)
        response = c_v.T @ (fock_of(rotated(u, 1.0)) - fock_of(rotated(u, -1.0))) @ c_o / (2 * step)
        return np.asarray(cupy.asnumpy((d_eps * u + response).ravel()), dtype=np.float64)

    roots = min(n_roots, nv * n_occ)
    result = _core.davidson(
        matvec,
        nv * n_occ,
        n_roots=roots,
        tolerance=1e-6,
        max_matvecs=80,
        diagonal=np.asarray(cupy.asnumpy(d_eps.ravel()), dtype=np.float64),
        guess=np.zeros((nv * n_occ, 0)),
    )
    if not result["converged"]:
        raise SCFError("stability analysis did not converge")
    return np.asarray(result["eigenvalues"], dtype=np.float64)


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


def rks(
    handle: Handle,
    basis: BasisSet,
    auxiliary: BasisSet,
    symbols: Sequence[str],
    coords_bohr: npt.ArrayLike,
    *,
    functional: str | None = None,
    grid: GridSpec | None = None,
    solvent: PCMModel | None = None,
    gradient: bool = False,
    guess: Any | None = None,
    stability: bool = False,
    charge: int = 0,
    threshold_pq: float = 1e-14,
    orbital_gradient_tol: float = 1e-8,
    energy_tol: float = 1e-10,
    max_iterations: int = 100,
) -> SCFResult:
    """Closed-shell SCF: HF when functional is None, otherwise RKS on `grid`.

    F = h + J[D] - K_df[C_occ] + Vxc + Vnl, where the DF exchange carries the
    functional's own exact-exchange fractions (read from the cuEST XC plan).
    E = sum D h + 1/2 sum D J - 1/2 sum D K + Exc + Enl + E_nuc.
    """
    z = [elements.atomic_number(s) for s in symbols]
    n_electrons = sum(z) - charge
    if n_electrons <= 0 or n_electrons % 2:
        raise SCFError(
            f"closed-shell SCF needs an even, positive electron count, got {n_electrons}"
        )
    if functional is not None and grid is None:
        raise SCFError("a DFT calculation needs an integration grid")
    n_occ = n_electrons // 2
    xyz = np.asarray(coords_bohr, dtype=np.float64).ravel()

    resources: list[Any] = []
    try:
        ao = AOBasis(handle, basis, symbols)
        resources.append(ao)
        aux = AOBasis(handle, auxiliary, symbols)
        resources.append(aux)
        pairs = PairList(handle, ao, xyz, threshold_pq=threshold_pq)
        resources.append(pairs)
        oei = OEIntPlan(handle, ao, pairs)
        resources.append(oei)
        xc = None
        if functional is not None:
            assert grid is not None
            mgrid = MolecularGrid(handle, symbols, xyz, grid)
            resources.append(mgrid)
            xc = XCIntPlan(handle, ao, mgrid, functional)
            resources.append(xc)
            # cuEST reports omega = -1 when there is no long-range exchange.
            omega = xc.lrc_omega if xc.lrc_exchange_scale != 0.0 else 0.0
            fractions = (xc.exchange_scale, xc.lrc_exchange_scale, omega)
        else:
            fractions = (1.0, 0.0, 0.0)
        df = DFIntPlan(
            handle,
            ao,
            aux,
            pairs,
            exchange_fraction=fractions[0],
            lrc_exchange_fraction=fractions[1],
            lrc_omega=fractions[2],
        )
        resources.append(df)
        pcm = None
        if solvent is not None:
            pcm = PCMIntPlan(handle, oei, symbols, solvent)
            resources.append(pcm)
        q_pcm = None
        e_pcm = 0.0
        needs_k = fractions[0] != 0.0 or fractions[1] != 0.0

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

        if guess is not None:
            c = orthonormalize_guess(cupy.asarray(guess), s, n_occ)
        else:
            eps, c = diagonalize(h)  # core guess
        e_nuc = nuclear_repulsion(xyz, z)

        def build_fock(
            c_occ: Any, q_guess: Any
        ) -> tuple[Any, Any, float, float, float, float, Any]:
            """Fock matrix, density and energy terms for doubly occupied orbitals."""
            c_rows = cupy.ascontiguousarray(c_occ.T)
            d = 2.0 * c_occ @ c_occ.T
            j = df.coulomb(d)
            k = df.exchange(c_rows) if needs_k else cupy.zeros_like(j)
            fock = h + j - k
            e_xc = e_nl = e_pcm = 0.0
            q_out = q_guess
            if xc is not None:
                e_xc, vxc = xc.rks_potential(c_rows)
                e_nl, vnl = xc.rks_nonlocal_potential(c_rows)
                fock = fock + vxc + vnl
            if pcm is not None:
                pcm_result = pcm.potential(d, q_guess)  # raises PCMNotConverged
                q_out = pcm_result.charges
                e_pcm = pcm_result.energy
                fock = fock + pcm_result.potential
            e_elec = (
                float(cupy.vdot(d, h) + 0.5 * cupy.vdot(d, j) - 0.5 * cupy.vdot(d, k))
                + e_xc
                + e_nl
                + e_pcm
            )
            return fock, d, e_elec, e_xc, e_nl, e_pcm, q_out

        diis = DIIS()
        history: list[float] = []
        e_old = 0.0
        grad = float("inf")
        converged = False
        iterations = 0
        e_xc = e_nl = 0.0
        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            fock, d, e_elec, e_xc, e_nl, e_pcm, q_pcm = build_fock(c[:, :n_occ], q_pcm)
            err_o = x.T @ (fock @ d @ s - s @ d @ fock) @ x
            grad = float(cupy.abs(err_o).max())
            history.append(e_elec + e_nuc)
            if grad < orbital_gradient_tol and abs(e_elec - e_old) < energy_tol:
                converged = True
                break
            e_old = e_elec
            eps, c = diagonalize(diis.extrapolate(fock, err_o))
        eps, c = diagonalize(fock)
        stab_eigs = None
        stable = None
        if stability:
            if not converged:
                raise SCFError("refusing to analyse stability of an unconverged SCF")
            stab_eigs = internal_stability(
                c, eps, n_occ, rank, s, lambda co: build_fock(co, q_pcm)[0]
            )
            stable = bool(stab_eigs[0] > STABILITY_THRESHOLD)
        grad_total = None
        terms: dict[str, npt.NDArray[np.float64]] = {}
        if gradient:
            if not converged:
                raise SCFError("refusing to compute a gradient from an unconverged SCF")
            c_occ = c[:, :n_occ]
            c_rows = cupy.ascontiguousarray(c_occ.T)
            p_half = c_occ @ c_occ.T
            d_tot = 2.0 * p_half
            w = 2.0 * (c_occ * eps[:n_occ]) @ c_occ.T
            terms["kinetic"] = cupy.asnumpy(oei.kinetic_gradient(d_tot))
            vb, vc = oei.potential_gradient(xyz, -cupy.asarray(z, dtype=cupy.float64), d_tot)
            terms["nuclear_attraction"] = cupy.asnumpy(vb + vc)
            terms["overlap"] = -cupy.asnumpy(oei.overlap_gradient(w))
            terms["coulomb_exchange"] = cupy.asnumpy(
                df.jk_gradient(
                    RKS_DENSITY_SCALE,
                    p_half,
                    RKS_COEFFICIENT_SCALE if needs_k else 0.0,
                    [c_rows],
                )
            )
            if xc is not None:
                terms["xc"] = cupy.asnumpy(xc.rks_gradient(c_rows))
                if xc.is_vv10:
                    terms["vv10"] = cupy.asnumpy(xc.rks_nonlocal_gradient(c_rows))
            if pcm is not None:
                terms["pcm"] = cupy.asnumpy(pcm.gradient(d_tot, q_pcm))
            terms["nuclear_repulsion"] = nuclear_repulsion_gradient(xyz, z)
            grad_total = np.sum(np.stack(list(terms.values())), axis=0)
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
            xc_energy=e_xc,
            nonlocal_energy=e_nl,
            functional=functional,
            grid=None if grid is None else str(grid),
            solvation_energy=e_pcm,
            gradient=grad_total,
            gradient_terms=terms,
            stability_eigenvalues=stab_eigs,
            orbitals_stable=stable,
            solvent=None if solvent is None else f"{solvent.method} eps={solvent.epsilon}",
        )
    finally:
        for r in reversed(resources):
            r.close()


def rhf(
    handle: Handle,
    basis: BasisSet,
    auxiliary: BasisSet,
    symbols: Sequence[str],
    coords_bohr: npt.ArrayLike,
    **kwargs: Any,
) -> SCFResult:
    """Restricted Hartree-Fock (rks without a functional)."""
    return rks(handle, basis, auxiliary, symbols, coords_bohr, functional=None, **kwargs)
