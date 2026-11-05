"""One-electron integral plan: overlap, kinetic and nuclear attraction (spec §8)."""

from typing import Any

import cupy
import numpy.typing as npt

from saddle.cuest.basis import AOBasis, PairList
from saddle.cuest.bindings import cb
from saddle.cuest.core import (
    CuestError,
    Handle,
    Parameters,
    Workspace,
    WorkspaceDescriptor,
    check,
    device_pointer,
    run_with_workspace,
)

P = cb.CuestParametersType


class OEIntPlan:
    """cuestOEIntPlan for one basis and pair list (hence one geometry)."""

    def __init__(self, handle: Handle, basis: AOBasis, pairs: PairList) -> None:
        self.handle = handle
        self.nao = basis.num_ao
        self.natom = len(basis.symbols)
        self.raw = cb.cuestOEIntPlanHandle()
        self._open = False
        args = dict(handle=handle.raw, basis=basis.raw, pairList=pairs.raw, outPlan=self.raw)
        with Parameters(P.CUEST_OEINTPLAN_PARAMETERS) as params:
            persistent, temporary = WorkspaceDescriptor(), WorkspaceDescriptor()
            check(
                cb.cuestOEIntPlanCreateWorkspaceQuery(
                    parameters=params.raw,
                    persistentWorkspaceDescriptor=persistent.pointer,
                    temporaryWorkspaceDescriptor=temporary.pointer,
                    **args,
                ),
                "cuestOEIntPlanCreateWorkspaceQuery",
            )
            self._persistent = Workspace(persistent)
            tmp = Workspace(temporary)
            try:
                check(
                    cb.cuestOEIntPlanCreate(
                        parameters=params.raw,
                        persistentWorkspace=self._persistent.pointer,
                        temporaryWorkspace=tmp.pointer,
                        **args,
                    ),
                    "cuestOEIntPlanCreate",
                )
            finally:
                tmp.release()
        self._open = True

    def _matrix(self, name: str, params_type: Any, out_kw: str, **extra: Any) -> Any:
        out = cupy.empty((self.nao, self.nao))
        with Parameters(params_type) as params:
            run_with_workspace(
                f"cuest{name}Compute",
                getattr(cb, f"cuest{name}ComputeWorkspaceQuery"),
                getattr(cb, f"cuest{name}Compute"),
                outputs=[out],
                handle=self.handle.raw,
                plan=self.raw,
                parameters=params.raw,
                **{out_kw: device_pointer(out, (self.nao, self.nao))},
                **extra,
            )
        return out

    def overlap(self) -> Any:
        return self._matrix("Overlap", P.CUEST_OVERLAPCOMPUTE_PARAMETERS, "outSMatrix")

    def kinetic(self) -> Any:
        return self._matrix("Kinetic", P.CUEST_KINETICCOMPUTE_PARAMETERS, "outTMatrix")

    def potential(self, xyz_bohr: npt.ArrayLike, charges: npt.ArrayLike) -> Any:
        """V_pq = sum_A q_A <p| 1/|r - R_A| |q>; pass q = -Z for nuclear attraction."""
        q = cupy.asarray(charges, dtype=cupy.float64).ravel()
        xyz = cupy.asarray(xyz_bohr, dtype=cupy.float64).ravel()
        if xyz.size != 3 * q.size or q.size == 0:
            raise ValueError(f"need 3 coordinates per charge, got {xyz.size} for {q.size} charges")
        return self._matrix(
            "Potential",
            P.CUEST_POTENTIALCOMPUTE_PARAMETERS,
            "outVMatrix",
            numCharges=int(q.size),
            xyz=device_pointer(xyz, (xyz.size,)),
            q=device_pointer(q, (q.size,)),
        )

    def nuclear_attraction(self, xyz_bohr: npt.ArrayLike, atomic_numbers: npt.ArrayLike) -> Any:
        return self.potential(xyz_bohr, -cupy.asarray(atomic_numbers, dtype=cupy.float64))

    def _gradient(self, name: str, params_type: Any, density: Any, **extra: Any) -> Any:
        d = cupy.ascontiguousarray(density, dtype=cupy.float64)
        out = cupy.empty((self.natom, 3))
        with Parameters(params_type) as params:
            run_with_workspace(
                f"cuest{name}DerivativeCompute",
                getattr(cb, f"cuest{name}DerivativeComputeWorkspaceQuery"),
                getattr(cb, f"cuest{name}DerivativeCompute"),
                outputs=[out],
                handle=self.handle.raw,
                plan=self.raw,
                parameters=params.raw,
                densityMatrix=device_pointer(d, (self.nao, self.nao)),
                outGradient=device_pointer(out, (self.natom, 3)),
                **extra,
            )
        return out

    def overlap_gradient(self, weighted_density: Any) -> Any:
        """sum_pq W_pq dS_pq/dR (natom, 3)."""
        return self._gradient(
            "Overlap", P.CUEST_OVERLAPDERIVATIVECOMPUTE_PARAMETERS, weighted_density
        )

    def kinetic_gradient(self, density: Any) -> Any:
        """sum_pq D_pq dT_pq/dR (natom, 3)."""
        return self._gradient("Kinetic", P.CUEST_KINETICDERIVATIVECOMPUTE_PARAMETERS, density)

    def potential_gradient(
        self, xyz_bohr: npt.ArrayLike, charges: npt.ArrayLike, density: Any
    ) -> tuple[Any, Any]:
        """(basis-centre gradient (natom, 3), charge-position gradient (ncharge, 3)) of sum D V."""
        q = cupy.asarray(charges, dtype=cupy.float64).ravel()
        xyz = cupy.asarray(xyz_bohr, dtype=cupy.float64).ravel()
        d = cupy.ascontiguousarray(density, dtype=cupy.float64)
        basis_grad = cupy.empty((self.natom, 3))
        charge_grad = cupy.empty((q.size, 3))
        with Parameters(P.CUEST_POTENTIALDERIVATIVECOMPUTE_PARAMETERS) as params:
            run_with_workspace(
                "cuestPotentialDerivativeCompute",
                cb.cuestPotentialDerivativeComputeWorkspaceQuery,
                cb.cuestPotentialDerivativeCompute,
                outputs=[basis_grad, charge_grad],
                handle=self.handle.raw,
                plan=self.raw,
                parameters=params.raw,
                numCharges=int(q.size),
                xyz=device_pointer(xyz, (xyz.size,)),
                q=device_pointer(q, (q.size,)),
                densityMatrix=device_pointer(d, (self.nao, self.nao)),
                outBasisGradient=device_pointer(basis_grad, (self.natom, 3)),
                outChargeGradient=device_pointer(charge_grad, (q.size, 3)),
            )
        return basis_grad, charge_grad

    def close(self) -> None:
        if self._open:
            self._open = False
            check(cb.cuestOEIntPlanDestroy(handle=self.raw), "cuestOEIntPlanDestroy")
            self._persistent.release()

    def __del__(self) -> None:
        if getattr(self, "_open", False):
            try:
                self.close()
            except CuestError:
                pass
