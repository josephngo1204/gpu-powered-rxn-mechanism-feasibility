"""Density-fitted Coulomb and exchange (spec §2: J and K always use DF in cuEST)."""

from collections.abc import Sequence
from typing import Any

import cupy

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
A = cb.CuestDFIntPlanParametersAttributes

# Scratch for DF exchange. NVIDIA suggests 8 GB on large GPUs and down to 2 GB on
# smaller ones; the MIG slice has 18 GB in total.
DEFAULT_EXCHANGE_BUFFER_BYTES = 2 << 30


class DFIntPlan:
    """DF integral plan for one primary basis, auxiliary basis and pair list.

    Plan creation is expensive and coordinate-bound (spec §2), so each
    displaced geometry needs its own plan.
    """

    def __init__(
        self,
        handle: Handle,
        primary: AOBasis,
        auxiliary: AOBasis,
        pairs: PairList,
        *,
        exchange_fraction: float = 1.0,
        lrc_exchange_fraction: float = 0.0,
        lrc_omega: float = 0.0,
        exchange_buffer_bytes: int = DEFAULT_EXCHANGE_BUFFER_BYTES,
    ) -> None:
        self.handle = handle
        self.nao = primary.num_ao
        self.naux = auxiliary.num_ao
        self.natom = len(primary.symbols)
        self.exchange_fraction = float(exchange_fraction)
        self.lrc_exchange_fraction = float(lrc_exchange_fraction)
        self.lrc_omega = float(lrc_omega)
        self._exchange_buffer = WorkspaceDescriptor(0, int(exchange_buffer_bytes))
        self.raw = cb.cuestDFIntPlanHandle()
        self._open = False
        args = dict(
            handle=handle.raw,
            primaryBasis=primary.raw,
            auxiliaryBasis=auxiliary.raw,
            pairList=pairs.raw,
            outPlan=self.raw,
        )
        with Parameters(P.CUEST_DFINTPLAN_PARAMETERS) as params:
            for attribute, value in (
                (A.CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION, self.exchange_fraction),
                (A.CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION, self.lrc_exchange_fraction),
                (A.CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA, self.lrc_omega),
            ):
                holder = cb.data_double()
                holder.value = value
                check(
                    cb.cuestParametersConfigure(
                        parametersType=P.CUEST_DFINTPLAN_PARAMETERS,
                        parameters=params.raw,
                        attribute=attribute,
                        attributeValue=holder,
                    ),
                    f"cuestParametersConfigure({attribute})",
                )
            persistent, temporary = WorkspaceDescriptor(), WorkspaceDescriptor()
            check(
                cb.cuestDFIntPlanCreateWorkspaceQuery(
                    parameters=params.raw,
                    persistentWorkspaceDescriptor=persistent.pointer,
                    temporaryWorkspaceDescriptor=temporary.pointer,
                    **args,
                ),
                "cuestDFIntPlanCreateWorkspaceQuery",
            )
            self._persistent = Workspace(persistent)
            tmp = Workspace(temporary)
            try:
                check(
                    cb.cuestDFIntPlanCreate(
                        parameters=params.raw,
                        persistentWorkspace=self._persistent.pointer,
                        temporaryWorkspace=tmp.pointer,
                        **args,
                    ),
                    "cuestDFIntPlanCreate",
                )
            finally:
                tmp.release()
        self._open = True

    def coulomb(self, density: Any) -> Any:
        """J_pq = sum_rs (pq|rs) D_rs for the supplied (total) density."""
        out = cupy.empty((self.nao, self.nao))
        d = cupy.ascontiguousarray(density, dtype=cupy.float64)
        with Parameters(P.CUEST_DFCOULOMBCOMPUTE_PARAMETERS) as params:
            run_with_workspace(
                "cuestDFCoulombCompute",
                cb.cuestDFCoulombComputeWorkspaceQuery,
                cb.cuestDFCoulombCompute,
                outputs=[out],
                handle=self.handle.raw,
                plan=self.raw,
                parameters=params.raw,
                densityMatrix=device_pointer(d, (self.nao, self.nao)),
                outCoulombMatrix=device_pointer(out, (self.nao, self.nao)),
            )
        return out

    def exchange(self, occupied: Any) -> Any:
        """K_pq = sum_i sum_rs [f_HF (pr|qs) + f_LRC (pr|erf|qs)] C_ir C_is.

        `occupied` is (n_occ, nao), row-major (one orbital per row).
        """
        c = cupy.ascontiguousarray(occupied, dtype=cupy.float64)
        if c.ndim != 2 or c.shape[1] != self.nao or c.shape[0] == 0:
            raise ValueError(f"occupied coefficients must be (n_occ, {self.nao}), got {c.shape}")
        out = cupy.empty((self.nao, self.nao))
        with Parameters(P.CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS) as params:
            run_with_workspace(
                "cuestDFSymmetricExchangeCompute",
                cb.cuestDFSymmetricExchangeComputeWorkspaceQuery,
                cb.cuestDFSymmetricExchangeCompute,
                outputs=[out],
                handle=self.handle.raw,
                plan=self.raw,
                parameters=params.raw,
                variableBufferSize=self._exchange_buffer.pointer,
                numOccupied=int(c.shape[0]),
                coefficientMatrix=device_pointer(c, c.shape),
                outExchangeMatrix=device_pointer(out, (self.nao, self.nao)),
            )
        return out

    def jk_gradient(
        self,
        density_scale: float,
        density: Any,
        coefficient_scale: float,
        occupied_sets: Sequence[Any],
        *,
        last_use: bool = False,
    ) -> Any:
        """d/dR [s_D E_J(D) + s_C sum_N E_K(C_N)] (natom, 3).

        The spin formulation decides the scales (spec §8); the typed callers in
        saddle.scf pass them (RKS: s_D = 2 with D = C C^T, s_C = -1, one set).
        `last_use=True` lets cuEST overwrite the plan's tensors, after which the
        plan is unusable (spec §8 hazard), so it defaults to recompute.
        """
        d = cupy.ascontiguousarray(density, dtype=cupy.float64)
        mats = [cupy.ascontiguousarray(c, dtype=cupy.float64) for c in occupied_sets]
        for c in mats:
            if c.ndim != 2 or c.shape[1] != self.nao:
                raise ValueError(f"coefficient sets must be (n_occ, {self.nao})")
        stacked = cupy.concatenate([c.ravel() for c in mats]) if mats else cupy.zeros(1)
        out = cupy.empty((self.natom, 3))
        policy_enum = cb.CuestDFSymmetricDerivativeComputeMemoryPolicy
        policy = (
            policy_enum.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_MEMORY_POLICY_OVERWRITE
            if last_use
            else policy_enum.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_MEMORY_POLICY_RECOMPUTE
        )
        attr = cb.CuestDFSymmetricDerivativeComputeParametersAttributes
        with Parameters(P.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS) as params:
            check(
                cb.cuestParametersConfigure(
                    parametersType=P.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS,
                    parameters=params.raw,
                    attribute=attr.CUEST_DFSYMMETRICDERIVATIVECOMPUTE_PARAMETERS_MEMORY_POLICY,
                    attributeValue=cb.data_cuestDFSymmetricDerivativeComputeMemoryPolicy_t(policy),
                ),
                "cuestParametersConfigure(MEMORY_POLICY)",
            )
            run_with_workspace(
                "cuestDFSymmetricDerivativeCompute",
                cb.cuestDFSymmetricDerivativeComputeWorkspaceQuery,
                cb.cuestDFSymmetricDerivativeCompute,
                outputs=[out],
                handle=self.handle.raw,
                plan=self.raw,
                parameters=params.raw,
                variableBufferSize=self._exchange_buffer.pointer,
                densityScale=float(density_scale),
                densityMatrix=device_pointer(d, (self.nao, self.nao)),
                coefficientScale=float(coefficient_scale),
                numCoefficientMatrices=len(mats),
                numOccupied=[int(c.shape[0]) for c in mats],
                coefficientMatrices=device_pointer(stacked, stacked.shape),
                outGradient=device_pointer(out, (self.natom, 3)),
            )
        if last_use:
            self.usable = False
        return out

    def close(self) -> None:
        if self._open:
            self._open = False
            check(cb.cuestDFIntPlanDestroy(handle=self.raw), "cuestDFIntPlanDestroy")
            self._persistent.release()

    def __del__(self) -> None:
        if getattr(self, "_open", False):
            try:
                self.close()
            except CuestError:
                pass
