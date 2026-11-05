"""PCM plan and potential (spec §8 hazard: PCM can return success unconverged).

Every call reads the results object and raises PCMNotConverged when the
charges did not converge, so a false success never reaches the SCF.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cupy

from saddle import elements
from saddle.cuest.basis import _query_uint
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
from saddle.cuest.oei import OEIntPlan
from saddle.solvation import ZETA, PCMModel

P = cb.CuestParametersType


class PCMNotConverged(RuntimeError):
    def __init__(self, residual: float, iterations: int) -> None:
        super().__init__(
            f"PCM charges did not converge (residual {residual:.2e}, {iterations} iterations)"
        )
        self.residual = residual
        self.iterations = iterations


@dataclass
class PCMResult:
    energy: float  # dielectric energy, Eh
    potential: Any  # cupy (nao, nao)
    charges: Any  # cupy (npoint,)
    converged: bool
    residual: float
    iterations: int


class _Results:
    def __init__(self) -> None:
        self.raw = cb.Results()
        check(
            cb.cuestResultsCreate(
                resultsType=cb.CuestResultsType.CUEST_PCM_RESULTS, outResults=self.raw
            ),
            "cuestResultsCreate",
        )

    def query(self, attribute: Any, holder: Any) -> Any:
        check(
            cb.cuestResultsQuery(
                resultsType=cb.CuestResultsType.CUEST_PCM_RESULTS,
                results=self.raw,
                attribute=attribute,
                attributeValue=holder,
            ),
            f"cuestResultsQuery({attribute})",
        )
        return holder.value

    def close(self) -> None:
        check(
            cb.cuestResultsDestroy(
                resultsType=cb.CuestResultsType.CUEST_PCM_RESULTS, results=self.raw
            ),
            "cuestResultsDestroy",
        )


class PCMIntPlan:
    def __init__(
        self, handle: Handle, oei: OEIntPlan, symbols: Sequence[str], model: PCMModel
    ) -> None:
        self.handle = handle
        self.model = model
        self.nao = oei.nao
        self.natom = len(symbols)
        self.raw = cb.cuestPCMIntPlanHandle()
        self._open = False
        n = len(symbols)
        args = dict(
            handle=handle.raw,
            intPlan=oei.raw,
            numAngularPointsPerAtom=[model.lebedev_points] * n,
            epsilon=float(model.epsilon),
            zetas=[ZETA[model.lebedev_points]] * n,
            atomicRadii=model.radii_bohr(symbols),
            effectiveNuclearCharges=[float(elements.atomic_number(s)) for s in symbols],
            outPlan=self.raw,
        )
        attrs = cb.CuestPCMIntPlanParametersAttributes
        with Parameters(P.CUEST_PCMINTPLAN_PARAMETERS) as params:
            for attribute, value in (
                (attrs.CUEST_PCMINTPLAN_PARAMETERS_CUTOFF, model.weight_cutoff),
                (attrs.CUEST_PCMINTPLAN_PARAMETERS_X_PREFACTOR, model.x_prefactor),
            ):
                check(
                    cb.cuestParametersConfigure(
                        parametersType=P.CUEST_PCMINTPLAN_PARAMETERS,
                        parameters=params.raw,
                        attribute=attribute,
                        attributeValue=cb.data_double(value),
                    ),
                    f"cuestParametersConfigure({attribute})",
                )
            persistent, temporary = WorkspaceDescriptor(), WorkspaceDescriptor()
            check(
                cb.cuestPCMIntPlanCreateWorkspaceQuery(
                    parameters=params.raw,
                    persistentWorkspaceDescriptor=persistent.pointer,
                    temporaryWorkspaceDescriptor=temporary.pointer,
                    **args,
                ),
                "cuestPCMIntPlanCreateWorkspaceQuery",
            )
            self._persistent = Workspace(persistent)
            tmp = Workspace(temporary)
            try:
                check(
                    cb.cuestPCMIntPlanCreate(
                        parameters=params.raw,
                        persistentWorkspace=self._persistent.pointer,
                        temporaryWorkspace=tmp.pointer,
                        **args,
                    ),
                    "cuestPCMIntPlanCreate",
                )
            finally:
                tmp.release()
        self._open = True
        kind = cb.CuestType.CUEST_PCMINTPLAN
        self.num_point = _query_uint(
            handle, kind, self.raw, cb.CuestPCMIntPlanAttributes.CUEST_PCMINTPLAN_NUM_POINT
        )

    def potential(
        self, density: Any, guess: Any | None = None, *, max_iterations: int | None = None
    ) -> PCMResult:
        """Solve for the surface charges of the total density; warm-start from `guess`."""
        d = cupy.ascontiguousarray(density, dtype=cupy.float64)
        q_in = (
            cupy.zeros(self.num_point)
            if guess is None
            else cupy.ascontiguousarray(guess, dtype=cupy.float64)
        )
        q_out = cupy.empty(self.num_point)
        v = cupy.empty((self.nao, self.nao))
        results = _Results()
        attrs = cb.CuestPCMPotentialComputeParametersAttributes
        try:
            with Parameters(P.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS) as params:
                for attribute, value in (
                    (
                        attrs.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS_CONVERGENCE_THRESHOLD,
                        cb.data_double(self.model.convergence_threshold),
                    ),
                    (
                        attrs.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS_MAX_ITERATIONS,
                        cb.data_uint64_t(int(max_iterations or self.model.max_iterations)),
                    ),
                ):
                    status = cb.cuestParametersConfigure(
                        parametersType=P.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS,
                        parameters=params.raw,
                        attribute=attribute,
                        attributeValue=value,
                    )
                    if (
                        attribute == attrs.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS_MAX_ITERATIONS
                        and status != cb.CuestStatus.CUEST_STATUS_SUCCESS
                    ):
                        status = cb.cuestParametersConfigure(
                            parametersType=P.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS,
                            parameters=params.raw,
                            attribute=attribute,
                            attributeValue=cb.data_int32_t(
                                int(max_iterations or self.model.max_iterations)
                            ),
                        )
                    check(status, f"cuestParametersConfigure({attribute})")
                run_with_workspace(
                    "cuestPCMPotentialCompute",
                    cb.cuestPCMPotentialComputeWorkspaceQuery,
                    cb.cuestPCMPotentialCompute,
                    outputs=[v, q_out],
                    handle=self.handle.raw,
                    plan=self.raw,
                    parameters=params.raw,
                    densityMatrix=device_pointer(d, (self.nao, self.nao)),
                    inQ=device_pointer(q_in, (self.num_point,)),
                    outQ=device_pointer(q_out, (self.num_point,)),
                    outPCMResults=results.raw,
                    outPCMPotentialMatrix=device_pointer(v, (self.nao, self.nao)),
                )
            a = cb.CuestPCMResultAttributes
            energy = float(results.query(a.CUEST_PCMRESULT_PCM_DIELECTRIC_ENERGY, cb.data_double()))
            converged = bool(results.query(a.CUEST_PCMRESULT_CONVERGED, cb.data_int32_t()))
            residual = float(results.query(a.CUEST_PCMRESULT_CONVERGED_RESIDUAL, cb.data_double()))
            iterations = int(
                results.query(a.CUEST_PCMRESULT_NUM_ITERATIONS_TAKEN, cb.data_uint64_t())
            )
        finally:
            results.close()
        if not converged:
            v.fill(float("nan"))
            raise PCMNotConverged(residual, iterations)
        return PCMResult(energy, v, q_out, converged, residual, iterations)

    def gradient(self, density: Any, guess: Any | None = None) -> Any:
        """Nuclear gradient of the PCM dielectric energy (natom, 3); raises if unconverged."""
        d = cupy.ascontiguousarray(density, dtype=cupy.float64)
        q_in = (
            cupy.zeros(self.num_point)
            if guess is None
            else cupy.ascontiguousarray(guess, dtype=cupy.float64)
        )
        q_out = cupy.empty(self.num_point)
        out = cupy.empty((self.natom, 3))
        results = _Results()
        attrs = cb.CuestPCMDerivativeComputeParametersAttributes
        try:
            with Parameters(P.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS) as params:
                check(
                    cb.cuestParametersConfigure(
                        parametersType=P.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS,
                        parameters=params.raw,
                        attribute=attrs.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS_CONVERGENCE_THRESHOLD,
                        attributeValue=cb.data_double(self.model.convergence_threshold),
                    ),
                    "cuestParametersConfigure(CONVERGENCE_THRESHOLD)",
                )
                run_with_workspace(
                    "cuestPCMDerivativeCompute",
                    cb.cuestPCMDerivativeComputeWorkspaceQuery,
                    cb.cuestPCMDerivativeCompute,
                    outputs=[out],
                    handle=self.handle.raw,
                    plan=self.raw,
                    parameters=params.raw,
                    densityMatrix=device_pointer(d, (self.nao, self.nao)),
                    inQ=device_pointer(q_in, (self.num_point,)),
                    outQ=device_pointer(q_out, (self.num_point,)),
                    outPCMResults=results.raw,
                    outPCMGradient=device_pointer(out, (self.natom, 3)),
                )
            a = cb.CuestPCMResultAttributes
            converged = bool(results.query(a.CUEST_PCMRESULT_CONVERGED, cb.data_int32_t()))
            residual = float(results.query(a.CUEST_PCMRESULT_CONVERGED_RESIDUAL, cb.data_double()))
            iterations = int(
                results.query(a.CUEST_PCMRESULT_NUM_ITERATIONS_TAKEN, cb.data_uint64_t())
            )
        finally:
            results.close()
        if not converged:
            out.fill(float("nan"))
            raise PCMNotConverged(residual, iterations)
        return out

    def close(self) -> None:
        if self._open:
            self._open = False
            check(cb.cuestPCMIntPlanDestroy(handle=self.raw), "cuestPCMIntPlanDestroy")
            self._persistent.release()

    def __del__(self) -> None:
        if getattr(self, "_open", False):
            try:
                self.close()
            except CuestError:
                pass
