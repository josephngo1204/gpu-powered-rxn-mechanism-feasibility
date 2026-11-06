"""Exchange-correlation integral plans and RKS potentials.

The functional's exact-exchange fractions and VV10 parameters are read back
from the cuEST plan, never hard-coded, so the DF exchange always matches the
functional. HF is routed around XC (cuEST rejects HF in potential routines).
"""

from typing import Any

import cupy

from saddle.cuest.basis import AOBasis, _query_flag
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
from saddle.cuest.grid import MolecularGrid

P = cb.CuestParametersType
F = cb.CuestXCIntPlanParametersFunctional
A = cb.CuestXCIntPlanAttributes

FUNCTIONALS = {
    "SVWN5": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_SVWN5,
    "PBE": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE,
    "BLYP": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_BLYP,
    "PBE0": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE0,
    "B3LYP5": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP5,
    "B3LYP1": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B3LYP1,
    "B97": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97,
    "M06-L": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06L,
    "M06": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M06,
    "M06-2X": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_M062X,
    "R2SCAN": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_R2SCAN,
    "CAM-B3LYP": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_CAMB3LYP,
    "HSE06": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HSE06,
    "LC-WPBE": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_LCWPBE,
    "LC-WPBEH": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_LCWPBEH,
    "WB97X": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97X,
    "WB97X-V": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97XV,
    "WB97M-V": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_WB97MV,
    "B97M-V": F.CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_B97MV,
}

DEFAULT_XC_BUFFER_BYTES = 2 << 30


def functional_key(name: str) -> str:
    key = name.upper().replace("Ω", "W").replace("OMEGA", "W")
    if key in ("HF", "RHF"):
        raise ValueError("HF has no XC part; route it around XC")
    if key not in FUNCTIONALS:
        raise ValueError(f"functional {name!r} is not in cuEST's table: {sorted(FUNCTIONALS)}")
    return key


def _query_double(handle: Handle, obj: Any, attribute: Any) -> float:
    holder = cb.data_double()
    check(
        cb.cuestQuery(
            handle=handle.raw,
            type=cb.CuestType.CUEST_XCINTPLAN,
            object=obj,
            attribute=attribute,
            attributeValue=holder,
        ),
        f"cuestQuery({attribute})",
    )
    return float(holder.value)


class XCIntPlan:
    def __init__(
        self,
        handle: Handle,
        basis: AOBasis,
        grid: MolecularGrid,
        functional: str,
        *,
        buffer_bytes: int = DEFAULT_XC_BUFFER_BYTES,
    ) -> None:
        self.handle = handle
        self.functional = functional_key(functional)
        self.nao = basis.num_ao
        self.natom = len(basis.symbols)
        self.grid = grid
        self._buffer = WorkspaceDescriptor(0, int(buffer_bytes))
        self.raw = cb.cuestXCIntPlanHandle()
        self._open = False
        args = dict(
            handle=handle.raw,
            basis=basis.raw,
            grid=grid.raw,
            functional=FUNCTIONALS[self.functional],
            outPlan=self.raw,
        )
        with Parameters(P.CUEST_XCINTPLAN_PARAMETERS) as params:
            persistent, temporary = WorkspaceDescriptor(), WorkspaceDescriptor()
            check(
                cb.cuestXCIntPlanCreateWorkspaceQuery(
                    parameters=params.raw,
                    persistentWorkspaceDescriptor=persistent.pointer,
                    temporaryWorkspaceDescriptor=temporary.pointer,
                    **args,
                ),
                "cuestXCIntPlanCreateWorkspaceQuery",
            )
            self._persistent = Workspace(persistent)
            tmp = Workspace(temporary)
            try:
                check(
                    cb.cuestXCIntPlanCreate(
                        parameters=params.raw,
                        persistentWorkspace=self._persistent.pointer,
                        temporaryWorkspace=tmp.pointer,
                        **args,
                    ),
                    "cuestXCIntPlanCreate",
                )
            finally:
                tmp.release()
        self._open = True
        kind = cb.CuestType.CUEST_XCINTPLAN
        self.is_hybrid = _query_flag(handle, kind, self.raw, A.CUEST_XCINTPLAN_IS_HYBRID)
        self.is_lrc = _query_flag(handle, kind, self.raw, A.CUEST_XCINTPLAN_IS_LRC_HYBRID)
        self.is_vv10 = _query_flag(handle, kind, self.raw, A.CUEST_XCINTPLAN_IS_VV10)
        self.exchange_scale = _query_double(handle, self.raw, A.CUEST_XCINTPLAN_EXCHANGE_SCALE)
        self.lrc_exchange_scale = _query_double(
            handle, self.raw, A.CUEST_XCINTPLAN_LRC_EXCHANGE_SCALE
        )
        self.lrc_omega = _query_double(handle, self.raw, A.CUEST_XCINTPLAN_LRC_OMEGA)
        if self.is_vv10:
            self.vv10_b = _query_double(handle, self.raw, A.CUEST_XCINTPLAN_VV10_B)
            self.vv10_c = _query_double(handle, self.raw, A.CUEST_XCINTPLAN_VV10_C)
            self.vv10_scale = _query_double(handle, self.raw, A.CUEST_XCINTPLAN_VV10_SCALE)

    @property
    def needs_exchange(self) -> bool:
        return self.exchange_scale != 0.0 or self.lrc_exchange_scale != 0.0

    def _potential(
        self, name: str, params_type: Any, occupied: Any, configure: dict[Any, float]
    ) -> tuple[float, Any]:
        c = cupy.ascontiguousarray(occupied, dtype=cupy.float64)
        if c.ndim != 2 or c.shape[1] != self.nao or c.shape[0] == 0:
            raise ValueError(f"occupied coefficients must be (n_occ, {self.nao}), got {c.shape}")
        out = cupy.empty((self.nao, self.nao))
        exc = cb.data_double()
        with Parameters(params_type) as params:
            for attribute, value in configure.items():
                check(
                    cb.cuestParametersConfigure(
                        parametersType=params_type,
                        parameters=params.raw,
                        attribute=attribute,
                        attributeValue=cb.data_double(value),
                    ),
                    f"cuestParametersConfigure({attribute})",
                )
            run_with_workspace(
                name,
                getattr(cb, f"{name}WorkspaceQuery"),
                getattr(cb, name),
                outputs=[out],
                handle=self.handle.raw,
                plan=self.raw,
                parameters=params.raw,
                variableBufferSize=self._buffer.pointer,
                numOccupied=int(c.shape[0]),
                coefficientMatrix=device_pointer(c, c.shape),
                outXCEnergy=exc,
                outXCPotentialMatrix=device_pointer(out, (self.nao, self.nao)),
            )
        return float(exc.value), out

    def rks_potential(self, occupied: Any) -> tuple[float, Any]:
        """Local (semi-)local XC energy and potential for doubly occupied orbitals (n_occ, nao)."""
        return self._potential(
            "cuestXCPotentialRKSCompute", P.CUEST_XCPOTENTIALRKSCOMPUTE_PARAMETERS, occupied, {}
        )

    def rks_nonlocal_potential(self, occupied: Any) -> tuple[float, Any]:
        """VV10 nonlocal correlation energy and potential (zero if the functional has none)."""
        if not self.is_vv10:
            return 0.0, cupy.zeros((self.nao, self.nao))
        attrs = cb.CuestNonlocalXCPotentialRKSComputeParametersAttributes
        return self._potential(
            "cuestNonlocalXCPotentialRKSCompute",
            P.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS,
            occupied,
            {
                attrs.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_B: self.vv10_b,
                attrs.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_C: self.vv10_c,
                attrs.CUEST_NONLOCALXCPOTENTIALRKSCOMPUTE_PARAMETERS_VV10_SCALE: self.vv10_scale,
            },
        )

    def _gradient(
        self, name: str, params_type: Any, occupied: Any, configure: dict[Any, float]
    ) -> Any:
        c = cupy.ascontiguousarray(occupied, dtype=cupy.float64)
        out = cupy.empty((self.natom, 3))
        with Parameters(params_type) as params:
            for attribute, value in configure.items():
                check(
                    cb.cuestParametersConfigure(
                        parametersType=params_type,
                        parameters=params.raw,
                        attribute=attribute,
                        attributeValue=cb.data_double(value),
                    ),
                    f"cuestParametersConfigure({attribute})",
                )
            run_with_workspace(
                name,
                getattr(cb, f"{name}WorkspaceQuery"),
                getattr(cb, name),
                outputs=[out],
                handle=self.handle.raw,
                plan=self.raw,
                parameters=params.raw,
                variableBufferSize=self._buffer.pointer,
                numOccupied=int(c.shape[0]),
                coefficientMatrix=device_pointer(c, c.shape),
                outGradient=device_pointer(out, (self.natom, 3)),
            )
        return out

    def rks_gradient(self, occupied: Any) -> Any:
        return self._gradient(
            "cuestXCDerivativeRKSCompute", P.CUEST_XCDERIVATIVERKSCOMPUTE_PARAMETERS, occupied, {}
        )

    def rks_nonlocal_gradient(self, occupied: Any) -> Any:
        if not self.is_vv10:
            return cupy.zeros((self.natom, 3))
        attrs = cb.CuestNonlocalXCDerivativeRKSComputeParametersAttributes
        return self._gradient(
            "cuestNonlocalXCDerivativeRKSCompute",
            P.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS,
            occupied,
            {
                attrs.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_B: self.vv10_b,
                attrs.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_C: self.vv10_c,
                attrs.CUEST_NONLOCALXCDERIVATIVERKSCOMPUTE_PARAMETERS_VV10_SCALE: self.vv10_scale,
            },
        )

    def close(self) -> None:
        if self._open:
            self._open = False
            check(cb.cuestXCIntPlanDestroy(handle=self.raw), "cuestXCIntPlanDestroy")
            self._persistent.release()

    def __del__(self) -> None:
        if getattr(self, "_open", False):
            try:
                self.close()
            except CuestError:
                pass
