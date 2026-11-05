"""cuEST AO basis and pair-list objects built from saddle.basis data."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from saddle.basis import BasisSet
from saddle.cuest.bindings import cb
from saddle.cuest.core import (
    AOShell,
    CuestError,
    Handle,
    Parameters,
    Workspace,
    WorkspaceDescriptor,
    check,
)


def _query(handle: Handle, kind: Any, obj: Any, attribute: Any, holder: Any) -> Any:
    """Query an attribute; cuEST type-checks the holder (INVALID_TYPE/SIZE on mismatch)."""
    check(
        cb.cuestQuery(
            handle=handle.raw, type=kind, object=obj, attribute=attribute, attributeValue=holder
        ),
        f"cuestQuery({attribute})",
    )
    return holder.value


def _query_uint(handle: Handle, kind: Any, obj: Any, attribute: Any) -> int:
    return int(_query(handle, kind, obj, attribute, cb.data_uint64_t()))


def _query_flag(handle: Handle, kind: Any, obj: Any, attribute: Any) -> bool:
    # cuEST stores flags such as IS_PURE as int32 (probed: data_bool is INVALID_TYPE).
    return bool(_query(handle, kind, obj, attribute, cb.data_int32_t()))


class AOBasis:
    """A cuEST AO basis for an ordered list of atoms. Shells live as long as the basis."""

    def __init__(self, handle: Handle, basis: BasisSet, symbols: Sequence[str]) -> None:
        self.handle = handle
        self.basis = basis
        self.symbols = tuple(symbols)
        self.raw = cb.cuestAOBasisHandle()
        self._open = False
        self._shells = [
            AOShell(
                handle,
                is_pure=s.pure,
                angular_momentum=s.angular_momentum,
                exponents=s.exponents,
                coefficients=s.normalized_coefficients(),
            )
            for sym in self.symbols
            for s in basis.shells[sym]
        ]
        per_atom = [len(basis.shells[sym]) for sym in self.symbols]
        shells = [s.raw for s in self._shells]
        args = dict(
            handle=handle.raw,
            numAtoms=len(self.symbols),
            numShellsPerAtom=per_atom,
            shells=shells,
            outBasis=self.raw,
        )
        with Parameters(cb.CuestParametersType.CUEST_AOBASIS_PARAMETERS) as params:
            persistent, temporary = WorkspaceDescriptor(), WorkspaceDescriptor()
            check(
                cb.cuestAOBasisCreateWorkspaceQuery(
                    parameters=params.raw,
                    persistentWorkspaceDescriptor=persistent.pointer,
                    temporaryWorkspaceDescriptor=temporary.pointer,
                    **args,
                ),
                "cuestAOBasisCreateWorkspaceQuery",
            )
            self._persistent = Workspace(persistent)
            tmp = Workspace(temporary)
            try:
                check(
                    cb.cuestAOBasisCreate(
                        parameters=params.raw,
                        persistentWorkspace=self._persistent.pointer,
                        temporaryWorkspace=tmp.pointer,
                        **args,
                    ),
                    "cuestAOBasisCreate",
                )
            finally:
                tmp.release()
        self._open = True
        kind = cb.CuestType.CUEST_AOBASIS
        attrs = cb.CuestAOBasisAttributes
        self.num_ao = _query_uint(handle, kind, self.raw, attrs.CUEST_AOBASIS_NUM_AO)
        self.num_shell = _query_uint(handle, kind, self.raw, attrs.CUEST_AOBASIS_NUM_SHELL)
        self.num_atom = _query_uint(handle, kind, self.raw, attrs.CUEST_AOBASIS_NUM_ATOM)
        self.max_l = _query_uint(handle, kind, self.raw, attrs.CUEST_AOBASIS_MAX_L)
        self.is_pure = _query_flag(handle, kind, self.raw, attrs.CUEST_AOBASIS_IS_PURE)
        expected = basis.n_functions(self.symbols)
        if self.num_ao != expected:
            raise CuestError(
                "COUNT_MISMATCH", f"cuEST reports {self.num_ao} AOs, expected {expected}"
            )

    def close(self) -> None:
        if self._open:
            self._open = False
            check(cb.cuestAOBasisDestroy(handle=self.raw), "cuestAOBasisDestroy")
            self._persistent.release()
            for s in self._shells:
                s.close()

    def __del__(self) -> None:
        if getattr(self, "_open", False):
            try:
                self.close()
            except CuestError:
                pass


class PairList:
    """Screened AO pair list. It captures coordinates, so each geometry needs its own (spec §2)."""

    def __init__(
        self, handle: Handle, basis: AOBasis, coords_bohr: npt.ArrayLike, *, threshold_pq: float
    ) -> None:
        xyz = np.asarray(coords_bohr, dtype=np.float64).ravel()
        if xyz.size != 3 * len(basis.symbols):
            raise ValueError(f"expected {3 * len(basis.symbols)} coordinates, got {xyz.size}")
        if not (threshold_pq >= 0.0 and np.isfinite(threshold_pq)):
            raise ValueError("threshold_pq must be a finite non-negative number")
        self.threshold_pq = float(threshold_pq)
        self.raw = cb.cuestAOPairListHandle()
        self._open = False
        args = dict(
            handle=handle.raw,
            basis=basis.raw,
            numAtoms=len(basis.symbols),
            xyz=[float(v) for v in xyz],
            thresholdPQ=self.threshold_pq,
            outPairList=self.raw,
        )
        with Parameters(cb.CuestParametersType.CUEST_AOPAIRLIST_PARAMETERS) as params:
            persistent, temporary = WorkspaceDescriptor(), WorkspaceDescriptor()
            check(
                cb.cuestAOPairListCreateWorkspaceQuery(
                    parameters=params.raw,
                    persistentWorkspaceDescriptor=persistent.pointer,
                    temporaryWorkspaceDescriptor=temporary.pointer,
                    **args,
                ),
                "cuestAOPairListCreateWorkspaceQuery",
            )
            self._persistent = Workspace(persistent)
            tmp = Workspace(temporary)
            try:
                check(
                    cb.cuestAOPairListCreate(
                        parameters=params.raw,
                        persistentWorkspace=self._persistent.pointer,
                        temporaryWorkspace=tmp.pointer,
                        **args,
                    ),
                    "cuestAOPairListCreate",
                )
            finally:
                tmp.release()
        self._open = True

    def close(self) -> None:
        if self._open:
            self._open = False
            check(cb.cuestAOPairListDestroy(handle=self.raw), "cuestAOPairListDestroy")
            self._persistent.release()

    def __del__(self) -> None:
        if getattr(self, "_open", False):
            try:
                self.close()
            except CuestError:
                pass
