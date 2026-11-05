"""cuEST atom and molecular integration grids built from saddle.grids quadratures."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from saddle.cuest.basis import _query_uint
from saddle.cuest.bindings import cb
from saddle.cuest.core import (
    CuestError,
    Handle,
    Parameters,
    Workspace,
    WorkspaceDescriptor,
    check,
)
from saddle.grids import GridSpec, becke_midpoint_bohr, becke_radial

P = cb.CuestParametersType


class _AtomGrid:
    def __init__(self, handle: Handle, symbol: str, spec: GridSpec) -> None:
        r, w = becke_radial(spec.radial, becke_midpoint_bohr(symbol))
        self.raw = cb.cuestAtomGridHandle()
        self._open = False
        with Parameters(P.CUEST_ATOMGRID_PARAMETERS) as params:
            check(
                cb.cuestAtomGridCreate(
                    handle=handle.raw,
                    numRadialPoints=spec.radial,
                    radialNodes=[float(v) for v in r],
                    radialWeights=[float(v) for v in w],
                    numAngularPoints=[spec.angular] * spec.radial,
                    parameters=params.raw,
                    outAtomGrid=self.raw,
                ),
                "cuestAtomGridCreate",
            )
        self._open = True

    def close(self) -> None:
        if self._open:
            self._open = False
            check(cb.cuestAtomGridDestroy(atomGrid=self.raw), "cuestAtomGridDestroy")


class MolecularGrid:
    """Unpruned atom-centred grid; geometry-bound, so each geometry needs a new one."""

    def __init__(
        self, handle: Handle, symbols: Sequence[str], coords_bohr: npt.ArrayLike, spec: GridSpec
    ) -> None:
        xyz = np.asarray(coords_bohr, dtype=np.float64).ravel()
        if xyz.size != 3 * len(symbols):
            raise ValueError(f"expected {3 * len(symbols)} coordinates, got {xyz.size}")
        self.spec = spec
        self.symbols = tuple(symbols)
        self.raw = cb.cuestMolecularGridHandle()
        self._open = False
        self._atoms = [_AtomGrid(handle, s, spec) for s in self.symbols]
        args = dict(
            handle=handle.raw,
            numAtoms=len(self.symbols),
            atomGrid=[a.raw for a in self._atoms],
            xyz=[float(v) for v in xyz],
            outGrid=self.raw,
        )
        try:
            with Parameters(P.CUEST_MOLECULARGRID_PARAMETERS) as params:
                persistent, temporary = WorkspaceDescriptor(), WorkspaceDescriptor()
                check(
                    cb.cuestMolecularGridCreateWorkspaceQuery(
                        parameters=params.raw,
                        persistentWorkspaceDescriptor=persistent.pointer,
                        temporaryWorkspaceDescriptor=temporary.pointer,
                        **args,
                    ),
                    "cuestMolecularGridCreateWorkspaceQuery",
                )
                self._persistent = Workspace(persistent)
                tmp = Workspace(temporary)
                try:
                    check(
                        cb.cuestMolecularGridCreate(
                            parameters=params.raw,
                            persistentWorkspace=self._persistent.pointer,
                            temporaryWorkspace=tmp.pointer,
                            **args,
                        ),
                        "cuestMolecularGridCreate",
                    )
                finally:
                    tmp.release()
        except Exception:
            for a in self._atoms:
                a.close()
            raise
        self._open = True
        kind = cb.CuestType.CUEST_MOLECULARGRID
        attrs = cb.CuestMolecularGridAttributes
        self.num_atom = _query_uint(handle, kind, self.raw, attrs.CUEST_MOLECULARGRID_NUM_ATOM)
        self.num_point = _query_uint(handle, kind, self.raw, attrs.CUEST_MOLECULARGRID_NUM_POINT)
        self.max_angular_point = _query_uint(
            handle, kind, self.raw, attrs.CUEST_MOLECULARGRID_MAX_ANGULAR_POINT
        )

    def close(self) -> None:
        if self._open:
            self._open = False
            check(cb.cuestMolecularGridDestroy(grid=self.raw), "cuestMolecularGridDestroy")
            self._persistent.release()
            for a in self._atoms:
                a.close()

    def __del__(self) -> None:
        if getattr(self, "_open", False):
            try:
                self.close()
            except CuestError:
                pass
