"""cuEST adapter core: status checks, lifetimes, workspaces, device buffers (spec §8).

Hazards handled here:
- every status is checked and mapped to CuestError;
- outputs of a failed compute are overwritten with NaN, so they cannot be used;
- device buffers are checked for shape, dtype, contiguity and residency;
- workspace sizing and compute are bundled into one call.
"""

import enum
from collections.abc import Callable, Sequence
from typing import Any

import cupy
import numpy as np

from saddle.cuest.bindings import cb

_SUCCESS = cb.CuestStatus.CUEST_STATUS_SUCCESS


class CuestError(RuntimeError):
    def __init__(self, status: Any, operation: str) -> None:
        name = getattr(status, "name", str(status)).removeprefix("CUEST_STATUS_")
        super().__init__(f"{operation} failed with {name}")
        self.status = status
        self.status_name = name
        self.operation = operation


def check(status: Any, operation: str) -> None:
    if status != _SUCCESS:
        raise CuestError(status, operation)


class MathMode(enum.Enum):
    NATIVE_FP64 = "native-fp64"
    DEFAULT = "default"

    @property
    def raw(self) -> int:
        m = cb.CuestMathMode
        return int(
            m.CUEST_NATIVE_FP64_MATH_MODE
            if self is MathMode.NATIVE_FP64
            else m.CUEST_DEFAULT_MATH_MODE
        )

    @classmethod
    def from_raw(cls, value: int) -> "MathMode":
        if value == int(cb.CuestMathMode.CUEST_NATIVE_FP64_MATH_MODE):
            return cls.NATIVE_FP64
        if value == int(cb.CuestMathMode.CUEST_DEFAULT_MATH_MODE):
            return cls.DEFAULT
        raise ValueError(f"unknown cuEST math mode {value}")


class Parameters:
    """A cuEST parameters object of one type, destroyed on close."""

    def __init__(self, parameters_type: Any) -> None:
        self.type = parameters_type
        self.raw = cb.Parameters()
        check(
            cb.cuestParametersCreate(parametersType=parameters_type, outParameters=self.raw),
            "cuestParametersCreate",
        )
        self._open = True

    def close(self) -> None:
        if self._open:
            self._open = False
            check(
                cb.cuestParametersDestroy(parametersType=self.type, parameters=self.raw),
                "cuestParametersDestroy",
            )

    def __enter__(self) -> "Parameters":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_open", False):
            try:
                self.close()
            except CuestError:
                pass


class Handle:
    """One cuEST handle. Handles never cross processes (spec §9)."""

    def __init__(self, math_mode: MathMode = MathMode.NATIVE_FP64) -> None:
        self._raw = cb.cuestHandle()
        with Parameters(cb.CuestParametersType.CUEST_HANDLE_PARAMETERS) as params:
            check(cb.cuestCreate(parameters=params.raw, handle=self._raw), "cuestCreate")
        self.closed = False
        self.set_math_mode(math_mode)

    @property
    def raw(self) -> Any:
        if self.closed:
            raise CuestError("CLOSED", "handle use after close (handle is closed)")
        return self._raw

    def set_math_mode(self, mode: MathMode) -> None:
        check(cb.cuestSetMathMode(handle=self.raw, mode=mode.raw), "cuestSetMathMode")

    @property
    def math_mode(self) -> MathMode:
        out = cb.data_cuestMathMode_t()
        check(cb.cuestGetMathMode(handle=self.raw, mode=out), "cuestGetMathMode")
        return MathMode.from_raw(int(out.value))

    @property
    def version(self) -> str:
        parts = []
        for fn, name in (
            (cb.cuestGetMajorVersion, "majorVersion"),
            (cb.cuestGetMinorVersion, "minorVersion"),
            (cb.cuestGetPatchVersion, "patchVersion"),
        ):
            out = cb.data_uint32_t()
            check(fn(handle=self.raw, **{name: out}), fn.__name__)
            parts.append(str(int(out.value)))
        return ".".join(parts)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            check(cb.cuestDestroy(handle=self._raw), "cuestDestroy")

    def __enter__(self) -> "Handle":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "closed", True):
            try:
                self.close()
            except CuestError:
                pass


class WorkspaceDescriptor:
    """cuestWorkspaceDescriptor_t: {size_t hostBufferSizeInBytes; size_t deviceBufferSizeInBytes}.

    Layout from the NVIDIA cuEST C API docs (c-api/types/structs, workspace_api.h).
    """

    dtype = np.dtype(
        [("hostBufferSizeInBytes", np.uint64), ("deviceBufferSizeInBytes", np.uint64)], align=True
    )

    def __init__(self, host_bytes: int = 0, device_bytes: int = 0) -> None:
        self.struct = np.array((host_bytes, device_bytes), dtype=self.dtype)

    @property
    def pointer(self) -> int:
        return int(self.struct.ctypes.data)

    @property
    def host_bytes(self) -> int:
        return int(self.struct["hostBufferSizeInBytes"])

    @property
    def device_bytes(self) -> int:
        return int(self.struct["deviceBufferSizeInBytes"])


class Workspace:
    """cuestWorkspace_t backed by CuPy device memory and pinned host memory.

    Layout: {uintptr_t hostBuffer; size_t hostBufferSizeInBytes;
             uintptr_t deviceBuffer; size_t deviceBufferSizeInBytes}.
    """

    dtype = np.dtype(
        [
            ("hostBuffer", np.uintp),
            ("hostBufferSizeInBytes", np.uint64),
            ("deviceBuffer", np.uintp),
            ("deviceBufferSizeInBytes", np.uint64),
        ],
        align=True,
    )

    def __init__(self, descriptor: WorkspaceDescriptor) -> None:
        self.struct = np.zeros((), dtype=self.dtype)
        self._device: Any = None
        self._host: Any = None
        if descriptor.device_bytes:
            self._device = cupy.cuda.alloc(descriptor.device_bytes)
            self.struct["deviceBuffer"] = self._device.ptr
            self.struct["deviceBufferSizeInBytes"] = descriptor.device_bytes
        if descriptor.host_bytes:
            self._host = cupy.cuda.alloc_pinned_memory(descriptor.host_bytes)
            self.struct["hostBuffer"] = self._host.ptr
            self.struct["hostBufferSizeInBytes"] = descriptor.host_bytes

    @property
    def pointer(self) -> int:
        return int(self.struct.ctypes.data)

    def release(self) -> None:
        cupy.cuda.Device().synchronize()
        self._device = None
        self._host = None
        self.struct["deviceBuffer"] = 0
        self.struct["hostBuffer"] = 0
        self.struct["deviceBufferSizeInBytes"] = 0
        self.struct["hostBufferSizeInBytes"] = 0


def device_pointer(array: Any, shape: Sequence[int]) -> Any:
    """cuEST Pointer to a float64, C-contiguous CuPy array of exactly `shape`."""
    if not isinstance(array, cupy.ndarray):
        raise ValueError("cuEST buffers must be CuPy device arrays")
    if array.dtype != cupy.float64:
        raise ValueError(f"cuEST buffers must be float64, got {array.dtype}")
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"buffer shape {tuple(array.shape)} != required {tuple(shape)}")
    if not array.flags.c_contiguous:
        raise ValueError("cuEST buffers must be C-contiguous")
    p = cb.Pointer()
    p.value = np.intp(array.data.ptr)
    return p


def run_with_workspace(
    operation: str,
    query: Callable[..., Any],
    compute: Callable[..., Any],
    *,
    outputs: Sequence[Any] = (),
    **arguments: Any,
) -> None:
    """Query the temporary workspace, allocate it, and compute, as one operation.

    The query and the compute receive identical arguments, which keeps the
    buffer sizes unchanged between them (the XC hazard in spec §8). On any
    failure the outputs are overwritten with NaN before raising.
    """
    descriptor = WorkspaceDescriptor()
    try:
        check(
            query(temporaryWorkspaceDescriptor=descriptor.pointer, **arguments),
            f"{operation}WorkspaceQuery",
        )
        workspace = Workspace(descriptor)
        try:
            check(compute(temporaryWorkspace=workspace.pointer, **arguments), operation)
        finally:
            workspace.release()
    except Exception:
        for out in outputs:
            out.fill(np.nan)
        raise


class AOShell:
    """One AO shell. Contraction coefficients must arrive normalized (spec §8)."""

    def __init__(
        self,
        handle: Handle,
        *,
        is_pure: bool,
        angular_momentum: int,
        exponents: Sequence[float],
        coefficients: Sequence[float],
    ) -> None:
        if len(exponents) != len(coefficients):
            raise ValueError("exponents and coefficients differ in length")
        self.raw = cb.cuestAOShellHandle()
        self.is_pure = bool(is_pure)
        self.angular_momentum = int(angular_momentum)
        self._open = False
        with Parameters(cb.CuestParametersType.CUEST_AOSHELL_PARAMETERS) as params:
            check(
                cb.cuestAOShellCreate(
                    handle=handle.raw,
                    isPure=int(is_pure),
                    L=int(angular_momentum),
                    numPrimitive=len(exponents),
                    exponents=[float(e) for e in exponents],
                    coefficients=[float(c) for c in coefficients],
                    parameters=params.raw,
                    outShell=self.raw,
                ),
                "cuestAOShellCreate",
            )
        self._open = True

    def close(self) -> None:
        if self._open:
            self._open = False
            check(cb.cuestAOShellDestroy(handle=self.raw), "cuestAOShellDestroy")

    def __del__(self) -> None:
        if getattr(self, "_open", False):
            try:
                self.close()
            except CuestError:
                pass
