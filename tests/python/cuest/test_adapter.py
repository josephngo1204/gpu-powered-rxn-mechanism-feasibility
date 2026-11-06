import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from saddle.cuest import (
    AOShell,
    CuestError,
    Handle,
    MathMode,
    Workspace,
    WorkspaceDescriptor,
    check,
    device_pointer,
    run_with_workspace,
)
from saddle.cuest.bindings import cb

pytestmark = pytest.mark.gpu


def test_workspace_struct_layout_matches_nvidia_docs() -> None:
    assert WorkspaceDescriptor.dtype.itemsize == 16
    assert Workspace.dtype.itemsize == 32
    assert Workspace.dtype.names == (
        "hostBuffer",
        "hostBufferSizeInBytes",
        "deviceBuffer",
        "deviceBufferSizeInBytes",
    )


def test_check_maps_status_to_error() -> None:
    check(cb.CuestStatus.CUEST_STATUS_SUCCESS, "noop")
    with pytest.raises(CuestError, match="INVALID_ARGUMENT") as info:
        check(cb.CuestStatus.CUEST_STATUS_INVALID_ARGUMENT, "cuestFoo")
    assert info.value.operation == "cuestFoo"


def test_handle_lifecycle_and_math_mode() -> None:
    with Handle(MathMode.NATIVE_FP64) as h:
        assert h.math_mode is MathMode.NATIVE_FP64
        assert h.version == "0.2.2"
        h.set_math_mode(MathMode.DEFAULT)
        assert h.math_mode is MathMode.DEFAULT
    assert h.closed
    h.close()
    with pytest.raises(CuestError, match="closed"):
        _ = h.raw


def test_real_cuest_failure_raises() -> None:
    with Handle(MathMode.NATIVE_FP64) as h, pytest.raises(CuestError) as info:
        AOShell(h, is_pure=True, angular_momentum=0, exponents=[], coefficients=[])
    assert info.value.status_name in {"NULL_POINTER", "INVALID_ARGUMENT"}


def test_valid_shell_is_created_and_destroyed() -> None:
    with Handle(MathMode.NATIVE_FP64) as h:
        shell = AOShell(
            h, is_pure=True, angular_momentum=1, exponents=[1.0, 0.3], coefficients=[0.6, 0.5]
        )
        shell.close()
        shell.close()


def test_workspace_allocation_and_release() -> None:
    pool = cupy.get_default_memory_pool()
    empty = Workspace(WorkspaceDescriptor(0, 0))
    assert int(empty.struct["deviceBuffer"]) == 0 and int(empty.struct["hostBuffer"]) == 0
    before = pool.used_bytes()
    ws = Workspace(WorkspaceDescriptor(4096, 1 << 20))
    assert int(ws.struct["deviceBuffer"]) != 0 and int(ws.struct["hostBuffer"]) != 0
    assert pool.used_bytes() >= before + (1 << 20)
    ws.release()
    assert pool.used_bytes() == before
    assert int(ws.struct["deviceBuffer"]) == 0


def test_device_pointer_validates_shape_dtype_and_layout() -> None:
    a = cupy.zeros((4, 4))
    p = device_pointer(a, (4, 4))
    assert p.value == a.data.ptr
    with pytest.raises(ValueError, match="shape"):
        device_pointer(a, (4, 5))
    with pytest.raises(ValueError, match="float64"):
        device_pointer(a.astype(cupy.float32), (4, 4))
    with pytest.raises(ValueError, match="contiguous"):
        device_pointer(a[:, ::2], (4, 2))
    with pytest.raises(ValueError, match="device"):
        device_pointer(np.zeros((4, 4)), (4, 4))  # type: ignore[arg-type]


def test_failed_compute_invalidates_outputs() -> None:
    out = cupy.ones((3, 3))

    def query(**kw: object) -> object:
        return cb.CuestStatus.CUEST_STATUS_SUCCESS

    def compute(**kw: object) -> object:
        return cb.CuestStatus.CUEST_STATUS_INVALID_SIZE

    with pytest.raises(CuestError, match="INVALID_SIZE"):
        run_with_workspace("cuestFake", query, compute, outputs=[out])
    assert bool(cupy.isnan(out).all())


def test_successful_compute_keeps_outputs() -> None:
    out = cupy.ones(2)
    calls: dict[str, object] = {}

    def query(**kw: object) -> object:
        calls["query"] = kw
        return cb.CuestStatus.CUEST_STATUS_SUCCESS

    def compute(**kw: object) -> object:
        calls["compute"] = kw
        return cb.CuestStatus.CUEST_STATUS_SUCCESS

    run_with_workspace("cuestFake", query, compute, outputs=[out], plan="p")
    assert "temporaryWorkspaceDescriptor" in calls["query"]  # type: ignore[operator]
    assert "temporaryWorkspace" in calls["compute"]  # type: ignore[operator]
    assert calls["compute"]["plan"] == "p"  # type: ignore[index]
    assert bool((out == 1).all())
