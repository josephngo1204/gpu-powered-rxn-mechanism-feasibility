import threading
import time
from pathlib import Path

import pytest

from saddle.store.tasks import Task, TaskError, TaskStore


def t(kind: str, n: int = 0, deps: tuple[str, ...] = ()) -> Task:
    return Task(
        kind=kind, config={"n": n, "profile": "search"}, geometry_hash="ab" * 32, depends_on=deps
    )


def test_task_id_is_content_addressed_and_order_independent() -> None:
    a = Task("hessian", {"x": 1, "y": 2}, "h")
    b = Task("hessian", {"y": 2, "x": 1}, "h")
    assert a.task_id == b.task_id
    assert a.task_id != Task("hessian", {"x": 1, "y": 3}, "h").task_id


def test_task_config_is_immutable() -> None:
    task = t("scf")
    with pytest.raises(TypeError):
        task.config["n"] = 5  # type: ignore[index]


def test_submit_is_idempotent(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    a = store.submit(t("scf"))
    assert store.submit(t("scf")) == a
    assert store.counts() == {"pending": 1}


def test_claim_respects_dependencies(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    first = store.submit(t("opt"))
    second = store.submit(t("hessian", deps=(first,)))
    claimed = store.claim("gpu0")
    assert claimed is not None and claimed.task_id == first
    assert store.claim("gpu1") is None
    store.complete(first, "gpu0", "runs/opt.h5")
    nxt = store.claim("gpu1")
    assert nxt is not None and nxt.task_id == second


def test_unknown_dependency_rejected(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    with pytest.raises(TaskError, match="unknown"):
        store.submit(t("hessian", deps=("nope",)))


def test_retries_then_failure_is_kept(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite", max_attempts=2)
    tid = store.submit(t("scf"))
    for _ in range(2):
        claimed = store.claim("gpu0")
        assert claimed is not None
        store.fail(tid, "gpu0", "SCF did not converge")
    assert store.claim("gpu0") is None
    info = store.status(tid)
    assert info.state == "failed"
    assert info.attempts == 2
    assert "SCF" in (info.error or "")


def test_only_owner_can_complete(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite")
    tid = store.submit(t("scf"))
    store.claim("gpu0")
    with pytest.raises(TaskError, match="not running on"):
        store.complete(tid, "gpu1", "x.h5")


def test_restart_requeues_stale_running_tasks(tmp_path: Path) -> None:
    path = tmp_path / "tasks.sqlite"
    store = TaskStore(path)
    tid = store.submit(t("scf"))
    store.claim("gpu0")
    reopened = TaskStore(path)
    assert reopened.status(tid).state == "running"
    time.sleep(0.01)
    assert reopened.requeue_stale(older_than_s=0.0) == [tid]
    claimed = reopened.claim("gpu1")
    assert claimed is not None and claimed.task_id == tid


def test_concurrent_workers_never_double_claim(tmp_path: Path) -> None:
    path = tmp_path / "tasks.sqlite"
    store = TaskStore(path)
    for i in range(40):
        store.submit(t("stencil", i))
    claimed: list[str] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        own = TaskStore(path)
        while (task := own.claim(name)) is not None:
            with lock:
                claimed.append(task.task_id)
            own.complete(task.task_id, name, f"{task.task_id}.h5")

    threads = [threading.Thread(target=worker, args=(f"gpu{i}",)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(claimed) == 40
    assert len(set(claimed)) == 40
    assert TaskStore(path).counts() == {"done": 40}
