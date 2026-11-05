"""SQLite task store (spec §7 "Task", §9 "Execution model").

Tasks are immutable and content-addressed: the id hashes the kind, the
config and the geometry hash, so resubmitting the same work is a no-op.
Workers claim tasks atomically; dependencies must be done first. Failures
are retried up to max_attempts and then kept as 'failed', never deleted.
"""

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task (
    task_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    config TEXT NOT NULL,
    geometry_hash TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'done', 'failed')),
    worker TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    result_path TEXT,
    submitted REAL NOT NULL,
    started REAL,
    finished REAL
);
CREATE TABLE IF NOT EXISTS dependency (
    task_id TEXT NOT NULL REFERENCES task(task_id),
    depends_on TEXT NOT NULL REFERENCES task(task_id),
    PRIMARY KEY (task_id, depends_on)
);
"""


class TaskError(RuntimeError):
    pass


@dataclass(frozen=True)
class Task:
    kind: str
    config: Mapping[str, Any]
    geometry_hash: str
    depends_on: tuple[str, ...] = ()
    task_id: str = field(init=False)

    def __post_init__(self) -> None:
        frozen = MappingProxyType(json.loads(json.dumps(dict(self.config), sort_keys=True)))
        object.__setattr__(self, "config", frozen)
        text = json.dumps(
            {"kind": self.kind, "config": dict(frozen), "geometry": self.geometry_hash},
            sort_keys=True,
            allow_nan=False,
        )
        object.__setattr__(self, "task_id", hashlib.sha256(text.encode()).hexdigest()[:32])


@dataclass(frozen=True)
class TaskStatus:
    task_id: str
    state: str
    worker: str | None
    attempts: int
    error: str | None
    result_path: str | None


class TaskStore:
    def __init__(self, path: str | Path, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self._conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def _tx(self) -> "_Transaction":
        return _Transaction(self._conn)

    def submit(self, task: Task) -> str:
        with self._tx() as c:
            for dep in task.depends_on:
                if c.execute("SELECT 1 FROM task WHERE task_id = ?", (dep,)).fetchone() is None:
                    raise TaskError(f"unknown dependency {dep}")
            c.execute(
                "INSERT OR IGNORE INTO task"
                " (task_id, kind, config, geometry_hash, state, submitted)"
                " VALUES (?, ?, ?, ?, 'pending', ?)",
                (
                    task.task_id,
                    task.kind,
                    json.dumps(dict(task.config), sort_keys=True),
                    task.geometry_hash,
                    time.time(),
                ),
            )
            c.executemany(
                "INSERT OR IGNORE INTO dependency (task_id, depends_on) VALUES (?, ?)",
                [(task.task_id, d) for d in task.depends_on],
            )
        return task.task_id

    def claim(self, worker: str) -> Task | None:
        with self._tx() as c:
            row = c.execute(
                "SELECT task_id, kind, config, geometry_hash FROM task t WHERE state = 'pending'"
                " AND NOT EXISTS (SELECT 1 FROM dependency d"
                "                 JOIN task u ON u.task_id = d.depends_on"
                "                 WHERE d.task_id = t.task_id AND u.state != 'done')"
                " ORDER BY submitted, task_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            task_id, kind, config, geometry_hash = row
            c.execute(
                "UPDATE task SET state = 'running', worker = ?, started = ? WHERE task_id = ?",
                (worker, time.time(), task_id),
            )
            deps = tuple(
                r[0]
                for r in c.execute(
                    "SELECT depends_on FROM dependency WHERE task_id = ?", (task_id,)
                )
            )
        task = Task(kind, json.loads(config), geometry_hash, deps)
        if task.task_id != task_id:
            raise TaskError(f"stored task {task_id} does not match its content")
        return task

    def _require_running(self, c: sqlite3.Connection, task_id: str, worker: str) -> int:
        row = c.execute(
            "SELECT state, worker, attempts FROM task WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None or row[0] != "running" or row[1] != worker:
            raise TaskError(f"task {task_id} is not running on {worker}")
        return int(row[2])

    def complete(self, task_id: str, worker: str, result_path: str) -> None:
        with self._tx() as c:
            self._require_running(c, task_id, worker)
            c.execute(
                "UPDATE task SET state = 'done', result_path = ?, finished = ?, error = NULL"
                " WHERE task_id = ?",
                (result_path, time.time(), task_id),
            )

    def fail(self, task_id: str, worker: str, error: str) -> str:
        with self._tx() as c:
            attempts = self._require_running(c, task_id, worker) + 1
            state = "failed" if attempts >= self.max_attempts else "pending"
            c.execute(
                "UPDATE task SET state = ?, attempts = ?, error = ?, worker = NULL, finished = ?"
                " WHERE task_id = ?",
                (state, attempts, error, time.time(), task_id),
            )
        return state

    def requeue_stale(self, older_than_s: float) -> list[str]:
        """Return running tasks whose worker went silent to the queue (restart)."""
        cutoff = time.time() - older_than_s
        with self._tx() as c:
            rows = c.execute(
                "SELECT task_id FROM task WHERE state = 'running' AND started <= ?", (cutoff,)
            ).fetchall()
            ids = [r[0] for r in rows]
            c.executemany(
                "UPDATE task SET state = 'pending', worker = NULL WHERE task_id = ?",
                [(i,) for i in ids],
            )
        return ids

    def status(self, task_id: str) -> TaskStatus:
        row = self._conn.execute(
            "SELECT task_id, state, worker, attempts, error, result_path"
            " FROM task WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise TaskError(f"unknown task {task_id}")
        return TaskStatus(*row)

    def counts(self) -> dict[str, int]:
        return dict(
            self._conn.execute("SELECT state, COUNT(*) FROM task GROUP BY state").fetchall()
        )

    def close(self) -> None:
        self._conn.close()


class _Transaction:
    """BEGIN IMMEDIATE ... COMMIT/ROLLBACK, so claims are atomic across processes."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self._conn.execute("BEGIN IMMEDIATE")
        return self._conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._conn.execute("COMMIT" if exc_type is None else "ROLLBACK")
