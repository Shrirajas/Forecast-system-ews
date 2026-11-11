"""Background jobs for the console: imports, refresh, book scoring, training.

Jobs run one at a time on a single worker thread, in submission order, so two
long writes never fight over the database. Status is kept in memory (the last
100 jobs) and polled by the UI.
"""

from __future__ import annotations

import queue
import threading
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from utils.logging import logger
from utils.timefmt import to_ist


@dataclass
class Job:
    id: str
    kind: str
    description: str
    status: str = "queued"  # queued / running / done / failed
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: str | None = None
    finished_at: str | None = None
    result: Any = None
    error: str | None = None

    def view(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("submitted_at", "started_at", "finished_at"):  # naive local (IST) -> explicit +05:30
            d[k] = to_ist(d[k], naive="ist")
        return d


class JobManager:
    def __init__(self, keep: int = 100):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._fns: dict[str, Callable[[], Any]] = {}
        self._done: dict[str, threading.Event] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._keep = keep
        self._worker = threading.Thread(target=self._run, name="ews-jobs", daemon=True)
        self._worker.start()

    def submit(self, kind: str, description: str, fn: Callable[[], Any]) -> Job:
        job = Job(id="job_" + uuid.uuid4().hex[:10], kind=kind, description=description)
        with self._lock:
            self._jobs[job.id] = job
            self._fns[job.id] = fn
            self._done[job.id] = threading.Event()
            while len(self._jobs) > self._keep:
                old_id, old = next(iter(self._jobs.items()))
                if old.status in ("queued", "running"):
                    break
                self._jobs.pop(old_id)
                self._done.pop(old_id, None)
        self._q.put(job.id)
        return job

    def active(self) -> list[Job]:
        """Jobs that are queued or running (a dataset switch is refused while any exist)."""
        with self._lock:
            return [j for j in self._jobs.values() if j.status in ("queued", "running")]

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [j.view() for j in reversed(self._jobs.values())]

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """Block until a job finishes (used by tests and ?wait=true)."""
        self._done[job_id].wait(timeout)
        return self._jobs[job_id]

    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            job, fn = self._jobs.get(job_id), self._fns.pop(job_id, None)
            if job is None or fn is None:
                continue
            job.status, job.started_at = "running", datetime.now().isoformat(timespec="seconds")
            try:
                job.result = fn()
                job.status = "done"
            except Exception as exc:  # report, never kill the worker
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                logger.error("job {} ({}) failed:\n{}", job.id, job.kind, traceback.format_exc())
            finally:
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                self._done[job_id].set()
