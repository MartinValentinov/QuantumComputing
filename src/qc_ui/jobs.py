"""A minimal background job runner.

Ingesting a disease takes minutes and an agent question can take longer. Both
are far past what a browser will wait on a single request, so they run on a
thread and the page polls a job id.

In-memory on purpose: jobs are ephemeral, and persisting them would mean
inventing a second store alongside the graph for no benefit.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"          # running | done | error
    progress: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress[-40:],
            "result": self.result,
            "error": self.error,
        }


class JobRunner:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn: Callable[[Callable[[str], None]], Any]) -> Job:
        """Run `fn` on a thread. `fn` receives a `log` callable for progress."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job

        def log(line: str) -> None:
            job.progress.append(line)

        def target() -> None:
            try:
                job.result = fn(log)
                job.status = "done"
            except Exception as e:
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.progress.append(traceback.format_exc(limit=3))

        threading.Thread(target=target, daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)
