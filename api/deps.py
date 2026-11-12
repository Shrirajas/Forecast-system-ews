"""Shared service objects, created once per process in the app lifespan."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from fastapi import HTTPException, Request

from api.broker import AlertBroker
from api.jobs import JobManager
from config.settings import Settings
from data_engine.ingestion import IngestionEngine
from models.pipeline import EWSPipeline

_LOCAL = {"127.0.0.1", "::1", "localhost", "testclient"}


@dataclass
class Services:
    settings: Settings
    pipeline: EWSPipeline
    ingestion: IngestionEngine
    broker: AlertBroker
    jobs: JobManager
    #: Registered dataset this service points at (None = a custom database chosen in the console).
    dataset_id: str | None = None
    inflight: int = 0
    retired: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @contextmanager
    def busy(self) -> Iterator[None]:
        """Mark a scoring/forecast call in flight: a dataset switch waits for none, and a call that
        arrives after the switch is refused instead of touching the closed database."""
        with self._lock:
            if self.retired:
                raise HTTPException(409, {"code": "dataset_switched", "message": "the dataset was switched; retry"})
            self.inflight += 1
        try:
            yield
        finally:
            with self._lock:
                self.inflight -= 1

    def retire(self) -> bool:
        """Close the door to new calls; False (and nothing changes) when a call is in flight."""
        with self._lock:
            if self.inflight:
                return False
            self.retired = True
            return True


def services(request: Request) -> Services:
    return request.app.state.services


def local_only(request: Request) -> None:
    svc: Services = request.app.state.services
    host = request.client.host if request.client else ""
    if not svc.settings.admin_allow_remote and host not in _LOCAL:
        raise HTTPException(403, "console admin endpoints are limited to this machine (admin_allow_remote=false)")
