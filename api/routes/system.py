"""Health, system information and run history."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from api.deps import Services, services
from utils.device import get_device_context

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request, svc: Annotated[Services, Depends(services)]) -> dict:
    cache = svc.pipeline.cache.load()
    return {
        "status": "ok",
        "storage": svc.settings.storage.backend,
        "cache_version": cache.meta.version if cache else None,
        "model_backend": svc.settings.model.backend,
        "model_loaded": svc.pipeline._engine is not None,
        "dataset_id": svc.dataset_id,
        "active_model": request.app.state.runtime.active_model,
    }


@router.get("/api/v1/system/info")
async def system_info(svc: Annotated[Services, Depends(services)]) -> dict:
    def collect() -> dict:
        cache = svc.pipeline.cache.load()
        engine = None
        try:
            engine = svc.pipeline.engine.describe()
        except Exception as exc:  # model not trained yet
            engine = {"error": str(exc)}
        m = svc.settings.model
        return {
            "app": svc.settings.app_name,
            "environment": svc.settings.environment,
            "device": get_device_context(m.amp_dtype, m.cuda_memory_fraction).describe(),
            "storage": {"backend": svc.settings.storage.backend, "tables": svc.pipeline.storage.table_counts()},
            "vector_cache": cache.meta.__dict__ if cache else None,
            "engine": engine,
            "sse_subscribers": svc.broker.subscribers,
        }

    return await run_in_threadpool(collect)


@router.get("/api/v1/runs")
async def recent_runs(svc: Annotated[Services, Depends(services)], limit: int = 20) -> list[dict]:
    return await run_in_threadpool(svc.pipeline.alerts.recent_runs, min(max(limit, 1), 500))
