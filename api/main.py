"""FastAPI application: Banking EWS & Loan Loss Predictor.

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Every blocking job (DuckDB/Postgres I/O, aggregation, PyTorch inference) runs
on a worker thread, so the event loop only handles HTTP and SSE.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from api.broker import AlertBroker
from api.deps import Services
from api.routes import alerts, ingest, predict, system
from config.settings import Settings, get_settings
from data_engine.ingestion import IngestionEngine
from data_engine.storage import get_storage
from models.pipeline import EWSPipeline
from utils.logging import logger, setup_logging


def _warm(pipeline: EWSPipeline) -> None:
    try:
        removed = pipeline.alerts.cleanup_orphans()
        if any(removed.values()):
            logger.info("removed rows of unfinished runs: {}", removed)
        has_data = pipeline.storage.scalar("SELECT COUNT(*) FROM daily_balances")
        if has_data:
            pipeline.refresh()
        _ = pipeline.engine  # load model weights once, up front
        if has_data and pipeline.settings.rescore_after_refresh:
            pipeline.score_book()  # no-op when a stored run already matches data + model
    except Exception as exc:  # the API still serves ingestion without a model
        logger.warning("warm-up incomplete: {}", exc)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = max(limiter.total_tokens, settings.worker_threads * 4)
        storage = get_storage(settings)
        pipeline = EWSPipeline(settings, storage=storage)
        app.state.services = Services(
            settings=settings,
            pipeline=pipeline,
            ingestion=IngestionEngine(storage, batch_rows=settings.storage.stream_batch_rows),
            broker=AlertBroker(),
        )
        if settings.warm_cache_on_startup:
            await run_in_threadpool(_warm, pipeline)
        logger.info("{} ready (storage={}, model={})", settings.app_name, settings.storage.backend, settings.model.backend)
        yield

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Early Warning System for retail & SME lending: multivariate spatial-temporal "
            "forecasting of balances and credit-line utilisation, SMA-0/1/2 classification, "
            "PD x EAD x LGD expected-loss estimation and deposit-attrition alerts."
        ),
        lifespan=lifespan,
    )
    for r in (system.router, ingest.router, predict.router, alerts.router):
        app.include_router(r)
    return app


app = create_app()
