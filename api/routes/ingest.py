"""Ingestion endpoints: JSON batches, streamed file uploads, server-side datasets."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Annotated, Literal

import pyarrow as pa
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from api.deps import Services, services
from api.schemas import TABLE_FIELDS, DatasetIngestRequest, IngestRequest, IngestResponse, TableIngestStats
from data_engine.ingestion import IngestResult
from data_engine.schema import ALL_TABLES, INGESTIBLE_TABLES, conform
from utils.logging import logger

router = APIRouter(prefix="/api/v1", tags=["ingestion"])
IngestTable = Literal["accounts", "daily_balances", "transactions", "credit_profiles", "loan_schedules", "holidays"]


def _stats(r: IngestResult) -> TableIngestStats:
    return TableIngestStats(rows=r.rows, min_event_date=r.min_event_date, max_event_date=r.max_event_date)


async def _refresh(svc: Services, wait: bool, background: BackgroundTasks) -> tuple[str, dict | None]:
    if wait:
        result = await run_in_threadpool(svc.pipeline.refresh)
        background.add_task(_safe_rescore, svc)  # the book rescore still runs after the response
        return "completed", result
    background.add_task(_safe_refresh, svc)
    return "scheduled", None


def _safe_refresh(svc: Services) -> None:
    try:
        svc.pipeline.refresh_and_rescore()
    except Exception:  # background task: log, never crash the worker
        logger.exception("background refresh / book rescore failed")


def _safe_rescore(svc: Services) -> None:
    if not svc.settings.rescore_after_refresh:
        return
    try:
        svc.pipeline.score_book()
    except Exception:
        logger.exception("background book rescore failed")


@router.post("/ingest", response_model=IngestResponse, summary="Ingest a daily batch (JSON)")
async def ingest_batch(
    payload: IngestRequest, background: BackgroundTasks, svc: Annotated[Services, Depends(services)]
) -> IngestResponse:
    """Validates and upserts account master data, EOD balances, transactions,
    bureau pulls, EMI schedules and holidays. Tables are loaded in dependency
    order (master and calendar data first), each as one Arrow batch."""
    t0 = time.perf_counter()
    batch_id = payload.batch_id or "batch_" + uuid.uuid4().hex[:12]

    def load() -> dict[str, TableIngestStats]:
        out: dict[str, TableIngestStats] = {}
        for name in TABLE_FIELDS:
            records = getattr(payload, name)
            if not records:
                continue
            tbl = conform(ALL_TABLES[name], pa.Table.from_pylist([r.model_dump() for r in records]))
            out[name] = _stats(svc.ingestion.ingest(name, tbl))
        return out

    tables = await run_in_threadpool(load)
    status, result = ("skipped", None)
    if payload.refresh_features:
        status, result = await _refresh(svc, payload.wait_for_refresh, background)
    return IngestResponse(
        batch_id=batch_id,
        tables=tables,
        refresh=status,
        refresh_result=result,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@router.post("/ingest/file", response_model=IngestResponse, summary="Stream a CSV / Parquet / Arrow file into a table")
async def ingest_file(
    background: BackgroundTasks,
    svc: Annotated[Services, Depends(services)],
    file: Annotated[UploadFile, File(description="CSV, Parquet or Arrow IPC file")],
    table: Annotated[IngestTable, Query()],
    format: Annotated[Literal["csv", "parquet", "ipc"] | None, Query()] = None,
    refresh_features: bool = True,
    wait_for_refresh: bool = False,
) -> IngestResponse:
    """The upload is spooled to disk by the server and parsed by PyArrow's
    multi-threaded streaming readers in bounded record batches, so a
    multi-GB ledger never has to fit in memory."""
    t0 = time.perf_counter()
    fmt = format or {".csv": "csv", ".parquet": "parquet", ".arrow": "ipc", ".feather": "ipc"}.get(
        Path(file.filename or "").suffix.lower()
    )
    if fmt is None:
        raise HTTPException(422, "cannot infer file format; pass ?format=csv|parquet|ipc")
    try:
        result = await run_in_threadpool(svc.ingestion.ingest_file, table, file.file, fmt)
    except (pa.ArrowInvalid, ValueError, KeyError) as exc:
        raise HTTPException(422, f"could not ingest file: {exc}") from exc
    finally:
        await file.close()
    status, res = ("skipped", None)
    if refresh_features:
        status, res = await _refresh(svc, wait_for_refresh, background)
    return IngestResponse(
        batch_id="file_" + uuid.uuid4().hex[:12],
        tables={table: _stats(result)},
        refresh=status,
        refresh_result=res,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@router.post("/ingest/dataset", response_model=IngestResponse, summary="Bulk-load a server-side Parquet/CSV dataset")
async def ingest_dataset(
    req: DatasetIngestRequest, background: BackgroundTasks, svc: Annotated[Services, Depends(services)]
) -> IngestResponse:
    """For terabyte-scale backfills already on the server's storage: a
    multi-file, multi-threaded Arrow dataset scan streamed into the
    database. Paths are restricted to the configured data directory."""
    t0 = time.perf_counter()
    root = svc.settings.data_dir.resolve()
    target = (root / req.path).resolve() if not Path(req.path).is_absolute() else Path(req.path).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(403, "path must be inside the server data directory")
    if not target.exists():
        raise HTTPException(404, f"{req.path} not found")
    if req.table not in INGESTIBLE_TABLES:
        raise HTTPException(422, "table not ingestible")
    try:
        result = await run_in_threadpool(svc.ingestion.ingest_dataset, req.table, target, req.format)
    except (pa.ArrowInvalid, ValueError) as exc:
        raise HTTPException(422, f"could not ingest dataset: {exc}") from exc
    status, res = ("skipped", None)
    if req.refresh_features:
        status, res = await _refresh(svc, False, background)
    return IngestResponse(
        batch_id="dataset_" + uuid.uuid4().hex[:12],
        tables={req.table: _stats(result)},
        refresh=status,
        refresh_result=res,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@router.post("/aggregate/refresh", summary="Run the incremental nightly aggregation + cache rebuild")
async def aggregate_refresh(
    svc: Annotated[Services, Depends(services)], background: BackgroundTasks, full: bool = False
) -> dict:
    result = await run_in_threadpool(svc.pipeline.refresh, full, full)
    background.add_task(_safe_rescore, svc)
    return result
