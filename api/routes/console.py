"""Console endpoints: database connection, data import, background jobs, overview.

These change what the service reads and writes, so they only answer requests
from this machine unless ``admin_allow_remote`` is set (put real auth in front
before you do that).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api import connection as conn
from api.deps import Services, services
from data_engine import importer
from data_engine.ingestion import IngestionEngine
from data_engine.storage import get_storage
from models.engine import reset_engine
from models.pipeline import EWSPipeline
from models.train import cpu_profile, train_model

_LOCAL = {"127.0.0.1", "::1", "localhost", "testclient"}
UPLOAD_SUFFIXES = {".csv": "csv", ".txt": "csv", ".parquet": "parquet", ".pq": "parquet"}


def local_only(request: Request) -> None:
    svc: Services = request.app.state.services
    host = request.client.host if request.client else ""
    if not svc.settings.admin_allow_remote and host not in _LOCAL:
        raise HTTPException(403, "console admin endpoints are limited to this machine (admin_allow_remote=false)")


router = APIRouter(prefix="/api/v1", tags=["console"], dependencies=[Depends(local_only)])
Svc = Annotated[Services, Depends(services)]


# ------------------------------------------------------------------------------------- overview
@router.get("/console/overview", summary="Everything the console home page shows")
async def overview(svc: Svc) -> dict[str, Any]:
    def collect() -> dict[str, Any]:
        p = svc.pipeline
        cache = p.cache.load()
        model: dict[str, Any] = {"backend": svc.settings.model.backend,
                                 "checkpoint": str(svc.settings.model.checkpoint_path),
                                 "exists": Path(svc.settings.model.checkpoint_path).exists()}
        if model["exists"] or svc.settings.model.backend != "stformer":
            try:
                model.update(p.engine.describe())
            except Exception as exc:
                model["error"] = str(exc)
        book = None
        runs = [r for r in p.alerts.recent_runs(20) if r.get("scope") == "BOOK"
                and r.get("horizon") == svc.settings.book_alert_horizon]
        if runs:
            r = runs[0]
            tiers = p.storage.query_arrow(
                "SELECT risk_tier, COUNT(*) AS n, SUM(expected_loss) AS el FROM account_scores "
                "WHERE run_id = :r GROUP BY risk_tier", {"r": r["run_id"]}).to_pylist()
            book = {"run_id": r["run_id"], "scored_at": r["created_at"], "horizon": r["horizon"],
                    "accounts": r["n_accounts"], "alerts": r["n_alerts"],
                    "total_expected_loss": r["total_expected_loss"],
                    "up_to_date": bool(cache and r.get("cache_version") == cache.meta.version),
                    "tiers": {t["risk_tier"]: t["n"] for t in tiers}}
        return {
            "connection": conn.describe(svc.settings),
            "tables": p.storage.table_counts(),
            "data": None if cache is None else {"accounts": cache.n_accounts, "first_day": cache.start,
                                                "last_day": cache.end, "cache_version": cache.meta.version},
            "model": model,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "latest_book_run": book,
        }

    return await run_in_threadpool(collect)


# ------------------------------------------------------------------------------------ connection
@router.get("/console/connection", summary="Current database (password hidden)")
async def get_connection(svc: Svc) -> dict[str, Any]:
    return conn.describe(svc.settings)


@router.post("/console/connection/test", summary="Check a database and create the EWS tables in it")
async def test_connection(spec: conn.ConnectionSpec, svc: Svc) -> dict[str, Any]:
    try:
        return await run_in_threadpool(conn.test_connection, conn.settings_for(svc.settings, spec))
    except Exception as exc:
        raise HTTPException(400, f"could not connect: {type(exc).__name__}: {exc}") from exc


@router.post("/console/connection", summary="Switch the service to another database and remember it")
async def switch_connection(spec: conn.ConnectionSpec, request: Request, svc: Svc) -> dict[str, Any]:
    new_settings = conn.settings_for(svc.settings, spec)

    def build() -> Services:
        conn.test_connection(new_settings)
        storage = get_storage(new_settings)
        return Services(
            settings=new_settings,
            pipeline=EWSPipeline(new_settings, storage=storage),
            ingestion=IngestionEngine(storage, batch_rows=new_settings.storage.stream_batch_rows),
            broker=svc.broker,
            jobs=svc.jobs,
        )

    try:
        new = await run_in_threadpool(build)
    except Exception as exc:
        raise HTTPException(400, f"could not connect: {type(exc).__name__}: {exc}") from exc
    request.app.state.services = new
    conn.save(new_settings, spec)
    job = None
    if new.pipeline.storage.scalar("SELECT COUNT(*) FROM daily_balances"):
        job = new.jobs.submit("refresh", "Refresh features and score the book after switching database",
                              new.pipeline.refresh_and_rescore).view()
    return {"connection": conn.describe(new_settings), "job": job}


# ---------------------------------------------------------------------------------------- import
@router.get("/import/tables", summary="Columns of every importable table")
async def import_tables() -> dict[str, Any]:
    return importer.describe_tables()


@router.get("/import/template/{table}", response_class=PlainTextResponse, summary="CSV template for a table")
async def import_template(table: str) -> PlainTextResponse:
    try:
        body = importer.template_csv(table)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return PlainTextResponse(body, media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{table}_template.csv"'})


def _spool(svc: Services) -> Path:
    d = svc.settings.upload_spool_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean_spool(d: Path, max_age_s: int = 24 * 3600) -> None:
    now = time.time()
    for f in d.glob("upl_*"):
        if now - f.stat().st_mtime > max_age_s:
            f.unlink(missing_ok=True)


@router.post("/import/upload", summary="Upload a CSV or Parquet file and preview it")
async def import_upload(svc: Svc, file: Annotated[UploadFile, File()], table: str = "daily_balances") -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    fmt = UPLOAD_SUFFIXES.get(suffix)
    if fmt is None:
        raise HTTPException(422, f"unsupported file type '{suffix}'; upload .csv or .parquet")
    spool = _spool(svc)
    _clean_spool(spool)
    upload_id = "upl_" + uuid.uuid4().hex[:12]
    dest = spool / f"{upload_id}{'.csv' if fmt == 'csv' else '.parquet'}"
    sha = hashlib.sha1()

    def save() -> int:
        size = 0
        with dest.open("wb") as out:
            while chunk := file.file.read(8 << 20):
                sha.update(chunk)
                out.write(chunk)
                size += len(chunk)
        return size

    size = await run_in_threadpool(save)
    await file.close()
    try:
        reader = importer.csv_columns_and_sample if fmt == "csv" else importer.parquet_columns_and_sample
        columns, sample = await run_in_threadpool(reader, str(dest))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"could not read the file: {exc}") from exc
    (spool / f"{upload_id}.sha1").write_text(sha.hexdigest())
    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "format": fmt,
        "bytes": size,
        "columns": columns,
        "sample": sample,
        "suggested_mapping": importer.suggest_mapping(table, columns) if table in importer.INGESTIBLE_TABLES else {},
    }


class SuggestRequest(BaseModel):
    table: str
    columns: list[str]


@router.post("/import/suggest", summary="Suggest a column mapping for a table")
async def import_suggest(req: SuggestRequest) -> dict[str, Any]:
    try:
        return {"mapping": importer.suggest_mapping(req.table, req.columns)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


class SqlSource(BaseModel):
    url: str = Field(description="SQLAlchemy URL of YOUR source database, e.g. "
                     "postgresql+psycopg2://user:pass@host/core or sqlite:///path/to/file.db")
    query: str = Field(description="One SELECT statement")


@router.post("/import/sql/preview", summary="Run a SELECT on a source database and preview the rows")
async def import_sql_preview(src: SqlSource, table: str = "daily_balances") -> dict[str, Any]:
    try:
        columns, sample = await run_in_threadpool(importer.sql_columns_and_sample, src.url, src.query)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, f"query failed: {type(exc).__name__}: {exc}") from exc
    return {"columns": columns, "sample": sample,
            "suggested_mapping": importer.suggest_mapping(table, columns) if table in importer.INGESTIBLE_TABLES else {}}


class CommitRequest(BaseModel):
    table: str
    upload_id: str | None = None
    sql: SqlSource | None = None
    mapping: dict[str, str | None]
    constants: dict[str, Any] = {}
    date_format: str | None = Field(None, description='strptime format, e.g. "%d/%m/%Y"; empty = ISO')
    signed_amounts: bool = True
    decimal_comma: bool = False
    refresh_after: bool = Field(True, description="Refresh features and rescore the book afterwards")


@router.post("/import/commit", summary="Import an uploaded file or a SQL result into a table (background job)")
async def import_commit(req: CommitRequest, svc: Svc, wait: bool = False) -> dict[str, Any]:
    if (req.upload_id is None) == (req.sql is None):
        raise HTTPException(422, "give exactly one source: upload_id or sql")
    opts = importer.ImportOptions(
        table=req.table, mapping=req.mapping, constants=req.constants, date_format=req.date_format or None,
        signed_amounts=req.signed_amounts, decimal_comma=req.decimal_comma,
    )
    if req.upload_id:
        spool = _spool(svc)
        files = [p for p in spool.glob(f"{req.upload_id}.*") if p.suffix in (".csv", ".parquet")]
        if not req.upload_id.startswith("upl_") or not files:
            raise HTTPException(404, "upload not found (uploads are kept for 24 hours)")
        path = files[0]
        sha_file = spool / f"{req.upload_id}.sha1"
        opts.source_key = sha_file.read_text() if sha_file.exists() else req.upload_id
        if path.suffix == ".csv":
            columns, _ = await run_in_threadpool(importer.csv_columns_and_sample, str(path))
            make_batches = lambda: importer.csv_batches(str(path), columns)  # noqa: E731
        else:
            columns, _ = await run_in_threadpool(importer.parquet_columns_and_sample, str(path))
            make_batches = lambda: importer.parquet_batches(str(path))  # noqa: E731
        label = f"Import {path.name} into {req.table}"
    else:
        try:
            columns, _ = await run_in_threadpool(importer.sql_columns_and_sample, req.sql.url, req.sql.query)
        except Exception as exc:
            raise HTTPException(400, f"query failed: {type(exc).__name__}: {exc}") from exc
        opts.source_key = hashlib.sha1(f"{req.sql.url}|{req.sql.query}".encode()).hexdigest()
        make_batches = lambda: importer.sql_batches(req.sql.url, req.sql.query)  # noqa: E731
        label = f"Import SQL query into {req.table}"
    try:
        problems = opts.validate(columns)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if problems:
        raise HTTPException(422, {"problems": problems})

    pipeline, ingestion, refresh = svc.pipeline, svc.ingestion, req.refresh_after

    def run() -> dict[str, Any]:
        res = importer.run_import(ingestion, make_batches(), opts)
        out: dict[str, Any] = {"table": res.table, "rows": res.rows,
                               "first_date": res.min_event_date, "last_date": res.max_event_date}
        if refresh and pipeline.storage.scalar("SELECT COUNT(*) FROM daily_balances"):
            out["refresh"] = pipeline.refresh_and_rescore()
        return out

    job = svc.jobs.submit("import", label, run)
    if wait:
        job = await run_in_threadpool(svc.jobs.wait, job.id)
    return {"job": job.view()}


# ------------------------------------------------------------------------------------------ jobs
@router.get("/jobs", summary="Recent background jobs")
async def list_jobs(svc: Svc) -> list[dict[str, Any]]:
    return svc.jobs.list()


@router.get("/jobs/{job_id}", summary="One background job")
async def get_job(job_id: str, svc: Svc) -> dict[str, Any]:
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.view()


class TrainRequest(BaseModel):
    epochs: int = Field(6, ge=1, le=200)


@router.post("/jobs/{kind}", summary="Start a background job: refresh, score-book or train")
async def start_job(
    kind: Literal["refresh", "score-book", "train"], svc: Svc, train: TrainRequest | None = None, wait: bool = False
) -> dict[str, Any]:
    p, settings = svc.pipeline, svc.settings
    if kind == "refresh":
        job = svc.jobs.submit("refresh", "Refresh features, rebuild the cache and score the book",
                              p.refresh_and_rescore)
    elif kind == "score-book":
        job = svc.jobs.submit("score-book", "Score the whole book (30/60/90 days)", lambda: p.score_book(force=True))
    else:
        epochs = (train or TrainRequest()).epochs

        def do_train() -> dict[str, Any]:
            p.refresh()
            m = settings.model if torch.cuda.is_available() else cpu_profile(settings.model)
            m = m.model_copy(update={"epochs": epochs})
            result = train_model(p.cache.current_dir(), m, checkpoint_path=settings.model.checkpoint_path)
            reset_engine()
            p._engine = None  # pick up the new weights
            return {"training": result.__dict__, "book": p.score_book(force=True)}

        job = svc.jobs.submit("train", f"Train the forecaster ({epochs} epochs) and rescore the book", do_train)
    if wait:
        job = await run_in_threadpool(svc.jobs.wait, job.id)
    return {"job": job.view()}

