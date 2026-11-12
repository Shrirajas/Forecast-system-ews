"""Console endpoints: database connection, data import, background jobs, overview.

These change what the service reads and writes, so they only answer requests
from this machine unless ``admin_allow_remote`` is set (put real auth in front
before you do that).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
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
from api.deps import Services, local_only, services
from api import compare_store
from config.registry import dataset_info, match_dataset
from models.risk import tier_label
from utils.timefmt import to_ist
from data_engine import importer
from data_engine.ingestion import IngestionEngine
from data_engine.storage import get_storage
from models.engine import get_engine
from models.pipeline import EWSPipeline
from models.train import cpu_profile, train_model

UPLOAD_SUFFIXES = {".csv": "csv", ".txt": "csv", ".parquet": "parquet", ".pq": "parquet"}


router = APIRouter(prefix="/api/v1", tags=["console"], dependencies=[Depends(local_only)])
Svc = Annotated[Services, Depends(services)]


# ------------------------------------------------------------------------------------- overview
@router.get("/console/overview", summary="Everything the console home page shows")
async def overview(request: Request, svc: Svc) -> dict[str, Any]:
    rt = request.app.state.runtime

    def collect() -> dict[str, Any]:
        p = svc.pipeline
        spec = rt.registry.get(svc.dataset_id or "")
        cache = p.cache.load()
        model: dict[str, Any] = {"backend": svc.settings.model.backend,
                                 "checkpoint": str(svc.settings.model.checkpoint_path),
                                 "exists": Path(svc.settings.model.checkpoint_path).exists()}
        if model["exists"] or svc.settings.model.backend != "stformer":
            try:
                model.update(p.engine.describe())
                model["trained_at"] = to_ist(model.get("trained_at"), naive="ist")  # checkpoints store local time
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
            book = {"run_id": r["run_id"], "scored_at": to_ist(r["created_at"]), "horizon": r["horizon"],
                    "accounts": r["n_accounts"], "alerts": r["n_alerts"],
                    "total_expected_loss": r["total_expected_loss"],
                    "up_to_date": bool(cache and r.get("cache_version") == cache.meta.version),
                    "tiers": {t["risk_tier"]: t["n"] for t in tiers}}
        counts = p.alerts.counts()
        alert_counts = {"active": counts.get("ACTIVE", 0), "acknowledged": counts.get("ACKNOWLEDGED", 0),
                        "superseded": counts.get("SUPERSEDED", 0), "pending": counts.get("PENDING", 0),
                        "total_rows_all_runs": sum(counts.values())}
        extra = _kpis(svc, spec, book, alert_counts, runs, cache)
        return {
            **extra,
            "model_status": rt.model_info(rt.active_model),
            "alert_counts": alert_counts,
            "tables_note": "Row counts of the stored tables. ews_alerts counts the rows of ALL runs "
                           "(active and superseded); the headline alert count is alert_counts.active.",
            "connection": conn.describe(svc.settings),
            "tables": p.storage.table_counts(),
            "data": None if cache is None else {"accounts": cache.n_accounts, "first_day": cache.start,
                                                "last_day": cache.end, "cache_version": cache.meta.version},
            "model": model,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "latest_book_run": book,
        }

    return await run_in_threadpool(collect)


def _kpis(svc: Services, spec, book: dict | None, ac: dict, runs: list, cache) -> dict[str, Any]:
    """Overview extras. ``None`` = unavailable (never 0)."""
    cfg, p = svc.settings.risk, svc.pipeline
    thr = cfg.high_risk_pd_threshold
    dist, high, note = None, None, None
    if book:
        rid = {"r": book["run_id"]}
        if spec is None or spec.capabilities.get("projected_sma"):
            t = book["tiers"]
            dist = [{"id": k, "label": tier_label(k), "count": int(t.get(k, 0)), "tone": tone}
                    for k, tone in (("SMA-2", "red"), ("SMA-1", "orange"), ("SMA-0", "yellow"), ("GREEN", "green"))]
            note = "Projected tiers are forecasts, not regulatory classifications. GREEN is not a guarantee of low credit risk."
        else:
            rows = p.storage.query_arrow(
                "SELECT CASE WHEN probability_of_default IS NULL THEN 'na' WHEN probability_of_default > :t THEN 'high' "
                "WHEN probability_of_default > :t / 2 THEN 'mid' ELSE 'low' END AS b, COUNT(*) AS n "
                "FROM account_scores WHERE run_id = :r GROUP BY 1", {**rid, "t": thr}).to_pylist()
            n = {r["b"]: int(r["n"]) for r in rows}
            dist = [{"id": "high", "label": f"PD above {thr:.0%}", "count": n.get("high", 0), "tone": "red"},
                    {"id": "mid", "label": f"PD {thr / 2:.0%} to {thr:.0%}", "count": n.get("mid", 0), "tone": "orange"},
                    {"id": "low", "label": f"PD below {thr / 2:.0%}", "count": n.get("low", 0), "tone": "green"},
                    {"id": "na", "label": "No credit exposure (N/A)", "count": n.get("na", 0), "tone": "na"}]
            note = "PD is a bureau-based estimate, not calibrated; this is not a regulatory classification."
        hr = int(p.storage.scalar(
            "SELECT COUNT(*) FROM account_scores WHERE run_id = :r AND expected_loss IS NOT NULL "
            "AND (risk_tier <> 'GREEN' OR probability_of_default > :t)", {**rid, "t": thr}) or 0)
        high = {"supported": True, "count": hr,
                "definition": f"credit products with a projected SMA tier, or PD above {thr:.0%}"}
    days: dict[str, dict[str, int]] = {}
    for r in p.alerts.recent_runs(200):
        d = (to_ist(r["created_at"]) or "")[:10]
        e = days.setdefault(d, {"runs": 0, "accounts": 0})
        e["runs"] += 1
        e["accounts"] += int(r["n_accounts"] or 0)
    saved = compare_store.list_saved(spec) if spec else []
    return {
        "timezone": "IST (UTC+05:30)",
        "dataset": dataset_info(spec, svc.dataset_id) if spec else None,
        "kpis": {"total_accounts": None if cache is None else cache.n_accounts,
                 "accounts_forecasted": book["accounts"] if book else None,
                 "high_risk": high or {"supported": False, "count": None, "definition": None},
                 "active_alerts": ac["active"] if book else None,
                 "last_forecast_at": book["scored_at"] if book else None},
        "risk_distribution": dist, "risk_distribution_note": note,
        "activity": [{"date": d, **v} for d, v in sorted(days.items())][-14:],
        "recent_comparison": saved[0] if saved else None,
    }


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
            dataset_id=match_dataset(new_settings, request.app.state.runtime.registry),
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
    #: Cap optimiser steps per epoch so a large book trains in minutes (None = every window).
    steps_per_epoch: int | None = Field(None, ge=1, le=100_000)


LOSS_NOTE = ("Training error is averaged over each epoch with dropout on; the check error is measured in eval mode on "
             "accounts held out of training, so the training error can sit above the check error.")


@router.get("/console/training", summary="Live progress of the current / last training run")
async def training_progress(request: Request, svc: Svc) -> dict[str, Any]:
    rt = request.app.state.runtime
    ck = Path(rt.training_ckpt or svc.settings.model.checkpoint_path)
    path = ck.with_suffix(".progress.json")
    meta = {"dataset_id": svc.dataset_id, "model_id": "stformer", "loss_note": LOSS_NOTE}
    if not path.exists():
        return {"status": "none", "checkpoint": str(ck), **meta}
    out = json.loads(path.read_text())
    out["started_at"] = to_ist(out.get("started_at"), naive="ist")
    return {**out, **meta}


@router.post("/jobs/{kind}", summary="Start a background job: refresh, score-book or train")
async def start_job(
    kind: Literal["refresh", "score-book", "train"], request: Request, svc: Svc, train: TrainRequest | None = None, wait: bool = False
) -> dict[str, Any]:
    p, settings = svc.pipeline, svc.settings
    if kind == "refresh":
        job = svc.jobs.submit("refresh", "Refresh features, rebuild the cache and score the book",
                              p.refresh_and_rescore)
    elif kind == "score-book":
        job = svc.jobs.submit("score-book", "Score the whole book (30/60/90 days)", lambda: p.score_book(force=True))
    else:
        rt = request.app.state.runtime
        if not rt.models[rt.active_model].supports_training:
            raise HTTPException(422, {"code": "training_not_supported", "message":
                                      f"{rt.models[rt.active_model].display_name} is a pretrained zero-shot model: there is nothing to train"})
        req = train or TrainRequest()
        epochs = req.epochs
        # a NEW file every run: existing checkpoints are never overwritten
        base = rt.registry[svc.dataset_id].checkpoint_path if svc.dataset_id else Path(settings.model.checkpoint_path)
        ckpt = base.with_name(f"{base.stem}_{datetime.now().strftime('%Y%m%dT%H%M%S')}{base.suffix}")
        rt.training_ckpt = str(ckpt)

        def do_train() -> dict[str, Any]:
            p.refresh()
            m = settings.model if torch.cuda.is_available() else cpu_profile(settings.model)
            m = m.model_copy(update={"epochs": epochs})
            try:
                result = train_model(p.cache.current_dir(), m, checkpoint_path=ckpt, max_steps_per_epoch=req.steps_per_epoch)
            except Exception as e:
                prog = ckpt.with_suffix(".progress.json")
                if prog.exists():  # tell the Training tab instead of leaving it on "running"
                    prog.write_text(json.dumps({**json.loads(prog.read_text()), "status": "failed", "error": str(e)}))
                raise
            if svc.dataset_id:  # remember the newest checkpoint of this dataset (resolve_checkpoint reads it)
                spec = rt.registry[svc.dataset_id]
                spec.checkpoint_path.with_name(f"{spec.id}_stformer.latest.json").write_text(json.dumps({"checkpoint": str(ckpt)}))
            settings.model.checkpoint_path = ckpt
            p._engine = None  # pick up the new weights
            get_engine(settings, "stformer", reload=True)
            rt.engine_reloaded()
            return {"training": result.__dict__, "book": p.score_book(force=True)}

        job = svc.jobs.submit("train", f"Train the forecaster ({epochs} epochs) and rescore the book", do_train)
    if wait:
        job = await run_in_threadpool(svc.jobs.wait, job.id)
    return {"job": job.view()}

