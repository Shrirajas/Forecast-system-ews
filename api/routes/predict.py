"""Risk inference & forecasting endpoint."""

from __future__ import annotations

from typing import Annotated

import orjson
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from api.deps import Services, services
from api.presentation import account_display
from utils.timefmt import to_ist
from api.schemas import PredictRequest, PredictResponse
from models.pipeline import ScoreRequest

router = APIRouter(prefix="/api/v1", tags=["prediction"])


@router.post("/predict/ews", response_model=PredictResponse, summary="Run the EWS forecast for a batch of accounts")
async def predict_ews(req: PredictRequest, request: Request, svc: Annotated[Services, Depends(services)]) -> Response:
    """Runs the multivariate forecaster over the selected accounts (or the
    whole book) and returns, per account:

    * probabilistic balance and utilisation paths (selected quantile tracks),
    * the EWS tier (GREEN / SMA-0 / SMA-1 / SMA-2), days to the first projected
      EMI shortfall on the liability trigger track and days until cash reaches zero,
    * PD (cash-shortfall, utilisation and bureau components), EAD, LGD and
      expected loss (EAD x PD x LGD) for SMA accounts,
    * behaviour metrics and deposit-attrition / cost-of-funds flags.

    Inference runs on a worker thread (GPU autocast when available) so the
    event loop keeps serving other requests.
    """
    if req.include_paths is None:
        paths = "none" if req.account_ids is None else "all"
    else:
        paths = "flagged" if req.include_paths == "flagged" else ("all" if req.include_paths else "none")
    sreq = ScoreRequest(
        account_ids=req.account_ids,
        horizon=req.horizon,
        as_of_date=req.as_of_date,
        liability_quantile=req.quantile_triggers.liability,
        asset_quantile=req.quantile_triggers.asset,
        return_quantiles=req.return_quantiles,
        paths=paths,
        persist_alerts=req.persist_alerts,
        auto_refresh=False,  # refreshes happen after ingest / nightly, never inside a request
    )
    try:
        with svc.busy():
            out = await run_in_threadpool(svc.pipeline.score, sreq)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    if req.persist_alerts and out.get("alerts"):
        svc.broker.publish(out["alerts"])
    accounts = out["accounts"]
    spec = request.app.state.runtime.registry.get(svc.dataset_id or "")
    for a in accounts:
        a["display"] = account_display(a, spec, svc.settings.risk, out["horizon"])
    if req.tiers:
        wanted = set(req.tiers)
        accounts = [a for a in accounts if a["risk_tier"] in wanted]
    body = {
        "run_id": out["run_id"],
        "as_of_date": out["as_of_date"],
        "horizon": out["horizon"],
        "source": out["source"],
        "scored_at": to_ist(out["scored_at"]),
        "cache_version": out["cache_version"],
        "model_version": out["model_version"],
        "backend": out.get("backend"),
        "quantile_triggers": req.quantile_triggers.model_dump(),
        "summary": out["summary"],
        "unknown_accounts": out["unknown_accounts"],
        "accounts": accounts,
    }
    # The pipeline builds this payload from typed, validated columns; serialise it directly
    # (orjson) instead of re-validating thousands of accounts through Pydantic.
    return Response(orjson.dumps(body, option=orjson.OPT_NAIVE_UTC), media_type="application/json")


@router.post("/score/book", summary="Score the whole book now and store it (write path)")
async def score_book(
    svc: Annotated[Services, Depends(services)], background: BackgroundTasks, wait: bool = False, force: bool = False
) -> dict:
    """Runs the nightly whole-book job: one forward pass, stored per horizon, alerts
    from the 90-day run. Whole-book /predict/ews reads are served from it."""
    if wait:
        try:
            return await run_in_threadpool(svc.pipeline.score_book, None, force)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
    background.add_task(svc.pipeline.score_book, None, force)
    return {"status": "scheduled"}
