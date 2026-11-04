"""Risk inference & forecasting endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from api.deps import Services, services
from api.schemas import PredictRequest, PredictResponse
from models.pipeline import ScoreRequest

router = APIRouter(prefix="/api/v1", tags=["prediction"])


@router.post("/predict/ews", response_model=PredictResponse, summary="Run the EWS forecast for a batch of accounts")
async def predict_ews(req: PredictRequest, svc: Annotated[Services, Depends(services)]) -> PredictResponse:
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
    sreq = ScoreRequest(
        account_ids=req.account_ids,
        horizon=req.horizon,
        as_of_date=req.as_of_date,
        liability_quantile=req.quantile_triggers.liability,
        asset_quantile=req.quantile_triggers.asset,
        return_quantiles=req.return_quantiles,
        include_paths=req.include_paths,
        persist_alerts=req.persist_alerts,
    )
    try:
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
    if req.tiers:
        wanted = set(req.tiers)
        accounts = [a for a in accounts if a["risk_tier"] in wanted]
    return PredictResponse(
        run_id=out["run_id"],
        as_of_date=out["as_of_date"],
        horizon=out["horizon"],
        backend=out.get("backend"),
        quantile_triggers=req.quantile_triggers,
        summary=out["summary"],
        unknown_accounts=out["unknown_accounts"],
        accounts=accounts,
    )
