"""Single-account forecast for the active dataset and a chosen model (never persists alerts)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from api.presentation import account_display
from api.runtime import Runtime, err
from api.schemas_v2 import ForecastOut, ForecastRequest
from config.registry import DatasetSpec
from models.engine import EngineUnavailable
from models.pipeline import ScoreRequest
from models.risk import monotone_quantiles
from utils.timefmt import now_ist

router = APIRouter(prefix="/api/v1", tags=["forecast"])
QN = ("q10", "q50", "q90")
CH = {"eod_balance": "balance", "utilization": "utilization"}


def _series(path: dict[str, list[float]], warnings: list[str], label: str) -> dict[str, list[float]]:
    q = np.array([path[k] for k in QN], float).T  # [H, 3]
    fixed, n = monotone_quantiles(q)
    if n:
        warnings.append(f"{label}: quantiles crossed on {n} day(s) and were re-ordered")
    return {k: fixed[:, i].round(4).tolist() for i, k in enumerate(QN)}


def _target_out(t, spec: DatasetSpec) -> dict[str, Any]:
    return {**t.model_dump(exclude={"applies_to"}), "currency": spec.currency.code if t.unit == spec.currency.code else None,
            "applies_to": list(t.applies_to)}


def run_forecast(rt: Runtime, req: ForecastRequest) -> dict[str, Any]:
    svc = rt.svc
    rt.require_active(req.dataset_id)
    model_id = req.model_id or rt.active_model
    rt.check_supported(model_id, svc.dataset_id)
    spec = rt.registry.get(svc.dataset_id or "")
    if spec is None:
        raise err(422, "dataset_unavailable", "forecasting needs a registered dataset; select one first")
    target = spec.target(req.target)
    if target is None or not target.available:
        raise err(422, "unsupported_target", (target.reason if target else f"unknown target '{req.target}'"),
                  targets=[t.id for t in spec.targets if t.available])
    pipe = svc.pipeline
    cache = pipe.loaded_cache(False)
    rows, _ = cache.rows_for([req.account_id])
    if len(rows) == 0:
        raise err(404, "unknown_account", f"account '{req.account_id}' is not in the {spec.display_name} dataset")
    acc = cache.accounts.iloc[int(rows[0])]
    has_line = bool(acc["credit_limit"] and acc["credit_limit"] > 0)
    if target.kind == "secondary" and target.applies_to and acc["product_type"] not in target.applies_to:
        raise err(422, "unsupported_target", f"'{target.label}' does not apply to {acc['product_type']} accounts",
                  applies_to=list(target.applies_to))
    if target.channel == "utilization" and not has_line:
        raise err(422, "unsupported_target", "the account has no credit line, so utilisation is not forecast")
    warnings: list[str] = []
    try:
        with svc.busy():
            out = pipe.score(ScoreRequest(account_ids=[req.account_id], horizon=req.horizon, as_of_date=req.as_of_date,
                                          backend=model_id, paths="all", persist_alerts=False))
            as_of = out["as_of_date"]
            n_hist = min(req.history_days, pipe.engine_for(model_id).context_length)
            win = cache.window(rows, as_of, n_hist, pipe.engine_for(model_id).horizon_pad)["past"][0]
    except EngineUnavailable as exc:
        raise err(503, "model_unavailable", str(exc)) from exc
    a = out["accounts"][0]
    fc = a.pop("forecast")
    a["display"] = account_display(a, spec, svc.settings.risk, req.horizon)
    dates = fc["dates"]
    hist_dates = [as_of - timedelta(days=n_hist - 1 - i) for i in range(n_hist)]

    def hist(ch: int) -> dict[str, list]:
        return {"dates": hist_dates, "values": [None if np.isnan(v) else round(float(v), 4) for v in win[:, ch]]}

    def block(t) -> dict[str, Any]:
        ch = 0 if t.channel == "eod_balance" else 1
        return {"target": _target_out(t, spec), "history": hist(ch),
                "forecast": {"dates": dates, "quantiles": _series(fc[CH[t.channel]], warnings, t.label)}}

    main = block(target)
    other = next((t for t in spec.targets if t.id != target.id and t.available and t.channel != target.channel
                  and (not t.applies_to or acc["product_type"] in t.applies_to)
                  and (t.channel != "utilization" or has_line)), None)
    med = np.array(main["forecast"]["quantiles"]["q50"])
    hv = [v for v in main["history"]["values"] if v is not None]
    hint = a["display"]["trend"] if target.channel == "eod_balance" else None
    eng = pipe.engine_for(model_id)
    m = rt.models[model_id]
    return {
        "dataset": {"id": spec.id, "display_name": spec.display_name, "currency": spec.currency.model_dump()},
        "model": {"id": m.id, "display_name": m.display_name, "backend": out["backend"], "version": out["model_version"],
                  "device": out.get("device"), "kind": m.kind, "badge": m.badge},
        "account": {"account_id": acc["account_id"], "product_type": acc["product_type"], "segment": acc["segment"],
                    "rm_id": acc["rm_id"], "credit_limit": float(acc["credit_limit"]) if has_line else None,
                    "open_date": acc["open_date"]},
        "target": main["target"], "horizon": req.horizon, "origin": as_of, "quantile_levels": [0.1, 0.5, 0.9],
        "history": main["history"], "forecast": main["forecast"], "secondary": None if other is None else block(other),
        "emi_schedule": fc["emi_schedule"] if spec.capabilities.get("emi") else None,
        "summary": {"current_value": hv[-1] if hv else None, "forecast_day1_median": float(med[0]),
                    "forecast_mean_median": float(med.mean()), "forecast_end_median": float(med[-1]),
                    "interval_end": [main["forecast"]["quantiles"]["q10"][-1], main["forecast"]["quantiles"]["q90"][-1]],
                    "trend": hint},
        "risk": a, "risk_unavailable_reason": None,
        "description": (f"{target.label} ({target.unit}): {target.description} Forecast for the next {req.horizon} days after "
                        f"{as_of}; the line is the median (q50) and the band the q10-q90 range."),
        "timing": {"forecasted_at": now_ist(), "forecast_ms": out["summary"]["latency_ms"], "timezone": "IST"},
        "warnings": warnings, "persisted": False,
    }


@router.post("/forecast", response_model=ForecastOut, summary="Forecast one account with the chosen model")
async def forecast(req: ForecastRequest, request: Request) -> dict:
    return await run_in_threadpool(run_forecast, request.app.state.runtime, req)
