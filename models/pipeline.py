"""End-to-end EWS scoring: cache -> forecast -> risk/behaviour -> alerts.

Framework-agnostic: the FastAPI layer, the CLI scripts and the Ray batch
scorer all call :class:`EWSPipeline`.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pyarrow as pa

from config.settings import QUANTILES, Settings
from data_engine.aggregation import FeatureAggregator
from data_engine.alerts import AlertStore, utcnow
from data_engine.schema import EWS_ALERTS, conform
from data_engine.storage import StorageAdapter, get_storage
from data_engine.vector_cache import CIBIL_CH, PAST_CHANNELS, LoadedCache, VectorCache
from models.behavior import BehaviorResult, evaluate_behavior
from models.engine import ForecastEngine, get_engine
from models.risk import AccountSnapshot, RiskResult, evaluate_risk
from utils.logging import logger

MISSED_CH = PAST_CHANNELS.index("missed_salary_flag")
RETURN_CH = PAST_CHANNELS.index("payment_return_flag")
MAX_UTILIZATION = 1.1


@dataclass
class ScoreRequest:
    account_ids: list[str] | None = None
    horizon: int = 90
    as_of_date: date | None = None
    liability_quantile: float = 0.1
    asset_quantile: float = 0.9
    return_quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    include_paths: bool = True
    persist_alerts: bool = True
    auto_refresh: bool = True


@dataclass
class ScoredBatch:
    """Vectorised scoring output for a set of cache rows."""

    rows: np.ndarray
    as_of: date
    horizon: int
    q: np.ndarray  # [B, 2, H, 9]
    known_future: np.ndarray  # [B, H, 3]
    past: np.ndarray  # [B, L, 6]
    snapshot: AccountSnapshot
    risk: RiskResult
    behavior: BehaviorResult


def _f(x: Any, nd: int = 2) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def _s(x: Any) -> str | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def _day(x: int) -> int | None:
    return int(x) if x > 0 else None


class EWSPipeline:
    def __init__(
        self,
        settings: Settings,
        storage: StorageAdapter | None = None,
        cache: VectorCache | None = None,
        engine: ForecastEngine | None = None,
    ):
        self.settings = settings
        self.storage = storage or get_storage(settings)
        self.cache = cache or VectorCache(settings.cache.root)
        self._engine = engine
        self.alerts = AlertStore(self.storage)
        self._refresh_lock = threading.Lock()

    @property
    def engine(self) -> ForecastEngine:
        if self._engine is None:
            self._engine = get_engine(self.settings)
        return self._engine

    # ---- freshness ------------------------------------------------------------------------------
    def refresh(self, full: bool = False, force_cache: bool = False) -> dict[str, Any]:
        """Incremental feature aggregation + cache rebuild when the data moved."""
        with self._refresh_lock:
            agg = FeatureAggregator(self.storage).refresh(full=full)
            rebuilt = None
            if force_cache or agg.mode != "noop" or self.cache.is_stale(self.storage):
                pad = -(-self.settings.model.max_horizon // self.settings.model.patch_length) * (
                    self.settings.model.patch_length
                )
                rebuilt = self.cache.build(self.storage, horizon_pad=pad, workers=self.settings.worker_threads)
            return {
                "aggregation": agg.__dict__,
                "cache_version": rebuilt.version if rebuilt else self.cache.load().meta.version,
                "cache_rebuilt": rebuilt is not None,
            }

    def loaded_cache(self, auto_refresh: bool) -> LoadedCache:
        if auto_refresh and self.cache.is_stale(self.storage):
            self.refresh()
        cache = self.cache.load()
        if cache is None:
            raise RuntimeError("vector cache not built yet: ingest data then POST /api/v1/aggregate/refresh")
        return cache

    # ---- vectorised scoring ------------------------------------------------------------------------
    def score_rows(
        self,
        cache: LoadedCache,
        rows: np.ndarray,
        as_of: date,
        horizon: int,
        liability_quantile: float,
        asset_quantile: float,
    ) -> ScoredBatch:
        eng = self.engine
        L, Hp = eng.context_length, eng.horizon_pad
        win = cache.window(rows, as_of, L, Hp)
        past, known = win["past"], win["known"]
        acc = cache.accounts.iloc[rows]
        q = eng.forecast(past, known, horizon, has_credit_line=acc["credit_limit"].fillna(0).to_numpy(float) > 0)
        # Physical bounds: deposit balances that never went negative stay >= 0;
        # utilisation lives in [0, max_utilization] (small over-limit allowed).
        nonneg = ~(np.nan_to_num(past[:, :, 0], nan=0.0) < 0).any(1)
        q[:, 0] = np.where(nonneg[:, None, None], np.maximum(q[:, 0], 0.0), q[:, 0])
        q[:, 1] = np.clip(q[:, 1], 0.0, MAX_UTILIZATION)

        last = past[:, -1, :]
        cib = past[:, :, CIBIL_CH]
        snap = AccountSnapshot(
            current_balance=np.nan_to_num(last[:, 0]),
            current_utilization=np.nan_to_num(last[:, 1]),
            credit_limit=acc["credit_limit"].fillna(0).to_numpy(float),
            outstanding_principal=acc["outstanding_principal"].fillna(0).to_numpy(float),
            lgd=acc["lgd"].fillna(0.45).to_numpy(float),
            cibil=np.nan_to_num(cib[:, -1], nan=700.0),
            missed_salary=np.nan_to_num(last[:, MISSED_CH]),
            payment_return=np.nan_to_num(last[:, RETURN_CH]),
            cibil_change_90d=np.nan_to_num(cib[:, -1] - cib[:, -91]),
        )
        known_future = known[:, L : L + horizon]
        risk = evaluate_risk(
            q[:, 0], q[:, 1], known_future[..., 0], snap, self.settings.risk, liability_quantile, asset_quantile
        )
        stress = (snap.missed_salary > 0) | (snap.payment_return > 0)
        behavior = evaluate_behavior(
            past[:, :, 0], q[:, 0], stress, self.settings.behavior, QUANTILES.index(liability_quantile)
        )
        return ScoredBatch(rows, as_of, horizon, q, known_future, past, snap, risk, behavior)

    # ---- request-level API ------------------------------------------------------------------------------
    def score(self, req: ScoreRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        cache = self.loaded_cache(req.auto_refresh)
        rows, unknown = cache.rows_for(req.account_ids)
        as_of = req.as_of_date or cache.end
        run_id = "run_" + uuid.uuid4().hex[:12]
        if len(rows) == 0:
            return {"run_id": run_id, "as_of_date": as_of, "horizon": req.horizon, "accounts": [],
                    "unknown_accounts": unknown, "summary": self._summary([], 0, t0)}

        sb = self.score_rows(cache, rows, as_of, req.horizon, req.liability_quantile, req.asset_quantile)
        accounts = self._account_payloads(cache, sb, req)
        alerts = self.alert_table(cache, sb, run_id)
        if req.persist_alerts:
            scope = cache.accounts["account_id"].iloc[rows].tolist()
            self.alerts.record_run(self._run_row(run_id, as_of, req, len(rows), alerts), scope, alerts)
        alert_rows = alerts.to_pylist()
        return {
            "run_id": run_id,
            "as_of_date": as_of,
            "horizon": req.horizon,
            "backend": self.engine.name,
            "accounts": accounts,
            "unknown_accounts": unknown,
            "alerts": alert_rows,
            "summary": self._summary(accounts, len(alert_rows), t0),
        }

    def _run_row(self, run_id: str, as_of: date, req: ScoreRequest, n: int, alerts: pa.Table) -> dict[str, Any]:
        el = alerts.column("expected_loss").to_pylist() if alerts.num_rows else []
        return {
            "run_id": run_id,
            "created_at": utcnow(),
            "as_of_date": as_of,
            "horizon": req.horizon,
            "backend": self.engine.name,
            "n_accounts": n,
            "n_alerts": alerts.num_rows,
            "total_expected_loss": float(sum(x for x in el if x is not None)),
            "params": json.dumps(
                {"liability_quantile": req.liability_quantile, "asset_quantile": req.asset_quantile,
                 "accounts": None if req.account_ids is None else len(req.account_ids)}
            ),
        }

    def alert_table(self, cache: LoadedCache, sb: ScoredBatch, run_id: str) -> pa.Table:
        r, bh = sb.risk, sb.behavior
        acc = cache.accounts.iloc[sb.rows]
        ids = acc["account_id"].to_numpy()
        rm = np.array([_s(x) for x in acc["rm_id"]], dtype=object)
        credit = r.tier != "GREEN"
        attr = bh.attrition_alert
        now = utcnow()
        parts = []
        for mask, kind in ((credit, "CREDIT_EWS"), (attr, "DEPOSIT_ATTRITION")):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            dtb = r.days_to_breach[idx]
            dtz = r.days_to_zero[idx]
            ned = r.next_emi_day[idx]
            parts.append(
                {
                    "alert_id": [f"{run_id}-{a}-{kind}" for a in ids[idx]],
                    "run_id": [run_id] * idx.size,
                    "account_id": ids[idx].tolist(),
                    "rm_id": rm[idx].tolist(),
                    "created_at": [now] * idx.size,
                    "as_of_date": [sb.as_of] * idx.size,
                    "alert_type": [kind] * idx.size,
                    "risk_tier": r.tier[idx].tolist(),
                    "priority_score": (r.priority[idx] if kind == "CREDIT_EWS" else bh.cof_impact_annual[idx]).tolist(),
                    "current_balance": sb.snapshot.current_balance[idx].tolist(),
                    "exposure_at_default": r.ead[idx].tolist(),
                    "probability_of_default": r.pd[idx].tolist(),
                    "loss_given_default": r.lgd[idx].tolist(),
                    # EL lives on the credit alert only, so summing alerts never double counts
                    "expected_loss": [None if kind != "CREDIT_EWS" or not np.isfinite(x) else float(x)
                                      for x in r.expected_loss[idx]],
                    "days_to_breach": [_day(x) for x in dtb],
                    "breach_date": [sb.as_of + timedelta(days=int(x)) if x > 0 else None for x in dtb],
                    "days_to_zero": [_day(x) for x in dtz],
                    "next_emi_amount": r.next_emi_amount[idx].tolist(),
                    "next_emi_date": [sb.as_of + timedelta(days=int(x)) if x > 0 else None for x in ned],
                    "projected_balance_drop": bh.projected_balance_drop[idx].tolist(),
                    "cof_impact_annual": bh.cof_impact_annual[idx].tolist(),
                    "triggers": [json.dumps(r.triggers[i]) for i in idx],
                    "status": ["PENDING"] * idx.size,
                }
            )
        if not parts:
            return EWS_ALERTS.arrow_schema.empty_table()
        merged = {k: sum((p[k] for p in parts), []) for k in parts[0]}
        return conform(EWS_ALERTS, pa.table(merged))

    def _account_payloads(self, cache: LoadedCache, sb: ScoredBatch, req: ScoreRequest) -> list[dict[str, Any]]:
        r, bh, sn = sb.risk, sb.behavior, sb.snapshot
        acc = cache.accounts.iloc[sb.rows].reset_index(drop=True)
        dates = [(sb.as_of + timedelta(days=i + 1)).isoformat() for i in range(sb.horizon)]
        qidx = [QUANTILES.index(q) for q in req.return_quantiles]
        out = []
        for i in range(len(sb.rows)):
            in_sma = r.tier[i] != "GREEN"
            item: dict[str, Any] = {
                "account_id": acc.at[i, "account_id"],
                "rm_id": _s(acc.at[i, "rm_id"]),
                "segment": _s(acc.at[i, "segment"]),
                "product_type": _s(acc.at[i, "product_type"]),
                "risk_tier": r.tier[i],
                "days_to_breach": _day(r.days_to_breach[i]),
                "breach_date": (sb.as_of + timedelta(days=int(r.days_to_breach[i]))).isoformat()
                if r.days_to_breach[i] > 0 else None,
                "breach_shortfall": _f(r.breach_shortfall[i]) if in_sma else None,
                "days_to_zero": _day(r.days_to_zero[i]),
                "current_balance": _f(sn.current_balance[i]),
                "current_utilization": _f(sn.current_utilization[i], 4),
                "next_emi": {
                    "date": (sb.as_of + timedelta(days=int(r.next_emi_day[i]))).isoformat(),
                    "amount": _f(r.next_emi_amount[i]),
                } if r.next_emi_day[i] > 0 else None,
                "probability_of_default": _f(r.pd[i], 4),
                "pd_components": {
                    "cash_shortfall": _f(r.pd_cash[i], 4),
                    "utilization_breach": _f(r.p_util[i], 4),
                    "bureau": _f(r.pd_bureau[i], 4),
                },
                "exposure_at_default": _f(r.ead[i]),
                "loss_given_default": _f(r.lgd[i], 4),
                "expected_loss": _f(r.expected_loss[i]) if in_sma else None,
                "priority_score": _f(r.priority[i]),
                "cibil_score": _f(sn.cibil[i], 0),
                "triggers": r.triggers[i],
                "behavior": {
                    "trailing_avg_balance_90d": _f(bh.trailing_avg_90d[i]),
                    "baseline_avg_balance": _f(bh.baseline_avg[i]),
                    "projected_avg_balance": _f(bh.projected_avg[i]),
                    "projected_change_pct": _f(bh.projected_change_pct[i] * 100, 2),
                    "projected_min_balance_worst_case": _f(bh.projected_min_worst[i]),
                    "trend": bh.trend[i],
                    "attrition_alert": bool(bh.attrition_alert[i]),
                    "projected_balance_drop": _f(bh.projected_balance_drop[i]),
                    "cof_impact_annual": _f(bh.cof_impact_annual[i]),
                },
            }
            if req.include_paths:
                item["forecast"] = {
                    "dates": dates,
                    "balance": {f"q{int(round(QUANTILES[k] * 100))}": np.round(sb.q[i, 0, :, k], 2).tolist() for k in qidx},
                    "utilization": {
                        f"q{int(round(QUANTILES[k] * 100))}": np.round(sb.q[i, 1, :, k], 4).tolist() for k in qidx
                    },
                    "emi_schedule": [
                        {"date": dates[d], "amount": _f(sb.known_future[i, d, 0])}
                        for d in np.flatnonzero(sb.known_future[i, :, 0] > 0)
                    ],
                }
            out.append(item)
        return out

    @staticmethod
    def _summary(accounts: list[dict[str, Any]], n_alerts: int, t0: float) -> dict[str, Any]:
        tiers = {t: 0 for t in ("GREEN", "SMA-0", "SMA-1", "SMA-2")}
        el = ead_sma = 0.0
        attr = 0
        cof = 0.0
        for a in accounts:
            tiers[a["risk_tier"]] += 1
            if a["risk_tier"] != "GREEN":
                el += a["expected_loss"] or 0.0
                ead_sma += a["exposure_at_default"] or 0.0
            if a["behavior"]["attrition_alert"]:
                attr += 1
                cof += a["behavior"]["cof_impact_annual"] or 0.0
        return {
            "accounts_scored": len(accounts),
            "tier_counts": tiers,
            "sma_exposure_at_default": round(ead_sma, 2),
            "total_expected_loss": round(el, 2),
            "attrition_alerts": attr,
            "attrition_cof_impact_annual": round(cof, 2),
            "alerts_raised": n_alerts,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


_pipeline: EWSPipeline | None = None
_lock = threading.Lock()


def get_pipeline(settings: Settings) -> EWSPipeline:
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = EWSPipeline(settings)
            logger.info("EWS pipeline initialised")
        return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    with _lock:
        _pipeline = None
