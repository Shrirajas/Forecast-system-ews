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
from data_engine.schema import ACCOUNT_SCORES, EWS_ALERTS, conform
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
    #: Forecast paths to return: "none", "flagged" (non-GREEN accounts only) or "all".
    paths: str = "all"
    persist_alerts: bool = True
    #: Refresh features/cache inside the call when data moved. Off on the API path:
    #: refreshes run after ingest and in the nightly job, never inside a request.
    auto_refresh: bool = False
    #: Serve whole-book requests from the latest stored book run when it matches.
    use_stored: bool = True
    #: Forecast backend for this request ("stformer" | "timesfm3"); None = the process default.
    #: An unavailable backend raises EngineUnavailable, it is never replaced by another one.
    backend: str | None = None


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


def bound_quantiles(q: np.ndarray, past: np.ndarray) -> np.ndarray:
    """Physical bounds on raw quantiles ``[B, 2, H, 9]``: balances that never went negative stay >= 0,
    utilisation lives in [0, MAX_UTILIZATION] (small over-limit allowed). Returns a bounded copy."""
    q = q.copy()
    nonneg = ~(np.nan_to_num(past[:, :, 0], nan=0.0) < 0).any(1)
    q[:, 0] = np.where(nonneg[:, None, None], np.maximum(q[:, 0], 0.0), q[:, 0])
    q[:, 1] = np.clip(q[:, 1], 0.0, MAX_UTILIZATION)
    return q


def _s(x: Any) -> str | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def _nest(r: dict[str, Any]) -> dict[str, Any]:
    """One flat account_scores row -> the /predict/ews account shape."""
    return {
        "account_id": r["account_id"], "rm_id": r["rm_id"], "segment": r["segment"],
        "product_type": r["product_type"], "risk_tier": r["risk_tier"],
        "days_to_breach": r["days_to_breach"], "breach_date": r["breach_date"],
        "breach_shortfall": r["breach_shortfall"], "days_to_zero": r["days_to_zero"],
        "current_balance": r["current_balance"], "current_utilization": r["current_utilization"],
        "next_emi": {"date": r["next_emi_date"], "amount": r["next_emi_amount"]} if r["next_emi_date"] else None,
        "probability_of_default": r["probability_of_default"],
        "pd_components": {"cash_shortfall": r["pd_cash_shortfall"],
                          "utilization_breach": r["pd_utilization_breach"], "bureau": r["pd_bureau"]},
        "exposure_at_default": r["exposure_at_default"], "loss_given_default": r["loss_given_default"],
        "expected_loss": r["expected_loss"], "priority_score": r["priority_score"], "cibil_score": r["cibil_score"],
        "triggers": json.loads(r["triggers"] or "[]"),
        "behavior": {
            "trailing_avg_balance_90d": r["trailing_avg_balance_90d"],
            "baseline_avg_balance": r["baseline_avg_balance"],
            "projected_avg_balance": r["projected_avg_balance"],
            "projected_change_pct": r["projected_change_pct"],
            "projected_min_balance_worst_case": r["projected_min_balance_worst_case"],
            "trend": r["trend"], "attrition_alert": bool(r["attrition_alert"]),
            "projected_balance_drop": r["projected_balance_drop"], "cof_impact_annual": r["cof_impact_annual"],
        },
    }


def _nan_to_none(a: np.ndarray) -> list[float | None]:
    """N/A (NaN, e.g. no credit exposure) must reach the alert as null, never as NaN (invalid JSON)."""
    return [float(x) if np.isfinite(x) else None for x in a]


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
        self._book_lock = threading.Lock()

    @property
    def engine(self) -> ForecastEngine:
        if self._engine is None:
            self._engine = get_engine(self.settings)
        return self._engine

    def engine_for(self, backend: str | None = None) -> ForecastEngine:
        """The engine for a per-request backend choice (None or the default = ``self.engine``)."""
        if backend is None or backend == self.engine.name:
            return self.engine
        return get_engine(self.settings, backend)

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

    def refresh_and_rescore(self) -> dict[str, Any]:
        """Write path, run after ingest and nightly: refresh, then store a fresh book run."""
        out = self.refresh()
        if self.settings.rescore_after_refresh:
            out["book"] = self.score_book()
        return out

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
        precomputed: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        engine: ForecastEngine | None = None,
    ) -> ScoredBatch:
        eng = engine or self.engine
        L, Hp = eng.context_length, eng.horizon_pad
        acc = cache.accounts.iloc[rows]
        if precomputed is None:
            win = cache.window(rows, as_of, L, Hp)
            past, known = win["past"], win["known"]
            q = eng.forecast(
                past, known, horizon, has_credit_line=acc["credit_limit"].fillna(0).to_numpy(float) > 0
            )
        else:  # a longer forecast already made for these rows: the first `horizon` days are the same
            q_full, past, known = precomputed
            q = q_full[:, :, :horizon].copy()
        q = bound_quantiles(q, past)

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
        """Answer a /predict/ews request.

        Whole-book requests for the latest day are read from the stored book run
        (no forward pass) when one matches the current data, model, horizon and
        triggers; paths for flagged accounts then need a forward pass over those
        accounts only. Everything else is scored live."""
        t0 = time.perf_counter()
        eng = self.engine_for(req.backend)
        cache = self.loaded_cache(req.auto_refresh)
        as_of = req.as_of_date or cache.end
        meta = {"cache_version": cache.meta.version, "model_version": eng.model_version, "backend": eng.name,
                "device": eng.describe().get("device", {}).get("device")}
        if req.account_ids is None and req.use_stored and req.paths != "all" and as_of == cache.end:
            run = self.alerts.latest_book_run(
                cache.meta.version, req.horizon, req.liability_quantile, req.asset_quantile, eng.model_version
            )
            if run is not None:
                return self._from_stored(cache, run, req, t0, meta)

        rows, unknown = cache.rows_for(req.account_ids)
        run_id = "run_" + uuid.uuid4().hex[:12]
        base = {"run_id": run_id, "as_of_date": as_of, "horizon": req.horizon, "source": "live",
                "scored_at": utcnow(), **meta, "unknown_accounts": unknown}
        if len(rows) == 0:
            return {**base, "accounts": [], "alerts": [], "summary": self._summary([], 0, t0)}
        sb = self.score_rows(cache, rows, as_of, req.horizon, req.liability_quantile, req.asset_quantile, engine=eng)
        scores = self.score_table(cache, sb, run_id)
        accounts = [_nest(r) for r in scores.to_pylist()]
        self._attach_paths(accounts, sb, np.arange(len(rows)), req)
        alerts = self.alert_table(cache, sb, run_id)
        # Only the default engine writes alerts: a zero-shot research backend must not feed the alert book.
        if req.persist_alerts and eng is self.engine:
            scope = cache.accounts["account_id"].iloc[rows].tolist()
            self.alerts.record_run(self._run_row(run_id, as_of, req, len(rows), alerts), scope, alerts)
        alert_rows = alerts.to_pylist()
        return {**base, "accounts": accounts, "alerts": alert_rows,
                "summary": self._summary(accounts, len(alert_rows), t0)}

    def _from_stored(self, cache: LoadedCache, run: dict, req: ScoreRequest, t0: float, meta: dict) -> dict[str, Any]:
        accounts = [_nest(r) for r in self.alerts.scores(run["run_id"]).to_pylist()]
        if req.paths == "flagged":
            flagged = [i for i, a in enumerate(accounts) if a["risk_tier"] != "GREEN"]
            if flagged:
                rows, _ = cache.rows_for([accounts[i]["account_id"] for i in flagged])
                sb = self.score_rows(cache, rows, cache.end, req.horizon, req.liability_quantile, req.asset_quantile)
                sub = [accounts[i] for i in flagged]
                self._attach_paths(sub, sb, np.arange(len(rows)), ScoreRequest(
                    paths="all", return_quantiles=req.return_quantiles))
        return {"run_id": run["run_id"], "as_of_date": run["as_of_date"], "horizon": run["horizon"],
                "source": "stored", "scored_at": run["created_at"], **meta, "unknown_accounts": [],
                "accounts": accounts, "alerts": [], "summary": self._summary(accounts, int(run["n_alerts"]), t0)}

    def _attach_paths(self, accounts: list[dict], sb: ScoredBatch, local: np.ndarray, req: ScoreRequest) -> None:
        """Add forecast paths in place, vectorised: one rounding and one tolist per channel."""
        if req.paths == "none" or len(local) == 0:
            return
        if req.paths == "flagged":
            local = np.array([i for i in local if accounts[i]["risk_tier"] != "GREEN"], dtype=int)
            if local.size == 0:
                return
        qidx = [QUANTILES.index(q) for q in req.return_quantiles]
        names = [f"q{int(round(QUANTILES[k] * 100))}" for k in qidx]
        dates = [sb.as_of + timedelta(days=d + 1) for d in range(sb.horizon)]
        bal = np.round(sb.q[local][:, 0][:, :, qidx], 2).transpose(0, 2, 1).tolist()
        util = np.round(sb.q[local][:, 1][:, :, qidx], 4).transpose(0, 2, 1).tolist()
        emi = sb.known_future[local, :, 0]
        for j, i in enumerate(local):
            due = np.flatnonzero(emi[j] > 0)
            accounts[i]["forecast"] = {
                "dates": dates,
                "balance": dict(zip(names, bal[j], strict=True)),
                "utilization": dict(zip(names, util[j], strict=True)),
                "emi_schedule": [{"date": dates[d], "amount": round(float(emi[j, d]), 2)} for d in due],
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
            "scope": "ACCOUNTS",  # request-time runs are never served as stored book results
            "liability_quantile": req.liability_quantile,
            "asset_quantile": req.asset_quantile,
            "model_version": self.engine.model_version,
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
                    "exposure_at_default": _nan_to_none(r.ead[idx]),
                    "probability_of_default": _nan_to_none(r.pd[idx]),
                    "loss_given_default": _nan_to_none(r.lgd[idx]),
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

    def score_table(self, cache: LoadedCache, sb: ScoredBatch, run_id: str) -> pa.Table:
        """Vectorised per-account results (everything a response holds except paths)."""
        r, bh, sn = sb.risk, sb.behavior, sb.snapshot
        acc = cache.accounts.iloc[sb.rows]
        n = len(sb.rows)
        base = np.datetime64(sb.as_of, "D")
        in_sma = r.tier != "GREEN"

        def days(x):
            return pa.array(np.where(x > 0, x, 0).astype(np.int32), mask=x <= 0)

        def dates(x):
            return pa.array(base + np.where(x > 0, x, 0).astype("timedelta64[D]"), mask=x <= 0)

        def num(x, nd=2, hide=None):
            v = np.round(np.asarray(x, dtype=float), nd)
            m = ~np.isfinite(v) if hide is None else (~np.isfinite(v) | hide)
            return pa.array(np.where(m, 0.0, v), mask=m)

        def text(col):
            return pa.array([_s(x) for x in acc[col]] if col in acc else [None] * n, pa.string())

        return conform(ACCOUNT_SCORES, pa.table({
            "run_id": pa.array([run_id] * n),
            "account_id": pa.array(acc["account_id"].astype(str).tolist()),
            "rm_id": text("rm_id"), "segment": text("segment"), "product_type": text("product_type"),
            "as_of_date": pa.array([sb.as_of] * n, pa.date32()),
            "horizon": pa.array(np.full(n, sb.horizon, np.int32)),
            "risk_tier": pa.array(r.tier.astype(str).tolist()),
            "days_to_breach": days(r.days_to_breach), "breach_date": dates(r.days_to_breach),
            "breach_shortfall": num(r.breach_shortfall, hide=~in_sma),
            "days_to_zero": days(r.days_to_zero),
            "current_balance": num(sn.current_balance), "current_utilization": num(sn.current_utilization, 4),
            "next_emi_date": dates(r.next_emi_day),
            "next_emi_amount": num(r.next_emi_amount, hide=r.next_emi_day <= 0),
            "probability_of_default": num(r.pd, 4), "pd_cash_shortfall": num(r.pd_cash, 4),
            "pd_utilization_breach": num(r.p_util, 4), "pd_bureau": num(r.pd_bureau, 4),
            "exposure_at_default": num(r.ead), "loss_given_default": num(r.lgd, 4),
            "expected_loss": num(r.expected_loss), "priority_score": num(r.priority),
            "cibil_score": num(sn.cibil, 0),
            "triggers": pa.array([json.dumps(t) for t in r.triggers]),
            "trailing_avg_balance_90d": num(bh.trailing_avg_90d), "baseline_avg_balance": num(bh.baseline_avg),
            "projected_avg_balance": num(bh.projected_avg),
            "projected_change_pct": num(bh.projected_change_pct * 100),
            "projected_min_balance_worst_case": num(bh.projected_min_worst),
            "trend": pa.array(bh.trend.astype(str).tolist()),
            "attrition_alert": pa.array(bh.attrition_alert.astype(np.int16)),
            "projected_balance_drop": num(bh.projected_balance_drop), "cof_impact_annual": num(bh.cof_impact_annual),
        }))

    def score_book(self, horizons: tuple[int, ...] | None = None, force: bool = False) -> dict[str, Any]:
        """Score every account once per horizon and store the results for fast reads.

        One forward pass at the longest horizon serves every horizon. Only
        ``book_alert_horizon`` raises (and supersedes) alerts; the other runs
        store scores only, so horizons never cancel each other's alerts."""
        st, rs = self.settings, self.settings.risk
        hs = tuple(sorted(set(horizons or st.book_horizons)))
        alert_h = st.book_alert_horizon if st.book_alert_horizon in hs else max(hs)
        with self._book_lock:
            cache = self.loaded_cache(auto_refresh=False)
            mv = self.engine.model_version
            lq, aq = rs.liability_quantile, rs.asset_quantile
            if not force and all(self.alerts.latest_book_run(cache.meta.version, h, lq, aq, mv) for h in hs):
                return {"skipped": True, "reason": "stored runs already match the current data and model"}
            t0 = time.perf_counter()
            runs = {h: "run_" + uuid.uuid4().hex[:12] for h in hs}
            totals = {h: {"n_alerts": 0, "el": 0.0} for h in hs}
            ids = cache.accounts["account_id"].astype(str).to_numpy()
            L, Hp, H = self.engine.context_length, self.engine.horizon_pad, max(hs)
            for a in range(0, cache.n_accounts, st.book_shard_size):
                rows = np.arange(a, min(a + st.book_shard_size, cache.n_accounts))
                win = cache.window(rows, cache.end, L, Hp)
                acc = cache.accounts.iloc[rows]
                q = self.engine.forecast(
                    win["past"], win["known"], H, has_credit_line=acc["credit_limit"].fillna(0).to_numpy(float) > 0
                )
                for h in hs:
                    sb = self.score_rows(cache, rows, cache.end, h, lq, aq, precomputed=(q, win["past"], win["known"]))
                    alerts = self.alert_table(cache, sb, runs[h]) if h == alert_h else EWS_ALERTS.arrow_schema.empty_table()
                    self.alerts.write_shard(runs[h], ids[rows].tolist(), alerts, self.score_table(cache, sb, runs[h]))
                    totals[h]["n_alerts"] += alerts.num_rows
                    if alerts.num_rows:
                        el = alerts.column("expected_loss").to_numpy(zero_copy_only=False)
                        totals[h]["el"] += float(np.nansum(np.asarray(el, dtype=float)))
            # Publish the alert horizon last, so the active list is never left without its alerts.
            for h in sorted(hs, key=lambda x: x == alert_h):
                self.alerts.finalize_run(
                    {
                        "run_id": runs[h], "created_at": utcnow(), "as_of_date": cache.end, "horizon": h,
                        "backend": self.engine.name, "n_accounts": cache.n_accounts,
                        "n_alerts": totals[h]["n_alerts"], "total_expected_loss": totals[h]["el"],
                        "params": json.dumps({"alert_horizon": alert_h}), "scope": "BOOK",
                        "cache_version": cache.meta.version, "liability_quantile": lq, "asset_quantile": aq,
                        "model_version": mv,
                    },
                    supersede=h == alert_h,
                )
                self.alerts.prune_book_runs(h, st.keep_book_runs)
            logger.info("book scored: {} accounts x horizons {} in {:.1f}s", cache.n_accounts, hs, time.perf_counter() - t0)
            return {"runs": runs, "alert_horizon": alert_h, "alerts": totals[alert_h]["n_alerts"],
                    "seconds": round(time.perf_counter() - t0, 2)}

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
