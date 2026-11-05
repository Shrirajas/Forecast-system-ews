"""Persistence for forecast runs and EWS alerts.

New alerts are written as ``PENDING`` and switched to ``ACTIVE`` in the same
transaction that supersedes the previous alerts for the scored accounts, so
readers never see a half-applied run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from data_engine.schema import ACCOUNT_SCORES, EWS_ALERTS, RUN_SCOPE, conform
from data_engine.storage import StorageAdapter

_ORDER = (
    "CASE alert_type WHEN 'CREDIT_EWS' THEN 0 ELSE 1 END, "
    "priority_score DESC, exposure_at_default DESC NULLS LAST, days_to_zero ASC NULLS LAST, account_id"
)


class AlertStore:
    def __init__(self, storage: StorageAdapter):
        self.storage = storage

    def write_shard(
        self, run_id: str, scope_account_ids: list[str], alerts: pa.Table, scores: pa.Table | None = None
    ) -> None:
        """Stage one shard of a run: alerts land as PENDING and stay invisible until finalize_run."""
        self.storage.write_arrow(
            "run_scope",
            conform(RUN_SCOPE, pa.table({"run_id": [run_id] * len(scope_account_ids), "account_id": scope_account_ids})),
            mode="ignore",
        )
        if alerts.num_rows:
            self.storage.write_arrow("ews_alerts", conform(EWS_ALERTS, alerts), mode="upsert")
        if scores is not None and scores.num_rows:
            self.storage.write_arrow("account_scores", conform(ACCOUNT_SCORES, scores), mode="upsert")

    def finalize_run(self, run: dict[str, Any], supersede: bool = True) -> None:
        """Atomically publish a staged run: supersede older alerts of its accounts,
        activate its own alerts and record the run row."""
        run_id = run["run_id"]
        cols = ", ".join(run.keys())
        vals = ", ".join(f":{k}" for k in run)
        stmts: list = []
        if supersede:
            stmts.append(
                (
                    "UPDATE ews_alerts SET status = 'SUPERSEDED' "
                    "WHERE status IN ('ACTIVE', 'ACKNOWLEDGED') AND run_id <> :r "
                    "AND account_id IN (SELECT account_id FROM run_scope WHERE run_id = :r)",
                    {"r": run_id},
                )
            )
        stmts += [
            ("UPDATE ews_alerts SET status = 'ACTIVE' WHERE run_id = :r AND status = 'PENDING'", {"r": run_id}),
            (f"INSERT INTO forecast_runs ({cols}) VALUES ({vals})", run),
            ("DELETE FROM run_scope WHERE run_id = :r", {"r": run_id}),
        ]
        self.storage.execute_many(stmts)

    def record_run(
        self,
        run: dict[str, Any],
        scope_account_ids: list[str],
        alerts: pa.Table,
        scores: pa.Table | None = None,
        supersede: bool = True,
    ) -> None:
        self.write_shard(run["run_id"], scope_account_ids, alerts, scores)
        self.finalize_run(run, supersede)

    def active(
        self,
        alert_type: str | None = None,
        tier: str | None = None,
        rm_id: str | None = None,
        min_priority: float = 0.0,
        include_acknowledged: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where = ["status IN ('ACTIVE', 'ACKNOWLEDGED')" if include_acknowledged else "status = 'ACTIVE'"]
        params: dict[str, Any] = {"mp": min_priority, "lim": limit, "off": offset}
        where.append("priority_score >= :mp")
        if alert_type:
            where.append("alert_type = :t")
            params["t"] = alert_type
        if tier:
            where.append("risk_tier = :tier")
            params["tier"] = tier
        if rm_id:
            where.append("rm_id = :rm")
            params["rm"] = rm_id
        clause = " AND ".join(where)
        total = int(self.storage.scalar(f"SELECT COUNT(*) FROM ews_alerts WHERE {clause}", params) or 0)
        rows = self.storage.query_arrow(
            f"SELECT * FROM ews_alerts WHERE {clause} ORDER BY {_ORDER} LIMIT :lim OFFSET :off", params
        ).to_pylist()
        for i, r in enumerate(rows):
            r["rank"] = offset + i + 1
        return rows, total

    def acknowledge(self, alert_id: str) -> bool:
        exists = self.storage.scalar(
            "SELECT COUNT(*) FROM ews_alerts WHERE alert_id = :a AND status = 'ACTIVE'", {"a": alert_id}
        )
        if not exists:
            return False
        self.storage.execute("UPDATE ews_alerts SET status = 'ACKNOWLEDGED' WHERE alert_id = :a", {"a": alert_id})
        return True

    def latest_book_run(
        self, cache_version: str, horizon: int, liability_q: float, asset_q: float, backend: str
    ) -> dict[str, Any] | None:
        """Most recent whole-book run that scored exactly the current data."""
        rows = self.storage.query_arrow(
            "SELECT * FROM forecast_runs WHERE scope = 'BOOK' AND cache_version = :v AND horizon = :h "
            "AND liability_quantile = :lq AND asset_quantile = :aq AND backend = :b "
            "ORDER BY created_at DESC LIMIT 1",
            {"v": cache_version, "h": horizon, "lq": liability_q, "aq": asset_q, "b": backend},
        ).to_pylist()
        return rows[0] if rows else None

    def scores(self, run_id: str, tiers: list[str] | None = None) -> pa.Table:
        params: dict[str, Any] = {"r": run_id}
        where = "run_id = :r"
        if tiers:
            names = []
            for i, t in enumerate(tiers):
                params[f"t{i}"] = t
                names.append(f":t{i}")
            where += f" AND risk_tier IN ({', '.join(names)})"
        return self.storage.query_arrow(f"SELECT * FROM account_scores WHERE {where} ORDER BY account_id", params)

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.storage.query_arrow(
            "SELECT * FROM forecast_runs ORDER BY created_at DESC LIMIT :n", {"n": limit}
        ).to_pylist()


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
