"""Persistence for forecast runs and EWS alerts.

New alerts are written as ``PENDING`` and switched to ``ACTIVE`` in the same
transaction that supersedes the previous alerts for the scored accounts, so
readers never see a half-applied run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from data_engine.schema import EWS_ALERTS, RUN_SCOPE, conform
from data_engine.storage import StorageAdapter

_ORDER = (
    "CASE alert_type WHEN 'CREDIT_EWS' THEN 0 ELSE 1 END, "
    "priority_score DESC, exposure_at_default DESC NULLS LAST, days_to_zero ASC NULLS LAST, account_id"
)


class AlertStore:
    def __init__(self, storage: StorageAdapter):
        self.storage = storage

    def record_run(self, run: dict[str, Any], scope_account_ids: list[str], alerts: pa.Table) -> None:
        run_id = run["run_id"]
        self.storage.write_arrow(
            "run_scope",
            conform(RUN_SCOPE, pa.table({"run_id": [run_id] * len(scope_account_ids), "account_id": scope_account_ids})),
            mode="ignore",
        )
        if alerts.num_rows:
            self.storage.write_arrow("ews_alerts", conform(EWS_ALERTS, alerts), mode="upsert")
        cols = ", ".join(run.keys())
        vals = ", ".join(f":{k}" for k in run)
        self.storage.execute_many(
            [
                (
                    "UPDATE ews_alerts SET status = 'SUPERSEDED' "
                    "WHERE status IN ('ACTIVE', 'ACKNOWLEDGED') AND run_id <> :r "
                    "AND account_id IN (SELECT account_id FROM run_scope WHERE run_id = :r)",
                    {"r": run_id},
                ),
                ("UPDATE ews_alerts SET status = 'ACTIVE' WHERE run_id = :r AND status = 'PENDING'", {"r": run_id}),
                (f"INSERT INTO forecast_runs ({cols}) VALUES ({vals})", run),
                ("DELETE FROM run_scope WHERE run_id = :r", {"r": run_id}),
            ]
        )

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

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.storage.query_arrow(
            "SELECT * FROM forecast_runs ORDER BY created_at DESC LIMIT :n", {"n": limit}
        ).to_pylist()


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
