"""Persistence for forecast runs and EWS alerts.

New alerts are written as ``PENDING`` and switched to ``ACTIVE`` in the same
transaction that supersedes the previous alerts for the scored accounts, so
readers never see a half-applied run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
        now = utcnow()
        n = len(scope_account_ids)
        self.storage.write_arrow(
            "run_scope",
            conform(
                RUN_SCOPE,
                pa.table({"run_id": [run_id] * n, "account_id": scope_account_ids, "created_at": [now] * n}),
            ),
            mode="ignore",
        )
        if alerts.num_rows:
            self.storage.write_arrow("ews_alerts", conform(EWS_ALERTS, alerts), mode="upsert")
        if scores is not None and scores.num_rows:
            if "created_at" not in scores.column_names:
                scores = scores.append_column("created_at", pa.array([now] * scores.num_rows, pa.timestamp("us")))
            self.storage.write_arrow("account_scores", conform(ACCOUNT_SCORES, scores), mode="upsert")

    def finalize_run(self, run: dict[str, Any], supersede: bool = True) -> None:
        """Atomically publish a staged run: supersede older alerts of its accounts,
        activate its own alerts and record the run row.

        Book runs of several horizons should raise alerts from one of them only:
        finalise the others with ``supersede=False`` and no alerts, otherwise
        whichever horizon finishes last decides the active alert list."""
        run = {**run}
        for k in ("liability_quantile", "asset_quantile"):
            if run.get(k) is not None:
                run[k] = round(float(run[k]), 2)  # stored rounded: lookups compare rounded values
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

    def prune_book_runs(self, horizon: int, keep: int) -> int:
        """Delete stored scores of all but the newest ``keep`` book runs for a horizon.
        Run rows stay as history; only the per-account rows are removed."""
        old = self.storage.query_arrow(
            "SELECT run_id FROM forecast_runs WHERE scope = 'BOOK' AND horizon = :h "
            "ORDER BY created_at DESC LIMIT 1000000 OFFSET :k",
            {"h": horizon, "k": max(keep, 1)},
        ).column("run_id").to_pylist()
        for rid in old:
            self.storage.execute("DELETE FROM account_scores WHERE run_id = :r", {"r": rid})
        return len(old)

    def cleanup_orphans(self, older_than: timedelta = timedelta(hours=1)) -> dict[str, int]:
        """Remove rows staged by runs that never finalised (crashed mid-way).

        Readers only see finalised runs, so these are never served; this only
        stops them piling up. Rows younger than ``older_than`` are kept, as
        they may belong to a run still in progress in another process."""
        cutoff = utcnow() - older_than
        orphan = "run_id NOT IN (SELECT run_id FROM forecast_runs) AND created_at < :c"
        counts = {}
        for table, extra in (("ews_alerts", " AND status = 'PENDING'"), ("account_scores", ""), ("run_scope", "")):
            counts[table] = int(self.storage.scalar(f"SELECT COUNT(*) FROM {table} WHERE {orphan}{extra}", {"c": cutoff}) or 0)
            if counts[table]:
                self.storage.execute(f"DELETE FROM {table} WHERE {orphan}{extra}", {"c": cutoff})
        return counts

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
        q: str | None = None,
        product_type: str | None = None,
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
        if q:  # LIKE with escaped wildcards: a plain substring search on the account id
            where.append("LOWER(account_id) LIKE :q ESCAPE '\\'")
            params["q"] = "%" + q.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        if product_type:
            where.append("account_id IN (SELECT account_id FROM accounts WHERE product_type = :pt)")
            params["pt"] = product_type
        clause = " AND ".join(where)
        total = int(self.storage.scalar(f"SELECT COUNT(*) FROM ews_alerts WHERE {clause}", params) or 0)
        rows = self.storage.query_arrow(
            f"SELECT * FROM ews_alerts WHERE {clause} ORDER BY {_ORDER} LIMIT :lim OFFSET :off", params
        ).to_pylist()
        for i, r in enumerate(rows):
            r["rank"] = offset + i + 1
        return rows, total

    def product_types(self, account_ids: list[str]) -> dict[str, str]:
        if not account_ids:
            return {}
        names = {f"a{i}": a for i, a in enumerate(account_ids)}
        rows = self.storage.query_arrow(
            f"SELECT account_id, product_type FROM accounts WHERE account_id IN ({', '.join(':' + k for k in names)})", names
        ).to_pylist()
        return {r["account_id"]: r["product_type"] for r in rows}

    def counts(self) -> dict[str, int]:
        """Alert rows by status: the overview shows ACTIVE; the tables panel counts every run's rows."""
        rows = self.storage.query_arrow("SELECT status, COUNT(*) AS n FROM ews_alerts GROUP BY status").to_pylist()
        return {r["status"]: int(r["n"]) for r in rows}

    def acknowledge(self, alert_id: str) -> bool:
        exists = self.storage.scalar(
            "SELECT COUNT(*) FROM ews_alerts WHERE alert_id = :a AND status = 'ACTIVE'", {"a": alert_id}
        )
        if not exists:
            return False
        self.storage.execute("UPDATE ews_alerts SET status = 'ACKNOWLEDGED' WHERE alert_id = :a", {"a": alert_id})
        return True

    def latest_book_run(
        self, cache_version: str, horizon: int, liability_q: float, asset_q: float, model_version: str
    ) -> dict[str, Any] | None:
        """Most recent whole-book run that scored exactly the current data with exactly
        the current weights (a retrain changes ``model_version``, so old runs stop matching)."""
        rows = self.storage.query_arrow(
            "SELECT * FROM forecast_runs WHERE scope = 'BOOK' AND cache_version = :v AND horizon = :h "
            "AND ROUND(liability_quantile, 2) = :lq AND ROUND(asset_quantile, 2) = :aq "
            "AND model_version = :mv ORDER BY created_at DESC LIMIT 1",
            {"v": cache_version, "h": horizon, "lq": round(float(liability_q), 2),
             "aq": round(float(asset_q), 2), "mv": model_version},
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
