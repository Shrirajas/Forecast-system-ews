"""Incremental nightly aggregation of raw ledgers into the rolling feature matrix.

The feature matrix holds one dense row per account-day:

* targets: EOD balance and credit-line utilisation,
* past covariates: forward-filled CIBIL score, 30-day rolling standard
  deviation of outward payments, missed-salary and payment-return indicators.

Only the affected date range is recomputed: from the earlier of
``watermark + 1`` and the earliest late-arriving event (``dirty_from``) up
to the newest balance date. Rolling windows read a fixed lookback before
that range, so an incremental pass yields exactly the same rows as a full
rebuild. All work runs inside the database engine with window functions;
nothing is looped over in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from data_engine.ingestion import FEATURE_WATERMARK
from data_engine.storage import StorageAdapter
from utils.logging import logger

#: Longest rolling window used below (missed-salary detection) + safety margin.
LOOKBACK_DAYS = 60
MISSED_SALARY_DAYS = 35

_FEATURE_SQL = f"""
WITH bal AS (
    SELECT account_id, as_of_date, eod_balance, credit_limit, credit_drawn
    FROM daily_balances
    WHERE as_of_date BETWEEN CAST(:lookback_start AS DATE) AND CAST(:end_date AS DATE)
),
flows AS (
    SELECT account_id,
           txn_date AS as_of_date,
           SUM(CASE WHEN direction = 'DR' AND status = 'POSTED' THEN amount ELSE 0 END) AS outward_total,
           SUM(CASE WHEN txn_type = 'SALARY' AND direction = 'CR' AND status = 'POSTED' THEN amount ELSE 0 END)
               AS salary_amt,
           SUM(CASE WHEN status = 'RETURNED' THEN 1 ELSE 0 END) AS returns_cnt
    FROM transactions
    WHERE txn_date BETWEEN CAST(:lookback_start AS DATE) AND CAST(:end_date AS DATE)
    GROUP BY account_id, txn_date
),
carry AS (  -- last bureau score known before the window (forward-fill seed)
    SELECT account_id, cibil_score
    FROM (
        SELECT account_id, cibil_score,
               ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY pull_date DESC) AS rn
        FROM credit_profiles
        WHERE pull_date < CAST(:lookback_start AS DATE)
    ) c
    WHERE rn = 1
),
joined AS (
    SELECT b.account_id, b.as_of_date, b.eod_balance,
           CASE WHEN b.credit_limit > 0 THEN b.credit_drawn / b.credit_limit ELSE 0 END AS utilization,
           COALESCE(f.outward_total, 0) AS outward_total,
           COALESCE(f.salary_amt, 0) AS salary_amt,
           COALESCE(f.returns_cnt, 0) AS returns_cnt,
           cp.cibil_score AS cibil_pull
    FROM bal b
    LEFT JOIN flows f ON f.account_id = b.account_id AND f.as_of_date = b.as_of_date
    LEFT JOIN credit_profiles cp ON cp.account_id = b.account_id AND cp.pull_date = b.as_of_date
),
grouped AS (
    SELECT j.*,
           COUNT(cibil_pull) OVER (PARTITION BY account_id ORDER BY as_of_date
                                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cibil_grp
    FROM joined j
),
windowed AS (
    SELECT g.account_id, g.as_of_date, g.eod_balance, g.utilization,
           MAX(g.cibil_pull) OVER (PARTITION BY g.account_id, g.cibil_grp) AS cibil_ffill,
           STDDEV_SAMP(g.outward_total) OVER w30 AS outward_std_30d,
           MAX(CASE WHEN g.salary_amt > 0 THEN g.as_of_date END) OVER w60 AS last_salary_date,
           SUM(g.returns_cnt) OVER w30 AS returns_30d
    FROM grouped g
    WINDOW w30 AS (PARTITION BY g.account_id ORDER BY g.as_of_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW),
           w60 AS (PARTITION BY g.account_id ORDER BY g.as_of_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
)
SELECT w.account_id,
       w.as_of_date,
       w.eod_balance,
       w.utilization,
       CAST(COALESCE(w.cibil_ffill, c.cibil_score) AS DOUBLE PRECISION) AS cibil_score,
       COALESCE(w.outward_std_30d, 0) AS outward_std_30d,
       CAST(CASE WHEN a.salary_day IS NOT NULL
                  AND w.as_of_date - a.open_date > {MISSED_SALARY_DAYS}
                  AND (w.last_salary_date IS NULL OR w.as_of_date - w.last_salary_date > {MISSED_SALARY_DAYS})
                 THEN 1 ELSE 0 END AS SMALLINT) AS missed_salary_flag,
       CAST(CASE WHEN w.returns_30d > 0 THEN 1 ELSE 0 END AS SMALLINT) AS payment_return_flag
FROM windowed w
LEFT JOIN carry c ON c.account_id = w.account_id
LEFT JOIN accounts a ON a.account_id = w.account_id
WHERE w.as_of_date >= CAST(:start_date AS DATE)
"""

_COLUMNS = (
    "account_id, as_of_date, eod_balance, utilization, cibil_score, outward_std_30d, "
    "missed_salary_flag, payment_return_flag"
)


@dataclass
class AggregationResult:
    mode: str  # full / incremental / noop
    start_date: date | None
    end_date: date | None
    rows: int
    seconds: float


class FeatureAggregator:
    def __init__(self, storage: StorageAdapter):
        self.storage = storage

    def _state(self) -> tuple[date | None, date | None]:
        rows = self.storage.query_arrow(
            "SELECT watermark, dirty_from FROM agg_watermarks WHERE name = :n", {"n": FEATURE_WATERMARK}
        ).to_pylist()
        return (rows[0]["watermark"], rows[0]["dirty_from"]) if rows else (None, None)

    def refresh(self, full: bool = False) -> AggregationResult:
        t0 = datetime.now()
        bounds = self.storage.query_arrow(
            "SELECT MIN(as_of_date) AS lo, MAX(as_of_date) AS hi FROM daily_balances"
        ).to_pylist()[0]
        lo, hi = bounds["lo"], bounds["hi"]
        if hi is None:
            return AggregationResult("noop", None, None, 0, 0.0)

        watermark, dirty_from = self._state()
        if full or watermark is None:
            mode, start = "full", lo
        else:
            candidates = [watermark + timedelta(days=1)] + ([dirty_from] if dirty_from else [])
            start = max(lo, min(candidates))
            mode = "incremental"
            if start > hi:
                return AggregationResult("noop", None, watermark, 0, 0.0)

        params = {
            "start_date": start,
            "end_date": hi,
            "lookback_start": start - timedelta(days=LOOKBACK_DAYS),
        }
        now = datetime.now(UTC).replace(tzinfo=None)
        self.storage.execute_many(
            [
                ("DELETE FROM feature_matrix WHERE as_of_date >= CAST(:start_date AS DATE)", params),
                (f"INSERT INTO feature_matrix ({_COLUMNS}) {_FEATURE_SQL}", params),
                ("DELETE FROM agg_watermarks WHERE name = :n", {"n": FEATURE_WATERMARK}),
                (
                    "INSERT INTO agg_watermarks (name, watermark, dirty_from, updated_at) VALUES (:n, :wm, NULL, :ts)",
                    {"n": FEATURE_WATERMARK, "wm": hi, "ts": now},
                ),
            ]
        )
        rows = int(
            self.storage.scalar(
                "SELECT COUNT(*) FROM feature_matrix WHERE as_of_date >= CAST(:s AS DATE)", {"s": start}
            )
            or 0
        )
        secs = (datetime.now() - t0).total_seconds()
        logger.info("feature matrix {} refresh {} .. {}: {} rows in {:.2f}s", mode, start, hi, rows, secs)
        return AggregationResult(mode, start, hi, rows, secs)
