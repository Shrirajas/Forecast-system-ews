"""Single source of truth for table layouts.

Each table is declared once with portable logical types and rendered to
DuckDB or PostgreSQL DDL, plus a matching Arrow schema used for zero-copy
streaming between PyArrow and the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa

# logical type -> (duckdb, postgres, arrow)
_TYPES: dict[str, tuple[str, str, pa.DataType]] = {
    "str": ("VARCHAR", "TEXT", pa.string()),
    "date": ("DATE", "DATE", pa.date32()),
    "ts": ("TIMESTAMP", "TIMESTAMP", pa.timestamp("us")),
    "f64": ("DOUBLE", "DOUBLE PRECISION", pa.float64()),
    "i32": ("INTEGER", "INTEGER", pa.int32()),
    "i16": ("SMALLINT", "SMALLINT", pa.int16()),
}


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[tuple[str, str, bool], ...]  # (name, logical type, nullable)
    primary_key: tuple[str, ...]
    indexes: tuple[tuple[str, ...], ...] = field(default_factory=tuple)

    @property
    def column_names(self) -> list[str]:
        return [c[0] for c in self.columns]

    @property
    def arrow_schema(self) -> pa.Schema:
        return pa.schema(
            [pa.field(n, _TYPES[t][2], nullable=nullable) for n, t, nullable in self.columns]
        )

    def column_sql(self, name: str, dialect: str) -> str:
        idx = 0 if dialect == "duckdb" else 1
        n, t, nullable = next(c for c in self.columns if c[0] == name)
        return f"{n} {_TYPES[t][idx]}{'' if nullable else ' NOT NULL'}"

    def ddl(self, dialect: str) -> list[str]:
        idx = 0 if dialect == "duckdb" else 1
        cols = ",\n  ".join(
            f"{n} {_TYPES[t][idx]}{'' if nullable else ' NOT NULL'}" for n, t, nullable in self.columns
        )
        stmts = [
            f"CREATE TABLE IF NOT EXISTS {self.name} (\n  {cols},\n"
            f"  PRIMARY KEY ({', '.join(self.primary_key)})\n)"
        ]
        for cols_ in self.indexes:
            stmts.append(
                f"CREATE INDEX IF NOT EXISTS ix_{self.name}_{'_'.join(cols_)} "
                f"ON {self.name} ({', '.join(cols_)})"
            )
        return stmts


ACCOUNTS = Table(
    "accounts",
    (
        ("account_id", "str", False),
        ("customer_id", "str", False),
        ("rm_id", "str", True),
        ("segment", "str", False),
        ("product_type", "str", False),
        ("open_date", "date", False),
        ("sanctioned_amount", "f64", False),
        ("outstanding_principal", "f64", False),
        ("emi_amount", "f64", False),
        ("emi_day", "i16", False),
        ("salary_day", "i16", True),
        ("credit_limit", "f64", False),
        ("lgd", "f64", False),
        ("interest_rate", "f64", False),
    ),
    ("account_id",),
    (("rm_id",),),
)

DAILY_BALANCES = Table(
    "daily_balances",
    (
        ("account_id", "str", False),
        ("as_of_date", "date", False),
        ("eod_balance", "f64", False),
        ("credit_limit", "f64", False),
        ("credit_drawn", "f64", False),
    ),
    ("account_id", "as_of_date"),
    (("as_of_date",),),
)

TRANSACTIONS = Table(
    "transactions",
    (
        ("txn_id", "str", False),
        ("account_id", "str", False),
        ("txn_date", "date", False),
        ("amount", "f64", False),
        ("direction", "str", False),  # CR / DR
        ("txn_type", "str", False),
        ("status", "str", False),  # POSTED / RETURNED
    ),
    ("txn_id",),
    (("account_id", "txn_date"), ("txn_date",)),
)

CREDIT_PROFILES = Table(
    "credit_profiles",
    (
        ("account_id", "str", False),
        ("pull_date", "date", False),
        ("cibil_score", "i16", False),
        ("enquiries_6m", "i16", True),
    ),
    ("account_id", "pull_date"),
)

LOAN_SCHEDULES = Table(
    "loan_schedules",
    (
        ("account_id", "str", False),
        ("due_date", "date", False),
        ("emi_amount", "f64", False),
        ("installment_no", "i32", True),
    ),
    ("account_id", "due_date"),
    (("due_date",),),
)

HOLIDAYS = Table("holidays", (("holiday_date", "date", False), ("name", "str", True)), ("holiday_date",))

FEATURE_MATRIX = Table(
    "feature_matrix",
    (
        ("account_id", "str", False),
        ("as_of_date", "date", False),
        ("eod_balance", "f64", False),
        ("utilization", "f64", False),
        ("cibil_score", "f64", True),
        ("outward_std_30d", "f64", False),
        ("missed_salary_flag", "i16", False),
        ("payment_return_flag", "i16", False),
    ),
    ("account_id", "as_of_date"),
    (("as_of_date",),),
)

AGG_WATERMARKS = Table(
    "agg_watermarks",
    (
        ("name", "str", False),
        ("watermark", "date", True),
        ("dirty_from", "date", True),
        ("updated_at", "ts", True),
    ),
    ("name",),
)

FORECAST_RUNS = Table(
    "forecast_runs",
    (
        ("run_id", "str", False),
        ("created_at", "ts", False),
        ("as_of_date", "date", False),
        ("horizon", "i32", False),
        ("backend", "str", False),
        ("n_accounts", "i32", False),
        ("n_alerts", "i32", False),
        ("total_expected_loss", "f64", False),
        ("params", "str", True),
        ("scope", "str", True),  # BOOK (whole portfolio) / ACCOUNTS
        ("cache_version", "str", True),  # data version the run scored
        ("liability_quantile", "f64", True),
        ("asset_quantile", "f64", True),
        ("model_version", "str", True),  # exact weights that produced the run
    ),
    ("run_id",),
    (("scope", "cache_version"),),
)

#: One row per account per run: everything /predict/ews returns except paths,
#: so whole-book reads come straight from the latest matching run.
ACCOUNT_SCORES = Table(
    "account_scores",
    (
        ("run_id", "str", False),
        ("account_id", "str", False),
        ("rm_id", "str", True),
        ("segment", "str", True),
        ("product_type", "str", True),
        ("as_of_date", "date", False),
        ("horizon", "i32", False),
        ("risk_tier", "str", False),
        ("days_to_breach", "i32", True),
        ("breach_date", "date", True),
        ("breach_shortfall", "f64", True),
        ("days_to_zero", "i32", True),
        ("current_balance", "f64", True),
        ("current_utilization", "f64", True),
        ("next_emi_date", "date", True),
        ("next_emi_amount", "f64", True),
        ("probability_of_default", "f64", True),
        ("pd_cash_shortfall", "f64", True),
        ("pd_utilization_breach", "f64", True),
        ("pd_bureau", "f64", True),
        ("exposure_at_default", "f64", True),
        ("loss_given_default", "f64", True),
        ("expected_loss", "f64", True),
        ("priority_score", "f64", True),
        ("cibil_score", "f64", True),
        ("triggers", "str", True),
        ("trailing_avg_balance_90d", "f64", True),
        ("baseline_avg_balance", "f64", True),
        ("projected_avg_balance", "f64", True),
        ("projected_change_pct", "f64", True),
        ("projected_min_balance_worst_case", "f64", True),
        ("trend", "str", True),
        ("attrition_alert", "i16", True),
        ("projected_balance_drop", "f64", True),
        ("cof_impact_annual", "f64", True),
        ("created_at", "ts", True),
    ),
    ("run_id", "account_id"),
    (("run_id",),),
)

RUN_SCOPE = Table(
    "run_scope",
    (("run_id", "str", False), ("account_id", "str", False), ("created_at", "ts", True)),
    ("run_id", "account_id"),
)

EWS_ALERTS = Table(
    "ews_alerts",
    (
        ("alert_id", "str", False),
        ("run_id", "str", False),
        ("account_id", "str", False),
        ("rm_id", "str", True),
        ("created_at", "ts", False),
        ("as_of_date", "date", False),
        ("alert_type", "str", False),  # CREDIT_EWS / DEPOSIT_ATTRITION
        ("risk_tier", "str", False),
        ("priority_score", "f64", False),
        ("current_balance", "f64", True),
        ("exposure_at_default", "f64", True),
        ("probability_of_default", "f64", True),
        ("loss_given_default", "f64", True),
        ("expected_loss", "f64", True),
        ("days_to_breach", "i32", True),
        ("breach_date", "date", True),
        ("days_to_zero", "i32", True),
        ("next_emi_amount", "f64", True),
        ("next_emi_date", "date", True),
        ("projected_balance_drop", "f64", True),
        ("cof_impact_annual", "f64", True),
        ("triggers", "str", True),
        ("status", "str", False),  # ACTIVE / ACKNOWLEDGED / SUPERSEDED
    ),
    ("alert_id",),
    (("status", "alert_type"), ("account_id",)),
)

ALL_TABLES: dict[str, Table] = {
    t.name: t
    for t in (
        ACCOUNTS,
        DAILY_BALANCES,
        TRANSACTIONS,
        CREDIT_PROFILES,
        LOAN_SCHEDULES,
        HOLIDAYS,
        FEATURE_MATRIX,
        AGG_WATERMARKS,
        FORECAST_RUNS,
        ACCOUNT_SCORES,
        RUN_SCOPE,
        EWS_ALERTS,
    )
}

#: Tables that clients may load through the ingestion API.
INGESTIBLE_TABLES: tuple[str, ...] = (
    "accounts",
    "daily_balances",
    "transactions",
    "credit_profiles",
    "loan_schedules",
    "holidays",
)

#: Date column per ingestible table, used to mark the feature matrix dirty.
EVENT_DATE_COLUMN: dict[str, str] = {
    "daily_balances": "as_of_date",
    "transactions": "txn_date",
    "credit_profiles": "pull_date",
}


def conform(table: Table, data: pa.Table) -> pa.Table:
    """Select/order columns and cast to the canonical Arrow schema."""
    schema = table.arrow_schema
    cols = []
    for f in schema:
        if f.name in data.column_names:
            cols.append(data.column(f.name).cast(f.type))
        elif f.nullable:
            cols.append(pa.nulls(data.num_rows, type=f.type))
        else:
            raise ValueError(f"{table.name}: missing required column '{f.name}'")
    return pa.Table.from_arrays(cols, schema=schema)
