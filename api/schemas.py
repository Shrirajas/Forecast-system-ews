"""Pydantic contracts for the EWS REST API."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config.settings import QUANTILES
from utils.timefmt import to_ist

AccountId = Annotated[str, Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_\-]+$")]
Money = Annotated[float, Field(ge=0)]
Tier = Literal["GREEN", "SMA-0", "SMA-1", "SMA-2"]
AlertType = Literal["CREDIT_EWS", "DEPOSIT_ATTRITION"]
MAX_JSON_ROWS = 500_000


def _quantile(v: float) -> float:
    r = round(float(v), 2)
    if r not in QUANTILES:
        raise ValueError(f"quantile must be one of {list(QUANTILES)}")
    return r


# ---------------------------------------------------------------------------- ingestion
class AccountRecord(BaseModel):
    account_id: AccountId
    customer_id: str = Field(min_length=1, max_length=64)
    rm_id: str | None = Field(None, max_length=32)
    segment: str = Field(max_length=32, examples=["RETAIL_SALARIED"])
    product_type: str = Field(max_length=32, examples=["HOME"])
    open_date: date
    sanctioned_amount: Money
    outstanding_principal: Money
    emi_amount: Money
    emi_day: int = Field(ge=1, le=31)
    salary_day: int | None = Field(None, ge=1, le=31)
    credit_limit: Money = 0.0
    lgd: float = Field(0.45, ge=0, le=1)
    interest_rate: float = Field(0.1, ge=0, le=1)


class DailyBalanceRecord(BaseModel):
    account_id: AccountId
    as_of_date: date
    eod_balance: float
    credit_limit: Money = 0.0
    credit_drawn: Money = 0.0


class TransactionRecord(BaseModel):
    txn_id: str = Field(min_length=1, max_length=64)
    account_id: AccountId
    txn_date: date
    amount: float = Field(gt=0)
    direction: Literal["CR", "DR"]
    txn_type: str = Field(max_length=32, examples=["SALARY", "EMI", "UPI_POS"])
    status: Literal["POSTED", "RETURNED"] = "POSTED"

    @field_validator("txn_type")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class CreditProfileRecord(BaseModel):
    account_id: AccountId
    pull_date: date
    cibil_score: int = Field(ge=300, le=900)
    enquiries_6m: int | None = Field(None, ge=0)


class LoanScheduleRecord(BaseModel):
    account_id: AccountId
    due_date: date
    emi_amount: float = Field(gt=0)
    installment_no: int | None = Field(None, ge=1)


class HolidayRecord(BaseModel):
    holiday_date: date
    name: str | None = None


class IngestRequest(BaseModel):
    batch_id: str | None = Field(None, max_length=64)
    accounts: list[AccountRecord] = []
    daily_balances: list[DailyBalanceRecord] = []
    transactions: list[TransactionRecord] = []
    credit_profiles: list[CreditProfileRecord] = []
    loan_schedules: list[LoanScheduleRecord] = []
    holidays: list[HolidayRecord] = []
    refresh_features: bool = Field(True, description="Run the incremental aggregation + cache rebuild after loading")
    wait_for_refresh: bool = Field(False, description="Block until the refresh finishes (default: background)")

    @model_validator(mode="after")
    def _non_empty(self) -> IngestRequest:
        n = sum(len(getattr(self, t)) for t in TABLE_FIELDS)
        if n == 0:
            raise ValueError("payload contains no records")
        if n > MAX_JSON_ROWS:
            raise ValueError(f"{n} rows exceeds the JSON limit {MAX_JSON_ROWS}; use /api/v1/ingest/file")
        return self


TABLE_FIELDS = ("accounts", "loan_schedules", "holidays", "credit_profiles", "daily_balances", "transactions")


class TableIngestStats(BaseModel):
    rows: int
    min_event_date: date | None = None
    max_event_date: date | None = None


class IngestResponse(BaseModel):
    batch_id: str
    tables: dict[str, TableIngestStats]
    refresh: Literal["completed", "scheduled", "skipped"]
    refresh_result: dict[str, Any] | None = None
    elapsed_ms: float


class DatasetIngestRequest(BaseModel):
    table: Literal["accounts", "daily_balances", "transactions", "credit_profiles", "loan_schedules", "holidays"]
    path: str = Field(description="File or directory (e.g. a partitioned Parquet dataset) under the server data dir")
    format: Literal["parquet", "csv"] = "parquet"
    refresh_features: bool = True


# ---------------------------------------------------------------------------- prediction
class QuantileTriggers(BaseModel):
    liability: float = Field(0.1, description="Worst-case cash track (q10..q50) used for the EMI-shortfall trigger")
    asset: float = Field(0.9, description="Worst-case utilisation track (q50..q90) used for EAD / limit breach")

    @field_validator("liability", "asset")
    @classmethod
    def _q(cls, v: float) -> float:
        return _quantile(v)

    @model_validator(mode="after")
    def _sides(self) -> QuantileTriggers:
        if self.liability > 0.5:
            raise ValueError("liability trigger must be a lower quantile (<= 0.5)")
        if self.asset < 0.5:
            raise ValueError("asset trigger must be an upper quantile (>= 0.5)")
        return self


class PredictRequest(BaseModel):
    account_ids: list[AccountId] | None = Field(
        None, max_length=200_000, description="Accounts to score; omit to score the whole portfolio"
    )
    horizon: Literal[30, 60, 90] = 90
    as_of_date: date | None = Field(None, description="Forecast origin (defaults to the latest data day)")
    quantile_triggers: QuantileTriggers = Field(default_factory=QuantileTriggers)
    return_quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    include_paths: bool | Literal["flagged"] | None = Field(
        None,
        description="Forecast paths: true = every account, \"flagged\" = non-GREEN accounts only, false = none. "
        "Default: none for whole-book requests (served from the stored book run), all for account lists.",
    )
    tiers: list[Tier] | None = Field(None, description="Only return accounts in these tiers")
    persist_alerts: bool = True

    @field_validator("return_quantiles")
    @classmethod
    def _qs(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("return_quantiles must not be empty")
        return sorted({_quantile(x) for x in v})

    @field_validator("account_ids")
    @classmethod
    def _dedupe(cls, v: list[str] | None) -> list[str] | None:
        return list(dict.fromkeys(v)) if v is not None else None


class NextEMI(BaseModel):
    date: date
    amount: float | None


class PDComponents(BaseModel):
    cash_shortfall: float | None
    utilization_breach: float | None
    bureau: float | None


class BehaviorOut(BaseModel):
    trailing_avg_balance_90d: float | None
    baseline_avg_balance: float | None = Field(None, description="Average balance over days -180..-30")
    projected_avg_balance: float | None
    projected_change_pct: float | None
    projected_min_balance_worst_case: float | None
    trend: Literal["RISING", "STABLE", "DECLINING"]
    attrition_alert: bool
    projected_balance_drop: float | None
    cof_impact_annual: float | None


class EMIPoint(BaseModel):
    date: date
    amount: float | None


class ForecastPaths(BaseModel):
    dates: list[date]
    balance: dict[str, list[float]]
    utilization: dict[str, list[float]]
    emi_schedule: list[EMIPoint]


class AccountForecast(BaseModel):
    account_id: str
    rm_id: str | None
    segment: str | None
    product_type: str | None
    risk_tier: Tier
    days_to_breach: int | None
    breach_date: date | None
    breach_shortfall: float | None
    days_to_zero: int | None
    current_balance: float | None
    current_utilization: float | None
    next_emi: NextEMI | None
    probability_of_default: float | None
    pd_components: PDComponents
    exposure_at_default: float | None
    loss_given_default: float | None
    expected_loss: float | None
    priority_score: float | None
    cibil_score: float | None
    triggers: list[str]
    behavior: BehaviorOut
    forecast: ForecastPaths | None = None
    display: dict[str, Any] | None = Field(None, description="Server-side labels: tier wording, N/A reasons, PD basis, triggers")


class PredictSummary(BaseModel):
    accounts_scored: int
    tier_counts: dict[str, int]
    sma_exposure_at_default: float
    total_expected_loss: float
    attrition_alerts: int
    attrition_cof_impact_annual: float
    alerts_raised: int
    latency_ms: float


class PredictResponse(BaseModel):
    run_id: str
    as_of_date: date
    horizon: int
    source: Literal["stored", "live"] = Field(description="stored = read from the latest book run, no forward pass")
    scored_at: datetime
    cache_version: str
    model_version: str
    backend: str | None = None
    quantile_triggers: QuantileTriggers
    summary: PredictSummary
    unknown_accounts: list[str]
    accounts: list[AccountForecast]


# ---------------------------------------------------------------------------- alerts
class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rank: int | None = Field(None, description="Position in the ranked list (null for live-pushed alerts)")
    alert_id: str
    run_id: str
    account_id: str
    rm_id: str | None
    alert_type: AlertType
    risk_tier: Tier
    priority_score: float
    status: str
    created_at: datetime
    as_of_date: date
    current_balance: float | None
    exposure_at_default: float | None
    probability_of_default: float | None
    loss_given_default: float | None
    expected_loss: float | None
    days_to_breach: int | None
    breach_date: date | None
    days_to_zero: int | None
    next_emi_amount: float | None
    next_emi_date: date | None
    projected_balance_drop: float | None
    cof_impact_annual: float | None
    triggers: list[str]
    product_type: str | None = None
    display: dict[str, Any] | None = None

    @field_validator("created_at", mode="before")
    @classmethod
    def _ist(cls, v: Any) -> Any:  # stored as naive UTC; the API speaks IST
        return datetime.fromisoformat(to_ist(v)) if v is not None else v

    @field_validator("triggers", mode="before")
    @classmethod
    def _parse(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v) if v else []
        return v or []


class ActiveAlertsResponse(BaseModel):
    generated_at: datetime
    total: int
    returned: int
    alerts: list[AlertOut]


class AckResponse(BaseModel):
    alert_id: str
    status: Literal["ACKNOWLEDGED"]
