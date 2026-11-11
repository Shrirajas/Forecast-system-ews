"""Dataset and model registry: what each dataset/model is, can forecast and needs.

Static facts live here as data; dynamic facts (account count, date range, checkpoint present, model
status) are read from the cache ``meta.json``, the checkpoint sidecar and the engine status, never by
opening a DuckDB file. The UI reads all of it from the API and hardcodes none of it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from config.settings import PROJECT_ROOT, CacheSettings, Settings

Category = Literal["predictive_warning", "spending_behaviour_change", "repayment_failure", "regulatory_classification"]
HORIZONS = (30, 60, 90)
TIMESFM_BADGE = "TimesFM 3.0 · research licence · prototype only"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Currency(Frozen):
    code: str
    symbol: str


class TargetSpec(Frozen):
    id: str
    label: str
    unit: str
    channel: Literal["eod_balance", "utilization"]
    kind: Literal["primary", "secondary"]
    description: str
    available: bool = True
    reason: str | None = None
    applies_to: tuple[str, ...] = ()  # product types; empty = all


class FeatureSpec(Frozen):
    id: str
    label: str
    available: bool
    reason: str | None = None
    source: str | None = None


class AlertTypeSpec(Frozen):
    id: Literal["CREDIT_EWS", "DEPOSIT_ATTRITION"]
    label: str
    category: Category
    description: str


class DatasetSpec(Frozen):
    id: str
    display_name: str
    description: str
    source_kind: Literal["synthetic", "real"]
    source_name: str
    source_notes: str
    currency: Currency
    duckdb_path: Path
    cache_root: Path
    checkpoint_path: Path
    targets: tuple[TargetSpec, ...]
    features: tuple[FeatureSpec, ...]
    capabilities: dict[str, Any]
    supported_models: tuple[str, ...] = ("stformer", "timesfm3")
    default_model: str = "stformer"
    alert_types: tuple[AlertTypeSpec, ...]
    labels: dict[str, str]
    risk_overrides: dict[str, Any] = Field(default_factory=dict)
    behavior_overrides: dict[str, Any] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    @property
    def primary_target(self) -> TargetSpec:
        return next(t for t in self.targets if t.kind == "primary")

    def target(self, target_id: str | None) -> TargetSpec | None:
        return self.primary_target if target_id is None else next((t for t in self.targets if t.id == target_id), None)


class ModelSpec(Frozen):
    id: str
    display_name: str
    backend: str
    kind: Literal["trained", "pretrained_zero_shot"]
    description: str
    supported_datasets: tuple[str, ...]
    targets: tuple[str, ...]  # channels it can forecast
    horizons: tuple[int, ...] = HORIZONS
    supports_quantiles: bool = True
    supports_training: bool
    licence_name: str
    licence_restriction: str | None = None
    badge: str | None = None
    checkpoint: str | None = None  # pretrained model id (zero-shot models)


# ---------------------------------------------------------------------------------------- vocabularies
def _vocab(**kw: str) -> dict[str, str]:
    base = {
        "zero_floor_label": "Cash near zero (day {day})", "zero_floor_detail": "Worst-case (q10) balance at or below the floor.",
        "return_label": "Recent payment return", "return_detail": "A payment was returned in the latest data.",
        "attrition_label": "Deposit attrition", "credit_alert_label": "Predictive credit warning",
        "bureau_name": "", "series": "End-of-day balance", "series_short": "Balance", "account_noun": "account",
    }
    return {**base, **kw}


def build_registry(root: Path = PROJECT_ROOT) -> dict[str, DatasetSpec]:
    d, m = root / "data", root / "artifacts" / "models"
    card_types = ("CREDIT_CARD", "DEBIT_CARD", "PREPAID_CARD")
    cards = DatasetSpec(
        id="cards", display_name="Cards (IBM TabFormer)",
        description="Synthetic US card transactions. The forecast target is each card's rolling 30-day net spend, "
                    "not an account balance and not outstanding debt.",
        source_kind="synthetic", source_name="IBM TabFormer synthetic credit-card transactions + sd254 card/user files",
        source_notes="Simulated people, no real customers. FICO scores, limits and card types come from the sd254 files.",
        currency=Currency(code="USD", symbol="$"),
        duckdb_path=d / "cards" / "ews.duckdb", cache_root=d / "cards" / "vector_cache", checkpoint_path=m / "cards_stformer.pt",
        targets=(
            TargetSpec(id="rolling_30d_net_spend", label="Rolling 30-day net spend", unit="USD", channel="eod_balance",
                       kind="primary", applies_to=card_types,
                       description="Sum of approved card spend over the last 30 days, refunds subtracted. It is not an account "
                                   "balance and not outstanding credit-card debt."),
            TargetSpec(id="estimated_utilization", label="Estimated utilisation", unit="ratio", channel="utilization",
                       kind="secondary", applies_to=("CREDIT_CARD",),
                       description="30-day spend divided by the credit limit; assumes each bill is paid in full every month "
                                   "(no payment data exists). Credit cards only."),
        ),
        features=(
            FeatureSpec(id="credit_limit", label="Credit limit", available=True, source="sd254_cards (credit cards only)"),
            FeatureSpec(id="bureau_score", label="FICO score", available=True, source="FICO, one snapshot per user (not CIBIL)"),
            FeatureSpec(id="emi", label="EMI / instalments", available=False, reason="The card data has no loan schedules."),
            FeatureSpec(id="loan_outcomes", label="Loan outcomes", available=False, reason="No loans in the card data."),
            FeatureSpec(id="declined_insufficient_funds", label="Insufficient-funds declines", available=True,
                        source="card transactions (declined, not returned payments)"),
        ),
        capabilities=dict(credit_limit=True, utilization=True, bureau=True, bureau_name="FICO", loan_schedules=False, emi=False,
                          projected_sma=False, expected_loss=True, loan_outcomes=False, deposit_attrition=False, spending_decline=True),
        alert_types=(
            AlertTypeSpec(id="CREDIT_EWS", label="Predictive credit warning", category="predictive_warning",
                          description="Forecast cash/limit stress. Cards have no EMIs, so this rarely fires."),
            AlertTypeSpec(id="DEPOSIT_ATTRITION", label="Spending decline", category="spending_behaviour_change",
                          description="Projected 30-day spend well below the card's own recent level. Not deposit attrition."),
        ),
        labels=_vocab(
            zero_floor_label="Spend below floor (day {day})",
            zero_floor_detail="Worst-case (q10) 30-day spend falls to or below the spend floor. It is a floor, not zero.",
            return_label="Recent insufficient-funds decline",
            return_detail="A card transaction was declined for insufficient balance. It is a declined transaction, not a returned payment.",
            attrition_label="Spending decline", bureau_name="FICO", series="Rolling 30-day net spend",
            series_short="30-day spend", account_noun="card"),
        risk_overrides={"zero_balance_threshold": 500.0},
        behavior_overrides={"attrition_min_balance": 500.0},
        limitations=(
            "The target is rolling 30-day net spend, not a balance; there is no bill-payment data.",
            "No loans or EMIs: projected SMA tiers and EMI forecasts do not apply.",
            "PD is essentially a bureau (FICO) estimate and is not calibrated against defaults.",
            "FICO is a single snapshot treated as known for the whole period.",
        ),
    )
    berka = DatasetSpec(
        id="berka", display_name="Berka (Czech bank)",
        description="Real anonymised Czech bank data, 1993-1998: daily account balances, 682 loans with instalment schedules and loan outcomes.",
        source_kind="real", source_name="PKDD'99 Discovery Challenge (Berka) financial dataset",
        source_notes="Loan status is an end-of-data label (A/B finished, C/D running; B and D are bad) on only 682 loans.",
        currency=Currency(code="CZK", symbol="Kč"),
        duckdb_path=d / "berka" / "ews.duckdb", cache_root=d / "berka" / "cache", checkpoint_path=m / "berka_stformer.pt",
        targets=(
            TargetSpec(id="eod_balance", label="End-of-day account balance", unit="CZK", channel="eod_balance", kind="primary",
                       description="The bank's reported balance after the day's last transaction, carried forward over days with "
                                   "no transactions. Negative values are overdrafts."),
            TargetSpec(id="utilization", label="Credit-line utilisation", unit="ratio", channel="utilization", kind="secondary",
                       description="Drawn amount divided by the credit limit.", available=False,
                       reason="The data has no credit limits or overdraft limits."),
        ),
        features=(
            FeatureSpec(id="credit_limit", label="Credit limit", available=False, reason="No overdraft or credit-limit data."),
            FeatureSpec(id="bureau_score", label="Bureau score", available=False, reason="The dataset has no bureau scores."),
            FeatureSpec(id="emi", label="Loan instalments (EMI)", available=True, source="loan.asc payments, built by the adapter"),
            FeatureSpec(id="loan_outcomes", label="Loan outcomes (A/B/C/D)", available=True, source="data/berka/parquet/loans.parquet"),
        ),
        capabilities=dict(credit_limit=False, utilization=False, bureau=False, bureau_name=None, loan_schedules=True, emi=True,
                          projected_sma=True, expected_loss=True, loan_outcomes=True, deposit_attrition=True, spending_decline=False),
        alert_types=(
            AlertTypeSpec(id="CREDIT_EWS", label="Predictive credit warning", category="predictive_warning",
                          description="Forecast balance below a scheduled loan instalment (projected SMA tier)."),
            AlertTypeSpec(id="DEPOSIT_ATTRITION", label="Deposit attrition", category="spending_behaviour_change",
                          description="Projected balance well below the account's own earlier level while income keeps arriving."),
        ),
        labels=_vocab(bureau_name="", series="End-of-day balance (CZK)", series_short="Balance", account_noun="account"),
        risk_overrides={"bureau_enabled": False},
        limitations=(
            "No bureau score: the PD bureau component is not available and is not imputed.",
            "Outstanding principal is estimated by the adapter (amount minus instalments paid), ignoring interest.",
            "Loan status is an end-of-data label and there are only 682 loans: tier/PD separation is indicative.",
        ),
    )
    sandbox = DatasetSpec(
        id="sandbox", display_name="Sandbox (generated EWS data)",
        description="Generated retail and SME loan book used to demonstrate the original balance, EMI and EWS workflow.",
        source_kind="synthetic", source_name="data_engine/synth.py generator", source_notes="Synthetic; INR; loans, EMIs, credit lines and bureau scores.",
        currency=Currency(code="INR", symbol="₹"),
        duckdb_path=d / "ews.duckdb", cache_root=d / "vector_cache", checkpoint_path=m / "ews_stformer.pt",
        targets=(
            TargetSpec(id="eod_balance", label="End-of-day account balance", unit="INR", channel="eod_balance", kind="primary",
                       description="Current-account end-of-day balance."),
            TargetSpec(id="utilization", label="Credit-line utilisation", unit="ratio", channel="utilization", kind="secondary",
                       description="Drawn amount divided by the credit limit."),
        ),
        features=(
            FeatureSpec(id="credit_limit", label="Credit limit", available=True),
            FeatureSpec(id="bureau_score", label="CIBIL score", available=True, source="synthetic CIBIL"),
            FeatureSpec(id="emi", label="Loan instalments (EMI)", available=True),
            FeatureSpec(id="loan_outcomes", label="Default labels", available=True, source="labels.parquet (synthetic)"),
        ),
        capabilities=dict(credit_limit=True, utilization=True, bureau=True, bureau_name="CIBIL", loan_schedules=True, emi=True,
                          projected_sma=True, expected_loss=True, loan_outcomes=True, deposit_attrition=True, spending_decline=False),
        alert_types=(
            AlertTypeSpec(id="CREDIT_EWS", label="Predictive credit warning", category="predictive_warning",
                          description="Forecast balance below a scheduled EMI (projected SMA tier)."),
            AlertTypeSpec(id="DEPOSIT_ATTRITION", label="Deposit attrition", category="spending_behaviour_change",
                          description="Projected balance well below the account's own earlier level."),
        ),
        labels=_vocab(bureau_name="CIBIL", series="End-of-day balance (INR)"),
        limitations=("Synthetic data: results show the workflow, not real credit behaviour.",),
    )
    return {s.id: s for s in (sandbox, cards, berka)}


def build_models(datasets: dict[str, DatasetSpec]) -> dict[str, ModelSpec]:
    ids = tuple(datasets)
    return {
        "stformer": ModelSpec(
            id="stformer", display_name="STFormer", backend="stformer", kind="trained", supported_datasets=ids,
            targets=("eod_balance", "utilization"), supports_training=True, licence_name="Project code (this repository)",
            description="The project's own spatial-temporal transformer, trained locally per dataset; forecasts 9 quantiles of "
                        "balance and utilisation up to 90 days ahead."),
        "timesfm3": ModelSpec(
            id="timesfm3", display_name="TimesFM 3.0", backend="timesfm3", kind="pretrained_zero_shot", supported_datasets=ids,
            targets=("eod_balance", "utilization"), supports_training=False, checkpoint="google/timesfm-3.0-pytorch",
            licence_name="TimesFM Non-Commercial License v1.0",
            licence_restriction="Weights may be used for testing, evaluation and research only, not in production or commercially.",
            badge=TIMESFM_BADGE,
            description="Google's pretrained time-series foundation model, run zero-shot (no training on this data)."),
    }


REGISTRY: dict[str, DatasetSpec] = build_registry()
MODELS: dict[str, ModelSpec] = build_models(REGISTRY)


# ------------------------------------------------------------------------------------------ dynamic facts
def cache_meta(spec: DatasetSpec) -> dict[str, Any] | None:
    cur = spec.cache_root / "CURRENT"
    try:
        return json.loads((spec.cache_root / cur.read_text().strip() / "meta.json").read_text())
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=8)
def _accounts(path: str, version: str):  # version is part of the key: a rebuilt cache is re-read
    return pq.read_table(path).to_pandas()


def account_table(spec: DatasetSpec):
    meta = cache_meta(spec)
    if meta is None:
        return None
    return _accounts(str(spec.cache_root / meta["version"] / "accounts.parquet"), meta["version"])


def resolve_checkpoint(spec: DatasetSpec) -> Path:
    """Newest trained checkpoint of this dataset: a ``<id>_stformer.latest.json`` pointer written after a
    console training run, else the registered default. Existing checkpoint files are never overwritten."""
    ptr = spec.checkpoint_path.with_name(f"{spec.id}_stformer.latest.json")
    try:
        p = Path(json.loads(ptr.read_text())["checkpoint"])
        if p.exists():
            return p
    except (OSError, ValueError, KeyError):
        pass
    return spec.checkpoint_path


def settings_for_dataset(base: Settings, spec: DatasetSpec) -> Settings:
    """``base`` pointed at one dataset: its database, cache, checkpoint and risk/behaviour calibration."""
    return base.model_copy(update={
        "storage": base.storage.model_copy(update={"backend": "duckdb", "duckdb_path": spec.duckdb_path}),
        "cache": CacheSettings(root=spec.cache_root),
        "model": base.model.model_copy(update={"checkpoint_path": resolve_checkpoint(spec)}),
        "risk": base.risk.model_copy(update=spec.risk_overrides),
        "behavior": base.behavior.model_copy(update=spec.behavior_overrides),
    })


def match_dataset(settings: Settings, registry: dict[str, DatasetSpec] = REGISTRY) -> str | None:
    """Which registered dataset a settings object points at (by database path), else None = custom database."""
    if settings.storage.backend != "duckdb":
        return None
    here = Path(settings.storage.duckdb_path).resolve()
    return next((s.id for s in registry.values() if s.duckdb_path.resolve() == here), None)


def dataset_status(spec: DatasetSpec) -> dict[str, Any]:
    db, meta = spec.duckdb_path.exists(), cache_meta(spec)
    state, reason = "ready", None
    if not db:
        state, reason = "missing_database", f"Database not found at {spec.duckdb_path.relative_to(PROJECT_ROOT) if spec.duckdb_path.is_relative_to(PROJECT_ROOT) else spec.duckdb_path}."
    elif meta is None:
        state, reason = "missing_cache", "Feature cache not built yet (refresh the dataset)."
    return {"available": state == "ready", "state": state, "reason": reason, "database_exists": db,
            "cache_exists": meta is not None, "checkpoint_exists": resolve_checkpoint(spec).exists()}


def dataset_info(spec: DatasetSpec, active: str | None) -> dict[str, Any]:
    meta, acc = cache_meta(spec), account_table(spec)
    return {
        "id": spec.id, "display_name": spec.display_name, "description": spec.description, "active": spec.id == active,
        "source": {"kind": spec.source_kind, "name": spec.source_name, "notes": spec.source_notes},
        "currency": spec.currency.model_dump(), "status": dataset_status(spec),
        "data": None if meta is None else {"accounts": meta["n_accounts"], "first_day": meta["start_date"],
                                           "last_day": meta["end_date"], "days": meta["n_days"], "cache_version": meta["version"]},
        "primary_target": spec.primary_target.id,
        "targets": [{**t.model_dump(exclude={"applies_to"}), "currency": spec.currency.code if t.unit == spec.currency.code else None,
                     "applies_to": list(t.applies_to)} for t in spec.targets],
        "features": [f.model_dump() for f in spec.features],
        "capabilities": dict(spec.capabilities),
        "supported_models": list(spec.supported_models), "default_model": spec.default_model,
        "alert_types": [a.model_dump() for a in spec.alert_types],
        "labels": dict(spec.labels),
        "risk": {"zero_floor": spec.risk_overrides.get("zero_balance_threshold", 500.0),
                 "high_risk_pd_threshold": spec.risk_overrides.get("high_risk_pd_threshold", 0.20),
                 "pd_basis_label": "bureau-based estimate, not calibrated" if spec.capabilities["bureau"] else "forecast cash-shortfall + utilisation drivers"},
        "product_types": {} if acc is None else {k: int(v) for k, v in acc["product_type"].value_counts().items()},
        "limitations": list(spec.limitations),
    }
