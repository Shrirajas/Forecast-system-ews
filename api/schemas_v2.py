"""Pydantic contracts for the dataset / model / forecast / compare endpoints (API_CONTRACT.md v1)."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from api.schemas import AccountId


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorOut(BaseModel):
    detail: ErrorDetail


class SelectDatasetRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=32)


class SelectModelRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=32)
    allow_download: bool = Field(False, description="Allow fetching pretrained weights that are not cached yet")


class DatasetsOut(BaseModel):
    active: str | None
    datasets: list[dict[str, Any]]


class SelectDatasetOut(BaseModel):
    active: str
    dataset: dict[str, Any]
    model: dict[str, Any]
    job: dict[str, Any] | None = None


class ModelsOut(BaseModel):
    dataset_id: str | None
    active: str
    models: list[dict[str, Any]]


class SelectModelOut(BaseModel):
    active: str
    model: dict[str, Any]


class AccountItem(BaseModel):
    account_id: str
    customer_id: str | None
    product_type: str | None
    segment: str | None
    rm_id: str | None
    credit_limit: float | None
    open_date: date | None
    has_credit_line: bool


class AccountsOut(BaseModel):
    dataset_id: str
    total: int
    returned: int
    offset: int
    product_types: dict[str, int]
    accounts: list[AccountItem]


class ForecastRequest(BaseModel):
    dataset_id: str | None = None
    model_id: str | None = None
    account_id: AccountId
    horizon: Literal[30, 60, 90] = 90
    target: str | None = Field(None, description="Target id from the dataset; default = its primary target")
    as_of_date: date | None = None
    history_days: int = Field(180, ge=30, le=512)


class ForecastOut(BaseModel):
    dataset: dict[str, Any]
    model: dict[str, Any]
    account: dict[str, Any]
    target: dict[str, Any]
    horizon: int
    origin: date
    quantile_levels: list[float]
    history: dict[str, list]
    forecast: dict[str, Any]
    secondary: dict[str, Any] | None = None
    emi_schedule: list[dict[str, Any]] | None = None
    summary: dict[str, Any]
    risk: dict[str, Any] | None = None
    risk_unavailable_reason: str | None = None
    description: str
    timing: dict[str, Any]
    warnings: list[str] = []
    persisted: bool = False


class CompareRequest(BaseModel):
    dataset_id: str | None = None
    scope: Literal["account", "dataset_sample"] = "account"
    account_id: AccountId | None = None
    horizon: Literal[30, 60, 90] = 90
    origins: list[date] | None = None
    models: list[Literal["stformer", "timesfm3"]] = Field(default_factory=lambda: ["stformer", "timesfm3"], min_length=1, max_length=2)
    seed: int = Field(7, description="Seed of the random account holdout (used when no customer-level split file exists)")
