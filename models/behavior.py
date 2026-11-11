"""Customer behaviour engine: balance trajectories and deposit-attrition alerts.

A falling balance is not always credit stress. When the median (q50)
trajectory projects the average balance over the horizon to fall well below
the customer's established balance (the average over days -180..-30, which
excludes the most recent month so an exit already under way is still
measured against the pre-exit level), while payroll keeps arriving and no
payment has bounced, the customer is probably moving deposits to another
bank. Losing low-cost CASA balances means replacing them with dearer
funding, so the alert carries an annualised cost-of-funds impact:

``cof_impact_annual = projected_balance_drop * replacement_funding_spread``

Attrition alerts can coexist with an SMA tier. A customer who has moved
funds out can be left too thin to meet the next EMI, and the relationship
manager needs both signals.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from config.settings import QUANTILES, BehaviorSettings

MEDIAN = QUANTILES.index(0.5)


@dataclass
class BehaviorResult:
    trailing_avg_90d: np.ndarray
    baseline_avg: np.ndarray
    projected_avg: np.ndarray
    projected_change_pct: np.ndarray  # vs trailing 90-day average
    projected_min_worst: np.ndarray
    trend: np.ndarray  # RISING / STABLE / DECLINING
    attrition_alert: np.ndarray  # bool
    projected_balance_drop: np.ndarray  # vs baseline, only where attrition is flagged
    cof_impact_annual: np.ndarray


def _nanmean(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows (no history in window)
        return np.nanmean(a, axis=1)


def evaluate_behavior(
    past_balance: np.ndarray,
    q_bal: np.ndarray,
    stress_evidence: np.ndarray,
    cfg: BehaviorSettings,
    liability_index: int = 0,
) -> BehaviorResult:
    """past_balance ``[B, L]`` (NaN = missing), q_bal ``[B, H, 9]``,
    stress_evidence ``[B]`` bool (missed salary / recent payment return)."""
    trailing = np.nan_to_num(_nanmean(past_balance[:, -90:]))
    baseline = _nanmean(past_balance[:, -180:-30])
    baseline = np.where(np.isnan(baseline), trailing, baseline)
    projected = q_bal[..., MEDIAN].mean(1)
    change = (projected - trailing) / np.maximum(np.abs(trailing), 1.0)
    drop_vs_base = (baseline - projected) / np.maximum(np.abs(baseline), 1.0)
    trend = np.where(change <= -0.10, "DECLINING", np.where(change >= 0.10, "RISING", "STABLE")).astype(object)
    attrition = (
        ~np.asarray(stress_evidence, dtype=bool)
        & (baseline >= cfg.attrition_min_balance)
        & (drop_vs_base >= cfg.attrition_drop_threshold)
    )
    drop = np.clip(baseline - projected, 0.0, None)
    return BehaviorResult(
        trailing_avg_90d=trailing,
        baseline_avg=baseline,
        projected_avg=projected,
        projected_change_pct=change,
        projected_min_worst=q_bal[..., liability_index].min(1),
        trend=trend,
        attrition_alert=attrition,
        projected_balance_drop=np.where(attrition, drop, 0.0),
        cof_impact_annual=np.where(attrition, drop * cfg.replacement_funding_spread, 0.0),
    )


def trend_explanation(trailing: float | None, projected: float | None, change_pct: float | None, horizon: int,
                      series: str) -> dict:
    """Name both averages behind a trend label: the forecast mean is NOT compared with today's value."""
    if trailing is None or projected is None:
        return {"label": None, "change_pct": None, "forecast_mean": projected, "trailing_mean_90d": trailing,
                "explanation": "Not enough history to compare."}
    label = "DECLINING" if change_pct is not None and change_pct <= -10 else (
        "RISING" if change_pct is not None and change_pct >= 10 else "STABLE")
    return {
        "label": label, "change_pct": change_pct, "forecast_mean": projected, "trailing_mean_90d": trailing,
        "explanation": (f"Mean of the {horizon}-day median forecast of {series} ({projected:,.0f}) vs the mean of the "
                        f"last 90 days ({trailing:,.0f}); not a comparison with today's value."),
    }


def attrition_label(kind: str, vocab: dict) -> str:
    return vocab["attrition_label"] if kind == "DEPOSIT_ATTRITION" else vocab["credit_alert_label"]
