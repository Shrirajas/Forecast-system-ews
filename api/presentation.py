"""Server-side labels for the console: tier wording, N/A reasons, PD basis, triggers, alert categories.

Pure functions over the account / alert dicts the pipeline already produces, so the stored format is unchanged.
"""

from __future__ import annotations

from typing import Any

from config.registry import _vocab, DatasetSpec
from config.settings import RiskSettings
from models.behavior import trend_explanation
from models.risk import GREEN_NOTE, TIER_TOOLTIP, money_na, pd_display, tier_label, trigger_info

CATEGORY_LABELS = {
    "predictive_warning": "Predictive warning", "spending_behaviour_change": "Spending behaviour change",
    "repayment_failure": "Actual repayment failure", "regulatory_classification": "Regulatory classification",
}
ALERT_CATEGORY = {"CREDIT_EWS": "predictive_warning", "DEPOSIT_ATTRITION": "spending_behaviour_change"}
CATEGORY_ALERT = {v: k for k, v in ALERT_CATEGORY.items()}
FORMULA = {
    "expected_loss": "expected loss (EAD x PD x LGD) x 0.5^(days to first EMI shortfall or cash floor / 30)",
    "cost_of_funds": "annual cost of replacing the projected balance drop (drop x funding spread); ranks within its own type",
}
NA_DEBIT = "Debit and prepaid cards are not credit exposures: no PD, EAD, LGD or expected loss."


def vocab(spec: DatasetSpec | None, cfg: RiskSettings) -> dict[str, str]:
    v = dict(spec.labels) if spec else _vocab()
    v["floor"] = f"{cfg.zero_balance_threshold:,.0f}"
    return v


def _triggers(codes: list[str], v: dict[str, str], unit: str) -> list[dict]:
    out = [trigger_info(c, v) for c in codes]
    for t in out:
        if t["code"].startswith("CASH_TO_ZERO"):
            t["detail"] = f"{t['detail']} Floor: {v['floor']} {unit}.".strip()
    return out


def _na(product: str | None) -> str | None:
    return NA_DEBIT if product in ("DEBIT_CARD", "PREPAID_CARD") else None


def account_display(a: dict[str, Any], spec: DatasetSpec | None, cfg: RiskSettings, horizon: int) -> dict[str, Any]:
    v, unit = vocab(spec, cfg), spec.currency.code if spec else ""
    pdc, beh = a["pd_components"], a["behavior"]
    tier, attr = a["risk_tier"], bool(beh.get("attrition_alert"))
    cat = "predictive_warning" if tier != "GREEN" else ("spending_behaviour_change" if attr else None)
    na = _na(a.get("product_type"))
    credit = a.get("exposure_at_default") is not None
    bureau = spec.capabilities.get("bureau_name") if spec else None
    return {
        "tier_label": tier_label(tier), "tier_tooltip": TIER_TOOLTIP if tier != "GREEN" else None,
        "green_note": GREEN_NOTE if tier == "GREEN" else None,
        "alert_label": None if cat is None else (v["credit_alert_label"] if cat == "predictive_warning" else v["attrition_label"]),
        "category": cat, "category_label": CATEGORY_LABELS.get(cat),
        "priority": {"score": a.get("priority_score") if credit else None, "basis": "expected_loss" if credit else None,
                     "formula": FORMULA["expected_loss"] if credit else None},
        "pd": pd_display(a.get("probability_of_default"), pdc.get("cash_shortfall"), pdc.get("utilization_breach"),
                         pdc.get("bureau"), cfg, bureau),
        "ead": money_na(a.get("exposure_at_default"), na), "expected_loss": money_na(a.get("expected_loss"), na),
        "triggers": _triggers(a.get("triggers", []), v, unit),
        "trend": trend_explanation(beh.get("trailing_avg_balance_90d"), beh.get("projected_avg_balance"),
                                   beh.get("projected_change_pct"), horizon, v["series"]),
    }


def alert_display(al: dict[str, Any], spec: DatasetSpec | None, cfg: RiskSettings) -> dict[str, Any]:
    v, unit = vocab(spec, cfg), spec.currency.code if spec else ""
    kind = al["alert_type"]
    cat = ALERT_CATEGORY[kind]
    basis = "expected_loss" if kind == "CREDIT_EWS" else "cost_of_funds"
    na = _na(al.get("product_type"))
    return {
        "tier_label": tier_label(al["risk_tier"]) if kind == "CREDIT_EWS" else None,
        "tier_tooltip": TIER_TOOLTIP if kind == "CREDIT_EWS" else None,
        "alert_label": v["credit_alert_label"] if kind == "CREDIT_EWS" else v["attrition_label"],
        "category": cat, "category_label": CATEGORY_LABELS[cat],
        "priority": {"score": al.get("priority_score"), "basis": basis, "formula": FORMULA[basis]},
        "ead": money_na(al.get("exposure_at_default"), na), "expected_loss": money_na(al.get("expected_loss"), na if kind == "CREDIT_EWS" else
                                                                                      "Behaviour alerts carry no expected loss (it lives on the credit alert)."),
        "pd": {"value": al.get("probability_of_default"), "calibrated": False},
        "triggers": _triggers(al.get("triggers", []), v, unit),
    }
