"""Section 11 risk-display rules: EL for credit products, N/A (not zero) elsewhere, labels, PD basis, trends."""

from __future__ import annotations

import numpy as np

from api.presentation import account_display
from config.registry import REGISTRY
from config.settings import RiskSettings
from models.behavior import trend_explanation
from models.risk import AccountSnapshot, evaluate_risk, monotone_quantiles, money_na, pd_display, tier_label, trigger_info


def _snap(limit, principal, cibil=700.0):
    n = len(limit)
    return AccountSnapshot(
        current_balance=np.full(n, 1_000.0), current_utilization=np.full(n, 0.2), credit_limit=np.asarray(limit, float),
        outstanding_principal=np.asarray(principal, float), lgd=np.full(n, 0.5), cibil=np.full(n, cibil),
        missed_salary=np.zeros(n), payment_return=np.zeros(n), cibil_change_90d=np.zeros(n))


def _risk(limit, principal, cfg=None, emi=None):
    n, h = len(limit), 30
    q = np.broadcast_to(np.linspace(800, 1_200, 9), (n, h, 9)).copy()
    util = np.full((n, h, 9), 0.2)
    return evaluate_risk(q, util, np.zeros((n, h)) if emi is None else emi, _snap(limit, principal), cfg or RiskSettings())


def test_item14_el_for_every_credit_product_and_na_for_the_rest():
    # credit card, loan, debit/prepaid/deposit (no limit, no principal)
    r = _risk([5_000, 0, 0], [0, 20_000, 0])
    assert np.isfinite(r.expected_loss[:2]).all() and (r.expected_loss[:2] > 0).all()
    assert np.isnan([r.expected_loss[2], r.ead[2], r.pd[2], r.lgd[2]]).all()  # N/A, never 0
    assert r.priority[0] > 0 and r.priority[2] == 0 and r.tier.tolist() == ["GREEN"] * 3
    assert r.credit_product.tolist() == [True, True, False]


def test_item14_a_scheduled_emi_makes_it_a_credit_product():
    emi = np.zeros((1, 30))
    emi[0, 10] = 100.0
    assert _risk([0], [0], emi=emi).credit_product.tolist() == [True]


def test_item14_no_bureau_data_drops_the_term_instead_of_imputing_a_score():
    with_b = _risk([5_000], [0])
    without = _risk([5_000], [0], RiskSettings(bureau_enabled=False))
    assert np.isnan(without.pd_bureau[0]) and np.isfinite(with_b.pd_bureau[0])
    assert without.pd[0] < with_b.pd[0] and np.isclose(without.pd[0], 0.0)  # no cash/util driver here either


def test_projected_tier_wording_and_green_is_not_a_guarantee():
    assert tier_label("SMA-1") == "Projected SMA-1" and tier_label("GREEN") == "No projected shortfall"
    d = account_display(_account(), REGISTRY["sandbox"], RiskSettings(), 90)
    assert d["green_note"] and "not a guarantee" in d["green_note"] and d["tier_tooltip"] is None
    d = account_display({**_account(), "risk_tier": "SMA-1"}, REGISTRY["sandbox"], RiskSettings(), 90)
    assert d["tier_label"] == "Projected SMA-1" and "days already overdue" in d["tier_tooltip"]


def test_pd_is_labelled_bureau_based_when_the_bureau_term_dominates():
    cfg = RiskSettings()
    cards = pd_display(0.235, 0.0, 0.0, 0.94, cfg, "FICO")
    assert cards["basis"] == "bureau-based estimate, not calibrated" and cards["calibrated"] is False
    assert [c["id"] for c in cards["components"]] == ["cash_shortfall", "utilization_breach", "bureau"]
    assert cards["components"][2]["weight"] == cfg.bureau_pd_weight and cards["components"][2]["label"].startswith("FICO")
    loan = pd_display(0.4, 0.35, 0.1, 0.1, cfg, "CIBIL")
    assert loan["basis"].startswith("forecast")
    none = pd_display(0.1, 0.1, 0.0, None, cfg, None)  # Berka: no bureau
    assert none["components"][2]["value"] is None and none["components"][2]["contribution"] is None


def test_card_trigger_wording():
    v = REGISTRY["cards"].labels
    assert trigger_info("CASH_TO_ZERO_DAY_3", v)["label"] == "Spend below floor (day 3)"
    ret = trigger_info("RECENT_PAYMENT_RETURN", v)
    assert "insufficient-funds decline" in ret["label"] and "not a returned payment" in ret["detail"]
    assert trigger_info("RECENT_PAYMENT_RETURN", REGISTRY["sandbox"].labels)["label"] == "Recent payment return"
    assert REGISTRY["cards"].labels["attrition_label"] == "Spending decline"
    assert REGISTRY["berka"].labels["attrition_label"] == "Deposit attrition"


def test_missing_values_are_na_with_a_reason_not_zero():
    m = money_na(None)
    assert m["value"] is None and m["na_reason"]
    assert money_na(float("nan"))["value"] is None and money_na(0.0) == {"value": 0.0, "na_reason": None}
    debit = {**_account(), "product_type": "DEBIT_CARD", "exposure_at_default": None, "expected_loss": None,
             "probability_of_default": None, "priority_score": 0.0}
    d = account_display(debit, REGISTRY["cards"], RiskSettings(), 90)
    assert d["ead"]["value"] is None and "not credit exposures" in d["ead"]["na_reason"]
    assert d["expected_loss"]["value"] is None and d["priority"]["score"] is None


def test_trend_names_both_averages_and_the_floor_is_labelled():
    t = trend_explanation(1_640.0, 1_234.0, -24.7, 90, "Rolling 30-day net spend")
    assert t["label"] == "DECLINING" and "1,234" in t["explanation"] and "1,640" in t["explanation"]
    assert "not a comparison with today" in t["explanation"]
    d = account_display({**_account(), "triggers": ["CASH_TO_ZERO_DAY_1"]}, REGISTRY["cards"],
                        RiskSettings(zero_balance_threshold=500.0), 90)
    assert "Floor: 500 USD" in d["triggers"][0]["detail"] and "floor, not zero" in d["triggers"][0]["detail"]


def test_quantile_order_is_enforced():
    q = np.array([[[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]]])
    fixed, n = monotone_quantiles(q)
    assert n == 1 and (np.diff(fixed, axis=-1) >= 0).all()


def _account():
    return {"account_id": "X1", "product_type": "AUTO", "risk_tier": "GREEN", "triggers": [], "priority_score": 5.0,
            "probability_of_default": 0.2, "exposure_at_default": 1_000.0, "expected_loss": 100.0,
            "pd_components": {"cash_shortfall": 0.0, "utilization_breach": 0.0, "bureau": 0.5},
            "behavior": {"trailing_avg_balance_90d": 100.0, "projected_avg_balance": 100.0,
                         "projected_change_pct": 0.0, "attrition_alert": False}}
