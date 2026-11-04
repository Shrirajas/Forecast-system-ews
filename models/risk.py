"""Predictive risk engine: SMA classification, PD, EAD, LGD and expected loss.

Everything is vectorised over the account batch.

**EWS trigger.** On the liability (worst-case cash) quantile track, default
q10, the balance available going into each scheduled EMI date (previous
day's EOD) is compared with that EMI. The first shortfall within the
evaluation horizon sets ``days_to_breach``, counted in days from the
as-of date, and the tier:

=========  =========================
SMA-0      shortfall 1-30 days out
SMA-1      shortfall 31-60 days out
SMA-2      shortfall 61-90 days out
GREEN      no shortfall in the horizon
=========  =========================

**PD.** Three default drivers are combined with a noisy-OR:

* ``pd_cash``: probability, from the full predictive distribution (the 9
  quantiles with exponential tails), that the balance falls short of an EMI,
  taken as the maximum over due dates (comonotonic: balances on nearby days
  move together),
* ``p_util``: probability that utilisation exceeds the breach level on any
  day of the horizon,
* ``pd_bureau``: a logistic mapping of the latest CIBIL score.

``PD = 1 - (1 - pd_cash)(1 - w_u * p_util)(1 - w_b * pd_bureau)``

**EAD** = outstanding loan principal (including arrears) + credit-line
exposure, taking the larger of today's drawn amount and the asset-track
(default q90) projected drawn amount at the breach day (or horizon end).

**Expected loss** = EAD x PD x LGD, computed for accounts in an SMA tier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.settings import QUANTILES, RiskSettings

TAUS = np.asarray(QUANTILES, dtype=np.float64)
TIERS = ("GREEN", "SMA-0", "SMA-1", "SMA-2")
TIER_BUCKETS = ((1, 30, "SMA-0"), (31, 60, "SMA-1"), (61, 90, "SMA-2"))


def quantile_cdf(q: np.ndarray, x: np.ndarray, taus: np.ndarray = TAUS) -> np.ndarray:
    """P(Y <= x) from quantile values ``q [..., Q]`` (piecewise-linear interior,
    exponential tails). ``x`` broadcasts against ``q[..., 0]``."""
    q = np.sort(np.asarray(q, dtype=np.float64), axis=-1)
    x = np.broadcast_to(np.asarray(x, dtype=np.float64), q.shape[:-1])
    nq = q.shape[-1]
    k = np.clip((q <= x[..., None]).sum(-1), 1, nq - 1)
    lo = np.take_along_axis(q, (k - 1)[..., None], -1)[..., 0]
    hi = np.take_along_axis(q, k[..., None], -1)[..., 0]
    tl, th = taus[k - 1], taus[k]
    span = hi - lo
    interior = np.where(span > 0, tl + (x - lo) / np.where(span > 0, span, 1.0) * (th - tl), th)

    eps = 1e-9
    b_lo = np.maximum((q[..., 1] - q[..., 0]) / np.log(taus[1] / taus[0]), eps)
    b_hi = np.maximum((q[..., -1] - q[..., -2]) / np.log((1 - taus[-2]) / (1 - taus[-1])), eps)
    with np.errstate(over="ignore", under="ignore"):
        lower = taus[0] * np.exp(np.minimum((x - q[..., 0]) / b_lo, 0.0))
        upper = 1 - (1 - taus[-1]) * np.exp(-np.maximum((x - q[..., -1]) / b_hi, 0.0))
    out = np.where(x < q[..., 0], lower, np.where(x > q[..., -1], upper, interior))
    return np.clip(out, 0.0, 1.0)


@dataclass
class AccountSnapshot:
    """Per-account state at the as-of date (arrays of length B)."""

    current_balance: np.ndarray
    current_utilization: np.ndarray
    credit_limit: np.ndarray
    outstanding_principal: np.ndarray
    lgd: np.ndarray
    cibil: np.ndarray
    missed_salary: np.ndarray
    payment_return: np.ndarray
    cibil_change_90d: np.ndarray


@dataclass
class RiskResult:
    tier: np.ndarray  # str [B]
    days_to_breach: np.ndarray  # int [B], -1 = none
    breach_shortfall: np.ndarray  # EMI - available at breach (worst track)
    days_to_zero: np.ndarray  # int [B], -1 = none
    next_emi_day: np.ndarray  # int [B] (1-based days out), -1 = none in horizon
    next_emi_amount: np.ndarray
    pd: np.ndarray
    pd_cash: np.ndarray
    p_util: np.ndarray
    pd_bureau: np.ndarray
    ead: np.ndarray
    lgd: np.ndarray
    expected_loss: np.ndarray  # NaN for GREEN
    priority: np.ndarray
    util_breach_day: np.ndarray  # int [B], -1 = none
    triggers: list[list[str]]


def _first_true(mask: np.ndarray) -> np.ndarray:
    """1-based index of the first True along the last axis, -1 if none."""
    any_ = mask.any(-1)
    return np.where(any_, mask.argmax(-1) + 1, -1)


def tier_for_days(days: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(days.shape, "GREEN", dtype=object)
    for lo, hi, name in TIER_BUCKETS:
        out[(days >= lo) & (days <= min(hi, horizon))] = name
    return out


def evaluate_risk(
    q_bal: np.ndarray,
    q_util: np.ndarray,
    emi_due: np.ndarray,
    snap: AccountSnapshot,
    cfg: RiskSettings,
    liability_quantile: float | None = None,
    asset_quantile: float | None = None,
) -> RiskResult:
    """q_bal/q_util ``[B, H, 9]`` raw units, emi_due ``[B, H]`` (amount due on each future day)."""
    b, h, _ = q_bal.shape
    li = QUANTILES.index(liability_quantile if liability_quantile is not None else cfg.liability_quantile)
    ai = QUANTILES.index(asset_quantile if asset_quantile is not None else cfg.asset_quantile)

    # Balance available entering day t = EOD of day t-1 (today's actual balance for t=0).
    avail_q = np.concatenate(
        [np.repeat(snap.current_balance[:, None, None], len(QUANTILES), axis=2), q_bal[:, :-1]], axis=1
    )
    due = emi_due > 0
    shortfall_worst = due & (avail_q[..., li] < emi_due)
    dtb = _first_true(shortfall_worst)
    tier = tier_for_days(dtb, h)
    idx = np.clip(dtb - 1, 0, h - 1)
    breach_gap = np.where(dtb > 0, emi_due[np.arange(b), idx] - avail_q[np.arange(b), idx, li], 0.0)

    # PD from the predictive distribution at each EMI date
    p_short = quantile_cdf(avail_q, emi_due)  # [B, H]
    p_short[:, 0] = np.where(due[:, 0], (snap.current_balance < emi_due[:, 0]).astype(float), 0.0)
    pd_cash = np.where(due, p_short, 0.0).max(-1)
    p_util = (1.0 - quantile_cdf(q_util, np.full((b, h), cfg.utilization_breach))).max(-1)
    pd_bureau = 1.0 / (1.0 + np.exp((snap.cibil - cfg.bureau_midpoint) / cfg.bureau_scale))
    pd = 1.0 - (1.0 - pd_cash) * (1.0 - cfg.util_pd_weight * p_util) * (1.0 - cfg.bureau_pd_weight * pd_bureau)
    pd = np.clip(pd, 0.0, 1.0)

    ead_idx = np.where(dtb > 0, dtb - 1, h - 1)
    proj_util = np.clip(q_util[np.arange(b), ead_idx, ai], 0.0, None)
    drawn_now = np.clip(snap.current_utilization, 0.0, None) * snap.credit_limit
    ead = snap.outstanding_principal + np.maximum(drawn_now, proj_util * snap.credit_limit)
    lgd = np.clip(snap.lgd, 0.0, 1.0)
    in_sma = tier != "GREEN"
    el = np.where(in_sma, ead * pd * lgd, np.nan)

    dtz = _first_true(q_bal[..., li] <= cfg.zero_balance_threshold)
    next_emi_day = _first_true(due)
    next_emi_amount = np.where(next_emi_day > 0, emi_due[np.arange(b), np.clip(next_emi_day - 1, 0, h - 1)], 0.0)
    util_breach_day = _first_true(q_util[..., ai] >= cfg.utilization_breach)

    urgency_days = np.where(dtz > 0, np.minimum(dtz, np.where(dtb > 0, dtb, dtz)), dtb).astype(float)
    urgency = np.power(0.5, np.clip(urgency_days, 0, None) / cfg.urgency_half_life_days)
    priority = np.where(in_sma, np.nan_to_num(el) * urgency, 0.0)

    triggers: list[list[str]] = []
    qname = f"q{int(round(QUANTILES[li] * 100))}"
    for i in range(b):
        t: list[str] = []
        if dtb[i] > 0:
            t.append(f"EMI_SHORTFALL_{qname}_DAY_{dtb[i]}")
        if dtz[i] > 0:
            t.append(f"CASH_TO_ZERO_DAY_{dtz[i]}")
        if util_breach_day[i] > 0:
            t.append(f"UTILIZATION_SPIKE_DAY_{util_breach_day[i]}")
        if snap.missed_salary[i] > 0:
            t.append("MISSED_SALARY")
        if snap.payment_return[i] > 0:
            t.append("RECENT_PAYMENT_RETURN")
        if snap.cibil_change_90d[i] <= -30:
            t.append(f"BUREAU_DROP_{int(round(-snap.cibil_change_90d[i]))}PTS")
        triggers.append(t)

    return RiskResult(
        tier=tier,
        days_to_breach=dtb,
        breach_shortfall=breach_gap,
        days_to_zero=dtz,
        next_emi_day=next_emi_day,
        next_emi_amount=next_emi_amount,
        pd=pd,
        pd_cash=pd_cash,
        p_util=p_util,
        pd_bureau=pd_bureau,
        ead=ead,
        lgd=lgd,
        expected_loss=el,
        priority=priority,
        util_breach_day=util_breach_day,
        triggers=triggers,
    )
