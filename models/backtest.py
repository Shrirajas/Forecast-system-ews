"""Out-of-time backtest of the EWS against realised outcomes.

Scores the book as of ``as_of`` (at least ``horizon`` days before the last
data day) and compares the result with what actually happened in the
following ``horizon`` days:

* **credit**: did the account have an EMI payment returned? Reported as
  precision / recall of the SMA flag, ROC-AUC and Brier score of the PD,
  and the timing error between the predicted breach day and the first
  actual return;
* **attrition**: did the realised average balance fall at least the
  configured threshold below the pre-period baseline, without stress?
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np

from models.pipeline import EWSPipeline


def backtest(pipeline: EWSPipeline, as_of: date | None = None, horizon: int = 90) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    cache = pipeline.loaded_cache(auto_refresh=False)
    as_of = as_of or cache.end - timedelta(days=horizon)
    if as_of + timedelta(days=horizon) > cache.end:
        raise ValueError("backtest window must end on or before the last cached day")
    rows = np.arange(cache.n_accounts)
    rs = pipeline.settings.risk
    sb = pipeline.score_rows(cache, rows, as_of, horizon, rs.liability_quantile, rs.asset_quantile)
    ids = cache.accounts["account_id"].to_numpy()

    ret = pipeline.storage.query_df(
        "SELECT account_id, MIN(txn_date) AS first_return FROM transactions "
        "WHERE txn_type = 'EMI' AND status = 'RETURNED' AND txn_date > CAST(:a AS DATE) "
        "AND txn_date <= CAST(:b AS DATE) GROUP BY account_id",
        {"a": as_of, "b": as_of + timedelta(days=horizon)},
    )
    first = dict(zip(ret.account_id, ret.first_return, strict=True))
    actual = np.array([a in first for a in ids])
    flagged = sb.risk.tier != "GREEN"
    tp = int((flagged & actual).sum())
    fp = int((flagged & ~actual).sum())
    fn = int((~flagged & actual).sum())
    tn = int((~flagged & ~actual).sum())

    timing = [
        abs(int(sb.risk.days_to_breach[i]) - (first[ids[i]] - as_of).days)
        for i in np.flatnonzero(flagged & actual)
    ]
    pd_ = np.clip(sb.risk.pd, 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(actual, pd_)) if 0 < actual.sum() < len(actual) else None

    t0 = cache.day_index(as_of) + 1
    realised = np.nanmean(np.asarray(cache.past[rows, t0 : t0 + horizon, 0]), axis=1)
    bh = sb.behavior
    stress = (sb.snapshot.missed_salary > 0) | (sb.snapshot.payment_return > 0) | actual
    realised_drop = (bh.baseline_avg - realised) / np.maximum(np.abs(bh.baseline_avg), 1.0)
    attr_actual = (
        ~stress
        & (bh.baseline_avg >= pipeline.settings.behavior.attrition_min_balance)
        & (realised_drop >= pipeline.settings.behavior.attrition_drop_threshold)
    )
    attr_pred = bh.attrition_alert
    a_tp = int((attr_pred & attr_actual).sum())

    return {
        "as_of_date": as_of.isoformat(),
        "horizon_days": horizon,
        "accounts": int(len(ids)),
        "credit_ews": {
            "accounts_with_emi_return": int(actual.sum()),
            "flagged_sma": int(flagged.sum()),
            "precision": round(tp / max(tp + fp, 1), 3),
            "recall": round(tp / max(tp + fn, 1), 3),
            "false_positive_rate": round(fp / max(fp + tn, 1), 3),
            "pd_roc_auc": round(auc, 3) if auc is not None else None,
            "pd_brier": round(float(brier_score_loss(actual, pd_)), 4),
            "breach_timing_mae_days": round(float(np.mean(timing)), 1) if timing else None,
            "tier_counts": {t: int((sb.risk.tier == t).sum()) for t in ("GREEN", "SMA-0", "SMA-1", "SMA-2")},
        },
        "deposit_attrition": {
            "realised_attrition": int(attr_actual.sum()),
            "flagged": int(attr_pred.sum()),
            "precision": round(a_tp / max(int(attr_pred.sum()), 1), 3),
            "recall": round(a_tp / max(int(attr_actual.sum()), 1), 3),
        },
    }
