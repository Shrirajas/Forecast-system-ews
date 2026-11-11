"""Model comparison on identical forecast origins, accounts and windows.

For each origin every model sees the same ``[B, 512, ...]`` window that ends AT the origin and is scored
on the same ``horizon`` days after it, which exist in the data (``run_comparison`` refuses an origin whose
actuals are missing, so nothing is scored on values unavailable at the origin). The compared target is
the balance channel (``eod_balance``) only: it is the one target every engine and baseline forecasts for
every account. Metrics are kept as sums so aggregating over origins is exact and checkable.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config.settings import QUANTILES
from data_engine.vector_cache import LoadedCache
from models.engine import ForecastEngine
from models.forecast import check_ordered
from models.pipeline import bound_quantiles

TAUS = np.asarray(QUANTILES)
BUCKETS = ((1, 1), (2, 7), (8, 30), (31, 60), (61, 90))
BASELINES = ("persistence", "seasonal_naive_30d", "empirical_quantiles_180d")
MIN_ACTUAL_SHARE = 0.8  # an account needs this share of the horizon observed to be scored


# ---------------------------------------------------------------------------- metrics
def sums(y: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Sufficient statistics. y ``[N, H]`` actual (NaN ignored), q ``[N, H, 9]`` quantile forecasts."""
    m = ~np.isnan(y)
    yy = np.nan_to_num(y)
    err = yy[..., None] - q
    pin = np.maximum(TAUS * err, (TAUS - 1) * err) * m[..., None]
    med = np.abs(yy - q[..., 4]) * m
    return {
        "n": float(m.sum()), "abs_y": float(np.abs(yy * m).sum()), "abs_err": float(med.sum()),
        "sq_err": float((((yy - q[..., 4]) * m) ** 2).sum()), "pin": float(pin.sum()),
        "cover": float((((yy >= q[..., 0]) & (yy <= q[..., -1])) & m).sum()),
    }


def add_sums(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: a.get(k, 0.0) + b.get(k, 0.0) for k in set(a) | set(b)}


def metrics_from_sums(s: dict[str, float]) -> dict[str, float]:
    n, denom = max(s["n"], 1.0), max(s["abs_y"], 1e-12)
    return {
        "mae": round(s["abs_err"] / n, 2), "rmse": round(float(np.sqrt(s["sq_err"] / n)), 2),
        "wape": round(s["abs_err"] / denom, 4), "wql": round(2 * s["pin"] / len(TAUS) / denom, 4),
        "coverage_10_90": round(s["cover"] / n, 4), "n_points": int(s["n"]),
    }


def forecast_metrics(y: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Metrics of one forecast set; also keeps the legacy ``wape_median`` key of the Berka report."""
    m = metrics_from_sums(sums(y, q))
    return {**m, "wape_median": m["wape"]}


def bucket_sums(y: np.ndarray, q: np.ndarray) -> dict[str, dict[str, float]]:
    h = y.shape[1]
    return {f"{a}-{b}" if a != b else f"{a}": sums(y[:, a - 1 : min(b, h)], q[:, a - 1 : min(b, h)])
            for a, b in BUCKETS if a <= h}


# ---------------------------------------------------------------------------- baselines
def baseline_forecasts(past: np.ndarray, horizon: int = 90) -> dict[str, np.ndarray]:
    """past ``[N, L]`` balances up to the origin (last column). Returns ``[N, horizon, 9]`` each."""
    n, L = past.shape
    last = pd.DataFrame(past.T).ffill().to_numpy().T[:, -1]
    persistence = np.repeat(last[:, None, None], horizon, 1).repeat(len(TAUS), 2)
    h = np.arange(1, horizon + 1)
    idx = L - 1 + h - 30 * np.ceil(h / 30).astype(int)
    seasonal = np.repeat(past[:, idx][..., None], len(TAUS), 2)
    seasonal = np.where(np.isnan(seasonal), persistence, seasonal)
    clim = np.nanquantile(past[:, -180:], TAUS, axis=1).T  # [N, 9]
    clim = np.where(np.isnan(clim), last[:, None], clim)
    return {
        "persistence": persistence,
        "seasonal_naive_30d": seasonal,
        "empirical_quantiles_180d": np.repeat(clim[:, None, :], horizon, 1),
    }


# ---------------------------------------------------------------------------- runner
def eligible_rows(cache: LoadedCache, rows: np.ndarray, origin: date, horizon: int) -> np.ndarray:
    """Rows observed at the origin, open 90+ days before it, with most of the horizon observed."""
    t = cache.day_index(origin)
    if t + horizon >= cache.meta.n_days:
        raise ValueError(f"origin {origin}: actuals through {origin + timedelta(days=horizon)} are not in the data "
                         f"(ends {cache.end})")
    bal = cache.past
    last, old = np.asarray(bal[rows, t, 0]), np.asarray(bal[rows, t - 90, 0])
    fut = np.asarray(bal[rows, t + 1 : t + 1 + horizon, 0])
    ok = ~np.isnan(last) & ~np.isnan(old) & ((~np.isnan(fut)).mean(1) >= MIN_ACTUAL_SHARE)
    return rows[ok]


def pick_single_account(cache: LoadedCache, eligible: list[np.ndarray], origins: list[date]) -> int:
    """Deterministic, model-blind: of the rows eligible at every origin, the one whose balance level is
    closest to the median level (a typical account, not a cherry-picked one)."""
    common = eligible[0]
    for e in eligible[1:]:
        common = np.intersect1d(common, e)
    if common.size == 0:
        raise ValueError("no account is eligible at every origin")
    t = cache.day_index(origins[-1])
    level = np.nanmean(np.asarray(cache.past[common, t - 89 : t + 1, 0]), axis=1)
    return int(common[np.argmin(np.abs(level - np.nanmedian(level)))])


def run_comparison(
    cache: LoadedCache,
    rows: np.ndarray,
    origins: list[date],
    horizon: int,
    engines: dict[str, ForecastEngine],
    single_account_row: int | None = None,
    history_days: int = 120,
) -> dict:
    """Score every engine and baseline on identical accounts/windows at each origin."""
    names = [*engines, *BASELINES]
    by_origin: dict[str, dict] = {}
    agg: dict[str, dict] = {n: {"all": {}, "buckets": {}} for n in names}
    eligible = [eligible_rows(cache, rows, o, horizon) for o in origins]
    if single_account_row is None:
        single_account_row = pick_single_account(cache, eligible, origins)
    single: dict = {"account_id": str(cache.accounts["account_id"].iloc[single_account_row]), "origins": {}}
    line = cache.accounts["credit_limit"].fillna(0).to_numpy(float) > 0
    for origin, er in zip(origins, eligible, strict=True):
        t = cache.day_index(origin)
        y = np.asarray(cache.past[er, t + 1 : t + 1 + horizon, 0], dtype=float)
        forecasts, elapsed = {}, {}
        first_past = None
        for name, eng in engines.items():
            win = cache.window(er, origin, eng.context_length, eng.horizon_pad)
            first_past = win["past"] if first_past is None else first_past
            t0 = time.perf_counter()
            q = eng.forecast(win["past"], win["known"], horizon, has_credit_line=line[er])
            elapsed[name] = round((time.perf_counter() - t0) * 1000, 1)
            q = bound_quantiles(q, win["past"])
            check_ordered(q)
            forecasts[name] = q[:, 0]
        forecasts.update(baseline_forecasts(first_past[:, :, 0], horizon))
        by_origin[origin.isoformat()] = {"n_accounts": int(len(er)), "elapsed_ms": elapsed, "models": {}}
        for name, q in forecasts.items():
            s_all, s_b = sums(y, q), bucket_sums(y, q)
            by_origin[origin.isoformat()]["models"][name] = {
                "metrics": metrics_from_sums(s_all), "sums": s_all, "by_horizon": {
                    k: metrics_from_sums(v) for k, v in s_b.items()}, "bucket_sums": s_b}
            agg[name]["all"] = add_sums(agg[name]["all"], s_all)
            for k, v in s_b.items():
                agg[name]["buckets"][k] = add_sums(agg[name]["buckets"].get(k, {}), v)
        pos = np.flatnonzero(er == single_account_row)
        if pos.size:
            i = int(pos[0])
            h = np.asarray(cache.past[single_account_row, t - history_days + 1 : t + 1, 0], dtype=float)
            single["origins"][origin.isoformat()] = {
                "history_dates": [(origin - timedelta(days=history_days - 1 - d)).isoformat() for d in range(history_days)],
                "history": [None if np.isnan(v) else round(float(v), 2) for v in h],
                "dates": [(origin + timedelta(days=d + 1)).isoformat() for d in range(horizon)],
                "actual": [None if np.isnan(v) else round(float(v), 2) for v in y[i]],
                "models": {n: {"q10": np.round(q[i, :, 0], 2).tolist(), "q50": np.round(q[i, :, 4], 2).tolist(),
                               "q90": np.round(q[i, :, -1], 2).tolist(), "metrics": forecast_metrics(y[i : i + 1], q[i : i + 1])}
                           for n, q in forecasts.items()},
            }
        else:
            single["origins"][origin.isoformat()] = {"skipped": "account not eligible at this origin"}
    aggregate = {n: {"metrics": metrics_from_sums(a["all"]), "sums": a["all"],
                     "by_horizon": {k: metrics_from_sums(v) for k, v in a["buckets"].items()}}
                 for n, a in agg.items()}
    return {"by_origin": by_origin, "aggregate": aggregate, "single_account": single,
            "eligible_accounts": {o.isoformat(): cache.accounts["account_id"].iloc[e].tolist()
                                  for o, e in zip(origins, eligible, strict=True)}}
