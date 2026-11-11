"""One forecast response for every backend, deliberately limited to what both can forecast.

Both engines output the two model channels, ``eod_balance`` and ``utilization``, as 9 quantiles per
day. Utilisation exists only for accounts with a credit line. Nothing else (SMA tiers, EMI, PD, EAD)
is part of this response: those are risk-engine outputs, not forecasts. A target that an account or
model cannot genuinely forecast is reported in ``skipped`` with the reason, never invented.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from config.settings import QUANTILES
from models.engine import TIMESFM_BADGE
from models.pipeline import EWSPipeline, bound_quantiles

TARGETS: dict[str, int] = {"eod_balance": 0, "utilization": 1}
QNAMES = tuple(f"q{int(round(q * 100))}" for q in QUANTILES)


class UnsupportedTarget(ValueError):
    pass


class UnknownAccount(ValueError):
    pass


@dataclass
class ForecastResponse:
    account_id: str
    target: str
    unit: str | None  # supplied by the dataset registry; never guessed here
    backend: str
    model_version: str
    device: str | None
    origin: date  # last day of data the model saw
    horizon: int
    dates: list[date]
    quantiles: dict[str, list[float]]  # q10..q90 -> one value per day
    history_dates: list[date]
    history: list[float | None]
    actual: list[float | None] | None  # backtests only: what really happened after the origin
    elapsed_ms: float  # wall time of the whole batch this account was forecast in
    licence_badge: str | None = None

    @property
    def median(self) -> list[float]:
        return self.quantiles["q50"]

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "median": self.median}


@dataclass
class ForecastBatch:
    responses: list[ForecastResponse] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)  # {"account_id", "target", "reason"}


def check_ordered(q: np.ndarray) -> None:
    """q10 <= q20 <= ... <= q90 on every day of every series, or raise (never silently re-sort)."""
    d = np.diff(q, axis=-1)
    tol = 1e-5 * np.maximum(1.0, np.abs(q[..., 1:]))
    bad = int((d < -tol).sum())
    if bad:
        raise ValueError(f"forecast quantiles are not ordered in {bad} places")


def forecast_accounts(
    pipe: EWSPipeline,
    account_ids: list[str],
    *,
    backend: str | None = None,
    horizon: int = 90,
    as_of: date | None = None,
    targets: tuple[str, ...] = ("eod_balance",),
    history_days: int = 120,
    with_actual: bool = False,
    units: dict[str, str] | None = None,
) -> ForecastBatch:
    """Forecast ``targets`` for ``account_ids`` as of ``as_of`` (default: last day of data) with the
    chosen backend. Raises ``EngineUnavailable`` if that backend cannot load: no other backend answers."""
    bad = [t for t in targets if t not in TARGETS]
    if bad:
        raise UnsupportedTarget(f"unsupported target(s) {bad}; choose from {sorted(TARGETS)}")
    eng = pipe.engine_for(backend)
    if horizon < 1 or horizon > eng.horizon_pad:
        raise ValueError(f"horizon must be 1..{eng.horizon_pad}")
    cache = pipe.loaded_cache(False)
    as_of = as_of or cache.end
    rows, unknown = cache.rows_for(account_ids)
    if unknown:
        raise UnknownAccount(f"unknown account id(s) in this dataset: {unknown[:5]}")
    acc = cache.accounts.iloc[rows]
    line = acc["credit_limit"].fillna(0).to_numpy(float) > 0
    win = cache.window(rows, as_of, eng.context_length, eng.horizon_pad)
    t0 = time.perf_counter()
    q = eng.forecast(win["past"], win["known"], horizon, has_credit_line=line)
    elapsed = (time.perf_counter() - t0) * 1000
    q = bound_quantiles(q, win["past"])
    check_ordered(q)

    desc = eng.describe()
    device = (desc.get("device") or {}).get("device") if isinstance(desc.get("device"), dict) else None
    badge = TIMESFM_BADGE if eng.name == "timesfm3" else None
    dates = [as_of + timedelta(days=d + 1) for d in range(horizon)]
    n_hist = min(history_days, win["past"].shape[1])
    hist_dates = [as_of - timedelta(days=n_hist - 1 - i) for i in range(n_hist)]
    t_idx = cache.day_index(as_of)
    out = ForecastBatch()
    for j, row in enumerate(rows):
        aid = str(acc["account_id"].iloc[j])
        for tgt in targets:
            ch = TARGETS[tgt]
            if tgt == "utilization" and not line[j]:
                out.skipped.append({"account_id": aid, "target": tgt, "reason": "account has no credit line"})
                continue
            hist = [None if np.isnan(v) else round(float(v), 4) for v in win["past"][j, -n_hist:, ch]]
            actual = None
            if with_actual:
                fut = np.asarray(cache.past[row, t_idx + 1 : t_idx + 1 + horizon, ch], dtype=float)
                fut = np.concatenate([fut, np.full(horizon - len(fut), np.nan)])
                actual = [None if np.isnan(v) else round(float(v), 4) for v in fut]
            out.responses.append(
                ForecastResponse(
                    account_id=aid, target=tgt, unit=(units or {}).get(tgt), backend=eng.name,
                    model_version=eng.model_version, device=device, origin=as_of, horizon=horizon, dates=dates,
                    quantiles={n: np.round(q[j, ch, :, k], 4).tolist() for k, n in enumerate(QNAMES)},
                    history_dates=hist_dates, history=hist, actual=actual, elapsed_ms=round(elapsed, 1),
                    licence_badge=badge,
                )
            )
    return out
