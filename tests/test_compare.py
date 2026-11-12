"""Comparison core: hand-computed metrics, no scoring on unavailable actuals, identical windows for all models."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from models.compare import BASELINES, baseline_forecasts, forecast_metrics, run_comparison, sums
from tests.test_engine_backends import StubEngine


def _flat(median):  # all 9 quantiles at the median
    return np.repeat(np.asarray(median, float)[..., None], 9, -1)


def test_metrics_match_hand_computation():
    y = np.array([[10.0, 20.0]])
    m = forecast_metrics(y, _flat([[12.0, 15.0]]))
    assert m["mae"] == 3.5 and m["rmse"] == round(float(np.sqrt((4 + 25) / 2)), 2)
    assert m["wape"] == round(7 / 30, 4) and m["wape_median"] == m["wape"]
    assert m["wql"] == m["wape"]  # equal quantiles: mean pinball over taus = |e|/2, wQL = 2*sum/sum|y|
    assert m["coverage_10_90"] == 0.0 and m["n_points"] == 2
    assert forecast_metrics(y, _flat([[10.0, 20.0]]))["coverage_10_90"] == 1.0


def test_nan_actuals_are_ignored():
    y = np.array([[10.0, np.nan, 20.0]])
    m = forecast_metrics(y, _flat([[12.0, 999.0, 15.0]]))
    assert m["n_points"] == 2 and m["mae"] == 3.5


def test_baselines_on_a_known_series():
    past = np.arange(1.0, 301.0)[None]  # 1..300, last = 300
    b = baseline_forecasts(past, 45)
    assert set(b) == set(BASELINES)
    assert (b["persistence"] == 300).all()
    assert b["seasonal_naive_30d"][0, 0, 0] == 271 and b["seasonal_naive_30d"][0, 30, 0] == 271  # same weekday-of-month, one season back
    assert b["empirical_quantiles_180d"][0, 0, 4] == np.nanquantile(past[0, -180:], 0.5)


def _setup(env, monkeypatch):
    _, pipe = env
    cache = pipe.cache.load()
    rows = np.arange(cache.n_accounts)
    engines = {"a": StubEngine("a"), "b": StubEngine("b")}
    origins = [cache.end - timedelta(days=d) for d in (150, 120, 90)]
    return cache, rows, engines, origins


def test_origin_without_actuals_is_refused(env, monkeypatch):
    cache, rows, engines, _ = _setup(env, monkeypatch)
    with pytest.raises(ValueError, match="not in the data"):
        run_comparison(cache, rows, [cache.end - timedelta(days=30)], 90, engines)


def test_every_model_scores_the_same_accounts_and_aggregation_is_exact(env, monkeypatch):
    cache, rows, engines, origins = _setup(env, monkeypatch)
    out = run_comparison(cache, rows, origins, 90, engines)
    names = {"a", "b", *BASELINES}
    for o, blk in out["by_origin"].items():
        assert set(blk["models"]) == names
        n = {m["metrics"]["n_points"] for m in blk["models"].values()}
        assert len(n) == 1  # identical scored points for every model and baseline
        assert blk["n_accounts"] == len(out["eligible_accounts"][o])
        for m in blk["models"].values():  # horizon buckets partition the overall sums
            assert sum(b["n"] for b in m["bucket_sums"].values()) == m["sums"]["n"]
    for name in names:
        assert out["aggregate"][name]["sums"]["n"] == sum(b["models"][name]["sums"]["n"] for b in out["by_origin"].values())
        assert out["aggregate"][name]["sums"]["abs_err"] == pytest.approx(
            sum(b["models"][name]["sums"]["abs_err"] for b in out["by_origin"].values()))


def test_single_account_view_uses_only_history_up_to_the_origin(env, monkeypatch):
    cache, rows, engines, origins = _setup(env, monkeypatch)
    out = run_comparison(cache, rows, origins, 90, engines)
    sa = out["single_account"]
    row = cache.rows_for([sa["account_id"]])[0][0]
    for o, v in sa["origins"].items():
        t = cache.day_index(__import__("datetime").date.fromisoformat(o))
        assert v["history_dates"][-1] == o and v["dates"][0] > o
        assert v["history"][-1] == pytest.approx(float(cache.past[row, t, 0]), abs=0.01)
        assert v["models"]["persistence"]["q50"] == pytest.approx([v["history"][-1]] * 90, abs=0.01)
        assert v["actual"][0] == pytest.approx(float(cache.past[row, t + 1, 0]), abs=0.01)
