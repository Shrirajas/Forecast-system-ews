"""The common, limited forecast response shared by every backend."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

import models.engine as eng_mod
from models.engine import TIMESFM_BADGE, _key
from models.forecast import QNAMES, UnknownAccount, UnsupportedTarget, forecast_accounts
from tests.test_engine_backends import StubEngine  # noqa: E402


def _ids(pipe):
    c = pipe.cache.load()
    line = (c.accounts.credit_limit.fillna(0) > 0).to_numpy()
    return c, c.accounts.account_id[line].tolist(), c.accounts.account_id[~line].tolist()


def test_response_carries_provenance_and_ordered_quantiles(env):
    _, pipe = env
    c, with_line, _ = _ids(pipe)
    origin = c.end - timedelta(days=60)
    out = forecast_accounts(pipe, with_line[:2], horizon=30, as_of=origin, targets=("eod_balance", "utilization"),
                            with_actual=True, history_days=45)
    assert len(out.responses) == 4 and not out.skipped
    for r in out.responses:
        assert r.backend == "stformer" and r.model_version.startswith("stformer:") and r.licence_badge is None
        assert r.device and r.origin == origin and r.horizon == 30 and len(r.dates) == 30
        assert r.dates[0] == origin + timedelta(days=1) and r.history_dates[-1] == origin and len(r.history) == 45
        assert tuple(r.quantiles) == QNAMES and all(len(v) == 30 for v in r.quantiles.values())
        q = np.array(list(r.quantiles.values()))
        assert (np.diff(q, axis=0) >= -1e-6).all()
        assert r.median == r.quantiles["q50"] and r.to_dict()["median"] == r.median
        assert r.unit is None  # units come from the dataset registry, never guessed


def test_actuals_are_the_days_after_the_origin_only(env):
    _, pipe = env
    c, ids, _ = _ids(pipe)
    origin = c.end - timedelta(days=40)
    r = forecast_accounts(pipe, ids[:1], horizon=30, as_of=origin, with_actual=True).responses[0]
    t = c.day_index(origin)
    row = c.rows_for(ids[:1])[0][0]
    np.testing.assert_allclose(r.actual, np.round(np.asarray(c.past[row, t + 1 : t + 31, 0], float), 4), atol=1e-3)
    np.testing.assert_allclose(r.history[-1], float(c.past[row, t, 0]), atol=1e-3)  # history ends AT the origin
    assert forecast_accounts(pipe, ids[:1], horizon=30, as_of=origin).responses[0].actual is None


def test_utilization_is_skipped_for_accounts_without_a_credit_line(env, monkeypatch):
    _, pipe = env
    c, with_line, _ = _ids(pipe)
    no_line = with_line[0]  # make one account line-less on a copy of the metadata (restored by monkeypatch)
    acc = c.accounts.copy()
    acc.loc[acc.account_id == no_line, "credit_limit"] = 0.0
    monkeypatch.setattr(c, "accounts", acc)
    out = forecast_accounts(pipe, [no_line, with_line[1]], horizon=30, targets=("eod_balance", "utilization"))
    assert [(r.account_id, r.target) for r in out.responses] == [
        (no_line, "eod_balance"), (with_line[1], "eod_balance"), (with_line[1], "utilization")]
    assert out.skipped == [{"account_id": no_line, "target": "utilization", "reason": "account has no credit line"}]


def test_rejects_unsupported_target_unknown_account_and_bad_horizon(env):
    _, pipe = env
    c, ids, _ = _ids(pipe)
    with pytest.raises(UnsupportedTarget):
        forecast_accounts(pipe, ids[:1], targets=("probability_of_default",))
    with pytest.raises(UnknownAccount):
        forecast_accounts(pipe, ["NOPE_1"])
    with pytest.raises(ValueError, match="horizon"):
        forecast_accounts(pipe, ids[:1], horizon=500)


def test_timesfm_response_names_backend_and_shows_licence_badge(env, monkeypatch):
    settings, pipe = env
    _, ids, _ = _ids(pipe)
    monkeypatch.setitem(eng_mod._engines, _key(settings, "timesfm3"), StubEngine("timesfm3"))
    r = forecast_accounts(pipe, ids[:1], backend="timesfm3", horizon=30).responses[0]
    assert r.backend == "timesfm3" and r.model_version == "timesfm3:stub" and r.licence_badge == TIMESFM_BADGE


def test_crossing_quantiles_from_an_engine_raise_instead_of_being_sorted(env, monkeypatch):
    settings, pipe = env
    _, ids, _ = _ids(pipe)

    class Crossing(StubEngine):
        def forecast(self, *a, **k):
            return super().forecast(*a, **k)[..., ::-1].copy()

    monkeypatch.setitem(eng_mod._engines, _key(settings, "timesfm3"), Crossing("timesfm3"))
    with pytest.raises(ValueError, match="not ordered"):
        forecast_accounts(pipe, ids[:1], backend="timesfm3", horizon=30)


def test_na_credit_fields_reach_alerts_as_null_not_nan():
    from models.pipeline import _nan_to_none

    assert _nan_to_none(np.array([1.5, np.nan, np.inf])) == [1.5, None, None]
