from __future__ import annotations

import numpy as np
import pytest
import torch

from config.settings import QUANTILES, BehaviorSettings, RiskSettings
from models.behavior import evaluate_behavior
from models.risk import AccountSnapshot, evaluate_risk, quantile_cdf, tier_for_days
from models.stformer import EWSForecaster, STFormerConfig
from utils.device import get_device_context, run_with_oom_backoff


def test_device_context_cpu_fallback():
    ctx = get_device_context(force_cpu=True)
    assert ctx.device.type == "cpu" and ctx.amp_dtype is None and not ctx.pin_memory
    with ctx.autocast():
        assert torch.ones(2).sum().item() == 2


def test_oom_backoff_halves_batch():
    calls = []

    def fn(a, b):
        calls.append(b - a)
        if b - a > 2:
            raise torch.cuda.OutOfMemoryError("fake")
        return list(range(a, b))

    out = [x for chunk in run_with_oom_backoff(fn, 7, 8) for x in chunk]
    assert out == list(range(7)) and calls[:3] == [7, 4, 2]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return EWSForecaster(STFormerConfig(d_model=32, n_heads=2, n_layers=1, ff_mult=2)).eval()


def test_forward_shapes_and_non_crossing(model):
    past = torch.rand(4, 512, 6) * 1e5
    past[:, :200] = float("nan")  # short history -> padded
    known = torch.zeros(4, 608, 3)
    q = model(past, known)
    assert q.shape == (4, 2, 96, 9)
    assert torch.isfinite(q).all()
    assert (q.diff(dim=-1) >= -1e-4).all()


def test_normalisation_is_scale_free(model):
    """Reversible normalisation: scaling every money channel by 10x leaves the
    normalised tokens unchanged (only the static log-level features differ)."""
    torch.manual_seed(1)
    past = torch.rand(2, 512, 6)
    past[..., 0] = past[..., 0] * 5e4 + 1e4
    past[..., 3] *= 2e3
    known = torch.zeros(2, 608, 3)
    known[:, ::30, 0] = 8_000.0
    x1, _, k1, s1, sc1 = model._prepare(past, known)
    p2, k2 = past.clone(), known.clone()
    for ch in (0, 3):
        p2[..., ch] *= 10
    k2[..., 0] *= 10
    x2, _, kk2, s2, sc2 = model._prepare(p2, k2)
    torch.testing.assert_close(x1, x2, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k1, kk2, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(sc2.bal_scale, sc1.bal_scale * 10, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(s1[:, 0], s2[:, 0], rtol=1e-4, atol=1e-4)  # zero-balance position


def test_loss_backward(model):
    model.train()
    past = torch.rand(3, 512, 6) * 1e4
    known = torch.zeros(3, 608, 3)
    fut = torch.rand(3, 96, 2) * 1e4
    fut[:, 90:] = float("nan")
    loss = model.loss(past, known, fut)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    model.eval()


def test_quantile_cdf_hits_levels():
    q = np.array([[10.0, 20, 30, 40, 50, 60, 70, 80, 90]])
    for tau, v in zip(QUANTILES, q[0], strict=True):
        assert np.isclose(quantile_cdf(q, np.array([v]))[0], tau)
    xs = np.linspace(-100, 300, 50)
    f = quantile_cdf(np.repeat(q, 50, 0), xs)
    assert (np.diff(f) >= -1e-12).all() and f[0] < 0.01 and f[-1] > 0.99


def test_tier_buckets():
    days = np.array([-1, 1, 30, 31, 60, 61, 90])
    assert tier_for_days(days, 90).tolist() == ["GREEN", "SMA-0", "SMA-0", "SMA-1", "SMA-1", "SMA-2", "SMA-2"]
    assert tier_for_days(days, 30).tolist() == ["GREEN", "SMA-0", "SMA-0", "GREEN", "GREEN", "GREEN", "GREEN"]


def _snapshot(b: int, balance: float) -> AccountSnapshot:
    return AccountSnapshot(
        current_balance=np.full(b, balance),
        current_utilization=np.full(b, 0.2),
        credit_limit=np.full(b, 100_000.0),
        outstanding_principal=np.full(b, 500_000.0),
        lgd=np.full(b, 0.4),
        cibil=np.full(b, 760.0),
        missed_salary=np.zeros(b),
        payment_return=np.zeros(b),
        cibil_change_90d=np.zeros(b),
    )


def test_risk_engine_breach_day_tier_and_expected_loss():
    h = 90
    spread = np.linspace(-0.4, 0.4, 9)
    q_bal = np.empty((3, h, 9))
    base = np.full((3, h), 50_000.0)
    base[0, 40:] = 5_000.0  # account 0 runs low from day 41 (EMI on day 45 -> SMA-1)
    base[1, :] = 200_000.0  # comfortably funded -> GREEN
    base[2, 3:] = 1_000.0  # account 2 drains immediately (EMI on day 10 -> SMA-0)
    q_bal[:] = base[..., None] * (1 + spread)
    q_util = np.full((3, h, 9), 0.3)
    emi = np.zeros((3, h))
    emi[:, [9, 44, 74]] = 20_000.0
    r = evaluate_risk(q_bal, q_util, emi, _snapshot(3, 60_000.0), RiskSettings())
    assert r.days_to_breach.tolist() == [45, -1, 10]
    assert r.tier.tolist() == ["SMA-1", "GREEN", "SMA-0"]
    # section 11: EL is computed for every credit product, so a GREEN account with a limit has one too
    assert np.isclose(r.expected_loss[1], r.ead[1] * r.pd[1] * r.lgd[1])
    for i in (0, 2):
        assert np.isclose(r.expected_loss[i], r.ead[i] * r.pd[i] * r.lgd[i])
    assert r.pd[2] > r.pd[1]
    assert r.priority[2] > 0 and r.priority[1] > 0 and r.priority[2] > r.priority[1]
    # EAD uses the larger of current drawn and the q90 projected drawn amount
    assert np.isclose(r.ead[1], 500_000 + 0.3 * 100_000)


def test_attrition_uses_pre_period_baseline_and_skips_stressed_accounts():
    past = np.full((3, 512), 300_000.0)
    past[2, -60:] = 120_000.0  # exit already under way: trailing 90d is depressed, baseline is not
    q_bal = np.empty((3, 90, 9))
    q_bal[0] = 100_000.0  # -67% vs baseline -> attrition
    q_bal[1] = 290_000.0  # stable
    q_bal[2] = 110_000.0
    res = evaluate_behavior(past, q_bal, np.array([False, False, False]), BehaviorSettings())
    assert res.attrition_alert.tolist() == [True, False, True]
    assert np.isclose(res.cof_impact_annual[0], 200_000 * 0.035)
    stressed = evaluate_behavior(past, q_bal, np.array([True, False, True]), BehaviorSettings())
    assert stressed.attrition_alert.tolist() == [False, False, False]


@pytest.fixture(scope="module")
def tiny_timesfm(tmp_path_factory):
    """A tiny randomly initialised TimesFM 3.0 model: exercises the real library API, no download."""
    tf = pytest.importorskip("timesfm3.torch.timesfm3_forecaster")
    from timesfm3.torch import configs

    tmp = tmp_path_factory.mktemp("tfm")
    rb = configs.ResidualBlockConfig(hidden_dims=32, output_dims=32, use_bias=False, activation="relu")
    tc = configs.StackedTransformersConfig(
        num_layers=1,
        transformer=configs.TransformerConfig(
            model_dims=32, hidden_dims=32, num_heads=2, attention_norm="rms", feedforward_norm="rms",
            qk_norm="rms", use_rope_seq=True, use_rope_var=True, use_bias=False, ff_activation="relu",
            deterministic=True,
        ),
    )
    cfg = tf._ModelConfig(checkpoint_path=str(tmp / "tiny.pt"), residual_block_config=rb,
                          transformer_config=tc, device="cpu")
    torch.manual_seed(0)
    torch.save(tf._make_torch_model(cfg).state_dict(), tmp / "tiny.pt")
    return tf.TimesFM3Forecaster(cfg)


def _book(seed: int = 0):
    rng = np.random.default_rng(seed)
    past = rng.random((5, 512, 6)).astype(np.float32)
    scale = np.array([1e2, 1e4, 1e6, 1e3, 1e5], dtype=np.float32)
    past[..., 0] = past[..., 0] * scale[:, None] + scale[:, None]
    past[[1, 3], :, 1] = 0.0  # CASA-only accounts: utilisation always zero
    past[0, :400] = np.nan  # short histories
    past[3, :200] = np.nan
    return past, np.zeros((5, 608, 3), np.float32)


def test_timesfm3_adapter_with_tiny_random_model(tiny_timesfm):
    from models.engine import TimesFM3Engine

    calls = []
    real = tiny_timesfm.predict_batch

    def spy(**kw):
        calls.append(kw)
        return real(**kw)

    eng = TimesFM3Engine("unused", get_device_context(force_cpu=True), 512, 96, forecaster=tiny_timesfm, batch_size=16)
    eng.model.predict_batch = spy
    past, known = _book()
    q = eng.forecast(past, known, 60)
    assert q.shape == (5, 2, 60, 9) and np.isfinite(q).all()
    assert (np.diff(q, axis=-1) >= 0).all()
    assert eng.model.config.per_core_batch_size == 16
    assert all(c["use_symmetric_averaging"] is False for c in calls)
    n_targets = sorted(x.shape[0] for c in calls for x in c["contexts"])
    assert n_targets == [1, 1, 2, 2, 2]  # the two CASA-only accounts forecast balance alone
    assert np.all(q[[1, 3], 1] == 0.0)  # ...and get a zero utilisation track
    lengths = sorted(c["contexts"][i].shape[1] for c in calls for i in range(len(c["contexts"])))
    assert lengths == [96, 288, 512, 512, 512]  # leading NaNs stripped, trimmed to whole patches
    for c in calls:  # one exact length per library call: nothing to pad
        assert len({x.shape[1] for x in c["contexts"]}) == 1
    # results come back in the caller's order despite grouping and sorting
    med = q[:, 0, :, 4].mean(1)
    assert list(np.argsort(med)) == [0, 3, 1, 4, 2]


def test_timesfm3_chunking_matches_single_pass(tiny_timesfm):
    from models.engine import TimesFM3Engine

    ctx = get_device_context(force_cpu=True)
    past, known = _book(1)
    whole = TimesFM3Engine("u", ctx, 512, 96, forecaster=tiny_timesfm, chunk_size=512).forecast(past, known, 30)
    chunked = TimesFM3Engine("u", ctx, 512, 96, forecaster=tiny_timesfm, chunk_size=2).forecast(past, known, 30)
    np.testing.assert_allclose(chunked, whole, rtol=1e-3, atol=1e-3)


def test_timesfm3_forecast_does_not_depend_on_batch_neighbours(tiny_timesfm):
    from models.engine import TimesFM3Engine

    eng = TimesFM3Engine("u", get_device_context(force_cpu=True), 512, 96, forecaster=tiny_timesfm)
    past, known = _book(2)
    together = eng.forecast(past, known, 30)
    for i in range(len(past)):
        alone = eng.forecast(past[i : i + 1], known[i : i + 1], 30)
        np.testing.assert_allclose(alone[0], together[i], rtol=1e-4, atol=1e-3)


def test_timesfm3_same_bucket_lengths_are_exact(tiny_timesfm):
    """Histories of 300 and 318 days fall in the same patch bucket; with trimming the
    library never pads, so each account's forecast is bit-identical alone or mixed."""
    from models.engine import TimesFM3Engine

    eng = TimesFM3Engine("u", get_device_context(force_cpu=True), 512, 96, forecaster=tiny_timesfm)
    past, known = _book(3)
    past[0, : 512 - 300] = np.nan
    past[2, : 512 - 318] = np.nan
    mixed = eng.forecast(past, known, 30)
    for i in range(len(past)):
        np.testing.assert_array_equal(eng.forecast(past[i : i + 1], known[i : i + 1], 30)[0], mixed[i])


def test_timesfm3_credit_line_flag_overrides_history(tiny_timesfm):
    from models.engine import TimesFM3Engine

    eng = TimesFM3Engine("u", get_device_context(force_cpu=True), 512, 96, forecaster=tiny_timesfm)
    calls = []
    real = eng.model.predict_batch
    eng.model.predict_batch = lambda **kw: (calls.append(kw), real(**kw))[1]
    past, known = _book(4)
    past[:, :, 1] = 0.0  # nobody has ever drawn on a line ...
    lines = np.array([True, False, True, False, False])  # ... but accounts 0 and 2 have one
    q = eng.forecast(past, known, 30, has_credit_line=lines)
    assert np.all(q[~lines, 1] == 0.0)
    # undrawn lines are still forecast as a second target, not assumed to be zero
    assert sorted(x.shape[0] for c in calls for x in c["contexts"]) == [1, 1, 1, 2, 2]


def test_timesfm3_requires_licence_ack_in_any_environment():
    from config.settings import ModelSettings, Settings
    from models.engine import get_engine, reset_engine

    reset_engine()
    with pytest.raises(RuntimeError, match="non-commercial"):
        get_engine(Settings(environment="sandbox", model=ModelSettings(backend="timesfm3")))
    reset_engine()
