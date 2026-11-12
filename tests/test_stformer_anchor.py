"""anchor_last: forecasts are normalised around today's level; the default path is unchanged."""

from __future__ import annotations

import numpy as np
import torch

from models.stformer import EWSForecaster, STFormerConfig

CFG = dict(d_model=32, n_heads=2, n_layers=1, ff_mult=2)


def _past():
    rng = np.random.default_rng(0)
    p = rng.random((3, 512, 6)).astype(np.float32) * 100 + 50
    p[1, 401:] = np.nan  # last observed value is at index 400, not at the end
    p[2, :300] = np.nan  # short history
    return torch.from_numpy(p), torch.zeros(3, 512 + 96, 3)


def test_anchored_level_is_the_last_observed_value_and_legacy_is_the_mean():
    past, known = _past()
    legacy = EWSForecaster(STFormerConfig(**CFG))
    anch = EWSForecaster(STFormerConfig(**CFG, anchor_last=True))
    s_old = legacy._prepare(past, known)[4]
    s_new = anch._prepare(past, known)[4]
    expect_last = torch.stack([past[0, -1, 0], past[1, 400, 0], past[2, -1, 0]])
    torch.testing.assert_close(s_new.bal_loc, expect_last)
    torch.testing.assert_close(s_old.bal_loc, torch.nan_to_num(past[..., 0]).sum(1) / (~torch.isnan(past[..., 0])).sum(1))
    torch.testing.assert_close(s_new.bal_scale, s_old.bal_scale)  # only the level moves


def test_all_missing_series_anchor_to_zero_without_nan():
    past, known = _past()
    past[0, :, 0] = float("nan")
    m = EWSForecaster(STFormerConfig(**CFG, anchor_last=True)).eval()
    q = m(past, known)
    assert torch.isfinite(q).all() and m._prepare(past, known)[4].bal_loc[0] == 0


def test_old_checkpoint_configs_still_load_with_the_flag_off():
    cfg = STFormerConfig(**{k: v for k, v in STFormerConfig(**CFG).to_dict().items() if k != "anchor_last"})
    assert cfg.anchor_last is False


def test_horizon_ramp_scales_offsets_by_sqrt_day_and_keeps_quantile_order():
    past, known = _past()
    plain = EWSForecaster(STFormerConfig(**CFG, anchor_last=True)).eval()
    ramp = EWSForecaster(STFormerConfig(**CFG, anchor_last=True, horizon_ramp=True)).eval()
    ramp.load_state_dict(plain.state_dict())
    q0, _ = plain.forward_normalized(past, known)
    q1, _ = ramp.forward_normalized(past, known)
    w = torch.sqrt(torch.arange(1, q0.shape[2] + 1) / q0.shape[2])
    torch.testing.assert_close(q1, q0 * w[None, None, :, None])
    assert w[0] < 0.11 and (q1.diff(dim=-1) >= 0).all()
