"""Multivariate spatial-temporal patch transformer with quantile outputs.

A TimesFM-style forecaster adapted to banking ledgers:

* **Patching**: each variate's daily series is cut into 32-day patches and
  embedded with a residual MLP over (values, observed-mask), as in TimesFM.
* **Token grid**: ``variates x time-patches``. Six past variates (EOD
  balance, utilisation, CIBIL, outward-payment volatility, missed-salary and
  payment-return flags) cover the context; their future positions are learned
  forecast-query tokens. Three known-in-advance variates (EMI schedule, bank
  holidays, payroll days) carry real values over both context and horizon.
* **Temporal attention**: each variate attends across its own time patches
  (long-run behaviour, seasonality, payment cadence).
* **Spatial attention**: at every time patch the variates attend to each
  other (for example a rise in card utilisation alongside a falling current
  account balance, or an EMI due date against the payroll day).
* **Probabilistic head**: target tokens at future positions decode to
  9 non-crossing quantiles (q10..q90) per day for balance and utilisation.

Reversible instance normalisation keeps the model scale-free across a
₹10k savings account and a ₹10cr SME current account; per-series scale
statistics are fed back in as a static conditioning vector.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from config.settings import QUANTILES

N_PAST, N_KNOWN, N_TARGETS = 6, 3, 2
BAL, UTIL, CIBIL, OUT_STD, MISSED_SAL, PAY_RET = range(N_PAST)
EMI, HOLIDAY, SALARY_DAY = range(N_KNOWN)
N_STATIC = 6


@dataclass
class STFormerConfig:
    context_length: int = 512
    max_horizon: int = 90
    patch_length: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1
    quantiles: list[float] = field(default_factory=lambda: list(QUANTILES))
    #: Normalise balance/utilisation around the LAST observed value instead of the context mean, so the
    #: forecast starts at today's level. Off by default: checkpoints trained without it must keep working.
    anchor_last: bool = False
    #: With anchor_last: scale the forecast offsets (from today's level) by sqrt(day / horizon_pad), so the
    #: first days start at today's value. The network can still learn large offsets for later days.
    horizon_ramp: bool = False

    @property
    def horizon_pad(self) -> int:
        return math.ceil(self.max_horizon / self.patch_length) * self.patch_length

    @property
    def n_past_patches(self) -> int:
        return self.context_length // self.patch_length

    @property
    def n_future_patches(self) -> int:
        return self.horizon_pad // self.patch_length

    def to_dict(self) -> dict:
        return asdict(self)


class ResidualPatchEmbed(nn.Module):
    """Per-variate residual MLP: (values ‖ mask) of one patch -> d_model."""

    def __init__(self, n_vars: int, patch: int, d_model: int):
        super().__init__()
        d_in = 2 * patch
        self.w1 = nn.Parameter(torch.randn(n_vars, d_in, d_model) / math.sqrt(d_in))
        self.b1 = nn.Parameter(torch.zeros(n_vars, d_model))
        self.w2 = nn.Parameter(torch.randn(n_vars, d_model, d_model) / math.sqrt(d_model))
        self.b2 = nn.Parameter(torch.zeros(n_vars, d_model))
        self.ws = nn.Parameter(torch.randn(n_vars, d_in, d_model) / math.sqrt(d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, V, N, 2P]
        h = F.silu(torch.einsum("bvnp,vpd->bvnd", x, self.w1) + self.b1[:, None])
        out = torch.einsum("bvnd,vde->bvne", h, self.w2) + self.b2[:, None]
        return out + torch.einsum("bvnp,vpd->bvnd", x, self.ws)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_valid: torch.Tensor | None) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        mask = key_valid[:, None, None, :] if key_valid is not None else None
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class STBlock(nn.Module):
    """Temporal attention -> spatial (cross-variate) attention -> FFN, pre-norm."""

    def __init__(self, cfg: STFormerConfig):
        super().__init__()
        d = cfg.d_model
        self.norm_t = nn.LayerNorm(d)
        self.attn_t = Attention(d, cfg.n_heads, cfg.dropout)
        self.norm_s = nn.LayerNorm(d)
        self.attn_s = Attention(d, cfg.n_heads, cfg.dropout)
        self.norm_f = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ff_mult * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.ff_mult * d, d)
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, v, n, d = x.shape
        # temporal: sequences over time patches, one per (sample, variate)
        xt = x.reshape(b * v, n, d)
        xt = xt + self.drop(self.attn_t(self.norm_t(xt), valid.reshape(b * v, n)))
        x = xt.view(b, v, n, d)
        # spatial: sequences over variates, one per (sample, time patch)
        xs = x.transpose(1, 2).reshape(b * n, v, d)
        vs = valid.transpose(1, 2).reshape(b * n, v)
        xs = xs + self.drop(self.attn_s(self.norm_s(xs), vs))
        x = xs.view(b, n, v, d).transpose(1, 2)
        return x + self.drop(self.ffn(self.norm_f(x)))


class SpatioTemporalForecaster(nn.Module):
    """Operates on normalised inputs; see :class:`EWSForecaster` for raw I/O."""

    def __init__(self, cfg: STFormerConfig):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.d_model, cfg.patch_length
        self.n_vars = N_PAST + N_KNOWN
        self.n_pos = cfg.n_past_patches + cfg.n_future_patches
        self.embed = ResidualPatchEmbed(self.n_vars, p, d)
        self.var_emb = nn.Parameter(torch.randn(self.n_vars, d) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(self.n_pos, d) * 0.02)
        self.query_emb = nn.Parameter(torch.randn(N_PAST, d) * 0.02)
        self.static = nn.Sequential(nn.Linear(N_STATIC, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(STBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(d)
        self.n_q = len(cfg.quantiles)
        self.median_idx = cfg.quantiles.index(0.5)
        self.head = nn.Linear(d, p * self.n_q)

    def _non_crossing(self, raw: torch.Tensor) -> torch.Tensor:
        m = self.median_idx
        mid = raw[..., m : m + 1]
        up = mid + torch.cumsum(F.softplus(raw[..., m + 1 :]), dim=-1)
        down = mid - torch.cumsum(F.softplus(raw[..., :m].flip(-1)), dim=-1)
        return torch.cat([down.flip(-1), mid, up], dim=-1)

    def forward(
        self, past: torch.Tensor, past_mask: torch.Tensor, known: torch.Tensor, static: torch.Tensor
    ) -> torch.Tensor:
        """past/past_mask [B, L, 6], known [B, L+Hp, 3], static [B, 6] -> [B, 2, Hp, Q]."""
        cfg = self.cfg
        b, L, _ = past.shape
        p, np_, nf = cfg.patch_length, cfg.n_past_patches, cfg.n_future_patches
        hp = cfg.horizon_pad

        pv = past.transpose(1, 2).reshape(b, N_PAST, np_, p)
        pm = past_mask.transpose(1, 2).reshape(b, N_PAST, np_, p)
        fut_zeros = pv.new_zeros(b, N_PAST, nf, p)
        past_tok = torch.cat([torch.cat([pv, fut_zeros], 2), torch.cat([pm, fut_zeros], 2)], dim=-1)
        kv = known.transpose(1, 2).reshape(b, N_KNOWN, np_ + nf, p)
        known_tok = torch.cat([kv, torch.ones_like(kv)], dim=-1)
        tokens = torch.cat([past_tok, known_tok], dim=1)  # [B, V, N, 2P]

        x = self.embed(tokens) + self.var_emb[None, :, None] + self.pos_emb[None, None]
        query = torch.zeros_like(x)
        query[:, :N_PAST, np_:] = self.query_emb[None, :, None]
        x = x + query + self.static(static)[:, None, None]

        valid = torch.ones(b, self.n_vars, np_ + nf, dtype=torch.bool, device=past.device)
        valid[:, :N_PAST, :np_] = pm.amax(-1) > 0  # drop fully-missing patches as keys
        for blk in self.blocks:
            x = blk(x, valid)

        h = self.norm(x[:, :N_TARGETS, np_:])  # [B, 2, Nf, D]
        raw = self.head(h).view(b, N_TARGETS, nf, p, self.n_q).reshape(b, N_TARGETS, hp, self.n_q)
        return self._non_crossing(raw.float())


@dataclass
class Scaling:
    bal_loc: torch.Tensor
    bal_scale: torch.Tensor
    util_loc: torch.Tensor
    util_scale: torch.Tensor


class EWSForecaster(nn.Module):
    """Raw-unit wrapper: handles NaNs, reversible normalisation and the loss."""

    def __init__(self, cfg: STFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.net = SpatioTemporalForecaster(cfg)
        self.register_buffer("taus", torch.tensor(cfg.quantiles, dtype=torch.float32), persistent=False)

    @staticmethod
    def _masked_stats(x: torch.Tensor, m: torch.Tensor, floor_abs: float, floor_rel: float):
        cnt = m.sum(1).clamp_min(1.0)
        mean = (x * m).sum(1) / cnt
        var = (((x - mean[:, None]) * m) ** 2).sum(1) / cnt
        mean_abs = (x.abs() * m).sum(1) / cnt
        std = torch.maximum(torch.sqrt(var).clamp_min(floor_abs), floor_rel * mean_abs)
        return mean, std

    @staticmethod
    def _last_obs(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Last observed value along time (0 where nothing was observed)."""
        idx = (m * torch.arange(1, x.shape[1] + 1, device=x.device)).argmax(1)
        return x.gather(1, idx[:, None])[:, 0] * (m.sum(1) > 0)

    def _prepare(self, past: torch.Tensor, known: torch.Tensor):
        with torch.autocast(device_type=past.device.type, enabled=False):
            past = past.float()
            known = torch.nan_to_num(known.float(), nan=0.0)
            mask = (~torch.isnan(past)).float()
            x = torch.nan_to_num(past, nan=0.0)
            bal_loc, bal_scale = self._masked_stats(x[..., BAL], mask[..., BAL], 1.0, 0.01)
            util_loc, util_scale = self._masked_stats(x[..., UTIL], mask[..., UTIL], 0.05, 0.0)
            if self.cfg.anchor_last:  # scale still comes from the whole context; only the level moves
                bal_loc = self._last_obs(x[..., BAL], mask[..., BAL])
                util_loc = self._last_obs(x[..., UTIL], mask[..., UTIL])
            s = Scaling(bal_loc, bal_scale, util_loc, util_scale)
            bs, bl = bal_scale[:, None], bal_loc[:, None]
            xn = torch.stack(
                [
                    (x[..., BAL] - bl) / bs,
                    (x[..., UTIL] - util_loc[:, None]) / util_scale[:, None],
                    (x[..., CIBIL] - 700.0) / 100.0,
                    x[..., OUT_STD] / bs,
                    x[..., MISSED_SAL],
                    x[..., PAY_RET],
                ],
                dim=-1,
            )
            xn = (xn * mask).clamp(-50, 50)
            kn = torch.stack([known[..., EMI] / bs, known[..., HOLIDAY], known[..., SALARY_DAY]], dim=-1).clamp(
                -50, 50
            )
            static = torch.stack(
                [
                    (-bal_loc / bal_scale).clamp(-50, 50),  # where "zero balance" sits in normalised units
                    torch.log1p(bal_loc.clamp_min(0)) / 15.0,
                    torch.log1p(bal_scale) / 15.0,
                    util_loc,
                    util_scale,
                    mask[..., BAL].mean(1),  # history completeness
                ],
                dim=-1,
            )
        return xn, mask, kn, static, s

    def forward_normalized(self, past: torch.Tensor, known: torch.Tensor):
        xn, mask, kn, static, s = self._prepare(past, known)
        qn = self.net(xn, mask, kn, static)
        if self.cfg.anchor_last and self.cfg.horizon_ramp:
            hp = qn.shape[2]
            ramp = torch.sqrt(torch.arange(1, hp + 1, device=qn.device, dtype=qn.dtype) / hp)
            qn = qn * ramp[None, None, :, None]  # positive scale: quantile order is preserved
        return qn, s

    def forward(self, past: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        """Raw ``past [B,L,6]`` (NaN = missing), ``known [B,L+Hp,3]`` -> raw quantiles ``[B,2,Hp,Q]``."""
        qn, s = self.forward_normalized(past, known)
        bal = qn[:, BAL] * s.bal_scale[:, None, None] + s.bal_loc[:, None, None]
        util = qn[:, UTIL] * s.util_scale[:, None, None] + s.util_loc[:, None, None]
        return torch.stack([bal, util], dim=1)

    def loss(self, past: torch.Tensor, known: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Pinball loss in normalised space. ``future [B, Hp, 2]`` raw, NaN = no target."""
        qn, s = self.forward_normalized(past, known)
        fut = future.float()
        tgt_mask = (~torch.isnan(fut)).float()
        fut = torch.nan_to_num(fut, nan=0.0)
        yb = (fut[..., BAL] - s.bal_loc[:, None]) / s.bal_scale[:, None]
        yu = (fut[..., UTIL] - s.util_loc[:, None]) / s.util_scale[:, None]
        y = torch.stack([yb, yu], dim=1).clamp(-50, 50)  # [B, 2, Hp]
        m = tgt_mask.transpose(1, 2)  # [B, 2, Hp]
        err = y[..., None] - qn  # [B, 2, Hp, Q]
        taus = self.taus.to(err.device)
        pin = torch.maximum(taus * err, (taus - 1) * err)
        return (pin.mean(-1) * m).sum() / m.sum().clamp_min(1.0)
