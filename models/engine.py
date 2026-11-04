"""Forecast engines behind one interface.

``stformer``  the in-repo spatial-temporal transformer trained on your own ledgers.
``timesfm3``  Google's pretrained TimesFM 3.0 (``google/timesfm-3.0-pytorch``),
              zero-shot, multivariate with past-only and past-and-future
              covariates. Needs Hugging Face access on first load.

Both return raw-unit quantiles ``[B, 2, horizon, 9]`` (balance, utilisation).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from config.settings import QUANTILES, Settings
from models.stformer import EWSForecaster, STFormerConfig
from utils.device import DeviceContext, get_device_context, run_with_oom_backoff
from utils.logging import logger


class ForecastEngine(ABC):
    name: str
    context_length: int
    horizon_pad: int

    @abstractmethod
    def forecast(self, past: np.ndarray, known: np.ndarray, horizon: int) -> np.ndarray:
        """past ``[B, L, 6]`` (NaN = missing), known ``[B, L + horizon_pad, 3]``."""

    def describe(self) -> dict:
        return {"backend": self.name, "context_length": self.context_length, "horizon_pad": self.horizon_pad}


class STFormerEngine(ForecastEngine):
    name = "stformer"

    def __init__(self, checkpoint: Path, ctx: DeviceContext, batch_size: int = 512):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"model checkpoint not found at {checkpoint}; run `python scripts/bootstrap.py` "
                "or `python scripts/train.py` first"
            )
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = STFormerConfig(**blob["config"])
        self.model = EWSForecaster(self.cfg)
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(ctx.device).eval()
        self.ctx = ctx
        self.batch_size = batch_size
        self.context_length = self.cfg.context_length
        self.horizon_pad = self.cfg.horizon_pad
        self.meta = {k: blob.get(k) for k in ("metrics", "epoch", "trained_at")}
        self._lock = threading.Lock()  # one forward at a time per model instance

    def forecast(self, past: np.ndarray, known: np.ndarray, horizon: int) -> np.ndarray:
        if horizon > self.cfg.max_horizon:
            raise ValueError(f"horizon {horizon} > model max_horizon {self.cfg.max_horizon}")
        past_t = torch.from_numpy(np.ascontiguousarray(past, dtype=np.float32))
        known_t = torch.from_numpy(np.ascontiguousarray(known, dtype=np.float32))
        if self.ctx.pin_memory:
            past_t, known_t = past_t.pin_memory(), known_t.pin_memory()

        def run(a: int, b: int) -> np.ndarray:
            p, k = self.ctx.to_device(past_t[a:b], known_t[a:b])
            with torch.inference_mode(), self.ctx.autocast():
                q = self.model(p, k)
            return q[:, :, :horizon].float().cpu().numpy()

        with self._lock:
            outs = list(run_with_oom_backoff(run, past_t.shape[0], self.batch_size))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, 2, horizon, len(QUANTILES)), np.float32)

    def describe(self) -> dict:
        return {**super().describe(), **self.meta, "device": self.ctx.describe()}


class TimesFM3Engine(ForecastEngine):
    name = "timesfm3"

    def __init__(
        self, checkpoint: str, ctx: DeviceContext, context_length: int, horizon_pad: int, forecaster=None
    ):
        if forecaster is None:
            try:
                from timesfm3 import TimesFM3Forecaster
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("timesfm3 backend needs `pip install 'timesfm[torch]>=3.0'`") from exc
            forecaster = TimesFM3Forecaster.from_pretrained(checkpoint, device=str(ctx.device))
        self.model = forecaster
        self.ctx = ctx
        self.context_length = context_length
        self.horizon_pad = horizon_pad
        self.model_quantiles = np.asarray(self.model.config.quantiles, dtype=np.float64)
        self._lock = threading.Lock()

    def forecast(self, past: np.ndarray, known: np.ndarray, horizon: int) -> np.ndarray:
        L = past.shape[1]
        contexts = [past[i, :, :2].T.astype(np.float32) for i in range(len(past))]
        past_only = [past[i, :, 2:].T.astype(np.float32) for i in range(len(past))]
        past_future = [known[i, : L + horizon].T.astype(np.float32) for i in range(len(past))]
        with self._lock, torch.inference_mode():
            outs = list(
                self.model.predict_batch(
                    contexts=contexts,
                    horizon=horizon,
                    past_only_covariates=past_only,
                    past_future_covariates=past_future,
                    return_quantiles=True,
                )
            )
        q = np.stack([o.quantiles for o in outs]).astype(np.float32)  # [B, 2, H, Qm]
        if q.shape[-1] == len(self.model_quantiles) + 1:  # some checkpoints prepend the mean
            q = q[..., 1:]
        if len(self.model_quantiles) != len(QUANTILES) or not np.allclose(self.model_quantiles, QUANTILES):
            q = np.apply_along_axis(lambda v: np.interp(QUANTILES, self.model_quantiles, v), -1, q)
        return np.sort(q, axis=-1).astype(np.float32)


_engine: ForecastEngine | None = None
_engine_lock = threading.Lock()


def get_engine(settings: Settings, reload: bool = False) -> ForecastEngine:
    global _engine
    with _engine_lock:
        if _engine is None or reload:
            m = settings.model
            ctx = get_device_context(m.amp_dtype, m.cuda_memory_fraction)
            if m.backend == "timesfm3":
                hp = -(-m.max_horizon // m.patch_length) * m.patch_length
                _engine = TimesFM3Engine(m.timesfm_checkpoint, ctx, m.context_length, hp)
            else:
                _engine = STFormerEngine(m.checkpoint_path, ctx, m.inference_batch_size)
            logger.info("forecast engine loaded: {}", _engine.name)
        return _engine


def reset_engine() -> None:
    global _engine
    with _engine_lock:
        _engine = None
