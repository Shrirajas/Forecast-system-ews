"""Forecast engines behind one interface.

``stformer``  the in-repo spatial-temporal transformer trained on your own ledgers.
``timesfm3``  Google's pretrained TimesFM 3.0 (``google/timesfm-3.0-pytorch``),
              zero-shot, multivariate with past-only and past-and-future
              covariates. Needs Hugging Face access on first load. Weights are
              non-commercial: a research baseline, not a production engine.

Both return raw-unit quantiles ``[B, 2, horizon, 9]`` (balance, utilisation).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
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

    #: Identifies the exact weights, so stored results are never served after a retrain.
    model_version: str = "unknown"

    @abstractmethod
    def forecast(
        self, past: np.ndarray, known: np.ndarray, horizon: int, has_credit_line: np.ndarray | None = None
    ) -> np.ndarray:
        """past ``[B, L, 6]`` (NaN = missing), known ``[B, L + horizon_pad, 3]``,
        has_credit_line ``[B]`` bool from the account master (credit_limit > 0)."""

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model_version": self.model_version,
            "context_length": self.context_length,
            "horizon_pad": self.horizon_pad,
        }


class STFormerEngine(ForecastEngine):
    name = "stformer"

    def __init__(self, checkpoint: Path, ctx: DeviceContext, batch_size: int = 512, compile_model: bool = False):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"model checkpoint not found at {checkpoint}; run `python scripts/bootstrap.py` "
                "or `python scripts/train.py` first"
            )
        self.model_version = "stformer:" + hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:16]
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = STFormerConfig(**blob["config"])
        self.model = EWSForecaster(self.cfg)
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(ctx.device).eval()
        if compile_model and ctx.is_cuda:
            # Batch sizes vary per request, so compile with dynamic shapes to avoid recompiles.
            self.model.net.forward = torch.compile(self.model.net.forward, dynamic=True)
        self.ctx = ctx
        self.batch_size = batch_size
        self.context_length = self.cfg.context_length
        self.horizon_pad = self.cfg.horizon_pad
        self.meta = {k: blob.get(k) for k in ("metrics", "epoch", "trained_at")}
        self._lock = threading.Lock()  # one forward at a time per model instance

    def forecast(
        self, past: np.ndarray, known: np.ndarray, horizon: int, has_credit_line: np.ndarray | None = None
    ) -> np.ndarray:
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
    """Pretrained TimesFM 3.0, zero-shot.

    * Balance and utilisation are multivariate targets, the four behavioural
      features are past-only covariates, and the EMI / holiday / payroll
      calendars are past-and-future covariates.
    * Accounts without a credit line (``credit_limit == 0`` in the account
      master, e.g. CASA only) are forecast with balance as the only target;
      their utilisation track is 0. An undrawn line still gets a forecast.
    * Each context is trimmed to a whole number of patches and only equal
      lengths share a batch, so the library never pads and results do not
      depend on batch composition.
    * Work is chunked with OOM backoff, runs under autocast on CUDA, and never
      uses symmetric averaging (which would double the forward passes).

    Licence: the TimesFM 3.0 weights are for non-commercial, non-production
    use. Use this engine as a research baseline only.
    """

    name = "timesfm3"

    def __init__(
        self,
        checkpoint: str,
        ctx: DeviceContext,
        context_length: int,
        horizon_pad: int,
        forecaster=None,
        batch_size: int = 64,
        chunk_size: int = 512,
        autocast: bool = True,
    ):
        if forecaster is None:
            try:
                from timesfm3 import TimesFM3Forecaster
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("timesfm3 backend needs `pip install 'timesfm[torch]>=3.0'`") from exc
            forecaster = TimesFM3Forecaster.from_pretrained(
                checkpoint, device=str(ctx.device), per_core_batch_size=batch_size
            )
        else:
            forecaster.config = dataclasses.replace(forecaster.config, per_core_batch_size=batch_size)
        self.model = forecaster
        self.ctx = ctx
        self.context_length = context_length
        self.horizon_pad = horizon_pad
        self.chunk_size = chunk_size
        self.use_autocast = autocast
        self.model_quantiles = np.asarray(self.model.config.quantiles, dtype=np.float64)
        self.model_version = f"timesfm3:{checkpoint}:{getattr(self.model.config, 'revision', None) or 'main'}"
        self._lock = threading.Lock()

    def _to_ours(self, q: np.ndarray) -> np.ndarray:
        if q.shape[-1] == len(self.model_quantiles) + 1:  # some checkpoints prepend the mean
            q = q[..., 1:]
        if len(self.model_quantiles) != len(QUANTILES) or not np.allclose(self.model_quantiles, QUANTILES):
            q = np.apply_along_axis(lambda v: np.interp(QUANTILES, self.model_quantiles, v), -1, q)
        return np.sort(q, axis=-1).astype(np.float32)

    def _predict(self, past, known, start, rows: np.ndarray, n_targets: int, horizon: int) -> np.ndarray:
        L = past.shape[1]

        def run(a: int, b: int) -> np.ndarray:
            sel = rows[a:b]
            ctx_ = [past[i, start[i] :, :n_targets].T.astype(np.float32) for i in sel]
            po = [past[i, start[i] :, 2:].T.astype(np.float32) for i in sel]
            pf = [known[i, start[i] : L + horizon].T.astype(np.float32) for i in sel]
            amp = self.ctx.autocast() if self.use_autocast else contextlib.nullcontext()
            with torch.inference_mode(), amp:
                outs = list(
                    self.model.predict_batch(
                        contexts=ctx_,
                        horizon=horizon,
                        past_only_covariates=po,
                        past_future_covariates=pf,
                        return_quantiles=True,
                        use_symmetric_averaging=False,
                    )
                )
            return np.stack([o.quantiles for o in outs]).astype(np.float32)  # [n, n_targets, H, Qm]

        return np.concatenate(list(run_with_oom_backoff(run, len(rows), self.chunk_size)), axis=0)

    def forecast(
        self, past: np.ndarray, known: np.ndarray, horizon: int, has_credit_line: np.ndarray | None = None
    ) -> np.ndarray:
        b, L = past.shape[:2]
        out = np.zeros((b, 2, horizon, len(QUANTILES)), dtype=np.float32)
        if b == 0:
            return out
        observed = ~np.isnan(past[:, :, 0])
        length = np.where(observed.any(1), L - observed.argmax(1), 1)
        patch = int(getattr(self.model.config, "input_patch_length", 32))
        # The library pads every batch to its longest series and padding shifts the
        # forecast. Trim each context to a whole number of patches (dropping at most
        # patch-1 of the oldest days) and batch only equal lengths: no padding at
        # all, so an account's forecast is identical whatever shares its batch.
        length = np.where(length >= patch, (length // patch) * patch, length)
        start = L - length
        if has_credit_line is None:  # fall back to history when the master has no limit data
            has_credit_line = np.nan_to_num(np.abs(past[:, :, 1])).max(1) > 0
        has_credit_line = np.asarray(has_credit_line, dtype=bool)
        with self._lock:
            for line, n_targets in ((True, 2), (False, 1)):
                for n in np.unique(length[has_credit_line == line]):
                    rows = np.flatnonzero((has_credit_line == line) & (length == n))
                    q = self._to_ours(self._predict(past, known, start, rows, n_targets, horizon))
                    out[rows, :n_targets] = q
        return out


_engine: ForecastEngine | None = None
_engine_lock = threading.Lock()


def get_engine(settings: Settings, reload: bool = False) -> ForecastEngine:
    global _engine
    with _engine_lock:
        if _engine is None or reload:
            m = settings.model
            ctx = get_device_context(m.amp_dtype, m.cuda_memory_fraction)
            if m.backend == "timesfm3":
                if not m.allow_noncommercial_weights:
                    raise RuntimeError(
                        "TimesFM 3.0 weights are licensed for non-commercial, non-production use only "
                        "(that includes commercial demos and pilots). Use the stformer backend, or set "
                        "EWS_MODEL__ALLOW_NONCOMMERCIAL_WEIGHTS=true for genuinely non-commercial research."
                    )
                logger.warning("TimesFM 3.0 weights: non-commercial licence, research use only")
                hp = -(-m.max_horizon // m.patch_length) * m.patch_length
                _engine = TimesFM3Engine(
                    m.timesfm_checkpoint, ctx, m.context_length, hp,
                    batch_size=m.timesfm_batch_size, chunk_size=m.inference_batch_size,
                    autocast=m.timesfm_autocast,
                )
            else:
                _engine = STFormerEngine(m.checkpoint_path, ctx, m.inference_batch_size, m.compile_model)
            logger.info("forecast engine loaded: {}", _engine.name)
        return _engine


def reset_engine() -> None:
    global _engine
    with _engine_lock:
        _engine = None
