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
import importlib.util
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from config.settings import QUANTILES, Settings
from models.stformer import EWSForecaster, STFormerConfig
from utils.device import DeviceContext, get_device_context, is_cuda_oom, run_with_oom_backoff
from utils.logging import logger


BACKENDS = ("stformer", "timesfm3")

#: Shown wherever TimesFM output appears. The weights forbid commercial and production use.
TIMESFM_BADGE = "TimesFM 3.0 · research licence · prototype only"


class EngineUnavailable(RuntimeError):
    """The backend that was asked for cannot be used. Callers report this; they never
    answer with a different backend (no silent fallback from TimesFM to STFormer)."""


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
        self.meta = {k: blob.get(k) for k in ("metrics", "epoch", "trained_at", "meta")}
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


def _hf_snapshot(repo: str) -> str | None:
    """Commit hash of the cached Hugging Face snapshot, so a stored result names the exact weights."""
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(repo, "config.json")
        return Path(hit).parent.name if isinstance(hit, str) else None
    except Exception:  # local directory checkpoint or no hub: version falls back to "unversioned"
        return None


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
        self.model_version = f"timesfm3:{checkpoint}@{_hf_snapshot(checkpoint) or 'unversioned'}"
        self.device_note: str | None = None
        self._lock = threading.Lock()

    def describe(self) -> dict:
        return {**super().describe(), "zero_shot": True, "trained_here": False, "licence_badge": TIMESFM_BADGE,
                "device": self.ctx.describe(), "device_note": self.device_note}

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


_engines: dict[tuple[str, str], ForecastEngine] = {}
_registry_lock = threading.Lock()
_load_locks: dict[tuple[str, str], threading.Lock] = {}


def _key(settings: Settings, backend: str) -> tuple[str, str]:
    m = settings.model
    return (backend, str(m.timesfm_checkpoint) if backend == "timesfm3" else str(m.checkpoint_path))


def _timesfm_problems(settings: Settings) -> list[str]:
    """Why TimesFM cannot be used right now (empty = usable). Never loads weights."""
    out = []
    if not settings.model.allow_noncommercial_weights:
        out.append(
            "TimesFM 3.0 weights are licensed for non-commercial, non-production use only (that includes "
            "commercial demos and pilots); set EWS_MODEL__ALLOW_NONCOMMERCIAL_WEIGHTS=true for genuinely "
            "non-commercial research"
        )
    if importlib.util.find_spec("timesfm3") is None:
        out.append("the timesfm3 package is not installed (pip install 'timesfm[torch]>=3.0')")
    return out


def _build_timesfm(settings: Settings) -> TimesFM3Engine:
    m = settings.model
    problems = _timesfm_problems(settings)
    if problems:
        raise EngineUnavailable("; ".join(problems))
    logger.warning("TimesFM 3.0 weights: non-commercial licence, research use only")
    hp = -(-m.max_horizon // m.patch_length) * m.patch_length

    def make(ctx: DeviceContext) -> TimesFM3Engine:
        return TimesFM3Engine(
            m.timesfm_checkpoint, ctx, m.context_length, hp, batch_size=m.timesfm_batch_size,
            chunk_size=m.inference_batch_size, autocast=m.timesfm_autocast,
        )

    ctx = get_device_context(m.amp_dtype, m.cuda_memory_fraction)
    try:
        return make(ctx)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not is_cuda_oom(exc):
            raise EngineUnavailable(f"TimesFM 3.0 could not be loaded ({type(exc).__name__}: {exc})") from exc
        # Same model on the CPU, reported in describe(); never a different model.
        torch.cuda.empty_cache()
        logger.warning("TimesFM did not fit on the GPU (shared card); loading it on the CPU instead")
        eng = make(get_device_context(force_cpu=True))
        eng.device_note = "cpu: CUDA out of memory while loading (GPU is shared)"
        return eng
    except Exception as exc:  # missing weights / no network / incompatible files
        raise EngineUnavailable(f"TimesFM 3.0 could not be loaded ({type(exc).__name__}: {exc})") from exc


def _build_stformer(settings: Settings) -> STFormerEngine:
    m = settings.model
    ctx = get_device_context(m.amp_dtype, m.cuda_memory_fraction)
    try:
        return STFormerEngine(m.checkpoint_path, ctx, m.inference_batch_size, m.compile_model)
    except FileNotFoundError as exc:
        raise EngineUnavailable(str(exc)) from exc


def get_engine(settings: Settings, backend: str | None = None, reload: bool = False) -> ForecastEngine:
    """The loaded engine for ``backend`` (default: the process default, ``stformer``).

    Each (backend, checkpoint) is loaded once and reused. A new engine is built *before* it replaces
    anything, so a failed load leaves every loaded engine untouched. Raises ``EngineUnavailable``;
    it never substitutes another backend."""
    backend = backend or settings.model.backend
    if backend not in BACKENDS:
        raise ValueError(f"unknown forecast backend {backend!r}; choose from {BACKENDS}")
    key = _key(settings, backend)
    with _registry_lock:
        lock = _load_locks.setdefault(key, threading.Lock())
    with lock:  # one load per key, and a slow TimesFM load never blocks STFormer requests
        eng = _engines.get(key)
        if eng is None or reload:
            eng = _build_timesfm(settings) if backend == "timesfm3" else _build_stformer(settings)
            _engines[key] = eng
            logger.info("forecast engine loaded: {}", eng.name)
        return eng


def engine_status(settings: Settings) -> dict[str, dict]:
    """Availability of every backend without loading any weights."""
    out: dict[str, dict] = {}
    for backend in BACKENDS:
        eng = _engines.get(_key(settings, backend))
        if backend == "timesfm3":
            problems = _timesfm_problems(settings)
            hit = None
            try:
                from huggingface_hub import try_to_load_from_cache

                hit = try_to_load_from_cache(settings.model.timesfm_checkpoint, "model.safetensors")
            except Exception:
                pass
            info = {"weights_cached": isinstance(hit, str), "licence_badge": TIMESFM_BADGE, "zero_shot": True}
        else:
            ck = Path(settings.model.checkpoint_path)
            problems = [] if ck.exists() else [f"model checkpoint not found at {ck}"]
            info = {"checkpoint": str(ck)}
        out[backend] = {
            "backend": backend, "available": not problems, "reasons": problems, "loaded": eng is not None,
            "model_version": eng.model_version if eng else None, **info,
        }
    return out


def reset_engine(backend: str | None = None) -> None:
    """Drop loaded engines (all, or one backend). Retrain code should reset only ``stformer`` so a
    1.3 GB TimesFM model is not reloaded for nothing."""
    with _registry_lock:
        for key in [k for k in _engines if backend is None or k[0] == backend]:
            del _engines[key]
