"""Training loop for the spatial-temporal forecaster.

Device-agnostic: on CUDA it uses pinned-memory loaders, non-blocking copies,
bf16/fp16 autocast (``GradScaler`` for fp16) and gradient accumulation so the
effective batch can exceed what fits in 16 GB; on CPU it runs plain fp32.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config.settings import ModelSettings
from models.dataset import CacheWindowDataset, WindowSpec, build_window_index, make_loader
from models.stformer import EWSForecaster, STFormerConfig
from utils.device import DeviceContext, get_device_context
from utils.logging import logger

MAX_VAL_WINDOWS = 8_192


@dataclass
class TrainResult:
    checkpoint: str
    epochs: int
    best_epoch: int
    train_windows: int
    val_windows: int
    best_val_pinball: float
    val_coverage_10_90: float
    val_median_mape: float
    seconds: float
    device: str


def cpu_profile(m: ModelSettings) -> ModelSettings:
    """Smaller network and fewer windows so CPU-only training takes minutes."""
    return m.model_copy(update=dict(d_model=96, n_layers=2, ff_mult=2, n_heads=4, window_stride=5, num_workers=0))


def config_from_settings(m: ModelSettings) -> STFormerConfig:
    return STFormerConfig(
        context_length=m.context_length,
        max_horizon=m.max_horizon,
        patch_length=m.patch_length,
        d_model=m.d_model,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        ff_mult=m.ff_mult,
        dropout=m.dropout,
    )


@torch.no_grad()
def evaluate(model: EWSForecaster, loader, ctx: DeviceContext, horizon: int) -> dict[str, float]:
    model.eval()
    tot, n = 0.0, 0
    cover, mape, cnt = 0.0, 0.0, 0
    for past, known, fut in loader:
        past, known, fut = ctx.to_device(past, known, fut)
        with ctx.autocast():
            loss = model.loss(past, known, fut)
            q = model(past, known)
        tot += float(loss) * past.shape[0]
        n += past.shape[0]
        y = fut[:, :horizon, 0]
        ok = ~torch.isnan(y)
        lo, med, hi = q[:, 0, :horizon, 0], q[:, 0, :horizon, 4], q[:, 0, :horizon, -1]
        cover += float(((y >= lo) & (y <= hi) & ok).sum())
        denom = torch.nan_to_num(y.abs(), nan=1.0).clamp_min(1_000.0)
        mape += float((torch.nan_to_num((med - y).abs() / denom) * ok).sum())
        cnt += int(ok.sum())
    return {
        "pinball": tot / max(n, 1),
        "coverage_10_90": cover / max(cnt, 1),
        "median_mape": mape / max(cnt, 1),
    }


def train_model(
    cache_dir: Path,
    m: ModelSettings,
    checkpoint_path: Path | None = None,
    max_steps_per_epoch: int | None = None,
    quiet: bool = False,
    max_target_day: int | None = None,
    init_checkpoint: Path | None = None,
) -> TrainResult:
    """Train on the cache at ``cache_dir``.

    ``max_target_day`` keeps every supervised target on or before that day
    index (out-of-time evaluation); ``init_checkpoint`` warm-starts from a
    compatible checkpoint (transfer learning / fine-tuning)."""
    t0 = time.perf_counter()
    torch.manual_seed(m.seed)
    np.random.seed(m.seed)
    ctx = get_device_context(m.amp_dtype, m.cuda_memory_fraction)
    cfg = config_from_settings(m)
    spec = WindowSpec(cfg.context_length, cfg.max_horizon, cfg.horizon_pad)

    past_shape = np.load(cache_dir / "past.npy", mmap_mode="r").shape
    train_idx, val_idx = build_window_index(
        past_shape[0], past_shape[1], spec, m.window_stride, m.val_fraction, m.seed,
        max_target_day=max_target_day,
    )
    # Drop windows whose account is not yet open at the cutoff or has no target.
    past_mm = np.load(cache_dir / "past.npy", mmap_mode="r")

    def observed(idx: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return idx
        last_ctx = past_mm[idx[:, 0], idx[:, 1] - 1, 0]
        first_tgt = past_mm[idx[:, 0], np.minimum(idx[:, 1], past_shape[1] - 1), 0]
        return idx[~np.isnan(last_ctx) & ~np.isnan(first_tgt)]

    train_idx, val_idx = observed(train_idx), observed(val_idx)
    if len(val_idx) > MAX_VAL_WINDOWS:  # keep per-epoch validation cheap on large books
        keep = np.random.default_rng(m.seed).choice(len(val_idx), MAX_VAL_WINDOWS, replace=False)
        val_idx = val_idx[np.sort(keep)]
    train_ds = CacheWindowDataset(cache_dir, train_idx, spec)
    val_ds = CacheWindowDataset(cache_dir, val_idx, spec)
    train_dl = make_loader(train_ds, m.batch_size, True, ctx, m.num_workers)
    val_dl = make_loader(val_ds, m.batch_size * 2, False, ctx, m.num_workers) if len(val_ds) else None

    model = EWSForecaster(cfg)
    if init_checkpoint is not None:
        blob = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if blob["config"] != cfg.to_dict():
            raise ValueError("init_checkpoint architecture differs from the training configuration")
        model.load_state_dict(blob["state_dict"])
        log_ = logger.debug if quiet else logger.info
        log_("warm start from {}", init_checkpoint)
    model = model.to(ctx.device)
    if m.compile_model and ctx.is_cuda:
        # compile the inner network's forward only: state_dict keys stay unchanged
        model.net.forward = torch.compile(model.net.forward)
    opt = torch.optim.AdamW(model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay)
    steps_per_epoch = max(1, len(train_dl) // m.grad_accum_steps)
    if max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=m.learning_rate, total_steps=steps_per_epoch * m.epochs, pct_start=0.1
    )
    scaler = ctx.grad_scaler()
    log = logger.debug if quiet else logger.info
    log(
        "training {} params on {} | {} train / {} val windows | batch {} x accum {}",
        sum(p.numel() for p in model.parameters()), ctx.device, len(train_ds), len(val_ds),
        m.batch_size, m.grad_accum_steps,
    )

    ckpt = Path(checkpoint_path or m.checkpoint_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, best_metrics = float("inf"), -1, {}
    for epoch in range(m.epochs):
        model.train()
        running, seen, opt_steps = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for step, (past, known, fut) in enumerate(train_dl):
            past, known, fut = ctx.to_device(past, known, fut)
            with ctx.autocast():
                loss = model.loss(past, known, fut)
            scaler.scale(loss / m.grad_accum_steps).backward()
            if (step + 1) % m.grad_accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), m.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                opt_steps += 1
            running += float(loss.detach()) * past.shape[0]
            seen += past.shape[0]
            if opt_steps >= steps_per_epoch:
                break
        metrics = evaluate(model, val_dl, ctx, cfg.max_horizon) if val_dl else {"pinball": running / max(seen, 1)}
        log(
            "epoch {}/{} train_pinball={:.4f} val={}",
            epoch + 1, m.epochs, running / max(seen, 1), {k: round(v, 4) for k, v in metrics.items()},
        )
        if metrics["pinball"] < best:
            best, best_epoch, best_metrics = metrics["pinball"], epoch + 1, metrics
            torch.save(
                {
                    "format": "ews-stformer/1",
                    "config": cfg.to_dict(),
                    "state_dict": model.state_dict(),
                    "metrics": metrics,
                    "epoch": epoch + 1,
                    "trained_at": datetime.now().isoformat(timespec="seconds"),
                    "torch_version": torch.__version__,
                },
                ckpt,
            )
        if ctx.is_cuda:
            torch.cuda.empty_cache()

    result = TrainResult(
        checkpoint=str(ckpt),
        epochs=m.epochs,
        best_epoch=best_epoch,
        train_windows=len(train_ds),
        val_windows=len(val_ds),
        best_val_pinball=round(best, 5),
        val_coverage_10_90=round(best_metrics.get("coverage_10_90", float("nan")), 4),
        val_median_mape=round(best_metrics.get("median_mape", float("nan")), 4),
        seconds=round(time.perf_counter() - t0, 1),
        device=str(ctx.device),
    )
    ckpt.with_suffix(".json").write_text(json.dumps(asdict(result), indent=2))
    log("training done: {}", result)
    return result
