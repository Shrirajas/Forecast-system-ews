"""Device-agnostic execution helpers (CUDA mixed precision vs CPU).

On a CUDA box (e.g. an RTX 4080 Super with 16 GB VRAM) this module:

* picks ``cuda`` automatically and enables TF32 matmuls / cuDNN autotuning,
* caps the per-process memory fraction so the driver/display keep headroom,
* selects bf16 autocast when the GPU supports it (Ada Lovelace does), falling
  back to fp16 + ``GradScaler``,
* exposes :func:`run_with_oom_backoff`, which halves the micro-batch and retries
  when ``torch.cuda.OutOfMemoryError`` is raised instead of crashing.

On CPU it degrades to fp32 with multi-threaded intra-op parallelism.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar

# Must be set before CUDA initialises; reduces fragmentation-driven OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from utils.logging import logger  # noqa: E402

T = TypeVar("T")


@dataclass(frozen=True)
class DeviceContext:
    device: torch.device
    amp_dtype: torch.dtype | None  # None => full fp32
    pin_memory: bool
    use_grad_scaler: bool

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def autocast(self) -> contextlib.AbstractContextManager:
        """Mixed-precision context (``torch.amp.autocast``; the non-deprecated
        form of ``torch.cuda.amp.autocast``). No-op on CPU / fp32."""
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def grad_scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(self.device.type, enabled=self.use_grad_scaler)

    def to_device(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # non_blocking only overlaps H2D copies when the source is pinned.
        return tuple(t.to(self.device, non_blocking=self.pin_memory) for t in tensors)

    def describe(self) -> dict:
        info: dict = {
            "device": str(self.device),
            "amp_dtype": str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else "fp32",
            "pin_memory": self.pin_memory,
            "torch_version": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
        }
        if self.is_cuda:
            props = torch.cuda.get_device_properties(self.device)
            info.update(
                gpu_name=props.name,
                vram_total_gb=round(props.total_memory / 1024**3, 2),
                vram_allocated_gb=round(torch.cuda.memory_allocated(self.device) / 1024**3, 3),
            )
        return info


_CTX: DeviceContext | None = None


def get_device_context(
    amp_dtype: str = "auto", cuda_memory_fraction: float = 0.92, force_cpu: bool = False
) -> DeviceContext:
    """Return (and memoise) the process-wide device context."""
    global _CTX
    if _CTX is not None and not force_cpu:
        return _CTX

    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda", torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        with contextlib.suppress(RuntimeError):
            torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, device)

        if amp_dtype == "fp32":
            dtype = None
        elif amp_dtype == "fp16" or (amp_dtype == "auto" and not torch.cuda.is_bf16_supported()):
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        ctx = DeviceContext(
            device=device,
            amp_dtype=dtype,
            pin_memory=True,
            use_grad_scaler=dtype is torch.float16,
        )
    else:
        device = torch.device("cpu")
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        # bf16 autocast on CPU is slower than fp32 on most server CPUs; stay fp32.
        ctx = DeviceContext(device=device, amp_dtype=None, pin_memory=False, use_grad_scaler=False)

    logger.info("Execution device: {}", ctx.describe())
    if not force_cpu:
        _CTX = ctx
    return ctx


def is_cuda_oom(exc: BaseException) -> bool:
    """CUDA out-of-memory, including cuBLAS/cuDNN handle allocation failures, which surface as a plain
    RuntimeError (not torch.cuda.OutOfMemoryError) when a shared GPU is nearly full."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and ("ALLOC_FAILED" in str(exc) or "out of memory" in str(exc).lower())
    )


def run_with_oom_backoff(
    fn: Callable[[int, int], T], n_items: int, batch_size: int, min_batch: int = 1
) -> Iterator[T]:
    """Call ``fn(start, stop)`` over ``range(n_items)`` in chunks.

    If a chunk raises CUDA OOM the cache is released, the chunk size is halved
    and the same chunk is retried, so large portfolio scoring never aborts on a
    16 GB card; it just slows down.
    """
    start = 0
    bs = max(min_batch, batch_size)
    while start < n_items:
        stop = min(n_items, start + bs)
        try:
            out = fn(start, stop)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not is_cuda_oom(exc) or bs <= min_batch:
                raise
            torch.cuda.empty_cache()
            bs = max(min_batch, bs // 2)
            logger.warning("CUDA OOM: retrying with batch size {}", bs)
            continue
        yield out
        start = stop
