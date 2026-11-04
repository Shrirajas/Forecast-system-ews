"""Train the spatial-temporal forecaster on the current vector cache.

    python scripts/train.py                     # settings defaults (GPU profile)
    python scripts/train.py --epochs 20 --d-model 192 --layers 4 --batch-size 128 --accum 2
    python scripts/train.py --cpu-profile       # small, fast CPU configuration

On Windows keep the ``if __name__ == "__main__"`` guard: DataLoader workers
use the spawn start method there.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import _bootstrap_path  # noqa: F401

from config.settings import get_settings
from data_engine.storage import get_storage
from models.pipeline import EWSPipeline
from models.train import train_model
from utils.logging import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--accum", type=int, dest="grad_accum_steps")
    ap.add_argument("--lr", type=float, dest="learning_rate")
    ap.add_argument("--d-model", type=int, dest="d_model")
    ap.add_argument("--layers", type=int, dest="n_layers")
    ap.add_argument("--heads", type=int, dest="n_heads")
    ap.add_argument("--stride", type=int, dest="window_stride")
    ap.add_argument("--workers", type=int, dest="num_workers")
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "fp32"], dest="amp_dtype")
    ap.add_argument("--compile", action="store_true", dest="compile_model", default=None)
    ap.add_argument("--cpu-profile", action="store_true")
    args = vars(ap.parse_args())

    settings = get_settings()
    setup_logging(settings.log_level)
    m = settings.model
    if args.pop("cpu_profile"):
        m = m.model_copy(update=dict(d_model=96, n_layers=2, ff_mult=2, window_stride=5, num_workers=0))
    m = m.model_copy(update={k: v for k, v in args.items() if v is not None})

    pipeline = EWSPipeline(settings, storage=get_storage(settings))
    pipeline.refresh()
    print(json.dumps(asdict(train_model(pipeline.cache.current_dir(), m)), indent=2))


if __name__ == "__main__":
    main()
