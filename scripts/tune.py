"""Optuna search over the forecaster's architecture and optimiser settings.

    python scripts/tune.py --trials 20 --steps-per-epoch 60 --epochs 3
"""

from __future__ import annotations

import argparse
import json

import _bootstrap_path  # noqa: F401

from config.settings import get_settings
from models.pipeline import EWSPipeline
from models.tuning import tune
from utils.logging import setup_logging

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=60)
    a = ap.parse_args()
    s = get_settings()
    setup_logging("WARNING")
    pipe = EWSPipeline(s)
    pipe.refresh()
    base = s.model.model_copy(update={"epochs": a.epochs, "num_workers": 0})
    study = tune(pipe.cache.current_dir(), base, a.trials, a.steps_per_epoch)
    print(json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2))
    print("apply with e.g. EWS_MODEL__D_MODEL=... EWS_MODEL__N_LAYERS=... python scripts/train.py")
