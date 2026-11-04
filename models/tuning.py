"""Optuna hyper-parameter search for the forecaster.

Each trial trains a short run and is scored by validation pinball loss on
held-out accounts. A median pruner stops weak trials early.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import optuna

from config.settings import ModelSettings
from models.train import train_model


def tune(cache_dir: Path, base: ModelSettings, n_trials: int = 20, steps_per_epoch: int | None = 60) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        d_model = trial.suggest_categorical("d_model", [64, 96, 128, 192])
        m = base.model_copy(
            update=dict(
                d_model=d_model,
                n_heads=trial.suggest_categorical("n_heads", [4, 8]),  # every d_model choice divides by 8
                n_layers=trial.suggest_int("n_layers", 1, 4),
                ff_mult=trial.suggest_categorical("ff_mult", [2, 4]),
                dropout=trial.suggest_float("dropout", 0.0, 0.3),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            res = train_model(cache_dir, m, Path(tmp) / "trial.pt", max_steps_per_epoch=steps_per_epoch, quiet=True)
        trial.set_user_attr("coverage_10_90", res.val_coverage_10_90)
        return res.best_val_pinball

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return study
