"""Train vs check error: both measured in eval mode; customer-level row split; provenance in the checkpoint."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from models.dataset import WindowSpec, build_window_index
from models.engine import STFormerEngine
from models.stformer import EWSForecaster, STFormerConfig
from models.train import evaluate, train_model
from utils.device import get_device_context


def test_train_history_has_eval_mode_train_loss_next_to_the_check_loss(env, tmp_path):
    settings, pipe = env
    m = settings.model.model_copy(update={"epochs": 2})
    ck = tmp_path / "t.pt"
    train_model(pipe.cache.current_dir(), m, checkpoint_path=ck, max_steps_per_epoch=3, quiet=True,
                anchor_last=True, meta={"split_rule": "test"})
    hist = json.loads(ck.with_suffix(".progress.json").read_text())["history"]
    assert len(hist) == 2
    for h in hist:
        assert {"train_pinball", "train_pinball_eval", "train_pinball_eval_recent", "pinball"} <= set(h)
        assert all(np.isfinite(h[k]) for k in ("train_pinball_eval", "train_pinball_eval_recent", "pinball"))
    blob = torch.load(ck, weights_only=False)
    assert blob["meta"] == {"split_rule": "test"} and blob["config"]["anchor_last"] is True
    assert json.loads(ck.with_suffix(".json").read_text())["meta"] == {"split_rule": "test"}
    assert STFormerEngine(ck, get_device_context(force_cpu=True)).cfg.anchor_last is True


def test_evaluate_runs_with_dropout_off_and_is_deterministic():
    cfg = STFormerConfig(d_model=32, n_heads=2, n_layers=1, ff_mult=2, dropout=0.5)
    model = EWSForecaster(cfg)
    model.train()  # evaluate() must switch to eval itself
    rng = np.random.default_rng(1)
    past = torch.from_numpy(rng.random((4, 512, 6)).astype(np.float32) * 10 + 1)
    known, fut = torch.zeros(4, 608, 3), torch.from_numpy(rng.random((4, 96, 2)).astype(np.float32))
    ctx = get_device_context(force_cpu=True)
    a = evaluate(model, [(past, known, fut)], ctx, 90)
    assert not model.training
    assert evaluate(model, [(past, known, fut)], ctx, 90) == a


def test_row_split_is_respected_and_overlap_is_refused():
    spec = WindowSpec(512, 90, 96)
    train, val = np.arange(0, 80), np.arange(80, 100)
    tr, va = build_window_index(100, 1000, spec, 30, 0.2, 7, row_split=(train, val))
    assert set(tr[:, 0]) == set(train) and set(va[:, 0]) == set(val)
    with pytest.raises(ValueError, match="both"):
        build_window_index(100, 1000, spec, 30, 0.2, 7, row_split=(train, np.arange(70, 100)))


def test_group_split_keeps_every_customer_on_one_side():
    from models.dataset import split_rows_by_group

    groups = np.array([f"CC{i % 40:04d}" for i in range(200)])  # 40 customers, 5 cards each
    tr, va, te = split_rows_by_group(groups, 0.15, 0.25, seed=3)
    assert sorted(np.concatenate([tr, va, te])) == list(range(200))
    sets = [set(groups[x]) for x in (tr, va, te)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(sets[2]) == 10 and len(sets[1]) == 6


def test_warm_start_accepts_old_configs_and_refuses_an_anchor_mismatch(env, tmp_path):
    settings, pipe = env
    m = settings.model.model_copy(update={"epochs": 1})
    base = tmp_path / "base.pt"
    train_model(pipe.cache.current_dir(), m, checkpoint_path=base, max_steps_per_epoch=2, quiet=True)
    blob = torch.load(base, weights_only=False)
    blob["config"].pop("anchor_last")  # checkpoints written before the flag existed
    torch.save(blob, base)
    train_model(pipe.cache.current_dir(), m, checkpoint_path=tmp_path / "ft.pt", max_steps_per_epoch=2,
                quiet=True, init_checkpoint=base)
    with pytest.raises(ValueError, match="architecture differs"):
        train_model(pipe.cache.current_dir(), m, checkpoint_path=tmp_path / "bad.pt", max_steps_per_epoch=2,
                    quiet=True, init_checkpoint=base, anchor_last=True)
