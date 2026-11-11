"""Per-request forecast backends: load once, never fall back, never disturb a loaded engine."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import models.engine as eng_mod
from config.settings import ModelSettings, Settings
from models.engine import (
    EngineUnavailable, ForecastEngine, _key, engine_status, get_engine, reset_engine,
)
from models.pipeline import ScoreRequest


class StubEngine(ForecastEngine):
    context_length, horizon_pad = 512, 96

    def __init__(self, name: str):
        self.name, self.model_version, self.calls = name, f"{name}:stub", 0

    def forecast(self, past, known, horizon, has_credit_line=None):
        self.calls += 1
        base = np.nan_to_num(past[:, -1, :2])[:, :, None, None]
        return (base + np.arange(9, dtype=np.float32) * 1.0 + np.zeros((1, 1, horizon, 1))).astype(np.float32)


@pytest.fixture
def clean_registry():
    reset_engine()
    yield
    reset_engine()


def _settings(allow: bool, tmp_path=None) -> Settings:
    m = ModelSettings(allow_noncommercial_weights=allow)
    if tmp_path is not None:
        m = m.model_copy(update={"checkpoint_path": tmp_path / "missing.pt"})
    return Settings(environment="sandbox", model=m)


def test_defaults_are_stformer_and_licence_flag_off():
    f = ModelSettings.model_fields
    assert f["backend"].default == "stformer" and f["allow_noncommercial_weights"].default is False


def test_timesfm_refused_without_licence_flag_and_nothing_else_answers(clean_registry):
    with pytest.raises(EngineUnavailable, match="non-commercial"):
        get_engine(_settings(False), "timesfm3")
    assert eng_mod._engines == {}  # no STFormer got loaded as a "fallback"


def test_missing_weights_raise_instead_of_falling_back(clean_registry, monkeypatch):
    def boom(*a, **k):
        raise OSError("offline: weights not cached")

    monkeypatch.setattr(eng_mod, "TimesFM3Engine", boom)
    with pytest.raises(EngineUnavailable, match="could not be loaded"):
        get_engine(_settings(True), "timesfm3")
    assert eng_mod._engines == {}


def test_failed_load_leaves_loaded_engines_untouched(clean_registry, monkeypatch):
    s = _settings(True)
    stub = StubEngine("stformer")
    monkeypatch.setitem(eng_mod._engines, _key(s, "stformer"), stub)
    monkeypatch.setattr(eng_mod, "TimesFM3Engine", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(EngineUnavailable):
        get_engine(s, "timesfm3")
    assert get_engine(s, "stformer") is stub


def test_each_backend_is_loaded_once_and_reused(clean_registry, monkeypatch):
    s, built = _settings(True), []
    monkeypatch.setattr(eng_mod, "_build_timesfm", lambda st: built.append(1) or StubEngine("timesfm3"))
    a = get_engine(s, "timesfm3")
    assert get_engine(s, "timesfm3") is a and len(built) == 1
    assert get_engine(s, "timesfm3", reload=True) is not a and len(built) == 2


def test_cuda_oom_at_load_retries_the_same_model_on_cpu(clean_registry, monkeypatch):
    devices = []

    class Fake(StubEngine):
        def __init__(self, checkpoint, ctx, *a, **k):
            devices.append(ctx.device.type)
            if len(devices) == 1:
                raise torch.cuda.OutOfMemoryError("oom")
            super().__init__("timesfm3")
            self.device_note = None

    monkeypatch.setattr(eng_mod, "TimesFM3Engine", Fake)
    monkeypatch.setattr(eng_mod, "get_device_context", lambda *a, force_cpu=False, **k: type(
        "C", (), {"device": torch.device("cpu" if force_cpu else "cuda")})())
    e = get_engine(_settings(True), "timesfm3")
    assert e.name == "timesfm3" and devices == ["cuda", "cpu"] and e.device_note.startswith("cpu")


def test_unknown_backend_is_rejected(clean_registry):
    with pytest.raises(ValueError, match="unknown forecast backend"):
        get_engine(_settings(False), "prophet")


def test_status_reports_without_loading(clean_registry, monkeypatch, tmp_path):
    monkeypatch.setattr(eng_mod, "_build_timesfm", lambda *a: pytest.fail("status must not load weights"))
    st = engine_status(_settings(False, tmp_path))
    assert st["timesfm3"]["available"] is False and "non-commercial" in st["timesfm3"]["reasons"][0]
    assert st["stformer"]["available"] is False and not st["stformer"]["loaded"]
    assert engine_status(_settings(True, tmp_path))["timesfm3"]["reasons"] == []


def test_request_backend_reaches_the_engine_and_persists_no_alerts(env, monkeypatch):
    settings, pipe = env
    ids = pipe.cache.load().accounts["account_id"].tolist()[:3]
    stub = StubEngine("timesfm3")
    monkeypatch.setitem(eng_mod._engines, _key(settings, "timesfm3"), stub)
    out = pipe.score(ScoreRequest(account_ids=ids, horizon=30, backend="timesfm3", paths="none"))
    assert out["backend"] == "timesfm3" and out["model_version"] == "timesfm3:stub" and stub.calls == 1
    stored = {r["run_id"] for r in pipe.alerts.recent_runs(10_000)}
    assert out["run_id"] not in stored  # research backend wrote nothing to the alert book
    default = pipe.score(ScoreRequest(account_ids=ids, horizon=30, paths="none"))
    assert default["backend"] == "stformer"
    assert default["run_id"] in {r["run_id"] for r in pipe.alerts.recent_runs(10_000)}


def test_unavailable_backend_on_a_request_raises(env):
    _, pipe = env
    with pytest.raises(EngineUnavailable):
        pipe.score(ScoreRequest(account_ids=None, horizon=30, backend="timesfm3"))


def test_cublas_alloc_failure_is_treated_as_oom_and_halves_the_batch():
    from utils.device import run_with_oom_backoff

    sizes = []

    def fn(a, b):
        sizes.append(b - a)
        if len(sizes) == 1:
            raise RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`")
        return b - a

    assert sum(run_with_oom_backoff(fn, 8, 8)) == 8 and sizes[:2] == [8, 4]
    with pytest.raises(RuntimeError, match="boom"):  # other RuntimeErrors are not swallowed
        list(run_with_oom_backoff(lambda a, b: (_ for _ in ()).throw(RuntimeError("boom")), 8, 8))
