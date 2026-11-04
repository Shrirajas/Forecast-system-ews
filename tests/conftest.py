from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import CacheSettings, ModelSettings, Settings, StorageSettings  # noqa: E402
from data_engine.ingestion import IngestionEngine  # noqa: E402
from data_engine.storage import get_storage, reset_storage_cache  # noqa: E402
from data_engine.synth import SandboxConfig, generate_sandbox  # noqa: E402

LOAD_ORDER = ("holidays", "accounts", "loan_schedules", "credit_profiles", "daily_balances", "transactions")
N_ACCOUNTS, N_DAYS = 30, 700


def make_settings(root: Path) -> Settings:
    return Settings(
        data_dir=root,
        sandbox_dir=root / "sandbox",
        warm_cache_on_startup=True,
        worker_threads=2,
        storage=StorageSettings(duckdb_path=root / "ews.duckdb", duckdb_memory_limit="1GB"),
        cache=CacheSettings(root=root / "cache"),
        model=ModelSettings(
            checkpoint_path=root / "model.pt",
            d_model=32,
            n_heads=2,
            n_layers=1,
            ff_mult=2,
            epochs=1,
            batch_size=32,
            window_stride=40,
            num_workers=0,
        ),
    )


@pytest.fixture(scope="session")
def sandbox_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("sandbox")
    generate_sandbox(SandboxConfig(n_accounts=N_ACCOUNTS, n_days=N_DAYS, end_date=date(2026, 10, 31), seed=3), d)
    return d


@pytest.fixture(scope="session")
def env(tmp_path_factory, sandbox_dir):
    """Loaded database + vector cache + tiny trained model shared by the suite."""
    from models.engine import reset_engine
    from models.pipeline import EWSPipeline
    from models.train import train_model

    root = tmp_path_factory.mktemp("ews")
    settings = make_settings(root)
    storage = get_storage(settings)
    eng = IngestionEngine(storage)
    for t in LOAD_ORDER:
        eng.ingest_file(t, sandbox_dir / f"{t}.parquet", "parquet")
    pipe = EWSPipeline(settings, storage=storage)
    pipe.refresh(full=True, force_cache=True)
    train_model(pipe.cache.current_dir(), settings.model, max_steps_per_epoch=3, quiet=True)
    reset_engine()
    yield settings, pipe
    reset_engine()
    reset_storage_cache()
