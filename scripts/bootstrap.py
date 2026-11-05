"""One-shot sandbox bring-up.

    python scripts/bootstrap.py            # generate -> ingest -> aggregate -> cache -> train -> score
    python scripts/bootstrap.py --skip-train
    python scripts/bootstrap.py --accounts 20000 --days 800   # larger synthetic book

Steps:
 1. synthesise the 100,000-row sandbox ledger (Parquet) under data/sandbox,
 2. stream every table into the configured database,
 3. run the incremental aggregation and build the memory-mapped tensor cache,
 4. train the spatial-temporal forecaster (CUDA AMP if available, else a
    lighter CPU profile),
 5. score the portfolio, persist alerts, and backtest the EWS out-of-time
    (score as of 90 days before the last data day, compare with the EMI
    returns and balance outflows that actually followed).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import _bootstrap_path  # noqa: F401
import torch

from config.settings import get_settings
from data_engine.ingestion import IngestionEngine
from data_engine.storage import get_storage
from data_engine.synth import SandboxConfig, generate_sandbox
from models.backtest import backtest
from models.pipeline import EWSPipeline, ScoreRequest
from models.train import train_model
from utils.logging import logger, setup_logging

LOAD_ORDER = ("holidays", "accounts", "loan_schedules", "credit_profiles", "daily_balances", "transactions")


def cpu_profile(m):
    """Smaller network and fewer windows so CPU-only training takes minutes."""
    return m.model_copy(update=dict(d_model=96, n_layers=2, ff_mult=2, n_heads=4, window_stride=5, num_workers=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accounts", type=int, default=125)
    ap.add_argument("--days", type=int, default=800)
    ap.add_argument("--end-date", type=date.fromisoformat, default=date(2026, 10, 31))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-generate", action="store_true", help="reuse existing data/sandbox parquet files")
    args = ap.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    sandbox = settings.sandbox_dir

    if not args.skip_generate:
        cfg = SandboxConfig(n_accounts=args.accounts, n_days=args.days, end_date=args.end_date, seed=args.seed)
        counts = generate_sandbox(cfg, sandbox)
        logger.info("sandbox generated: {}", counts)

    storage = get_storage(settings)
    ingestion = IngestionEngine(storage, batch_rows=settings.storage.stream_batch_rows)
    for table in LOAD_ORDER:
        ingestion.ingest_file(table, sandbox / f"{table}.parquet", "parquet")

    pipeline = EWSPipeline(settings, storage=storage)
    refresh = pipeline.refresh(full=True, force_cache=True)
    logger.info("aggregation + cache: {}", refresh)

    m = settings.model
    if not torch.cuda.is_available():
        m = cpu_profile(m)
        logger.info("no CUDA device: using CPU training profile (d_model={}, layers={})", m.d_model, m.n_layers)
    if args.epochs:
        m = m.model_copy(update={"epochs": args.epochs})
    if not args.skip_train or not Path(m.checkpoint_path).exists():
        cache_dir = pipeline.cache.current_dir()
        result = train_model(cache_dir, m)
        logger.info("model trained: {}", result)

    out = pipeline.score(ScoreRequest(horizon=90, paths="none", use_stored=False))
    print(json.dumps(out["summary"], indent=2))
    top, total = pipeline.alerts.active(limit=10)
    print(f"\nTop alerts ({total} active):")
    for a in top:
        print(
            f"  #{a['rank']:<2} {a['account_id']} {a['alert_type']:<17} {a['risk_tier']:<5} "
            f"priority={a['priority_score']:>12,.0f}  EL={a['expected_loss'] or 0:>12,.0f}  "
            f"breach_day={a['days_to_breach']}  zero_day={a['days_to_zero']}"
        )
    print("\nOut-of-time backtest (realised outcomes over the following 90 days):")
    print(json.dumps(backtest(pipeline), indent=2))


if __name__ == "__main__":
    main()
