"""Generate the synthetic sandbox portfolio as Parquet.

    python scripts/generate_sandbox.py                       # 125 accounts x 800 days = 100,000 rows
    python scripts/generate_sandbox.py --accounts 50000 --out data/big_sandbox
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import _bootstrap_path  # noqa: F401

from config.settings import get_settings
from data_engine.synth import SandboxConfig, generate_sandbox
from utils.logging import setup_logging

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", type=int, default=125)
    ap.add_argument("--days", type=int, default=800)
    ap.add_argument("--end-date", type=date.fromisoformat, default=date(2026, 10, 31))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-accounts", type=int, default=20_000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    setup_logging()
    cfg = SandboxConfig(a.accounts, a.days, a.end_date, a.seed, chunk_accounts=a.chunk_accounts)
    print(json.dumps(generate_sandbox(cfg, a.out or get_settings().sandbox_dir), indent=2))
