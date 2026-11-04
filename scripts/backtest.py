"""Out-of-time EWS backtest against realised EMI returns and balance outflows.

    python scripts/backtest.py                       # as of 90 days before the last data day
    python scripts/backtest.py --as-of 2026-06-30 --horizon 60 --liability-q 0.2
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import _bootstrap_path  # noqa: F401

from config.settings import get_settings
from models.backtest import backtest
from models.pipeline import EWSPipeline
from utils.logging import setup_logging

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=date.fromisoformat, default=None)
    ap.add_argument("--horizon", type=int, choices=[30, 60, 90], default=90)
    ap.add_argument("--liability-q", type=float, default=None)
    a = ap.parse_args()
    s = get_settings()
    setup_logging("WARNING")
    if a.liability_q is not None:
        s.risk.liability_quantile = a.liability_q
    print(json.dumps(backtest(EWSPipeline(s), a.as_of, a.horizon), indent=2))
