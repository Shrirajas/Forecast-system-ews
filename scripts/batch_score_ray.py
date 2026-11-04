"""Nightly portfolio scoring fanned out over Ray actors.

    python scripts/batch_score_ray.py --actors 4 --shard-size 5000
    python scripts/batch_score_ray.py --address auto --gpus-per-actor 0.25   # existing cluster, shared GPU
"""

from __future__ import annotations

import argparse
import json

import _bootstrap_path  # noqa: F401

from config.settings import get_settings
from models.distributed import score_portfolio
from utils.logging import setup_logging

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", type=int, default=2)
    ap.add_argument("--shard-size", type=int, default=5_000)
    ap.add_argument("--horizon", type=int, choices=[30, 60, 90], default=90)
    ap.add_argument("--gpus-per-actor", type=float, default=0.0)
    ap.add_argument("--address", default=None)
    a = ap.parse_args()
    s = get_settings()
    setup_logging(s.log_level)
    print(json.dumps(score_portfolio(s, a.horizon, a.actors, a.shard_size, a.gpus_per_actor, a.address), indent=2, default=str))
