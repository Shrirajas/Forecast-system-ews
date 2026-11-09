"""Measure /api/v1/predict/ews latency in-process (no network), on any loaded book.

    python scripts/bench_api_latency.py --work data/berka --checkpoint artifacts/models/berka_stformer.pt

Scenarios:
  whole_book_default        every account, default options (stored book run when available)
  whole_book_flagged_paths  every account, paths for flagged accounts only
  whole_book_all_paths      every account, every path (always live)
  whole_book_live_no_paths  every account, no paths, triggers with no stored run (q20): answered live
  accounts_20_paths         20 named accounts, paths included
  after_ingest              a one-row ingest (no refresh requested), then 20 accounts:
                            shows any freshness work done inside the request path
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import _bootstrap_path  # noqa: F401
from fastapi.testclient import TestClient

from api.main import create_app
from config.settings import PROJECT_ROOT, CacheSettings, ModelSettings, Settings, StorageSettings


def timed(client: TestClient, body: dict) -> tuple[float, int, dict]:
    t = time.perf_counter()
    r = client.post("/api/v1/predict/ews", json=body)
    ms = (time.perf_counter() - t) * 1000
    r.raise_for_status()
    return ms, len(r.content), r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--duckdb", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/models/ews_stformer.pt")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    settings = Settings(
        data_dir=a.work,
        storage=StorageSettings(duckdb_path=a.duckdb or a.work / "ews.duckdb", duckdb_memory_limit="6GB"),
        cache=CacheSettings(root=a.work / ("cache" if (a.work / "cache").exists() else "vector_cache")),
        model=ModelSettings(checkpoint_path=a.checkpoint),
        log_level="WARNING",
    )
    results: dict[str, dict] = {}
    with TestClient(create_app(settings)) as client:
        ids = [r["account_id"] for r in client.post(
            "/api/v1/predict/ews", json={"include_paths": False, "persist_alerts": False}).json()["accounts"][:20]]
        scenarios = {
            "whole_book_default": {},
            "whole_book_flagged_paths": {"include_paths": "flagged"},
            "whole_book_all_paths": {"include_paths": True},
            "whole_book_live_no_paths": {"include_paths": False, "quantile_triggers": {"liability": 0.2, "asset": 0.9}},
            "accounts_20_paths": {"account_ids": ids, "include_paths": True},
        }
        for name, body in scenarios.items():
            runs = [timed(client, body) for _ in range(a.repeat)]
            results[name] = {
                "median_ms": round(statistics.median(r[0] for r in runs), 1),
                "response_mb": round(runs[-1][1] / 1e6, 3),
                "accounts": len(runs[-1][2]["accounts"]),
                "source": runs[-1][2].get("source", "live"),
            }
        # one-row re-ingest of an existing balance: data unchanged, but the ingest marker moves
        row = client.app.state.services.pipeline.storage.query_arrow(
            "SELECT * FROM daily_balances WHERE account_id = :a ORDER BY as_of_date DESC LIMIT 1", {"a": ids[0]}
        ).to_pylist()[0]
        payload = {"daily_balances": [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}],
                   "refresh_features": False}
        client.post("/api/v1/ingest", json=payload).raise_for_status()
        ms, size, body = timed(client, {"account_ids": ids, "include_paths": True})
        results["after_ingest"] = {"ms": round(ms, 1), "response_mb": round(size / 1e6, 3),
                                   "source": body.get("source", "live")}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
