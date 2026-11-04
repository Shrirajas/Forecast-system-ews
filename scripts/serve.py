"""Launch the API with Uvicorn.

    python scripts/serve.py --port 8000

One worker per DuckDB file (DuckDB allows a single writer process). To scale
horizontally, set EWS_STORAGE__BACKEND=postgres and run more workers/replicas.
"""

from __future__ import annotations

import argparse

import _bootstrap_path  # noqa: F401
import uvicorn

from config.settings import get_settings

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    if s.storage.backend == "duckdb" and a.workers > 1:
        raise SystemExit("DuckDB backend supports a single API worker; use the postgres backend to scale out")
    uvicorn.run("api.main:app", host=a.host, port=a.port, workers=a.workers, reload=a.reload,
                log_level=s.log_level.lower())
