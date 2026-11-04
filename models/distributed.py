"""Portfolio-wide nightly scoring on Ray.

Each actor loads the model once and memory-maps the same vector cache, so
the OS page cache shares the tensors across actors on a node. The driver
shards the account rows, collects alert tables and persists one run. On a
multi-node cluster put the cache on shared storage, or rebuild it per node.
"""

from __future__ import annotations

import uuid
from datetime import date

import numpy as np
import pyarrow as pa

from config.settings import Settings
from data_engine.alerts import utcnow
from data_engine.schema import EWS_ALERTS
from models.pipeline import EWSPipeline
from utils.logging import logger


def score_portfolio(
    settings: Settings,
    horizon: int = 90,
    num_actors: int = 2,
    shard_size: int = 5_000,
    gpus_per_actor: float = 0.0,
    address: str | None = None,
) -> dict:
    import ray

    if not ray.is_initialized():
        ray.init(address=address, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)

    @ray.remote(num_gpus=gpus_per_actor)
    class Scorer:
        def __init__(self, s: Settings):
            from data_engine.vector_cache import VectorCache

            self.pipe = EWSPipeline(s, storage=_ReadOnlyStorage())
            self.cache = VectorCache(s.cache.root).load()
            self.lq, self.aq = s.risk.liability_quantile, s.risk.asset_quantile

        def score(self, rows: np.ndarray, as_of: date, run_id: str) -> pa.Table:
            sb = self.pipe.score_rows(self.cache, rows, as_of, horizon, self.lq, self.aq)
            return self.pipe.alert_table(self.cache, sb, run_id)

    driver = EWSPipeline(settings)
    driver.refresh()
    cache = driver.cache.load()
    as_of = cache.end
    run_id = "run_" + uuid.uuid4().hex[:12]
    actors = [Scorer.remote(settings) for _ in range(max(1, num_actors))]
    shards = [np.arange(a, min(a + shard_size, cache.n_accounts)) for a in range(0, cache.n_accounts, shard_size)]
    futures = [actors[i % len(actors)].score.remote(rows, as_of, run_id) for i, rows in enumerate(shards)]
    tables = [t for t in ray.get(futures) if t.num_rows]
    alerts = pa.concat_tables(tables) if tables else None
    n_alerts = alerts.num_rows if alerts is not None else 0
    el = 0.0
    if alerts is not None:
        el = float(np.nansum(np.asarray(alerts.column("expected_loss").to_pylist(), dtype=float)))
    driver.alerts.record_run(
        {
            "run_id": run_id,
            "created_at": utcnow(),
            "as_of_date": as_of,
            "horizon": horizon,
            "backend": settings.model.backend + "+ray",
            "n_accounts": cache.n_accounts,
            "n_alerts": n_alerts,
            "total_expected_loss": el,
            "params": f'{{"actors": {len(actors)}, "shards": {len(shards)}}}',
        },
        cache.accounts["account_id"].tolist(),
        alerts if alerts is not None else EWS_ALERTS.arrow_schema.empty_table(),
    )
    logger.info("ray run {}: {} accounts, {} alerts across {} actors", run_id, cache.n_accounts, n_alerts, len(actors))
    return {"run_id": run_id, "accounts": cache.n_accounts, "alerts": n_alerts, "total_expected_loss": el,
            "actors": len(actors), "shards": len(shards)}


class _ReadOnlyStorage:
    """Actors only read the memory-mapped cache; the database stays with the driver
    (an embedded DuckDB file can only be opened by one process for writing)."""

    def __getattr__(self, name):
        raise RuntimeError("storage is not available inside Ray scoring actors")
