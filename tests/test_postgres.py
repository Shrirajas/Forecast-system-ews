"""PostgreSQL adapter parity test (opt-in).

    EWS_TEST_POSTGRES_DSN=postgresql+psycopg2://user@localhost:5432/ews_test pytest tests/test_postgres.py

The target database is wiped (EWS tables dropped) before the run.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from config.settings import CacheSettings, Settings, StorageSettings
from data_engine.aggregation import FeatureAggregator
from data_engine.ingestion import IngestionEngine
from data_engine.schema import ALL_TABLES
from data_engine.storage import DuckDBStorage, PostgresStorage
from tests.conftest import LOAD_ORDER

DSN = os.environ.get("EWS_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set EWS_TEST_POSTGRES_DSN to run against PostgreSQL")


def test_postgres_matches_duckdb(tmp_path, sandbox_dir):
    pg = PostgresStorage(Settings(storage=StorageSettings(backend="postgres", postgres_dsn=DSN)))
    for t in ALL_TABLES:
        pg.execute(f"DROP TABLE IF EXISTS {t}")
    pg.init_schema()
    dk = DuckDBStorage(Settings(storage=StorageSettings(duckdb_path=tmp_path / "d.duckdb"), cache=CacheSettings(root=tmp_path)))
    dk.init_schema()
    for st in (pg, dk):
        eng = IngestionEngine(st)
        for t in LOAD_ORDER:
            eng.ingest_file(t, sandbox_dir / f"{t}.parquet", "parquet")
        eng.ingest_file("daily_balances", sandbox_dir / "daily_balances.parquet", "parquet")  # idempotent
        FeatureAggregator(st).refresh(full=True)
    q = "SELECT * FROM feature_matrix ORDER BY account_id, as_of_date"
    a, b = pg.query_df(q), dk.query_df(q)
    assert len(a) == len(b) > 0
    cols = ["eod_balance", "utilization", "cibil_score", "outward_std_30d", "missed_salary_flag", "payment_return_flag"]
    np.testing.assert_allclose(a[cols].astype(float).fillna(-1).values, b[cols].astype(float).fillna(-1).values)
    assert pg.table_counts()["daily_balances"] == dk.table_counts()["daily_balances"]
    pg.close()
