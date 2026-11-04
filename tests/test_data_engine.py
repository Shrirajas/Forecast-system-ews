from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config.settings import CacheSettings, Settings, StorageSettings
from data_engine import calendar as cal
from data_engine.aggregation import FeatureAggregator
from data_engine.ingestion import IngestionEngine
from data_engine.storage import DuckDBStorage
from data_engine.synth import SandboxConfig, generate_sandbox
from tests.conftest import LOAD_ORDER, N_ACCOUNTS, N_DAYS


def _fresh_storage(tmp_path) -> DuckDBStorage:
    s = Settings(storage=StorageSettings(duckdb_path=tmp_path / "t.duckdb"), cache=CacheSettings(root=tmp_path / "c"))
    st = DuckDBStorage(s)
    st.init_schema()
    return st


def test_default_sandbox_is_100k_rows(tmp_path):
    counts = generate_sandbox(SandboxConfig(), tmp_path)
    assert counts["daily_balances"] == 100_000
    assert counts["accounts"] == 125
    assert counts["transactions"] > 100_000


def test_sandbox_shape_and_invariants(sandbox_dir):
    db = pq.read_table(sandbox_dir / "daily_balances.parquet").to_pandas()
    assert len(db) == N_ACCOUNTS * N_DAYS
    assert (db.eod_balance >= 0).all()
    assert (db.credit_drawn <= db.credit_limit * 1.05 + 1).all()
    sched = pq.read_table(sandbox_dir / "loan_schedules.parquet").to_pandas()
    assert pd.to_datetime(sched.due_date).max() > pd.Timestamp("2027-01-31")  # covers the forecast horizon
    tx = pq.read_table(sandbox_dir / "transactions.parquet").to_pandas()
    assert tx.txn_id.is_unique
    assert set(tx.status) <= {"POSTED", "RETURNED"}


def test_sandbox_is_deterministic(tmp_path):
    cfg = SandboxConfig(n_accounts=5, n_days=200, seed=11)
    generate_sandbox(cfg, tmp_path / "a")
    generate_sandbox(cfg, tmp_path / "b")
    a = pq.read_table(tmp_path / "a" / "daily_balances.parquet")
    b = pq.read_table(tmp_path / "b" / "daily_balances.parquet")
    assert a.equals(b)


def test_calendar_rules():
    days = cal.day_range(date(2026, 11, 1), 30)
    hol = np.array(["2026-11-08"], dtype="datetime64[D]")
    nb = cal.non_business_mask(days, hol)
    as_str = {str(d) for d in days[nb]}
    assert "2026-11-01" in as_str  # Sunday
    assert "2026-11-14" in as_str  # 2nd Saturday
    assert "2026-11-07" not in as_str  # 1st Saturday is a working day
    assert "2026-11-08" in as_str  # holiday
    rolled = cal.monthly_dates(np.array([1]), date(2026, 11, 1), date(2026, 11, 30), hol, roll=+1)
    assert str(rolled[0, 0]) == "2026-11-02"


def test_upsert_is_idempotent_and_dedupes_within_batch(tmp_path):
    st = _fresh_storage(tmp_path)
    eng = IngestionEngine(st)
    rows = pa.table(
        {
            "account_id": ["AC1", "AC1", "AC2"],
            "as_of_date": [date(2026, 1, 1)] * 3,
            "eod_balance": [1.0, 2.0, 3.0],
            "credit_limit": [10.0] * 3,
            "credit_drawn": [0.0] * 3,
        }
    )
    eng.ingest("daily_balances", rows)
    eng.ingest("daily_balances", rows)
    out = st.query_df("SELECT account_id, eod_balance FROM daily_balances ORDER BY account_id")
    assert out.to_dict("list") == {"account_id": ["AC1", "AC2"], "eod_balance": [2.0, 3.0]}  # last write wins


def test_incremental_refresh_matches_full_rebuild(tmp_path, sandbox_dir):
    st = _fresh_storage(tmp_path)
    eng = IngestionEngine(st)
    for t in ("holidays", "accounts", "loan_schedules", "credit_profiles"):
        eng.ingest_file(t, sandbox_dir / f"{t}.parquet", "parquet")
    cut = pa.scalar(date(2026, 9, 1), pa.date32())
    db = pq.read_table(sandbox_dir / "daily_balances.parquet")
    tx = pq.read_table(sandbox_dir / "transactions.parquet")
    agg = FeatureAggregator(st)

    eng.ingest("daily_balances", db.filter(pc.less(db["as_of_date"], cut)))
    eng.ingest("transactions", tx.filter(pc.less(tx["txn_date"], cut)))
    assert agg.refresh().mode == "full"
    eng.ingest("daily_balances", db.filter(pc.greater_equal(db["as_of_date"], cut)))
    eng.ingest("transactions", tx.filter(pc.greater_equal(tx["txn_date"], cut)))
    r = agg.refresh()
    assert r.mode == "incremental" and r.start_date == date(2026, 9, 1)
    incremental = st.query_df("SELECT * FROM feature_matrix ORDER BY account_id, as_of_date")
    agg.refresh(full=True)
    full = st.query_df("SELECT * FROM feature_matrix ORDER BY account_id, as_of_date")
    assert len(incremental) == len(full) == N_ACCOUNTS * N_DAYS
    num = incremental.select_dtypes("number").columns
    np.testing.assert_allclose(incremental[num].fillna(-1).values, full[num].fillna(-1).values, rtol=1e-9)
    assert agg.refresh().mode == "noop"


def test_late_arriving_data_rewinds_watermark(tmp_path, sandbox_dir):
    st = _fresh_storage(tmp_path)
    eng = IngestionEngine(st)
    for t in LOAD_ORDER:
        eng.ingest_file(t, sandbox_dir / f"{t}.parquet", "parquet")
    agg = FeatureAggregator(st)
    agg.refresh()
    late = pa.table(
        {
            "txn_id": ["LATE-1"],
            "account_id": ["AC00000001"],
            "txn_date": [date(2026, 3, 10)],
            "amount": [123.0],
            "direction": ["DR"],
            "txn_type": ["EMI"],
            "status": ["RETURNED"],
        }
    )
    eng.ingest("transactions", late)
    r = agg.refresh()
    assert r.mode == "incremental" and r.start_date == date(2026, 3, 10)
    flag = st.scalar(
        "SELECT payment_return_flag FROM feature_matrix WHERE account_id = 'AC00000001' AND as_of_date = DATE '2026-03-20'"
    )
    assert flag == 1


def test_rolling_features_match_pandas(env, sandbox_dir):
    _, pipe = env
    a = "AC00000003"
    f = pipe.storage.query_df(f"SELECT * FROM feature_matrix WHERE account_id = '{a}' ORDER BY as_of_date")
    tx = pq.read_table(sandbox_dir / "transactions.parquet").to_pandas()
    out = tx[(tx.account_id == a) & (tx.direction == "DR") & (tx.status == "POSTED")].groupby("txn_date").amount.sum()
    daily = pd.Series(f.as_of_date.map(out).fillna(0).values)
    ref = daily.rolling(30, min_periods=2).std().fillna(0).values
    np.testing.assert_allclose(f.outward_std_30d.values, ref, atol=1e-6)


def test_vector_cache_alignment(env):
    _, pipe = env
    pipe.refresh()
    cache = pipe.cache.load()
    assert cache.past.shape[0] >= N_ACCOUNTS and cache.past.shape[1:] == (N_DAYS, 6)
    rows, missing = cache.rows_for(["AC00000002", "ZZZ"])
    assert missing == ["ZZZ"]
    w = cache.window(rows, cache.end, 512, 96)
    assert w["past"].shape == (1, 512, 6) and w["known"].shape == (1, 608, 3)
    last = pipe.storage.query_df(
        "SELECT eod_balance FROM feature_matrix WHERE account_id = 'AC00000002' ORDER BY as_of_date DESC LIMIT 1"
    )
    assert np.isclose(w["past"][0, -1, 0], last.eod_balance[0])
    sched = pipe.storage.query_df(
        "SELECT due_date, emi_amount FROM loan_schedules WHERE account_id = 'AC00000002' "
        "AND due_date > CAST(:d AS DATE) ORDER BY due_date LIMIT 3",
        {"d": cache.end},
    )
    future = w["known"][0, 512:, 0]
    for d, amt in zip(sched.due_date, sched.emi_amount, strict=True):
        day = (d - cache.end).days - 1
        if day < 96:
            assert np.isclose(future[day], amt)
    assert not pipe.cache.is_stale(pipe.storage)


def test_cache_goes_stale_on_ingest_and_refresh_fixes_it(env):
    _, pipe = env
    pipe.refresh()
    assert not pipe.cache.is_stale(pipe.storage)
    IngestionEngine(pipe.storage).ingest(
        "holidays", pa.table({"holiday_date": [date(2027, 8, 15)], "name": ["Independence Day"]})
    )
    assert pipe.cache.is_stale(pipe.storage)
    assert pipe.refresh()["cache_rebuilt"] is True
    assert not pipe.cache.is_stale(pipe.storage)
