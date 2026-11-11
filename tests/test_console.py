"""Web console: importer mapping/conversion, data import (file and SQL), database switching, jobs."""

from __future__ import annotations

import sqlite3
from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from config.settings import CacheSettings, Settings, StorageSettings
from data_engine import importer as imp


# ------------------------------------------------------------------------------------- importer
def test_suggest_mapping_understands_common_bank_exports():
    m = imp.suggest_mapping("daily_balances", ["Acct No", "Bal Date", "Closing Balance", "OD Limit", "Drawn"])
    assert m == {"account_id": "Acct No", "as_of_date": "Bal Date", "eod_balance": "Closing Balance",
                 "credit_limit": "OD Limit", "credit_drawn": "Drawn"}
    t = imp.suggest_mapping("transactions", ["Account", "Posting Dt", "Amt", "Dr/Cr", "Ref"])
    assert (t["account_id"], t["txn_date"], t["amount"], t["direction"], t["txn_id"]) == (
        "Account", "Posting Dt", "Amt", "Dr/Cr", "Ref")
    s = imp.suggest_mapping("loan_schedules", ["acct", "EMI Date", "Installment Amount"])
    assert s["account_id"] == "acct" and s["due_date"] == "EMI Date" and s["emi_amount"] == "Installment Amount"


def test_transform_converts_dates_numbers_and_debit_credit():
    src = pa.table({
        "Account": ["A1", "A1", "A2", "A2"],
        "Posting Dt": ["31/12/2024", "01/01/2025", "02/01/2025", "03/01/2025"],
        "Amt": ["1,234.50", "(500.00)", "₹ 2,000", "-75"],
        "Dr/Cr": ["D", "credit", "CR", "DR"],
    })
    opts = imp.ImportOptions("transactions", imp.suggest_mapping("transactions", src.column_names),
                             date_format="%d/%m/%Y", source_key="abc")
    assert opts.validate(src.column_names) == []
    out = imp.transform_batch(src, opts).to_pylist()
    assert [r["txn_date"] for r in out] == [date(2024, 12, 31), date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
    assert [r["amount"] for r in out] == [1234.5, 500.0, 2000.0, 75.0]
    assert [r["direction"] for r in out] == ["DR", "CR", "CR", "DR"]
    assert out[0]["status"] == "POSTED" and out[0]["txn_type"] == "OTHER"
    other = imp.transform_batch(src, imp.ImportOptions("transactions", opts.mapping, date_format="%d/%m/%Y",
                                                       source_key="xyz")).column("txn_id").to_pylist()
    assert set(other).isdisjoint(r["txn_id"] for r in out)  # generated ids never collide across sources


def test_signed_amounts_decimal_comma_and_defaults():
    src = pa.table({"acct": ["A1", "A1"], "date": ["2025-01-01", "2025-01-02"], "amount": ["-1.234,50", "10,00"]})
    opts = imp.ImportOptions("transactions", imp.suggest_mapping("transactions", src.column_names), decimal_comma=True)
    out = imp.transform_batch(src, opts).to_pylist()
    assert [(r["amount"], r["direction"]) for r in out] == [(1234.5, "DR"), (10.0, "CR")]
    acc = imp.transform_batch(pa.table({"acno": ["A9"]}), imp.ImportOptions("accounts", {"account_id": "acno"},
                              constants={"lgd": "0.6"})).to_pylist()[0]
    assert acc["customer_id"] == "A9" and acc["lgd"] == 0.6 and acc["segment"] == "RETAIL"


def test_validation_reports_missing_required_columns_and_bad_codes():
    opts = imp.ImportOptions("daily_balances", {"account_id": "a", "eod_balance": "b"})
    assert any("as_of_date" in p for p in opts.validate(["a", "b"]))
    bad = imp.ImportOptions("transactions", {"account_id": "a", "txn_date": "d", "amount": "x", "direction": "t"})
    with pytest.raises(ValueError, match="debit/credit"):
        imp.transform_batch(pa.table({"a": ["A"], "d": ["2025-01-01"], "x": ["1"], "t": ["MAYBE"]}), bad)
    with pytest.raises(ValueError, match="SELECT"):
        imp.sql_columns_and_sample("sqlite://", "DELETE FROM x")


# ------------------------------------------------------------------------------------- API
@pytest.fixture()
def console(env, tmp_path):
    base, _ = env
    s = Settings(
        data_dir=tmp_path, upload_spool_dir=tmp_path / "spool", warm_cache_on_startup=True,
        storage=StorageSettings(duckdb_path=tmp_path / "fresh.duckdb", duckdb_memory_limit="1GB"),
        cache=CacheSettings(root=tmp_path / "cache"), model=base.model, book_shard_size=1000,
    )
    with TestClient(create_app(s)) as c:
        yield c


def _sandbox_csv(sandbox_dir, accounts: int = 5) -> bytes:
    db = pq.read_table(sandbox_dir / "daily_balances.parquet").to_pandas()
    db = db[db.account_id.isin(sorted(db.account_id.unique())[:accounts])]
    out = db.rename(columns={"account_id": "Acct No", "as_of_date": "Bal Date", "eod_balance": "Closing Balance"})
    out["Bal Date"] = out["Bal Date"].map(lambda d: d.strftime("%d/%m/%Y"))
    out["Closing Balance"] = out["Closing Balance"].map(lambda v: f"{v:,.2f}")
    return out[["Acct No", "Bal Date", "Closing Balance"]].to_csv(index=False).encode()


def test_console_page_and_reference_endpoints(console):
    assert "EWS Console" in console.get("/ui/").text
    assert console.get("/", follow_redirects=False).status_code in (302, 307)
    tables = console.get("/api/v1/import/tables").json()
    assert {"accounts", "daily_balances", "transactions"} <= set(tables)
    tpl = console.get("/api/v1/import/template/daily_balances").text
    assert tpl.startswith("account_id,as_of_date,eod_balance")


def test_upload_map_import_then_score(console, sandbox_dir):
    r = console.post("/api/v1/import/upload", params={"table": "daily_balances"},
                     files={"file": ("export.csv", _sandbox_csv(sandbox_dir), "text/csv")})
    assert r.status_code == 200, r.text
    up = r.json()
    assert up["suggested_mapping"]["as_of_date"] == "Bal Date" and len(up["sample"]) == 20
    body = {"table": "daily_balances", "upload_id": up["upload_id"], "mapping": up["suggested_mapping"],
            "date_format": "%d/%m/%Y", "refresh_after": True}
    job = console.post("/api/v1/import/commit", params={"wait": "true"}, json=body).json()["job"]
    assert job["status"] == "done", job
    assert job["result"]["rows"] == 5 * 700 and job["result"]["refresh"]["book"]["alerts"] >= 0
    ov = console.get("/api/v1/console/overview").json()
    assert ov["tables"]["daily_balances"] == 3500 and ov["latest_book_run"]["up_to_date"] is True
    book = console.post("/api/v1/predict/ews", json={"horizon": 90}).json()
    assert book["source"] == "stored" and book["summary"]["accounts_scored"] == 5
    # same file again: idempotent
    up2 = console.post("/api/v1/import/upload", params={"table": "daily_balances"},
                       files={"file": ("export.csv", _sandbox_csv(sandbox_dir), "text/csv")}).json()
    body["upload_id"], body["refresh_after"] = up2["upload_id"], False
    console.post("/api/v1/import/commit", params={"wait": "true"}, json=body)
    assert console.get("/api/v1/console/overview").json()["tables"]["daily_balances"] == 3500
    # an unmapped required column is refused up front
    bad = {**body, "mapping": {**body["mapping"], "as_of_date": None}}
    assert console.post("/api/v1/import/commit", json=bad).status_code == 422


def test_import_from_a_sql_source(console, tmp_path):
    src = tmp_path / "core.db"
    with sqlite3.connect(src) as db:
        db.execute("CREATE TABLE cust (acct_no TEXT, emi REAL, od_limit REAL)")
        db.executemany("INSERT INTO cust VALUES (?, ?, ?)", [("Z1", 1000, 5000), ("Z2", 2000, 0)])
    sql = {"url": f"sqlite:///{src}", "query": "SELECT acct_no, emi, od_limit FROM cust"}
    prev = console.post("/api/v1/import/sql/preview", params={"table": "accounts"}, json=sql).json()
    assert prev["suggested_mapping"]["account_id"] == "acct_no" and prev["suggested_mapping"]["emi_amount"] == "emi"
    job = console.post("/api/v1/import/commit", params={"wait": "true"}, json={
        "table": "accounts", "sql": sql, "mapping": prev["suggested_mapping"], "refresh_after": False}).json()["job"]
    assert job["status"] == "done" and job["result"]["rows"] == 2
    bad = {"url": f"sqlite:///{src}", "query": "DROP TABLE cust"}
    assert console.post("/api/v1/import/sql/preview", json=bad).status_code == 422


def test_switch_database_and_remember_it(console, tmp_path):
    spec = {"backend": "duckdb", "duckdb_path": str(tmp_path / "other.duckdb")}
    assert console.post("/api/v1/console/connection/test", json=spec).json()["ok"] is True
    r = console.post("/api/v1/console/connection", json=spec).json()
    assert r["connection"]["duckdb_path"].endswith("other.duckdb") and r["job"] is None  # empty: nothing to score
    assert console.get("/api/v1/console/overview").json()["tables"]["daily_balances"] == 0
    assert (tmp_path / "connection.json").exists()
    bad = console.post("/api/v1/console/connection/test", json={"backend": "postgres", "postgres_dsn": "mysql://x"})
    assert bad.status_code == 422


def test_jobs_and_remote_guard(console, env):
    job = console.post("/api/v1/jobs/score-book", params={"wait": "true"}).json()["job"]
    assert job["status"] in ("done", "failed")  # empty book fails cleanly, never hangs
    assert any(j["id"] == job["id"] for j in console.get("/api/v1/jobs").json())
    remote = TestClient(console.app, client=("203.0.113.9", 5000))
    assert remote.get("/api/v1/console/overview").status_code == 403
    assert remote.get("/health").status_code == 200  # the normal API is unaffected


def test_training_progress_endpoint(console, env):
    settings, _ = env
    p = console.get("/api/v1/console/training").json()
    assert p["status"] == "done" and p["steps_per_epoch"] == 3
    assert [r["epoch"] for r in p["history"]] == list(range(1, settings.model.epochs + 1))
    assert p["history"][0]["saved"] and p["checkpoint"] == str(settings.model.checkpoint_path)
    assert {"train_pinball", "pinball", "coverage_10_90", "median_mape"} <= set(p["history"][-1])
