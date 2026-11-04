from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture(scope="module")
def client(env):
    settings, _ = env
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_and_info(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["cache_version"] and h["model_loaded"]
    info = client.get("/api/v1/system/info").json()
    assert info["storage"]["tables"]["daily_balances"] > 0
    assert info["engine"]["backend"] == "stformer"


@pytest.mark.parametrize("horizon", [30, 60, 90])
def test_predict_contract(client, horizon):
    r = client.post(
        "/api/v1/predict/ews",
        json={"horizon": horizon, "return_quantiles": [0.1, 0.5, 0.9], "persist_alerts": False},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["horizon"] == horizon and body["summary"]["accounts_scored"] == 30
    a = body["accounts"][0]
    assert set(a["forecast"]["balance"]) == {"q10", "q50", "q90"}
    assert len(a["forecast"]["dates"]) == horizon
    assert a["risk_tier"] in {"GREEN", "SMA-0", "SMA-1", "SMA-2"}
    for acc in body["accounts"]:
        q10, q50, q90 = (acc["forecast"]["balance"][k] for k in ("q10", "q50", "q90"))
        assert all(lo <= mid <= hi for lo, mid, hi in zip(q10, q50, q90, strict=True))
        if acc["risk_tier"] == "GREEN":
            assert acc["expected_loss"] is None
        else:
            assert acc["days_to_breach"] <= horizon
            assert acc["expected_loss"] == pytest.approx(
                acc["exposure_at_default"] * acc["probability_of_default"] * acc["loss_given_default"], rel=1e-3
            )


def test_predict_validation(client):
    assert client.post("/api/v1/predict/ews", json={"horizon": 45}).status_code == 422
    assert client.post("/api/v1/predict/ews", json={"return_quantiles": [0.15]}).status_code == 422
    bad = {"quantile_triggers": {"liability": 0.9, "asset": 0.1}}
    assert client.post("/api/v1/predict/ews", json=bad).status_code == 422
    r = client.post("/api/v1/predict/ews", json={"account_ids": ["AC00000001", "UNKNOWN01"], "include_paths": False})
    body = r.json()
    assert body["unknown_accounts"] == ["UNKNOWN01"] and len(body["accounts"]) == 1
    assert body["accounts"][0].get("forecast") is None


def test_alerts_ranked_json_sse_and_ack(client):
    r = client.post("/api/v1/predict/ews", json={"horizon": 90, "include_paths": False})
    assert r.status_code == 200
    raised = r.json()["summary"]["alerts_raised"]
    data = client.get("/api/v1/alerts/active", params={"limit": 500}).json()
    assert data["total"] == raised
    alerts = data["alerts"]
    assert [a["rank"] for a in alerts] == list(range(1, len(alerts) + 1))
    credit = [a for a in alerts if a["alert_type"] == "CREDIT_EWS"]
    assert credit == alerts[: len(credit)]  # credit alerts rank ahead of attrition
    pr = [a["priority_score"] for a in credit]
    assert pr == sorted(pr, reverse=True)

    with client.stream("GET", "/api/v1/alerts/active", params={"stream": "true", "limit": 5}) as s:
        assert s.headers["content-type"].startswith("text/event-stream")
        text = "".join(s.iter_text())
    events = [line[len("event: ") :] for line in text.splitlines() if line.startswith("event: ")]
    assert events[0] == "snapshot_start" and events[-1] == "snapshot_end"
    assert events.count("alert") == min(5, raised)

    if alerts:
        aid = alerts[0]["alert_id"]
        assert client.post(f"/api/v1/alerts/{aid}/ack").json()["status"] == "ACKNOWLEDGED"
        assert client.post(f"/api/v1/alerts/{aid}/ack").status_code == 404
        after = client.get("/api/v1/alerts/active", params={"limit": 500}).json()
        assert after["total"] == raised - 1
    # a re-run supersedes the previous run's alerts instead of duplicating them
    client.post("/api/v1/predict/ews", json={"horizon": 90, "include_paths": False})
    again = client.get("/api/v1/alerts/active", params={"limit": 500}).json()
    assert again["total"] == raised


def test_ingest_json_then_refresh(client):
    payload = {
        "accounts": [
            {
                "account_id": "NEW0000001",
                "customer_id": "CUNEW1",
                "segment": "RETAIL_SALARIED",
                "product_type": "PERSONAL",
                "open_date": "2026-01-01",
                "sanctioned_amount": 300000,
                "outstanding_principal": 250000,
                "emi_amount": 9000,
                "emi_day": 5,
                "salary_day": 1,
                "credit_limit": 50000,
                "lgd": 0.6,
                "interest_rate": 0.14,
            }
        ],
        "daily_balances": [
            {"account_id": "NEW0000001", "as_of_date": f"2026-10-{d:02d}", "eod_balance": 20000 + d * 100,
             "credit_limit": 50000, "credit_drawn": 1000}
            for d in range(1, 32)
        ],
        "loan_schedules": [{"account_id": "NEW0000001", "due_date": "2026-11-05", "emi_amount": 9000}],
        "transactions": [
            {"txn_id": "N1", "account_id": "NEW0000001", "txn_date": "2026-10-01", "amount": 45000,
             "direction": "CR", "txn_type": "salary"}
        ],
        "wait_for_refresh": True,
    }
    r = client.post("/api/v1/ingest", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tables"]["daily_balances"]["rows"] == 31 and body["refresh"] == "completed"
    assert body["refresh_result"]["cache_rebuilt"] is True
    p = client.post("/api/v1/predict/ews", json={"account_ids": ["NEW0000001"], "horizon": 30, "persist_alerts": False})
    acc = p.json()["accounts"][0]
    assert acc["next_emi"] == {"date": "2026-11-05", "amount": 9000.0}

    bad = {"daily_balances": [{"account_id": "x", "as_of_date": "2026-10-01", "eod_balance": 1}]}
    assert client.post("/api/v1/ingest", json=bad).status_code == 422
    assert client.post("/api/v1/ingest", json={}).status_code == 422


def test_ingest_file_streams_csv_and_parquet(client):
    csv = "account_id,pull_date,cibil_score,enquiries_6m\nAC00000001,2026-10-31,701,1\nAC00000002,2026-10-31,655,\n"
    r = client.post(
        "/api/v1/ingest/file",
        params={"table": "credit_profiles", "refresh_features": "false"},
        files={"file": ("bureau.csv", csv, "text/csv")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["tables"]["credit_profiles"]["rows"] == 2

    tbl = pa.table({"holiday_date": pa.array([__import__("datetime").date(2027, 1, 26)]), "name": ["Republic Day"]})
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    r = client.post(
        "/api/v1/ingest/file",
        params={"table": "holidays", "refresh_features": "false"},
        files={"file": ("h.parquet", buf.getvalue(), "application/octet-stream")},
    )
    assert r.status_code == 200 and r.json()["tables"]["holidays"]["rows"] == 1

    r = client.post(
        "/api/v1/ingest/file",
        params={"table": "transactions", "format": "csv"},
        files={"file": ("t.csv", "not,a,valid\nfile", "text/csv")},
    )
    assert r.status_code == 422


def test_dataset_ingest_rejects_paths_outside_data_dir(client):
    r = client.post("/api/v1/ingest/dataset", json={"table": "holidays", "path": "/etc"})
    assert r.status_code == 403
    assert json.loads(r.text)["detail"].startswith("path must be inside")


def test_run_expected_loss_matches_portfolio_summary(client):
    body = client.post("/api/v1/predict/ews", json={"horizon": 90, "include_paths": False}).json()
    runs = client.get("/api/v1/runs", params={"limit": 1}).json()
    assert runs[0]["run_id"] == body["run_id"]
    assert runs[0]["total_expected_loss"] == pytest.approx(body["summary"]["total_expected_loss"], abs=1.0)
    alerts = client.get("/api/v1/alerts/active", params={"alert_type": "DEPOSIT_ATTRITION", "limit": 500}).json()
    assert all(a["expected_loss"] is None for a in alerts["alerts"])
