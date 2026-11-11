"""Dataset switching, models, accounts, forecast, compare store, overview and alert additions (tmp datasets only).

Reuses the three tiny synthetic datasets of tests/test_acceptance_items.py (QA): never a real data/*.duckdb.
"""

from __future__ import annotations

import json
import threading

import pytest
from test_acceptance_items import DS, V1, accounts, api, forecast, multi, select  # noqa: F401  (fixtures)

from api import compare_store


def rt(api):
    return api.app.state.runtime


def test_item01_dataset_list_has_capabilities_and_one_active(api):
    select(api, "sandbox")
    body = api.get(f"{V1}/datasets").json()
    assert body["active"] == "sandbox" and [d["active"] for d in body["datasets"]].count(True) == 1
    d = next(x for x in body["datasets"] if x["id"] == "cards")
    for key in ("targets", "features", "capabilities", "supported_models", "alert_types", "labels", "currency", "status", "data"):
        assert key in d
    assert d["data"]["accounts"] == DS["cards"][2] and d["data"]["last_day"] == DS["cards"][4].isoformat()
    assert api.get(f"{V1}/datasets/active").json()["id"] == "sandbox"
    assert api.get(f"{V1}/datasets/nope").status_code == 404


def test_item04_everything_follows_the_active_dataset(api):
    for ds in ("berka", "cards", "sandbox"):
        select(api, ds)
        assert api.get("/health").json()["dataset_id"] == ds
        ov = api.get(f"{V1}/console/overview").json()
        assert ov["dataset"]["id"] == ds and ov["data"]["accounts"] == DS[ds][2] and ov["data"]["last_day"] == DS[ds][4].isoformat()
        for a in api.get(f"{V1}/alerts/active", params={"limit": 500}).json()["alerts"]:
            assert a["account_id"].startswith(DS[ds][0])  # alerts of another dataset never show up
        book = api.post(f"{V1}/predict/ews", json={"horizon": 30}).json()
        assert all(a["account_id"].startswith(DS[ds][0]) for a in book["accounts"])


def test_item08_switch_is_refused_while_a_job_runs_and_state_is_kept(api):
    select(api, "sandbox")
    gate, started = threading.Event(), threading.Event()
    job = api.app.state.services.jobs.submit("test", "blocking job", lambda: (started.set(), gate.wait(10)))
    assert started.wait(5)
    try:
        r = api.post(f"{V1}/datasets/select", json={"dataset_id": "cards"})
        assert r.status_code == 409 and r.json()["detail"]["code"] == "busy" and job.id in r.json()["detail"]["jobs"]
        assert api.get("/health").json()["dataset_id"] == "sandbox"
        assert api.post(f"{V1}/forecast", json={"account_id": accounts(api, "sandbox")["accounts"][0]["account_id"],
                                                 "horizon": 30}).status_code == 200  # the old dataset still serves
    finally:
        gate.set()
        api.app.state.services.jobs.wait(job.id, 10)
    select(api, "cards")  # job finished: the switch is allowed again


def test_item08_a_call_in_flight_blocks_the_switch_and_a_late_call_is_refused(api):
    select(api, "sandbox")
    svc = api.app.state.services
    with svc.busy():
        r = api.post(f"{V1}/datasets/select", json={"dataset_id": "cards"})
        assert r.status_code == 409 and r.json()["detail"]["code"] == "busy"
    select(api, "cards")
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:  # the retired services refuse new work instead of using a closed database
        with svc.busy():
            pass
    assert e.value.status_code == 409


def test_item08_a_dataset_that_cannot_open_is_refused_and_the_active_one_keeps_working(api):
    select(api, "sandbox")
    reg = rt(api).registry
    ghost = reg["berka"].model_copy(update={"duckdb_path": reg["berka"].duckdb_path.with_name("missing.duckdb")})
    original, reg["berka"] = reg["berka"], ghost
    try:
        r = api.post(f"{V1}/datasets/select", json={"dataset_id": "berka"})
        assert r.status_code == 422 and r.json()["detail"]["code"] == "dataset_unavailable"
        assert api.get(f"{V1}/datasets").json()["datasets"][0]["status"]["available"] in (True, False)
    finally:
        reg["berka"] = original
    assert api.get("/health").json()["dataset_id"] == "sandbox" and api.get(f"{V1}/console/overview").status_code == 200


def test_new_endpoints_are_in_the_openapi_schema_with_models(api):
    spec = api.get("/openapi.json").json()
    for path in ("/api/v1/datasets", "/api/v1/datasets/select", "/api/v1/datasets/{dataset_id}/accounts", "/api/v1/models",
                 "/api/v1/models/{model_id}/status", "/api/v1/models/select", "/api/v1/forecast", "/api/v1/compare",
                 "/api/v1/compare/{comparison_id}"):
        assert path in spec["paths"], path
    schemas = spec["components"]["schemas"]
    assert {"SelectDatasetRequest", "SelectModelRequest", "ForecastRequest", "ForecastOut", "AccountsOut"} <= set(schemas)
    assert schemas["ForecastRequest"]["properties"]["horizon"]["enum"] == [30, 60, 90]


def test_models_report_status_licence_and_load_state(api):
    select(api, "cards")
    body = api.get(f"{V1}/models").json()
    ms = {m["id"]: m for m in body["models"]}
    assert body["dataset_id"] == "cards" and set(ms) == {"stformer", "timesfm3"}
    assert ms["stformer"]["status"]["state"] in {"ready", "loaded"} and ms["stformer"]["version"].startswith("stformer:")
    assert ms["stformer"]["targets"] == ["rolling_30d_net_spend", "estimated_utilization"]
    assert ms["timesfm3"]["licence"]["badge"] == "TimesFM 3.0 · research licence · prototype only"
    assert api.get(f"{V1}/models", params={"dataset_id": "nope"}).status_code == 404
    sel = api.post(f"{V1}/models/select", json={"model_id": "stformer"})
    assert sel.status_code == 200 and sel.json()["model"]["status"]["state"] == "loaded" and sel.json()["model"]["status"]["loaded_at"]
    assert api.get(f"{V1}/models/stformer/status").json()["status"]["loaded"] is True


def test_accounts_list_filters_and_pages(api):
    a = accounts(api, "berka")
    assert a["total"] == DS["berka"][2] and a["accounts"][0]["account_id"].startswith("BK")
    one = api.get(f"{V1}/datasets/berka/accounts", params={"q": a["accounts"][2]["account_id"].lower()}).json()
    assert one["total"] == 1
    page = api.get(f"{V1}/datasets/berka/accounts", params={"limit": 3, "offset": 3}).json()
    assert page["returned"] == 3 and page["offset"] == 3 and page["accounts"][0]["account_id"] == a["accounts"][3]["account_id"]
    assert api.get(f"{V1}/datasets/berka/accounts", params={"product_type": "NOPE"}).json()["total"] == 0
    assert api.get(f"{V1}/datasets/nope/accounts").status_code == 404


def test_forecast_never_persists_runs_or_alerts(api):
    select(api, "sandbox")
    before = api.get(f"{V1}/runs", params={"limit": 500}).json()
    alerts_before = api.get(f"{V1}/alerts/active", params={"limit": 500}).json()["total"]
    r = forecast(api, "sandbox", sorted(a["account_id"] for a in accounts(api, "sandbox")["accounts"])[0], horizon=90)
    assert r.status_code == 200 and r.json()["persisted"] is False and r.json()["risk"]["display"]["pd"]["calibrated"] is False
    assert api.get(f"{V1}/runs", params={"limit": 500}).json() == before
    assert api.get(f"{V1}/alerts/active", params={"limit": 500}).json()["total"] == alerts_before


def test_item11_training_is_refused_for_a_model_that_cannot_train(api):
    select(api, "sandbox")
    r = rt(api)
    r.active_model = "timesfm3"
    try:
        out = api.post(f"{V1}/jobs/train", json={"epochs": 1})
        assert out.status_code == 422 and out.json()["detail"]["code"] == "training_not_supported"
    finally:
        r.active_model = "stformer"


def test_ist_timestamps_and_active_vs_all_run_alert_counts(api):
    select(api, "sandbox")
    api.post(f"{V1}/score/book", params={"wait": "true", "force": "true"}).raise_for_status()
    assert api.get(f"{V1}/console/overview").json()["kpis"]["high_risk"]["supported"] is True
    ov = api.get(f"{V1}/console/overview").json()
    assert ov["timezone"].startswith("IST") and ov["latest_book_run"]["scored_at"].endswith("+05:30")
    ac = ov["alert_counts"]
    assert ac["total_rows_all_runs"] == sum(v for k, v in ac.items() if k != "total_rows_all_runs") and "ews_alerts" in ov["tables_note"]
    k = ov["kpis"]
    assert k["total_accounts"] == DS["sandbox"][2] and k["active_alerts"] == ac["active"] and k["last_forecast_at"].endswith("+05:30")
    assert k["high_risk"]["supported"] is True and sum(d["count"] for d in ov["risk_distribution"]) == k["accounts_forecasted"]
    assert ov["model_status"]["id"] == "stformer"
    job = api.post(f"{V1}/jobs/refresh", params={"wait": "true"}).json()["job"]
    assert job["submitted_at"].endswith("+05:30") and job["finished_at"].endswith("+05:30")
    runs = api.get(f"{V1}/alerts/active", params={"limit": 5}).json()
    assert runs["generated_at"].endswith("+05:30")
    if runs["alerts"]:
        assert runs["alerts"][0]["created_at"].endswith("+05:30")


def test_alert_filters_and_display_taxonomy(api):
    select(api, "sandbox")
    api.post(f"{V1}/score/book", params={"wait": "true"}).raise_for_status()
    base = api.get(f"{V1}/alerts/active", params={"limit": 500}).json()
    assert base["total"] > 0
    first = base["alerts"][0]
    d = first["display"]
    assert d["category"] in {"predictive_warning", "spending_behaviour_change"} and d["category_label"] and d["priority"]["basis"]
    assert first["display"]["ead"].keys() == {"value", "na_reason"}
    q = api.get(f"{V1}/alerts/active", params={"q": first["account_id"][3:8], "limit": 500}).json()
    assert 0 < q["total"] <= base["total"] and all(first["account_id"][3:8].lower() in a["account_id"].lower() for a in q["alerts"])
    pred = api.get(f"{V1}/alerts/active", params={"category": "predictive_warning", "limit": 500}).json()
    assert all(a["alert_type"] == "CREDIT_EWS" for a in pred["alerts"])
    for cat in ("repayment_failure", "regulatory_classification"):  # no overdue data exists: none are invented
        assert api.get(f"{V1}/alerts/active", params={"category": cat}).json()["total"] == 0
    assert api.get(f"{V1}/alerts/active", params={"category": "bogus"}).status_code == 422
    pt = first["product_type"]
    assert pt and api.get(f"{V1}/alerts/active", params={"product_type": pt, "limit": 500}).json()["total"] > 0
    assert api.get(f"{V1}/alerts/active", params={"limit": 2, "offset": 1}).json()["alerts"][0]["rank"] == 2


def test_saved_comparisons_are_listed_read_and_path_safe(api, multi):
    root, reg, _ = multi
    select(api, "sandbox")
    d = compare_store.eval_dir(reg["sandbox"])
    d.mkdir(parents=True, exist_ok=True)
    doc = {"comparison_id": "cmp-1", "created_at": "2026-11-11T13:00:00", "scope": "account", "horizon": 30,
           "models": ["stformer", "timesfm3"], "origins": ["2026-09-01"],
           "aggregate": {"stformer": {"metrics": {"mae": 1.0, "wape": 0.1}}, "timesfm3": {"metrics": {"mae": 2.0, "wape": 0.2}}}}
    (d / "cmp-1.json").write_text(json.dumps(doc))
    (d / "broken.json").write_text("{not json")
    lst = api.get(f"{V1}/compare").json()
    assert [c["comparison_id"] for c in lst["comparisons"]] == ["cmp-1"] and lst["comparisons"][0]["created_at"].endswith("+05:30")
    assert lst["comparisons"][0]["headline"]["timesfm3"] == {"mae": 2.0, "wape": 0.2}
    assert api.get(f"{V1}/compare/cmp-1").json()["horizon"] == 30
    assert api.get(f"{V1}/compare/nope").status_code == 404
    assert api.get(f"{V1}/compare/..%2F..%2Fsecret").status_code == 404
    assert api.get(f"{V1}/compare", params={"dataset_id": "cards"}).json()["comparisons"] == []
    assert api.get(f"{V1}/console/overview").json()["recent_comparison"]["comparison_id"] == "cmp-1"


def test_item09_compare_runs_the_ml_script_with_identical_origins_for_every_model(api, multi, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    _, reg, _ = multi
    select(api, "sandbox")
    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        d = compare_store.eval_dir(reg["sandbox"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "cmp-run.json").write_text(json.dumps({"id": "cmp-run", "created_at": "2026-11-11T14:00:00+05:30"}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("api.routes.compare.subprocess.run", fake_run)
    acc = accounts(api, "sandbox")["accounts"][0]["account_id"]
    r = api.post(f"{V1}/compare", json={"dataset_id": "sandbox", "scope": "account", "account_id": acc, "horizon": 30})
    assert r.status_code == 202, r.text
    job = api.app.state.services.jobs.wait(r.json()["job"]["id"], 20)
    assert job.status == "done" and job.result["comparison_id"] == "cmp-run" and job.result["fair"] is False
    a = calls[0]
    origins = a[a.index("--origins") + 1].split(",")
    assert len(origins) >= 3 and a[a.index("--unit") + 1] == "INR" and "--timesfm" in a and a[a.index("--single-account") + 1] == acc
    assert all(o <= "2026-10-01" for o in origins)  # horizon 30: the actuals exist at every origin
    bad = api.post(f"{V1}/compare", json={"account_id": acc, "horizon": 30, "origins": ["2026-10-30"]})
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "invalid_origins"
    assert api.post(f"{V1}/compare", json={"dataset_id": "cards", "account_id": acc}).status_code == 409
    assert api.post(f"{V1}/compare", json={"models": ["nope"], "account_id": acc}).status_code == 422
