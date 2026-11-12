"""TASK.md section 9 acceptance tests (QA), written against .team/ews/API_CONTRACT.md.

Three tiny *synthetic* datasets live in a tmp dir (never data/*.duckdb): same generator, distinct id prefixes, end dates and sizes,
so a leak between datasets shows up as a wrong id prefix or a wrong history date. Data-dependent behaviour of the real
Cards/Berka files (FICO, loan outcomes, ...) is checked by the live smoke in docs/UPGRADE_REPORT.md, not here.
Test names carry ``test_itemNN_`` = section 9 item NN (the QA ledger counts them).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from conftest import LOAD_ORDER, make_settings
from fastapi.testclient import TestClient

from api.main import create_app
from config.registry import build_registry, settings_for_dataset
from data_engine.ingestion import IngestionEngine
from data_engine.storage import get_storage, release_storage
from data_engine.synth import SandboxConfig, generate_sandbox

# dataset -> (account prefix, customer prefix, accounts, days, last day, seed)
DS = {
    "sandbox": ("AC", "CU", 10, 600, date(2026, 10, 31), 3),
    "cards": ("CC", "UC", 12, 500, date(2026, 9, 30), 4),
    "berka": ("BK", "UB", 9, 550, date(2026, 8, 31), 5),
}


@pytest.fixture(scope="module")
def multi(tmp_path_factory):
    from models.engine import reset_engine
    from models.pipeline import EWSPipeline
    from models.train import train_model

    root = tmp_path_factory.mktemp("multi")
    reg, base = build_registry(root), make_settings(root)
    opened = []  # only these databases are released: reset_storage_cache() would also close the session `env` fixture's
    for ds, (ap, cp, n, days, end, seed) in DS.items():
        raw = root / "raw" / ds
        raw.mkdir(parents=True)
        generate_sandbox(SandboxConfig(n_accounts=n, n_days=days, end_date=end, seed=seed), raw)
        for t in LOAD_ORDER:
            tbl = pq.read_table(raw / f"{t}.parquet")
            for col, old, new in (("account_id", "AC", ap), ("customer_id", "CU", cp)):
                if col in tbl.column_names:
                    tbl = tbl.set_column(tbl.column_names.index(col), col, pc.replace_substring_regex(tbl[col], f"^{old}", new))
            pq.write_table(tbl, raw / f"{t}.parquet")
        spec = reg[ds]
        for p in (spec.duckdb_path, spec.checkpoint_path):
            p.parent.mkdir(parents=True, exist_ok=True)
        s = settings_for_dataset(base, spec)
        opened.append(s)
        storage = get_storage(s)
        eng = IngestionEngine(storage)
        for t in LOAD_ORDER:
            eng.ingest_file(t, raw / f"{t}.parquet", "parquet")
        pipe = EWSPipeline(s, storage=storage)
        pipe.refresh(full=True, force_cache=True)
        train_model(pipe.cache.current_dir(), s.model, max_steps_per_epoch=3, quiet=True)
    for s in opened:
        release_storage(s)
    reset_engine()
    base = base.model_copy(update={"storage": base.storage.model_copy(update={"duckdb_path": reg["sandbox"].duckdb_path})})
    yield root, reg, base
    reset_engine()
    for s in opened:
        release_storage(s)


@pytest.fixture(scope="module")
def api(multi):
    _, reg, base = multi
    with TestClient(create_app(base, registry=reg)) as c:
        yield c


V1 = "/api/v1"


def select(api, ds):
    r = api.post(f"{V1}/datasets/select", json={"dataset_id": ds})
    assert r.status_code == 200, r.text
    return r.json()


def accounts(api, ds):
    r = api.get(f"{V1}/datasets/{ds}/accounts", params={"limit": 200})
    assert r.status_code == 200, r.text
    return r.json()


def forecast(api, ds, account_id, **kw):
    return api.post(f"{V1}/forecast", json={"dataset_id": ds, "account_id": account_id, "horizon": 30, **kw})


def assert_ordered(fc):
    q10, q50, q90 = (np.asarray(fc["quantiles"][k], float) for k in ("q10", "q50", "q90"))
    assert np.isfinite([q10, q50, q90]).all() and (q10 <= q50).all() and (q50 <= q90).all()


# ------------------------------------------------------------------------------ items 1-3: selecting
@pytest.mark.parametrize("item,ds", [("01", "cards"), ("02", "berka"), ("03", "sandbox")])
def test_item01_each_dataset_can_be_selected(api, item, ds):
    out = select(api, ds)
    assert out["active"] == ds and out["dataset"]["id"] == ds and out["dataset"]["status"]["available"] is True
    assert out["dataset"]["data"]["accounts"] == DS[ds][2]
    assert api.get("/health").json()["dataset_id"] == ds
    assert api.get(f"{V1}/datasets").json()["active"] == ds


def test_item01_cards_target_is_30day_spend_in_usd_not_balance(api):
    d = select(api, "cards")["dataset"]
    t = next(x for x in d["targets"] if x["id"] == d["primary_target"])
    assert t["label"] == "Rolling 30-day net spend" and t["unit"] == "USD" and d["currency"]["code"] == "USD"
    assert "balance" not in t["label"].lower()


def test_item02_berka_is_czk_and_has_no_utilisation_or_bureau(api):
    d = select(api, "berka")["dataset"]
    assert d["currency"]["code"] == "CZK" and d["capabilities"]["bureau"] is False and d["capabilities"]["utilization"] is False
    r = forecast(api, "berka", accounts(api, "berka")["accounts"][0]["account_id"], target="estimated_utilization")
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unsupported_target"


def test_item03_sandbox_keeps_balance_emi_and_utilisation_workflow(api):
    select(api, "sandbox")
    acc = next(a for a in accounts(api, "sandbox")["accounts"] if a["has_credit_line"])
    r = forecast(api, "sandbox", acc["account_id"])
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["target"]["id"] == "eod_balance" and j["secondary"] is not None and j["risk"] is not None
    assert api.post(f"{V1}/predict/ews", json={"horizon": 30}).status_code == 200  # existing API unchanged


# ------------------------------------------------------------------------------ item 4: no mixing
def test_item04_switching_never_mixes_accounts_or_histories(api):
    ids = {}
    for ds in DS:
        select(api, ds)
        a = accounts(api, ds)
        ids[ds] = {x["account_id"] for x in a["accounts"]}
        assert a["total"] == DS[ds][2] and all(i.startswith(DS[ds][0]) for i in ids[ds])
        f = forecast(api, ds, sorted(ids[ds])[0]).json()
        assert f["dataset"]["id"] == ds and f["origin"] == DS[ds][4].isoformat()  # its own last day, not another dataset's
        assert f["history"]["dates"][-1] == DS[ds][4].isoformat()
    assert not (ids["cards"] & ids["berka"] or ids["cards"] & ids["sandbox"] or ids["berka"] & ids["sandbox"])
    select(api, "cards")
    other = sorted(ids["berka"])[0]
    r = forecast(api, "cards", other)  # a Berka id must not resolve while Cards is active
    assert r.status_code == 404 and r.json()["detail"]["code"] == "unknown_account"
    assert accounts(api, "berka")["accounts"][0]["account_id"].startswith("BK")  # per-dataset listing, whatever is active


# ------------------------------------------------------------------------------ item 5: STFormer
def test_item05_stformer_forecast_is_real_and_ordered(api):
    select(api, "cards")
    acc = sorted(a["account_id"] for a in accounts(api, "cards")["accounts"])[0]
    r = forecast(api, "cards", acc, model_id="stformer", horizon=60)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"]["id"] == "stformer" and j["horizon"] == 60 and len(j["forecast"]["dates"]) == 60 and j["persisted"] is False
    assert j["model"]["version"] and len(j["history"]["dates"]) > 30
    assert_ordered(j["forecast"])
    assert j["forecast"]["dates"][0] > j["origin"]  # forecast starts after the origin
    assert j["description"] and j["timing"]["timezone"] == "IST"


# ------------------------------------------------------------------------------ item 7: TimesFM unavailable
def test_item07_timesfm_unavailable_is_reported_not_faked(api):
    select(api, "cards")
    m = api.get(f"{V1}/models/timesfm3/status").json()
    assert m["status"]["loaded"] is False and m["status"]["state"] in {"licence_not_enabled", "weights_not_downloaded", "package_missing"}
    assert m["status"]["message"] and m["licence"]["badge"] == "TimesFM 3.0 · research licence · prototype only"
    r = api.post(f"{V1}/models/select", json={"model_id": "timesfm3"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "model_unavailable"
    assert api.get(f"{V1}/models").json()["active"] == "stformer"  # failed load leaves the selection alone
    acc = accounts(api, "cards")["accounts"][0]["account_id"]
    r = forecast(api, "cards", acc, model_id="timesfm3")
    assert r.status_code == 503 and r.json()["detail"]["code"] == "model_unavailable"  # no silent STFormer fallback
    assert forecast(api, "cards", acc).status_code == 200  # STFormer still healthy afterwards


# ------------------------------------------------------------------------------ item 8: rejections
def test_item08_unsupported_requests_are_rejected_and_state_is_kept(api):
    select(api, "cards")
    acc = accounts(api, "cards")["accounts"][0]["account_id"]
    for url, body, status, code in [
        (f"{V1}/datasets/select", {"dataset_id": "nope"}, 404, "unknown_dataset"),
        (f"{V1}/models/select", {"model_id": "nope"}, 404, "unknown_model"),
        (f"{V1}/forecast", {"account_id": acc, "horizon": 30, "model_id": "nope"}, 404, "unknown_model"),
        (f"{V1}/forecast", {"account_id": acc, "horizon": 30, "target": "nope"}, 422, "unsupported_target"),
        (f"{V1}/forecast", {"account_id": acc, "horizon": 30, "dataset_id": "berka"}, 409, "dataset_not_active"),
    ]:
        r = api.post(url, json=body)
        assert r.status_code == status and r.json()["detail"]["code"] == code, (body, r.text)
    assert api.post(f"{V1}/forecast", json={"account_id": acc, "horizon": 45}).status_code == 422  # only 30/60/90
    assert api.get(f"{V1}/datasets").json()["active"] == "cards" and api.get(f"{V1}/models").json()["active"] == "stformer"


# ------------------------------------------------------------------------------ item 10: units and labels
@pytest.mark.parametrize("ds,unit,label_has", [("cards", "USD", "30-day"), ("berka", "CZK", "balance"), ("sandbox", "INR", "balance")])
def test_item10_forecast_target_label_unit_and_description(api, ds, unit, label_has):
    select(api, ds)
    j = forecast(api, ds, sorted(a["account_id"] for a in accounts(api, ds)["accounts"])[0]).json()
    assert j["target"]["unit"] == unit and label_has in j["target"]["label"].lower() and j["dataset"]["currency"]["code"] == unit
    assert j["description"] and (j["target"]["label"] in j["description"] or unit in j["description"])


# ------------------------------------------------------------------------------ item 11: training capability
def test_item11_only_trainable_models_advertise_training(api):
    select(api, "cards")
    ms = {m["id"]: m for m in api.get(f"{V1}/models").json()["models"]}
    assert ms["stformer"]["supports_training"] is True and ms["stformer"]["training"] is not None and ms["stformer"]["zero_shot"] is None
    assert ms["timesfm3"]["supports_training"] is False and ms["timesfm3"]["training"] is None and ms["timesfm3"]["zero_shot"] is not None


# ------------------------------------------------------------------------------ item 14: nothing fabricated
def test_item14_bureau_is_null_where_the_dataset_has_none(api):
    def bureau(ds):
        select(api, ds)
        acc = next(a for a in accounts(api, ds)["accounts"] if a["has_credit_line"])
        pd_ = forecast(api, ds, acc["account_id"]).json()["risk"]["display"]["pd"]
        return next(c for c in pd_["components"] if c["id"] == "bureau"), pd_

    comp, pd_ = bureau("berka")  # registry says bureau disabled: no invented 700 score, even though the table has one
    assert comp["value"] is None and comp["contribution"] is None and pd_["bureau_name"] is None
    comp, pd_ = bureau("cards")
    assert comp["value"] is not None and pd_["bureau_name"] == "FICO" and pd_["calibrated"] is False
    assert pd_["basis"]  # the label itself is asserted in BACKEND's test_risk_display on real cards-like inputs


# ------------------------------------------------------------------------------ item 4/8: safe switching
def _open_files() -> set[str]:
    import os

    out = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            out.add(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            pass
    return out


def test_item04_switch_is_refused_while_a_job_runs_and_state_is_kept(api):
    import threading

    select(api, "sandbox")
    hold, jobs = threading.Event(), api.app.state.services.jobs
    job = jobs.submit("qa-hold", "QA: keeps the job queue busy", hold.wait)
    try:
        r = api.post(f"{V1}/datasets/select", json={"dataset_id": "cards"})
        assert r.status_code == 409 and r.json()["detail"]["code"] == "busy"
        assert api.get(f"{V1}/datasets").json()["active"] == "sandbox"
        assert accounts(api, "sandbox")["accounts"][0]["account_id"].startswith("AC")  # still serving the old dataset
    finally:
        hold.set()
        jobs.wait(job.id, 10)
    assert select(api, "cards")["active"] == "cards"  # and the switch works once the job is done


def test_item04_failed_switch_keeps_old_dataset_and_connection(api, multi):
    select(api, "sandbox")
    rt = api.app.state.runtime
    good = rt.registry["cards"]
    rt.registry["cards"] = good.model_copy(update={"cache_root": multi[0] / "no-such-cache"})
    try:
        r = api.post(f"{V1}/datasets/select", json={"dataset_id": "cards"})
        assert r.status_code == 422 and r.json()["detail"]["code"] == "dataset_unavailable"
    finally:
        rt.registry["cards"] = good
    assert api.get(f"{V1}/datasets").json()["active"] == "sandbox"
    assert forecast(api, "sandbox", accounts(api, "sandbox")["accounts"][0]["account_id"]).status_code == 200


@pytest.mark.skipif(not __import__("os").path.isdir("/proc/self/fd"), reason="needs /proc (Linux)")
def test_item04_old_duckdb_file_is_closed_after_a_switch(api, multi):
    import time

    reg = multi[1]
    old, new = str(reg["sandbox"].duckdb_path), str(reg["cards"].duckdb_path)
    select(api, "sandbox")
    assert old in _open_files()  # positive control: the check can see an open database
    select(api, "cards")
    time.sleep(2.6)  # CLOSE_GRACE_S = 2.0
    files = _open_files()
    assert old not in files, "old DuckDB file still open after the grace period"
    assert new in files, "new DuckDB file not open"


def test_item04_fast_switch_back_does_not_close_the_active_database(api, multi):
    """A -> B -> A inside the 2 s close grace: the first switch's pending close must not kill the dataset that is active again."""
    import time

    select(api, "cards")
    select(api, "sandbox")  # schedules a close of the Cards database ...
    select(api, "cards")  # ... which is active again before it fires
    time.sleep(2.6)
    still_open = str(multi[1]["cards"].duckdb_path) in _open_files()
    acc = accounts(api, "cards")["accounts"][0]["account_id"]
    r = forecast(api, "cards", acc)
    reads = {u: api.get(f"{V1}{u}").status_code for u in ("/alerts/active", "/console/overview")}
    reads["predict"] = api.post(f"{V1}/predict/ews", json={"horizon": 30}).status_code
    assert still_open and r.status_code == 200 and set(reads.values()) == {200}, (
        f"stale close timer: db open={still_open}, forecast={r.status_code}, db-backed reads={reads}")
