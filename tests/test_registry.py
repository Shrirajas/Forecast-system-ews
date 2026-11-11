"""Dataset / model registry: static facts, dynamic facts from the cache, per-dataset settings, IST helper."""

from __future__ import annotations

from datetime import datetime

from config.registry import (MODELS, REGISTRY, TIMESFM_BADGE, build_models, build_registry, dataset_info, dataset_status,
                             match_dataset, resolve_checkpoint, settings_for_dataset)
from config.settings import Settings
from utils.timefmt import to_ist


def test_item01_registry_describes_the_three_datasets():
    assert set(REGISTRY) == {"sandbox", "cards", "berka"}
    cards, berka, sandbox = REGISTRY["cards"], REGISTRY["berka"], REGISTRY["sandbox"]
    assert (cards.currency.code, berka.currency.code, sandbox.currency.code) == ("USD", "CZK", "INR")
    t = cards.primary_target
    assert t.label == "Rolling 30-day net spend" and t.channel == "eod_balance" and "not an account balance" in t.description
    assert [x.id for x in cards.targets] == ["rolling_30d_net_spend", "estimated_utilization"]
    assert cards.capabilities["bureau_name"] == "FICO" and cards.capabilities["emi"] is False
    assert cards.capabilities["projected_sma"] is False
    for spec in REGISTRY.values():  # every dataset names its own paths: none share a database or a cache
        assert spec.duckdb_path.name == "ews.duckdb" and set(spec.supported_models) == {"stformer", "timesfm3"}
    assert len({s.duckdb_path for s in REGISTRY.values()}) == 3 == len({s.cache_root for s in REGISTRY.values()})


def test_item02_berka_only_declares_what_the_data_supports():
    b = REGISTRY["berka"]
    util = next(t for t in b.targets if t.id == "utilization")
    assert util.available is False and "no credit limits" in util.reason
    feats = {f.id: f for f in b.features}
    assert feats["bureau_score"].available is False and feats["credit_limit"].available is False
    assert feats["emi"].available and feats["loan_outcomes"].available
    assert b.risk_overrides["bureau_enabled"] is False


def test_models_declare_training_and_licence():
    st, tf = MODELS["stformer"], MODELS["timesfm3"]
    assert st.supports_training and st.kind == "trained" and st.badge is None
    assert not tf.supports_training and tf.kind == "pretrained_zero_shot" and tf.badge == TIMESFM_BADGE
    assert "research" in tf.licence_restriction and tf.checkpoint == "google/timesfm-3.0-pytorch"
    assert set(build_models(REGISTRY)) == {"stformer", "timesfm3"}


def test_dataset_settings_are_derived_from_the_base_and_never_accumulate(tmp_path):
    base = Settings()
    berka = settings_for_dataset(base, REGISTRY["berka"])
    cards = settings_for_dataset(base, REGISTRY["cards"])
    assert berka.risk.bureau_enabled is False and cards.risk.bureau_enabled is True  # no leak from berka to cards
    assert cards.behavior.attrition_min_balance == 500.0 and base.behavior.attrition_min_balance != 500.0
    assert berka.storage.duckdb_path == REGISTRY["berka"].duckdb_path and berka.cache.root == REGISTRY["berka"].cache_root
    assert base.risk.bureau_enabled is True  # the base object is untouched


def test_dynamic_facts_come_from_the_cache_without_opening_duckdb(tmp_path):
    reg = build_registry(tmp_path)  # nothing exists under tmp_path
    st = dataset_status(reg["cards"])
    assert st["available"] is False and st["state"] == "missing_database" and st["reason"]
    info = dataset_info(reg["cards"], None)
    assert info["data"] is None and info["product_types"] == {} and info["status"]["checkpoint_exists"] is False
    assert all(str(s.duckdb_path).startswith(str(tmp_path)) for s in reg.values())
    real = dataset_info(REGISTRY["cards"], "cards")  # real cache metadata (read-only json/parquet, no DuckDB)
    if real["status"]["cache_exists"]:
        assert real["active"] and real["data"]["accounts"] == sum(real["product_types"].values())


def test_match_dataset_and_checkpoint_resolution(tmp_path):
    s = Settings()
    assert match_dataset(s.model_copy(update={"storage": s.storage.model_copy(update={"duckdb_path": REGISTRY["cards"].duckdb_path})})) == "cards"
    assert match_dataset(s.model_copy(update={"storage": s.storage.model_copy(update={"duckdb_path": tmp_path / "x.duckdb"})})) is None
    reg = build_registry(tmp_path)
    spec = reg["berka"]
    spec.checkpoint_path.parent.mkdir(parents=True)
    assert resolve_checkpoint(spec) == spec.checkpoint_path  # no pointer: the registered default
    newer = spec.checkpoint_path.with_name("berka_stformer_20260101T000000.pt")
    newer.write_bytes(b"x")
    spec.checkpoint_path.with_name("berka_stformer.latest.json").write_text('{"checkpoint": "%s"}' % newer)
    assert resolve_checkpoint(spec) == newer


def test_every_timestamp_is_ist_with_an_offset():
    assert to_ist(datetime(2026, 11, 11, 5, 42, 8)) == "2026-11-11T11:12:08+05:30"  # naive UTC (DB)
    assert to_ist("2026-11-11T11:11:37", naive="ist") == "2026-11-11T11:11:37+05:30"  # naive local (training, jobs)
    assert to_ist("2026-11-11T05:42:08+00:00") == "2026-11-11T11:12:08+05:30" and to_ist(None) is None
