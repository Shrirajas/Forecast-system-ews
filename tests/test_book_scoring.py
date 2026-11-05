"""Whole-book scoring and stored-run bookkeeping (review fixes for PR #1)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from config.settings import Settings, StorageSettings
from data_engine.alerts import utcnow
from data_engine.storage import DuckDBStorage


def _active(pipe) -> int:
    return pipe.alerts.active(limit=10_000)[1]


def test_book_rescore_keeps_alerts_from_the_alert_horizon_only(env):
    _, pipe = env
    out = pipe.score_book(force=True)
    assert out["alert_horizon"] == 90
    runs = {r["horizon"]: r for r in pipe.alerts.recent_runs(10) if r["scope"] == "BOOK"}
    assert {30, 60, 90} <= set(runs)
    assert runs[30]["n_alerts"] == runs[60]["n_alerts"] == 0
    # every active alert comes from the 90-day run, and all of its alerts are active
    active, total = pipe.alerts.active(limit=10_000)
    assert total == runs[90]["n_alerts"] == out["alerts"]
    assert {a["run_id"] for a in active} <= {out["runs"][90]}
    # a second rescore replaces, not accumulates
    pipe.score_book(force=True)
    assert _active(pipe) == total
    # each horizon stores one score row per account
    for h in (30, 60, 90):
        assert pipe.alerts.scores(pipe.alerts.latest_book_run(
            pipe.cache.load().meta.version, h, 0.1, 0.9, pipe.engine.model_version)["run_id"]).num_rows == pipe.cache.load().n_accounts


def test_stored_run_lookup_matches_model_version_and_rounded_quantiles(env):
    _, pipe = env
    pipe.score_book(force=True)
    v, mv = pipe.cache.load().meta.version, pipe.engine.model_version
    assert pipe.alerts.latest_book_run(v, 90, 0.1, 0.9, mv)
    assert pipe.alerts.latest_book_run(v, 90, 0.1 + 1e-12, 0.30000000000000004 * 3, mv)  # float noise
    assert pipe.alerts.latest_book_run(v, 90, 0.1, 0.9, "stformer:retrained") is None  # new weights
    assert pipe.alerts.latest_book_run("v-other-data", 90, 0.1, 0.9, mv) is None  # new data
    assert pipe.score_book()["skipped"] is True  # nothing changed: no rescore


def test_old_book_runs_are_pruned(env):
    settings, pipe = env
    for _ in range(settings.keep_book_runs + 2):
        pipe.score_book(force=True)
    kept = pipe.storage.scalar(
        "SELECT COUNT(DISTINCT s.run_id) FROM account_scores s JOIN forecast_runs r USING (run_id) "
        "WHERE r.scope = 'BOOK' AND r.horizon = 90"
    )
    assert kept == settings.keep_book_runs


def test_orphans_of_crashed_runs_are_cleaned_up(env):
    _, pipe = env
    cache = pipe.cache.load()
    import numpy as np

    rows = np.arange(3)
    sb = pipe.score_rows(cache, rows, cache.end, 90, 0.1, 0.9)
    pipe.alerts.write_shard("run_crashed", cache.accounts["account_id"].iloc[rows].tolist(),
                            pipe.alert_table(cache, sb, "run_crashed"), pipe.score_table(cache, sb, "run_crashed"))
    assert pipe.alerts.cleanup_orphans()["account_scores"] == 0  # too recent: may still be running
    old = utcnow() - timedelta(hours=2)
    for t in ("account_scores", "run_scope", "ews_alerts"):
        pipe.storage.execute(f"UPDATE {t} SET created_at = :c WHERE run_id = 'run_crashed'", {"c": old})
    removed = pipe.alerts.cleanup_orphans()
    assert removed["account_scores"] == 3 and removed["run_scope"] == 3
    for t in ("account_scores", "run_scope", "ews_alerts"):
        assert pipe.storage.scalar(f"SELECT COUNT(*) FROM {t} WHERE run_id = 'run_crashed'") == 0


def test_migrate_refuses_missing_not_null_columns(tmp_path):
    st = DuckDBStorage(Settings(storage=StorageSettings(duckdb_path=tmp_path / "m.duckdb")))
    st.execute("CREATE TABLE forecast_runs (run_id VARCHAR PRIMARY KEY)")  # an old, partial table
    with pytest.raises(RuntimeError, match="NOT NULL"):
        st.migrate()
