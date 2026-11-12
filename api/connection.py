"""Choosing, testing, switching and remembering the EWS database from the console."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from config.settings import CacheSettings, Settings, StorageSettings
from data_engine.storage import DuckDBStorage, PostgresStorage
from utils.logging import logger


class ConnectionSpec(BaseModel):
    backend: Literal["duckdb", "postgres"]
    duckdb_path: str | None = Field(None, description="Path to a DuckDB file (created if missing)")
    postgres_dsn: str | None = Field(
        None, description="SQLAlchemy URL, e.g. postgresql+psycopg2://user:password@host:5432/dbname"
    )

    @model_validator(mode="after")
    def _target(self) -> ConnectionSpec:
        if self.backend == "duckdb" and not self.duckdb_path:
            raise ValueError("duckdb_path is required for the duckdb backend")
        if self.backend == "postgres":
            if not self.postgres_dsn:
                raise ValueError("postgres_dsn is required for the postgres backend")
            if not self.postgres_dsn.startswith(("postgresql", "postgres")):
                raise ValueError("postgres_dsn must be a postgresql:// or postgresql+psycopg2:// URL")
        return self


def mask_dsn(dsn: str | None) -> str | None:
    if not dsn:
        return None
    try:
        from sqlalchemy.engine import make_url

        return make_url(dsn).render_as_string(hide_password=True)
    except Exception:
        return "<invalid DSN>"


def describe(settings: Settings) -> dict[str, Any]:
    st = settings.storage
    return {
        "backend": st.backend,
        "duckdb_path": str(st.duckdb_path) if st.backend == "duckdb" else None,
        "postgres_dsn": mask_dsn(st.postgres_dsn) if st.backend == "postgres" else None,
        "cache_root": str(settings.cache.root),
    }


def settings_for(base: Settings, spec: ConnectionSpec) -> Settings:
    """A copy of ``base`` pointed at another database, with its own tensor cache."""
    if spec.backend == "duckdb":
        path = Path(spec.duckdb_path).expanduser()
        if not path.is_absolute():
            path = (base.data_dir / path).resolve()
        storage = base.storage.model_copy(update={"backend": "duckdb", "duckdb_path": path})
        key = f"duckdb:{path}"
    else:
        storage = base.storage.model_copy(update={"backend": "postgres", "postgres_dsn": spec.postgres_dsn})
        key = f"postgres:{mask_dsn(spec.postgres_dsn)}"
    default_db = StorageSettings().duckdb_path.resolve()
    if spec.backend == "duckdb" and Path(storage.duckdb_path).resolve() == default_db:
        cache = CacheSettings()  # the default database keeps the default cache location
    else:
        slug = hashlib.sha1(key.encode()).hexdigest()[:10]
        cache = CacheSettings(root=base.data_dir / "connections" / slug / "cache")
    return base.model_copy(update={"storage": storage, "cache": cache})


def test_connection(settings: Settings) -> dict[str, Any]:
    """Open the database, ensure the EWS schema exists, and report what is already there."""
    adapter = DuckDBStorage(settings) if settings.storage.backend == "duckdb" else PostgresStorage(settings)
    try:
        adapter.init_schema()
        adapter.migrate()
        counts = {
            t: int(adapter.scalar(f"SELECT COUNT(*) FROM {t}") or 0)
            for t in ("accounts", "daily_balances", "transactions", "credit_profiles", "loan_schedules")
        }
    finally:
        adapter.close()
    return {"ok": True, "tables": counts}


def _saved_path(settings: Settings) -> Path:
    return settings.data_dir / "connection.json"


def save(settings: Settings, spec: ConnectionSpec) -> None:
    path = _saved_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.model_dump(), indent=2))
    try:
        os.chmod(path, 0o600)  # may hold a database password
    except OSError:
        pass


def load_saved(settings: Settings) -> Settings:
    """Apply the connection chosen in the console, unless env vars pick the storage."""
    if not settings.use_saved_connection or any(k.startswith("EWS_STORAGE__") for k in os.environ):
        return settings
    path = _saved_path(settings)
    if not path.exists():
        return settings
    try:
        spec = ConnectionSpec(**json.loads(path.read_text()))
    except Exception as exc:
        logger.warning("ignoring unreadable {}: {}", path, exc)
        return settings
    logger.info("using the database saved in the console ({})", spec.backend)
    return settings_for(settings, spec)
