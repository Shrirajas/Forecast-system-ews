"""Storage adapters: DuckDB (embedded, columnar) and PostgreSQL.

Both adapters speak Arrow on the way in and out, so ingestion, aggregation
and the tensor cache stream record batches without building Python objects
per row. SQL is written once with ``:name`` parameters; the DuckDB adapter
rewrites them to ``$name``.
"""

from __future__ import annotations

import io
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from config.settings import Settings, get_settings
from data_engine.schema import ALL_TABLES, Table, conform
from utils.logging import logger

WriteMode = Literal["upsert", "ignore", "append"]
ArrowLike = pa.Table | pa.RecordBatch | pa.RecordBatchReader | Iterable[pa.RecordBatch]

_PARAM_RE = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")


def _iter_tables(data: ArrowLike) -> Iterator[pa.Table]:
    if isinstance(data, pa.Table):
        yield data
    elif isinstance(data, pa.RecordBatch):
        yield pa.Table.from_batches([data])
    else:
        for batch in data:
            yield batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])


def _duck_relation(cur, sql: str, params: dict[str, Any] | None):
    # Relations stream straight to Arrow (DB-API results make an extra copy on DuckDB >= 2).
    return cur.sql(sql, params=params) if params else cur.sql(sql)


def _duck_table(rel) -> pa.Table:
    fn = getattr(rel, "to_arrow_table", None) or rel.arrow  # DuckDB 2.x / 1.x
    return fn()


def _duck_reader(rel, batch_rows: int) -> pa.RecordBatchReader:
    fn = getattr(rel, "to_arrow_reader", None) or rel.fetch_arrow_reader
    return fn(batch_rows)


def _with_row_number(tbl: pa.Table) -> pa.Table:
    return tbl.append_column("__rn", pa.array(np.arange(tbl.num_rows, dtype=np.int64)))


class StorageAdapter(ABC):
    dialect: str

    @abstractmethod
    def init_schema(self) -> None: ...

    @abstractmethod
    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def execute_many(self, statements: list[tuple[str, dict[str, Any] | None]]) -> None:
        """Run several statements atomically (single transaction)."""

    @abstractmethod
    def query_arrow(self, sql: str, params: dict[str, Any] | None = None) -> pa.Table: ...

    @abstractmethod
    def stream_arrow(
        self, sql: str, params: dict[str, Any] | None = None, batch_rows: int | None = None
    ) -> Iterator[pa.RecordBatch]: ...

    @abstractmethod
    def write_arrow(self, table: str, data: ArrowLike, mode: WriteMode = "upsert") -> int:
        """Write Arrow data into ``table``; returns rows written."""

    @abstractmethod
    def close(self) -> None: ...

    # ---- conveniences shared by both adapters ---------------------------------
    def query_df(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        return self.query_arrow(sql, params).to_pandas()

    def scalar(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        tbl = self.query_arrow(sql, params)
        if tbl.num_rows == 0 or tbl.num_columns == 0:
            return None
        return tbl.column(0)[0].as_py()

    def migrate(self) -> list[str]:
        """Add columns that newer releases declare but an existing database lacks.

        Only nullable columns can be added safely to a live table; a missing
        NOT NULL column means the database needs a manual migration, so this
        raises instead of letting later inserts fail obscurely. (Secondary
        indexes are created by ``init_schema`` on every start, so existing
        PostgreSQL databases pick up new ones; DuckDB uses none by design.)"""
        added, blocked = [], []
        for t in ALL_TABLES.values():
            have = {
                r["column_name"].lower()
                for r in self.query_arrow(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = :t "
                    "AND table_schema = current_schema()", {"t": t.name}
                ).to_pylist()
            }
            if not have:
                continue  # table just created by init_schema
            for name, _, nullable in t.columns:
                if name in have:
                    continue
                if not nullable:
                    blocked.append(f"{t.name}.{name}")
                    continue
                self.execute(f"ALTER TABLE {t.name} ADD COLUMN {t.column_sql(name, self.dialect)}")
                added.append(f"{t.name}.{name}")
        if blocked:
            raise RuntimeError(
                "database schema is missing NOT NULL column(s) that cannot be added automatically: "
                + ", ".join(blocked) + ". Migrate these tables manually (add with a default, then backfill)."
            )
        if added:
            logger.info("schema migrated: added {}", ", ".join(added))
        return added

    def write_rows(self, table: str, rows: list[dict[str, Any]], mode: WriteMode = "upsert") -> int:
        if not rows:
            return 0
        t = ALL_TABLES[table]
        return self.write_arrow(table, conform(t, pa.Table.from_pylist(rows)), mode)

    def table_counts(self) -> dict[str, int]:
        return {name: int(self.scalar(f"SELECT COUNT(*) FROM {name}") or 0) for name in ALL_TABLES}


class DuckDBStorage(StorageAdapter):
    """Embedded columnar engine with vectorised, multi-threaded execution.

    One process-wide connection; every thread gets its own cursor (DuckDB's
    supported concurrency model). Writes are serialised with a lock.
    """

    dialect = "duckdb"

    def __init__(self, settings: Settings):
        import duckdb

        cfg = settings.storage
        cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        config: dict[str, Any] = {"memory_limit": cfg.duckdb_memory_limit}
        if cfg.duckdb_threads:
            config["threads"] = cfg.duckdb_threads
        self._con = duckdb.connect(str(cfg.duckdb_path), config=config)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self.batch_rows = cfg.stream_batch_rows
        self.path = cfg.duckdb_path

    def _cursor(self):
        cur = getattr(self._local, "cur", None)
        if cur is None:
            cur = self._con.cursor()
            self._local.cur = cur
        return cur

    @staticmethod
    def _sql(sql: str, params: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
        if not params:
            return sql, None
        used = set(_PARAM_RE.findall(sql))
        return _PARAM_RE.sub(r"$\1", sql), {k: v for k, v in params.items() if k in used}

    def init_schema(self) -> None:
        with self._write_lock:
            cur = self._cursor()
            for t in ALL_TABLES.values():
                # Only the PK DDL: DuckDB prunes scans with zone maps, and extra
                # ART indexes slow bulk upserts without helping analytics.
                cur.execute(t.ddl("duckdb")[0])

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        q, p = self._sql(sql, params)
        with self._write_lock:
            self._cursor().execute(q, p)

    def execute_many(self, statements: list[tuple[str, dict[str, Any] | None]]) -> None:
        with self._write_lock:
            cur = self._cursor()
            cur.execute("BEGIN TRANSACTION")
            try:
                for sql, params in statements:
                    q, p = self._sql(sql, params)
                    cur.execute(q, p)
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    def query_arrow(self, sql: str, params: dict[str, Any] | None = None) -> pa.Table:
        q, p = self._sql(sql, params)
        return _duck_table(_duck_relation(self._cursor(), q, p))

    def stream_arrow(
        self, sql: str, params: dict[str, Any] | None = None, batch_rows: int | None = None
    ) -> Iterator[pa.RecordBatch]:
        q, p = self._sql(sql, params)
        # A dedicated cursor so a long-lived stream never collides with other
        # statements issued from the same thread.
        cur = self._con.cursor()
        try:
            reader = _duck_reader(_duck_relation(cur, q, p), batch_rows or self.batch_rows)
            yield from reader
        finally:
            cur.close()

    def write_arrow(self, table: str, data: ArrowLike, mode: WriteMode = "upsert") -> int:
        t: Table = ALL_TABLES[table]
        cols = ", ".join(t.column_names)
        pk = ", ".join(t.primary_key)
        verb = {"upsert": "INSERT OR REPLACE", "ignore": "INSERT OR IGNORE", "append": "INSERT"}[mode]
        n = 0
        with self._write_lock:
            cur = self._cursor()
            cur.execute("BEGIN TRANSACTION")
            try:
                for chunk in _iter_tables(data):
                    if chunk.num_rows == 0:
                        continue
                    staged = _with_row_number(conform(t, chunk))
                    cur.register("__ingest_batch", staged)
                    # Keep the last occurrence of each key inside the batch so a
                    # single statement never touches the same row twice.
                    cur.execute(
                        f"{verb} INTO {table} ({cols}) SELECT {cols} FROM __ingest_batch "
                        f"QUALIFY row_number() OVER (PARTITION BY {pk} ORDER BY __rn DESC) = 1"
                    )
                    cur.unregister("__ingest_batch")
                    n += chunk.num_rows
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
        return n

    def close(self) -> None:
        self._con.close()


class PostgresStorage(StorageAdapter):
    """PostgreSQL via SQLAlchemy + psycopg2.

    Bulk writes stream each Arrow batch as CSV through ``COPY ... FROM STDIN``
    into a temp staging table, then merge with ``INSERT ... ON CONFLICT``.
    Reads use server-side cursors so large scans never land in memory at once.
    """

    dialect = "postgres"

    def __init__(self, settings: Settings):
        from sqlalchemy import create_engine

        self.engine = create_engine(
            settings.storage.postgres_dsn, pool_pre_ping=True, pool_size=settings.worker_threads
        )
        self.batch_rows = settings.storage.stream_batch_rows

    def init_schema(self) -> None:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            for t in ALL_TABLES.values():
                for stmt in t.ddl("postgres"):
                    conn.execute(text(stmt))

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            conn.execute(text(sql), params or {})

    def execute_many(self, statements: list[tuple[str, dict[str, Any] | None]]) -> None:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            for sql, params in statements:
                conn.execute(text(sql), params or {})

    def _rows_to_arrow(self, keys: list[str], rows: list) -> pa.Table:
        cols = {k: [r[i] for r in rows] for i, k in enumerate(keys)}
        return pa.table(cols)

    def query_arrow(self, sql: str, params: dict[str, Any] | None = None) -> pa.Table:
        from sqlalchemy import text

        with self.engine.connect() as conn:
            res = conn.execute(text(sql), params or {})
            keys = list(res.keys())
            return self._rows_to_arrow(keys, res.fetchall())

    def stream_arrow(
        self, sql: str, params: dict[str, Any] | None = None, batch_rows: int | None = None
    ) -> Iterator[pa.RecordBatch]:
        from sqlalchemy import text

        n = batch_rows or self.batch_rows
        with self.engine.connect().execution_options(stream_results=True, max_row_buffer=n) as conn:
            res = conn.execute(text(sql), params or {})
            keys = list(res.keys())
            for part in res.partitions(n):
                yield from self._rows_to_arrow(keys, part).to_batches()

    def write_arrow(self, table: str, data: ArrowLike, mode: WriteMode = "upsert") -> int:
        t: Table = ALL_TABLES[table]
        cols = ", ".join(t.column_names)
        pk = ", ".join(t.primary_key)
        if mode == "upsert":
            sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in t.column_names if c not in t.primary_key)
            conflict = f"ON CONFLICT ({pk}) DO UPDATE SET {sets}" if sets else f"ON CONFLICT ({pk}) DO NOTHING"
        elif mode == "ignore":
            conflict = f"ON CONFLICT ({pk}) DO NOTHING"
        else:
            conflict = ""
        n = 0
        raw = self.engine.raw_connection()
        try:
            cur = raw.cursor()
            stg = f"stg_{table}"
            cur.execute(f"CREATE TEMP TABLE IF NOT EXISTS {stg} (LIKE {table} INCLUDING DEFAULTS)")
            cur.execute(f"ALTER TABLE {stg} ADD COLUMN IF NOT EXISTS __rn BIGINT")
            for chunk in _iter_tables(data):
                if chunk.num_rows == 0:
                    continue
                staged = _with_row_number(conform(t, chunk))
                buf = io.BytesIO()
                pacsv.write_csv(staged, buf)
                buf.seek(0)
                cur.execute(f"TRUNCATE {stg}")
                cur.copy_expert(f"COPY {stg} ({cols}, __rn) FROM STDIN WITH (FORMAT csv, HEADER true)", buf)
                cur.execute(
                    f"INSERT INTO {table} ({cols}) "
                    f"SELECT {cols} FROM (SELECT DISTINCT ON ({pk}) * FROM {stg} ORDER BY {pk}, __rn DESC) d "
                    f"{conflict}"
                )
                n += chunk.num_rows
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()
        return n

    def close(self) -> None:
        self.engine.dispose()


_lock = threading.Lock()
_ADAPTERS: dict[tuple[str, str], StorageAdapter] = {}


def get_storage(settings: Settings | None = None) -> StorageAdapter:
    """Process-wide adapter for the configured backend (schema ensured once)."""
    s = settings or get_settings()
    key = str(s.storage.duckdb_path) if s.storage.backend == "duckdb" else s.storage.postgres_dsn
    with _lock:
        adapter = _ADAPTERS.get((s.storage.backend, key))
        if adapter is None:
            adapter = DuckDBStorage(s) if s.storage.backend == "duckdb" else PostgresStorage(s)
            adapter.init_schema()
            adapter.migrate()
            _ADAPTERS[(s.storage.backend, key)] = adapter
            logger.info("Storage ready: {} ({})", s.storage.backend, key)
        return adapter


def reset_storage_cache() -> None:
    """Close and forget cached adapters (used by tests and CLI tools)."""
    with _lock:
        for adapter in _ADAPTERS.values():
            adapter.close()
        _ADAPTERS.clear()


def release_storage(settings: Settings) -> bool:
    """Close and forget the adapter of one database (a dataset switch releases the DuckDB file so other
    processes can open it). Returns False when none was open."""
    key = str(settings.storage.duckdb_path) if settings.storage.backend == "duckdb" else settings.storage.postgres_dsn
    with _lock:
        adapter = _ADAPTERS.pop((settings.storage.backend, key), None)
    if adapter is not None:
        adapter.close()
    return adapter is not None
