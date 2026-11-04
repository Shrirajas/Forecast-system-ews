"""Streaming ingestion into the storage layer.

Inbound data never needs to be materialised in Python: files are parsed by
PyArrow's multi-threaded readers into bounded record batches, a background
thread keeps the parser ahead of the database writer (see
:func:`utils.streaming.prefetch`), and each batch is upserted zero-copy.
Every load marks the feature matrix dirty from the earliest event date it
touched so the nightly aggregation only recomputes what changed.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import BinaryIO, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from data_engine.schema import ALL_TABLES, EVENT_DATE_COLUMN, INGESTIBLE_TABLES, conform
from data_engine.storage import ArrowLike, StorageAdapter, WriteMode, _iter_tables
from utils.logging import logger
from utils.streaming import prefetch

FileFormat = Literal["csv", "parquet", "ipc"]
FEATURE_WATERMARK = "feature_matrix"
INGEST_MARKER = "ingest"


@dataclass
class IngestResult:
    table: str
    rows: int
    min_event_date: date | None
    max_event_date: date | None


class IngestionEngine:
    def __init__(self, storage: StorageAdapter, batch_rows: int = 65_536, prefetch_depth: int = 4):
        self.storage = storage
        self.batch_rows = batch_rows
        self.prefetch_depth = prefetch_depth
        self._mark_lock = threading.Lock()

    # ---- core ---------------------------------------------------------------------------------
    def ingest(self, table: str, data: ArrowLike, mode: WriteMode | None = None) -> IngestResult:
        if table not in INGESTIBLE_TABLES:
            raise ValueError(f"table '{table}' is not ingestible; choose from {INGESTIBLE_TABLES}")
        spec = ALL_TABLES[table]
        mode = mode or ("ignore" if table == "transactions" else "upsert")
        date_col = EVENT_DATE_COLUMN.get(table)
        stats: dict[str, date | None] = {"min": None, "max": None}

        def conformed() -> Iterator[pa.Table]:
            for chunk in _iter_tables(data):
                tbl = conform(spec, chunk)
                if date_col and tbl.num_rows:
                    mm = pc.min_max(tbl.column(date_col)).as_py()
                    lo, hi = mm["min"], mm["max"]
                    if lo is not None:
                        stats["min"] = lo if stats["min"] is None else min(stats["min"], lo)
                        stats["max"] = hi if stats["max"] is None else max(stats["max"], hi)
                yield tbl

        rows = self.storage.write_arrow(table, prefetch(conformed(), self.prefetch_depth), mode)
        self._mark_ingested(stats["min"])
        logger.info("ingested {} rows into {} (events {} .. {})", rows, table, stats["min"], stats["max"])
        return IngestResult(table, rows, stats["min"], stats["max"])

    # ---- sources ------------------------------------------------------------------------------
    def _csv_batches(self, source: str | Path | BinaryIO, table: str) -> Iterator[pa.RecordBatch]:
        schema = ALL_TABLES[table].arrow_schema
        reader = pacsv.open_csv(
            source,
            read_options=pacsv.ReadOptions(block_size=8 << 20, use_threads=True),
            convert_options=pacsv.ConvertOptions(
                column_types={f.name: f.type for f in schema},
                strings_can_be_null=True,
            ),
        )
        yield from reader

    def _parquet_batches(self, source: str | Path | BinaryIO) -> Iterator[pa.RecordBatch]:
        pf = pq.ParquetFile(source, memory_map=isinstance(source, str | Path))
        yield from pf.iter_batches(batch_size=self.batch_rows, use_threads=True)

    def _ipc_batches(self, source: str | Path | BinaryIO) -> Iterator[pa.RecordBatch]:
        src = pa.memory_map(str(source)) if isinstance(source, str | Path) else source
        try:
            reader = pa.ipc.open_file(src)
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)
        except pa.ArrowInvalid:
            if hasattr(src, "seek"):
                src.seek(0)
            yield from pa.ipc.open_stream(src)

    def ingest_file(
        self, table: str, source: str | Path | BinaryIO, fmt: FileFormat, mode: WriteMode | None = None
    ) -> IngestResult:
        if fmt == "csv":
            batches = self._csv_batches(source, table)
        elif fmt == "parquet":
            batches = self._parquet_batches(source)
        elif fmt == "ipc":
            batches = self._ipc_batches(source)
        else:
            raise ValueError(f"unsupported format {fmt}")
        return self.ingest(table, batches, mode)

    def ingest_dataset(
        self, table: str, path: str | Path, fmt: Literal["csv", "parquet"] = "parquet", mode: WriteMode | None = None
    ) -> IngestResult:
        """Directory / glob / partitioned dataset (multi-file, multi-threaded scan)."""
        dataset = ds.dataset(str(path), format=fmt)
        batches = dataset.to_batches(batch_size=self.batch_rows, use_threads=True)
        return self.ingest(table, batches, mode)

    # ---- watermark bookkeeping -------------------------------------------------------------------
    def _mark_ingested(self, min_event: date | None) -> None:
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._mark_lock:
            stmts: list = [
                ("DELETE FROM agg_watermarks WHERE name = :n", {"n": INGEST_MARKER}),
                (
                    "INSERT INTO agg_watermarks (name, watermark, dirty_from, updated_at) VALUES (:n, NULL, NULL, :ts)",
                    {"n": INGEST_MARKER, "ts": now},
                ),
            ]
            if min_event is not None:
                cur = self.storage.query_arrow(
                    "SELECT watermark, dirty_from FROM agg_watermarks WHERE name = :n", {"n": FEATURE_WATERMARK}
                ).to_pylist()
                wm = cur[0]["watermark"] if cur else None
                dirty = cur[0]["dirty_from"] if cur else None
                # Only events at or before the watermark are "late" and need a rewind;
                # newer days are picked up by the normal incremental pass.
                if wm is not None and min_event <= wm:
                    dirty = min_event if dirty is None else min(dirty, min_event)
                stmts += [
                    ("DELETE FROM agg_watermarks WHERE name = :n", {"n": FEATURE_WATERMARK}),
                    (
                        "INSERT INTO agg_watermarks (name, watermark, dirty_from, updated_at) "
                        "VALUES (:n, :wm, :dirty, :ts)",
                        {"n": FEATURE_WATERMARK, "wm": wm, "dirty": dirty, "ts": now},
                    ),
                ]
            self.storage.execute_many(stmts)


def last_ingest_time(storage: StorageAdapter) -> datetime | None:
    return storage.scalar("SELECT updated_at FROM agg_watermarks WHERE name = :n", {"n": INGEST_MARKER})
