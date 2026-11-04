"""Memory-mapped tensor cache ("vector cache") feeding the forecasting engine.

Layout of one cache version (``<root>/v<timestamp>/``):

* ``past.npy``  float32 ``[accounts, days, 6]``: targets + past-only covariates,
  NaN where a day is missing (the model masks those positions),
* ``known.npy`` float32 ``[accounts, days + horizon_pad, 3]``: covariates
  known in advance (EMI schedule, bank holidays, payroll days), extending
  past the last observed day so future windows can be cut,
* ``accounts.parquet``: static attributes plus each account's row index,
* ``meta.json``.

Arrays are written with ``numpy.lib.format.open_memmap``. Shards of accounts
are filled in parallel threads, each streaming its own Arrow record batches
from the database, so a portfolio far larger than RAM can be cached: the OS
pages data in and out instead of the process holding it. Readers map the
files read-only and slice only the accounts they need. A ``CURRENT`` pointer
is swapped atomically, so a rebuild never disturbs in-flight readers.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config.settings import FUTURE_COVARIATES, PAST_COVARIATES, TARGET_FEATURES
from data_engine import calendar as cal
from data_engine.ingestion import FEATURE_WATERMARK, last_ingest_time
from data_engine.storage import StorageAdapter
from utils.logging import logger

PAST_CHANNELS: tuple[str, ...] = TARGET_FEATURES + PAST_COVARIATES
KNOWN_CHANNELS: tuple[str, ...] = FUTURE_COVARIATES
CIBIL_CH = PAST_CHANNELS.index("cibil_score")
DEFAULT_CIBIL = 700.0
_EPOCH = date(1970, 1, 1)


@dataclass
class CacheMeta:
    version: str
    start_date: str
    end_date: str
    n_accounts: int
    n_days: int
    horizon_pad: int
    past_channels: list[str]
    known_channels: list[str]
    built_at: str
    ingest_marker: str | None
    feature_watermark: str | None
    build_seconds: float


def _row_index(account_ids: pa.Array, ids: pa.Array) -> np.ndarray:
    """Vectorised hash lookup of account ids -> cache rows (-1 if unknown)."""
    return pc.fill_null(pc.index_in(account_ids, value_set=ids), -1).to_numpy().astype(np.int64)


def _ffill_bfill(a: np.ndarray, default: float) -> np.ndarray:
    """Forward- then back-fill NaNs along axis 1 of ``[n, t]``."""
    n, t = a.shape
    valid = ~np.isnan(a)
    idx = np.where(valid, np.arange(t)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = a[np.arange(n)[:, None], idx]
    first = np.where(valid.any(1), valid.argmax(1), 0)
    lead = np.isnan(out)
    out[lead] = np.broadcast_to(a[np.arange(n), first][:, None], out.shape)[lead]
    out[np.isnan(out)] = default
    return out


class LoadedCache:
    """Read-only view over one cache version."""

    def __init__(self, vdir: Path):
        self.dir = vdir
        self.meta = CacheMeta(**json.loads((vdir / "meta.json").read_text()))
        self.past = np.load(vdir / "past.npy", mmap_mode="r")
        self.known = np.load(vdir / "known.npy", mmap_mode="r")
        self.accounts: pd.DataFrame = pq.read_table(vdir / "accounts.parquet").to_pandas()
        self._row = {a: i for i, a in enumerate(self.accounts["account_id"])}
        self.start = date.fromisoformat(self.meta.start_date)
        self.end = date.fromisoformat(self.meta.end_date)

    @property
    def n_accounts(self) -> int:
        return self.meta.n_accounts

    def rows_for(self, account_ids: list[str] | None) -> tuple[np.ndarray, list[str]]:
        """Row indices for ``account_ids`` (all when None) plus unknown ids."""
        if account_ids is None:
            return np.arange(self.n_accounts), []
        rows, missing = [], []
        for a in account_ids:
            r = self._row.get(a)
            (missing.append(a) if r is None else rows.append(r))
        return np.asarray(rows, dtype=np.int64), missing

    def day_index(self, d: date) -> int:
        return (d - self.start).days

    def window(self, rows: np.ndarray, as_of: date, context: int, horizon_pad: int) -> dict[str, np.ndarray]:
        """Cut model inputs for forecasting the days after ``as_of``.

        Returns ``past [B, context, 6]`` (NaN = missing), ``known
        [B, context + horizon_pad, 3]`` and the last observed day's raw values.
        """
        end = self.day_index(as_of) + 1  # exclusive
        if end <= 0 or end > self.meta.n_days:
            raise ValueError(f"as_of {as_of} outside cached range {self.start}..{self.end}")
        if horizon_pad > self.meta.horizon_pad:
            raise ValueError("requested horizon exceeds cached known-covariate horizon")
        lo = end - context
        b = len(rows)
        past = np.full((b, context, len(PAST_CHANNELS)), np.nan, dtype=np.float32)
        known = np.zeros((b, context + horizon_pad, len(KNOWN_CHANNELS)), dtype=np.float32)
        src_lo = max(lo, 0)
        order = np.argsort(rows)  # sorted fancy-index reads are sequential on disk
        srows = rows[order]
        past[order, src_lo - lo :] = self.past[srows, src_lo:end]
        known[order, src_lo - lo :] = self.known[srows, src_lo : end + horizon_pad]
        return {"past": past, "known": known}


class VectorCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._loaded: LoadedCache | None = None

    # ---- read side -------------------------------------------------------------------------
    def current_dir(self) -> Path | None:
        ptr = self.root / "CURRENT"
        if not ptr.exists():
            return None
        vdir = self.root / ptr.read_text().strip()
        return vdir if (vdir / "meta.json").exists() else None

    def load(self, refresh: bool = False) -> LoadedCache | None:
        vdir = self.current_dir()
        if vdir is None:
            return None
        if refresh or self._loaded is None or self._loaded.dir != vdir:
            self._loaded = LoadedCache(vdir)
        return self._loaded

    def is_stale(self, storage: StorageAdapter) -> bool:
        cache = self.load()
        if cache is None:
            return True
        marker = last_ingest_time(storage)
        wm = storage.scalar("SELECT watermark FROM agg_watermarks WHERE name = :n", {"n": FEATURE_WATERMARK})
        return (str(marker) if marker else None) != cache.meta.ingest_marker or (
            str(wm) if wm else None
        ) != cache.meta.feature_watermark

    # ---- build side ------------------------------------------------------------------------
    def build(
        self, storage: StorageAdapter, horizon_pad: int = 96, workers: int = 4, history_days: int | None = None
    ) -> CacheMeta:
        t0 = time.perf_counter()
        marker = last_ingest_time(storage)
        wm = storage.scalar("SELECT watermark FROM agg_watermarks WHERE name = :n", {"n": FEATURE_WATERMARK})
        b = storage.query_arrow("SELECT MIN(as_of_date) AS lo, MAX(as_of_date) AS hi FROM feature_matrix").to_pylist()[0]
        if b["hi"] is None:
            raise RuntimeError("feature_matrix is empty: ingest data and run the aggregation first")
        hi: date = b["hi"]
        lo: date = b["lo"] if history_days is None else max(b["lo"], hi - timedelta(days=history_days - 1))
        n_days = (hi - lo).days + 1

        ids_tbl = storage.query_arrow(
            "SELECT DISTINCT account_id FROM feature_matrix WHERE as_of_date >= CAST(:lo AS DATE) ORDER BY account_id",
            {"lo": lo},
        )
        ids = ids_tbl.column("account_id").combine_chunks()
        n = len(ids)
        static = storage.query_df("SELECT * FROM accounts")
        accounts = pd.DataFrame({"account_id": ids.to_pylist()}).merge(static, on="account_id", how="left")
        accounts["row"] = np.arange(n)

        version = "v" + datetime.now().strftime("%Y%m%dT%H%M%S%f")
        vdir = self.root / version
        vdir.mkdir(parents=True, exist_ok=True)
        past = np.lib.format.open_memmap(vdir / "past.npy", "w+", np.float32, (n, n_days, len(PAST_CHANNELS)))
        known = np.lib.format.open_memmap(
            vdir / "known.npy", "w+", np.float32, (n, n_days + horizon_pad, len(KNOWN_CHANNELS))
        )
        lo_int = (lo - _EPOCH).days
        horizon_end = hi + timedelta(days=horizon_pad)

        holidays = storage.query_arrow("SELECT holiday_date FROM holidays").column(0).to_numpy(zero_copy_only=False)
        holidays = holidays.astype("datetime64[D]")
        cal_days = cal.day_range(lo, n_days + horizon_pad)
        holiday_flag = cal.non_business_mask(cal_days, holidays).astype(np.float32)

        salary_day = self._salary_days(storage, accounts, hi)
        accounts["salary_day_effective"] = salary_day

        n_workers = max(1, min(workers, n))
        bounds = np.linspace(0, n, n_workers + 1).astype(int)
        shards = [(int(a), int(z)) for a, z in zip(bounds[:-1], bounds[1:], strict=True) if z > a]
        cols = ", ".join(PAST_CHANNELS)

        def fill_shard(shard: tuple[int, int]) -> int:
            r0, r1 = shard
            past[r0:r1] = np.nan
            known[r0:r1] = 0.0
            params = {"a": ids[r0].as_py(), "z": ids[r1 - 1].as_py(), "lo": lo, "hi": hi, "he": horizon_end}
            rows_written = 0
            for batch in storage.stream_arrow(
                f"SELECT account_id, as_of_date, {cols} FROM feature_matrix "
                "WHERE account_id BETWEEN :a AND :z AND as_of_date BETWEEN CAST(:lo AS DATE) AND CAST(:hi AS DATE)",
                params,
            ):
                r = _row_index(batch.column("account_id"), ids)
                t = batch.column("as_of_date").cast(pa.int32()).to_numpy() - lo_int
                vals = np.column_stack(
                    [batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in PAST_CHANNELS]
                )
                past[r, t] = vals
                rows_written += batch.num_rows
            # forward-fill bureau scores so every day carries the latest known value
            past[r0:r1, :, CIBIL_CH] = _ffill_bfill(np.asarray(past[r0:r1, :, CIBIL_CH]), DEFAULT_CIBIL)

            for batch in storage.stream_arrow(
                "SELECT account_id, due_date, emi_amount FROM loan_schedules "
                "WHERE account_id BETWEEN :a AND :z AND due_date BETWEEN CAST(:lo AS DATE) AND CAST(:he AS DATE)",
                params,
            ):
                r = _row_index(batch.column("account_id"), ids)
                ok = r >= 0
                t = batch.column("due_date").cast(pa.int32()).to_numpy() - lo_int
                emi = batch.column("emi_amount").to_numpy(zero_copy_only=False)
                blk = np.zeros((r1 - r0, n_days + horizon_pad), dtype=np.float32)
                np.add.at(blk, (r[ok] - r0, t[ok]), emi[ok])
                known[r0:r1, :, 0] += blk

            known[r0:r1, :, 1] = holiday_flag[None, :]
            sd = salary_day[r0:r1]
            pay = cal.monthly_dates(np.where(sd > 0, sd, 0), lo, horizon_end, holidays, roll=-1)
            rr, cc = np.nonzero(~np.isnat(pay))
            ti = (pay[rr, cc] - np.datetime64(lo, "D")).astype(np.int64)
            known[r0 + rr, ti, 2] = 1.0
            return rows_written

        with ThreadPoolExecutor(max_workers=len(shards), thread_name_prefix="cache") as ex:
            total = sum(ex.map(fill_shard, shards))
        past.flush()
        known.flush()

        pq.write_table(pa.Table.from_pandas(accounts, preserve_index=False), vdir / "accounts.parquet")
        meta = CacheMeta(
            version=version,
            start_date=lo.isoformat(),
            end_date=hi.isoformat(),
            n_accounts=n,
            n_days=n_days,
            horizon_pad=horizon_pad,
            past_channels=list(PAST_CHANNELS),
            known_channels=list(KNOWN_CHANNELS),
            built_at=datetime.now().isoformat(timespec="seconds"),
            ingest_marker=str(marker) if marker else None,
            feature_watermark=str(wm) if wm else None,
            build_seconds=round(time.perf_counter() - t0, 3),
        )
        (vdir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
        tmp = self.root / "CURRENT.tmp"
        tmp.write_text(version)
        os.replace(tmp, self.root / "CURRENT")
        self._cleanup(keep={version, self._loaded.dir.name if self._loaded else ""})
        logger.info(
            "vector cache {}: {} accounts x {} days ({} feature rows) in {:.2f}s using {} threads",
            version, n, n_days, total, meta.build_seconds, len(shards),
        )
        return meta

    @staticmethod
    def _salary_days(storage: StorageAdapter, accounts: pd.DataFrame, hi: date) -> np.ndarray:
        """Payroll day per account: master data first, else the modal credit day."""
        sd = accounts["salary_day"].fillna(0).astype(int).to_numpy() if "salary_day" in accounts else None
        if sd is None:
            sd = np.zeros(len(accounts), dtype=int)
        hist = storage.query_df(
            "SELECT account_id, EXTRACT(DAY FROM txn_date) AS dom, COUNT(*) AS c FROM transactions "
            "WHERE txn_type = 'SALARY' AND direction = 'CR' AND txn_date >= CAST(:s AS DATE) "
            "GROUP BY account_id, EXTRACT(DAY FROM txn_date)",
            {"s": hi - timedelta(days=180)},
        )
        if not hist.empty:
            hist["c"] = hist["c"].astype(int)
            mode = hist.sort_values("c", ascending=False).drop_duplicates("account_id").set_index("account_id")["dom"]
            inferred = accounts["account_id"].map(mode).fillna(0).astype(int).to_numpy()
            sd = np.where(sd > 0, sd, inferred)
        return sd.astype(np.int64)

    def _cleanup(self, keep: set[str]) -> None:
        versions = sorted(p for p in self.root.glob("v*") if p.is_dir())
        for p in versions[:-2]:
            if p.name not in keep:
                shutil.rmtree(p, ignore_errors=True)  # may fail on Windows while mapped
