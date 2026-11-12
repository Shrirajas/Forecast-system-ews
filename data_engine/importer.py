"""Bring-your-own-data import: map any CSV, Parquet or SQL result onto the EWS tables.

Real exports never use our column names. The importer:

* suggests a mapping from source columns to each table's columns (known synonyms
  such as ``acct_no`` -> ``account_id``, then fuzzy matching),
* reads every source column as text and converts it here, so a bad value in row
  5,000,000 cannot break type inference half-way through a file,
* parses dates in ISO or a user-given format (``%d/%m/%Y`` ...), strips thousands
  separators, normalises debit/credit codes and can derive the direction from a
  signed amount,
* fills columns that are not in the source from user constants or safe defaults,
* streams record batches into :class:`~data_engine.ingestion.IngestionEngine`,
  so large files and query results never sit in memory at once.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from data_engine.ingestion import IngestionEngine, IngestResult
from data_engine.schema import ALL_TABLES, INGESTIBLE_TABLES, Table, conform

#: Accepted source names per target column (compared after lower-casing and
#: dropping everything that is not a letter or digit).
SYNONYMS: dict[str, list[str]] = {
    "account_id": ["account_id", "acct_id", "account", "account_no", "account_number", "acc_no", "acno",
                   "acct_no", "acct", "acc", "acc_id", "loan_account", "customer_account", "account_num"],
    "customer_id": ["customer_id", "cust_id", "client_id", "cif", "cif_id", "customer"],
    "rm_id": ["rm_id", "relationship_manager", "rm", "rm_code", "officer", "branch_code"],
    "segment": ["segment", "customer_segment", "category"],
    "product_type": ["product_type", "product", "loan_type", "scheme", "product_code"],
    "open_date": ["open_date", "account_open_date", "opening_date", "opened_on", "start_date"],
    "sanctioned_amount": ["sanctioned_amount", "sanction_amount", "loan_amount", "principal", "disbursed_amount"],
    "outstanding_principal": ["outstanding_principal", "outstanding", "principal_outstanding", "pos",
                              "balance_outstanding", "loan_outstanding"],
    "emi_amount": ["emi_amount", "emi", "installment_amount", "instalment", "monthly_payment", "payment",
                   "payments"],
    "emi_day": ["emi_day", "emi_date_day", "due_day", "installment_day"],
    "salary_day": ["salary_day", "payroll_day", "salary_date_day"],
    "credit_limit": ["credit_limit", "limit", "od_limit", "cc_limit", "sanctioned_limit", "drawing_power"],
    "lgd": ["lgd", "loss_given_default"],
    "interest_rate": ["interest_rate", "rate", "roi", "rate_of_interest"],
    "as_of_date": ["as_of_date", "date", "balance_date", "bal_date", "eod_date", "business_date", "value_date",
                   "asofdate", "report_date", "snapshot_date"],
    "eod_balance": ["eod_balance", "balance", "closing_balance", "ledger_balance", "end_of_day_balance",
                    "clear_balance", "available_balance"],
    "credit_drawn": ["credit_drawn", "drawn", "utilised", "utilized", "od_outstanding", "drawn_amount"],
    "txn_id": ["txn_id", "transaction_id", "trans_id", "tran_id", "reference", "ref_no", "reference_number",
               "utr", "ref", "txn_ref"],
    "txn_date": ["txn_date", "transaction_date", "tran_date", "posting_date", "date", "value_date"],
    "amount": ["amount", "txn_amount", "transaction_amount", "tran_amount", "amt"],
    "direction": ["direction", "dr_cr", "drcr", "cr_dr", "debit_credit", "type", "txn_direction"],
    "txn_type": ["txn_type", "transaction_type", "tran_type", "category", "k_symbol", "purpose", "channel"],
    "status": ["status", "txn_status", "transaction_status"],
    "pull_date": ["pull_date", "report_date", "bureau_date", "score_date", "date"],
    "cibil_score": ["cibil_score", "cibil", "credit_score", "bureau_score", "score"],
    "enquiries_6m": ["enquiries_6m", "enquiries", "inquiries", "inquiries_6m"],
    "due_date": ["due_date", "emi_date", "installment_date", "instalment_date", "date"],
    "installment_no": ["installment_no", "instalment_no", "emi_no", "installment_number", "seq"],
    "holiday_date": ["holiday_date", "date", "holiday"],
    "name": ["name", "holiday_name", "description"],
}

#: Values used for required columns the source does not have (overridable per import).
DEFAULTS: dict[str, dict[str, Any]] = {
    "accounts": {"segment": "RETAIL", "product_type": "UNKNOWN", "open_date": "1970-01-01",
                 "sanctioned_amount": 0.0, "outstanding_principal": 0.0, "emi_amount": 0.0, "emi_day": 1,
                 "credit_limit": 0.0, "lgd": 0.45, "interest_rate": 0.0},
    "daily_balances": {"credit_limit": 0.0, "credit_drawn": 0.0},
    "transactions": {"status": "POSTED", "txn_type": "OTHER"},
}
#: Required columns copied from another column when absent.
COPY_FROM: dict[str, dict[str, str]] = {"accounts": {"customer_id": "account_id"}}

_DEBIT = {"DR", "D", "DEBIT", "DB", "WITHDRAWAL", "OUT", "VYDAJ", "VYBER", "-"}
_CREDIT = {"CR", "C", "CREDIT", "DEPOSIT", "IN", "PRIJEM", "+"}
_NUMERIC = {"f64", "i32", "i16"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def table_spec(table: str) -> Table:
    if table not in INGESTIBLE_TABLES:
        raise ValueError(f"unknown table '{table}'; choose from {list(INGESTIBLE_TABLES)}")
    return ALL_TABLES[table]


def describe_tables() -> dict[str, list[dict[str, Any]]]:
    """Columns per importable table, for the mapping UI."""
    out = {}
    for t in INGESTIBLE_TABLES:
        spec = ALL_TABLES[t]
        out[t] = [
            {
                "name": n,
                "type": typ,
                "required": not nullable,
                "default": DEFAULTS.get(t, {}).get(n),
                "copy_from": COPY_FROM.get(t, {}).get(n),
            }
            for n, typ, nullable in spec.columns
        ]
    return out


def suggest_mapping(table: str, source_columns: list[str]) -> dict[str, str | None]:
    """Best source column for every target column (None when nothing fits)."""
    spec = table_spec(table)
    by_norm = {_norm(c): c for c in source_columns}
    used: set[str] = set()
    mapping: dict[str, str | None] = {}
    for target, _, _ in spec.columns:  # exact synonym matches first, in synonym order
        for syn in [target, *SYNONYMS.get(target, [])]:
            src = by_norm.get(_norm(syn))
            if src and src not in used:
                mapping[target] = src
                used.add(src)
                break
    date_target = next((n for n, typ, nullable in spec.columns if typ == "date" and not nullable), None)
    for target, _, _ in spec.columns:  # then rules and fuzzy matches on what is left
        if target in mapping:
            continue
        free = [c for c in source_columns if c not in used]
        rule = None
        if target == "account_id":
            rule = next((c for c in free if _norm(c).startswith(("acct", "acc", "account"))), None)
        elif target == date_target:
            rule = next((c for c in free if "date" in _norm(c) or _norm(c).endswith("dt")), None)
        if rule:
            mapping[target] = rule
            used.add(rule)
            continue
        best = difflib.get_close_matches(_norm(target), [_norm(c) for c in free], n=1, cutoff=0.8)
        if best:
            src = next(c for c in free if _norm(c) == best[0])
            mapping[target] = src
            used.add(src)
        else:
            mapping[target] = None
    return mapping


@dataclass
class ImportOptions:
    table: str
    #: target column -> source column (None/absent = not in the source)
    mapping: dict[str, str | None]
    #: target column -> constant used when the column is not mapped
    constants: dict[str, Any] = field(default_factory=dict)
    #: strptime format for date columns, e.g. "%d/%m/%Y"; None = ISO (YYYY-MM-DD)
    date_format: str | None = None
    #: transactions only: when no direction column is mapped, negative amounts are debits
    signed_amounts: bool = True
    decimal_comma: bool = False
    #: identifies the source (content hash of a file, hash of url + query) so that
    #: generated transaction ids are unique across sources and stable on re-import
    source_key: str = "src"

    def validate(self, source_columns: list[str]) -> list[str]:
        """Problems that would stop the import (empty list = ready)."""
        spec = table_spec(self.table)
        errors = []
        cols = set(source_columns)
        for target, src in self.mapping.items():
            if src and src not in cols:
                errors.append(f"'{target}' is mapped to '{src}', which is not in the source")
        for target, _typ, nullable in spec.columns:
            if nullable or self.mapping.get(target) or target in self.constants:
                continue
            if target in DEFAULTS.get(self.table, {}) or target in COPY_FROM.get(self.table, {}):
                continue
            if self.table == "transactions" and target == "direction" and self.signed_amounts:
                continue
            if self.table == "transactions" and target == "txn_id":
                continue  # generated from the row content when missing
            errors.append(f"required column '{target}' is not mapped and has no constant")
        return errors


def _to_number(col: pa.ChunkedArray | pa.Array, decimal_comma: bool) -> pa.Array:
    if pa.types.is_integer(col.type) or pa.types.is_floating(col.type) or pa.types.is_decimal(col.type):
        return pc.cast(col, pa.float64())
    s = pc.utf8_trim_whitespace(pc.cast(col, pa.string()))
    if decimal_comma:
        s = pc.replace_substring(pc.replace_substring(s, ".", ""), ",", ".")
    else:
        s = pc.replace_substring(s, ",", "")
    s = pc.replace_substring_regex(s, r"^\((.*)\)$", r"-\1")  # accounting negatives: (1,234.00)
    s = pc.replace_substring_regex(s, r"[^0-9eE+\-.]", "")  # currency symbols, spaces
    s = pc.if_else(pc.equal(s, ""), pa.scalar(None, pa.string()), s)
    return pc.cast(s, pa.float64())


def _to_date(col: pa.ChunkedArray | pa.Array, fmt: str | None) -> pa.Array:
    if pa.types.is_date(col.type):
        return pc.cast(col, pa.date32())
    if pa.types.is_timestamp(col.type):
        return pc.cast(col, pa.date32())
    s = pc.utf8_trim_whitespace(pc.cast(col, pa.string()))
    s = pc.if_else(pc.equal(s, ""), pa.scalar(None, pa.string()), s)
    if fmt:
        return pc.cast(pc.strptime(s, format=fmt, unit="s", error_is_null=False), pa.date32())
    s = pc.utf8_slice_codeunits(s, 0, 10)  # "2024-01-31 00:00:00" -> "2024-01-31"
    return pc.cast(s, pa.date32())


def _direction(col: pa.ChunkedArray | pa.Array) -> pa.Array:
    codes = pc.utf8_upper(pc.utf8_trim_whitespace(pc.cast(col, pa.string())))
    is_dr = pc.is_in(codes, value_set=pa.array(sorted(_DEBIT)))
    is_cr = pc.is_in(codes, value_set=pa.array(sorted(_CREDIT)))
    bad = pc.and_(pc.invert(pc.or_(is_dr, is_cr)), pc.is_valid(codes))
    if pc.any(bad).as_py():
        example = pc.filter(codes, bad)[0].as_py()
        raise ValueError(f"unrecognised debit/credit value '{example}' (use DR/CR, D/C or DEBIT/CREDIT)")
    return pc.if_else(is_dr, "DR", "CR")


def transform_batch(batch: pa.RecordBatch | pa.Table, opts: ImportOptions, row_offset: int = 0) -> pa.Table:
    """Map, convert and fill one source batch into the target table's schema."""
    spec = table_spec(opts.table)
    tbl = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
    n = tbl.num_rows
    cols: dict[str, pa.Array] = {}
    types = {name: typ for name, typ, _ in spec.columns}

    def const(target: str, value: Any) -> pa.Array:
        return pa.array([value] * n)

    for target, typ, _nullable in spec.columns:
        src = opts.mapping.get(target)
        if src:
            raw = tbl.column(src)
        elif target in opts.constants and opts.constants[target] not in (None, ""):
            raw = const(target, opts.constants[target])
        elif target in COPY_FROM.get(opts.table, {}) and COPY_FROM[opts.table][target] in cols:
            cols[target] = cols[COPY_FROM[opts.table][target]]
            continue
        elif target in DEFAULTS.get(opts.table, {}):
            raw = const(target, DEFAULTS[opts.table][target])
        else:
            continue  # nullable and absent: conform() fills nulls
        if typ == "date":
            cols[target] = _to_date(raw, opts.date_format if src else None)
        elif typ in _NUMERIC:
            num = _to_number(raw, opts.decimal_comma)
            cols[target] = pc.cast(pc.round(num), pa.int32() if typ == "i32" else pa.int16()) if typ != "f64" else num
        else:
            cols[target] = pc.utf8_trim_whitespace(pc.cast(raw, pa.string()))

    if opts.table == "transactions":
        amount = cols["amount"]
        if "direction" in cols and opts.mapping.get("direction"):
            cols["direction"] = _direction(cols["direction"])
        elif opts.signed_amounts:
            cols["direction"] = pc.if_else(pc.less(amount, 0), "DR", "CR")
        cols["amount"] = pc.abs(amount)
        cols["txn_type"] = pc.utf8_upper(cols["txn_type"])
        cols["status"] = pc.utf8_upper(cols["status"])
        if "txn_id" not in cols:  # source + position: re-importing the same source is a no-op
            key = re.sub(r"[^A-Za-z0-9]", "", opts.source_key)[:12] or "src"
            cols["txn_id"] = pa.array([f"IMP-{key}-{row_offset + i:010d}" for i in range(n)])
    if opts.table == "accounts" and "customer_id" not in cols:
        cols["customer_id"] = cols["account_id"]

    out = pa.table({k: v for k, v in cols.items() if k in types})
    return conform(spec, out)


# ---------------------------------------------------------------------------------------- sources
def csv_columns_and_sample(source: str | Path | BinaryIO, rows: int = 20) -> tuple[list[str], list[dict]]:
    reader = pacsv.open_csv(source, read_options=pacsv.ReadOptions(block_size=1 << 20))
    first = reader.read_next_batch()
    names = first.schema.names
    sample = pa.Table.from_batches([first]).slice(0, rows)
    return names, [{k: (None if v is None else str(v)) for k, v in r.items()} for r in sample.to_pylist()]


def csv_batches(source: str | Path | BinaryIO, columns: list[str], block_size: int = 8 << 20) -> Iterator[pa.Table]:
    """All columns as text: conversion happens in transform_batch, never in type inference."""
    reader = pacsv.open_csv(
        source,
        read_options=pacsv.ReadOptions(block_size=block_size, use_threads=True),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in columns},
                                             strings_can_be_null=True),
    )
    for b in reader:
        yield pa.Table.from_batches([b])


def parquet_columns_and_sample(source: str | Path | BinaryIO, rows: int = 20) -> tuple[list[str], list[dict]]:
    pf = pq.ParquetFile(source)
    first = next(pf.iter_batches(batch_size=rows))
    return pf.schema_arrow.names, [{k: (None if v is None else str(v)) for k, v in r.items()}
                                   for r in first.to_pylist()]


def parquet_batches(source: str | Path | BinaryIO, batch_rows: int = 65_536) -> Iterator[pa.Table]:
    for b in pq.ParquetFile(source).iter_batches(batch_size=batch_rows):
        yield pa.Table.from_batches([b])


_READ_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def _check_query(query: str) -> str:
    q = query.strip().rstrip(";")
    if not _READ_ONLY.match(q) or ";" in q:
        raise ValueError("only a single SELECT (or WITH ... SELECT) query is allowed")
    return q


def sql_columns_and_sample(url: str, query: str, rows: int = 20) -> tuple[list[str], list[dict]]:
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            res = conn.execute(text(_check_query(query)))
            names = list(res.keys())
            data = res.fetchmany(rows)
            conn.rollback()
    finally:
        engine.dispose()
    return names, [{k: (None if v is None else str(v)) for k, v in zip(names, r, strict=True)} for r in data]


def sql_batches(url: str, query: str, batch_rows: int = 50_000) -> Iterator[pa.Table]:
    """Server-side cursor: rows arrive in batches, never all at once."""
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    try:
        with engine.connect().execution_options(stream_results=True, max_row_buffer=batch_rows) as conn:
            res = conn.execute(text(_check_query(query)))
            names = list(res.keys())
            for part in res.partitions(batch_rows):
                yield pa.table({n: [r[i] for r in part] for i, n in enumerate(names)})
            conn.rollback()
    finally:
        engine.dispose()


def run_import(ingestion: IngestionEngine, batches: Iterator[pa.Table], opts: ImportOptions) -> IngestResult:
    def mapped() -> Iterator[pa.Table]:
        offset = 0
        for b in batches:
            yield transform_batch(b, opts, offset)
            offset += b.num_rows

    return ingestion.ingest(opts.table, mapped())


def template_csv(table: str) -> str:
    """Header plus one example row, as a starting point for a new export."""
    examples = {
        "accounts": "AC0001,CU0001,RM001,RETAIL_SALARIED,HOME,2021-04-01,2500000,1980000,24000,5,1,150000,0.25,0.085",
        "daily_balances": "AC0001,2026-10-31,48250.75,150000,12000",
        "transactions": "TX000001,AC0001,2026-10-31,24000,DR,EMI,POSTED",
        "credit_profiles": "AC0001,2026-10-01,742,1",
        "loan_schedules": "AC0001,2026-11-05,24000,31",
        "holidays": "2026-11-08,Diwali",
    }
    spec = table_spec(table)
    return ",".join(spec.column_names) + "\n" + examples[table] + "\n"
