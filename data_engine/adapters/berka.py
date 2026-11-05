"""Berka / PKDD'99 Czech bank dataset -> EWS ingestion schema.

The PKDD'99 Discovery Challenge "financial" dataset holds real, anonymised
data from a Czech bank, 1993-1998: 4,500 accounts, 1.06M transactions with
the running balance after each one, and 682 loans with an outcome status:

    A  contract finished, no problems      B  contract finished, loan not paid
    C  running contract, OK so far         D  running contract, client in debt

Mapping:

* daily_balances: end-of-day balance (the bank's reported balance after the
  day's last transaction), forward-filled over days with no transactions. There is no overdraft-limit data, so ``credit_limit`` and
  ``credit_drawn`` are 0. Balances can go negative, which here means an
  unarranged overdraft.
* transactions: ``k_symbol`` / ``operation`` mapped to EWS types
  (UVER -> EMI, DUCHOD + transfers from other banks -> SALARY, SANKC. UROK
  -> PENALTY_INTEREST, ...).
* loan_schedules: monthly instalment ``payments`` for ``duration`` months
  from the month after the loan date, on the account's usual loan-payment day.
* accounts: one per account; owner as customer, district as RM territory.
* credit_profiles: none (the dataset has no bureau scores).
* holidays: Czech public holidays.

``loans.parquet`` (loan outcomes) is written next to the tables for
evaluation; it is not ingested.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb

END_DATE = date(1998, 12, 31)

_TXN_TYPE = """
CASE
    WHEN k_symbol = 'UVER' THEN 'EMI'
    WHEN k_symbol = 'DUCHOD' THEN 'SALARY'
    WHEN type = 'PRIJEM' AND operation = 'PREVOD Z UCTU' THEN 'SALARY'
    WHEN k_symbol = 'UROK' THEN 'INTEREST_CREDIT'
    WHEN k_symbol = 'SANKC. UROK' THEN 'PENALTY_INTEREST'
    WHEN k_symbol = 'SIPO' THEN 'HOUSEHOLD_BILL'
    WHEN k_symbol = 'POJISTNE' THEN 'INSURANCE'
    WHEN k_symbol = 'SLUZBY' THEN 'FEE'
    WHEN operation = 'VKLAD' THEN 'CASH_DEPOSIT'
    WHEN operation = 'VYBER KARTOU' THEN 'CARD'
    WHEN operation = 'VYBER' THEN 'CASH_WITHDRAWAL'
    WHEN operation = 'PREVOD NA UCET' THEN 'NEFT_OUT'
    ELSE 'OTHER'
END
"""


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def czech_holidays(first_year: int, last_year: int) -> list[tuple[date, str]]:
    fixed = [
        (1, 1, "New Year"), (5, 1, "Labour Day"), (5, 8, "Liberation Day"), (7, 5, "Cyril and Methodius"),
        (7, 6, "Jan Hus"), (10, 28, "Independence Day"), (12, 24, "Christmas Eve"), (12, 25, "Christmas Day"),
        (12, 26, "St Stephen"),
    ]
    out = [(date(y, m, d), n) for y in range(first_year, last_year + 1) for m, d, n in fixed]
    out += [(_easter(y) + timedelta(days=1), "Easter Monday") for y in range(first_year, last_year + 1)]
    return sorted(out)


def convert_berka(raw_dir: Path, out_dir: Path, lead_days: int = 180) -> dict[str, int]:
    """Read the raw ``*.asc`` files and write EWS-schema Parquet into ``out_dir``."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()

    def load(name: str) -> None:
        path = (raw_dir / f"{name}.asc").as_posix()
        c.execute(
            f'CREATE OR REPLACE TABLE "raw_{name}" AS SELECT * FROM read_csv(?, delim=\';\', header=true, quote=\'"\')',
            [path],
        )

    for name in ("trans", "loan", "account", "disp"):
        load(name)
    end = END_DATE.isoformat()

    c.execute(f"""
        CREATE TABLE tx AS
        SELECT 'BK' || lpad(CAST(account_id AS VARCHAR), 6, '0') AS account_id,
               trans_id,
               strptime(CAST(date AS VARCHAR), '%y%m%d')::DATE AS d,
               amount,
               balance,
               CASE WHEN type = 'PRIJEM' THEN amount ELSE -amount END AS signed,
               CASE WHEN type = 'PRIJEM' THEN 'CR' ELSE 'DR' END AS direction,
               {_TXN_TYPE} AS txn_type
        FROM raw_trans
    """)
    # End-of-day balance: the bank's reported balance of the day's final transaction.
    # Amounts are rounded while balances are not, so a recomputed running sum
    # drifts slightly; it is only used to pick which reported balance closes the day.
    c.execute("""
        CREATE TABLE cum AS
        SELECT account_id, d, SUM(SUM(signed)) OVER (PARTITION BY account_id ORDER BY d) AS running
        FROM tx GROUP BY account_id, d
    """)
    c.execute("""
        CREATE TABLE eod AS
        SELECT c.account_id, c.d,
               arg_min(t.balance, abs(t.balance - c.running)) AS eod_balance,
               min(abs(t.balance - c.running)) AS drift
        FROM cum c JOIN tx t ON t.account_id = c.account_id AND t.d = c.d
        GROUP BY c.account_id, c.d
    """)
    match = c.execute("SELECT AVG(CASE WHEN drift < 5 THEN 1.0 ELSE 0.0 END) FROM eod").fetchone()[0]
    c.execute(f"""
        CREATE TABLE daily AS
        WITH spans AS (SELECT account_id, MIN(d) AS d0 FROM eod GROUP BY account_id),
             grid AS (SELECT account_id, CAST(unnest(generate_series(d0, DATE '{end}', INTERVAL 1 DAY)) AS DATE) AS d
                      FROM spans)
        SELECT g.account_id, g.d AS as_of_date,
               last_value(e.eod_balance IGNORE NULLS) OVER (PARTITION BY g.account_id ORDER BY g.d
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS eod_balance,
               0.0 AS credit_limit, 0.0 AS credit_drawn
        FROM grid g LEFT JOIN eod e ON e.account_id = g.account_id AND e.d = g.d
    """)

    c.execute("""
        CREATE TABLE loans AS
        SELECT loan_id, 'BK' || lpad(CAST(account_id AS VARCHAR), 6, '0') AS account_id,
               strptime(CAST(date AS VARCHAR), '%y%m%d')::DATE AS loan_date,
               amount, duration, payments, status
        FROM raw_loan
    """)
    # Usual loan-payment day per account (mode of EMI debit days), else the loan date's day.
    c.execute("""
        CREATE TABLE emi_day AS
        SELECT l.account_id,
               COALESCE((SELECT mode(day(t.d)) FROM tx t WHERE t.account_id = l.account_id AND t.txn_type = 'EMI'),
                        day(l.loan_date)) AS emi_day
        FROM loans l
    """)
    c.execute(f"""
        CREATE TABLE schedules AS
        SELECT l.account_id,
               CAST(date_trunc('month', l.loan_date) + to_months(k)
                    + to_days(LEAST(e.emi_day, day(last_day(date_trunc('month', l.loan_date) + to_months(k)))) - 1)
                    AS DATE) AS due_date,
               l.payments AS emi_amount,
               CAST(k AS INTEGER) AS installment_no
        FROM loans l JOIN emi_day e USING (account_id),
             LATERAL (SELECT unnest(generate_series(1, l.duration)) AS k)
        WHERE date_trunc('month', l.loan_date) + to_months(k) <= DATE '{end}' + INTERVAL {lead_days} DAY
    """)
    c.execute(f"""
        CREATE TABLE accounts AS
        SELECT 'BK' || lpad(CAST(a.account_id AS VARCHAR), 6, '0') AS account_id,
               'CL' || CAST(COALESCE(o.client_id, a.account_id) AS VARCHAR) AS customer_id,
               'D' || lpad(CAST(a.district_id AS VARCHAR), 2, '0') AS rm_id,
               'RETAIL' AS segment,
               CASE WHEN l.loan_id IS NULL THEN 'CASA' ELSE 'LOAN' END AS product_type,
               strptime(CAST(a.date AS VARCHAR), '%y%m%d')::DATE AS open_date,
               COALESCE(l.amount, 0)::DOUBLE AS sanctioned_amount,
               COALESCE(GREATEST(l.amount - l.payments * GREATEST(datediff('month', l.loan_date, DATE '{end}'), 0), 0),
                        0)::DOUBLE AS outstanding_principal,
               COALESCE(l.payments, 0)::DOUBLE AS emi_amount,
               CAST(COALESCE(e.emi_day, 1) AS SMALLINT) AS emi_day,
               CAST(NULL AS SMALLINT) AS salary_day,
               0.0 AS credit_limit,
               0.45 AS lgd,
               0.0 AS interest_rate
        FROM raw_account a
        LEFT JOIN (SELECT account_id, MIN(client_id) AS client_id FROM raw_disp WHERE type = 'OWNER' GROUP BY 1) o
               ON o.account_id = a.account_id
        LEFT JOIN loans l ON l.account_id = 'BK' || lpad(CAST(a.account_id AS VARCHAR), 6, '0')
        LEFT JOIN emi_day e ON e.account_id = l.account_id
    """)

    hol = czech_holidays(1993, END_DATE.year + 1)
    c.execute("CREATE TABLE holidays (holiday_date DATE, name VARCHAR)")
    c.executemany("INSERT INTO holidays VALUES (?, ?)", hol)

    exports = {
        "accounts": "SELECT * FROM accounts",
        "daily_balances": "SELECT * FROM daily ORDER BY account_id, as_of_date",
        "transactions": "SELECT 'T' || CAST(trans_id AS VARCHAR) AS txn_id, account_id, d AS txn_date, amount, "
        "direction, txn_type, 'POSTED' AS status FROM tx ORDER BY d, account_id",
        "loan_schedules": "SELECT * FROM schedules ORDER BY account_id, due_date",
        "credit_profiles": "SELECT CAST(NULL AS VARCHAR) AS account_id, CAST(NULL AS DATE) AS pull_date, "
        "CAST(NULL AS SMALLINT) AS cibil_score, CAST(NULL AS SMALLINT) AS enquiries_6m WHERE false",
        "holidays": "SELECT * FROM holidays",
        "loans": "SELECT * FROM loans",
    }
    counts: dict[str, int] = {}
    for name, sql in exports.items():
        dest = (out_dir / f"{name}.parquet").as_posix()
        c.execute(f"COPY ({sql}) TO '{dest}' (FORMAT parquet, COMPRESSION zstd)")
        counts[name] = c.execute(f"SELECT COUNT(*) FROM read_parquet('{dest}')").fetchone()[0]
    counts["eod_balance_match_rate"] = round(float(match), 4)
    return counts
