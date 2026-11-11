"""IBM TabFormer synthetic credit-card transactions -> EWS ingestion schema.

``card_transaction.v1.csv`` holds 24M card swipes (2,000 users, ~6,100 cards,
1991-2020) with no balances, payments or loans. The optional companion files
``sd254_cards.csv`` (one row per card: type, credit limit, open date) and
``sd254_users.csv`` (one row per user, in user-id order: FICO, income, debt,
state) add the risk context. Mapping:

* one EWS account per card: ``CC<user>_<card>``,
* daily_balances.eod_balance = the card's rolling 30-day net spend (refunds
  subtract, declined swipes are ignored); a projected drop raises a
  "spend falling" (deposit-attrition) alert,
* credit cards: credit_limit = the card's limit and credit_drawn = 30-day spend,
  so utilisation (the model's second target) is spend / limit,
* credit_profiles: the user's FICO score as the bureau score (feeds PD),
* accounts: product_type CREDIT_CARD / DEBIT_CARD / PREPAID_CARD, the real open
  date, rm_id = the user's state, segment = debt-to-income band,
* transactions: approved swipes as CARD debits, refunds as credits,
  "Insufficient Balance" declines as RETURNED (the engine reads these as a
  payment-capacity signal); other declines (bad PIN, glitch...) moved no money
  and are dropped,
* no loans or bill payments, so there are no EMIs and SMA tiers stay GREEN,
* holidays: fixed-date US federal holidays.

Only cards with a swipe inside ``[start, end]`` are kept.
"""

from __future__ import annotations

import csv as _csv
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pyarrow as pa

WINDOW_DAYS = 30
CARD_LGD = 0.75  # unsecured card lending: higher loss-given-default than the 0.45 default


def _money(s: str) -> float | None:
    s = (s or "").replace("$", "").replace(",", "").strip()
    return float(s) if s else None


def _users_table(path: Path) -> pa.Table:
    """sd254_users.csv has no id column: the row order is the user id (row 0 = user 0)."""
    with open(path, newline="") as f:
        rows = list(_csv.DictReader(f))
    income = [_money(r["Yearly Income - Person"]) for r in rows]
    debt = [_money(r["Total Debt"]) for r in rows]
    return pa.table({
        "uid": list(range(len(rows))),
        "fico": [int(r["FICO Score"]) for r in rows],
        "state": [r["State"] or None for r in rows],
        "dti": [d / i if i else None for d, i in zip(debt, income)],
    })


def convert_cards(
    csv: Path, out_dir: Path, start: date, end: date,
    cards_csv: Path | None = None, users_csv: Path | None = None,
) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warm = start - timedelta(days=WINDOW_DAYS - 1)  # so the first kept day has a full 30-day window
    c = duckdb.connect()
    c.execute(f"""
        CREATE TABLE tx AS
        SELECT 'CC' || lpad(CAST("User" AS VARCHAR), 4, '0') || '_' || CAST("Card" AS VARCHAR) AS account_id,
               "User" AS uid, "Card" AS card,
               make_date("Year", "Month", "Day") AS d,
               CAST(replace("Amount", '$', '') AS DOUBLE) AS amt,
               "Errors?" IS NOT NULL AS declined,
               -- only a lack of funds is a payment-capacity signal; bad PIN / glitch declines are not
               coalesce(contains("Errors?", 'Insufficient Balance'), false) AS no_funds
        FROM read_csv(?, header = true)
        WHERE make_date("Year", "Month", "Day") BETWEEN DATE '{warm}' AND DATE '{end}'
    """, [Path(csv).as_posix()])

    if cards_csv:
        c.execute("""
            CREATE TABLE card_info AS
            SELECT CAST("User" AS INTEGER) AS uid, CAST("CARD INDEX" AS INTEGER) AS card,
                   CASE "Card Type" WHEN 'Credit' THEN 'CREDIT_CARD' WHEN 'Debit' THEN 'DEBIT_CARD'
                        WHEN 'Debit (Prepaid)' THEN 'PREPAID_CARD' ELSE 'CARD' END AS product_type,
                   CAST(replace("Credit Limit", '$', '') AS DOUBLE) AS card_limit,
                   CAST(strptime("Acct Open Date", '%m/%Y') AS DATE) AS opened
            FROM read_csv(?, header = true, all_varchar = true)
        """, [Path(cards_csv).as_posix()])
    else:
        c.execute("CREATE TABLE card_info (uid INTEGER, card INTEGER, product_type VARCHAR, card_limit DOUBLE, opened DATE)")
    users = _users_table(users_csv) if users_csv else pa.table(
        {"uid": pa.array([], pa.int64()), "fico": pa.array([], pa.int64()),
         "state": pa.array([], pa.string()), "dti": pa.array([], pa.float64())})
    c.register("users", users)

    c.execute(f"""
        CREATE TABLE cards AS
        SELECT t.account_id, t.uid, MIN(t.d) AS first_d,
               COALESCE(any_value(k.product_type), 'CARD') AS product_type,
               -- only credit cards have a credit line; a debit "limit" is not drawn against
               CASE WHEN any_value(k.product_type) = 'CREDIT_CARD' THEN COALESCE(any_value(k.card_limit), 0) ELSE 0 END
                   AS credit_limit,
               any_value(k.opened) AS opened
        FROM tx t LEFT JOIN card_info k ON k.uid = t.uid AND k.card = t.card
        GROUP BY t.account_id, t.uid HAVING MAX(t.d) >= DATE '{start}'
    """)
    c.execute(f"""
        CREATE TABLE daily AS
        WITH spend AS (SELECT account_id, d, SUM(amt) AS s FROM tx WHERE NOT declined GROUP BY 1, 2),
             grid AS (SELECT account_id, CAST(unnest(generate_series(GREATEST(first_d, DATE '{warm}'), DATE '{end}',
                                                                      INTERVAL 1 DAY)) AS DATE) AS d FROM cards),
             roll AS (SELECT g.account_id, g.d,
                             SUM(COALESCE(s.s, 0)) OVER (PARTITION BY g.account_id ORDER BY g.d
                                 ROWS BETWEEN {WINDOW_DAYS - 1} PRECEDING AND CURRENT ROW) AS bal
                      FROM grid g LEFT JOIN spend s USING (account_id, d))
        SELECT r.account_id, r.d AS as_of_date, round(r.bal, 2) AS eod_balance, k.credit_limit,
               -- shortcut: drawn = 30-day spend assumes the bill is paid in full monthly; no payment data exists
               CASE WHEN k.credit_limit > 0 THEN round(GREATEST(r.bal, 0), 2) ELSE 0.0 END AS credit_drawn
        FROM roll r JOIN cards k USING (account_id) WHERE r.d >= DATE '{start}'
    """)
    years = range(start.year, end.year + 2)
    fixed = [(1, 1, "New Year"), (7, 4, "Independence Day"), (11, 11, "Veterans Day"), (12, 25, "Christmas Day")]
    exports = {
        "accounts": f"""SELECT k.account_id, 'U' || CAST(k.uid AS VARCHAR) AS customer_id,
                       CASE WHEN u.state IS NULL THEN NULL ELSE 'ST-' || u.state END AS rm_id,
                       CASE WHEN u.dti IS NULL THEN 'RETAIL' WHEN u.dti < 1 THEN 'DTI_UNDER_1X'
                            WHEN u.dti < 2.5 THEN 'DTI_1_TO_2.5X' ELSE 'DTI_OVER_2.5X' END AS segment,
                       k.product_type, COALESCE(k.opened, LEAST(k.first_d, DATE '{start}')) AS open_date,
                       0.0 AS sanctioned_amount, 0.0 AS outstanding_principal, 0.0 AS emi_amount,
                       1::SMALLINT AS emi_day, NULL::SMALLINT AS salary_day, k.credit_limit,
                       CASE WHEN k.product_type = 'CREDIT_CARD' THEN {CARD_LGD} ELSE 0.45 END AS lgd,
                       0.0 AS interest_rate
                       FROM cards k LEFT JOIN users u ON u.uid = k.uid ORDER BY k.account_id""",
        "daily_balances": "SELECT * FROM daily ORDER BY account_id, as_of_date",
        "transactions": f"""SELECT 'C' || CAST(row_number() OVER () AS VARCHAR) AS txn_id, account_id, d AS txn_date,
                           abs(amt) AS amount, CASE WHEN amt < 0 THEN 'CR' ELSE 'DR' END AS direction,
                           CASE WHEN amt < 0 THEN 'OTHER' ELSE 'CARD' END AS txn_type,
                           CASE WHEN no_funds THEN 'RETURNED' ELSE 'POSTED' END AS status
                           FROM tx WHERE d >= DATE '{start}' AND account_id IN (SELECT account_id FROM cards)
                           AND (NOT declined OR no_funds)""",
        "loan_schedules": "SELECT NULL::VARCHAR AS account_id, NULL::DATE AS due_date, NULL::DOUBLE AS emi_amount, "
        "NULL::INTEGER AS installment_no WHERE false",
        # FICO is a single snapshot, recorded at the window start (no score history exists)
        "credit_profiles": f"""SELECT k.account_id, DATE '{start}' AS pull_date, CAST(u.fico AS SMALLINT) AS cibil_score,
                              NULL::SMALLINT AS enquiries_6m FROM cards k JOIN users u ON u.uid = k.uid
                              ORDER BY k.account_id""",
        "holidays": "SELECT * FROM (VALUES " + ",".join(
            f"(DATE '{date(y, m, d)}', '{n}')" for y in years for m, d, n in fixed) + ") t(holiday_date, name)",
    }
    counts: dict[str, int] = {}
    for name, sql in exports.items():
        dest = (out_dir / f"{name}.parquet").as_posix()
        c.execute(f"COPY ({sql}) TO '{dest}' (FORMAT parquet, COMPRESSION zstd)")
        counts[name] = c.execute(f"SELECT COUNT(*) FROM read_parquet('{dest}')").fetchone()[0]
    return counts


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Convert card_transaction.v1.csv (+ sd254 files) to EWS parquet tables.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cards", type=Path, help="sd254_cards.csv")
    ap.add_argument("--users", type=Path, help="sd254_users.csv")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2020, 2, 28))
    a = ap.parse_args()
    print(json.dumps(convert_cards(a.csv, a.out, a.start, a.end, a.cards, a.users)))
