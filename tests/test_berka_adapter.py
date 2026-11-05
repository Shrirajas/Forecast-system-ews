"""Berka converter on a tiny hand-made extract in the original *.asc format."""

from __future__ import annotations

from datetime import date

import pyarrow.parquet as pq

from data_engine.adapters.berka import convert_berka, czech_holidays

TRANS = """"trans_id";"account_id";"date";"type";"operation";"amount";"balance";"k_symbol";"bank";"account"
1;7;981001;"PRIJEM";"VKLAD";1000.00;1000.00;"";;
2;7;981003;"PRIJEM";"PREVOD Z UCTU";5000.04;6000.04;"";"AB";123
3;7;981003;"VYDAJ";"PREVOD NA UCET";2500.00;3500.04;"UVER";"CD";456
4;7;981010;"VYDAJ";"VYBER";4000.00;-499.96;"";;
5;7;981031;"VYDAJ";"VYBER";14.60;-514.56;"SANKC. UROK";;
6;8;981215;"PRIJEM";"VKLAD";300.00;300.00;"";;
"""
LOAN = """"loan_id";"account_id";"date";"amount";"duration";"payments";"status"
1;7;980905;30000;12;2500.00;"D"
"""
ACCOUNT = """"account_id";"district_id";"frequency";"date"
7;18;"POPLATEK MESICNE";980901
8;1;"POPLATEK MESICNE";981201
"""
DISP = """"disp_id";"client_id";"account_id";"type"
1;70;7;"OWNER"
2;80;8;"OWNER"
"""


def test_convert_berka_extract(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, text in {"trans": TRANS, "loan": LOAN, "account": ACCOUNT, "disp": DISP}.items():
        (raw / f"{name}.asc").write_text(text)
    counts = convert_berka(raw, tmp_path / "out")
    assert counts["accounts"] == 2 and counts["transactions"] == 6 and counts["loans"] == 1
    assert counts["eod_balance_match_rate"] == 1.0

    daily = pq.read_table(tmp_path / "out" / "daily_balances.parquet").to_pandas()
    a = daily[daily.account_id == "BK000007"].set_index("as_of_date").eod_balance
    assert a[date(1998, 10, 1)] == 1000.0
    assert a[date(1998, 10, 3)] == 3500.04  # reported balance after the day's last transaction
    assert a[date(1998, 10, 5)] == 3500.04  # forward-filled
    assert a[date(1998, 12, 31)] == -514.56  # negative balances preserved
    assert len(a) == (date(1998, 12, 31) - date(1998, 10, 1)).days + 1

    tx = pq.read_table(tmp_path / "out" / "transactions.parquet").to_pandas().set_index("txn_id")
    assert tx.loc["T3", "txn_type"] == "EMI" and tx.loc["T2", "txn_type"] == "SALARY"
    assert tx.loc["T5", "txn_type"] == "PENALTY_INTEREST" and tx.loc["T4", "direction"] == "DR"

    sched = pq.read_table(tmp_path / "out" / "loan_schedules.parquet").to_pandas()
    assert len(sched) == 9 and set(sched.emi_amount) == {2500.0}  # cut 180 days after the data ends
    assert sched.due_date.max() == date(1999, 6, 3)
    assert sched.due_date.min() == date(1998, 10, 3)  # month after the loan, on the usual payment day

    acc = pq.read_table(tmp_path / "out" / "accounts.parquet").to_pandas().set_index("account_id")
    assert acc.loc["BK000007", "product_type"] == "LOAN" and acc.loc["BK000008", "product_type"] == "CASA"
    assert acc.loc["BK000007", "customer_id"] == "CL70" and acc.loc["BK000007", "rm_id"] == "D18"


def test_czech_holidays_include_easter_monday():
    days = dict(czech_holidays(1998, 1998))
    assert days[date(1998, 4, 13)] == "Easter Monday"
    assert days[date(1998, 10, 28)] == "Independence Day"
