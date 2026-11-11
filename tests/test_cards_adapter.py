from datetime import date

import pyarrow.parquet as pq

from data_engine.adapters.cards import convert_cards

CSV = """User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,Zip,MCC,Errors?,Is Fraud?
1,0,2020,1,1,06:21,$100.00,Swipe Transaction,1,X,CA,1.0,5300,,No
1,0,2020,1,5,06:21,$50.00,Swipe Transaction,1,X,CA,1.0,5300,Insufficient Balance,No
1,0,2020,1,10,06:21,$-20.00,Swipe Transaction,1,X,CA,1.0,5300,,No
1,0,2020,1,11,06:21,$70.00,Swipe Transaction,1,X,CA,1.0,5300,"Bad PIN,",No
1,0,2020,2,5,06:21,$10.00,Swipe Transaction,1,X,CA,1.0,5300,,No
2,0,2019,1,1,06:21,$999.00,Swipe Transaction,1,X,CA,1.0,5300,,No
"""


def test_rolling_30_day_spend(tmp_path):
    src = tmp_path / "cards.csv"
    src.write_text(CSV)
    counts = convert_cards(src, tmp_path / "out", date(2020, 1, 1), date(2020, 2, 10))
    assert counts["accounts"] == 1  # user 2 has no swipe in the window
    bal = {r["as_of_date"]: r["eod_balance"] for r in pq.read_table(tmp_path / "out" / "daily_balances.parquet").to_pylist()}
    assert bal[date(2020, 1, 1)] == 100
    assert bal[date(2020, 1, 5)] == 100  # declined swipe ignored
    assert bal[date(2020, 1, 10)] == 80  # refund subtracts
    assert bal[date(2020, 1, 31)] == -20  # the 1 Jan swipe left the 30-day window
    assert bal[date(2020, 2, 5)] == -10
    assert len(bal) == 41
    txns = pq.read_table(tmp_path / "out" / "transactions.parquet").to_pylist()
    assert sorted(r["status"] for r in txns) == ["POSTED", "POSTED", "POSTED", "RETURNED"]  # bad-PIN decline dropped


CARDS = """User,CARD INDEX,Card Brand,Card Type,Card Number,Expires,CVV,Has Chip,Cards Issued,Credit Limit,Acct Open Date,Year PIN last Changed,Card on Dark Web
1,0,Visa,Credit,4000,12/2022,123,YES,1,$1000,09/2015,2015,No
"""
USERS = """Person,Current Age,Retirement Age,Birth Year,Birth Month,Gender,Address,Apartment,City,State,Zipcode,Latitude,Longitude,Per Capita Income - Zipcode,Yearly Income - Person,Total Debt,FICO Score,Num Credit Cards
A,50,66,1970,1,Female,x,,y,NY,1,0,0,$1,$10000,$5000,800,1
B,40,66,1980,1,Male,x,,y,CA,1,0,0,$1,$20000,$60000,590,1
"""


def test_companion_files_add_limit_fico_and_segment(tmp_path):
    for name, text in (("cards.csv", CSV), ("sd254_cards.csv", CARDS), ("sd254_users.csv", USERS)):
        (tmp_path / name).write_text(text)
    out = tmp_path / "out"
    convert_cards(tmp_path / "cards.csv", out, date(2020, 1, 1), date(2020, 2, 10),
                  tmp_path / "sd254_cards.csv", tmp_path / "sd254_users.csv")
    acct = pq.read_table(out / "accounts.parquet").to_pylist()[0]
    assert acct["product_type"] == "CREDIT_CARD" and acct["credit_limit"] == 1000
    assert acct["open_date"] == date(2015, 9, 1) and acct["lgd"] == 0.75
    assert acct["rm_id"] == "ST-CA" and acct["segment"] == "DTI_OVER_2.5X"  # user 1 = second row: debt 3x income
    day = {r["as_of_date"]: r for r in pq.read_table(out / "daily_balances.parquet").to_pylist()}
    assert day[date(2020, 1, 1)]["credit_drawn"] == 100 and day[date(2020, 1, 31)]["credit_drawn"] == 0  # refunds never go below 0
    prof = pq.read_table(out / "credit_profiles.parquet").to_pylist()
    assert prof == [{"account_id": "CC0001_0", "pull_date": date(2020, 1, 1), "cibil_score": 590, "enquiries_6m": None}]
