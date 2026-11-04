"""Sandbox generator: synthetic, bank-like retail & SME portfolio.

The simulation is vectorised across accounts (one numpy op per day for the
whole chunk) and written chunk-by-chunk with ``ParquetWriter``, so the same
code produces the default 100,000-row ledger in seconds or a multi-GB
portfolio without holding it in memory.

Behaviour that is modelled:

* salaried credits on the payroll day (rolled back to the previous business
  day), irregular self-employed receipts and seasonal SME turnover,
* an amortising loan whose EMI is presented by NACH on the (next) business
  day; insufficient funds cause a payment return, a bounce charge and arrears,
* a revolving credit line (card / overdraft) that absorbs shortfalls, accrues
  interest and is repaid from surplus balance,
* monthly CIBIL pulls whose score drifts with utilisation, arrears and bounces,
* latent scenarios: healthy, gradual income stress, sudden income shock,
  shock-then-recovery, and deposit attrition (balances swept to another bank).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_engine import calendar as cal
from data_engine.schema import (
    ACCOUNTS,
    CREDIT_PROFILES,
    DAILY_BALANCES,
    HOLIDAYS,
    LOAN_SCHEDULES,
    TRANSACTIONS,
)
from utils.logging import logger

SEGMENTS = np.array(["RETAIL_SALARIED", "SELF_EMPLOYED", "SME_CURRENT"])
SCENARIOS = np.array(["HEALTHY", "GRADUAL_STRESS", "INCOME_SHOCK", "RECOVERY", "DEPOSIT_ATTRITION"])
HEALTHY, GRADUAL, SHOCK, RECOVERY, ATTRITION = range(5)

PRODUCTS = {
    # name: (tenure_months, annual_rate, lgd)
    "HOME": (240, 0.085, 0.22),
    "AUTO": (60, 0.095, 0.40),
    "PERSONAL": (48, 0.135, 0.65),
    "BUSINESS": (60, 0.15, 0.55),
    "MSME_TERM": (72, 0.12, 0.45),
}

TXN_TYPES = np.array(
    [
        "SALARY",
        "BUSINESS_RECEIPT",
        "TRANSFER_IN",
        "EMI",
        "EMI_ARREARS",
        "BOUNCE_CHARGE",
        "RENT",
        "UPI_POS",
        "CARD_BILL",
        "SUPPLIER_PAYMENT",
        "PAYROLL",
        "LINE_REPAYMENT",
        "NEFT_OUT",
    ]
)
_T = {name: i for i, name in enumerate(TXN_TYPES)}
CR, DR = 0, 1
POSTED, RETURNED = 0, 1
BOUNCE_CHARGE = 590.0  # INR 500 + GST


@dataclass
class SandboxConfig:
    n_accounts: int = 125
    n_days: int = 800  # 125 x 800 = 100,000 daily ledger rows
    end_date: date = date(2026, 10, 31)
    seed: int = 42
    burn_in_days: int = 60
    schedule_lead_days: int = 180
    chunk_accounts: int = 20_000

    @property
    def start_date(self) -> date:
        return self.end_date - timedelta(days=self.n_days - 1)


class _TxnBuffer:
    def __init__(self) -> None:
        self.parts: list[tuple[np.ndarray, ...]] = []

    def add(self, mask: np.ndarray, amount: np.ndarray, day: int, ttype: str, direction: int, status: int = POSTED):
        idx = np.flatnonzero(mask & (amount > 0.005))
        if idx.size == 0:
            return
        n = idx.size
        self.parts.append(
            (
                idx.astype(np.int32),
                np.full(n, day, np.int32),
                np.round(amount[idx], 2),
                np.full(n, _T[ttype], np.int8),
                np.full(n, direction, np.int8),
                np.full(n, status, np.int8),
            )
        )

    def add_rows(self, acct_idx: np.ndarray, amount: np.ndarray, day: int, ttype: str, direction: int):
        keep = amount > 0.005
        acct_idx, amount = acct_idx[keep], amount[keep]
        if acct_idx.size == 0:
            return
        n = acct_idx.size
        self.parts.append(
            (
                acct_idx.astype(np.int32),
                np.full(n, day, np.int32),
                np.round(amount, 2),
                np.full(n, _T[ttype], np.int8),
                np.full(n, direction, np.int8),
                np.full(n, POSTED, np.int8),
            )
        )

    def to_table(self, account_ids: np.ndarray, days: np.ndarray, id_prefix: str) -> pa.Table:
        if not self.parts:
            return TRANSACTIONS.arrow_schema.empty_table()
        acct, day, amt, typ, dirn, st = (np.concatenate(c) for c in zip(*self.parts, strict=True))
        order = np.lexsort((typ, acct, day))
        acct, day, amt, typ, dirn, st = (a[order] for a in (acct, day, amt, typ, dirn, st))
        seq = np.arange(acct.size)
        txn_id = np.char.add(id_prefix, np.char.zfill(seq.astype(str), 10))
        return pa.table(
            {
                "txn_id": pa.array(txn_id),
                "account_id": pa.array(account_ids[acct]),
                "txn_date": pa.array(days[day].astype("datetime64[D]")),
                "amount": pa.array(amt),
                "direction": pa.array(np.where(dirn == CR, "CR", "DR")),
                "txn_type": pa.array(TXN_TYPES[typ]),
                "status": pa.array(np.where(st == POSTED, "POSTED", "RETURNED")),
            },
            schema=TRANSACTIONS.arrow_schema,
        )


def _annuity_balance(principal, r, emi, k):
    """Outstanding principal after k payments of an amortising loan."""
    g = np.power(1 + r, k)
    return np.maximum(0.0, principal * g - emi * (g - 1) / r)


def _simulate_chunk(
    cfg: SandboxConfig, chunk_idx: int, first_account: int, n: int, holidays: np.ndarray
) -> dict[str, pa.Table]:
    rng = np.random.default_rng(cfg.seed * 1_000_003 + chunk_idx)
    burn = cfg.burn_in_days
    sim_start = np.datetime64(cfg.start_date, "D") - np.timedelta64(burn, "D")
    n_sim = cfg.n_days + burn
    days = sim_start + np.arange(n_sim)
    sched_end = np.datetime64(cfg.end_date, "D") + np.timedelta64(cfg.schedule_lead_days, "D")

    account_ids = np.char.add("AC", np.char.zfill((first_account + np.arange(n) + 1).astype(str), 8))
    customer_ids = np.char.add("CU", np.char.zfill((first_account + np.arange(n) + 1).astype(str), 8))

    # ---- static account attributes ----------------------------------------------------------
    seg = rng.choice(3, size=n, p=[0.6, 0.2, 0.2])
    salaried, selfemp, sme = seg == 0, seg == 1, seg == 2
    income = np.where(
        salaried,
        rng.lognormal(np.log(70_000), 0.45, n),
        np.where(selfemp, rng.lognormal(np.log(90_000), 0.55, n), rng.lognormal(np.log(900_000), 0.6, n)),
    )
    product = np.empty(n, dtype=object)
    product[salaried] = rng.choice(["HOME", "PERSONAL", "AUTO"], size=salaried.sum(), p=[0.3, 0.4, 0.3])
    product[selfemp] = rng.choice(["BUSINESS", "PERSONAL", "AUTO"], size=selfemp.sum(), p=[0.5, 0.25, 0.25])
    product[sme] = "MSME_TERM"
    tenure = np.array([PRODUCTS[p][0] for p in product])
    rate = np.array([PRODUCTS[p][1] for p in product])
    lgd = np.clip(np.array([PRODUCTS[p][2] for p in product]) + rng.normal(0, 0.03, n), 0.1, 0.9)
    foir = np.where(sme, rng.uniform(0.04, 0.10, n), rng.uniform(0.15, 0.38, n))
    emi = np.round(income * foir, -1)
    r_m = rate / 12
    principal = emi * (1 - np.power(1 + r_m, -tenure)) / r_m
    window_months = int(np.ceil(n_sim / 30.4)) + int(np.ceil(cfg.schedule_lead_days / 30.4)) + 2
    paid_before = rng.integers(0, np.maximum(1, np.minimum(tenure - window_months - 1, 120)))

    credit_limit = np.round(
        np.where(sme, income * rng.uniform(0.3, 0.8, n), income * rng.uniform(1.0, 3.0, n)), -3
    )
    rent = np.where(salaried & (rng.random(n) < 0.5), income * rng.uniform(0.12, 0.25, n), 0.0)
    spend_ratio = np.clip(
        rng.uniform(0.90, 0.99, n) - foir - np.where(sme, 0.08, rent / income), 0.15, None
    )
    card_share = np.where(sme, 0.0, rng.uniform(0.2, 0.45, n))
    salary_day = np.where(salaried, rng.choice([1, 1, 1, 5, 7, 10, 25, 28, 30, 31], size=n), -1)
    emi_day = rng.choice([2, 5, 5, 7, 10, 15], size=n)
    rent_day = rng.integers(1, 6, size=n)
    bill_day = (emi_day + 12) % 28 + 1
    rm_ids = np.char.add("RM", np.char.zfill(rng.integers(1, 13, n).astype(str), 3))

    scenario = rng.choice(5, size=n, p=[0.66, 0.12, 0.10, 0.05, 0.07])
    onset = np.where(
        scenario == RECOVERY,
        rng.integers(n_sim - 420, n_sim - 260, n),
        rng.integers(n_sim - 220, n_sim - 15, n),
    )
    onset = np.where(scenario == HEALTHY, n_sim + 10_000, onset)
    recover_at = np.where(scenario == RECOVERY, onset + rng.integers(90, 150, n), n_sim + 10_000)
    stressed_scn = (scenario == GRADUAL) | (scenario == SHOCK) | (scenario == RECOVERY)

    # ---- calendars ----------------------------------------------------------------------------
    sal_dates = cal.monthly_dates(salary_day, days[0], days[-1], holidays, roll=-1)
    emi_dates = cal.monthly_dates(emi_day, days[0], sched_end, holidays, roll=+1)
    rent_dates = cal.monthly_dates(rent_day, days[0], days[-1], holidays, roll=+1)
    bill_dates = cal.monthly_dates(bill_day, days[0], days[-1], holidays, roll=+1)

    def to_matrix(dates_2d: np.ndarray) -> np.ndarray:
        m = np.zeros((n, n_sim), dtype=bool)
        valid = ~np.isnat(dates_2d)
        rows, cols = np.nonzero(valid)
        di = (dates_2d[rows, cols] - days[0]).astype(np.int64)
        ok = (di >= 0) & (di < n_sim)
        m[rows[ok], di[ok]] = True
        return m

    is_salary = to_matrix(sal_dates) & salaried[:, None]
    is_emi = to_matrix(emi_dates)
    is_rent = to_matrix(rent_dates) & (rent > 0)[:, None]
    is_bill = to_matrix(bill_dates) & ~sme[:, None]

    wd = cal.weekday(days)
    dom = cal.day_of_month(days)
    month = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
    biz = ~cal.non_business_mask(days, holidays)
    is_hol = np.isin(days, holidays)
    seasonal = np.select([np.isin(month, [10, 11]), month == 3, np.isin(month, [7, 8])], [1.25, 1.15, 0.85], 1.0)

    # ---- state ----------------------------------------------------------------------------------
    balance = np.where(sme, income * rng.uniform(0.1, 0.4, n), income * rng.uniform(0.3, 2.0, n))
    drawn = credit_limit * rng.uniform(0.0, 0.25, n)
    overdue = np.zeros(n)
    dpd = np.zeros(n, dtype=np.int64)
    base_cibil = np.clip(rng.normal(748, 42, n), 560, 880)
    cibil = base_cibil.copy()
    last_bounce = np.full(n, -10_000)
    n_bounces = np.zeros(n, dtype=np.int64)
    salary = income.copy()

    bal_hist = np.zeros((n, cfg.n_days))
    drawn_hist = np.zeros((n, cfg.n_days))
    txns = _TxnBuffer()
    bureau_rows: list[tuple[np.ndarray, int, np.ndarray, np.ndarray]] = []

    for d in range(n_sim):
        rec = d - burn  # recorded day index (negative during burn-in)
        t_rel = d - onset
        after = t_rel >= 0
        prog = np.clip(t_rel / 180.0, 0.0, 1.0)
        income_mult = np.ones(n)
        spend_mult = np.ones(n)
        g = (scenario == GRADUAL) & after
        income_mult[g] = 1.0 - 0.7 * prog[g]
        spend_mult[g] = 1.0 - 0.15 * prog[g]
        s = ((scenario == SHOCK) & after) | ((scenario == RECOVERY) & after & (d < recover_at))
        income_mult[s] = np.where(salaried[s], 0.0, 0.25)
        spend_mult[s] = 0.75
        att = (scenario == ATTRITION) & after
        spend_mult[att] = 0.5  # spending migrates to the competitor bank too
        cur_income = np.where(salaried, salary, income)

        # month start: line interest, salary revision, bureau pull
        if dom[d] == 1:
            drawn = np.minimum(drawn * (1 + np.where(sme, 0.12, 0.36) / 12), credit_limit * 1.02)
            if month[d] == 4:
                salary = salary * (1 + rng.uniform(0.03, 0.10, n))
            util = drawn / np.maximum(credit_limit, 1)
            target = (
                base_cibil
                - 40 * (util > 0.5)
                - 60 * (util > 0.8)
                - 90 * (dpd > 0)
                - 60 * (dpd > 30)
                - 25 * ((d - last_bounce) < 60)
            )
            cibil = np.clip(cibil + 0.35 * (target - cibil) + rng.normal(0, 4, n), 300, 900)
            if rec >= 0:
                enq = rng.poisson(0.4, n) + 2 * (stressed_scn & after)
                bureau_rows.append((np.arange(n), rec, np.round(cibil).astype(np.int16), enq.astype(np.int16)))

        # ---- credits ----
        missed = g & (prog > 0.3) & (rng.random(n) < 0.3)
        sal_amt = np.where(is_salary[:, d] & ~missed, salary * income_mult * (1 + rng.normal(0, 0.01, n)), 0.0)
        cnt = rng.poisson(6 / 30.4, n)
        se_amt = np.where(
            selfemp & (cnt > 0),
            cnt * income / 6 * rng.lognormal(-0.08, 0.4, n) * seasonal[d] * income_mult,
            0.0,
        )
        sme_amt = np.where(
            sme & biz[d], income / 22 * rng.lognormal(-0.06, 0.35, n) * seasonal[d] * income_mult, 0.0
        )
        gig = np.where(s & (rng.random(n) < 0.12), income * rng.uniform(0.03, 0.07, n), 0.0)
        balance += sal_amt + se_amt + sme_amt + gig
        if rec >= 0:
            txns.add(sal_amt > 0, sal_amt, rec, "SALARY", CR)
            txns.add(se_amt > 0, se_amt, rec, "BUSINESS_RECEIPT", CR)
            txns.add(sme_amt > 0, sme_amt, rec, "BUSINESS_RECEIPT", CR)
            txns.add(gig > 0, gig, rec, "TRANSFER_IN", CR)

        # ---- EMI presentation (NACH) ----
        due = is_emi[:, d]
        pay = due & (balance >= emi)
        bounce = due & ~pay
        balance = balance - np.where(pay, emi, 0.0)
        overdue = overdue + np.where(bounce, emi, 0.0)
        charge = bounce & (balance >= BOUNCE_CHARGE)
        balance = balance - np.where(charge, BOUNCE_CHARGE, 0.0)
        last_bounce = np.where(bounce, d, last_bounce)
        n_bounces = n_bounces + (bounce & (rec >= cfg.n_days - 90))
        if rec >= 0:
            txns.add(pay, emi, rec, "EMI", DR)
            txns.add(bounce, emi, rec, "EMI", DR, RETURNED)
            txns.add(charge, np.full(n, BOUNCE_CHARGE), rec, "BOUNCE_CHARGE", DR)
        cure = (overdue > 0) & ~due & (balance >= overdue + 0.25 * emi)
        cured_amt = np.where(cure, overdue, 0.0)
        balance -= cured_amt
        overdue = np.where(cure, 0.0, overdue)
        dpd = np.where(overdue > 0, dpd + 1, 0)
        if rec >= 0:
            txns.add(cure, cured_amt, rec, "EMI_ARREARS", DR)

        # ---- spending / obligations (credit line absorbs shortfalls) ----
        weekend = 1.2 if wd[d] >= 5 else 1.0
        fest = 1.5 if is_hol[d] else 1.0
        retail = ~sme
        k = rng.poisson(1.6 * weekend, n) * retail
        total_k = int(k.sum())
        spend_acct_idx = np.repeat(np.arange(n), k)
        daily_mean = cur_income * spend_ratio / 30.4 * spend_mult
        per_txn_mean = daily_mean / 1.6
        amts = rng.lognormal(np.log(np.maximum(per_txn_mean[spend_acct_idx], 1.0)) - 0.32, 0.8, total_k) * fest
        spend_total = np.bincount(spend_acct_idx, weights=amts, minlength=n)

        card_part = spend_total * card_share
        card_posted = np.minimum(card_part, np.maximum(credit_limit - drawn, 0))
        drawn += card_posted
        acct_spend = spend_total - card_posted  # declined card spend falls back to the account

        rent_due = np.where(is_rent[:, d], rent * cur_income / income, 0.0)
        supplier = np.where(
            sme & biz[d], income * spend_ratio / 22 * rng.lognormal(-0.045, 0.3, n) * spend_mult, 0.0
        )
        payroll = np.where(sme & (dom[d] == 1), income * 0.08, 0.0)
        need = acct_spend + rent_due + supplier + payroll
        from_bal = np.minimum(need, balance)
        line_draw = np.minimum(need - from_bal, np.maximum(credit_limit - drawn, 0))
        balance -= from_bal
        drawn += line_draw
        posted_ratio = np.where(need > 0, (from_bal + line_draw) / np.maximum(need, 1e-9), 0.0)
        if rec >= 0:
            acct_ratio = np.where(spend_total > 0, acct_spend / np.maximum(spend_total, 1e-9), 0.0)
            txns.add_rows(
                spend_acct_idx,
                amts * acct_ratio[spend_acct_idx] * posted_ratio[spend_acct_idx],
                rec,
                "UPI_POS",
                DR,
            )
            txns.add(rent_due > 0, rent_due * posted_ratio, rec, "RENT", DR)
            txns.add(supplier > 0, supplier * posted_ratio, rec, "SUPPLIER_PAYMENT", DR)
            txns.add(payroll > 0, payroll * posted_ratio, rec, "PAYROLL", DR)

        # ---- credit line repayment ----
        reserve = 0.1 * income
        bill = is_bill[:, d]
        bill_pay = np.where(bill, np.minimum(drawn, np.maximum(balance - reserve, 0)), 0.0)
        sweep_buf = 0.15 * income
        od_pay = np.where(sme & (drawn > 0), np.minimum(drawn, np.maximum(balance - sweep_buf, 0) * 0.6), 0.0)
        balance -= bill_pay + od_pay
        drawn -= bill_pay + od_pay
        if rec >= 0:
            txns.add(bill_pay > 0, bill_pay, rec, "CARD_BILL", DR)
            txns.add(od_pay > 0, od_pay, rec, "LINE_REPAYMENT", DR)

        # ---- deposit attrition: weekly sweep to another bank ----
        if wd[d] == 4:
            floor = 1.5 * emi + 0.25 * cur_income
            sweep = np.where(att, np.maximum(balance - floor, 0) * 0.5, 0.0)
            balance -= sweep
            if rec >= 0:
                txns.add(sweep > 0, sweep, rec, "NEFT_OUT", DR)

        balance = np.maximum(balance, 0.0)
        drawn = np.clip(drawn, 0.0, None)
        if rec >= 0:
            bal_hist[:, rec] = balance
            drawn_hist[:, rec] = drawn

    # ---- assemble tables ----------------------------------------------------------------------
    rec_days = days[burn:]
    open_date = (rec_days[0] - rng.integers(30, 2000, n).astype("timedelta64[D]")).astype("datetime64[D]")

    # Loan schedule from the first recorded day through the lead window.
    rows, cols = np.nonzero(~np.isnat(emi_dates) & (emi_dates >= rec_days[0]))
    first_col = np.full(n, emi_dates.shape[1])
    np.minimum.at(first_col, rows, cols)
    inst_no = paid_before[rows] + (cols - first_col[rows]) + 1
    keep = inst_no <= tenure[rows]
    sched = pa.table(
        {
            "account_id": pa.array(account_ids[rows[keep]]),
            "due_date": pa.array(emi_dates[rows[keep], cols[keep]].astype("datetime64[D]")),
            "emi_amount": pa.array(emi[rows[keep]]),
            "installment_no": pa.array(inst_no[keep].astype(np.int32)),
        },
        schema=LOAN_SCHEDULES.arrow_schema,
    )
    paid_by_end = paid_before + np.array(
        [int(((emi_dates[i] >= rec_days[0]) & (emi_dates[i] <= rec_days[-1])).sum()) for i in range(n)]
    )
    outstanding = _annuity_balance(principal, r_m, emi, paid_by_end) + overdue

    accounts = pa.table(
        {
            "account_id": pa.array(account_ids),
            "customer_id": pa.array(customer_ids),
            "rm_id": pa.array(rm_ids),
            "segment": pa.array(SEGMENTS[seg]),
            "product_type": pa.array(product.astype(str)),
            "open_date": pa.array(open_date),
            "sanctioned_amount": pa.array(np.round(principal, 2)),
            "outstanding_principal": pa.array(np.round(outstanding, 2)),
            "emi_amount": pa.array(emi),
            "emi_day": pa.array(emi_day.astype(np.int16)),
            "salary_day": pa.array(np.where(salary_day > 0, salary_day, 0).astype(np.int16), mask=salary_day <= 0),
            "credit_limit": pa.array(credit_limit),
            "lgd": pa.array(np.round(lgd, 4)),
            "interest_rate": pa.array(rate),
        },
        schema=ACCOUNTS.arrow_schema,
    )

    daily = pa.table(
        {
            "account_id": pa.array(np.repeat(account_ids, cfg.n_days)),
            "as_of_date": pa.array(np.tile(rec_days, n).astype("datetime64[D]")),
            "eod_balance": pa.array(np.round(bal_hist, 2).ravel()),
            "credit_limit": pa.array(np.repeat(credit_limit, cfg.n_days)),
            "credit_drawn": pa.array(np.round(drawn_hist, 2).ravel()),
        },
        schema=DAILY_BALANCES.arrow_schema,
    )

    if bureau_rows:
        b_idx = np.concatenate([r[0] for r in bureau_rows])
        b_day = np.concatenate([np.full(n, r[1]) for r in bureau_rows])
        b_score = np.concatenate([r[2] for r in bureau_rows])
        b_enq = np.concatenate([r[3] for r in bureau_rows])
    else:
        b_idx = b_day = b_score = b_enq = np.zeros(0, dtype=np.int64)
    bureau = pa.table(
        {
            "account_id": pa.array(account_ids[b_idx]),
            "pull_date": pa.array(rec_days[b_day].astype("datetime64[D]")),
            "cibil_score": pa.array(b_score.astype(np.int16)),
            "enquiries_6m": pa.array(b_enq.astype(np.int16)),
        },
        schema=CREDIT_PROFILES.arrow_schema,
    )

    onset_rec = onset - burn
    labels = pa.table(
        {
            "account_id": pa.array(account_ids),
            "scenario": pa.array(SCENARIOS[scenario]),
            "onset_date": pa.array(
                np.where(scenario == HEALTHY, np.datetime64("NaT", "D"), rec_days[0] + onset_rec.astype("timedelta64[D]"))
            ),
            "dpd_at_end": pa.array(dpd),
            "bounces_last_90d": pa.array(n_bounces),
        }
    )

    return {
        "accounts": accounts,
        "daily_balances": daily,
        "transactions": txns.to_table(account_ids, rec_days, f"TX{chunk_idx:03d}"),
        "credit_profiles": bureau,
        "loan_schedules": sched,
        "labels": labels,
    }


def generate_sandbox(cfg: SandboxConfig, out_dir: Path) -> dict[str, int]:
    """Generate the sandbox portfolio as Parquet files in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sched_end = cfg.end_date + timedelta(days=cfg.schedule_lead_days)
    hol = cal.bank_holidays(cfg.start_date - timedelta(days=cfg.burn_in_days + 31), sched_end)
    holidays = np.array([d for d, _ in hol], dtype="datetime64[D]")

    writers: dict[str, pq.ParquetWriter] = {}
    counts: dict[str, int] = {}
    try:
        for ci, first in enumerate(range(0, cfg.n_accounts, cfg.chunk_accounts)):
            n = min(cfg.chunk_accounts, cfg.n_accounts - first)
            tables = _simulate_chunk(cfg, ci, first, n, holidays)
            for name, tbl in tables.items():
                if name not in writers:
                    writers[name] = pq.ParquetWriter(out_dir / f"{name}.parquet", tbl.schema, compression="zstd")
                writers[name].write_table(tbl, row_group_size=262_144)
                counts[name] = counts.get(name, 0) + tbl.num_rows
            logger.info("sandbox chunk {} done ({} accounts)", ci, n)
    finally:
        for w in writers.values():
            w.close()

    hol_tbl = pa.table(
        {"holiday_date": pa.array([d for d, _ in hol], type=pa.date32()), "name": pa.array([nm for _, nm in hol])},
        schema=HOLIDAYS.arrow_schema,
    )
    pq.write_table(hol_tbl, out_dir / "holidays.parquet")
    counts["holidays"] = hol_tbl.num_rows
    (out_dir / "sandbox_config.json").write_text(
        __import__("json").dumps({**asdict(cfg), "end_date": cfg.end_date.isoformat(), "rows": counts}, indent=2)
    )
    return counts
