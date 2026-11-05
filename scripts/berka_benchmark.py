"""Real-data benchmark on the Berka / PKDD'99 Czech bank dataset.

    git clone --depth 1 https://github.com/jlacko/berka-dataset ../berka-dataset
    python scripts/berka_benchmark.py --raw ../berka-dataset

The book is scored as of ``--as-of`` (default 1998-10-02, 90 days before the
data ends) and compared with what really happened over the next 90 days.

Forecast accuracy (EOD balance, all accounts open for 90+ days):
    weighted quantile loss (wQL, mean over q10..q90), WAPE of the median and
    q10-q90 coverage, against persistence, 30-day seasonal-naive and
    per-account empirical-quantile baselines.

Credit early warning (loans with an instalment due in the window):
    SMA flag and PD against real outcomes: a scheduled loan payment that did
    not arrive, a negative balance, or the bank's own "in debt"/"not paid"
    status, against naive rules that use only today's data.

Models: the sandbox-trained forecaster used zero-shot, and the same network
fine-tuned on Berka windows whose targets all end on or before the as-of date
(strictly out-of-time; a 15 % account holdout is reported separately).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import _bootstrap_path  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from config.settings import PROJECT_ROOT, QUANTILES, CacheSettings, ModelSettings, Settings, StorageSettings
from data_engine.adapters.berka import convert_berka
from data_engine.ingestion import IngestionEngine
from data_engine.storage import get_storage
from models.dataset import holdout_split
from models.engine import STFormerEngine
from models.pipeline import EWSPipeline
from models.train import train_model
from utils.device import get_device_context
from utils.logging import logger, setup_logging

LOAD_ORDER = ("holidays", "accounts", "loan_schedules", "credit_profiles", "daily_balances", "transactions")
TAUS = np.asarray(QUANTILES)
H = 90


# ---------------------------------------------------------------------------- forecast metrics
def forecast_metrics(y: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """y [N, H] actual (NaN ignored), q [N, H, 9] quantile forecasts."""
    m = ~np.isnan(y)
    yy = np.nan_to_num(y)
    err = yy[..., None] - q
    pin = np.maximum(TAUS * err, (TAUS - 1) * err) * m[..., None]
    denom = np.abs(yy * m).sum()
    return {
        "wql": round(float(2 * pin.sum(axis=(0, 1)).mean() / denom), 4),
        "wape_median": round(float((np.abs(yy - q[..., 4]) * m).sum() / denom), 4),
        "coverage_10_90": round(float((((yy >= q[..., 0]) & (yy <= q[..., -1])) & m).sum() / m.sum()), 4),
    }


def baseline_forecasts(past: np.ndarray) -> dict[str, np.ndarray]:
    """past [N, L] balances up to the as-of day (last column)."""
    n, L = past.shape
    last = pd.DataFrame(past.T).ffill().to_numpy().T[:, -1]
    persistence = np.repeat(last[:, None, None], H, 1).repeat(len(TAUS), 2)
    h = np.arange(1, H + 1)
    idx = L - 1 + h - 30 * np.ceil(h / 30).astype(int)
    seasonal = np.repeat(past[:, idx][..., None], len(TAUS), 2)
    seasonal = np.where(np.isnan(seasonal), persistence, seasonal)
    clim = np.nanquantile(past[:, -180:], TAUS, axis=1).T  # [N, 9]
    clim = np.where(np.isnan(clim), last[:, None], clim)
    return {
        "persistence": persistence,
        "seasonal_naive_30d": seasonal,
        "empirical_quantiles_180d": np.repeat(clim[:, None, :], H, 1),
    }


# ---------------------------------------------------------------------------- credit outcomes
def credit_outcomes(storage, cache, raw: Path, as_of: date) -> pd.DataFrame:
    loans = pq.read_table(raw / "loans.parquet").to_pandas()
    end = as_of + timedelta(days=H)
    due = storage.query_df(
        "SELECT account_id, COUNT(*) AS due_n FROM loan_schedules "
        "WHERE due_date > CAST(:a AS DATE) AND due_date <= CAST(:b AS DATE) GROUP BY account_id",
        {"a": as_of, "b": end},
    )
    paid = storage.query_df(
        "SELECT account_id, COUNT(*) AS paid_n FROM transactions WHERE txn_type = 'EMI' "
        "AND txn_date > CAST(:a AS DATE) AND txn_date <= CAST(:b AS DATE) GROUP BY account_id",
        {"a": as_of - timedelta(days=3), "b": end + timedelta(days=3)},
    )
    df = loans.merge(due, on="account_id").merge(paid, on="account_id", how="left").fillna({"paid_n": 0})
    df = df[pd.to_datetime(df.loan_date) <= pd.Timestamp(as_of - timedelta(days=30))]
    rows, _ = cache.rows_for(df.account_id.tolist())
    t0 = cache.day_index(as_of)
    fut = np.asarray(cache.past[rows, t0 + 1 : t0 + 1 + H, 0])
    hist = np.asarray(cache.past[rows, t0 - 89 : t0 + 1, 0])
    df = df.assign(
        row=rows,
        missed_payment=(df.paid_n < df.due_n).to_numpy(),
        negative_balance=(np.nanmin(fut, axis=1) < 0),
        bank_status_bad=df.status.isin(["B", "D"]).to_numpy(),
        past_negative_90d=(np.nanmin(hist, axis=1) < 0),
        past_min_balance_90d=np.nanmin(hist, axis=1),
    )
    df["any_bad_outcome"] = df.missed_payment | df.negative_balance | df.bank_status_bad
    return df.reset_index(drop=True)


def classification(flag: np.ndarray, score: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
    tp = int((flag & truth).sum())
    fp = int((flag & ~truth).sum())
    fn = int((~flag & truth).sum())
    auc = float(roc_auc_score(truth, score)) if 0 < truth.sum() < len(truth) else None
    return {
        "positives": int(truth.sum()),
        "flagged": int(flag.sum()),
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(tp + fn, 1), 3),
        "auc": round(auc, 3) if auc is not None else None,
    }


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="directory with the Berka *.asc files")
    ap.add_argument("--work", type=Path, default=PROJECT_ROOT / "data" / "berka")
    ap.add_argument("--as-of", type=date.fromisoformat, default=date(1998, 10, 2))
    ap.add_argument("--zero-shot-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/models/ews_stformer.pt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=250)
    ap.add_argument("--skip-train", action="store_true", help="reuse artifacts/models/berka_stformer.pt")
    args = ap.parse_args()
    setup_logging("INFO")
    t_start = time.perf_counter()

    work = args.work
    parquet = work / "parquet"
    if not (parquet / "loans.parquet").exists():
        logger.info("converting raw Berka files: {}", convert_berka(args.raw, parquet))
    model_cfg = ModelSettings(
        checkpoint_path=PROJECT_ROOT / "artifacts/models/berka_stformer.pt",
        d_model=96, n_layers=2, ff_mult=2, n_heads=4, window_stride=7, num_workers=0,
        epochs=args.epochs,
    )
    settings = Settings(
        data_dir=work,
        warm_cache_on_startup=False,
        storage=StorageSettings(duckdb_path=work / "ews.duckdb", duckdb_memory_limit="6GB"),
        cache=CacheSettings(root=work / "cache"),
        model=model_cfg,
    )
    storage = get_storage(settings)
    if not storage.scalar("SELECT COUNT(*) FROM daily_balances"):
        ing = IngestionEngine(storage)
        for t in LOAD_ORDER:
            ing.ingest_file(t, parquet / f"{t}.parquet", "parquet")
    base = EWSPipeline(settings, storage=storage, engine=object())  # engine set per model below
    base.refresh(full=not base.cache.load(), force_cache=not base.cache.load())
    cache = base.cache.load()
    t0 = cache.day_index(args.as_of)
    logger.info("cache: {} accounts x {} days ({} .. {})", cache.n_accounts, cache.meta.n_days, cache.start, cache.end)

    # ---- models ---------------------------------------------------------------------------------------
    ctx = get_device_context(model_cfg.amp_dtype, model_cfg.cuda_memory_fraction)
    engines = {"stformer_sandbox_zero_shot": STFormerEngine(args.zero_shot_checkpoint, ctx)}
    train_info = None
    if not args.skip_train:
        train_info = train_model(
            base.cache.current_dir(), model_cfg, max_steps_per_epoch=args.steps_per_epoch,
            max_target_day=t0, init_checkpoint=args.zero_shot_checkpoint,
        )
    engines["stformer_berka_finetuned"] = STFormerEngine(model_cfg.checkpoint_path, ctx)

    # ---- forecast accuracy -----------------------------------------------------------------------------
    all_rows = np.arange(cache.n_accounts)
    open_long = ~np.isnan(np.asarray(cache.past[:, t0 - 90, 0]))
    rows = all_rows[open_long]
    val_rows, _ = holdout_split(cache.n_accounts, model_cfg.val_fraction, model_cfg.seed)
    is_holdout = np.isin(rows, val_rows)
    actual = np.asarray(cache.past[rows, t0 + 1 : t0 + 1 + H, 0])
    past_bal = np.asarray(cache.past[rows, t0 - 511 : t0 + 1, 0])

    forecasts = {k: v for k, v in baseline_forecasts(past_bal).items()}
    scored = {}
    for name, eng in engines.items():
        pipe = EWSPipeline(settings, storage=storage, engine=eng)
        sb = pipe.score_rows(cache, all_rows, args.as_of, H, 0.1, 0.9)
        scored[name] = (pipe, sb)
        forecasts[name] = sb.q[rows, 0]
    accuracy = {
        name: {
            "all_accounts": forecast_metrics(actual, q),
            "holdout_accounts": forecast_metrics(actual[is_holdout], q[is_holdout]),
        }
        for name, q in forecasts.items()
    }

    # ---- credit early warning -----------------------------------------------------------------------------
    out = credit_outcomes(storage, cache, parquet, args.as_of)
    labels = ["missed_payment", "negative_balance", "bank_status_bad", "any_bad_outcome"]
    orow = out.row.to_numpy()
    ews: dict[str, dict] = {}
    for name, (_, sb) in scored.items():
        flag = sb.risk.tier[orow] != "GREEN"
        pd_ = sb.risk.pd[orow]
        ews[name] = {lab: classification(flag, pd_, out[lab].to_numpy()) for lab in labels}
        ews[name]["tier_counts"] = {t: int((sb.risk.tier[orow] == t).sum()) for t in ("GREEN", "SMA-0", "SMA-1", "SMA-2")}
    emi_now = out.payments.to_numpy()
    cur = np.asarray(cache.past[orow, t0, 0])
    rules = {
        "rule_balance_below_next_emi": (cur < emi_now, -(cur - emi_now)),
        "rule_negative_balance_last_90d": (out.past_negative_90d.to_numpy(), -out.past_min_balance_90d.to_numpy()),
    }
    for name, (flag, score) in rules.items():
        ews[name] = {lab: classification(flag, score, out[lab].to_numpy()) for lab in labels}

    # ---- example: highest-priority flagged loan -----------------------------------------------------------
    pipe, sb = scored["stformer_berka_finetuned"]
    pr = sb.risk.priority[orow]
    i = int(np.argmax(pr))
    r = int(orow[i])
    acc_id = out.account_id[i]
    ex_days = [1, 10, 30, 60, 90]
    example = {
        "account_id": acc_id,
        "loan_status_at_end_of_data": out.status[i],
        "monthly_instalment": float(out.payments[i]),
        "balance_on_as_of": round(float(cache.past[r, t0, 0]), 2),
        "risk_tier": sb.risk.tier[r],
        "days_to_breach": int(sb.risk.days_to_breach[r]),
        "probability_of_default": round(float(sb.risk.pd[r]), 3),
        "expected_loss": round(float(np.nan_to_num(sb.risk.expected_loss[r])), 2),
        "realised": {
            "missed_payment": bool(out.missed_payment[i]),
            "negative_balance": bool(out.negative_balance[i]),
        },
        "path": [
            {
                "day": d,
                "date": (args.as_of + timedelta(days=d)).isoformat(),
                "q10": round(float(sb.q[r, 0, d - 1, 0]), 0),
                "q50": round(float(sb.q[r, 0, d - 1, 4]), 0),
                "q90": round(float(sb.q[r, 0, d - 1, -1]), 0),
                "actual": round(float(cache.past[r, t0 + d, 0]), 0),
            }
            for d in ex_days
        ],
    }

    report = {
        "dataset": "PKDD'99 Berka (Czech bank, 1993-1998)",
        "as_of_date": args.as_of.isoformat(),
        "horizon_days": H,
        "accounts_in_cache": cache.n_accounts,
        "daily_balance_rows": int(storage.scalar("SELECT COUNT(*) FROM daily_balances")),
        "transactions": int(storage.scalar("SELECT COUNT(*) FROM transactions")),
        "forecast_population": {"accounts": int(len(rows)), "holdout_accounts": int(is_holdout.sum())},
        "credit_population": {"loans": int(len(out)), "status_counts": out.status.value_counts().to_dict()},
        "training": train_info.__dict__ if train_info else "skipped",
        "forecast_accuracy_balance": accuracy,
        "early_warning": ews,
        "example_top_priority_loan": example,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    work.mkdir(parents=True, exist_ok=True)
    (work / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
