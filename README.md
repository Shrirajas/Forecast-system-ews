# Banking EWS & Loan Loss Predictor

An Early Warning System (EWS) for retail and SME lending. It forecasts every
account's end-of-day balance and credit-line utilisation 30/60/90 days ahead
as 9 quantile tracks (q10–q90). It then classifies accounts into SMA-0/1/2
when the worst-case cash track falls short of a scheduled EMI, estimates
**Expected Loss = EAD × PD × LGD**, and streams ranked alerts to relationship
managers through a FastAPI service.

```
                     [ INPUT DATA SOURCE ]
        daily ledgers · EOD balances · bureau pulls · EMI schedules
                               │  /api/v1/ingest (JSON · streamed files · datasets)
                               ▼
                  [ HIGH-PERFORMANCE DATA LAYER ]
   DuckDB / PostgreSQL ─ incremental window-function aggregation (data_engine/aggregation.py)
   └─ memory-mapped tensor cache [accounts × days × channels] (data_engine/vector_cache.py)
                               ▼
              [ MULTIVARIATE FORECASTING ENGINE ]
   stformer : spatial-temporal patch transformer (models/stformer.py)  ← trained in-repo
   timesfm3 : pretrained google/timesfm-3.0-pytorch with covariates    ← optional, zero-shot
                               │
         ┌─────────────────────┴──────────────────────┐
         ▼                                            ▼
 [ PREDICTIVE RISK ENGINE ]                 [ CUSTOMER BEHAVIOUR ENGINE ]
  SMA-0/1/2 · PD · EAD · LGD · EL            balance trajectories · deposit-attrition
  (models/risk.py)                           & cost-of-funds alerts (models/behavior.py)
                               ▼
                     [ CENTRAL API GATEWAY ]
   FastAPI (api/) · /api/v1/predict/ews · /api/v1/alerts/active (JSON + SSE)
```

## Quick start

Step-by-step setup for your own PC (Windows/macOS/Linux, GPU or CPU): [`docs/RUNNING_LOCALLY.md`](docs/RUNNING_LOCALLY.md).
Runbook to hand to an AI coding agent: [`AGENTS.md`](AGENTS.md).

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/bootstrap.py      # sandbox → DuckDB → features → cache → train → score
python scripts/serve.py          # http://localhost:8000/docs
```

`bootstrap.py` generates the 100,000-row sandbox (125 accounts × 800 days),
loads it, trains the forecaster, scores the book and prints the portfolio
summary, the top alerts and an out-of-time backtest. On a CPU-only machine
it switches to a lighter model profile, so training takes a few minutes.

```bash
curl -s localhost:8000/api/v1/predict/ews -H 'content-type: application/json' \
     -d '{"horizon": 90, "quantile_triggers": {"liability": 0.1, "asset": 0.9}, "tiers": ["SMA-0","SMA-1","SMA-2"]}'
curl -s 'localhost:8000/api/v1/alerts/active?limit=20'
curl -N 'localhost:8000/api/v1/alerts/active?stream=true&follow=true'   # Server-Sent Events
```

## Layout

| Path | Contents |
|---|---|
| `config/` | `settings.py`: pydantic-settings; every value overridable via `EWS_*` env vars |
| `data_engine/` | `schema.py` (DDL + Arrow schemas), `storage.py` (DuckDB / Postgres adapters), `adapters/berka.py` (public Czech bank dataset), `ingestion.py` (streaming loads), `aggregation.py` (incremental feature matrix), `vector_cache.py` (memmap tensors), `calendar.py` (RBI business-day rules), `synth.py` (sandbox generator), `alerts.py` (run/alert persistence) |
| `models/` | `stformer.py` (network), `dataset.py`, `train.py`, `engine.py` (stformer / TimesFM 3 backends), `risk.py`, `behavior.py`, `pipeline.py` (end-to-end scoring), `backtest.py`, `tuning.py` (Optuna), `distributed.py` (Ray) |
| `api/` | `main.py` (app + lifespan), `schemas.py` (Pydantic contracts), `routes/` |
| `utils/` | `device.py` (CUDA/AMP/OOM handling), `streaming.py`, `logging.py` |
| `scripts/` | `bootstrap.py`, `generate_sandbox.py`, `train.py`, `serve.py`, `backtest.py`, `berka_benchmark.py`, `tune.py`, `batch_score_ray.py` |
| `tests/` | pytest suite (data engine, model, risk maths, API contracts) |

## Data layer

**Tables:** `accounts`, `daily_balances`, `transactions`, `credit_profiles`,
`loan_schedules`, `holidays`, plus the derived `feature_matrix` and the
`forecast_runs` / `ews_alerts` outputs.

**Incremental aggregation.** The nightly job recomputes only the dates from
`min(watermark + 1, earliest late-arriving event)` up to the newest balance
date. It runs entirely in SQL window functions, with no Python row loops,
and produces per account-day:

| Group | Columns |
|---|---|
| Targets | `eod_balance`, `utilization` (drawn / limit) |
| Past covariates | forward-filled `cibil_score`, 30-day rolling std of outward payments, `missed_salary_flag` (no payroll credit in 35 days), `payment_return_flag` (a return in the last 30 days) |
| Future covariates (cache) | `emi_due_amount` from the loan schedule, `holiday_flag` (Sundays, 2nd/4th Saturdays, bank holidays), `salary_day_flag` (payroll day rolled to the previous business day; inferred from credits when master data lacks it) |

Rolling windows read a fixed 60-day lookback, so an incremental pass gives
exactly the same rows as a full rebuild. The test suite checks this.

**Vector cache.** `past.npy` `[accounts, days, 6]` and `known.npy`
`[accounts, days + 96, 3]` are written with `numpy.lib.format.open_memmap`.
Account shards fill in parallel threads, each streaming its own Arrow record
batches from the database. Readers memory-map the arrays read-only and slice
only the accounts they need, so the book can be much larger than RAM.
Versions are swapped through an atomic `CURRENT` pointer.

**Ingestion** never builds per-row Python objects for files.
`/api/v1/ingest/file` streams CSV, Parquet or Arrow IPC through PyArrow's
multi-threaded readers in bounded batches. A background prefetch thread
keeps parsing ahead of the database writer, and each batch is upserted
zero-copy (`INSERT OR REPLACE` in DuckDB; `COPY` into staging then
`ON CONFLICT` in Postgres). `/api/v1/ingest/dataset` scans multi-file
datasets already on the server, such as terabyte backfills.

## Forecasting engine (`models/stformer.py`)

* **Patching:** each daily series is cut into 32-day patches and embedded
  with a per-variate residual MLP over (values, observed-mask). 512 context
  days give 16 patch tokens per variate.
* **Token grid:** 9 variates × (16 past + 3 future) patches. Past variates
  get learned forecast-query tokens at future positions. Known covariates
  carry real values across the horizon.
* **Temporal attention:** within each variate across time (long-run
  behaviour, seasonality, payment cadence).
* **Spatial attention:** across variates at each time patch (utilisation
  against balance, EMI date against payroll day).
* **Output:** 9 non-crossing quantiles (median + cumulative softplus
  offsets) per day for balance and utilisation, trained with pinball loss.
* **Reversible instance normalisation:** inputs are normalised per series,
  so one model serves ₹10k savings accounts and ₹10cr SME accounts.
* **Device-agnostic:** CUDA + bf16/fp16 autocast + `GradScaler`,
  `pin_memory=True` loaders with non-blocking copies, gradient accumulation,
  a capped VRAM fraction, `expandable_segments`, and automatic batch halving
  on `torch.cuda.OutOfMemoryError`. Falls back to fp32 on CPU.

### Pretrained TimesFM 3.0 backend

`EWS_MODEL__BACKEND=timesfm3` uses Google's `google/timesfm-3.0-pytorch`
through the official `timesfm3` API. The two targets are passed as a
multivariate context, the four past covariates as `past_only_covariates`
and the three calendar/contract covariates as `past_future_covariates`.
The rest of the pipeline is unchanged. The checkpoint downloads from
Hugging Face on first use.

### RTX 4080 Super (16 GB) notes

* Install the CUDA wheel: `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
* The default network (`d_model=128`, 3 layers, ~1.2M parameters, batch 64)
  is small next to 16 GB. For larger models raise `--batch-size` and use
  `--accum` rather than pushing one batch to the limit, e.g.
  `python scripts/train.py --d-model 256 --layers 6 --batch-size 128 --accum 2 --compile`.
* Ada GPUs support bf16, so `amp_dtype=auto` picks bf16 with no loss
  scaling; `--amp fp16` enables `GradScaler`.
* On Windows, DataLoader workers use `spawn`. The datasets re-open their
  memmaps per worker, so nothing large gets pickled.

## Risk engine (`models/risk.py`)

| Item | Definition |
|---|---|
| Trigger | On the liability track (default q10), the balance available going into each EMI date (previous day's EOD) is compared with the EMI due. The first shortfall within the horizon is the breach. |
| SMA-0 / SMA-1 / SMA-2 | Breach 1–30 / 31–60 / 61–90 days out from the as-of date (limited to the requested horizon). |
| PD | Noisy-OR of (a) P(balance < EMI) from the full 9-quantile predictive distribution with exponential tails, max over due dates; (b) P(utilisation > 95 %) on any day; (c) a logistic bureau PD from CIBIL. Weights are in `RiskSettings`. |
| EAD | Outstanding principal (including arrears) + max(current drawn, asset-track q90 projected drawn at the breach day). |
| LGD | Account / product LGD from the master data. |
| Expected loss | EAD × PD × LGD, computed for SMA accounts. |
| Alert ranking | Credit EWS first, then `priority = EL × 0.5^(days / 30)` with days = min(days to zero, days to breach), then EAD, then days to zero. |

Note: the RBI's SMA buckets count **days past due** on an existing default.
Here, as specified, the tiers bucket **how far out the projected EMI
shortfall is**. An SMA-0 account in this system is therefore the most
imminent case, and ranking puts it first through the urgency weight.

**Deposit attrition** (`models/behavior.py`): an account is flagged as
attrition when its median track projects the horizon's average balance at
least 35 % below its established balance, and it shows no stress evidence
(no missed salary and no recent payment return). The established balance is
the average over days −180..−30, so an exit already under way is still
measured against the pre-exit level. The alert carries the annual
cost-of-funds impact, `drop × replacement-funding spread`. It can coexist
with an SMA alert, because moving money out can also leave the next EMI
underfunded. Expected loss is recorded only on the credit alert, so totals
never double count.

### Backtest

`python scripts/backtest.py [--as-of YYYY-MM-DD] [--horizon 90] [--liability-q 0.2]`
scores the book as of an earlier date and compares the result with what
happened next: precision and recall of the SMA flag against actual EMI
returns, PD ROC-AUC and Brier score, breach-timing error, and attrition
precision and recall against realised balance outflows. A higher liability
trigger quantile (q20/q30) trades a little recall for much higher precision.

## Real-data benchmark: Berka / PKDD'99 Czech bank

`scripts/berka_benchmark.py` runs the whole stack on the public PKDD'99
"financial" dataset: real, anonymised data from a Czech bank, 1993–1998,
with 4,500 accounts, 1.06M transactions and 682 loans with outcomes.
`data_engine/adapters/berka.py` maps it into the ingestion schema. The
end-of-day balance is the bank's reported balance after the day's last
transaction (5.59M account-days). Loan payments become EMIs, and pension
and inbound transfers become income.

```bash
git clone --depth 1 https://github.com/jlacko/berka-dataset ../berka-dataset
python scripts/berka_benchmark.py --raw ../berka-dataset      # ~5 min on a 4-core CPU; writes data/berka/report.json
```

The book is scored as of 1998-10-02 and compared with the following 90
days of real data. The fine-tuned model only saw training targets dated on
or before 1998-10-02. "Holdout" accounts (15 %) were never trained on.

| 90-day balance forecast (4,500 accounts) | wQL ↓ | WAPE (median) ↓ | q10–q90 coverage (target 0.80) |
|---|---|---|---|
| Persistence (last balance) | 0.212 | 0.212 | 0.06 |
| Seasonal naive (30 days) | 0.196 | 0.196 | 0.00 |
| Per-account empirical quantiles (180 days) | 0.179 | 0.225 | 0.57 |
| STFormer trained on the synthetic sandbox only (zero-shot) | 0.184 | 0.224 | 0.76 |
| **STFormer fine-tuned on Berka, out-of-time** | **0.136** | **0.174** | **0.81** |
| …same, holdout accounts only | 0.141 | 0.181 | 0.81 |

| Early warning, 437 loans with an instalment due (62 bad outcomes) | Flagged | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Fine-tuned model, SMA flag / PD vs any bad outcome¹ | 25 | 0.68 | 0.27 | **0.80** |
| …vs the bank's own "in debt / not paid" status | 25 | 0.64 | 0.35 | **0.87** |
| …vs a negative balance in the next 90 days | 25 | 0.40 | 0.83 | **0.97** |
| Rule: balance below next instalment today | 10 | 1.00 | 0.16 | 0.61 |
| Rule: negative balance in the last 90 days | 11 | 1.00 | 0.18 | 0.67 |
| Zero-shot sandbox model vs any bad outcome | 104 | 0.23 | 0.39 | 0.68 |

¹ Bad outcome = a scheduled loan payment that never arrived, a negative
balance in the window, or loan status B/D at the end of the data.

What this shows:

* Fine-tuned on real ledgers, the forecaster beats every naive baseline,
  with 24 % lower quantile loss than the best one, and its q10–q90 bands
  are calibrated (81 % coverage).
* As a risk ranker it beats simple rules on bad outcomes and on the
  bank's status (AUC 0.80–0.87 against 0.61–0.73). On predicting negative
  balances it matches the best rule (0.97 against 0.96).
* The default q10 trigger is conservative: high precision, low recall. Use
  a higher liability quantile, or rank by PD, for wider coverage.
* The model trained only on synthetic data does not transfer. Fine-tune on
  your own history before relying on it.

Caveats:

* Berka has no bureau scores or credit-line data, so those PD components
  are inactive.
* The loan status is a single end-of-1998 snapshot.
* Missed payments are inferred by comparing the reconstructed schedule with
  the loan-payment transactions (±3 days).

## API

| Method & path | Purpose |
|---|---|
| `POST /api/v1/ingest` | Validated JSON batch (accounts, balances, transactions, bureau, schedules, holidays); optional refresh, in the background or blocking |
| `POST /api/v1/ingest/file?table=…` | Streamed CSV / Parquet / Arrow upload |
| `POST /api/v1/ingest/dataset` | Server-side multi-file dataset load (restricted to the data directory) |
| `POST /api/v1/aggregate/refresh` | Incremental (or `?full=true`) aggregation + cache rebuild |
| `POST /api/v1/predict/ews` | Forecast paths, tiers, PD/EAD/LGD/EL, behaviour; persists and pushes alerts |
| `GET /api/v1/alerts/active` | Ranked active alerts; filters `alert_type`, `tier`, `rm_id`, `min_priority`; `stream=true[&follow=true]` for SSE |
| `POST /api/v1/alerts/{id}/ack` | Acknowledge an alert |
| `GET /health`, `GET /api/v1/system/info`, `GET /api/v1/runs` | Operations |

Interactive docs: `/docs`.

## Request latency: read path vs write path

`/predict/ews` never refreshes data or rebuilds the cache inside a request.
Ingest and `POST /api/v1/aggregate/refresh` refresh in the background, and
then rescore the whole book (`score_book`): one forward pass, with results
stored per horizon (30/60/90) in `account_scores`. Only the 90-day run raises
alerts. `POST /api/v1/score/book` runs the same job on demand, for example
nightly.

A whole-book request for the latest day is read from the stored run when it
matches the current data version, model version (a hash of the weights),
horizon and triggers; the response says `"source": "stored"`. Anything else
is scored live (`"source": "live"`). `include_paths` controls paths:

* omitted: none for the whole book, all for account lists
* `"flagged"`: non-GREEN accounts only
* `true`: everything

Responses are serialised with orjson.

Measured in-process on Berka (4,500 accounts, 4-core CPU), with `scripts/bench_api_latency.py`:

| Request | Before | After |
|---|---|---|
| Whole book, default | 5.8 s (live) | **47 ms** (stored) |
| Whole book, paths for flagged accounts | n/a | **87 ms** |
| Whole book, every path (explicitly live) | 6.6 s | 6.4 s |
| 20 accounts with paths | 47 ms | 53 ms |
| 20 accounts right after an ingest | 6.5 s (inline refresh) | **40 ms** |

## Scaling out

* **PostgreSQL:** `EWS_STORAGE__BACKEND=postgres` and
  `EWS_STORAGE__POSTGRES_DSN=postgresql+psycopg2://…`. The same SQL runs on
  both backends. Use Postgres for multi-worker or multi-replica APIs, since
  a DuckDB file has a single writer process.
* **Ray:** `python scripts/batch_score_ray.py --actors 8 --gpus-per-actor 0.25`
  fans nightly portfolio scoring over actors that share the memory-mapped
  cache.
* **Optuna:** `python scripts/tune.py --trials 30`.

## Tests

```bash
pytest -q                                                    # 32 tests: data engine, model, risk maths, API, TimesFM 3 adapter
EWS_TEST_POSTGRES_DSN=postgresql+psycopg2://user@host/db pytest tests/test_postgres.py   # Postgres parity (drops EWS tables!)
```

---

## Frontend (Next.js landing page)

The `app/` directory holds the original Next.js 15 / Shadcn UI / Tailwind
landing page:

```bash
npm install
npm run dev   # http://localhost:3000
```
