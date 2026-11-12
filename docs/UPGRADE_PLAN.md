# EWS console upgrade: Phase 1 plan

Branch `different-approach` (baseline `b58eb6a`). Written by the backend lead from a read of the code, with input from ML,
UI/UX and QA collected in the team chat. API shapes: `.team/ews/API_CONTRACT.md`, copied into the final report when the code lands.

## 1. Inspection findings (verified in code, not from the docs)

**Datasets on disk**

| | database | feature cache | checkpoint | accounts | days | notes |
|---|---|---|---|---|---|---|
| sandbox | `data/ews.duckdb` | `data/vector_cache` | `artifacts/models/ews_stformer.pt` | 125 | 2024-08-23 .. 2026-10-31 | synthetic, loans + EMI + credit lines, INR |
| cards | `data/cards/ews.duckdb` | `data/cards/vector_cache` | `cards_stformer.pt` | 5,264 | 2018-01-01 .. 2020-02-28 | IBM TabFormer; `eod_balance` = rolling 30-day net spend (USD) |
| berka | `data/berka/ews.duckdb` | `data/berka/cache` (not `vector_cache`) | none yet | 4,500 | 1993-01-01 .. 1998-12-31 | real Czech bank, CZK; 682 loans with EMI schedules, outcomes in `data/berka/parquet/loans.parquet`; no bureau, no credit lines |

**How the service works today**
* `api/main.py` builds one `Services(settings, pipeline, ingestion, broker, jobs)` in the lifespan and keeps it on
  `app.state.services`. `console.switch_connection` swaps it for another database; that is the only switching mechanism.
* `data_engine.storage.get_storage` caches one adapter per DuckDB path for the life of the process and nothing closes the
  old one after a switch. A DuckDB file open in the server blocks every other process (AGENTS.md rule), so a switch must close.
* `models.engine.get_engine` is a **process-wide singleton**: after a switch the new pipeline would silently keep the old
  dataset's weights. ML is replacing it with a registry keyed by backend; it must be keyed by (backend, checkpoint) as well.
* `JobManager` has no "busy" query, and there is no marker for in-flight scoring requests, so nothing can refuse a switch.
* Behaviour that differs per dataset lives in environment variables (`.local/cards.env`: attrition floor 500, checkpoint,
  database, cache). The registry replaces that for the three registered datasets.
* TimesFM: `TimesFM3Engine` exists and is complete (multivariate, covariates, equal-length batching, OOM backoff), gated by
  `allow_noncommercial_weights`. Package `timesfm 3.0.2` is installed; weights (1.32 GB, non-commercial licence) are not cached.
* Training writes straight onto `settings.model.checkpoint_path`, i.e. an existing checkpoint, and `/console/training` reads the
  progress file next to it.

**Root causes of the documented display problems**
* `days_to_zero = 1` with a positive worst case: `models/risk.py:163` tests `q10 <= zero_balance_threshold` (500, an INR-era
  absolute), not `<= 0`. For 30-day card spend almost every card is "below 500". It is a floor, not zero: make it dataset-aware and label it.
* "DECLINING (-24.7%)" beside a flat forecast: `models/behavior.py:63` compares the mean forecast median with the trailing 90-day
  mean, not with today's value. Both means are already in the payload; show both with the comparison spelled out.
* Expected loss only for SMA accounts: `risk.py` sets `el = where(in_sma, ...)` and `pipeline.score_table` masks it again with
  `hide=~in_sma`, so a GREEN card with PD 23.5 % has EL N/A and priority 0.
* PD for cards is `0.25 * logistic(FICO)`: cash and utilisation terms are 0 (no EMIs, utilisation rarely above 95 %).
* Berka has no bureau data but the cache fills `cibil_score` with 700 and the pipeline does `nan_to_num(..., 700)`, so PD would
  carry an invented bureau term. Needs a per-dataset switch (`bureau_enabled`).
* Overview "185 alerts" vs tables "ews_alerts 370": the tables panel counts every run's rows (active + superseded).
* Time zones: DB timestamps are naive UTC (`utcnow()`), jobs/training use naive local time (the machine is IST).

## 2. Safest approach (least change that works)

1. One **registry** (`config/registry.py`): immutable `DatasetSpec` / `ModelSpec` records. Static facts (paths, targets, currency, labels,
   capabilities, per-dataset settings overrides) are data; dynamic facts (account count, date range, checkpoint present, model
   status) are computed from the cache `meta.json`, the checkpoint sidecar and the engine status, without opening DuckDB.
2. A small **runtime** (`api/runtime.py`) next to `Services`: active dataset id, active model id, one switch lock. Dataset switching
   reuses the existing `Services` swap: build the new services off to the side (open DB, check schema, build pipeline), then
   swap `app.state.services`, then close the old DuckDB connection after a short grace period. Refuses (409 `busy`) while any job is
   queued/running or a scoring/forecast/compare call is in flight; a failed build leaves the old services in place.
3. Existing endpoints stay as they are; changes are additive (new keys and query parameters).
4. New routers: `datasets`, `models`, `forecast`, `compare`, with Pydantic models in `api/schemas_v2.py` (existing `schemas.py` untouched).
5. Display logic (labels, N/A reasons, PD basis, Projected SMA wording, trigger wording) is pure functions in `models/risk.py` and
   `models/behavior.py`, applied to account/alert dicts in `api/presentation.py`, so the pipeline's stored format is unchanged.
6. Forecasts for a chosen model use ML's engine registry through `ScoreRequest.backend`; non-default backends never persist alerts.

## 3. Dataset registry and switching

Fields per dataset: id, display name, description, source, currency, database path, cache root, checkpoint path, targets (id, label,
unit, source channel, availability + reason, applicable product types), features (available + reason), capability flags, supported
models, default model, alert-type vocabulary, UI labels, risk settings overrides (zero floor, attrition floor, bureau enabled),
limitations. Model fields: id, name, backend, kind (trained / pretrained zero-shot), supported datasets, targets, horizons,
quantile support, training capability, version, availability. Targets per dataset:

* **cards**: primary `rolling_30d_net_spend` (USD, "not a balance, not debt"); secondary `estimated_utilization` for credit cards only
  (30-day spend / limit, assumes the bill is paid monthly). No EMI/SMA/loan fields. FICO is labelled FICO.
* **berka**: primary `eod_balance` (CZK, real end-of-day balance). Utilisation unavailable (no credit lines). Bureau unavailable.
  Real EMI schedules and loan outcomes available, so projected SMA tiers and the loan-outcome evaluation are genuine for 682 loan accounts.
* **sandbox**: `eod_balance` (INR) + `utilization`, EMI, SMA, EAD, LGD as before.

Switching rules: refuse while busy; never reuse one dataset's account list for another (lists come from each dataset's own cache);
a model unsupported by the new dataset resets to its default and says so in the response; selecting a model never changes the
database; a failed model load leaves the previous model active. Persisting the choice across restarts is deliberately not done
(the process default from the environment stays the sandbox, as AGENTS.md expects).

## 4. API (summary; shapes in the contract)

`GET /datasets`, `GET /datasets/{id}`, `POST /datasets/select`, `GET /datasets/{id}/accounts`, `GET /models`, `GET /models/{id}/status`,
`POST /models/select`, `POST /forecast`, `POST /compare`, `GET /compare`, `GET /compare/{id}`; additive fields on `/console/overview`,
`/alerts/active`, `/predict/ews`, `/health`; `/jobs/train` refuses non-trainable models and writes a new checkpoint file.
Errors on new endpoints: `{"detail": {"code", "message"}}` with stable codes. Local-only guard on every state-changing call.

## 5. Risk-display fixes (section 11), kept small

* EL, EAD, PD, LGD computed for every credit product (credit limit > 0 or outstanding principal > 0); NaN (shown N/A, never 0) for
  debit, prepaid and deposit accounts, with an `na_reason`. Priority = EL x urgency for credit products.
* Cards: PD labelled "bureau-based estimate, not calibrated" with its three components and weights; "Insufficient Balance" declines are
  labelled declined transactions; `DEPOSIT_ATTRITION` shown as "Spending decline", `CASH_TO_ZERO_DAY_n` as "Spend below floor".
* Projected tiers shown as "Projected SMA-0/1/2" with the RBI tooltip; GREEN gets a "not a guarantee" note; alert taxonomy
  (predictive / spending change / repayment failure / regulatory), regulatory only with verified overdue data (none in these datasets).
* Zero floor per dataset and shown with the value; both trend averages shown; IST everywhere; active vs all-run alert counts;
  `bureau_enabled=false` for Berka; high-risk threshold configurable (`risk.high_risk_pd_threshold`, default 0.20).
* q10 <= q50 <= q90 enforced (ML in the engine, backend again in `/forecast`).

## 6. TimesFM

Default backend stays `stformer`. `EWS_MODEL__ALLOW_NONCOMMERCIAL_WEIGHTS=true` goes only in `.local/env.sh`; TimesFM is chosen
per request/selection. Status endpoint reports package / licence flag / weights cached / loaded / last error without loading
anything. No silent fallback: every forecast response names the model actually used; the badge "TimesFM 3.0 · research licence ·
prototype only" travels in `model.badge` and `licence.badge`. One instance is loaded once and reused across datasets. No download
unless the caller passes `allow_download`.

## 7. ML input (from chat, 12:43)

Engine registry `get_engine(settings, backend)` + `engine_status()` without loading weights; `ScoreRequest.backend`; no fallback
(`EngineUnavailable`); common `ForecastResponse`; weights download into `.local/hf` (GPU has ~2.3 GB free, OOM backoff then CPU).
Fair comparison needs a separate STFormer checkpoint trained before the origin, customer-level split, >= 3 origins, baselines,
results in `artifacts/eval/<dataset>/<id>.json`; fix `scripts/berka_benchmark.py` and train a Berka checkpoint. Backend asked for:
registry key (backend, checkpoint), evict hook, two NaN-safe edits in `pipeline.py`, `run_comparison`/`save_comparison`.

## 8. UI input (from chat, 12:55)

All dataset/model vocabulary read from the backend: names, currency, target label/units/definition, date range, account counts,
supported models, alert types, risk-label vocabulary, licence badge text, trainable flag, quantiles, horizons; null means
"unavailable"; pages show an explicit "backend not available" state until an endpoint exists. Existing console, alerts, jobs and
import endpoints stay untouched. QA input: baseline 57 passed + 1 skipped; QA is the only committer, one commit per phase.

## 9. Phases and ownership

| Phase | Backend (me) | Others |
|---|---|---|
| 1 plan | this document, contract, pytest config | UI screenshots (done), ML TimesFM facts |
| 2 datasets | registry, runtime/switching, datasets/models/accounts endpoints, risk fixes, overview additions | ML engine registry |
| 3 TimesFM | model status/select, `/forecast` | ML engine, weights, `.local/env.sh` flag |
| 4 compare | `/compare` wrapper, saved list/get | ML metrics, baselines, fair checkpoint, Berka checkpoint |
| 5 UI | alert filters/labels, overview KPIs | UI redesign |
| 6 tests | my three test files | QA suite and verification |

## 10. Risks

TimesFM weights/GPU headroom (shared GPU, ~2.3 GB free); switching races (handled by the in-flight counter and the lock);
Berka bureau and "outstanding principal" are adapter estimates (documented); concurrent edits of `models/pipeline.py` (ML owns
it, backend asks instead of editing); no migration of existing checkpoints (new files only).
