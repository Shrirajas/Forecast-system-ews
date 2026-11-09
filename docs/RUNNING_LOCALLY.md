# Running the EWS on your own PC

This guide takes a fresh machine to a running API with a trained model and real-data results.
It covers Windows, macOS and Linux, with and without an NVIDIA GPU (tested target: RTX 4080 Super, 16 GB).

> Handing this to an AI coding agent? Point it at [`AGENTS.md`](../AGENTS.md) instead: the same steps,
> written as commands with pass/fail checks.

---

## 1. What you need

| Item | Version | Notes |
|---|---|---|
| Python | **3.11, 3.12 or 3.13** (64-bit) | 3.13 is what the code was developed and tested on. 3.10 is likely to work but is untested. |
| Git | any recent | To clone the repo (and the optional real-data set). |
| Disk | ~10 GB free | ~4–5 GB is PyTorch + CUDA libraries; the rest is data and caches. |
| RAM | 8 GB minimum, 16 GB recommended | The real-data benchmark loads 5.6M daily rows. |
| NVIDIA GPU (optional) | driver supporting CUDA 12.8 | Without a GPU everything still runs on CPU, just slower. |

Check:

```bash
python --version      # Windows may need: py -3.13 --version
git --version
nvidia-smi            # GPU only: shows the driver and the CUDA version it supports
```

## 2. Get the code

```bash
git clone https://github.com/kazuma761/Forecast-system-ews.git
cd Forecast-system-ews
git checkout feature/fast-predict      # or main / develop, once merged
```

## 3. Create a virtual environment

**Windows (PowerShell)**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Every command below assumes the venv is active and you are in the repo root.

## 4. Install the packages

### 4a. PyTorch first (pick ONE)

The order matters: install PyTorch from the right index **before** `requirements.txt`, or pip may pull
a CPU-only or mismatched build.

| Machine | Command |
|---|---|
| NVIDIA GPU (Windows or Linux) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| CPU only (Windows / macOS / Linux) | `pip install torch torchvision torchaudio` |

Check the GPU is visible (GPU machines only):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# expected: 2.x.x+cu128 True NVIDIA GeForce RTX 4080 SUPER
```

If it prints `False`, update the NVIDIA driver and reinstall with the command above.

### 4b. Everything else

```bash
pip install -r requirements.txt
```

What gets installed and why:

| Package | Used for |
|---|---|
| `torch`, `torchvision`, `torchaudio` | The forecasting model (training and inference, CUDA or CPU). |
| `timesfm[torch]` | Optional pretrained TimesFM 3.0 engine (research only, see section 9). |
| `transformers`, `accelerate` | Model-loading utilities used by pretrained engines. |
| `numpy`, `pandas`, `scikit-learn` | Arrays, tables, metrics (ROC-AUC, Brier score). |
| `optuna` | Hyper-parameter search (`scripts/tune.py`). |
| `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `python-multipart`, `orjson` | The REST API, its validation, settings, file uploads and fast JSON. |
| `pyarrow`, `duckdb` | Columnar data streaming and the default embedded database. |
| `sqlalchemy`, `psycopg2-binary` | Optional PostgreSQL backend. |
| `ray[default]` | Optional distributed nightly scoring (`scripts/batch_score_ray.py`). |
| `loguru` | Logging. |
| `httpx`, `pytest` | Tests. |

If `ray` fails to install on your platform, you can skip it, because only `scripts/batch_score_ray.py`
needs it. Delete the `ray[default]` line from `requirements.txt` (or a copy of it) and rerun
`pip install -r requirements.txt`.

### 4c. Verify the install

```bash
pytest -q
```

Expected: `45 passed, 1 skipped` (the skipped test needs a PostgreSQL server). Takes about 15 s.

## 5. Bring up the sandbox (synthetic data + trained model)

```bash
python scripts/bootstrap.py
```

What it does, in order:

1. Generates the 100,000-row synthetic book (125 accounts × 800 days) into `data/sandbox/`.
2. Loads it into DuckDB (`data/ews.duckdb`).
3. Builds the daily features and the memory-mapped tensor cache (`data/vector_cache/`).
4. Trains the forecaster into `artifacts/models/ews_stformer.pt`.
   * GPU: the full model (d_model 128, 3 layers) with mixed precision.
   * CPU: a lighter profile; about 5 minutes on a 4-core CPU.
5. Scores the book, prints the summary and top alerts, and runs an out-of-time backtest.

Useful options:

| Option | Effect |
|---|---|
| `--epochs 10` | Train longer. |
| `--skip-train` | Reuse an existing checkpoint. |
| `--skip-generate` | Reuse the existing `data/sandbox/` files. |
| `--accounts 20000` | A bigger synthetic book (more rows, longer run). |

## 6. Run the API

```bash
python scripts/serve.py --port 8000
```

Then open <http://localhost:8000/docs> for interactive API docs.

At startup the server refreshes features and scores the whole book once, so the first whole-book
request is already served from stored results.

Try it:

```bash
# health
curl http://localhost:8000/health

# whole book (stored, ~50 ms): tiers, PD, EAD, LGD, expected loss, behaviour; no paths by default
curl -X POST http://localhost:8000/api/v1/predict/ews -H "content-type: application/json" -d "{\"horizon\": 90}"

# whole book with forecast paths for flagged (non-GREEN) accounts only
curl -X POST http://localhost:8000/api/v1/predict/ews -H "content-type: application/json" -d "{\"include_paths\": \"flagged\"}"

# a few accounts with full paths
curl -X POST http://localhost:8000/api/v1/predict/ews -H "content-type: application/json" -d "{\"account_ids\": [\"AC00000018\", \"AC00000004\"]}"

# ranked alerts, and a live stream of new ones (Server-Sent Events)
curl "http://localhost:8000/api/v1/alerts/active?limit=10"
curl -N "http://localhost:8000/api/v1/alerts/active?stream=true&follow=true"

# rescore the whole book now (the nightly job)
curl -X POST "http://localhost:8000/api/v1/score/book?wait=true"
```

Windows PowerShell: use `curl.exe`, or `Invoke-RestMethod`, e.g.
`Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/predict/ews -ContentType 'application/json' -Body '{"horizon": 90}'`.

Stop the server with `Ctrl+C`.

> **One process per DuckDB file.** While the server is running it holds `data/ews.duckdb`. Stop it
> before running `bootstrap.py`, `train.py`, `backtest.py` or any other script that uses the same
> database, or you'll get a "Could not set lock on file" error. For several processes at once, switch to
> PostgreSQL (section 8).

## 7. Real-data benchmark (Czech bank, Berka / PKDD'99)

```bash
git clone --depth 1 https://github.com/jlacko/berka-dataset ../berka-dataset
python scripts/berka_benchmark.py --raw ../berka-dataset
```

This loads 4,500 real accounts (5.6M daily balances, 1.06M transactions, 682 loans), fine-tunes the
model on data up to 1998-10-02, and compares its forecasts and loan warnings with what actually happened
in the next 90 days. It writes `data/berka/report.json` and prints the same report. It takes about 5
minutes on a 4-core CPU (faster on a GPU). It needs `artifacts/models/ews_stformer.pt` from step 5.

Measure API latency on that data:

```bash
python scripts/bench_api_latency.py --work data/berka --checkpoint artifacts/models/berka_stformer.pt
```

## 8. Other tools

| Command | What it does |
|---|---|
| `python scripts/train.py --epochs 20 --batch-size 128 --accum 2 --compile` | Retrain on the current cache (GPU options shown). |
| `python scripts/backtest.py --horizon 90 --liability-q 0.2` | Out-of-time EWS backtest with a different trigger quantile. |
| `python scripts/tune.py --trials 20` | Optuna search over model size and learning rate. |
| `python scripts/batch_score_ray.py --actors 4` | Nightly scoring spread over Ray workers. |
| `python scripts/generate_sandbox.py --accounts 50000 --out data/big` | Generate a larger synthetic book only. |

PostgreSQL instead of DuckDB (needed for several API workers):

```bash
# PowerShell: $env:EWS_STORAGE__BACKEND="postgres"; $env:EWS_STORAGE__POSTGRES_DSN="postgresql+psycopg2://user:pass@localhost:5432/ews"
export EWS_STORAGE__BACKEND=postgres
export EWS_STORAGE__POSTGRES_DSN=postgresql+psycopg2://user:pass@localhost:5432/ews
python scripts/bootstrap.py --skip-train
python scripts/serve.py --workers 4
```

## 9. Configuration

Every setting in `config/settings.py` can be overridden with an environment variable prefixed `EWS_`
(nested sections use `__`), or in a `.env` file in the repo root. The common ones:

| Variable | Default | Meaning |
|---|---|---|
| `EWS_STORAGE__BACKEND` | `duckdb` | `duckdb` or `postgres`. |
| `EWS_MODEL__CHECKPOINT_PATH` | `artifacts/models/ews_stformer.pt` | Which trained model the API uses. |
| `EWS_MODEL__AMP_DTYPE` | `auto` | `auto` picks bf16 on an RTX 4080; `fp16`, `fp32` also allowed. |
| `EWS_MODEL__INFERENCE_BATCH_SIZE` | `512` | Accounts per forward pass; halves itself if the GPU runs out of memory. |
| `EWS_MODEL__COMPILE_MODEL` | `false` | `torch.compile` on CUDA (first call is slow, later calls faster). |
| `EWS_RISK__LIABILITY_QUANTILE` | `0.1` | Worst-case cash track for the EMI-shortfall trigger (0.2–0.3 = fewer, more precise alerts). |
| `EWS_RESCORE_AFTER_REFRESH` | `true` | Rescore the whole book after every data refresh. |

**TimesFM 3.0 (pretrained, research only):** its weights are licensed for non-commercial,
non-production use, which rules out commercial demos too. To try it on a genuinely non-commercial
research setup, you need internet access to Hugging Face, and then:

```bash
export EWS_MODEL__BACKEND=timesfm3
export EWS_MODEL__ALLOW_NONCOMMERCIAL_WEIGHTS=true
python scripts/serve.py
```

The first run downloads `google/timesfm-3.0-pytorch`. bf16 autocast for TimesFM stays off
(`EWS_MODEL__TIMESFM_AUTOCAST=false`) until it has been checked against fp32 on your GPU.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` on a GPU machine | Reinstall PyTorch with the `cu128` command in 4a; update the NVIDIA driver. |
| `Could not set lock on file ... ews.duckdb` | Another process (usually the API server) has the database open. Stop it first. |
| `model checkpoint not found` (HTTP 503) | Run `python scripts/bootstrap.py` (or `scripts/train.py`) first. |
| `vector cache not built yet` (HTTP 503) | Ingest data, then `POST /api/v1/aggregate/refresh`, or rerun `bootstrap.py`. |
| CUDA out of memory while training | Lower `--batch-size` and raise `--accum` (e.g. `--batch-size 32 --accum 4`). |
| Windows: DataLoader worker errors | Use `--workers 0`, or keep running scripts as `python scripts/<name>.py` (they already have the required `if __name__ == "__main__"` guard). |
| Start from scratch | Stop the server, delete `data/` and `artifacts/`, rerun `bootstrap.py`. |
