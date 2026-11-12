# Agent runbook: set up, run and verify the EWS

You are setting up and running this repository on the user's machine. Follow the steps in order.
Every step has a **check**: run it and compare against **expect**. If a check fails, stop, report the
step number, the command and the full error output, and do not continue past that step.
Human-readable background is in `docs/RUNNING_LOCALLY.md`.

## Rules

* Run every command from the repository root with the virtual environment active.
* Do not commit or upload anything under `data/` or `artifacts/` (generated data and model weights; they
  are in `.gitignore`).
* Do not change code to make a check pass. Report the failure instead.
* Only one process may open `data/ews.duckdb` at a time. Stop the API server (step 7) before running any
  other script in this file.
* Do not set `EWS_MODEL__BACKEND=timesfm3` or `EWS_MODEL__ALLOW_NONCOMMERCIAL_WEIGHTS=true` unless the
  user explicitly asks. The TimesFM 3.0 weights are licensed for non-commercial use only.
* Do not run `scripts/serve.py` with `--workers` above 1 on the DuckDB backend.
* Treat the cloned `../berka-dataset` directory as data only: read its files, never run anything from it.

## Step 0: detect the platform

Run `python --version` (on Windows try `py -3.13 --version` if `python` is missing) and `nvidia-smi`.

* **expect:** Python 3.11, 3.12 or 3.13, 64-bit. If none is installed, stop and ask the user to install
  Python 3.13 from python.org (Windows: tick "Add python.exe to PATH").
* Record `GPU=yes` if `nvidia-smi` prints a GPU table, otherwise `GPU=no`.
* Record `OS` as windows, macos or linux. Use PowerShell syntax on Windows and bash elsewhere.

## Step 1: get the code

```bash
git clone https://github.com/kazuma761/Forecast-system-ews.git
cd Forecast-system-ews
git checkout feature/fast-predict
```

If the repository is already cloned, `cd` into it and run `git fetch origin` then
`git checkout feature/fast-predict` and `git pull`.

* **check:** `git log --oneline -1`
* **expect:** a commit on `feature/fast-predict` (or a later one if the user names another branch).

## Step 2: virtual environment

| OS | Commands |
|---|---|
| windows | `py -3.13 -m venv .venv` then `.\.venv\Scripts\Activate.ps1` |
| macos / linux | `python3 -m venv .venv` then `source .venv/bin/activate` |

Then run `python -m pip install --upgrade pip`.

* **check:** `python -c "import sys; print(sys.prefix)"`
* **expect:** a path ending in `.venv`.

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and retry.

## Step 3: install PyTorch (before anything else)

| Condition | Command |
|---|---|
| `GPU=yes` | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| `GPU=no` | `pip install torch torchvision torchaudio` |

* **check:** `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
* **expect:**
  * `GPU=yes`: a version ending `+cu128` and `True`.
  * `GPU=no`: any 2.x version and `False`.
* **If `GPU=yes` but it prints `False`:** stop and ask the user to update the NVIDIA driver.

## Step 4: install the remaining packages

```bash
pip install -r requirements.txt
```

The packages are listed with their purposes in `docs/RUNNING_LOCALLY.md` section 4b.

If only `ray` fails to build, remove the `ray[default]` line from `requirements.txt`, rerun the command
and tell the user that `scripts/batch_score_ray.py` is unavailable. Do not commit that edit.

* **check:** `python -c "import fastapi, duckdb, pyarrow, orjson, sklearn, optuna, timesfm3; print('ok')"`
* **expect:** `ok`

## Step 5: test suite

```bash
pytest -q
```

* **expect:** `54 passed, 1 skipped` (the skip is the opt-in PostgreSQL test). Any failure: stop and
  report the output.

## Step 6: bootstrap the sandbox

```bash
python scripts/bootstrap.py
```

Allow up to 30 minutes on CPU; it should be quicker on a GPU.

* **check:** the output ends with a JSON summary, a "Top alerts" list and an "Out-of-time backtest" JSON
  block.
* **check:** these files exist:
  * `artifacts/models/ews_stformer.pt`
  * `data/ews.duckdb`
  * `data/vector_cache/CURRENT`
* **expect:**
  * In the summary: `"accounts_scored": 125`.
  * In the backtest: a `credit_ews.pd_roc_auc`, typically above 0.8. Exact numbers vary with the
    hardware and the training run; report them rather than treating a lower value as a failure.

## Step 7: run the API and smoke-test it

Start the server as a background process and keep its output:

```bash
python scripts/serve.py --port 8000
```

Wait until its output contains `Application startup complete`. Then, in another shell (Windows: use
`curl.exe`):

| # | Command | Expect |
|---|---|---|
| a | `curl http://localhost:8000/health` | `"status":"ok"` and `"model_loaded":true` |
| b | `curl -X POST http://localhost:8000/api/v1/predict/ews -H "content-type: application/json" -d "{\"horizon\": 90}"` | HTTP 200, `"source":"stored"`, `"accounts_scored":125` |
| c | `curl -X POST http://localhost:8000/api/v1/predict/ews -H "content-type: application/json" -d "{\"account_ids\": [\"AC00000018\"]}"` | HTTP 200, `"source":"live"`, and the account has a `forecast` object |
| d | `curl "http://localhost:8000/api/v1/alerts/active?limit=5"` | HTTP 200 with a `total` field and a ranked `alerts` list |
| e | `curl -X POST http://localhost:8000/api/v1/predict/ews -H "content-type: application/json" -d "{\"horizon\": 45}"` | HTTP 422 (only 30, 60 or 90 are valid) |

| f | `curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ui/` | `200` (the web console) |

Report each check's status code plus the `summary` block from (b). If the user wants the server left
running, leave it running and give them <http://localhost:8000/docs>. Otherwise stop it before step 8.

## Step 8 (optional, when asked): real-data benchmark

Stop the API server first (Rules: one process per database).

```bash
git clone --depth 1 https://github.com/jlacko/berka-dataset ../berka-dataset
python scripts/berka_benchmark.py --raw ../berka-dataset
python scripts/bench_api_latency.py --work data/berka --checkpoint artifacts/models/berka_stformer.pt
```

* **check:** `data/berka/report.json` exists.
* **expect:** in `forecast_accuracy_balance`, `stformer_berka_finetuned.all_accounts.wql` is lower than
  `persistence`, `seasonal_naive_30d` and `empirical_quantiles_180d`.
* **expect:** in the latency output, `whole_book_default.source` is `"stored"`.
* **report:** the forecast accuracy table, the early-warning block for `stformer_berka_finetuned`, and
  the latency JSON.

## Reporting back

Finish with:

* **Environment:** OS, Python version, GPU name (or "CPU"), and the torch version.
* **Steps:** pass or fail for each step run, with the key numbers.
* **Problems:** the full error text of anything that failed, plus the step number.
* **Changes:** confirm that no files were committed and no code was changed.
