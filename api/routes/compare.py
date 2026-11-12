"""Model comparison: run one (background job around scripts/compare_models.py), list and read saved ones."""

from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from api import compare_store
from api.deps import local_only
from api.runtime import err
from api.schemas_v2 import CompareRequest
from config.registry import cache_meta
from config.settings import PROJECT_ROOT

router = APIRouter(prefix="/api/v1", tags=["compare"])


def _spec(request: Request, dataset_id: str | None):
    rt = request.app.state.runtime
    return rt.dataset(dataset_id or rt.dataset_id or "")


@router.post("/compare", status_code=202, dependencies=[Depends(local_only)],
             summary="Compare STFormer and TimesFM on identical origins and windows (background job)")
async def run_compare(req: CompareRequest, request: Request) -> dict[str, Any]:
    rt = request.app.state.runtime
    rt.require_active(req.dataset_id)
    spec = _spec(request, req.dataset_id)
    for m in req.models:
        rt.check_supported(m, spec.id)
    meta = cache_meta(spec)
    if meta is None:
        raise err(422, "dataset_unavailable", "feature cache not built for this dataset")
    last = date.fromisoformat(meta["end_date"])
    origins = sorted(req.origins) if req.origins else [last - timedelta(days=req.horizon + 30 * k) for k in (3, 2, 1, 0)]
    if len(origins) < 3 or any(o + timedelta(days=req.horizon) > last for o in origins):
        raise err(422, "invalid_origins", f"need at least 3 origins, each at least {req.horizon} days before the last data day "
                  f"({last}): the holdout actuals must exist")
    if req.scope == "account" and req.account_id is None:
        raise err(422, "unknown_account", "scope=account needs an account_id")
    models_dir = spec.checkpoint_path.parent
    holdout = models_dir / f"{spec.id}_stformer_holdout.pt"  # trained only on windows that end before the first origin
    split = models_dir / f"{spec.id}_stformer_holdout.split.json"
    args = [sys.executable, str(PROJECT_ROOT / "scripts" / "compare_models.py"), "--dataset", spec.id,
            "--cache-root", str(spec.cache_root), "--origins", ",".join(o.isoformat() for o in origins),
            "--horizon", str(req.horizon), "--unit", spec.currency.code, "--holdout-seed", str(req.seed)]
    if "stformer" in req.models:
        args += ["--stformer", f"{'holdout' if holdout.exists() else 'current'}={holdout if holdout.exists() else rt.svc.settings.model.checkpoint_path}"]
    if "timesfm3" in req.models:
        args.append("--timesfm")
    args += ["--test-split", str(split)] if split.exists() else ["--holdout-fraction", "0.2"]
    if req.account_id:
        args += ["--single-account", req.account_id]
    folder = compare_store.eval_dir(spec)

    def work() -> dict[str, Any]:
        before = set(folder.glob("*.json"))
        done = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError((done.stderr or done.stdout).strip().splitlines()[-1][:500])
        new = sorted(set(folder.glob("*.json")) - before)
        return {"comparison_id": new[-1].stem if new else None,
                "fair": holdout.exists(), "checkpoint": "holdout" if holdout.exists() else "current"}

    job = rt.svc.jobs.submit("compare", f"Compare {' vs '.join(req.models)} on {spec.display_name}", work)
    return {"job": job.view(), "comparison_id": None}


@router.get("/compare", summary="Saved comparisons of a dataset, newest first")
async def list_comparisons(request: Request, dataset_id: str | None = None) -> dict:
    spec = _spec(request, dataset_id)
    return {"dataset_id": spec.id, "comparisons": await run_in_threadpool(compare_store.list_saved, spec)}


@router.get("/compare/{comparison_id}", summary="One saved comparison (full result)")
async def get_comparison(comparison_id: str, request: Request, dataset_id: str | None = None) -> dict:
    return await run_in_threadpool(compare_store.load, _spec(request, dataset_id), comparison_id)
