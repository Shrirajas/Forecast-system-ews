"""Dataset registry, dataset switching, per-dataset account lists, models and model selection."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from api.deps import local_only
from api.runtime import Runtime, err
from api.schemas_v2 import (AccountsOut, DatasetsOut, ModelsOut, SelectDatasetOut, SelectDatasetRequest,
                            SelectModelOut, SelectModelRequest)
from config.registry import account_table, dataset_info

router = APIRouter(prefix="/api/v1", tags=["datasets & models"])


def runtime(request: Request) -> Runtime:
    return request.app.state.runtime


Rt = Annotated[Runtime, Depends(runtime)]


@router.get("/datasets", response_model=DatasetsOut, summary="Registered datasets with capabilities")
async def list_datasets(rt: Rt) -> dict:
    return await run_in_threadpool(rt.datasets)


@router.get("/datasets/active", summary="The active dataset")
async def active_dataset(rt: Rt) -> dict:
    if rt.dataset_id is None:
        raise err(404, "unknown_dataset", "a custom database is connected; it is not a registered dataset")
    return dataset_info(rt.dataset(rt.dataset_id), rt.dataset_id)


@router.get("/datasets/{dataset_id}", summary="One dataset")
async def get_dataset(dataset_id: str, rt: Rt) -> dict:
    return await run_in_threadpool(dataset_info, rt.dataset(dataset_id), rt.dataset_id)


@router.post("/datasets/select", response_model=SelectDatasetOut, dependencies=[Depends(local_only)],
             summary="Switch the active dataset (database, cache, checkpoint)")
async def select_dataset(req: SelectDatasetRequest, rt: Rt) -> dict:
    return await run_in_threadpool(rt.select_dataset, req.dataset_id)


@router.get("/datasets/{dataset_id}/accounts", response_model=AccountsOut, summary="Accounts of one dataset")
async def dataset_accounts(
    dataset_id: str, rt: Rt, q: str | None = None, product_type: str | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50, offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    spec = rt.dataset(dataset_id)

    def read() -> dict:
        acc = account_table(spec)  # that dataset's own cache: ids are never shared between datasets
        if acc is None:
            raise err(422, "dataset_unavailable", "feature cache not built for this dataset")
        counts = {k: int(v) for k, v in acc["product_type"].value_counts().items()}
        if product_type:
            acc = acc[acc["product_type"] == product_type]
        if q:
            acc = acc[acc["account_id"].str.contains(q, case=False, regex=False)]
        page = acc.sort_values("account_id").iloc[offset: offset + limit]
        items = [{"account_id": r.account_id, "customer_id": r.customer_id, "product_type": r.product_type,
                  "segment": r.segment, "rm_id": r.rm_id,
                  "credit_limit": float(r.credit_limit) if r.credit_limit and r.credit_limit > 0 else None,
                  "open_date": r.open_date, "has_credit_line": bool(r.credit_limit and r.credit_limit > 0)}
                 for r in page.itertuples()]
        return {"dataset_id": dataset_id, "total": len(acc), "returned": len(items), "offset": offset,
                "product_types": counts, "accounts": items}

    return await run_in_threadpool(read)


@router.get("/models", response_model=ModelsOut, summary="Forecasting models and their status")
async def list_models(rt: Rt, dataset_id: str | None = None) -> dict:
    return await run_in_threadpool(rt.models_list, dataset_id)


@router.get("/models/{model_id}/status", summary="Metadata and status of one model")
async def model_status(model_id: str, rt: Rt, dataset_id: str | None = None) -> dict:
    if dataset_id:
        rt.dataset(dataset_id)
    return await run_in_threadpool(rt.model_info, model_id, dataset_id)


@router.post("/models/select", response_model=SelectModelOut, dependencies=[Depends(local_only)],
             summary="Select (and load once) the forecasting model")
async def select_model(req: SelectModelRequest, rt: Rt) -> dict:
    return await run_in_threadpool(rt.select_model, req.model_id, req.allow_download)
