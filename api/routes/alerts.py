"""Proactive trigger dashboard: ranked active alerts (JSON or Server-Sent Events)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.deps import Services, services
from api.schemas import AckResponse, ActiveAlertsResponse, AlertOut, AlertType, Tier

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])
HEARTBEAT_SECONDS = 15.0


def _sse(event: str, data: Any, event_id: str | None = None) -> bytes:
    head = f"event: {event}\n" + (f"id: {event_id}\n" if event_id else "")
    return (head + f"data: {json.dumps(data, default=str)}\n\n").encode()


@router.get(
    "/active",
    response_model=ActiveAlertsResponse,
    summary="Ranked active alerts for relationship managers",
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def active_alerts(
    request: Request,
    svc: Annotated[Services, Depends(services)],
    alert_type: AlertType | None = None,
    tier: Tier | None = None,
    rm_id: str | None = None,
    min_priority: Annotated[float, Query(ge=0)] = 0.0,
    include_acknowledged: bool = False,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    stream: Annotated[bool, Query(description="Server-Sent Events instead of JSON")] = False,
    follow: Annotated[bool, Query(description="With stream: keep the connection open and push new alerts")] = False,
):
    """Alerts that breached the EWS safety bounds, ranked by:

    1. alert type (credit EWS ahead of deposit attrition),
    2. priority score: expected loss (EAD x PD x LGD) weighted by urgency,
       which halves every 30 days until the cash balance reaches zero or the
       first EMI shortfall, whichever comes first,
    3. exposure at default, then days to zero.

    ``?stream=true`` (or ``Accept: text/event-stream``) streams the ranked
    list as SSE ``alert`` events; add ``follow=true`` to keep receiving
    alerts as new forecast runs raise them.
    """
    rows, total = await run_in_threadpool(
        svc.pipeline.alerts.active, alert_type, tier, rm_id, min_priority, include_acknowledged, limit, offset
    )
    alerts = [AlertOut.model_validate(r) for r in rows]
    wants_sse = stream or "text/event-stream" in request.headers.get("accept", "")
    if not wants_sse:
        return ActiveAlertsResponse(generated_at=datetime.now(), total=total, returned=len(alerts), alerts=alerts)

    def matches(a: dict[str, Any]) -> bool:
        return (
            (alert_type is None or a["alert_type"] == alert_type)
            and (tier is None or a["risk_tier"] == tier)
            and (rm_id is None or a.get("rm_id") == rm_id)
            and float(a.get("priority_score") or 0) >= min_priority
        )

    async def events() -> AsyncIterator[bytes]:
        queue = svc.broker.subscribe() if follow else None
        try:
            yield _sse("snapshot_start", {"total": total, "returned": len(alerts)})
            for a in alerts:
                yield _sse("alert", a.model_dump(mode="json"), a.alert_id)
            yield _sse("snapshot_end", {"generated_at": datetime.now().isoformat()})
            while queue is not None:
                if await request.is_disconnected():
                    break
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except TimeoutError:
                    yield b": ping\n\n"
                    continue
                if matches(raw):
                    live = AlertOut.model_validate({**raw, "status": "ACTIVE", "rank": None})
                    yield _sse("alert", live.model_dump(mode="json"), live.alert_id)
        finally:
            if queue is not None:
                svc.broker.unsubscribe(queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{alert_id}/ack", response_model=AckResponse, summary="Acknowledge an alert")
async def acknowledge(alert_id: str, svc: Annotated[Services, Depends(services)]) -> AckResponse:
    ok = await run_in_threadpool(svc.pipeline.alerts.acknowledge, alert_id)
    if not ok:
        raise HTTPException(404, "no active alert with that id")
    return AckResponse(alert_id=alert_id, status="ACKNOWLEDGED")
