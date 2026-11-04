"""In-process pub/sub that pushes freshly raised alerts to SSE subscribers."""

from __future__ import annotations

import asyncio
from typing import Any


class AlertBroker:
    def __init__(self, max_queue: int = 1_000):
        self._subs: set[asyncio.Queue] = set()
        self._max_queue = max_queue

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    @property
    def subscribers(self) -> int:
        return len(self._subs)

    def publish(self, alerts: list[dict[str, Any]]) -> None:
        """Call from the event loop. Slow consumers drop alerts instead of
        blocking the scorer; they can always re-read /alerts/active."""
        for q in list(self._subs):
            for a in alerts:
                try:
                    q.put_nowait(a)
                except asyncio.QueueFull:
                    break
