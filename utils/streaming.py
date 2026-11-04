"""Bounded producer/consumer helpers for streaming record batches."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")
_SENTINEL = object()


def prefetch(source: Iterable[T], depth: int = 4) -> Iterator[T]:
    """Iterate ``source`` on a background thread, keeping at most ``depth``
    items in flight. Parsing/decoding (reader thread) overlaps with the
    consumer (e.g. the database writer) while memory stays bounded."""
    q: queue.Queue = queue.Queue(maxsize=max(1, depth))
    stop = threading.Event()

    def _run() -> None:
        try:
            for item in source:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            q.put(_SENTINEL)
        except BaseException as exc:  # forward producer errors to the consumer
            q.put(exc)

    t = threading.Thread(target=_run, name="prefetch", daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
