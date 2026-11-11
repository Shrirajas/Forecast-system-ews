"""Every timestamp the API returns is IST with an explicit offset (the console used to mix UTC and local time)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Literal

IST = timezone(timedelta(hours=5, minutes=30), "IST")


def to_ist(value: datetime | str | None, naive: Literal["utc", "ist"] = "utc") -> str | None:
    """ISO string with +05:30. ``naive`` says what a tz-less input means: DB timestamps are naive UTC
    (``utcnow()``), job/training timestamps are naive local time (the machine runs on IST)."""
    if value is None or value == "":
        return None
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC if naive == "utc" else IST)
    return dt.astimezone(IST).isoformat(timespec="seconds")


def now_ist() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")
