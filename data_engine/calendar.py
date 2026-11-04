"""Vectorised Indian banking calendar utilities.

Non-business days follow the RBI convention for scheduled banks: Sundays,
2nd and 4th Saturdays, and gazetted bank holidays. All functions operate on
``numpy.datetime64[D]`` arrays so they scale to portfolio-sized inputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np

# Movable festival holidays (sample national calendar; extend per state/year).
_MOVABLE: dict[str, list[str]] = {
    "Holi": ["2024-03-25", "2025-03-14", "2026-03-04", "2027-03-22", "2028-03-11"],
    "Good Friday": ["2024-03-29", "2025-04-18", "2026-04-03", "2027-03-26", "2028-04-14"],
    "Id-ul-Fitr": ["2024-04-11", "2025-03-31", "2026-03-20", "2027-03-10", "2028-02-27"],
    "Ganesh Chaturthi": ["2024-09-07", "2025-08-27", "2026-09-14", "2027-09-04", "2028-08-23"],
    "Dussehra": ["2024-10-12", "2025-10-02", "2026-10-20", "2027-10-09", "2028-09-27"],
    "Diwali": ["2024-11-01", "2025-10-21", "2026-11-08", "2027-10-29", "2028-10-17"],
}
_FIXED: list[tuple[int, int, str]] = [
    (1, 26, "Republic Day"),
    (5, 1, "Maharashtra Day / Labour Day"),
    (8, 15, "Independence Day"),
    (10, 2, "Gandhi Jayanti"),
    (12, 25, "Christmas"),
]


def bank_holidays(start: date, end: date) -> list[tuple[date, str]]:
    """Gazetted holidays in ``[start, end]`` (excludes weekends)."""
    out: dict[date, str] = {}
    for year in range(start.year, end.year + 1):
        for month, day, name in _FIXED:
            out[date(year, month, day)] = name
    for name, days in _MOVABLE.items():
        for d in days:
            out[date.fromisoformat(d)] = name
    return sorted((d, n) for d, n in out.items() if start <= d <= end)


def to_day_array(values: Iterable) -> np.ndarray:
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values).astype(
        "datetime64[D]"
    )


def day_range(start: date | np.datetime64, n_days: int) -> np.ndarray:
    return np.datetime64(start, "D") + np.arange(n_days)


def weekday(days: np.ndarray) -> np.ndarray:
    """Monday=0 ... Sunday=6 (1970-01-01 was a Thursday)."""
    return (days.astype("datetime64[D]").astype(np.int64) + 3) % 7


def day_of_month(days: np.ndarray) -> np.ndarray:
    d = days.astype("datetime64[D]")
    return (d - d.astype("datetime64[M]").astype("datetime64[D]")).astype(np.int64) + 1


def days_in_month(days: np.ndarray) -> np.ndarray:
    m = days.astype("datetime64[M]")
    return ((m + np.timedelta64(1, "M")).astype("datetime64[D]") - m.astype("datetime64[D]")).astype(np.int64)


def non_business_mask(days: np.ndarray, holidays: np.ndarray) -> np.ndarray:
    d = days.astype("datetime64[D]")
    wd = weekday(d)
    nth_week = (day_of_month(d) - 1) // 7 + 1
    second_fourth_sat = (wd == 5) & ((nth_week == 2) | (nth_week == 4))
    return (wd == 6) | second_fourth_sat | np.isin(d, holidays.astype("datetime64[D]"))


def roll_business_day(days: np.ndarray, holidays: np.ndarray, direction: int) -> np.ndarray:
    """Move non-business days to the previous (-1) or next (+1) business day."""
    d = days.astype("datetime64[D]").copy()
    for _ in range(10):
        m = non_business_mask(d, holidays)
        if not m.any():
            break
        d[m] += np.timedelta64(direction, "D")
    return d


def month_starts(start: date | np.datetime64, end: date | np.datetime64) -> np.ndarray:
    m0 = np.datetime64(start, "M")
    m1 = np.datetime64(end, "M")
    return np.arange(m0, m1 + np.timedelta64(1, "M")).astype("datetime64[D]")


def monthly_dates(
    day_of_month_target: np.ndarray,
    start: date | np.datetime64,
    end: date | np.datetime64,
    holidays: np.ndarray,
    roll: int,
) -> np.ndarray:
    """Per-account monthly dates, clipped to month length and business-day rolled.

    Returns ``[n_accounts, n_months]`` datetime64[D]; NaT where the target day
    is missing (``<= 0``) or the rolled date falls outside ``[start, end]``.
    """
    months = month_starts(start, end)
    dom = np.asarray(day_of_month_target, dtype=np.int64)[:, None]
    mlen = days_in_month(months)[None, :]
    raw = months[None, :] + (np.minimum(np.maximum(dom, 1), mlen) - 1).astype("timedelta64[D]")
    rolled = roll_business_day(raw.ravel(), holidays, roll).reshape(raw.shape)
    lo, hi = np.datetime64(start, "D"), np.datetime64(end, "D")
    out = rolled.astype("datetime64[D]")
    invalid = (dom <= 0) | (out < lo) | (out > hi)
    out = out.copy()
    out[np.broadcast_to(invalid, out.shape)] = np.datetime64("NaT", "D")
    return out
