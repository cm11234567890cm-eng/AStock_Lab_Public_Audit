"""Minimal, explicit A-share trading calendar used for data freshness checks.

It is deliberately independent from a computer's current date.  The 2026
National Day closure is included so 2026-10-02 resolves to 2026-09-30.
Before using another year in production, import and version an exchange
calendar into AStock_Lab/data/metadata; never infer holidays from missing bars.
"""
from __future__ import annotations

from datetime import date, timedelta

# Statutory closures needed by the Phase 1.5 calendar.  Weekends are handled
# separately.  The set is explicit and auditable rather than a hidden heuristic.
CN_A_SHARE_HOLIDAYS_2026 = frozenset(
    [date(2026, 1, day) for day in range(1, 4)]
    + [date(2026, 2, day) for day in range(15, 24)]
    + [date(2026, 4, day) for day in range(4, 7)]
    + [date(2026, 5, day) for day in range(1, 6)]
    + [date(2026, 6, day) for day in range(19, 22)]
    + [date(2026, 9, day) for day in range(25, 28)]
    + [date(2026, 10, day) for day in range(1, 8)]
)


def is_trading_day(value: date) -> bool:
    """True only for a weekday that is not in the explicit closure calendar."""
    return value.weekday() < 5 and value not in CN_A_SHARE_HOLIDAYS_2026


def previous_trade_date(value: date) -> date:
    """Return the most recent trading date on or before *value*."""
    candidate = value
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


EXPECTED_LATEST_TRADE_DATE = previous_trade_date(date(2026, 10, 2))
