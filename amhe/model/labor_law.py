"""Stałe Kodeksu pracy i stawki na rok 2026 — czas w slotach 30-minutowych."""

from __future__ import annotations

import numpy as np

SLOT_MINUTES: int = 30
SLOTS_PER_HOUR: int = 60 // SLOT_MINUTES   # 2
SLOTS_PER_DAY: int = 24 * SLOTS_PER_HOUR   # 48


def hours_to_slots(hours: float) -> int:
    return int(round(hours * SLOTS_PER_HOUR))


def slots_to_hours(slots: int) -> float:
    return slots / SLOTS_PER_HOUR


# limity czasu pracy
MAX_DAILY_HOURS: int = 8
MAX_DAILY_SLOTS: int = hours_to_slots(MAX_DAILY_HOURS)    # 16

MAX_WEEKLY_HOURS: int = 40
MAX_WEEKLY_SLOTS: int = hours_to_slots(MAX_WEEKLY_HOURS)  # 80

# minimalne odpoczynki
MIN_DAILY_REST_HOURS: int = 11
MIN_DAILY_REST_SLOTS: int = hours_to_slots(MIN_DAILY_REST_HOURS)   # 22

MIN_WEEKLY_REST_HOURS: int = 35
MIN_WEEKLY_REST_SLOTS: int = hours_to_slots(MIN_WEEKLY_REST_HOURS) # 70

DAYS_PER_WEEK: int = 7

# pora nocna: 21:00–7:00
NIGHT_START_HOUR: int = 21
NIGHT_END_HOUR: int = 7
NIGHT_START_SLOT: int = NIGHT_START_HOUR * SLOTS_PER_HOUR  # 42
NIGHT_END_SLOT: int = NIGHT_END_HOUR * SLOTS_PER_HOUR      # 14


def night_mask() -> np.ndarray:
    """Maska bool — które sloty doby są nocne (21:00–7:00)."""
    s = np.arange(SLOTS_PER_DAY)
    return (s >= NIGHT_START_SLOT) | (s < NIGHT_END_SLOT)


def is_night_slot(slot_in_day: int) -> bool:
    return slot_in_day >= NIGHT_START_SLOT or slot_in_day < NIGHT_END_SLOT


BREAK_MINUTES: int = 15


def breaks_required(shift_hours: float) -> int:
    """Liczba 15-min przerw wliczanych do czasu pracy: >=6h->1, >9h->2, >16h->3."""
    if shift_hours > 16:
        return 3
    if shift_hours > 9:
        return 2
    if shift_hours >= 6:
        return 1
    return 0


# stawki 2026
MIN_WAGE_MONTHLY: float = 4806.0
MIN_HOURLY: float = 31.40
BASE_PER_SLOT: float = MIN_HOURLY / SLOTS_PER_HOUR  # 15.70 zł / slot

NIGHT_BONUS_RATE: float = 0.20
NIGHT_BONUS_PER_SLOT: float = NIGHT_BONUS_RATE * MIN_HOURLY / SLOTS_PER_HOUR

SUNDAY_HOLIDAY_BONUS_RATE: float = 1.00
SUNDAY_HOLIDAY_BONUS_PER_SLOT: float = SUNDAY_HOLIDAY_BONUS_RATE * MIN_HOURLY / SLOTS_PER_HOUR

OVERTIME_BONUS_RATE: float = 0.50
