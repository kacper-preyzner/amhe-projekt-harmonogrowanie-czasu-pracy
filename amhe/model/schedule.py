"""Reprezentacja grafiku, pracownika i instancji problemu."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from amhe.model import labor_law as law

# preferowane okna godzinowe (h_od, h_do); None = brak preferencji
PREFERENCE_HOURS: dict[str, tuple[int, int] | None] = {
    "rano":    (6, 14),
    "dzien":   (8, 16),
    "wieczor": (14, 22),
    "noc":     (22, 30),  # przejście przez północ
    "dowolna": None,
}


def preference_mask(preference: str) -> np.ndarray:
    """Maska bool slotów zgodnych z preferencją."""
    window = PREFERENCE_HOURS.get(preference)
    if window is None:
        return np.ones(law.SLOTS_PER_DAY, dtype=bool)
    h_start, h_end = window
    s_start = (h_start * law.SLOTS_PER_HOUR) % law.SLOTS_PER_DAY
    s_end = (h_end * law.SLOTS_PER_HOUR) % law.SLOTS_PER_DAY
    s = np.arange(law.SLOTS_PER_DAY)
    if s_start <= s_end:
        return (s >= s_start) & (s < s_end)
    return (s >= s_start) | (s < s_end)


@dataclass
class Employee:
    id: int
    name: str
    preference: str = "dowolna"
    skill: float = 1.0

    @property
    def pref_mask(self) -> np.ndarray:
        return preference_mask(self.preference)


@dataclass
class ProblemInstance:
    employees: list[Employee]
    n_days: int
    demand: np.ndarray        # (n_days, SLOTS_PER_DAY)
    day_of_week: np.ndarray   # (n_days,) 0=pon..6=niedz
    is_holiday: np.ndarray    # (n_days,) bool
    name: str = "instance"

    def __post_init__(self) -> None:
        self.demand = np.asarray(self.demand, dtype=int)
        self.day_of_week = np.asarray(self.day_of_week, dtype=int)
        self.is_holiday = np.asarray(self.is_holiday, dtype=bool)
        if self.demand.shape != (self.n_days, law.SLOTS_PER_DAY):
            raise ValueError(
                f"demand ma kształt {self.demand.shape}, oczekiwano "
                f"{(self.n_days, law.SLOTS_PER_DAY)}"
            )

    @property
    def n_employees(self) -> int:
        return len(self.employees)

    @property
    def slots_per_day(self) -> int:
        return law.SLOTS_PER_DAY

    @property
    def n_weeks(self) -> int:
        return (self.n_days + law.DAYS_PER_WEEK - 1) // law.DAYS_PER_WEEK

    def is_sunday_or_holiday(self, day: int) -> bool:
        return bool(self.day_of_week[day] == 6 or self.is_holiday[day])


@dataclass
class Schedule:
    """Grafik (genotyp): macierze początków i długości zmian.

    Atrybuty:
        start:  macierz (n_employees, n_days) — indeks slotu początku zmiany w dobie,
        length: macierz (n_employees, n_days) — długość zmiany w slotach (0 = wolne).
    """
    start: np.ndarray
    length: np.ndarray

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=int)
        self.length = np.asarray(self.length, dtype=int)
        if self.start.shape != self.length.shape:
            raise ValueError("start i length muszą mieć ten sam kształt")

    @property
    def n_employees(self) -> int:
        return self.start.shape[0]

    @property
    def n_days(self) -> int:
        return self.start.shape[1]

    def copy(self) -> "Schedule":
        return Schedule(self.start.copy(), self.length.copy())

    @classmethod
    def empty(cls, n_employees: int, n_days: int) -> "Schedule":
        return cls(
            start=np.zeros((n_employees, n_days), dtype=int),
            length=np.zeros((n_employees, n_days), dtype=int),
        )


def shift_end(start: int, length: int) -> int:
    return start + length


def coverage(instance: ProblemInstance, schedule: Schedule) -> np.ndarray:
    """Obsada w każdym slocie: macierz (n_days, SLOTS_PER_DAY)."""
    cov = np.zeros((instance.n_days, law.SLOTS_PER_DAY), dtype=int)
    for e in range(schedule.n_employees):
        for d in range(schedule.n_days):
            ln = int(schedule.length[e, d])
            if ln <= 0:
                continue
            s = int(schedule.start[e, d])
            cov[d, s : s + ln] += 1
    return cov


def absolute_shifts(schedule: Schedule, e: int) -> list[tuple[int, int]]:
    """Zmiany pracownika e jako (start_abs, end_abs) w slotach od początku horyzontu."""
    shifts: list[tuple[int, int]] = []
    for d in range(schedule.n_days):
        ln = int(schedule.length[e, d])
        if ln <= 0:
            continue
        s = int(schedule.start[e, d])
        base = d * law.SLOTS_PER_DAY
        shifts.append((base + s, base + s + ln))
    shifts.sort()
    return shifts


def weekly_worked_slots(schedule: Schedule) -> np.ndarray:
    """Suma przepracowanych slotów: macierz (n_employees, n_weeks)."""
    n_weeks = (schedule.n_days + law.DAYS_PER_WEEK - 1) // law.DAYS_PER_WEEK
    out = np.zeros((schedule.n_employees, n_weeks), dtype=int)
    for d in range(schedule.n_days):
        w = d // law.DAYS_PER_WEEK
        out[:, w] += np.maximum(schedule.length[:, d], 0)
    return out
