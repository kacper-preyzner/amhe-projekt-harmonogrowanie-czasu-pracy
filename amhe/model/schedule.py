"""Reprezentacja grafiku, pracownika i instancji problemu.

Kodowanie grafiku jest zwarte i z natury wymusza ciągłość zmian: dla każdej pary
(pracownik, dzień) przechowujemy **początek** zmiany i jej **długość** w slotach.
Długość 0 oznacza dzień wolny. Dzięki temu zmiana jest zawsze spójnym blokiem,
a operatory genetyczne i naprawcze działają na małych macierzach liczb całkowitych.

Założenia upraszczające (opisane w raporcie):
    * pojedyncza zmiana mieści się w obrębie jednej doby (nie przechodzi przez północ),
    * horyzont planowania zaczyna się w poniedziałek (tydzień = ``dzień // 7``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amhe.model import labor_law as law

# --- preferencje pory dnia ----------------------------------------------------

#: preferowane okna godzinowe (godzina_od, godzina_do); ``None`` = brak preferencji
PREFERENCE_HOURS: dict[str, tuple[int, int] | None] = {
    "rano": (6, 14),       # zmiana poranna
    "dzien": (8, 16),      # zmiana dzienna
    "wieczor": (14, 22),   # zmiana popołudniowo-wieczorna
    "noc": (22, 30),       # zmiana nocna (22:00–6:00, z przejściem przez północ)
    "dowolna": None,       # pracownik bez preferencji
}


def preference_mask(preference: str) -> np.ndarray:
    """Maska logiczna długości :data:`law.SLOTS_PER_DAY` — sloty zgodne z preferencją.

    Dla preferencji ``"dowolna"`` zwraca same ``True`` (żaden slot nie jest karany).
    Okna z przejściem przez północ (np. noc 22:00–6:00) są obsłużone modulo doba.
    """
    window = PREFERENCE_HOURS.get(preference)
    if window is None:
        return np.ones(law.SLOTS_PER_DAY, dtype=bool)
    h_start, h_end = window
    s_start = (h_start * law.SLOTS_PER_HOUR) % law.SLOTS_PER_DAY
    s_end = (h_end * law.SLOTS_PER_HOUR) % law.SLOTS_PER_DAY
    s = np.arange(law.SLOTS_PER_DAY)
    if s_start <= s_end:
        return (s >= s_start) & (s < s_end)
    # okno przechodzi przez północ
    return (s >= s_start) | (s < s_end)


@dataclass
class Employee:
    """Pracownik call center.

    Atrybuty:
        id:          identyfikator,
        name:        nazwa (do wizualizacji),
        preference:  klucz preferencji pory dnia (patrz :data:`PREFERENCE_HOURS`),
        skill:       poziom kompetencji (do reoptymalizacji / dopasowania), domyślnie 1.0.
    """

    id: int
    name: str
    preference: str = "dowolna"
    skill: float = 1.0

    @property
    def pref_mask(self) -> np.ndarray:
        """Maska slotów zgodnych z preferencją pracownika."""
        return preference_mask(self.preference)


@dataclass
class ProblemInstance:
    """Pełny opis instancji problemu harmonogramowania.

    Atrybuty:
        employees:   lista pracowników,
        n_days:      długość horyzontu w dniach,
        demand:      macierz (n_days, SLOTS_PER_DAY) — wymagana liczba agentów w slocie,
        day_of_week: wektor (n_days,) — dzień tygodnia (0=pon … 6=niedz),
        is_holiday:  wektor (n_days,) bool — czy dzień jest świętem,
        name:        nazwa instancji (do raportów).
    """

    employees: list[Employee]
    n_days: int
    demand: np.ndarray
    day_of_week: np.ndarray
    is_holiday: np.ndarray
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
        """Liczba tygodni rozliczeniowych (horyzont zaczyna się w poniedziałek)."""
        return (self.n_days + law.DAYS_PER_WEEK - 1) // law.DAYS_PER_WEEK

    def is_sunday_or_holiday(self, day: int) -> bool:
        """Czy dany dzień jest niedzielą lub świętem (dodatek 100%)."""
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
        """Pusty grafik (wszyscy mają wolne)."""
        return cls(
            start=np.zeros((n_employees, n_days), dtype=int),
            length=np.zeros((n_employees, n_days), dtype=int),
        )


# --- funkcje pomocnicze na grafiku --------------------------------------------


def shift_end(start: int, length: int) -> int:
    """Slot zakończenia zmiany (wyłącznie) w obrębie doby."""
    return start + length


def coverage(instance: ProblemInstance, schedule: Schedule) -> np.ndarray:
    """Obsada w każdym slocie: macierz (n_days, SLOTS_PER_DAY) liczb pracowników.

    Liczy, ilu pracowników pracuje w danym slocie danego dnia. Granulacja
    30-minutowa; ewentualne 15-minutowe przerwy są pomijane przy liczeniu obsady
    (uproszczenie opisane w raporcie).
    """
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
    """Lista zmian pracownika ``e`` jako (start_abs, end_abs) w slotach od początku horyzontu.

    Posortowana rosnąco po czasie; dni wolne pomijane. Używana do sprawdzania
    odpoczynków dobowych i tygodniowych (liczonych w czasie ciągłym).
    """
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
    """Suma przepracowanych slotów na (pracownik, tydzień): macierz (n_employees, n_weeks)."""
    n_weeks = (schedule.n_days + law.DAYS_PER_WEEK - 1) // law.DAYS_PER_WEEK
    out = np.zeros((schedule.n_employees, n_weeks), dtype=int)
    for d in range(schedule.n_days):
        w = d // law.DAYS_PER_WEEK
        out[:, w] += np.maximum(schedule.length[:, d], 0)
    return out
