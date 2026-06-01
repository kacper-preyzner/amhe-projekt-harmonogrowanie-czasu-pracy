"""Stałe i pomocnicze funkcje wynikające z Kodeksu pracy oraz stawek na rok 2026.

Cały model czasu operuje na **slotach 30-minutowych**. Doba ma 48 slotów
(indeksy 0..47, slot ``s`` to przedział ``[s*30 min, (s+1)*30 min)`` licząc od północy).
Wszystkie limity prawne wyrażono zarówno w godzinach (czytelność), jak i w slotach
(obliczenia).

Źródła reguł: Kodeks pracy (czas pracy, odpoczynki, praca nocna, dodatki) oraz
ogłoszone stawki minimalne na 2026 r. (4806 zł/mies., 31,40 zł/godz.).
"""

from __future__ import annotations

import numpy as np

# --- dyskretyzacja czasu ------------------------------------------------------

#: długość pojedynczego slotu w minutach
SLOT_MINUTES: int = 30
#: liczba slotów w godzinie
SLOTS_PER_HOUR: int = 60 // SLOT_MINUTES  # 2
#: liczba slotów w dobie (praca 24/7)
SLOTS_PER_DAY: int = 24 * SLOTS_PER_HOUR  # 48


def hours_to_slots(hours: float) -> int:
    """Zamienia godziny na liczbę slotów (zaokrąglenie do najbliższego)."""
    return int(round(hours * SLOTS_PER_HOUR))


def slots_to_hours(slots: int) -> float:
    """Zamienia liczbę slotów na godziny."""
    return slots / SLOTS_PER_HOUR


# --- twarde ograniczenia czasu pracy ------------------------------------------

#: maksymalny dobowy czas pracy (8 h)
MAX_DAILY_HOURS: int = 8
MAX_DAILY_SLOTS: int = hours_to_slots(MAX_DAILY_HOURS)  # 16

#: maksymalny przeciętny tygodniowy czas pracy (40 h)
MAX_WEEKLY_HOURS: int = 40
MAX_WEEKLY_SLOTS: int = hours_to_slots(MAX_WEEKLY_HOURS)  # 80

#: minimalny nieprzerwany odpoczynek dobowy (11 h)
MIN_DAILY_REST_HOURS: int = 11
MIN_DAILY_REST_SLOTS: int = hours_to_slots(MIN_DAILY_REST_HOURS)  # 22

#: minimalny nieprzerwany odpoczynek tygodniowy (35 h)
MIN_WEEKLY_REST_HOURS: int = 35
MIN_WEEKLY_REST_SLOTS: int = hours_to_slots(MIN_WEEKLY_REST_HOURS)  # 70

#: liczba dni w tygodniu rozliczeniowym
DAYS_PER_WEEK: int = 7


# --- praca nocna (21:00–7:00) -------------------------------------------------

NIGHT_START_HOUR: int = 21
NIGHT_END_HOUR: int = 7
NIGHT_START_SLOT: int = NIGHT_START_HOUR * SLOTS_PER_HOUR  # 42
NIGHT_END_SLOT: int = NIGHT_END_HOUR * SLOTS_PER_HOUR      # 14


def night_mask() -> np.ndarray:
    """Maska logiczna długości :data:`SLOTS_PER_DAY` — które sloty są nocne (21:00–7:00)."""
    s = np.arange(SLOTS_PER_DAY)
    return (s >= NIGHT_START_SLOT) | (s < NIGHT_END_SLOT)


def is_night_slot(slot_in_day: int) -> bool:
    """Czy dany slot doby należy do pory nocnej (21:00–7:00)."""
    return slot_in_day >= NIGHT_START_SLOT or slot_in_day < NIGHT_END_SLOT


# --- przerwy w pracy ----------------------------------------------------------

BREAK_MINUTES: int = 15


def breaks_required(shift_hours: float) -> int:
    """Liczba 15-minutowych przerw należnych przy zmianie o danej długości.

    ≥6 h → 1 przerwa, >9 h → 2, >16 h → 3. Przerwy są wliczane do czasu pracy
    i przy granulacji 30-minutowej zawsze się „mieszczą", więc nie powodują
    nielegalności grafiku — funkcja służy do sprawozdawczości i kompletności modelu.
    """
    if shift_hours > 16:
        return 3
    if shift_hours > 9:
        return 2
    if shift_hours >= 6:
        return 1
    return 0


# --- wynagrodzenie (stawki 2026) ----------------------------------------------

#: minimalne wynagrodzenie miesięczne (2026), w zł
MIN_WAGE_MONTHLY: float = 4806.0
#: minimalna stawka godzinowa (2026), w zł/h
MIN_HOURLY: float = 31.40
#: podstawowy koszt pracy za jeden slot (30 min)
BASE_PER_SLOT: float = MIN_HOURLY / SLOTS_PER_HOUR  # 15.70

#: dodatek za pracę w nocy: 20% stawki minimalnej za każdą godzinę nocną
NIGHT_BONUS_RATE: float = 0.20
NIGHT_BONUS_PER_SLOT: float = NIGHT_BONUS_RATE * MIN_HOURLY / SLOTS_PER_HOUR

#: dodatek za pracę w niedzielę/święto: +100% (podwójna stawka)
SUNDAY_HOLIDAY_BONUS_RATE: float = 1.00
SUNDAY_HOLIDAY_BONUS_PER_SLOT: float = SUNDAY_HOLIDAY_BONUS_RATE * MIN_HOURLY / SLOTS_PER_HOUR

#: dodatki za nadgodziny (domyślnie nadgodziny są zabronione twardym limitem 8 h/dobę)
OVERTIME_BONUS_RATE: float = 0.50
