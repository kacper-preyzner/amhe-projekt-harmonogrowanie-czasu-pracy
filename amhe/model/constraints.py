"""Twarde ograniczenia Kodeksu pracy i sprawdzanie legalnosci grafiku.

Operator naprawczy (:mod:`amhe.repair`) gwarantuje, ze kazdy oceniany grafik jest
legalny; funkcje z tego modulu sluza do *weryfikacji* (testy, metryka "% spelnionych
ograniczen") oraz jako definicja referencyjna regul.

Sprawdzane ograniczenia:
    * struktura zmiany (miesci sie w dobie, dlugosc >= 0),
    * dobowy czas pracy <= 8 h,
    * tygodniowy czas pracy <= 40 h,
    * odpoczynek dobowy >= 11 h (miedzy kolejnymi zmianami, czas ciagly),
    * odpoczynek tygodniowy >= 35 h (ciagly, w kazdym tygodniu).

Praca nocna <= 8 h jest konsekwencja limitu dobowego, wiec nie wymaga osobnego testu.
"""

from __future__ import annotations

from amhe.model import labor_law as law
from amhe.model.schedule import ProblemInstance, Schedule, absolute_shifts


def max_free_gap(shifts, window_start, window_end):
    """Najdluzszy ciagly okres bez pracy w oknie [window_start, window_end) (w slotach).

    Uwzglednia wolne fragmenty przed pierwsza i po ostatniej zmianie w oknie.
    """
    relevant = [(s, e) for (s, e) in shifts if s < window_end and e > window_start]
    relevant.sort()
    if not relevant:
        return window_end - window_start
    gap = relevant[0][0] - window_start
    for (s_prev, e_prev), (s_next, e_next) in zip(relevant, relevant[1:]):
        gap = max(gap, s_next - e_prev)
    gap = max(gap, window_end - relevant[-1][1])
    return gap


def violations(instance: ProblemInstance, schedule: Schedule) -> dict:
    """Zwraca slownik liczby naruszen poszczegolnych twardych ograniczen.

    Wszystkie wartosci rowne 0 <=> grafik w pelni legalny.
    """
    counts = {
        "shift_bounds": 0,
        "daily_hours": 0,
        "weekly_hours": 0,
        "daily_rest": 0,
        "weekly_rest": 0,
    }

    # struktura zmiany i dobowy czas pracy
    for e in range(schedule.n_employees):
        for d in range(schedule.n_days):
            ln = int(schedule.length[e, d])
            s = int(schedule.start[e, d])
            if ln < 0 or (ln > 0 and (s < 0 or s + ln > law.SLOTS_PER_DAY)):
                counts["shift_bounds"] += 1
            if ln > law.MAX_DAILY_SLOTS:
                counts["daily_hours"] += 1

    # tygodniowy czas pracy
    for e in range(schedule.n_employees):
        for w in range(instance.n_weeks):
            days = [d for d in range(schedule.n_days) if d // law.DAYS_PER_WEEK == w]
            worked = sum(max(int(schedule.length[e, d]), 0) for d in days)
            if worked > law.MAX_WEEKLY_SLOTS:
                counts["weekly_hours"] += 1

    # odpoczynek dobowy (przerwy miedzy kolejnymi zmianami w czasie ciaglym)
    for e in range(schedule.n_employees):
        shifts = absolute_shifts(schedule, e)
        for (s_prev, end_prev), (start_next, e_next) in zip(shifts, shifts[1:]):
            if start_next - end_prev < law.MIN_DAILY_REST_SLOTS:
                counts["daily_rest"] += 1

    # odpoczynek tygodniowy
    for e in range(schedule.n_employees):
        shifts = absolute_shifts(schedule, e)
        for w in range(instance.n_weeks):
            days = [d for d in range(schedule.n_days) if d // law.DAYS_PER_WEEK == w]
            # odpoczynek tygodniowy egzekwujemy tylko dla pelnych 7-dniowych tygodni
            if len(days) < law.DAYS_PER_WEEK:
                continue
            ws = days[0] * law.SLOTS_PER_DAY
            we = (days[-1] + 1) * law.SLOTS_PER_DAY
            if max_free_gap(shifts, ws, we) < law.MIN_WEEKLY_REST_SLOTS:
                counts["weekly_rest"] += 1

    return counts


def is_legal(instance: ProblemInstance, schedule: Schedule) -> bool:
    """Czy grafik spelnia wszystkie twarde ograniczenia."""
    return all(v == 0 for v in violations(instance, schedule).values())


def satisfaction_rate(instance: ProblemInstance, schedule: Schedule) -> float:
    """Odsetek spelnionych jednostek twardych ograniczen (metryka kontrolna, cel 1.0).

    Dla kazdego pracownika liczymy zbior sprawdzen: struktura + dobowy limit (per dzien),
    tygodniowy limit (per tydzien), odpoczynek tygodniowy (per tydzien), odpoczynek
    dobowy (per para kolejnych zmian). Zwraca (spelnione / wszystkie).
    """
    counts = violations(instance, schedule)
    full_weeks = sum(
        1
        for w in range(instance.n_weeks)
        if len([d for d in range(schedule.n_days) if d // law.DAYS_PER_WEEK == w])
        == law.DAYS_PER_WEEK
    )
    total = 0
    total += 2 * schedule.n_employees * schedule.n_days          # struktura + dobowy limit
    total += schedule.n_employees * instance.n_weeks             # tygodniowy limit godzin
    total += schedule.n_employees * full_weeks                   # odpoczynek tygodniowy
    pairs = sum(max(len(absolute_shifts(schedule, e)) - 1, 0)
                for e in range(schedule.n_employees))
    total += pairs                                               # odpoczynek dobowy
    if total == 0:
        return 1.0
    violated = sum(counts.values())
    return max(0.0, 1.0 - violated / total)
