"""Operator naprawczy ("straznik prawa").

Dowolny (potencjalnie nielegalny) grafik przekształca w grafik **w pelni legalny**,
naprawiajac kolejno twarde ograniczenia Kodeksu pracy. Naprawa jest deterministyczna
i dziala niezaleznie dla kazdego pracownika (ograniczenia twarde sa per-pracownik;
popyt jest kryterium miekkim, wiec nie jest tu uwzgledniany).

Kolejnosc krokow dobrano tak, by zaden pozniejszy nie psul wczesniejszego:

    A. struktura + limit dobowy  — przytnij start/dlugosc do prawidlowego zakresu i <= 8 h,
    B. odpoczynek dobowy          — przesun zmiane pozniej (lub usun), by zachowac >= 11 h,
    C. tygodniowy limit godzin    — przytnij najdluzsze zmiany, az suma <= 40 h,
    D. odpoczynek tygodniowy      — w pelnym tygodniu wymus 2 kolejne dni wolne (>= 35 h).

Kroki B–D jedynie skracaja/usuwaja prace, wiec nie wprowadzaja nowych naruszen
ograniczen sprawdzonych wczesniej. Wynik jest gwarantowanie legalny.
"""

from __future__ import annotations

from amhe.model import labor_law as law
from amhe.model.constraints import max_free_gap
from amhe.model.schedule import Schedule, absolute_shifts


def repair(schedule: Schedule) -> Schedule:
    """Zwraca nowa, legalna kopie grafiku (oryginal nie jest modyfikowany)."""
    s = schedule.copy()
    start, length = s.start, s.length
    E, D = s.n_employees, s.n_days
    n_weeks = (D + law.DAYS_PER_WEEK - 1) // law.DAYS_PER_WEEK

    for e in range(E):
        _fix_bounds_and_daily(start, length, e, D)
        _fix_daily_rest(start, length, e, D)
        _fix_weekly_hours(start, length, e, D, n_weeks)
        _fix_weekly_rest(s, start, length, e, D, n_weeks)

    return s


def _fix_bounds_and_daily(start, length, e, D):
    """Krok A: dlugosc w [0, 8 h], zmiana mieszczaca sie w dobie."""
    for d in range(D):
        ln = int(length[e, d])
        if ln <= 0:
            length[e, d] = 0
            start[e, d] = 0
            continue
        ln = min(ln, law.MAX_DAILY_SLOTS)
        st = max(0, min(int(start[e, d]), law.SLOTS_PER_DAY - ln))
        start[e, d] = st
        length[e, d] = ln


def _fix_daily_rest(start, length, e, D):
    """Krok B: zachowaj >= 11 h przerwy miedzy kolejnymi zmianami (czas ciagly)."""
    prev_end = -10 ** 9
    for d in range(D):
        ln = int(length[e, d])
        if ln <= 0:
            continue
        st = int(start[e, d])
        start_abs = d * law.SLOTS_PER_DAY + st
        required = prev_end + law.MIN_DAILY_REST_SLOTS
        if start_abs < required:
            new_st = required - d * law.SLOTS_PER_DAY
            if new_st + ln > law.SLOTS_PER_DAY:
                # nie da sie zmiescic z zachowaniem odpoczynku — usun zmiane
                length[e, d] = 0
                start[e, d] = 0
                continue
            st = new_st
            start[e, d] = st
            start_abs = d * law.SLOTS_PER_DAY + st
        prev_end = start_abs + ln


def _fix_weekly_hours(start, length, e, D, n_weeks):
    """Krok C: przytnij najdluzsze zmiany, az tygodniowa suma <= 40 h."""
    for w in range(n_weeks):
        days = [d for d in range(D) if d // law.DAYS_PER_WEEK == w]
        excess = sum(int(length[e, d]) for d in days) - law.MAX_WEEKLY_SLOTS
        if excess <= 0:
            continue
        for d in sorted(days, key=lambda dd: int(length[e, dd]), reverse=True):
            if excess <= 0:
                break
            cut = min(int(length[e, d]), excess)
            length[e, d] -= cut
            excess -= cut
            if length[e, d] == 0:
                start[e, d] = 0


def _fix_weekly_rest(s, start, length, e, D, n_weeks):
    """Krok D: w pelnym tygodniu wymus >= 35 h ciaglego odpoczynku.

    Jezeli go brak, ustawia wolne w parze kolejnych dni o najmniejszej sumie godzin
    pracy (2 pelne dni wolne to 96 slotow ciaglego odpoczynku, czyli >= 70).
    """
    for w in range(n_weeks):
        days = [d for d in range(D) if d // law.DAYS_PER_WEEK == w]
        if len(days) < law.DAYS_PER_WEEK:
            continue
        ws = days[0] * law.SLOTS_PER_DAY
        we = (days[-1] + 1) * law.SLOTS_PER_DAY
        if max_free_gap(absolute_shifts(s, e), ws, we) >= law.MIN_WEEKLY_REST_SLOTS:
            continue
        best_pair = None
        best_val = None
        for i in range(len(days) - 1):
            d1, d2 = days[i], days[i + 1]
            val = int(length[e, d1]) + int(length[e, d2])
            if best_val is None or val < best_val:
                best_val = val
                best_pair = (d1, d2)
        for d in best_pair:
            length[e, d] = 0
            start[e, d] = 0
