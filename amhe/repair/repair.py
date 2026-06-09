"""Operator naprawczy — przekształca dowolny grafik w w pełni legalny.

Kolejność kroków: 
A) limit dobowy, 
B) odpoczynek dobowy,
C) tygodniowy limit godzin, 
D) odpoczynek tygodniowy.
"""

from __future__ import annotations

from amhe.model import labor_law as law
from amhe.model.constraints import max_free_gap
from amhe.model.schedule import Schedule, absolute_shifts


def repair(schedule: Schedule) -> Schedule:
    """Zwraca nową, legalną kopię grafiku."""
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
    """Krok A: długość w [0, 8h], zmiana mieści się w dobie."""
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
    """Krok B: >= 11h przerwy między kolejnymi zmianami."""
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
                length[e, d] = 0
                start[e, d] = 0
                continue
            st = new_st
            start[e, d] = st
            start_abs = d * law.SLOTS_PER_DAY + st
        prev_end = start_abs + ln


def _fix_weekly_hours(start, length, e, D, n_weeks):
    """Krok C: przytnij najdłuższe zmiany, aż tygodniowa suma <= 40h."""
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
    """Krok D: wymuś >= 35h ciągłego odpoczynku w pełnym tygodniu."""
    for w in range(n_weeks):
        days = [d for d in range(D) if d // law.DAYS_PER_WEEK == w]
        if len(days) < law.DAYS_PER_WEEK:
            continue
        ws = days[0] * law.SLOTS_PER_DAY
        we = (days[-1] + 1) * law.SLOTS_PER_DAY
        if max_free_gap(absolute_shifts(s, e), ws, we) >= law.MIN_WEEKLY_REST_SLOTS:
            continue
        # ustaw wolne w parze dni z najmniejszą sumą godzin
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
