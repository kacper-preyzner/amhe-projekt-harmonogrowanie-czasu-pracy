"""Kryteria oceny grafiku: koszt oraz kara za niedopasowanie preferencji.

Optymalizacja jest dwukryterialna (minimalizacja obu wartosci):

1. **Koszt** = wynagrodzenia (podstawa + dodatki: nocny, niedzielny/swiateczny)
   + kara za niedobor obsady wzgledem popytu (utracone polaczenia). Nadmiarowa
   obsada nie wymaga osobnej kary, bo jest naturalnie zniechecana przez koszt plac.
2. **Kara preferencji** = liczba przepracowanych slotow poza preferowanym przez
   pracownika oknem pory dnia (pracownik z preferencja "dowolna" nigdy nie jest karany).

Poniewaz operator naprawczy zapewnia legalnosc, front Pareto jest czystym
kompromisem koszt x dopasowanie preferencji.
"""

from __future__ import annotations

import numpy as np

from amhe.model import labor_law as law
from amhe.model.schedule import ProblemInstance, Schedule, coverage

#: kara za jeden brakujacy agento-slot (utracona obsluga); wyrazona w zl,
#: dobrana powyzej kosztu obsadzenia slotu, by oplacalo sie pokrywac popyt
UNDERSTAFF_PENALTY_PER_SLOT: float = 3.0 * law.BASE_PER_SLOT


def wage_cost(instance: ProblemInstance, schedule: Schedule) -> float:
    """Calkowity koszt wynagrodzen (podstawa + dodatki nocny i niedzielny/swiateczny)."""
    night = law.night_mask()
    total = 0.0
    for e in range(schedule.n_employees):
        for d in range(schedule.n_days):
            ln = int(schedule.length[e, d])
            if ln <= 0:
                continue
            s = int(schedule.start[e, d])
            total += ln * law.BASE_PER_SLOT
            total += int(night[s : s + ln].sum()) * law.NIGHT_BONUS_PER_SLOT
            if instance.is_sunday_or_holiday(d):
                total += ln * law.SUNDAY_HOLIDAY_BONUS_PER_SLOT
    return total


def coverage_shortfall(instance: ProblemInstance, schedule: Schedule) -> int:
    """Calkowity niedobor obsady (suma brakujacych agento-slotow wzgledem popytu)."""
    cov = coverage(instance, schedule)
    return int(np.maximum(instance.demand - cov, 0).sum())


def coverage_surplus(instance: ProblemInstance, schedule: Schedule) -> int:
    """Calkowita nadmiarowa obsada (suma agento-slotow ponad popyt)."""
    cov = coverage(instance, schedule)
    return int(np.maximum(cov - instance.demand, 0).sum())


def cost_objective(
    instance: ProblemInstance,
    schedule: Schedule,
    understaff_penalty: float = UNDERSTAFF_PENALTY_PER_SLOT,
) -> float:
    """Kryterium kosztu: wynagrodzenia + kara za niedobor obsady."""
    return wage_cost(instance, schedule) + understaff_penalty * coverage_shortfall(
        instance, schedule
    )


def preference_penalty(instance: ProblemInstance, schedule: Schedule) -> float:
    """Kryterium preferencji: liczba przepracowanych slotow poza preferowanym oknem."""
    penalty = 0.0
    for e, emp in enumerate(instance.employees):
        mask = emp.pref_mask
        for d in range(schedule.n_days):
            ln = int(schedule.length[e, d])
            if ln <= 0:
                continue
            s = int(schedule.start[e, d])
            penalty += int((~mask[s : s + ln]).sum())
    return penalty


def evaluate(
    instance: ProblemInstance,
    schedule: Schedule,
    understaff_penalty: float = UNDERSTAFF_PENALTY_PER_SLOT,
) -> np.ndarray:
    """Wektor kryteriow [koszt, kara_preferencji] (do minimalizacji przez NSGA-II)."""
    return np.array(
        [
            cost_objective(instance, schedule, understaff_penalty),
            preference_penalty(instance, schedule),
        ],
        dtype=float,
    )


def breakdown(instance: ProblemInstance, schedule: Schedule) -> dict:
    """Rozbicie kryteriow na skladniki (do raportow i analizy)."""
    wages = wage_cost(instance, schedule)
    shortfall = coverage_shortfall(instance, schedule)
    return {
        "wages": wages,
        "shortfall_slots": shortfall,
        "surplus_slots": coverage_surplus(instance, schedule),
        "understaff_penalty": UNDERSTAFF_PENALTY_PER_SLOT * shortfall,
        "cost": wages + UNDERSTAFF_PENALTY_PER_SLOT * shortfall,
        "preference_penalty": preference_penalty(instance, schedule),
        "worked_slots": int(np.maximum(schedule.length, 0).sum()),
    }
