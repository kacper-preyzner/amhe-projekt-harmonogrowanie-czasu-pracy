"""Reoptymalizacja w czasie rzeczywistym ("pogotowie") po absencji pracownika.

Gdy w trakcie tygodnia pracownik zglasza niedyspozycyjnosc (np. L4), nie ukladamy
calego grafiku od nowa. Zamiast tego:

    1. usuwamy wszystkie zmiany nieobecnego pracownika (powstaje luka w pokryciu),
    2. lokalnie latamy luke: dla kazdego dnia, w ktorym powstal niedobor, szukamy
       wolnego, kompetentnego zastepcy i dokladamy mu zmiane pokrywajaca brakujace
       sloty — o ile nie narusza to twardych ograniczen (sprawdzane operatorem
       naprawczym i porownaniem pokrycia).

Mierzymy czas reoptymalizacji oraz jakosc (wzrost kosztu, przywrocone pokrycie) —
to dane do scenariusza 3 eksperymentow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from amhe.model import labor_law as law
from amhe.model.constraints import is_legal
from amhe.model.objectives import coverage_shortfall, cost_objective
from amhe.model.schedule import ProblemInstance, Schedule, coverage
from amhe.repair import repair

#: dlugosc zmiany zastepczej (sloty) — krotka, by latac punktowo
PATCH_SHIFT_SLOTS = law.hours_to_slots(4)


@dataclass
class ReoptResult:
    """Wynik lokalnej reoptymalizacji po absencji."""

    schedule: Schedule
    absent_employee: int
    shortfall_before: int       # niedobor tuz po usunieciu nieobecnego
    shortfall_after: int        # niedobor po zalataniu
    cost_before: float          # koszt grafiku przed absencja
    cost_after: float           # koszt grafiku po reoptymalizacji
    recovered_slots: int        # ile agento-slotow niedoboru udalo sie odzyskac
    wall_time: float            # czas reoptymalizacji (s)


def remove_employee(schedule: Schedule, employee: int) -> Schedule:
    """Zwraca kopie grafiku z wyzerowanymi zmianami danego pracownika (absencja)."""
    s = schedule.copy()
    s.start[employee, :] = 0
    s.length[employee, :] = 0
    return s


def reoptimize_absence(instance: ProblemInstance, schedule: Schedule,
                       absent: int, skill_threshold: float = 0.0) -> ReoptResult:
    """Lokalnie latamy grafik po absencji pracownika ``absent``.

    Args:
        instance:         instancja problemu,
        schedule:         (legalny) grafik bazowy sprzed absencji,
        absent:           indeks nieobecnego pracownika,
        skill_threshold:  minimalny poziom kompetencji zastepcy.
    """
    t0 = time.perf_counter()
    cost_before = cost_objective(instance, schedule)

    patched = remove_employee(schedule, absent)
    shortfall_before = coverage_shortfall(instance, patched)

    for d in range(instance.n_days):
        _patch_day(instance, patched, d, absent, skill_threshold)

    patched = repair(patched)
    assert is_legal(instance, patched)

    shortfall_after = coverage_shortfall(instance, patched)
    cost_after = cost_objective(instance, patched)
    wall = time.perf_counter() - t0

    return ReoptResult(
        schedule=patched,
        absent_employee=absent,
        shortfall_before=shortfall_before,
        shortfall_after=shortfall_after,
        cost_before=cost_before,
        cost_after=cost_after,
        recovered_slots=shortfall_before - shortfall_after,
        wall_time=wall,
    )


def _patch_day(instance: ProblemInstance, schedule: Schedule, day: int,
               absent: int, skill_threshold: float) -> None:
    """Lata niedobor w jednym dniu, dokladajac zmiany wolnym, kompetentnym zastepcom."""
    cov = coverage(instance, schedule)
    deficit = instance.demand[day] - cov[day]
    if deficit.max() <= 0:
        return

    # kandydaci: wolni tego dnia, o wystarczajacych kompetencjach, nie nieobecny
    candidates = [
        e for e, emp in enumerate(instance.employees)
        if e != absent and schedule.length[e, day] == 0
        and emp.skill >= skill_threshold
    ]
    # preferuj pracownikow o wyzszych kompetencjach
    candidates.sort(key=lambda e: -instance.employees[e].skill)

    while deficit.max() > 0 and candidates:
        slot = int(np.argmax(deficit))
        e = candidates.pop(0)
        ln = PATCH_SHIFT_SLOTS
        st = int(np.clip(slot - ln // 2, 0, law.SLOTS_PER_DAY - ln))

        trial = schedule.copy()
        trial.start[e, day] = st
        trial.length[e, day] = ln
        trial = repair(trial)

        # zaakceptuj tylko, jesli zmniejsza niedobor i pozostaje legalny
        if (is_legal(instance, trial)
                and coverage_shortfall(instance, trial)
                < coverage_shortfall(instance, schedule)):
            schedule.start[:] = trial.start
            schedule.length[:] = trial.length
            cov = coverage(instance, schedule)
            deficit = instance.demand[day] - cov[day]
