"""Lokalna reoptymalizacja po absencji pracownika — łatanie grafiku bez przebudowy całości."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from amhe.model import labor_law as law
from amhe.model.constraints import is_legal
from amhe.model.objectives import coverage_shortfall, cost_objective
from amhe.model.schedule import ProblemInstance, Schedule, coverage
from amhe.repair import repair

PATCH_SHIFT_SLOTS = law.hours_to_slots(4)


@dataclass
class ReoptResult:
    schedule: Schedule
    absent_employee: int
    shortfall_before: int
    shortfall_after: int
    cost_before: float
    cost_after: float
    recovered_slots: int
    wall_time: float


def remove_employee(schedule: Schedule, employee: int) -> Schedule:
    """Kopia grafiku z wyzerowanymi zmianami danego pracownika."""
    s = schedule.copy()
    s.start[employee, :] = 0
    s.length[employee, :] = 0
    return s


def reoptimize_absence(instance: ProblemInstance, schedule: Schedule,
                       absent: int, skill_threshold: float = 0.0) -> ReoptResult:
    """Łata grafik po absencji pracownika absent — szuka zastępców dzień po dniu."""
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
    """Dokłada zmiany wolnym, kompetentnym zastępcom tam, gdzie jest niedobór."""
    cov = coverage(instance, schedule)
    deficit = instance.demand[day] - cov[day]
    if deficit.max() <= 0:
        return

    candidates = [
        e for e, emp in enumerate(instance.employees)
        if e != absent and schedule.length[e, day] == 0
        and emp.skill >= skill_threshold
    ]
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

        if (is_legal(instance, trial)
                and coverage_shortfall(instance, trial)
                < coverage_shortfall(instance, schedule)):
            schedule.start[:] = trial.start
            schedule.length[:] = trial.length
            cov = coverage(instance, schedule)
            deficit = instance.demand[day] - cov[day]
