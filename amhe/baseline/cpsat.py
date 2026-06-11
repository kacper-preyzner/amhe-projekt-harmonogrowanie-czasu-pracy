"""Solver CP-SAT (OR-Tools) jako baseline — minimalizuje f1 na małych instancjach."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

from amhe.model import labor_law as law
from amhe.model.objectives import UNDERSTAFF_PENALTY_PER_SLOT
from amhe.model.schedule import ProblemInstance, Schedule

MIN_SHIFT_SLOTS = law.hours_to_slots(4)

# koszty w groszach — CP-SAT wymaga całkowitej funkcji celu
_BASE_GR = round(law.BASE_PER_SLOT * 100)
_NIGHT_GR = round(law.NIGHT_BONUS_PER_SLOT * 100)
_SUNDAY_GR = round(law.SUNDAY_HOLIDAY_BONUS_PER_SLOT * 100)
_UNDERSTAFF_GR = round(UNDERSTAFF_PENALTY_PER_SLOT * 100)


@dataclass
class CPSATResult:
    schedule: Schedule
    cost: float
    status: str
    wall_time: float
    is_optimal: bool


def solve_cpsat(instance: ProblemInstance, max_time: float = 30.0,
                workers: int = 8) -> CPSATResult:
    """Rozwiązuje instancję solverem CP-SAT i zwraca najlepszy znaleziony grafik."""
    E, D, S = instance.n_employees, instance.n_days, law.SLOTS_PER_DAY
    night = law.night_mask()
    model = cp_model.CpModel()

    pres = {}
    start = {}
    size = {}
    work = {}

    for e in range(E):
        for d in range(D):
            pres[e, d] = model.NewBoolVar(f"pres_{e}_{d}")
            start[e, d] = model.NewIntVar(0, S, f"start_{e}_{d}")
            size[e, d] = model.NewIntVar(0, law.MAX_DAILY_SLOTS, f"size_{e}_{d}")
            end = model.NewIntVar(0, S, f"end_{e}_{d}")
            model.Add(end == start[e, d] + size[e, d])

            model.Add(size[e, d] == 0).OnlyEnforceIf(pres[e, d].Not())
            model.Add(size[e, d] >= MIN_SHIFT_SLOTS).OnlyEnforceIf(pres[e, d])
            model.Add(size[e, d] >= 1).OnlyEnforceIf(pres[e, d])

            covered_sum = []
            for s in range(S):
                after = model.NewBoolVar(f"af_{e}_{d}_{s}")
                model.Add(start[e, d] <= s).OnlyEnforceIf(after)
                model.Add(start[e, d] >= s + 1).OnlyEnforceIf(after.Not())
                before = model.NewBoolVar(f"bf_{e}_{d}_{s}")
                model.Add(end >= s + 1).OnlyEnforceIf(before)
                model.Add(end <= s).OnlyEnforceIf(before.Not())
                w = model.NewBoolVar(f"w_{e}_{d}_{s}")
                model.AddBoolAnd([pres[e, d], after, before]).OnlyEnforceIf(w)
                model.AddBoolOr([pres[e, d].Not(), after.Not(), before.Not()]).OnlyEnforceIf(w.Not())
                work[e, d, s] = w
                covered_sum.append(w)
            model.Add(sum(covered_sum) == size[e, d])

    # odpoczynek dobowy >= 11h między dniem d i d+1
    for e in range(E):
        for d in range(D - 1):
            end_d = model.NewIntVar(0, S, f"endd_{e}_{d}")
            model.Add(end_d == start[e, d] + size[e, d])
            model.Add(law.SLOTS_PER_DAY + start[e, d + 1] - end_d
                      >= law.MIN_DAILY_REST_SLOTS).OnlyEnforceIf(
                          [pres[e, d], pres[e, d + 1]])

    # tygodniowy limit godzin
    for e in range(E):
        for w_idx in range(instance.n_weeks):
            days = [d for d in range(D) if d // law.DAYS_PER_WEEK == w_idx]
            model.Add(sum(size[e, d] for d in days) <= law.MAX_WEEKLY_SLOTS)

    # funkcja celu (grosze)
    terms = []
    for e in range(E):
        for d in range(D):
            sunday = instance.is_sunday_or_holiday(d)
            for s in range(S):
                coef = _BASE_GR
                if night[s]:
                    coef += _NIGHT_GR
                if sunday:
                    coef += _SUNDAY_GR
                terms.append(coef * work[e, d, s])

    for d in range(D):
        for s in range(S):
            cov = sum(work[e, d, s] for e in range(E))
            short = model.NewIntVar(0, E, f"short_{d}_{s}")
            model.Add(short >= int(instance.demand[d, s]) - cov)
            terms.append(_UNDERSTAFF_GR * short)

    model.Minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_time)
    solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    sched = Schedule.empty(E, D)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for e in range(E):
            for d in range(D):
                if solver.Value(pres[e, d]) and solver.Value(size[e, d]) > 0:
                    sched.start[e, d] = solver.Value(start[e, d])
                    sched.length[e, d] = solver.Value(size[e, d])
        cost = solver.ObjectiveValue() / 100.0
    else:
        cost = float("inf")

    return CPSATResult(
        schedule=sched,
        cost=cost,
        status=status_name,
        wall_time=solver.WallTime(),
        is_optimal=(status == cp_model.OPTIMAL),
    )
