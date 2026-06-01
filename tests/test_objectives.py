"""Testy kryteriow: koszt (place + dodatki + niedobor) oraz kara preferencji."""

import numpy as np

from amhe.model import labor_law as law
from amhe.model.objectives import (
    UNDERSTAFF_PENALTY_PER_SLOT,
    cost_objective,
    coverage_shortfall,
    evaluate,
    preference_penalty,
    wage_cost,
)
from amhe.model.schedule import Employee, ProblemInstance, Schedule


def make_instance(prefs=("dowolna",), n_days=7, demand=None, dow=None, holidays=()):
    emps = [Employee(id=i, name=f"E{i}", preference=p) for i, p in enumerate(prefs)]
    if demand is None:
        demand = np.zeros((n_days, law.SLOTS_PER_DAY), dtype=int)
    if dow is None:
        dow = np.array([d % 7 for d in range(n_days)])
    hol = np.zeros(n_days, dtype=bool)
    for d in holidays:
        hol[d] = True
    return ProblemInstance(emps, n_days, demand, dow, hol)


def test_empty_schedule_cost_is_pure_understaffing():
    demand = np.zeros((7, law.SLOTS_PER_DAY), dtype=int)
    demand[0, 16] = 2
    demand[1, 20] = 1
    inst = make_instance(prefs=("dowolna",), demand=demand)
    sch = Schedule.empty(1, 7)
    assert wage_cost(inst, sch) == 0.0
    assert coverage_shortfall(inst, sch) == 3
    assert cost_objective(inst, sch) == 3 * UNDERSTAFF_PENALTY_PER_SLOT
    assert preference_penalty(inst, sch) == 0.0


def test_coverage_reduces_shortfall():
    demand = np.zeros((7, law.SLOTS_PER_DAY), dtype=int)
    demand[0, 16] = 1
    inst = make_instance(prefs=("dowolna",), demand=demand)
    sch = Schedule.empty(1, 7)
    sch.start[0, 0] = 16
    sch.length[0, 0] = 1
    assert coverage_shortfall(inst, sch) == 0


def test_night_bonus():
    inst = make_instance(prefs=("dowolna",))
    sch = Schedule.empty(1, 7)
    sch.start[0, 0] = 0   # 00:00–01:00, dwa sloty nocne
    sch.length[0, 0] = 2
    expected = 2 * law.BASE_PER_SLOT + 2 * law.NIGHT_BONUS_PER_SLOT
    assert wage_cost(inst, sch) == expected


def test_sunday_bonus():
    dow = np.array([6, 0, 1, 2, 3, 4, 5])  # dzien 0 = niedziela
    inst = make_instance(prefs=("dowolna",), dow=dow)
    sch = Schedule.empty(1, 7)
    sch.start[0, 0] = 16  # 08:00–09:00, nie noc
    sch.length[0, 0] = 2
    expected = 2 * law.BASE_PER_SLOT + 2 * law.SUNDAY_HOLIDAY_BONUS_PER_SLOT
    assert wage_cost(inst, sch) == expected


def test_preference_penalty_morning():
    inst = make_instance(prefs=("rano",))  # okno 6:00–14:00 -> sloty 12..27
    sch = Schedule.empty(1, 7)
    sch.start[0, 0] = 16   # 08:00–12:00 w oknie
    sch.length[0, 0] = 8
    assert preference_penalty(inst, sch) == 0.0
    sch2 = Schedule.empty(1, 7)
    sch2.start[0, 0] = 0   # 00:00–02:00 poza oknem
    sch2.length[0, 0] = 4
    assert preference_penalty(inst, sch2) == 4.0


def test_evaluate_returns_two_objectives():
    inst = make_instance(prefs=("dowolna",))
    sch = Schedule.empty(1, 7)
    out = evaluate(inst, sch)
    assert out.shape == (2,)
    assert out.dtype == float
