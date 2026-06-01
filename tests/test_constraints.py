"""Testy twardych ograniczeń (legalność grafiku)."""

import numpy as np

from amhe.model import labor_law as law
from amhe.model.constraints import (
    is_legal,
    max_free_gap,
    satisfaction_rate,
    violations,
)
from amhe.model.schedule import Employee, ProblemInstance, Schedule


def make_instance(n_employees=1, n_days=7):
    emps = [Employee(id=i, name=f"E{i}") for i in range(n_employees)]
    demand = np.zeros((n_days, law.SLOTS_PER_DAY), dtype=int)
    dow = np.array([d % 7 for d in range(n_days)])
    hol = np.zeros(n_days, dtype=bool)
    return ProblemInstance(emps, n_days, demand, dow, hol)


def legal_full_time_schedule():
    """1 pracownik: pon–pt 8 h od 08:00, weekend wolny — w pełni legalny."""
    sch = Schedule.empty(1, 7)
    for d in range(5):
        sch.start[0, d] = 16   # 08:00
        sch.length[0, d] = 16  # 8 h
    return sch


def test_legal_schedule_passes():
    inst = make_instance()
    sch = legal_full_time_schedule()
    assert is_legal(inst, sch)
    assert violations(inst, sch) == {
        "shift_bounds": 0, "daily_hours": 0, "weekly_hours": 0,
        "daily_rest": 0, "weekly_rest": 0,
    }
    assert satisfaction_rate(inst, sch) == 1.0


def test_daily_hours_violation():
    inst = make_instance()
    sch = Schedule.empty(1, 7)
    sch.start[0, 0] = 0
    sch.length[0, 0] = 20  # 10 h > 8 h
    assert violations(inst, sch)["daily_hours"] == 1
    assert not is_legal(inst, sch)


def test_shift_bounds_violation():
    inst = make_instance()
    sch = Schedule.empty(1, 7)
    sch.start[0, 0] = 40
    sch.length[0, 0] = 16  # 40+16=56 > 48
    assert violations(inst, sch)["shift_bounds"] == 1


def test_daily_rest_violation_isolated():
    inst = make_instance()
    sch = Schedule.empty(1, 7)
    sch.start[0, 0] = 16; sch.length[0, 0] = 16  # kończy o 24:00 (slot 32)
    sch.start[0, 1] = 0;  sch.length[0, 1] = 16  # zaczyna o 00:00 nast. dnia
    v = violations(inst, sch)
    assert v["daily_rest"] == 1
    assert v["weekly_hours"] == 0 and v["weekly_rest"] == 0


def test_weekly_rest_violation_isolated():
    inst = make_instance()
    sch = Schedule.empty(1, 7)
    for d in range(7):  # praca codziennie po 4 h od 10:00
        sch.start[0, d] = 20
        sch.length[0, d] = 8
    v = violations(inst, sch)
    assert v["weekly_rest"] == 1
    assert v["weekly_hours"] == 0   # 7*8=56 <= 80
    assert v["daily_hours"] == 0
    assert v["daily_rest"] == 0


def test_weekly_hours_violation():
    inst = make_instance()
    sch = Schedule.empty(1, 7)
    for d in range(6):  # 6 dni po 8 h = 48 h > 40 h
        sch.start[0, d] = 16
        sch.length[0, d] = 16
    assert violations(inst, sch)["weekly_hours"] == 1


def test_max_free_gap():
    # zmiany w slotach [10,20] i [40,50] w oknie [0,100)
    shifts = [(10, 20), (40, 50)]
    assert max_free_gap(shifts, 0, 100) == 50   # po ostatniej: 100-50
    assert max_free_gap([], 0, 70) == 70        # brak zmian = całe okno wolne


def test_satisfaction_rate_below_one_when_illegal():
    inst = make_instance()
    sch = Schedule.empty(1, 7)
    sch.start[0, 0] = 0
    sch.length[0, 0] = 30  # rażąco za długa
    assert satisfaction_rate(inst, sch) < 1.0
