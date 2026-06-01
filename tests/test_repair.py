"""Testy operatora naprawczego: dowolny grafik -> grafik legalny."""

import numpy as np
import pytest

from amhe.model import labor_law as law
from amhe.model.constraints import is_legal, violations
from amhe.model.schedule import Employee, ProblemInstance, Schedule
from amhe.repair import repair


def make_instance(n_employees, n_days):
    emps = [Employee(id=i, name=f"E{i}") for i in range(n_employees)]
    demand = np.zeros((n_days, law.SLOTS_PER_DAY), dtype=int)
    dow = np.array([d % 7 for d in range(n_days)])
    hol = np.zeros(n_days, dtype=bool)
    return ProblemInstance(emps, n_days, demand, dow, hol)


def random_schedule(n_employees, n_days, rng):
    """Losowy, najczesciej nielegalny grafik (zbyt dlugie zmiany, brak odpoczynkow)."""
    start = rng.integers(0, law.SLOTS_PER_DAY, size=(n_employees, n_days))
    # czeste dni wolne + czasem zbyt dlugie zmiany (do 15 h)
    length = rng.integers(0, 31, size=(n_employees, n_days))
    length[rng.random((n_employees, n_days)) < 0.3] = 0
    return Schedule(start, length)


@pytest.mark.parametrize("n_employees,n_days", [(1, 7), (5, 7), (3, 14), (8, 14)])
def test_repair_yields_legal(n_employees, n_days):
    inst = make_instance(n_employees, n_days)
    rng = np.random.default_rng(0)
    for _ in range(50):
        sch = random_schedule(n_employees, n_days, rng)
        fixed = repair(sch)
        assert is_legal(inst, fixed), violations(inst, fixed)


def test_repair_does_not_mutate_input():
    inst = make_instance(3, 7)
    rng = np.random.default_rng(1)
    sch = random_schedule(3, 7, rng)
    before_start = sch.start.copy()
    before_length = sch.length.copy()
    _ = repair(sch)
    assert np.array_equal(sch.start, before_start)
    assert np.array_equal(sch.length, before_length)


def test_repair_idempotent():
    inst = make_instance(5, 14)
    rng = np.random.default_rng(2)
    for _ in range(20):
        sch = random_schedule(5, 14, rng)
        once = repair(sch)
        twice = repair(once)
        assert np.array_equal(once.start, twice.start)
        assert np.array_equal(once.length, twice.length)
        assert is_legal(inst, once)


def test_already_legal_stays_legal():
    inst = make_instance(1, 7)
    sch = Schedule.empty(1, 7)
    for d in range(5):  # pon-pt po 8 h od 08:00
        sch.start[0, d] = 16
        sch.length[0, d] = 16
    fixed = repair(sch)
    assert is_legal(inst, fixed)
    # legalny pelnoetatowy grafik nie powinien byc okrojony
    assert int(fixed.length.sum()) == 5 * 16
