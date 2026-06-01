"""Testy baseline CP-SAT: legalnosc, optymalnosc na malej instancji, spojnosc kosztu."""

import numpy as np
import pytest

from amhe.baseline import solve_cpsat
from amhe.data.generator import generate_instance, scenario_cpsat
from amhe.model.constraints import is_legal
from amhe.model.objectives import cost_objective


def test_cpsat_returns_legal_schedule():
    inst = scenario_cpsat()
    res = solve_cpsat(inst, max_time=20.0)
    assert res.status in ("OPTIMAL", "FEASIBLE")
    assert is_legal(inst, res.schedule)


def test_cpsat_cost_matches_objective():
    inst = scenario_cpsat()
    res = solve_cpsat(inst, max_time=20.0)
    # koszt zwrocony przez solver zgodny z nasza funkcja celu (z dokladnoscia do groszy)
    recomputed = cost_objective(inst, res.schedule)
    assert res.cost == pytest.approx(recomputed, abs=0.5)


def test_cpsat_trivial_zero_demand_is_empty():
    # brak popytu => optymalny koszt to 0 (nikt nie pracuje)
    inst = generate_instance(3, 2, seed=0, target_peak_coverage=0.0)
    inst.demand[:] = 0
    res = solve_cpsat(inst, max_time=10.0)
    assert res.is_optimal
    assert res.cost == 0.0
    assert int(res.schedule.length.sum()) == 0


def test_cpsat_covers_demand_block():
    # blok popytu na 8 slotow (4 h) => pokrycie jedna zmiana sie oplaca:
    # kara za niedobor 8*understaff > koszt 8-slotowej zmiany
    inst = generate_instance(2, 1, seed=0, target_peak_coverage=0.0)
    inst.demand[:] = 0
    inst.demand[0, 16:24] = 1   # 08:00-12:00
    res = solve_cpsat(inst, max_time=10.0)
    assert res.is_optimal
    from amhe.model.schedule import coverage
    cov = coverage(inst, res.schedule)
    assert cov[0, 16:24].min() >= 1   # caly blok pokryty
