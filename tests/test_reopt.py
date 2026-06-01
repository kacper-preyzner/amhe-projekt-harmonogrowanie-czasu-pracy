"""Testy lokalnej reoptymalizacji po absencji pracownika."""

import numpy as np

from amhe.data.generator import scenario_small
from amhe.model.constraints import is_legal
from amhe.model.objectives import coverage_shortfall
from amhe.optim.memetic import NSGA2Config, run_nsga2
from amhe.realtime import reoptimize_absence, remove_employee


def optimized_schedule(inst, seed=0):
    res = run_nsga2(inst, NSGA2Config(pop_size=24, n_generations=20, seed=seed))
    # wybierz grafik o najmniejszym koszcie z frontu
    idx = int(np.argmin(res.pareto_objectives[:, 0]))
    return res.pareto_schedules[idx]


def test_remove_employee_clears_rows():
    inst = scenario_small()
    sch = optimized_schedule(inst)
    out = remove_employee(sch, 1)
    assert out.length[1].sum() == 0
    assert out.start[1].sum() == 0
    # pozostali nietknieci
    assert np.array_equal(out.length[0], sch.length[0])


def test_reopt_legal_and_reduces_shortfall():
    inst = scenario_small()
    sch = optimized_schedule(inst)
    res = reoptimize_absence(inst, sch, absent=2)
    assert is_legal(inst, res.schedule)
    # absencja nieobecnego rzeczywiscie usunieta
    assert res.schedule.length[2].sum() == 0
    # latanie nie zwieksza niedoboru wzgledem stanu tuz po absencji
    assert res.shortfall_after <= res.shortfall_before
    assert res.recovered_slots >= 0


def test_reopt_records_timing_and_cost():
    inst = scenario_small()
    sch = optimized_schedule(inst)
    res = reoptimize_absence(inst, sch, absent=0)
    assert res.wall_time >= 0.0
    assert np.isfinite(res.cost_before) and np.isfinite(res.cost_after)


def test_reopt_recovers_some_coverage_when_possible():
    # duzy zespol, jeden nieobecny: powinno udac sie zalatac czesc luki
    from amhe.data.generator import generate_instance
    inst = generate_instance(10, 7, seed=5, target_peak_coverage=0.5)
    sch = optimized_schedule(inst, seed=3)
    short_full = coverage_shortfall(inst, sch)
    res = reoptimize_absence(inst, sch, absent=1)
    # po zalataniu niedobor nie jest gorszy niz po samym usunieciu pracownika
    assert res.shortfall_after <= res.shortfall_before
