"""Testy memetycznego NSGA-II: poprawnosc frontu i zbieznosc."""

import numpy as np

from amhe.data.generator import scenario_small
from amhe.model.constraints import is_legal
from amhe.optim.memetic import NSGA2Config, hypervolume_2d, run_nsga2
from amhe.optim.nsga2 import dominates


def small_config(**kw):
    base = dict(pop_size=16, n_generations=8, seed=0)
    base.update(kw)
    return NSGA2Config(**base)


def test_run_returns_legal_pareto():
    inst = scenario_small()
    res = run_nsga2(inst, small_config())
    assert len(res.pareto_schedules) >= 1
    for s in res.pareto_schedules:
        assert is_legal(inst, s)
    assert res.pareto_objectives.shape[1] == 2


def test_pareto_front_is_nondominated():
    inst = scenario_small()
    res = run_nsga2(inst, small_config())
    P = res.pareto_objectives
    for i in range(len(P)):
        for j in range(len(P)):
            if i != j:
                assert not dominates(P[j], P[i])


def test_convergence_cost_non_increasing():
    inst = scenario_small()
    res = run_nsga2(inst, small_config(n_generations=12))
    h = res.history_best_cost
    # najlepszy koszt jest monotonicznie nierosnacy (elityzm mu+lambda)
    assert all(h[i + 1] <= h[i] + 1e-6 for i in range(len(h) - 1))


def test_reproducible_with_seed():
    inst = scenario_small()
    r1 = run_nsga2(inst, small_config(seed=7))
    r2 = run_nsga2(inst, small_config(seed=7))
    assert np.allclose(np.sort(r1.objectives[:, 0]), np.sort(r2.objectives[:, 0]))


def test_hypervolume_basic():
    # front {(1,3),(2,2),(3,1)} pod ref (4,4)
    pts = np.array([[1, 3], [2, 2], [3, 1]], dtype=float)
    hv = hypervolume_2d(pts, np.array([4.0, 4.0]))
    assert hv > 0
    # pojedynczy punkt (0,0) pod ref (4,4) => pole 16
    assert hypervolume_2d(np.array([[0.0, 0.0]]), np.array([4.0, 4.0])) == 16.0


def test_local_search_flag_runs():
    inst = scenario_small()
    res_ls = run_nsga2(inst, small_config(use_local_search=True))
    res_no = run_nsga2(inst, small_config(use_local_search=False))
    assert len(res_ls.pareto_schedules) >= 1
    assert len(res_no.pareto_schedules) >= 1
