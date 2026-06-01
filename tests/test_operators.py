"""Testy operatorow genetycznych i przeszukiwania lokalnego (zawsze legalny wynik)."""

import numpy as np

from amhe.data.generator import scenario_small
from amhe.model.constraints import is_legal
from amhe.model.objectives import cost_objective, preference_penalty
from amhe.optim import operators as ops
from amhe.optim.local_search import local_search


def test_initial_population_legal_and_sized():
    inst = scenario_small()
    rng = np.random.default_rng(0)
    pop = ops.initial_population(inst, 12, rng)
    assert len(pop) == 12
    for s in pop:
        assert s.start.shape == (inst.n_employees, inst.n_days)
        assert is_legal(inst, s)


def test_crossover_legal_and_new_objects():
    inst = scenario_small()
    rng = np.random.default_rng(1)
    a, b = ops.initial_population(inst, 2, rng)
    c1, c2 = ops.crossover(a, b, rng, inst)
    assert is_legal(inst, c1) and is_legal(inst, c2)
    assert c1 is not a and c2 is not b


def test_mutate_legal_and_changes_something():
    inst = scenario_small()
    rng = np.random.default_rng(2)
    s = ops.random_schedule(inst, rng)
    changed = False
    for _ in range(10):
        m = ops.mutate(s, rng, inst, rate=0.5)
        assert is_legal(inst, m)
        if not (np.array_equal(m.start, s.start) and np.array_equal(m.length, s.length)):
            changed = True
    assert changed


def test_local_search_legal_and_not_worse():
    inst = scenario_small()
    rng = np.random.default_rng(3)
    s = ops.random_schedule(inst, rng, work_prob=0.9)
    weight = 0.0   # czysta minimalizacja kosztu
    before = cost_objective(inst, s)
    improved = local_search(s, inst, rng, max_steps=30, weight=weight)
    assert is_legal(inst, improved)
    # przy stalej wadze hill-climbing nie moze pogorszyc skalaryzacji (tu: kosztu)
    assert cost_objective(inst, improved) <= before + 1e-9
