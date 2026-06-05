"""Operatory genetyczne: inicjalizacja, krzyżowanie, mutacja."""

from __future__ import annotations

import numpy as np

from amhe.model import labor_law as law
from amhe.model.schedule import ProblemInstance, Schedule
from amhe.repair import repair

MIN_INIT_SHIFT = law.hours_to_slots(4)
MAX_INIT_SHIFT = law.MAX_DAILY_SLOTS


def _demand_window(instance: ProblemInstance):
    """Zakres slotów [lo, hi), w których wystepuje popyt."""
    active = np.where(instance.demand.sum(axis=0) > 0)[0]
    if len(active) == 0:
        return 0, law.SLOTS_PER_DAY
    return int(active[0]), int(active[-1]) + 1


def random_schedule(instance: ProblemInstance, rng: np.random.Generator,
                    work_prob: float = 0.6) -> Schedule:
    """Losowy, naprawiony grafik."""
    E, D = instance.n_employees, instance.n_days
    lo, hi = _demand_window(instance)
    start = np.zeros((E, D), dtype=int)
    length = np.zeros((E, D), dtype=int)
    for e in range(E):
        for d in range(D):
            if rng.random() < work_prob:
                ln = int(rng.integers(MIN_INIT_SHIFT, MAX_INIT_SHIFT + 1))
                latest = max(lo, hi - ln)
                st = int(rng.integers(lo, latest + 1)) if latest > lo else lo
                st = min(st, law.SLOTS_PER_DAY - ln)
                start[e, d] = st
                length[e, d] = ln
    return repair(Schedule(start, length))


def initial_population(instance: ProblemInstance, size: int,
                       rng: np.random.Generator) -> list[Schedule]:
    """Populacja początkowa o zróżnicowanej gęstości pracy."""
    pop = []
    for i in range(size):
        wp = 0.4 + 0.5 * (i / max(size - 1, 1))
        pop.append(random_schedule(instance, rng, work_prob=wp))
    return pop


def crossover(parent_a: Schedule, parent_b: Schedule, rng: np.random.Generator,
              instance: ProblemInstance) -> tuple[Schedule, Schedule]:
    """Krzyżowanie jednorodne na poziomie pracownika (wymiana wierszy grafiku)."""
    E = parent_a.n_employees
    mask = rng.random(E) < 0.5
    a = parent_a.copy()
    b = parent_b.copy()
    a.start[mask] = parent_b.start[mask]
    a.length[mask] = parent_b.length[mask]
    b.start[mask] = parent_a.start[mask]
    b.length[mask] = parent_a.length[mask]
    return repair(a), repair(b)


def mutate(schedule: Schedule, rng: np.random.Generator, instance: ProblemInstance,
           rate: float = 0.1) -> Schedule:
    """Mutacja: przełącz wolne<->praca, przesuń start, zmień długość."""
    s = schedule.copy()
    E, D = s.n_employees, s.n_days
    lo, hi = _demand_window(instance)
    for e in range(E):
        for d in range(D):
            if rng.random() >= rate:
                continue
            move = rng.integers(0, 3)
            if move == 0:
                if s.length[e, d] > 0:
                    s.length[e, d] = 0
                    s.start[e, d] = 0
                else:
                    ln = int(rng.integers(MIN_INIT_SHIFT, MAX_INIT_SHIFT + 1))
                    s.length[e, d] = ln
                    s.start[e, d] = min(max(lo, 0), law.SLOTS_PER_DAY - ln)
            elif move == 1 and s.length[e, d] > 0:
                ln = int(s.length[e, d])
                delta = int(rng.integers(-4, 5))
                s.start[e, d] = int(np.clip(s.start[e, d] + delta, 0,
                                            law.SLOTS_PER_DAY - ln))
            elif move == 2 and s.length[e, d] > 0:
                delta = int(rng.integers(-4, 5))
                ln = int(np.clip(s.length[e, d] + delta, 0, law.MAX_DAILY_SLOTS))
                s.length[e, d] = ln
                if ln == 0:
                    s.start[e, d] = 0
                else:
                    s.start[e, d] = min(s.start[e, d], law.SLOTS_PER_DAY - ln)
    return repair(s)
