"""Przeszukiwanie lokalne (hill-climbing na ważonej skalaryzacji kryteriów)."""

from __future__ import annotations

import numpy as np

from amhe.model import labor_law as law
from amhe.model.objectives import (
    cost_objective,
    coverage_shortfall,
    preference_penalty,
)
from amhe.model.schedule import ProblemInstance, Schedule, coverage
from amhe.repair import repair


def _scalar(instance, schedule, weight):
    return cost_objective(instance, schedule) + weight * preference_penalty(
        instance, schedule
    )


def local_search(schedule: Schedule, instance: ProblemInstance,
                 rng: np.random.Generator, max_steps: int = 20,
                 weight: float | None = None) -> Schedule:
    """Zachłanny hill-climbing. Waga losowana losowo, by eksplorować różne kierunki frontu."""
    if weight is None:
        weight = float(rng.uniform(0.0, 2.0)) * law.BASE_PER_SLOT
    best = repair(schedule)
    best_val = _scalar(instance, best, weight)

    for _ in range(max_steps):
        candidate = _propose(best, instance, rng)
        if candidate is None:
            continue
        candidate = repair(candidate)
        val = _scalar(instance, candidate, weight)
        if val < best_val - 1e-9:
            best, best_val = candidate, val
    return best


def _propose(schedule: Schedule, instance: ProblemInstance,
             rng: np.random.Generator) -> Schedule | None:
    """Proponuje jeden ruch lokalny (None jeśli nie wykonalny)."""
    cov = coverage(instance, schedule)
    deficit = instance.demand - cov
    s = schedule.copy()
    move = rng.integers(0, 3)

    if move == 0:
        # usuń zmianę w slocie z nadwyżką
        working = [(e, d) for e in range(s.n_employees) for d in range(s.n_days)
                   if s.length[e, d] > 0]
        if not working:
            return None
        e, d = working[rng.integers(0, len(working))]
        st, ln = int(s.start[e, d]), int(s.length[e, d])
        if deficit[d, st:st + ln].max() <= 0:
            s.length[e, d] = 0
            s.start[e, d] = 0
            return s
        return None

    if move == 1:
        # dodaj krótką zmianę pokrywającą największy niedobór
        if deficit.max() <= 0:
            return None
        d, slot = np.unravel_index(int(np.argmax(deficit)), deficit.shape)
        free = [e for e in range(s.n_employees) if s.length[e, d] == 0]
        if not free:
            return None
        e = free[rng.integers(0, len(free))]
        ln = law.hours_to_slots(4)
        st = int(np.clip(slot - ln // 2, 0, law.SLOTS_PER_DAY - ln))
        s.start[e, d] = st
        s.length[e, d] = ln
        return s

    # move == 2: dosuń zmianę ku preferencji pracownika
    working = [(e, d) for e in range(s.n_employees) for d in range(s.n_days)
               if s.length[e, d] > 0]
    if not working:
        return None
    e, d = working[rng.integers(0, len(working))]
    mask = instance.employees[e].pref_mask
    pref_slots = np.where(mask)[0]
    if len(pref_slots) == 0:
        return None
    ln = int(s.length[e, d])
    target = int(pref_slots[len(pref_slots) // 2]) - ln // 2
    s.start[e, d] = int(np.clip(target, 0, law.SLOTS_PER_DAY - ln))
    return s
