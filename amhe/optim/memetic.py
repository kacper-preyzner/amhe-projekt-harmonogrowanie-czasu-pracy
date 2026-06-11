"""Memetyczny NSGA-II — główny algorytm projektu (NSGA-II + przeszukiwanie lokalne)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amhe.model.objectives import evaluate
from amhe.model.schedule import ProblemInstance, Schedule
from amhe.optim import operators as ops
from amhe.optim.local_search import local_search
from amhe.optim.nsga2 import (
    binary_tournament,
    environmental_selection,
    fast_non_dominated_sort,
    rank_and_crowding,
)


@dataclass
class NSGA2Config:
    pop_size: int = 40
    n_generations: int = 50
    crossover_prob: float = 0.9
    mutation_rate: float = 0.1
    use_local_search: bool = True
    local_search_steps: int = 15
    local_search_frac: float = 0.5
    seed: int = 0


@dataclass
class NSGA2Result:
    population: list[Schedule]
    objectives: np.ndarray
    pareto_schedules: list[Schedule]
    pareto_objectives: np.ndarray
    history_best_cost: list[float] = field(default_factory=list)
    history_hypervolume: list[float] = field(default_factory=list)
    n_evaluations: int = 0


def _evaluate_population(instance, population):
    return np.array([evaluate(instance, s) for s in population], dtype=float)


def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Hiperbjętość 2D (minimalizacja) frontu względem punktu odniesienia."""
    pts = np.asarray(points, dtype=float)
    pts = pts[(pts[:, 0] <= ref[0]) & (pts[:, 1] <= ref[1])]
    if len(pts) == 0:
        return 0.0
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    hv = 0.0
    prev_x = pts[0, 0]
    prev_y = ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


def run_nsga2(instance: ProblemInstance, config: NSGA2Config) -> NSGA2Result:
    """Uruchamia memetyczny (lub czysty) NSGA-II dla zadanej instancji."""
    rng = np.random.default_rng(config.seed)
    n_eval = 0

    population = ops.initial_population(instance, config.pop_size, rng)
    F = _evaluate_population(instance, population)
    n_eval += len(population)

    ref = F.max(axis=0) * 1.1 + 1.0

    history_cost: list[float] = []
    history_hv: list[float] = []

    def record():
        history_cost.append(float(F[:, 0].min()))
        fronts = fast_non_dominated_sort(F)
        history_hv.append(hypervolume_2d(F[fronts[0]], ref))

    record()

    for _ in range(config.n_generations):
        rank, crowd, _ = rank_and_crowding(F)

        offspring: list[Schedule] = []
        while len(offspring) < config.pop_size:
            ia = binary_tournament(rank, crowd, rng)
            ib = binary_tournament(rank, crowd, rng)
            if rng.random() < config.crossover_prob:
                c1, c2 = ops.crossover(population[ia], population[ib], rng, instance)
            else:
                c1, c2 = population[ia].copy(), population[ib].copy()
            c1 = ops.mutate(c1, rng, instance, config.mutation_rate)
            c2 = ops.mutate(c2, rng, instance, config.mutation_rate)
            offspring.extend([c1, c2])
        offspring = offspring[:config.pop_size]

        if config.use_local_search:
            for i in range(len(offspring)):
                if rng.random() < config.local_search_frac:
                    offspring[i] = local_search(
                        offspring[i], instance, rng,
                        max_steps=config.local_search_steps,
                    )

        F_off = _evaluate_population(instance, offspring)
        n_eval += len(offspring)

        combined = population + offspring
        F_comb = np.vstack([F, F_off])
        keep = environmental_selection(F_comb, config.pop_size)
        population = [combined[i] for i in keep]
        F = F_comb[keep]

        record()

    fronts = fast_non_dominated_sort(F)
    pareto_idx = fronts[0]
    return NSGA2Result(
        population=population,
        objectives=F,
        pareto_schedules=[population[i] for i in pareto_idx],
        pareto_objectives=F[pareto_idx],
        history_best_cost=history_cost,
        history_hypervolume=history_hv,
        n_evaluations=n_eval,
    )
