"""Trzy scenariusze eksperymentalne z dokumentacji wstepnej.

1. ``vs_cpsat``   — memetyk vs dokladny CP-SAT na malej instancji (luka kosztu, czas),
2. ``ablation``  — memetyk (z LS) vs czysty NSGA-II (bez LS) na sredniej instancji,
3. ``disruption``— absencja pracownika i lokalna reoptymalizacja (czas, jakosc).

Kazda funkcja zwraca liste rekordow (slownikow) gotowych do zapisania w CSV i dalszej
analizy. Przebiegi sa powtarzane dla wielu ziaren (reprodukowalnosc).
"""

from __future__ import annotations

import time

import numpy as np

from amhe.baseline import solve_cpsat
from amhe.data.generator import scenario_cpsat, scenario_medium, scenario_small
from amhe.model.objectives import breakdown, preference_penalty
from amhe.optim.memetic import NSGA2Config, hypervolume_2d, run_nsga2
from amhe.optim.nsga2 import fast_non_dominated_sort
from amhe.realtime import reoptimize_absence


def _best_cost_idx(pareto_objectives):
    return int(np.argmin(pareto_objectives[:, 0]))


def run_vs_cpsat(seeds=(1, 2, 3), gens=60, pop=40, cpsat_time=30.0):
    """Scenariusz 1: porownanie z dokladnym solverem CP-SAT."""
    inst = scenario_cpsat()
    records = []

    cp = solve_cpsat(inst, max_time=cpsat_time)
    records.append({
        "scenario": "vs_cpsat", "method": "CP-SAT", "seed": -1,
        "cost": cp.cost, "pref_penalty": preference_penalty(inst, cp.schedule),
        "wall_time": cp.wall_time, "status": cp.status, "optimal": cp.is_optimal,
        "n_eval": np.nan,
    })

    for seed in seeds:
        t0 = time.perf_counter()
        res = run_nsga2(inst, NSGA2Config(pop_size=pop, n_generations=gens, seed=seed))
        wall = time.perf_counter() - t0
        idx = _best_cost_idx(res.pareto_objectives)
        best = res.pareto_schedules[idx]
        bd = breakdown(inst, best)
        records.append({
            "scenario": "vs_cpsat", "method": "memetyk", "seed": seed,
            "cost": bd["cost"], "pref_penalty": bd["preference_penalty"],
            "wall_time": wall, "status": "FEASIBLE", "optimal": False,
            "n_eval": res.n_evaluations,
            "gap_pct": (bd["cost"] - cp.cost) / cp.cost * 100.0 if cp.cost > 0 else np.nan,
        })
    return records, {"cpsat": cp, "instance": inst}


def run_ablation(seeds=(1, 2, 3, 4, 5), gens=60, pop=40):
    """Scenariusz 2: wplyw przeszukiwania lokalnego (memetyk vs czysty NSGA-II)."""
    inst = scenario_medium()
    records = []
    histories = {"memetyk": [], "NSGA-II": []}
    pareto_example = {}

    # wspolny punkt odniesienia do hiperobjetosci (z pierwszego przebiegu)
    ref = None
    for use_ls, label in [(True, "memetyk"), (False, "NSGA-II")]:
        for seed in seeds:
            res = run_nsga2(inst, NSGA2Config(
                pop_size=pop, n_generations=gens, seed=seed, use_local_search=use_ls))
            if ref is None:
                ref = res.objectives.max(axis=0) * 1.1 + 1.0
            hv = hypervolume_2d(res.pareto_objectives, ref)
            idx = _best_cost_idx(res.pareto_objectives)
            bd = breakdown(inst, res.pareto_schedules[idx])
            records.append({
                "scenario": "ablation", "method": label, "seed": seed,
                "cost": bd["cost"], "pref_penalty": bd["preference_penalty"],
                "hypervolume": hv, "pareto_size": len(res.pareto_schedules),
                "n_eval": res.n_evaluations,
            })
            histories[label].append(res.history_best_cost)
            pareto_example[label] = res.pareto_objectives
    return records, {"histories": histories, "pareto": pareto_example,
                     "instance": inst, "ref": ref}


def run_disruption(seeds=(1, 2, 3, 4, 5), gens=50, pop=40, absent=1):
    """Scenariusz 3: absencja pracownika i lokalna reoptymalizacja."""
    inst = scenario_medium()
    records = []
    example = None
    for seed in seeds:
        res = run_nsga2(inst, NSGA2Config(pop_size=pop, n_generations=gens, seed=seed))
        idx = _best_cost_idx(res.pareto_objectives)
        base = res.pareto_schedules[idx]
        reo = reoptimize_absence(inst, base, absent=absent)
        records.append({
            "scenario": "disruption", "method": "reopt", "seed": seed,
            "absent": absent,
            "shortfall_before": reo.shortfall_before,
            "shortfall_after": reo.shortfall_after,
            "recovered_slots": reo.recovered_slots,
            "cost_before": reo.cost_before, "cost_after": reo.cost_after,
            "cost_increase": reo.cost_after - reo.cost_before,
            "reopt_time": reo.wall_time,
        })
        if example is None:
            example = {"instance": inst, "base": base, "reopt": reo}
    return records, example
