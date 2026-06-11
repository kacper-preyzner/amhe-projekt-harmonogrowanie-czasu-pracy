"""Własna implementacja NSGA-II: dominacja Pareto, sortowanie, crowding, turniej."""

from __future__ import annotations

import numpy as np


def dominates(a, b) -> bool:
    """Czy a dominuje b (wszystkie kryteria <=, co najmniej jedno <)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F) -> list:
    """Dzieli rozwiązania na fronty Pareto. Zwraca listę frontów (listy indeksów)."""
    F = np.asarray(F, dtype=float)
    n = len(F)
    dominated = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                dominated[p].append(q)
            elif dominates(F[q], F[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(F, front) -> np.ndarray:
    """Miara zagęszczenia dla jednego frontu."""
    F = np.asarray(F, dtype=float)
    length = len(front)
    dist = np.zeros(length, dtype=float)
    if length <= 2:
        dist[:] = np.inf
        return dist
    pts = F[front]
    for m in range(pts.shape[1]):
        order = np.argsort(pts[:, m])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = pts[order[-1], m] - pts[order[0], m]
        if span == 0:
            continue
        for k in range(1, length - 1):
            dist[order[k]] += (pts[order[k + 1], m] - pts[order[k - 1], m]) / span
    return dist


def rank_and_crowding(F):
    """Zwraca (rank, crowd, fronts) dla całej populacji."""
    n = len(F)
    fronts = fast_non_dominated_sort(F)
    rank = np.zeros(n, dtype=int)
    crowd = np.zeros(n, dtype=float)
    for r, front in enumerate(fronts):
        cd = crowding_distance(F, front)
        for idx, sol in enumerate(front):
            rank[sol] = r
            crowd[sol] = cd[idx]
    return rank, crowd, fronts


def environmental_selection(F, n_select: int) -> np.ndarray:
    """Wybiera n_select rozwiązań wg (ranga rosnąco, zagęszczenie malejąco)."""
    fronts = fast_non_dominated_sort(F)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
        else:
            cd = crowding_distance(F, front)
            order = sorted(range(len(front)), key=lambda i: -cd[i])
            need = n_select - len(selected)
            selected.extend(front[i] for i in order[:need])
            break
    return np.array(selected, dtype=int)


def binary_tournament(rank, crowd, rng) -> int:
    """Turniej binarny: lepsza ranga, przy remisie — większe zagęszczenie."""
    i, j = rng.integers(0, len(rank), size=2)
    if rank[i] < rank[j]:
        return int(i)
    if rank[j] < rank[i]:
        return int(j)
    return int(i) if crowd[i] >= crowd[j] else int(j)
