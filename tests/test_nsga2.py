"""Testy jadra NSGA-II: dominacja, sortowanie, crowding, selekcja."""

import numpy as np

from amhe.optim.nsga2 import (
    crowding_distance,
    dominates,
    environmental_selection,
    fast_non_dominated_sort,
    rank_and_crowding,
)


def test_dominates_basic():
    assert dominates([1, 1], [2, 2])
    assert dominates([1, 2], [1, 3])
    assert not dominates([1, 2], [2, 1])   # nieporownywalne
    assert not dominates([1, 1], [1, 1])   # rownosc nie jest dominacja


def test_fronts_partition_all_points():
    F = np.array([[1, 5], [2, 3], [4, 1], [5, 5], [3, 4]], dtype=float)
    fronts = fast_non_dominated_sort(F)
    flat = sorted(i for fr in fronts for i in fr)
    assert flat == list(range(len(F)))
    # pierwszy front to rozwiazania niezdominowane
    assert set(fronts[0]) == {0, 1, 2}


def test_front_zero_is_nondominated():
    F = np.array([[0, 2], [2, 0], [1, 1], [3, 3]], dtype=float)
    fronts = fast_non_dominated_sort(F)
    for i in fronts[0]:
        for j in range(len(F)):
            assert not dominates(F[j], F[i])


def test_crowding_endpoints_infinite():
    F = np.array([[0, 4], [1, 2], [2, 1], [4, 0]], dtype=float)
    front = [0, 1, 2, 3]
    cd = crowding_distance(F, front)
    assert np.isinf(cd[0]) and np.isinf(cd[3])
    assert np.isfinite(cd[1]) and np.isfinite(cd[2])


def test_environmental_selection_keeps_best_front():
    F = np.array([[0, 2], [2, 0], [1, 1], [3, 3], [4, 4]], dtype=float)
    sel = environmental_selection(F, 3)
    assert len(sel) == 3
    # najgorsze (zdominowane) rozwiazania nie powinny zostac wybrane
    assert 4 not in sel


def test_rank_and_crowding_consistent():
    F = np.array([[0, 2], [2, 0], [1, 1], [3, 3]], dtype=float)
    rank, crowd, fronts = rank_and_crowding(F)
    assert rank[3] == max(rank)            # [3,3] w najgorszym froncie
    assert rank[0] == 0 and rank[1] == 0   # skrajne niezdominowane
