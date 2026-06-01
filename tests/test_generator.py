"""Testy generatora danych syntetycznych."""

import numpy as np

from amhe.data.generator import (
    demand_profile,
    generate_instance,
    make_calendar,
    scenario_cpsat,
    scenario_medium,
    scenario_small,
)
from amhe.model import labor_law as law


def test_demand_profile_shape_and_peaks():
    p = demand_profile()
    assert p.shape == (law.SLOTS_PER_DAY,)
    assert p.max() == 1.0
    # szczyt w okolicach poludnia, dolek w nocy
    assert p[law.SLOTS_PER_HOUR * 3] < 0.2     # 3:00 — noc
    assert p.argmax() / law.SLOTS_PER_HOUR == 10.5


def test_scenarios_shapes():
    for inst, ne, nd in [
        (scenario_small(), 5, 7),
        (scenario_medium(), 15, 14),
        (scenario_cpsat(), 6, 3),
    ]:
        assert inst.n_employees == ne
        assert inst.n_days == nd
        assert inst.demand.shape == (nd, law.SLOTS_PER_DAY)


def test_demand_non_negative_integers():
    inst = scenario_medium()
    assert inst.demand.dtype.kind == "i"
    assert (inst.demand >= 0).all()


def test_peak_scales_with_workforce():
    inst = generate_instance(20, 7, seed=0, target_peak_coverage=0.5)
    # szczyt obsady ~ 0.5 * 20 = 10 (z drobnym szumem)
    assert 8 <= inst.demand.max() <= 12


def test_reproducible_with_seed():
    a = generate_instance(10, 7, seed=42)
    b = generate_instance(10, 7, seed=42)
    assert np.array_equal(a.demand, b.demand)
    c = generate_instance(10, 7, seed=43)
    assert not np.array_equal(a.demand, c.demand)


def test_calendar_and_holidays():
    dow, hol = make_calendar(14, start_weekday=0, holidays=(9,))
    assert dow[0] == 0 and dow[6] == 6 and dow[7] == 0   # poniedzialek..niedziela
    assert hol[9] and hol.sum() == 1


def test_weekend_demand_lower_than_weekday():
    inst = generate_instance(20, 7, seed=1, start_weekday=0)
    weekday_peak = inst.demand[2].max()   # sroda
    sunday_peak = inst.demand[6].max()    # niedziela
    assert sunday_peak < weekday_peak
