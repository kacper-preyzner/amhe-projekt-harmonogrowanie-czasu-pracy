"""Testy reprezentacji grafiku, masek preferencji, obsady i stałych prawnych."""

import numpy as np

from amhe.model import labor_law as law
from amhe.model.schedule import (
    Employee,
    ProblemInstance,
    Schedule,
    absolute_shifts,
    coverage,
    preference_mask,
    weekly_worked_slots,
)


def make_instance(n_employees=2, n_days=7):
    emps = [Employee(id=i, name=f"E{i}") for i in range(n_employees)]
    demand = np.zeros((n_days, law.SLOTS_PER_DAY), dtype=int)
    dow = np.array([d % 7 for d in range(n_days)])
    hol = np.zeros(n_days, dtype=bool)
    return ProblemInstance(emps, n_days, demand, dow, hol, name="test")


def test_law_constants():
    assert law.SLOTS_PER_DAY == 48
    assert law.MAX_DAILY_SLOTS == 16
    assert law.MAX_WEEKLY_SLOTS == 80
    assert law.MIN_DAILY_REST_SLOTS == 22
    assert law.MIN_WEEKLY_REST_SLOTS == 70
    assert law.BASE_PER_SLOT == law.MIN_HOURLY / 2


def test_night_mask():
    mask = law.night_mask()
    assert mask[0] and mask[13]          # 0:00 i 6:30 są nocne
    assert not mask[14]                   # 7:00 już nie
    assert mask[42] and mask[47]          # 21:00 i 23:30 nocne
    assert not mask[41]                   # 20:30 nie


def test_breaks_required():
    assert law.breaks_required(4) == 0
    assert law.breaks_required(6) == 1
    assert law.breaks_required(8) == 1
    assert law.breaks_required(10) == 2
    assert law.breaks_required(17) == 3


def test_preference_mask_morning():
    m = preference_mask("rano")  # 6:00-14:00 -> sloty 12..27
    assert not m[11] and m[12] and m[27] and not m[28]


def test_preference_mask_any_and_night_wrap():
    assert preference_mask("dowolna").all()
    noc = preference_mask("noc")  # 22:00-6:00, przez północ
    assert noc[44] and noc[0] and noc[11] and not noc[12] and not noc[43]


def test_coverage_counts_overlap():
    inst = make_instance(n_employees=2, n_days=3)
    sch = Schedule.empty(2, 3)
    sch.start[0, 0] = 16; sch.length[0, 0] = 4
    sch.start[1, 0] = 16; sch.length[1, 0] = 4
    cov = coverage(inst, sch)
    assert cov[0, 16] == 2 and cov[0, 19] == 2
    assert cov[0, 15] == 0 and cov[0, 20] == 0
    assert cov[1].sum() == 0


def test_absolute_shifts_sorted_and_offset():
    sch = Schedule.empty(1, 3)
    sch.start[0, 2] = 10; sch.length[0, 2] = 8
    sch.start[0, 0] = 16; sch.length[0, 0] = 16
    shifts = absolute_shifts(sch, 0)
    assert shifts == [(16, 32), (2 * 48 + 10, 2 * 48 + 18)]


def test_weekly_worked_slots():
    sch = Schedule.empty(1, 9)  # 2 tygodnie (7 + 2 dni)
    sch.length[0, 0] = 16
    sch.length[0, 7] = 8
    wws = weekly_worked_slots(sch)
    assert wws.shape == (1, 2)
    assert wws[0, 0] == 16 and wws[0, 1] == 8
