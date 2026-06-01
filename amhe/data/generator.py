"""Generator syntetycznych instancji problemu (profil typowego call center).

Popyt jest podawany wprost (bez warstwy predykcyjnej) jako wymagana liczba agentow
w kazdym 30-minutowym slocie. Profil dobowy ma dwa szczyty (poranny ok. 10:30 i
popoludniowy ok. 15:30), niski poziom nocny oraz nizsze natezenie w weekendy i swieta.
Skala popytu jest dobierana automatycznie do liczebnosci zespolu, dzieki czemu
instancje sa sensowne (wykonalne, ale nietrywialne) niezaleznie od rozmiaru.
"""

from __future__ import annotations

import numpy as np

from amhe.model import labor_law as law
from amhe.model.schedule import Employee, ProblemInstance

#: rotacja preferencji przy generowaniu pracownikow
PREFERENCES_CYCLE = ["rano", "dzien", "wieczor", "noc", "dowolna"]

#: ile polaczen obsluguje jeden agent w ciagu jednego slotu (30 min)
CALLS_PER_AGENT_PER_SLOT = 15

#: wspolczynniki natezenia ruchu wzgledem dnia roboczego
DAY_FACTOR_WEEKDAY = 1.0
DAY_FACTOR_SATURDAY = 0.6
DAY_FACTOR_SUNDAY_HOLIDAY = 0.4


def demand_profile() -> np.ndarray:
    """Znormalizowany dobowy profil natezenia ruchu (48 slotow, maksimum = 1).

    Suma niewielkiego tla nocnego oraz dwoch krzywych Gaussa (szczyt poranny i
    popoludniowy) odwzorowuje typowy rozklad polaczen w call center.
    """
    t = np.arange(law.SLOTS_PER_DAY)
    h = t / law.SLOTS_PER_HOUR  # godzina (0..24)
    baseline = 0.08
    morning = 0.55 * np.exp(-((h - 10.5) ** 2) / (2 * 1.8 ** 2))
    afternoon = 0.45 * np.exp(-((h - 15.5) ** 2) / (2 * 2.0 ** 2))
    profile = baseline + morning + afternoon
    return profile / profile.max()


def make_employees(n_employees: int, rng: np.random.Generator) -> list[Employee]:
    """Tworzy liste pracownikow z preferencjami (rotacyjnie) i drobnym zroznicowaniem kompetencji."""
    employees = []
    for i in range(n_employees):
        pref = PREFERENCES_CYCLE[i % len(PREFERENCES_CYCLE)]
        skill = float(np.round(rng.uniform(0.85, 1.15), 2))
        employees.append(Employee(id=i, name=f"P{i:02d}", preference=pref, skill=skill))
    return employees


def make_calendar(n_days: int, start_weekday: int, holidays):
    """Zwraca (day_of_week, is_holiday) dla horyzontu zaczynajacego sie w ``start_weekday``."""
    dow = np.array([(start_weekday + d) % 7 for d in range(n_days)], dtype=int)
    is_holiday = np.zeros(n_days, dtype=bool)
    for d in holidays:
        if 0 <= d < n_days:
            is_holiday[d] = True
    return dow, is_holiday


def make_demand(n_days, dow, is_holiday, peak_agents, rng, noise=0.05):
    """Macierz wymaganej obsady (n_days, 48) wg profilu dobowego i kalendarza."""
    profile = demand_profile()
    demand = np.zeros((n_days, law.SLOTS_PER_DAY), dtype=int)
    for d in range(n_days):
        if is_holiday[d] or dow[d] == 6:        # niedziela / swieto
            factor = DAY_FACTOR_SUNDAY_HOLIDAY
        elif dow[d] == 5:                        # sobota
            factor = DAY_FACTOR_SATURDAY
        else:
            factor = DAY_FACTOR_WEEKDAY
        day_curve = profile * factor * peak_agents
        if noise > 0:
            day_curve = day_curve * (1.0 + rng.normal(0.0, noise, size=law.SLOTS_PER_DAY))
        demand[d] = np.clip(np.round(day_curve), 0, None).astype(int)
    return demand


def generate_instance(
    n_employees: int,
    n_days: int,
    seed: int = 0,
    start_weekday: int = 0,
    holidays=(),
    target_peak_coverage: float = 0.6,
    name: str = "instance",
) -> ProblemInstance:
    """Generuje pelna instancje problemu.

    Args:
        n_employees:            liczba pracownikow,
        n_days:                 dlugosc horyzontu w dniach,
        seed:                   ziarno generatora (reprodukowalnosc),
        start_weekday:          dzien tygodnia pierwszego dnia (0=pon..6=niedz),
        holidays:               indeksy dni bedacych swietami,
        target_peak_coverage:   docelowa obsada w szczycie jako ulamek zespolu,
        name:                   nazwa instancji.
    """
    rng = np.random.default_rng(seed)
    employees = make_employees(n_employees, rng)
    dow, is_holiday = make_calendar(n_days, start_weekday, holidays)
    peak_agents = max(1, round(target_peak_coverage * n_employees))
    demand = make_demand(n_days, dow, is_holiday, peak_agents, rng)
    return ProblemInstance(
        employees=employees,
        n_days=n_days,
        demand=demand,
        day_of_week=dow,
        is_holiday=is_holiday,
        name=name,
    )


# --- predefiniowane scenariusze ----------------------------------------------


def scenario_small(seed: int = 1) -> ProblemInstance:
    """Maly scenariusz: 5 pracownikow, 7 dni (tydzien od poniedzialku)."""
    return generate_instance(5, 7, seed=seed, start_weekday=0, name="maly_5x7")


def scenario_medium(seed: int = 2) -> ProblemInstance:
    """Sredni scenariusz: 15 pracownikow, 14 dni, ze swietem w 10. dniu."""
    return generate_instance(15, 14, seed=seed, start_weekday=0, holidays=(9,),
                             name="sredni_15x14")


def scenario_cpsat(seed: int = 3) -> ProblemInstance:
    """Maly scenariusz odniesienia dla CP-SAT: 6 pracownikow, 3 dni."""
    return generate_instance(6, 3, seed=seed, start_weekday=0, name="cpsat_6x3")


#: rejestr scenariuszy
SCENARIOS = {
    "maly": scenario_small,
    "sredni": scenario_medium,
    "cpsat": scenario_cpsat,
}
