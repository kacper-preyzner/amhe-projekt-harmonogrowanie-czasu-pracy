"""Projekt AMHE nr 7 — harmonogramowanie pracy w call center.

Pakiet realizuje memetyczny algorytm ewolucyjny (własny NSGA-II + przeszukiwanie
lokalne) z operatorem naprawczym wymuszającym twarde ograniczenia Kodeksu pracy,
układający tygodniowy grafik pracy dla call center. Wynikiem jest front Pareto
kompromisów koszt × dopasowanie preferencji. Jako punkt odniesienia służy solver
CP-SAT (OR-Tools).

Podpakiety:
    * ``data``        — syntetyczny generator danych (pracownicy, popyt, kalendarz),
    * ``model``       — reprezentacja grafiku, ograniczenia (Kodeks pracy), kryteria,
    * ``repair``      — operator naprawczy egzekwujący legalność,
    * ``optim``       — NSGA-II, operatory genetyczne, przeszukiwanie lokalne, memetyk,
    * ``baseline``    — solver odniesienia CP-SAT,
    * ``realtime``    — lokalna reoptymalizacja po absencji,
    * ``experiments`` — uruchamianie eksperymentów,
    * ``analysis``    — wykresy, statystyki, tabele.
"""

__version__ = "0.1.0"
