# AMHE — Projekt 7: Harmonogramowanie pracy w call center

Memetyczny algorytm ewolucyjny (własny **NSGA-II** + przeszukiwanie lokalne) z
**operatorem naprawczym** wymuszającym twarde ograniczenia Kodeksu pracy, układający
tygodniowy grafik pracy dla call center. Optymalizacja jest **wielokryterialna**
(koszt × dopasowanie preferencji), więc wynikiem jest **front Pareto** kompromisów.
Jako punkt odniesienia służy dokładny solver **CP-SAT** (OR-Tools).

## Wymagania i instalacja

Projekt korzysta z menedżera [`uv`](https://docs.astral.sh/uv/) i Pythona 3.12.

```bash
make setup       # = uv sync : tworzy .venv i instaluje zależności
make test        # = uv run pytest : 67 testów jednostkowych
```

## Uruchamianie

Najprościej — skrypty `.sh` (działają z dowolnego katalogu):

```bash
./setup.sh                    # środowisko (uv sync)
./test.sh                     # testy jednostkowe
./run.sh                      # pełne 3 scenariusze
./run.sh --smoke              # szybki przebieg dymny (~10 s)
./run.sh --scenario ablation  # pojedynczy scenariusz
./report.sh                   # kompilacja raportu -> report/raport.pdf
./all.sh                      # cały pipeline: setup -> test -> run -> report
./all.sh --smoke              # jak wyżej, ale szybki przebieg dymny
```

To samo przez `make` (`setup` / `test` / `smoke` / `experiments` / `report`) lub
bezpośrednio:

```bash
uv run python -m amhe.experiments.run --all
uv run python -m amhe.experiments.run --scenario ablation
```

## Co jest liczone

| Scenariusz | Co bada | Pliki wynikowe |
|---|---|---|
| `vs_cpsat`   | memetyk vs dokładny CP-SAT (luka kosztu, czas) | `results/vs_cpsat.csv`, `figures/gantt_cpsat.*`, `coverage_cpsat.*` |
| `ablation`   | wpływ przeszukiwania lokalnego (memetyk vs NSGA-II) | `results/ablation.csv`, `figures/convergence_ablation.*`, `pareto_ablation.*`, `boxplot_ablation.*` |
| `disruption` | absencja pracownika + lokalna reoptymalizacja | `results/disruption.csv`, `figures/gantt_before_absence.*`, `gantt_after_reopt.*` |

Tabele LaTeX trafiają do `report/tables/`, figury (PNG+PDF, oraz interaktywny
Gantt HTML) do `report/figures/`.

## Struktura

```
amhe/
  data/          generator danych syntetycznych (pracownicy, popyt, kalendarz)
  model/         labor_law (stałe 2026), schedule (reprezentacja),
                 constraints (legalność), objectives (koszt + preferencje)
  repair/        operator naprawczy (gwarancja legalności)
  optim/         nsga2 (sort+crowding), operators, local_search, memetic
  baseline/      cpsat (solver odniesienia OR-Tools)
  realtime/      reopt (lokalna reoptymalizacja po absencji)
  experiments/   scenarios + run (CLI)
  analysis/      plots, gantt (mpl + plotly), stats, tables
tests/           67 testów jednostkowych
report/          raport.tex (PL), references.bib, figures/, tables/
results/  figures/   (generowane)
```

## Model — najważniejsze założenia

- Czas dyskretny: sloty 30-minutowe (48/dobę). Pojedyncza zmiana mieści się w dobie.
- Twarde ograniczenia (Kodeks pracy): ≤8 h/dobę, ≤40 h/tydzień, odpoczynek dobowy
  ≥11 h, tygodniowy ≥35 h, praca nocna ≤8 h.
- Stawki 2026: płaca min. 4806 zł/mies., 31,40 zł/h; dodatki nocny (+20%),
  niedzielny/świąteczny (+100%).
- Kryteria (min.): **koszt** = płace + kara za niedobór obsady; **kara preferencji**
  = sloty pracy poza preferowanym oknem pory dnia pracownika.

Pełny opis metodyki, hipotezy i analiza wyników — w `report/raport.tex`.
