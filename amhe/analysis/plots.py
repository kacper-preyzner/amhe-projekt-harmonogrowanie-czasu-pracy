"""Wykresy: krzywe zbieznosci, front Pareto, pokrycie popytu.

Wszystkie funkcje przyjmuja sciezke wyjsciowa i zapisuja rysunek (PNG + PDF), aby
mozna je bylo bezposrednio osadzic w raporcie LaTeX. Uzywany jest backend nieinteraktywny.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from amhe.model import labor_law as law  # noqa: E402
from amhe.model.schedule import ProblemInstance, Schedule, coverage  # noqa: E402


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_convergence(histories: dict, path, ylabel="najlepszy koszt",
                     title="Krzywa zbieznosci"):
    """Krzywe zbieznosci wielu wariantow.

    Args:
        histories: mapowanie etykieta -> lista przebiegow (kazdy to lista wartosci);
                   przebiegi sa usredniane, rysowany jest pas min-max.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, runs in histories.items():
        runs = np.asarray(runs, dtype=float)
        if runs.ndim == 1:
            runs = runs[None, :]
        gens = np.arange(runs.shape[1])
        mean = runs.mean(axis=0)
        ax.plot(gens, mean, label=label, linewidth=2)
        if runs.shape[0] > 1:
            ax.fill_between(gens, runs.min(axis=0), runs.max(axis=0), alpha=0.15)
    ax.set_xlabel("pokolenie")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_pareto(fronts: dict, path, title="Front Pareto: koszt vs preferencje"):
    """Fronty Pareto wielu wariantow w przestrzeni (koszt, kara preferencji).

    Args:
        fronts: mapowanie etykieta -> tablica (n, 2) wartosci kryteriow.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (label, pts) in enumerate(fronts.items()):
        pts = np.asarray(pts, dtype=float)
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        ax.plot(pts[:, 0], pts[:, 1], marker=markers[i % len(markers)],
                linestyle="--", label=label, alpha=0.8)
    ax.set_xlabel("koszt [zl]")
    ax.set_ylabel("kara preferencji [agento-sloty]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_coverage(instance: ProblemInstance, schedule: Schedule, path,
                  day: int = 0, title=None):
    """Porownanie popytu i obsady w wybranym dniu (slot po slocie)."""
    cov = coverage(instance, schedule)[day]
    dem = instance.demand[day]
    hours = np.arange(law.SLOTS_PER_DAY) / law.SLOTS_PER_HOUR
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(hours, dem, where="mid", label="popyt", linewidth=2, color="tab:red")
    ax.step(hours, cov, where="mid", label="obsada", linewidth=2, color="tab:blue")
    ax.fill_between(hours, dem, cov, where=(dem > cov), step="mid",
                    alpha=0.3, color="tab:red", label="niedobor")
    ax.set_xlabel("godzina")
    ax.set_ylabel("liczba agentow")
    ax.set_title(title or f"Pokrycie popytu — dzien {day}")
    ax.set_xticks(range(0, 25, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_boxplot(groups: dict, path, ylabel="koszt [zl]",
                 title="Rozklad wynikow"):
    """Boxplot rozkladu metryki dla wielu wariantow (np. po wielu ziarnach)."""
    labels = list(groups.keys())
    data = [np.asarray(groups[k], dtype=float) for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, path)
