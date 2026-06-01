"""Testy statystyczne do porownywania wariantow algorytmu.

Glownie nieparametryczny test Manna-Whitneya U (niezalezne proby z roznych ziaren)
oraz korekta Holma na wielokrotne porownania. Zwracane struktury sa gotowe do
zlozenia w tabele LaTeX.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    """Wynik porownania dwoch wariantow."""

    label_a: str
    label_b: str
    median_a: float
    median_b: float
    statistic: float
    p_value: float
    p_corrected: float = float("nan")
    significant: bool = False


def mann_whitney(label_a, a, label_b, b, alternative="two-sided") -> ComparisonResult:
    """Test Manna-Whitneya U dla dwoch niezaleznych prob."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    stat, p = stats.mannwhitneyu(a, b, alternative=alternative)
    return ComparisonResult(
        label_a=label_a, label_b=label_b,
        median_a=float(np.median(a)), median_b=float(np.median(b)),
        statistic=float(stat), p_value=float(p),
    )


def holm_correction(results, alpha: float = 0.05):
    """Koryguje p-wartosci metoda Holma (kontrola FWER). Modyfikuje liste w miejscu."""
    order = sorted(range(len(results)), key=lambda i: results[i].p_value)
    m = len(results)
    prev = 0.0
    for rank, idx in enumerate(order):
        corrected = min(1.0, (m - rank) * results[idx].p_value)
        corrected = max(corrected, prev)   # monotonicznosc korekty Holma
        prev = corrected
        results[idx].p_corrected = corrected
        results[idx].significant = corrected < alpha
    return results


def compare_variants(groups: dict, baseline: str | None = None,
                     alpha: float = 0.05):
    """Porownuje warianty parami testem Manna-Whitneya z korekta Holma.

    Args:
        groups:   mapowanie etykieta -> probka wynikow (np. koszty z wielu ziaren),
        baseline: jesli podany, porownuje kazdy wariant z baseline; w przeciwnym
                  razie porownuje wszystkie pary.
    """
    labels = list(groups.keys())
    results = []
    if baseline is not None:
        for lab in labels:
            if lab == baseline:
                continue
            results.append(mann_whitney(baseline, groups[baseline], lab, groups[lab]))
    else:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                results.append(
                    mann_whitney(labels[i], groups[labels[i]],
                                 labels[j], groups[labels[j]])
                )
    holm_correction(results, alpha=alpha)
    return results


def summary_statistics(sample) -> dict:
    """Podstawowe statystyki opisowe probki."""
    a = np.asarray(sample, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "iqr": float(np.subtract(*np.percentile(a, [75, 25]))),
    }
