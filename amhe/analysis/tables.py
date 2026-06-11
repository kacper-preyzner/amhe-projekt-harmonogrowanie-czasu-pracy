"""Generacja tabel LaTeX z wynikow eksperymentow (do bezposredniego \\input w raporcie)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from amhe.analysis.stats import summary_statistics


def _fmt(x, prec=1):
    if isinstance(x, str):
        return x
    if isinstance(x, float) and not np.isfinite(x):
        return "--"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{prec}f}"


def write_table(path, header, rows, caption, label, col_format=None):
    """Zapisuje tabele LaTeX (srodowisko table + tabular)."""
    ncol = len(header)
    col_format = col_format or ("l" + "r" * (ncol - 1))
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_format}}}",
        r"    \hline",
        "    " + " & ".join(header) + r" \\",
        r"    \hline",
    ]
    for row in rows:
        lines.append("    " + " & ".join(_fmt(c) for c in row) + r" \\")
    lines += [r"    \hline", r"  \end{tabular}", r"\end{table}", ""]
    text = "\n".join(lines)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def summary_table(path, groups: dict, caption, label, metric_name="koszt [zl]"):
    """Tabela statystyk opisowych (srednia, odch., mediana, min, max) per wariant."""
    header = ["wariant", "n", "srednia", "odch. std", "mediana", "min", "max"]
    rows = []
    for name, sample in groups.items():
        s = summary_statistics(sample)
        rows.append([name, s["n"], s["mean"], s["std"], s["median"],
                     s["min"], s["max"]])
    cap = f"{caption} ({metric_name})"
    return write_table(path, header, rows, cap, label)


def vs_cpsat_table(path, df_inst, caption, label):
    """Tabela memetyk vs CP-SAT dla jednej instancji: koszt, luka[%], czas[s], status."""
    header = ["Metoda", "Koszt (sr.)", "Luka kosztu [\\%]", "Czas [s]", "Status"]
    rows = []
    cp_row = df_inst[df_inst.method == "CP-SAT"].iloc[0]
    mem_rows = df_inst[df_inst.method == "memetyk"]
    rows.append([
        "CP-SAT",
        _fmt(cp_row["cost"]),
        "0.0",
        _fmt(cp_row["wall_time"], 2),
        str(cp_row["status"]),
    ])
    rows.append([
        "memetyk",
        _fmt(mem_rows["cost"].mean()),
        _fmt(mem_rows["gap_pct"].mean()),
        _fmt(mem_rows["wall_time"].mean(), 2),
        "FEASIBLE",
    ])
    return write_table(path, header, rows, caption, label, col_format="lrrrr")  # noqa: E501 — 5 cols but first is text


def comparison_table(path, results, caption, label):
    """Tabela porownan parami (mediany, p-wartosc, p po korekcie Holma, istotnosc)."""
    header = ["A", "B", "med. A", "med. B", "p", "p (Holm)", "istotne"]
    rows = []
    for r in results:
        rows.append([
            r.label_a, r.label_b, r.median_a, r.median_b,
            f"{r.p_value:.4f}", f"{r.p_corrected:.4f}",
            "tak" if r.significant else "nie",
        ])
    return write_table(path, header, rows, caption, label,
                       col_format="ll" + "r" * 4 + "c")
