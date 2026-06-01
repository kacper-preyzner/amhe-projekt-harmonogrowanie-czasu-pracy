"""Testy modulu analizy: statystyki, tabele LaTeX, generacja wykresow/plikow."""

import numpy as np

from amhe.analysis import gantt, plots, tables
from amhe.analysis.stats import (
    compare_variants,
    holm_correction,
    mann_whitney,
    summary_statistics,
)
from amhe.data.generator import scenario_small
from amhe.optim import operators as ops


def test_summary_statistics():
    s = summary_statistics([1, 2, 3, 4, 5])
    assert s["n"] == 5 and s["mean"] == 3.0 and s["median"] == 3.0
    assert s["min"] == 1.0 and s["max"] == 5.0


def test_mann_whitney_detects_difference():
    a = np.arange(0, 10)
    b = np.arange(100, 110)
    r = mann_whitney("A", a, "B", b)
    assert r.p_value < 0.05
    assert r.median_a < r.median_b


def test_holm_monotonic_and_bounds():
    r1 = mann_whitney("A", [1, 2, 3], "B", [1, 2, 3])
    r2 = mann_whitney("A", np.arange(10), "B", np.arange(100, 110))
    res = holm_correction([r1, r2])
    for r in res:
        assert 0.0 <= r.p_corrected <= 1.0


def test_compare_variants_runs():
    groups = {
        "x": np.random.default_rng(0).normal(10, 1, 8),
        "y": np.random.default_rng(1).normal(20, 1, 8),
        "z": np.random.default_rng(2).normal(10.5, 1, 8),
    }
    res = compare_variants(groups)
    assert len(res) == 3   # 3 pary


def test_summary_table_writes_latex(tmp_path):
    groups = {"a": [1, 2, 3], "b": [4, 5, 6]}
    out = tmp_path / "t.tex"
    text = tables.summary_table(out, groups, caption="Test", label="tab:test")
    assert out.exists()
    assert r"\begin{table}" in text and r"\end{table}" in text


def test_plots_create_files(tmp_path):
    hist = {"m": [[5, 4, 3, 2], [6, 5, 4, 3]]}
    plots.plot_convergence(hist, tmp_path / "conv")
    assert (tmp_path / "conv.png").exists() and (tmp_path / "conv.pdf").exists()

    fronts = {"m": np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])}
    plots.plot_pareto(fronts, tmp_path / "par")
    assert (tmp_path / "par.png").exists()


def test_gantt_create_files(tmp_path):
    inst = scenario_small()
    rng = np.random.default_rng(0)
    sch = ops.random_schedule(inst, rng, work_prob=0.8)
    gantt.gantt_matplotlib(inst, sch, tmp_path / "g")
    assert (tmp_path / "g.png").exists()
    gantt.gantt_plotly(inst, sch, tmp_path / "gp")
    assert (tmp_path / "gp.html").exists()
