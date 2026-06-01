"""CLI uruchamiajace eksperymenty, zapisujace CSV oraz generujace wykresy i tabele.

Przyklady:
    uv run python -m amhe.experiments.run --smoke           # szybki przebieg dymny
    uv run python -m amhe.experiments.run --all             # pelne 3 scenariusze
    uv run python -m amhe.experiments.run --scenario ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from amhe.analysis import gantt, plots, tables
from amhe.analysis.stats import compare_variants
from amhe.experiments import scenarios

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
REPORT_FIG_DIR = Path("report/figures")
REPORT_TAB_DIR = Path("report/tables")


def _save_csv(records, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return df


def _figpaths(stem):
    """Zapisujemy figury rownolegle do figures/ i report/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / stem, REPORT_FIG_DIR / stem


def experiment_vs_cpsat(cfg):
    print("[1/3] vs CP-SAT ...")
    records, extra = scenarios.run_vs_cpsat(
        seeds=cfg["seeds"], gens=cfg["gens"], pop=cfg["pop"],
        cpsat_time=cfg["cpsat_time"])
    df = _save_csv(records, "vs_cpsat")
    inst, cp = extra["instance"], extra["cpsat"]
    for stem in _figpaths("gantt_cpsat"):
        gantt.gantt_matplotlib(inst, cp.schedule, stem,
                               title="Harmonogram CP-SAT (optimum)")
    gantt.gantt_plotly(inst, cp.schedule, REPORT_FIG_DIR / "gantt_cpsat",
                       title="Harmonogram CP-SAT (interaktywny)")
    for stem in _figpaths("coverage_cpsat"):
        plots.plot_coverage(inst, cp.schedule, stem, day=1,
                            title="CP-SAT: pokrycie popytu (dzien 1)")
    print(df.to_string(index=False))
    return df


def experiment_ablation(cfg):
    print("[2/3] Ablacja (LS) ...")
    records, extra = scenarios.run_ablation(
        seeds=cfg["seeds"], gens=cfg["gens"], pop=cfg["pop"])
    df = _save_csv(records, "ablation")

    for stem in _figpaths("convergence_ablation"):
        plots.plot_convergence(extra["histories"], stem,
                               title="Zbieznosc: memetyk vs NSGA-II")
    for stem in _figpaths("pareto_ablation"):
        plots.plot_pareto(extra["pareto"], stem,
                          title="Fronty Pareto: memetyk vs NSGA-II")
    groups = {m: df[df.method == m]["cost"].values for m in df.method.unique()}
    for stem in _figpaths("boxplot_ablation"):
        plots.plot_boxplot(groups, stem, title="Koszt: memetyk vs NSGA-II")

    REPORT_TAB_DIR.mkdir(parents=True, exist_ok=True)
    tables.summary_table(REPORT_TAB_DIR / "ablation_summary.tex", groups,
                         caption="Statystyki kosztu — ablacja przeszukiwania lokalnego",
                         label="tab:ablation_summary")
    hv_groups = {m: df[df.method == m]["hypervolume"].values for m in df.method.unique()}
    tables.summary_table(REPORT_TAB_DIR / "ablation_hv.tex", hv_groups,
                         caption="Statystyki hiperobjetosci — ablacja",
                         label="tab:ablation_hv", metric_name="hiperobjetosc")
    if all(len(v) >= 2 for v in groups.values()):
        cmp = compare_variants(groups)
        tables.comparison_table(REPORT_TAB_DIR / "ablation_test.tex", cmp,
                               caption="Test Manna-Whitneya (koszt) — memetyk vs NSGA-II",
                               label="tab:ablation_test")
    print(df.to_string(index=False))
    return df


def experiment_disruption(cfg):
    print("[3/3] Zaburzenie (absencja + reopt) ...")
    records, example = scenarios.run_disruption(
        seeds=cfg["seeds"], gens=cfg["gens"], pop=cfg["pop"])
    df = _save_csv(records, "disruption")
    if example is not None:
        inst = example["instance"]
        for stem in _figpaths("gantt_before_absence"):
            gantt.gantt_matplotlib(inst, example["base"], stem,
                                   title="Grafik przed absencja")
        for stem in _figpaths("gantt_after_reopt"):
            gantt.gantt_matplotlib(inst, example["reopt"].schedule, stem,
                                   title="Grafik po reoptymalizacji")
    print(df.to_string(index=False))
    return df


SMOKE = dict(seeds=(1,), gens=6, pop=12, cpsat_time=5.0)
FULL = dict(seeds=(1, 2, 3, 4, 5), gens=60, pop=40, cpsat_time=30.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Eksperymenty AMHE — call center")
    parser.add_argument("--smoke", action="store_true",
                        help="szybki przebieg dymny (male budzety)")
    parser.add_argument("--all", action="store_true", help="wszystkie scenariusze")
    parser.add_argument("--scenario", choices=["vs_cpsat", "ablation", "disruption"],
                        help="pojedynczy scenariusz")
    args = parser.parse_args(argv)

    cfg = SMOKE if args.smoke else FULL
    run_all = args.all or (not args.scenario)

    if run_all or args.scenario == "vs_cpsat":
        experiment_vs_cpsat(cfg)
    if run_all or args.scenario == "ablation":
        experiment_ablation(cfg)
    if run_all or args.scenario == "disruption":
        experiment_disruption(cfg)
    print("Gotowe. Wyniki w results/, figury w figures/ i report/figures/.")


if __name__ == "__main__":
    main()
