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


def _save_csv(records, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return df


def _figpaths(stem):
    """Zapisujemy figury rownolegle do figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / stem


def experiment_vs_cpsat(cfg):
    print("[1/3] vs CP-SAT ...")
    records, extra = scenarios.run_vs_cpsat(
        seeds=cfg["seeds"], gens=cfg["gens"], pop=cfg["pop"],
        cpsat_time=cfg["cpsat_time"])
    df = _save_csv(records, "vs_cpsat")
    inst, cp = extra["instance"], extra["cpsat"]
    gantt.gantt_matplotlib(inst, cp.schedule, _figpaths("gantt_cpsat"),
                           title="Harmonogram CP-SAT (optimum)")
    plots.plot_coverage(inst, cp.schedule, _figpaths("coverage_cpsat"), day=1,
                        title="CP-SAT: pokrycie popytu (dzien 1)")
    print(df.to_string(index=False))
    return df


def experiment_ablation(cfg):
    print("[2/3] Ablacja (LS) ...")
    records, extra = scenarios.run_ablation(
        seeds=cfg["seeds"], gens=cfg["gens"], pop=cfg["pop"])
    df = _save_csv(records, "ablation")

    plots.plot_convergence(extra["histories"], _figpaths("convergence_ablation"),
                           title="Zbieznosc: memetyk vs NSGA-II")

    # krzywa hiperobjetosci znormalizowana do 1; punkt odniesienia HV jest
    # wspolny dla obu metod w ramach ziarna, wiec normalizujemy per ziarno
    # przez maksimum z obu metod
    hv_runs = {m: np.asarray(v, dtype=float)
               for m, v in extra["hv_histories"].items()}
    denom = np.maximum.reduce([a.max(axis=1) for a in hv_runs.values()])
    hv_norm = {m: a / denom[:, None] for m, a in hv_runs.items()}
    plots.plot_convergence(hv_norm, _figpaths("hv_ablation"),
                           ylabel="hiperobjetosc (znormalizowana)",
                           title="Hiperobjetosc frontu: memetyk vs NSGA-II")
    hv_rows = [
        {"method": m, "seed": seed, "generation": g, "hypervolume_norm": val}
        for m, runs in hv_norm.items()
        for seed, run in zip(cfg["seeds"], runs)
        for g, val in enumerate(run)
    ]
    _save_csv(hv_rows, "hv_history")

    plots.plot_pareto(extra["pareto"], _figpaths("pareto_ablation"),
                      title="Fronty Pareto: memetyk vs NSGA-II")
    groups = {m: df[df.method == m]["cost"].values for m in df.method.unique()}
    plots.plot_boxplot(groups, _figpaths("boxplot_ablation"), title="Koszt: memetyk vs NSGA-II")

    print(df.to_string(index=False))
    return df


def experiment_disruption(cfg):
    print("[3/3] Zaburzenie (absencja + reopt) ...")
    records, example = scenarios.run_disruption(
        seeds=cfg["seeds"], gens=cfg["gens"], pop=cfg["pop"])
    df = _save_csv(records, "disruption")
    if example is not None:
        inst = example["instance"]
        gantt.gantt_matplotlib(inst, example["base"], _figpaths("gantt_before_absence"),
                               title="Grafik przed absencja")
        gantt.gantt_matplotlib(inst, example["reopt"].schedule, _figpaths("gantt_after_reopt"),
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
    print("Gotowe. Wyniki w results/, figury w figures/.")


if __name__ == "__main__":
    main()
