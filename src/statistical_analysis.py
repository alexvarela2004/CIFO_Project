"""
statistical_analysis.py
-----------------------
Statistical comparison: Best model (p12_final) vs Baseline (p1_baseline).

Answers the core question: is our optimised GA statistically better than
the starting point?

Best model (p12_final):
    selection  = tournament_k10
    crossover  = blend
    mutation   = gaussian_decay  (sigma_max=80, rate=0.01)
    n_elites   = 3
    pop_size   = 50
    n_gens     = 1000

Baseline (p1_baseline):
    selection  = tournament_k3
    crossover  = uniform
    mutation   = gaussian_fixed
    n_elites   = 5
    pop_size   = 50
    n_gens     = 1000

Statistical pipeline:
    - Shapiro-Wilk normality test on each sample
    - Both normal  -> Welch t-test  + Cohen's d
    - Otherwise    -> Wilcoxon rank-sum + rank-biserial r
    - Significance at alpha=0.05

Outputs (in stat_analysis_outputs/):
    comparison_raw_rmse.json        all RMSE values
    comparison_raw_rmse.csv         same, columnar
    comparison_stat_summary.csv     statistical summary
    comparison_boxplot.png/.pdf     boxplot with significance annotation

Usage
-----
    python statistical_analysis.py --target data/girl_pearl.png
    python statistical_analysis.py --target data/girl_pearl.png --n_runs 30
    python statistical_analysis.py --target data/girl_pearl.png --n_runs 3 --fast
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fitness import RMSEFitness
from ga import GeneticAlgorithm, GAConfig, EarlyStopping, DiversityAwareEarlyStopping
from ga_operators.selection import TournamentSelection
from ga_operators.crossover import BlendCrossover, UniformCrossover
from ga_operators.mutation import GaussianMutation, SigmaDecayScheduler
from ga_utils import load_target

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stat_analysis")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POPULATION_SIZE = 50
N_GENERATIONS   = 1000
ALPHA           = 0.05
OUTPUT_DIR      = "stat_analysis_outputs"


# ---------------------------------------------------------------------------
# Config descriptor
# ---------------------------------------------------------------------------
@dataclass
class Config:
    name: str
    label: str
    description: str
    tournament_k: int
    crossover: str        # "blend" | "uniform"
    mutation: str         # "gaussian_decay" | "gaussian_fixed"
    n_elites: int
    crossover_rate: float = 0.8
    mut_rate: float       = 0.01
    sigma_max: float      = 80.0
    sigma_min: float      = 2.0
    init_strategy: str    = "random" 


BEST_MODEL = Config(
    name         = "best_model",
    label        = "Best Model\n(p12_final)",
    description  = "tournament_k10 + blend + gaussian_decay + 3 elites",
    tournament_k = 10,
    crossover    = "blend",
    mutation     = "gaussian_decay",
    n_elites     = 3,
    crossover_rate = 0.8,
    mut_rate     = 0.01,
    sigma_max    = 80.0,
    sigma_min    = 2.0,
    init_strategy = "quadrant"
)

BASELINE = Config(
    name         = "baseline",
    label        = "Baseline\n(p1_baseline)",
    description  = "tournament_k3 + uniform + gaussian_fixed + 5 elites",
    tournament_k = 3,
    crossover    = "uniform",
    mutation     = "gaussian_fixed",
    n_elites     = 5,
    crossover_rate = 0.8,
    mut_rate     = 0.05,
    sigma_max    = 15.0,
    sigma_min    = 15.0,
    init_strategy = "random"
)

CONFIGS = [BEST_MODEL, BASELINE]


# ---------------------------------------------------------------------------
# Operator factories
# ---------------------------------------------------------------------------
def make_selection(k: int):
    return TournamentSelection(tournament_size=k)


def make_crossover(name: str):
    if name == "blend":
        return BlendCrossover(alpha=0.5)
    if name == "uniform":
        return UniformCrossover(swap_prob=0.5)
    raise ValueError(f"Unknown crossover: {name}")


def make_mutation(cfg: Config, n_generations: int):
    if cfg.mutation == "gaussian_decay":
        op = GaussianMutation(
            mutation_rate=cfg.mut_rate,
            vertex_sigma=cfg.sigma_max,
            color_sigma=cfg.sigma_max,
        )
        scheduler = SigmaDecayScheduler(
            mutation=op,
            n_generations=n_generations,
            vertex_sigma_max=cfg.sigma_max,
            vertex_sigma_min=cfg.sigma_min,
            color_sigma_max=cfg.sigma_max,
            color_sigma_min=cfg.sigma_min,
        )
        return op, scheduler

    if cfg.mutation == "gaussian_fixed":
        op = GaussianMutation(
            mutation_rate=cfg.mut_rate,
            vertex_sigma=cfg.sigma_max,
            color_sigma=cfg.sigma_max,
        )
        return op, None

    raise ValueError(f"Unknown mutation: {cfg.mutation}")


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(cfg: Config, seed: int, target: np.ndarray,
               n_generations: int) -> float:
    fitness_fn = RMSEFitness(target)
    selection  = make_selection(cfg.tournament_k)
    crossover  = make_crossover(cfg.crossover)
    mutation_op, scheduler = make_mutation(cfg, n_generations)

    ga_config = GAConfig(
        population_size=POPULATION_SIZE,
        n_generations=n_generations,
        crossover_rate=cfg.crossover_rate,
        n_elites=cfg.n_elites,
        n_workers=1,
        early_stopping=EarlyStopping(patience=200, tolerance=1e-4),
        diversity_early_stopping=DiversityAwareEarlyStopping(
            patience=200, tolerance=1e-4,
            phenotypic_variance_threshold=0.01,
            genotypic_variance_threshold=1e-4,
            diversity_check_interval=10,
        ),
        checkpoint_interval=0,
        seed=seed,
    )

    ga = GeneticAlgorithm(
        fitness_fn=fitness_fn,
        selection=selection,
        crossover=crossover,
        mutation=mutation_op,
        config=ga_config,
    )

    def _cb(generation: int, best, stats_dict: dict) -> None:
        if scheduler is not None:
            scheduler(generation, best, stats_dict)

    best = ga.run(target=target, init_strategy=cfg.init_strategy, callback=_cb)
    return float(best.fitness)


# ---------------------------------------------------------------------------
# Run both configs
# ---------------------------------------------------------------------------
def collect_results(target: np.ndarray, n_runs: int,
                    n_generations: int) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {}
    total = len(CONFIGS) * n_runs
    done  = 0

    for cfg in CONFIGS:
        rmses: List[float] = []
        logger.info("=== Config: %s (%d runs) ===", cfg.name, n_runs)
        for seed in range(n_runs):
            t0 = time.time()
            try:
                rmse = run_single(cfg, seed, target, n_generations)
                rmses.append(rmse)
                done += 1
                logger.info(
                    "  [%d/%d] %s seed=%d  RMSE=%.4f  (%.1fs)",
                    done, total, cfg.name, seed, rmse, time.time() - t0,
                )
            except Exception as e:
                logger.error("  FAILED %s seed=%d: %s", cfg.name, seed, e, exc_info=True)
        results[cfg.name] = rmses

    return results


# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------
def normality_test(data: List[float]) -> Tuple[float, float, bool]:
    n = len(data)
    if n < 3:
        return float("nan"), float("nan"), False
    if n <= 50:
        stat, p = stats.shapiro(data)
    else:
        stat, p = stats.kstest(data, "norm", args=(np.mean(data), np.std(data, ddof=1)))
    return float(stat), float(p), bool(p > ALPHA)


def run_statistical_test(best_rmses: List[float],
                         baseline_rmses: List[float]) -> Dict[str, Any]:
    sw_stat_best,     sw_p_best,     normal_best     = normality_test(best_rmses)
    sw_stat_baseline, sw_p_baseline, normal_baseline = normality_test(baseline_rmses)
    both_normal = normal_best and normal_baseline

    if both_normal:
        test_name = "Welch t-test"
        stat, p_value = stats.ttest_ind(
            best_rmses, baseline_rmses, equal_var=False, alternative="two-sided"
        )
        pooled_std = np.sqrt(
            (np.std(best_rmses, ddof=1) ** 2 + np.std(baseline_rmses, ddof=1) ** 2) / 2
        )
        effect_size  = (np.mean(baseline_rmses) - np.mean(best_rmses)) / pooled_std \
                       if pooled_std > 0 else 0.0
        effect_label = "Cohen's d"
    else:
        test_name = "Wilcoxon rank-sum"
        stat, p_value = stats.ranksums(best_rmses, baseline_rmses)
        n1, n2 = len(best_rmses), len(baseline_rmses)
        u_stat, _ = stats.mannwhitneyu(best_rmses, baseline_rmses, alternative="two-sided")
        effect_size  = 1 - (2 * u_stat) / (n1 * n2)
        effect_label = "rank-biserial r"

    significant = bool(p_value < ALPHA)

    return {
        # normality
        "normality_test":        "Shapiro-Wilk" if len(best_rmses) <= 50 else "KS",
        "sw_p_best":             round(float(sw_p_best),      4),
        "sw_p_baseline":         round(float(sw_p_baseline),  4),
        "normal_best":           normal_best,
        "normal_baseline":       normal_baseline,
        "both_normal":           both_normal,
        # descriptive
        "n_best":                len(best_rmses),
        "n_baseline":            len(baseline_rmses),
        "mean_best":             round(float(np.mean(best_rmses)),           4),
        "mean_baseline":         round(float(np.mean(baseline_rmses)),       4),
        "std_best":              round(float(np.std(best_rmses,    ddof=1)), 4),
        "std_baseline":          round(float(np.std(baseline_rmses, ddof=1)), 4),
        "median_best":           round(float(np.median(best_rmses)),         4),
        "median_baseline":       round(float(np.median(baseline_rmses)),     4),
        # test
        "test":                  test_name,
        "statistic":             round(float(stat),       4),
        "p_value":               round(float(p_value),    6),
        "significant":           significant,
        "effect_size":           round(float(effect_size), 4),
        "effect_label":          effect_label,
        "conclusion":            "best model is significantly better than baseline"
                                 if significant and np.mean(best_rmses) < np.mean(baseline_rmses)
                                 else "no statistically significant difference",
    }


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
def save_outputs(results: Dict[str, List[float]],
                 test_result: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # raw JSON
    with open(os.path.join(out_dir, "comparison_raw_rmse.json"), "w") as f:
        json.dump(results, f, indent=2)

    # raw CSV
    raw_path = os.path.join(out_dir, "comparison_raw_rmse.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["best_model", "baseline"])
        max_runs = max(len(v) for v in results.values())
        for i in range(max_runs):
            writer.writerow([
                results["best_model"][i] if i < len(results["best_model"]) else "",
                results["baseline"][i]   if i < len(results["baseline"])   else "",
            ])
    logger.info("Raw RMSE saved -> %s", raw_path)

    # statistical summary CSV
    summary_path = os.path.join(out_dir, "comparison_stat_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_result.keys()))
        writer.writeheader()
        writer.writerow(test_result)
    logger.info("Statistical summary saved -> %s", summary_path)


# ---------------------------------------------------------------------------
# Boxplot
# ---------------------------------------------------------------------------
def make_boxplot(results: Dict[str, List[float]],
                 test_result: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    data   = [results["best_model"], results["baseline"]]
    labels = [BEST_MODEL.label, BASELINE.label]
    colors = ["#2196F3", "#FF7043"]

    bp = ax.boxplot(
        data, labels=labels, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=5, alpha=0.5),
        widths=0.45,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # significance bracket between the two boxes
    p     = test_result["p_value"]
    if   p < 0.001: stars = "***"
    elif p < 0.01:  stars = "**"
    elif p < 0.05:  stars = "*"
    else:            stars = "ns"

    ymax    = max(max(data[0]), max(data[1]))
    ymin    = min(min(data[0]), min(data[1]))
    y_range = ymax - ymin
    y_bar   = ymax + y_range * 0.07
    y_text  = y_bar + y_range * 0.03

    ax.plot([1, 1, 2, 2], [y_bar - y_range*0.01, y_bar, y_bar, y_bar - y_range*0.01],
            lw=1.5, color="black")
    sig_color = "#c62828" if test_result["significant"] else "#757575"
    ax.text(1.5, y_text, stars, ha="center", va="bottom",
            fontsize=15, color=sig_color, fontweight="bold")

    ax.set_ylim(ymin - y_range * 0.05, y_text + y_range * 0.1)
    ax.set_ylabel("Final Best RMSE", fontsize=12)
    ax.set_title(
        "Best Model vs Baseline\n"
        f"{test_result['test']}  p={test_result['p_value']:.4e}"
        f"  {test_result['effect_label']}={test_result['effect_size']:.3f}\n"
        f"(pop={POPULATION_SIZE}, gens={N_GENERATIONS}, n={test_result['n_best']} runs each"
        f" | * p<0.05  ** p<0.01  *** p<0.001  ns = not significant)",
        fontsize=10, pad=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"comparison_boxplot.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Boxplot saved -> %s", path)
    plt.close()


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
def print_summary(results: Dict[str, List[float]],
                  test_result: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("  STATISTICAL COMPARISON: Best Model vs Baseline")
    print(f"  pop={POPULATION_SIZE}  gens={N_GENERATIONS}  α={ALPHA}")
    print("=" * 70)
    print(f"  {'':25}  {'Best Model':>15}  {'Baseline':>15}")
    print(f"  {'n runs':25}  {test_result['n_best']:>15}  {test_result['n_baseline']:>15}")
    print(f"  {'mean RMSE':25}  {test_result['mean_best']:>15.4f}  {test_result['mean_baseline']:>15.4f}")
    print(f"  {'std RMSE':25}  {test_result['std_best']:>15.4f}  {test_result['std_baseline']:>15.4f}")
    print(f"  {'median RMSE':25}  {test_result['median_best']:>15.4f}  {test_result['median_baseline']:>15.4f}")
    print("-" * 70)
    print(f"  Normality test:  {test_result['normality_test']}")
    print(f"    best model  p={test_result['sw_p_best']:.4f}  "
          f"({'normal' if test_result['normal_best'] else 'NOT normal'})")
    print(f"    baseline    p={test_result['sw_p_baseline']:.4f}  "
          f"({'normal' if test_result['normal_baseline'] else 'NOT normal'})")
    print(f"  Test used:       {test_result['test']}")
    print(f"  Statistic:       {test_result['statistic']:.4f}")
    print(f"  p-value:         {test_result['p_value']:.6f}  "
          f"({'significant' if test_result['significant'] else 'NOT significant'})")
    print(f"  Effect size:     {test_result['effect_label']} = {test_result['effect_size']:.4f}")
    print("-" * 70)
    print(f"  Conclusion: {test_result['conclusion']}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical comparison: best model vs baseline."
    )
    parser.add_argument("--target",  type=str, default="data/Girl_Pearl_Earing.png",
                        help="Path to the target image.")
    parser.add_argument("--n_runs",  type=int, default=30,
                        help="Independent runs per config (default: 30).")
    parser.add_argument("--fast",    action="store_true",
                        help="Use 100 generations for quick testing.")
    parser.add_argument("--out_dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_generations = 100 if args.fast else N_GENERATIONS

    logger.info("Loading target image from %s", args.target)
    target = load_target(args.target)
    logger.info("Target shape: %s", target.shape)
    logger.info(
        "Running 2 configs × %d runs = %d total GA runs  (pop=%d, gens=%d)",
        args.n_runs, 2 * args.n_runs, POPULATION_SIZE, n_generations,
    )

    # ── Run ───────────────────────────────────────────────────────────────
    results = collect_results(target, args.n_runs, n_generations)

    # ── Statistical test ──────────────────────────────────────────────────
    test_result = run_statistical_test(results["best_model"], results["baseline"])

    # ── Output ────────────────────────────────────────────────────────────
    print_summary(results, test_result)
    save_outputs(results, test_result, args.out_dir)
    make_boxplot(results, test_result, args.out_dir)

    logger.info("All done. Outputs in %s/", args.out_dir)


if __name__ == "__main__":
    main()