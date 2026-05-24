"""
runner_ciede.py
---------------
Runs the GA with CIEDEFitness (perceptual colour metric) and cross-evaluates
the best individual against RMSEFitness for academic comparison.

Implements additional challenge option 1: CIEDE2000 as fitness function.

Usage
-----
    # Run CIEDE GA only (seeds 42, 43, 44)
    python runner_ciede.py --target ../data/girl_pearl.png

    # With cross-evaluation against a finished RMSE run
    python runner_ciede.py --target ../data/girl_pearl.png \\
        --rmse-best ../runner_outputs/<run>/seed_42/best_final_triangles.json

Output structure
----------------
runner_outputs/
    ciede_comparison/
        seed_42/
            best_final.png
            best_final_triangles.json
            generation_log.json
            ga_config.json
            checkpoints/
                gen_0000.png
                gen_0000_metrics.json
                ...
        seed_43/
        seed_44/
        comparison_results.csv

comparison_results.csv columns
-------------------------------
seed, ciede_run_rmse, ciede_run_ciede, rmse_run_rmse, rmse_run_ciede

  ciede_run_rmse  -- best CIEDE individual evaluated with RMSEFitness
  ciede_run_ciede -- best CIEDE individual evaluated with CIEDEFitness
  rmse_run_rmse   -- best RMSE individual evaluated with RMSEFitness   (--rmse-best only)
  rmse_run_ciede  -- best RMSE individual evaluated with CIEDEFitness  (--rmse-best only)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from typing import Optional

import numpy as np

from fitness import RMSEFitness, CIEDEFitness
from ga import GeneticAlgorithm, GAConfig, EarlyStopping, DiversityAwareEarlyStopping
from ga_operators.selection import TournamentSelection
from ga_operators.crossover import KPointCrossover
from ga_operators.mutation import GaussianMutation, SigmaDecayScheduler
from utils import load_target, save_render, render, triangles_to_json, triangles_from_json

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("runner_ciede")

# ---------------------------------------------------------------------------
# Hardcoded best config (matches best result from phases 1-11)
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]
N_GENERATIONS = 3000
POPULATION_SIZE = 50
CROSSOVER_RATE = 0.8
N_ELITES = 3
INIT_STRATEGY = "quadrant"
IMAGE_CHECKPOINT_INTERVAL = 100

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runner_outputs")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "ciede_comparison")
COMPARISON_CSV = os.path.join(OUTPUT_DIR, "comparison_results.csv")

# Mutation hyperparams — best from phase 10 grid search
_MUTATION_RATE = 0.01
_VERTEX_SIGMA_MAX = 80.0
_VERTEX_SIGMA_MIN = 2.0
_COLOR_SIGMA_MAX = 80.0
_COLOR_SIGMA_MIN = 2.0


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _append_csv(filepath: str, row: dict) -> None:
    file_exists = os.path.isfile(filepath)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, target: np.ndarray, rmse_best_path: Optional[str]) -> None:
    """
    Run GA with CIEDEFitness for one seed, save outputs, cross-evaluate,
    and append one row to comparison_results.csv.
    """
    seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
    checkpoints_dir = os.path.join(seed_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    logger.info("--- Starting CIEDE run | seed=%d ---", seed)

    ciede_fn = CIEDEFitness(target)
    rmse_fn = RMSEFitness(target)

    mutation_op = GaussianMutation(
        mutation_rate=_MUTATION_RATE,
        vertex_sigma=_VERTEX_SIGMA_MAX,
        color_sigma=_COLOR_SIGMA_MAX,
    )
    scheduler = SigmaDecayScheduler(
        mutation=mutation_op,
        n_generations=N_GENERATIONS,
        vertex_sigma_max=_VERTEX_SIGMA_MAX,
        vertex_sigma_min=_VERTEX_SIGMA_MIN,
        color_sigma_max=_COLOR_SIGMA_MAX,
        color_sigma_min=_COLOR_SIGMA_MIN,
    )

    ga_config = GAConfig(
        population_size=POPULATION_SIZE,
        n_generations=N_GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        n_elites=N_ELITES,
        n_workers=1,
        early_stopping=EarlyStopping(patience=100, tolerance=1e-4),
        diversity_early_stopping=DiversityAwareEarlyStopping(
            patience=100,
            tolerance=1e-4,
            phenotypic_variance_threshold=0.01,
            genotypic_variance_threshold=1e-4,
            diversity_check_interval=10,
        ),
        checkpoint_interval=0,
        seed=seed,
    )

    ga = GeneticAlgorithm(
        fitness_fn=ciede_fn,
        selection=TournamentSelection(tournament_size=10),
        crossover=KPointCrossover(k=2),
        mutation=mutation_op,
        config=ga_config,
    )

    def _callback(generation: int, best, stats: dict) -> None:
        scheduler(generation, best, stats)

        if generation % IMAGE_CHECKPOINT_INTERVAL == 0:
            rendered = render(best.triangles)
            save_render(rendered, os.path.join(checkpoints_dir, f"gen_{generation:04d}.png"))

            pop = ga.current_population
            div = pop.diversity_report()

            metrics = {
                "generation": generation,
                "best_fitness_ciede": round(best.fitness, 6),
                "mean_fitness": round(stats.get("mean", float("nan")), 6),
                "std_fitness": round(stats.get("std", float("nan")), 6),
                "worst_fitness": round(stats.get("worst", float("nan")), 6),
                "phenotypic_entropy": round(div["phenotypic_entropy"], 6),
                "genotypic_entropy": round(div["genotypic_entropy"], 6),
                "phenotypic_variance": round(div["phenotypic_variance"], 6),
                "genotypic_variance": round(div["genotypic_variance"], 6),
            }
            with open(
                os.path.join(checkpoints_dir, f"gen_{generation:04d}_metrics.json"),
                "w", encoding="utf-8",
            ) as f:
                json.dump(metrics, f, indent=2)

            logger.info(
                "Checkpoint gen %d | seed=%d | ciede=%.4f | pheno_var=%.6f",
                generation, seed, best.fitness, div["phenotypic_variance"],
            )

    t0 = time.time()
    best = ga.run(target=target, init_strategy=INIT_STRATEGY, callback=_callback)
    elapsed = time.time() - t0

    # Save best individual outputs
    save_render(render(best.triangles), os.path.join(seed_dir, "best_final.png"))
    triangles_to_json(list(best.triangles), os.path.join(seed_dir, "best_final_triangles.json"))
    ga.save_log(os.path.join(seed_dir, "generation_log.json"))
    ga.save_config(os.path.join(seed_dir, "ga_config.json"))

    # Cross-evaluate: best CIEDE individual scored with RMSE
    best_rendered = render(best.triangles)
    ciede_run_ciede = round(best.fitness, 6)
    ciede_run_rmse = round(rmse_fn.evaluate(best_rendered), 6)

    logger.info(
        "CIEDE run done | seed=%d | ciede=%.4f | rmse=%.4f | %.1fs",
        seed, ciede_run_ciede, ciede_run_rmse, elapsed,
    )

    # Cross-evaluate RMSE best if provided
    rmse_run_rmse: object = ""
    rmse_run_ciede: object = ""

    if rmse_best_path is not None:
        logger.info("Loading RMSE best from %s", rmse_best_path)
        rmse_triangles = triangles_from_json(rmse_best_path)
        rmse_rendered = render(rmse_triangles)
        rmse_run_rmse = round(rmse_fn.evaluate(rmse_rendered), 6)
        rmse_run_ciede = round(ciede_fn.evaluate(rmse_rendered), 6)
        logger.info(
            "RMSE cross-eval | seed=%d | rmse=%.4f | ciede=%.4f",
            seed, rmse_run_rmse, rmse_run_ciede,
        )

    _append_csv(COMPARISON_CSV, {
        "seed": seed,
        "ciede_run_rmse": ciede_run_rmse,
        "ciede_run_ciede": ciede_run_ciede,
        "rmse_run_rmse": rmse_run_rmse,
        "rmse_run_ciede": rmse_run_ciede,
    })
    logger.info("Comparison CSV updated -> %s", COMPARISON_CSV)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GA with CIEDEFitness and cross-evaluate against RMSEFitness."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="../data/girl_pearl.png",
        help="Path to the target image (default: ../data/girl_pearl.png).",
    )
    parser.add_argument(
        "--rmse-best",
        dest="rmse_best",
        type=str,
        default=None,
        help=(
            "Path to best_final_triangles.json from a finished RMSE run. "
            "When provided, the RMSE best is also evaluated with both metrics "
            "and the last two columns of comparison_results.csv are populated."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"Seeds to use. Defaults to {SEEDS}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else SEEDS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target = load_target(args.target)
    logger.info(
        "Target loaded from %s | shape=%s | seeds=%s",
        args.target, target.shape, seeds,
    )

    if args.rmse_best is not None:
        if not os.path.isfile(args.rmse_best):
            raise FileNotFoundError(
                f"RMSE best triangles not found at: {args.rmse_best}"
            )
        logger.info("RMSE best path: %s", args.rmse_best)
    else:
        logger.info(
            "No --rmse-best provided. rmse_run_* columns will be empty. "
            "Re-run with --rmse-best once the RMSE model finishes."
        )

    for seed in seeds:
        try:
            run_seed(seed, target, args.rmse_best)
        except Exception as e:
            logger.error("Seed %d failed: %s", seed, e, exc_info=True)

    logger.info("All seeds done. Results in %s", COMPARISON_CSV)


if __name__ == "__main__":
    main()
