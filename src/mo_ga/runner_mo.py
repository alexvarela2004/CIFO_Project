"""
mo_ga/runner_mo.py
------------------
Runner for the NSGA-II multi-objective GA.

Trains the GA optimising RMSE and CIEDE2000 simultaneously.
After each run, cross-evaluates the Pareto front against the single-objective
RMSE GA result (if provided) for academic comparison.

The comparison answers: does multi-objective optimisation produce a better
trade-off between perceptual metrics than single-objective RMSE minimisation?

Usage
-----
    # Run NSGA-II (3 objectives: RMSE + CIEDE)
    python -m mo_ga.runner_mo --target data/girl_pearl.png

    # With cross-evaluation against a finished RMSE run
    python -m mo_ga.runner_mo --target data/girl_pearl.png \\
        --rmse-best runner_outputs/p5_mut_gaussian_decay/seed_42/best_final_triangles.json

    # Only run specific seeds
    python -m mo_ga.runner_mo --target data/girl_pearl.png --seeds 42

Output structure
----------------
runner_outputs/
    mo_ga/
        seed_42/
            best_rmse_individual.png         <- front member with best RMSE
            best_ciede_individual.png        <- front member with best CIEDE
            pareto_front.json                <- all front members
            generation_log.json
            ga_config.json
            checkpoints/
                gen_0000.png
                gen_0000_metrics.json
                ...
        seed_43/
        seed_44/
        mo_comparison_results.csv

mo_comparison_results.csv columns
----------------------------------
seed, front_size,
mo_best_rmse, mo_best_ciede,
rmse_run_rmse, rmse_run_ciede,
elapsed_s, n_generations_run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from typing import List, Optional

import numpy as np

from fitness import RMSEFitness, CIEDEFitness
from ga import GAConfig, EarlyStopping, DiversityAwareEarlyStopping
from ga_operators.crossover import BlendCrossover
from ga_operators.mutation import GaussianMutation, SigmaDecayScheduler
from mo_ga.mo_ga import MOGA
from mo_ga.mo_individual import MOIndividual
from utils import load_target, save_render, render, triangles_to_json, triangles_from_json

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("runner_mo")

# ---------------------------------------------------------------------------
# Constants 
# ---------------------------------------------------------------------------

SEEDS                     = [42, 43, 44]
N_GENERATIONS             = 3000
POPULATION_SIZE           = 150
CROSSOVER_RATE            = 0.8
N_ELITES                  = 3        
IMAGE_CHECKPOINT_INTERVAL = 100

# Mutation hyperparams 
_MUTATION_RATE    = 0.01
_VERTEX_SIGMA_MAX = 80.0
_VERTEX_SIGMA_MIN = 2.0
_COLOR_SIGMA_MAX  = 80.0
_COLOR_SIGMA_MIN  = 2.0

_INIT_STRATEGY = "quadrant"   

OUTPUT_ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runner_outputs")
MO_OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, "mo_ga")
COMPARISON_CSV = os.path.join(MO_OUTPUT_DIR, "mo_comparison_results.csv")

# ---------------------------------------------------------------------------
# CSV helper
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

def run_seed(
    seed: int,
    target: np.ndarray,
    rmse_best_path: Optional[str],
) -> dict:
    """
    Run NSGA-II for one seed. Cross-evaluate Pareto front and RMSE best.

    Returns one result row for mo_comparison_results.csv.
    """
    seed_dir = os.path.join(MO_OUTPUT_DIR, f"seed_{seed}")
    checkpoints_dir = os.path.join(seed_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    logger.info("--- Starting NSGA-II | seed=%d ---", seed)

    # -- Build fitness functions
    rmse_fn  = RMSEFitness(target)
    ciede_fn = CIEDEFitness(target)
    fitness_fns = [rmse_fn, ciede_fn]

    # -- Build operators (same best config from phases 1-11)
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
        checkpoint_interval=0,   # handled manually in callback
        seed=seed,
    )

    moga = MOGA(
        fitness_fns=fitness_fns,
        crossover=BlendCrossover(alpha=0.5),
        mutation=mutation_op,
        config=ga_config,
        tournament_size=2,
    )

    # -- Callback for checkpoints and progress logging
    def _callback(
        generation: int,
        best: MOIndividual,
        stats: dict,
        pareto_front: List[MOIndividual],
    ) -> None:
        scheduler(generation, best, stats)

        if generation % IMAGE_CHECKPOINT_INTERVAL == 0:
            rendered = render(best.triangles)
            save_render(rendered, os.path.join(checkpoints_dir, f"gen_{generation:04d}.png"))

            pop = moga.current_population
            div = pop.diversity_report()
            front_stats = pop.pareto_front_stats()

            metrics = {
                "generation":         generation,
                "best_rmse":          round(stats.get("best", float("nan")), 6),
                "front_size":         front_stats["front_size"],
                "mean_fitness":       round(stats.get("mean", float("nan")), 6),
                "std_fitness":        round(stats.get("std",  float("nan")), 6),
                "phenotypic_variance": round(div["phenotypic_variance"], 6),
                "genotypic_variance":  round(div["genotypic_variance"], 6),
            }

            # Per-objective front min
            obj_names = ["rmse", "ciede"]
            for i, name in enumerate(obj_names):
                if i < len(front_stats.get("obj_mins", [])):
                    metrics[f"front_{name}_min"] = round(front_stats["obj_mins"][i], 6)

            with open(
                os.path.join(checkpoints_dir, f"gen_{generation:04d}_metrics.json"),
                "w", encoding="utf-8",
            ) as f:
                json.dump(metrics, f, indent=2)

            logger.info(
                "Checkpoint gen %d | seed=%d | best_rmse=%.4f | front_size=%d",
                generation, seed, stats.get("best", float("nan")), front_stats["front_size"],
            )

    # -- Run
    t0 = time.time()
    pareto_front = moga.run(
        target=target,
        init_strategy=_INIT_STRATEGY,
        callback=_callback,
    )
    elapsed = time.time() - t0
    n_generations_run = len(moga.generation_log) - 1

    # -- Save outputs
    # Best member by each objective
    if pareto_front:
        best_by_rmse  = min(pareto_front, key=lambda x: x.fitness_values[0])
        best_by_ciede = min(pareto_front, key=lambda x: x.fitness_values[1])

        save_render(render(best_by_rmse.triangles),
                    os.path.join(seed_dir, "best_rmse_individual.png"))
        save_render(render(best_by_ciede.triangles),
                    os.path.join(seed_dir, "best_ciede_individual.png"))

        triangles_to_json(
            list(best_by_rmse.triangles),
            os.path.join(seed_dir, "best_rmse_triangles.json"),
        )

    moga.save_pareto_front(
        os.path.join(seed_dir, "pareto_front.json"), pareto_front
    )
    moga.save_log(os.path.join(seed_dir, "generation_log.json"))
    moga.save_config(os.path.join(seed_dir, "ga_config.json"))

    # -- Pareto front statistics for the result row
    if pareto_front:
        mo_best_rmse  = min(ind.fitness_values[0] for ind in pareto_front)
        mo_best_ciede = min(ind.fitness_values[1] for ind in pareto_front)
    else:
        mo_best_rmse = mo_best_ciede = float("nan")

    logger.info(
        "NSGA-II done | seed=%d | front_size=%d | "
        "best_rmse=%.4f | best_ciede=%.4f | %.1fs",
        seed, len(pareto_front),
        mo_best_rmse, mo_best_ciede, elapsed,
    )

    # -- Cross-evaluate RMSE best if provided
    rmse_run_rmse = rmse_run_ciede = ""
    if rmse_best_path is not None:
        logger.info("Loading RMSE best from %s", rmse_best_path)
        rmse_triangles = triangles_from_json(rmse_best_path)
        rmse_rendered  = render(rmse_triangles)
        rmse_run_rmse  = round(rmse_fn.evaluate(rmse_rendered),  6)
        rmse_run_ciede = round(ciede_fn.evaluate(rmse_rendered), 6)
        logger.info(
            "RMSE cross-eval | seed=%d | rmse=%.4f | ciede=%.4f",
            seed, rmse_run_rmse, rmse_run_ciede
        )

    row = {
        "seed":             seed,
        "front_size":       len(pareto_front),
        "n_generations_run": n_generations_run,
        "elapsed_s":        round(elapsed, 1),
        "mo_best_rmse":     round(mo_best_rmse,  6) if not isinstance(mo_best_rmse, float) or not np.isnan(mo_best_rmse) else "",
        "mo_best_ciede":    round(mo_best_ciede, 6) if not isinstance(mo_best_ciede, float) or not np.isnan(mo_best_ciede) else "",
        "rmse_run_rmse":    rmse_run_rmse,
        "rmse_run_ciede":   rmse_run_ciede,
    }

    _append_csv(COMPARISON_CSV, row)
    logger.info("Comparison CSV updated -> %s", COMPARISON_CSV)
    return row

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NSGA-II multi-objective GA for triangle image approximation."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/girl_pearl.png",
        help="Path to the target image.",
    )
    parser.add_argument(
        "--rmse-best",
        dest="rmse_best",
        type=str,
        default=None,
        help=(
            "Path to best_final_triangles.json from a finished RMSE run. "
            "When provided, the RMSE best is cross-evaluated with all metrics."
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
    args  = parse_args()
    seeds = args.seeds if args.seeds is not None else SEEDS

    os.makedirs(MO_OUTPUT_DIR, exist_ok=True)

    target = load_target(args.target)
    logger.info(
        "Target loaded | shape=%s | seeds=%s",
        target.shape, seeds,
    )

    if args.rmse_best is not None and not os.path.isfile(args.rmse_best):
        raise FileNotFoundError(f"RMSE best not found: {args.rmse_best}")

    all_results = []
    for seed in seeds:
        try:
            row = run_seed(seed, target, args.rmse_best)
            all_results.append(row)
        except Exception as e:
            logger.error("Seed %d failed: %s", seed, e, exc_info=True)

    # Summary
    if all_results:
        print("\n--- NSGA-II Summary ---")
        print(f"{'Seed':>5} {'Front':>6} {'Best RMSE':>10} {'Best CIEDE':>12}")
        print("-" * 50)
        for r in all_results:
            print(
                f"{r['seed']:>5} "
                f"{r['front_size']:>6} "
                f"{r['mo_best_rmse']:>10} "
                f"{r['mo_best_ciede']:>12} "
            )

    logger.info("All done. Results in %s", COMPARISON_CSV)


if __name__ == "__main__":
    main()
