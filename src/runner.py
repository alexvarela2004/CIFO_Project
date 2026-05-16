"""
runner.py
---------
Experiment runner for the GA image approximation project.

Executes the full phase-based experiment plan (phases 1-5) defined in the
project tracker, saves all outputs to runner_outputs/<run_name>/ and appends
final results to runner_outputs/results.csv.

Output structure per run
------------------------
runner_outputs/
    results.csv                         <- global results table, one row per run
    <run_name>/
        config.json                     <- full run config serialised to JSON
        generation_log.json             <- per-generation stats (fitness + diversity)
        best_final.png                  <- best individual rendered at end of run
        best_final_triangles.json       <- best individual triangles (reloadable)
        checkpoints/
            gen_0000.png                <- rendered best every 100 generations
            gen_0000_metrics.json       <- fitness + all diversity metrics at that gen
            gen_0100.png
            gen_0100_metrics.json
            ...

Usage
-----
    python runner.py --target data/girl_pearl.png
    python runner.py --target data/girl_pearl.png --phases 1 2
    python runner.py --target data/girl_pearl.png --run baseline_1

Each phase can also be run individually via --phases flag for incremental
execution across sessions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# -- project imports
from fitness import RMSEFitness, build_fitness
from ga import GeneticAlgorithm, GAConfig, EarlyStopping, DiversityAwareEarlyStopping
from ga_operators.selection import TournamentSelection, RankSelection
from ga_operators.crossover import SinglePointCrossover, UniformCrossover, KPointCrossover
from ga_operators.mutation import (
    GaussianMutation,
    CreepMutation,
    ResetMutation,
    SwapMutation,
    CompositeMutation,
    SigmaDecayScheduler,
    DeltaDecayScheduler,
)
from population import Population
from utils import load_target, save_render, render, triangles_to_json

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("runner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_ROOT = "runner_outputs"
RESULTS_CSV = os.path.join(OUTPUT_ROOT, "results.csv")
N_GENERATIONS = 3000
POPULATION_SIZE = 50
CROSSOVER_RATE = 0.8
IMAGE_CHECKPOINT_INTERVAL = 100  # save image + metrics every N generations
SEED = 42

# ---------------------------------------------------------------------------
# Run config descriptor
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """
    Fully describes one experiment run: name, phase, hyperparameters and
    operator choices. Serialised to config.json inside each run folder.
    """
    name: str
    phase: int
    description: str
    selection: str
    crossover: str
    mutation: str
    n_elites: int
    population_size: int = POPULATION_SIZE
    n_generations: int = N_GENERATIONS
    crossover_rate: float = CROSSOVER_RATE
    seed: int = SEED
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def run_dir(name: str) -> str:
    """Return and create the output directory for a named run."""
    path = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path


def save_config(cfg: RunConfig, directory: str) -> None:
    path = os.path.join(directory, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    logger.info("Config saved -> %s", path)


def append_global_results(row: dict) -> None:
    """
    Append one result row to the global results CSV.
    Creates the file with headers on first write.
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Results appended to %s", RESULTS_CSV)


# ---------------------------------------------------------------------------
# Operator factories
# ---------------------------------------------------------------------------

def make_selection(name: str, **kwargs):
    if name == "tournament_k2":
        return TournamentSelection(tournament_size=2)
    if name == "tournament_k3":
        return TournamentSelection(tournament_size=3)
    if name == "tournament_k5":
        return TournamentSelection(tournament_size=5)
    if name == "tournament_k10":
        return TournamentSelection(tournament_size=10)
    if name == "rank":
        return RankSelection(selection_pressure=kwargs.get("selection_pressure", 1.5))
    raise ValueError(f"Unknown selection: {name}")


def make_crossover(name: str):
    if name == "uniform":
        return UniformCrossover(swap_prob=0.5)
    if name == "single_point":
        return SinglePointCrossover()
    if name == "two_point":
        return KPointCrossover(k=2)
    raise ValueError(f"Unknown crossover: {name}")


def make_mutation(name: str):
    """
    Returns (mutation_operator, scheduler_or_None).
    The scheduler, if present, must be passed as the GA callback.
    """
    if name == "gaussian_fixed":
        op = GaussianMutation(mutation_rate=0.05, vertex_sigma=15.0, color_sigma=15.0)
        return op, None

    if name == "creep_fixed":
        op = CreepMutation(mutation_rate=0.05, vertex_delta=15.0, color_delta=15.0)
        return op, None

    if name == "gaussian_decay":
        op = GaussianMutation(mutation_rate=0.05, vertex_sigma=40.0, color_sigma=40.0)
        scheduler = SigmaDecayScheduler(
            mutation=op,
            n_generations=N_GENERATIONS,
            vertex_sigma_max=40.0,
            vertex_sigma_min=2.0,
            color_sigma_max=40.0,
            color_sigma_min=2.0,
        )
        return op, scheduler

    if name == "creep_decay":
        op = CreepMutation(mutation_rate=0.05, vertex_delta=30.0, color_delta=30.0)
        scheduler = DeltaDecayScheduler(
            mutation=op,
            n_generations=N_GENERATIONS,
            vertex_delta_max=30.0,
            vertex_delta_min=1.0,
            color_delta_max=30.0,
            color_delta_min=1.0,
        )
        return op, scheduler

    if name == "composite_gaussian":
        gaussian = GaussianMutation(mutation_rate=0.05, vertex_sigma=15.0, color_sigma=15.0)
        op = CompositeMutation([
            gaussian,
            ResetMutation(mutation_rate=0.01),
            SwapMutation(mutation_rate=0.2, n_swaps=1),
        ])
        return op, None

    if name == "composite_creep":
        creep = CreepMutation(mutation_rate=0.05, vertex_delta=15.0, color_delta=15.0)
        op = CompositeMutation([
            creep,
            ResetMutation(mutation_rate=0.01),
            SwapMutation(mutation_rate=0.2, n_swaps=1),
        ])
        return op, None

    raise ValueError(f"Unknown mutation: {name}")


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def execute_run(cfg: RunConfig, target: np.ndarray) -> dict:
    """
    Execute a single GA run described by cfg against the given target image.

    Returns a results dict suitable for appending to the global CSV.
    """
    logger.info("=== Starting run: %s (phase %d) ===", cfg.name, cfg.phase)
    directory = run_dir(cfg.name)
    save_config(cfg, directory)

    fitness_fn = RMSEFitness(target)
    selection = make_selection(cfg.selection, **cfg.extra)
    crossover = make_crossover(cfg.crossover)
    mutation_op, scheduler = make_mutation(cfg.mutation)

    # -- Early stopping: plain + diversity-aware running in parallel
    plain_es = EarlyStopping(patience=100, tolerance=1e-4)
    diversity_es = DiversityAwareEarlyStopping(
        patience=100,
        tolerance=1e-4,
        phenotypic_variance_threshold=0.01,
        genotypic_variance_threshold=1e-4,
        diversity_check_interval=10,
    )

    ga_config = GAConfig(
        population_size=cfg.population_size,
        n_generations=cfg.n_generations,
        crossover_rate=cfg.crossover_rate,
        n_elites=cfg.n_elites,
        n_workers=1,
        early_stopping=plain_es,
        diversity_early_stopping=diversity_es,
        checkpoint_interval=0,  # we handle checkpointing in the callback below
        seed=cfg.seed,
    )

    ga = GeneticAlgorithm(
        fitness_fn=fitness_fn,
        selection=selection,
        crossover=crossover,
        mutation=mutation_op,
        config=ga_config,
    )

    def _callback(generation: int, best, stats: dict) -> None:
        # scheduler update (sigma/delta decay) — no-op if scheduler is None
        if scheduler is not None:
            scheduler(generation, best, stats)

        # checkpoint image + full diversity metrics every N generations
        if generation % IMAGE_CHECKPOINT_INTERVAL == 0:
            rendered = render(best.triangles)
            img_path = os.path.join(directory, "checkpoints", f"gen_{generation:04d}.png")
            save_render(rendered, img_path)

            # ga.current_population is updated every generation by the engine,
            # so we can compute all four diversity metrics here without any
            # extra coupling or state tracking in the callback closure.
            pop = ga.current_population
            div = pop.diversity_report()

            metrics = {
                "generation": generation,
                "best_fitness": round(best.fitness, 6),
                "mean_fitness": round(stats.get("mean", float("nan")), 6),
                "std_fitness": round(stats.get("std", float("nan")), 6),
                "worst_fitness": round(stats.get("worst", float("nan")), 6),
                "phenotypic_entropy": round(div["phenotypic_entropy"], 6),
                "genotypic_entropy": round(div["genotypic_entropy"], 6),
                "phenotypic_variance": round(div["phenotypic_variance"], 6),
                "genotypic_variance": round(div["genotypic_variance"], 6),
            }
            metrics_path = os.path.join(
                directory, "checkpoints", f"gen_{generation:04d}_metrics.json"
            )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            logger.info(
                "Checkpoint gen %d | fitness=%.4f | pheno_var=%.6f | geno_var=%.6f",
                generation, best.fitness,
                div["phenotypic_variance"], div["genotypic_variance"],
            )

    # -- Run
    t0 = time.time()
    best = ga.run(
        target=target,
        init_strategy="random",
        callback=_callback,
    )
    elapsed = time.time() - t0

    # -- Save final outputs
    final_rendered = render(best.triangles)
    save_render(final_rendered, os.path.join(directory, "best_final.png"))
    triangles_to_json(list(best.triangles), os.path.join(directory, "best_final_triangles.json"))
    ga.save_log(os.path.join(directory, "generation_log.json"))
    ga.save_config(os.path.join(directory, "ga_config.json"))

    n_generations_run = len(ga.generation_log) - 1  # subtract gen 0

    results_row = {
        "run_name": cfg.name,
        "phase": cfg.phase,
        "description": cfg.description,
        "selection": cfg.selection,
        "crossover": cfg.crossover,
        "mutation": cfg.mutation,
        "n_elites": cfg.n_elites,
        "population_size": cfg.population_size,
        "n_generations_max": cfg.n_generations,
        "n_generations_run": n_generations_run,
        "seed": cfg.seed,
        "final_best_fitness": round(best.fitness, 6),
        "elapsed_s": round(elapsed, 1),
        "stopped_early": n_generations_run < cfg.n_generations,
    }

    append_global_results(results_row)
    logger.info(
        "Run %s complete | fitness=%.4f | %d gens | %.1fs",
        cfg.name, best.fitness, n_generations_run, elapsed,
    )
    return results_row


# ---------------------------------------------------------------------------
# Experiment plan
# ---------------------------------------------------------------------------

def build_experiment_plan() -> List[RunConfig]:
    """
    Define all runs across phases 1-5.

    Phase 1 - baseline
    Phase 2 - elitism sweep
    Phase 3 - selection sweep
    Phase 4 - crossover sweep
    Phase 5 - mutation sweep (challenge 3 material)
    """
    runs: List[RunConfig] = []

    # ------------------------------------------------------------------
    # Phase 1: Baseline
    # ------------------------------------------------------------------
    runs.append(RunConfig(
        name="p1_baseline",
        phase=1,
        description="Baseline - tournament k3, uniform xover, gaussian fixed, 1 elite",
        selection="tournament_k3",
        crossover="uniform",
        mutation="gaussian_fixed",
        n_elites=1,
    ))

    # ------------------------------------------------------------------
    # Phase 2: Elitism sweep
    # ------------------------------------------------------------------
    for n_elites, label in [(0, "0"), (1, "1"), (3, "3"), (5, "5")]:
        runs.append(RunConfig(
            name=f"p2_elites_{label}",
            phase=2,
            description=f"Elitism sweep - {n_elites} elites",
            selection="tournament_k3",
            crossover="uniform",
            mutation="gaussian_fixed",
            n_elites=n_elites,
        ))

    # ------------------------------------------------------------------
    # Phase 3: Selection sweep (elites fixed at 1 - update after phase 2)
    # ------------------------------------------------------------------
    for sel_name, label in [
        ("tournament_k2", "tournament_k2"),
        ("tournament_k5", "tournament_k5"),
        ("tournament_k10", "tournament_k10"),
        ("rank", "rank"),
    ]:
        runs.append(RunConfig(
            name=f"p3_sel_{label}",
            phase=3,
            description=f"Selection sweep - {label}",
            selection=sel_name,
            crossover="uniform",
            mutation="gaussian_fixed",
            n_elites=1,
        ))

    # ------------------------------------------------------------------
    # Phase 4: Crossover sweep
    # ------------------------------------------------------------------
    for xover_name, label in [
        ("uniform", "uniform"),
        ("single_point", "single_point"),
        ("two_point", "two_point"),
    ]:
        runs.append(RunConfig(
            name=f"p4_xover_{label}",
            phase=4,
            description=f"Crossover sweep - {label}",
            selection="tournament_k3",  # update to best from phase 3
            crossover=xover_name,
            mutation="gaussian_fixed",
            n_elites=1,
        ))

    # ------------------------------------------------------------------
    # Phase 5: Mutation sweep
    # ------------------------------------------------------------------
    for mut_name in [
        "gaussian_fixed",
        "creep_fixed",
        "gaussian_decay",
        "creep_decay",
        "composite_gaussian",
        "composite_creep",
    ]:
        runs.append(RunConfig(
            name=f"p5_mut_{mut_name}",
            phase=5,
            description=f"Mutation sweep - {mut_name}",
            selection="tournament_k3",  # update to best from phase 3
            crossover="uniform",        # update to best from phase 4
            mutation=mut_name,
            n_elites=1,
        ))

    return runs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GA experiment runner for triangle image approximation."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/girl_pearl.png",
        help="Path to the target image file (default: data/girl_pearl.png).",
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=None,
        help="Phases to run (e.g. --phases 1 2). Defaults to all phases.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run a single named run (e.g. --run p1_baseline). Overrides --phases.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available run names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_experiment_plan()

    if args.list:
        print(f"{'Name':<30} {'Phase':<8} Description")
        print("-" * 80)
        for cfg in plan:
            print(f"{cfg.name:<30} {cfg.phase:<8} {cfg.description}")
        return

    target = load_target(args.target)
    logger.info("Target image loaded from %s, shape=%s", args.target, target.shape)

    # filter runs
    if args.run is not None:
        matching = [c for c in plan if c.name == args.run]
        if not matching:
            raise ValueError(
                f"No run named '{args.run}'. Use --list to see available runs."
            )
        selected = matching
    elif args.phases is not None:
        selected = [c for c in plan if c.phase in args.phases]
        if not selected:
            raise ValueError(f"No runs found for phases {args.phases}.")
    else:
        selected = plan

    logger.info("Running %d experiment(s).", len(selected))

    all_results = []
    for cfg in selected:
        try:
            result = execute_run(cfg, target)
            all_results.append(result)
        except Exception as e:
            logger.error("Run %s failed with error: %s", cfg.name, e, exc_info=True)
            # continue to next run rather than aborting the whole session
            continue

    logger.info("All runs complete. Results in %s", RESULTS_CSV)

    # print summary table to stdout
    if all_results:
        print("\n--- Summary ---")
        print(f"{'Run':<30} {'Fitness':>10} {'Gens':>6} {'Time(s)':>8}")
        print("-" * 60)
        for r in all_results:
            print(
                f"{r['run_name']:<30} "
                f"{r['final_best_fitness']:>10.4f} "
                f"{r['n_generations_run']:>6} "
                f"{r['elapsed_s']:>8.1f}"
            )


if __name__ == "__main__":
    main()