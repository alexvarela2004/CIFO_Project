"""
runner.py
---------
Experiment runner for the GA image approximation project.

Executes the full phase-based experiment plan (phases 1-5) defined in the
project tracker. Each configuration is run with multiple seeds to support
statistical comparison (Mann-Whitney U test, mean/std reporting).

Output structure
----------------
runner_outputs/
    results.csv                         <- global results, one row per seed run
    <config_name>/
        config_results.csv              <- results for this config across all seeds
        seed_42/
            config.json
            generation_log.json
            best_final.png
            best_final_triangles.json
            ga_config.json
            checkpoints/
                gen_0000.png
                gen_0000_metrics.json
                gen_0100.png
                gen_0100_metrics.json
                ...
        seed_43/
            ...
        seed_44/
            ...

Usage
-----
    python runner.py --target data/girl_pearl.png
    python runner.py --target data/girl_pearl.png --phases 1 2
    python runner.py --target data/girl_pearl.png --run p1_baseline
    python runner.py --target data/girl_pearl.png --run p1_baseline --seeds 42 43 44
    python runner.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from fitness import RMSEFitness, CIEDEFitness, SSIMFitness
from ga import GeneticAlgorithm, GAConfig, EarlyStopping, DiversityAwareEarlyStopping
from ga_operators.selection import TournamentSelection, RankSelection
from ga_operators.crossover import (
    SinglePointCrossover,
    UniformCrossover,
    KPointCrossover,
    SegmentShuffleCrossover,
    BlendCrossover,
)
from ga_operators.mutation import (
    GaussianMutation,
    CreepMutation,
    ResetMutation,
    SwapMutation,
    CompositeMutation,
    SigmaDecayScheduler,
    DeltaDecayScheduler,
    CompositeSigmaDecayScheduler
)
from ga_utils import load_target, save_render, render, triangles_to_json

# ---------------------------------------------------------------------------
# Logging
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

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runner_outputs")
RESULTS_CSV = os.path.join(OUTPUT_ROOT, "results.csv")

N_GENERATIONS = 3000
POPULATION_SIZE = 50
CROSSOVER_RATE = 0.8
IMAGE_CHECKPOINT_INTERVAL = 100
SEEDS = [42, 43, 44]  # default seeds used for every configuration

# ---------------------------------------------------------------------------
# Run config descriptor
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """
    Describes one experiment configuration (operator choices + hyperparams).
    Each config is run once per seed, producing one subfolder per seed under
    the config folder.
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
    init_strategy: str = "random"
    image_ratio: float = 0.5
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def config_dir(config_name: str) -> str:
    """Return and create the top-level directory for a named config."""
    path = os.path.join(OUTPUT_ROOT, config_name)
    os.makedirs(path, exist_ok=True)
    return path


def seed_dir(config_name: str, seed: int) -> str:
    """Return and create the seed subfolder inside a config folder."""
    path = os.path.join(OUTPUT_ROOT, config_name, f"seed_{seed}")
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _append_csv(filepath: str, row: dict) -> None:
    """Append a row to a CSV, writing headers on first write."""
    file_exists = os.path.isfile(filepath)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_global_results(row: dict) -> None:
    """Append one result row to the global results CSV."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    _append_csv(RESULTS_CSV, row)
    logger.info("Global results updated -> %s", RESULTS_CSV)


def append_config_results(config_name: str, row: dict) -> None:
    """Append one result row to the per-config results CSV."""
    path = os.path.join(OUTPUT_ROOT, config_name, "config_results.csv")
    _append_csv(path, row)
    logger.info("Config results updated -> %s", path)


# ---------------------------------------------------------------------------
# Operator factories
# ---------------------------------------------------------------------------

def make_selection(name: str, **kwargs):
    """Instantiate and return a selection operator by name."""
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
    """Instantiate and return a crossover operator by name."""
    if name == "uniform":
        return UniformCrossover(swap_prob=0.5)
    if name == "single_point":
        return SinglePointCrossover()
    if name == "two_point":
        return KPointCrossover(k=2)
    if name == "segment_shuffle":
        return SegmentShuffleCrossover(k_min=1, k_max=5)
    if name == "blend":
        return BlendCrossover(alpha=0.5)
    raise ValueError(f"Unknown crossover: {name}")


def make_mutation(name: str, n_generations: int, extra: dict = None):
    """
    Returns (mutation_operator, scheduler_or_None).
    n_generations is passed so decay schedulers are calibrated to the run length.
    decay_n_generations in extra overrides the scheduler window independently of
    n_generations - useful when the run budget is large but you want the sigma/delta
    to decay faster (e.g. a 20k-gen run decaying as if it were a 5k-gen run).
    """
    extra = extra or {}
    mutation_rate = extra.get("mutation_rate", 0.05)
    decay_n_generations = extra.get("decay_n_generations", n_generations)

    if name == "gaussian_fixed":
        op = GaussianMutation(
            mutation_rate=mutation_rate,
            vertex_sigma=extra.get("vertex_sigma_max", 15.0),
            color_sigma=extra.get("color_sigma_max",   15.0),
        )
        return op, None

    if name == "creep_fixed":
        op = CreepMutation(
            mutation_rate=mutation_rate,
            vertex_delta=extra.get("vertex_delta_max", 15.0),
            color_delta=extra.get("color_delta_max",   15.0),
        )
        return op, None

    if name == "gaussian_decay":
        vertex_sigma_max = extra.get("vertex_sigma_max", 40.0)
        vertex_sigma_min = extra.get("vertex_sigma_min", 2.0)
        color_sigma_max  = extra.get("color_sigma_max",  40.0)
        color_sigma_min  = extra.get("color_sigma_min",  2.0)
        op = GaussianMutation(
            mutation_rate=mutation_rate,
            vertex_sigma=vertex_sigma_max,
            color_sigma=color_sigma_max,
        )
        scheduler = SigmaDecayScheduler(
            mutation=op,
            n_generations=decay_n_generations,
            vertex_sigma_max=vertex_sigma_max,
            vertex_sigma_min=vertex_sigma_min,
            color_sigma_max=color_sigma_max,
            color_sigma_min=color_sigma_min,
        )
        return op, scheduler

    if name == "creep_decay":
        vertex_delta_max = extra.get("vertex_delta_max", 30.0)
        vertex_delta_min = extra.get("vertex_delta_min", 1.0)
        color_delta_max  = extra.get("color_delta_max",  30.0)
        color_delta_min  = extra.get("color_delta_min",  1.0)
        op = CreepMutation(
            mutation_rate=mutation_rate,
            vertex_delta=vertex_delta_max,
            color_delta=color_delta_max,
        )
        scheduler = DeltaDecayScheduler(
            mutation=op,
            n_generations=decay_n_generations,
            vertex_delta_max=vertex_delta_max,
            vertex_delta_min=vertex_delta_min,
            color_delta_max=color_delta_max,
            color_delta_min=color_delta_min,
        )
        return op, scheduler

    if name == "composite_gaussian":
        gaussian = GaussianMutation(
            mutation_rate=mutation_rate,
            vertex_sigma=extra.get("vertex_sigma_max", 15.0),
            color_sigma=extra.get("color_sigma_max",   15.0),
        )
        op = CompositeMutation([
            gaussian,
            ResetMutation(mutation_rate=0.01),
            SwapMutation(mutation_rate=0.2, n_swaps=1),
        ])
        return op, None

    if name == "composite_creep":
        creep = CreepMutation(
            mutation_rate=mutation_rate,
            vertex_delta=extra.get("vertex_delta_max", 15.0),
            color_delta=extra.get("color_delta_max",   15.0),
        )
        op = CompositeMutation([
            creep,
            ResetMutation(mutation_rate=0.01),
            SwapMutation(mutation_rate=0.2, n_swaps=1),
        ])
        return op, None
    
    if name == "composite_gaussian_decay":
        gaussian = GaussianMutation(mutation_rate=0.05, vertex_sigma=40.0, color_sigma=40.0)
        composite = CompositeMutation([
            gaussian,
            ResetMutation(mutation_rate=0.01),
            SwapMutation(mutation_rate=0.2, n_swaps=1),
        ])
        scheduler = CompositeSigmaDecayScheduler(
            gaussian_op=gaussian,
            n_generations=n_generations,
            vertex_sigma_max=40.0,
            vertex_sigma_min=2.0,
            color_sigma_max=40.0,
            color_sigma_min=2.0,
        )
        return composite, scheduler
        

    raise ValueError(f"Unknown mutation: {name}")

# ---------------------------------------------------------------------------
# Core single-seed run
# ---------------------------------------------------------------------------
def make_fitness(extra: dict, target: np.ndarray):
    """
    Instantiate the fitness function from the extra dict.
    Defaults to RMSEFitness if no 'fitness' key is present —
    preserves behaviour for all phases 1-11.
    """
    name = extra.get("fitness", "rmse")

    if name == "rmse":
        return RMSEFitness(target)
    if name == "ciede2000":
        return CIEDEFitness(target)
    if name == "ssim":
        return SSIMFitness(target)
    raise ValueError(f"Unknown fitness function: {name}")


def execute_single_run(cfg: RunConfig, seed: int, target: np.ndarray) -> dict:
    """
    Execute one GA run for the given config and seed.

    Outputs go to runner_outputs/<config_name>/seed_<seed>/.
    Returns a results dict appended to both the global and per-config CSVs.
    """
    directory = seed_dir(cfg.name, seed)
    logger.info("--- Starting %s | seed=%d ---", cfg.name, seed)

    # save config json into seed folder with seed field included
    config_path = os.path.join(directory, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        d = cfg.to_dict()
        d["seed"] = seed
        json.dump(d, f, indent=2)

    fitness_fn = make_fitness(cfg.extra, target)
    selection = make_selection(cfg.selection, **cfg.extra)
    crossover = make_crossover(cfg.crossover)
    mutation_op, scheduler = make_mutation(cfg.mutation, cfg.n_generations, extra=cfg.extra)

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

    def _callback(generation: int, best, stats: dict) -> None:
        if scheduler is not None:
            scheduler(generation, best, stats)

        if generation % IMAGE_CHECKPOINT_INTERVAL == 0:
            rendered = render(best.triangles)
            img_path = os.path.join(directory, "checkpoints", f"gen_{generation:04d}.png")
            save_render(rendered, img_path)

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
                "Checkpoint gen %d | seed=%d | fitness=%.4f | pheno_var=%.6f | geno_var=%.6f",
                generation, seed, best.fitness,
                div["phenotypic_variance"], div["genotypic_variance"],
            )

    t0 = time.time()
    best = ga.run(target=target, init_strategy=cfg.init_strategy, image_ratio=cfg.image_ratio, callback=_callback)
    elapsed = time.time() - t0

    save_render(render(best.triangles), os.path.join(directory, "best_final.png"))
    triangles_to_json(list(best.triangles), os.path.join(directory, "best_final_triangles.json"))
    ga.save_log(os.path.join(directory, "generation_log.json"))
    ga.save_config(os.path.join(directory, "ga_config.json"))

    n_generations_run = len(ga.generation_log) - 1

    row = {
        "run_name": cfg.name,
        "seed": seed,
        "phase": cfg.phase,
        "description": cfg.description,
        "selection": cfg.selection,
        "crossover": cfg.crossover,
        "mutation": cfg.mutation,
        "n_elites": cfg.n_elites,
        "population_size": cfg.population_size,
        "n_generations_max": cfg.n_generations,
        "n_generations_run": n_generations_run,
        "final_best_fitness": round(best.fitness, 6),
        "elapsed_s": round(elapsed, 1),
        "stopped_early": n_generations_run < cfg.n_generations,
    }

    append_global_results(row)
    append_config_results(cfg.name, row)

    logger.info(
        "Done %s | seed=%d | fitness=%.4f | %d gens | %.1fs",
        cfg.name, seed, best.fitness, n_generations_run, elapsed,
    )
    return row


# ---------------------------------------------------------------------------
# Multi-seed executor
# ---------------------------------------------------------------------------

def execute_run(cfg: RunConfig, seeds: List[int], target: np.ndarray) -> List[dict]:
    """
    Execute a config across all seeds sequentially.
    Errors on individual seeds are caught so remaining seeds still run.
    Returns list of per-seed result dicts.
    """
    logger.info("=== Config: %s | %d seeds ===", cfg.name, len(seeds))
    config_dir(cfg.name)  # ensure top-level config folder exists

    results = []
    for seed in seeds:
        try:
            row = execute_single_run(cfg, seed, target)
            results.append(row)
        except Exception as e:
            logger.error(
                "Config %s seed %d failed: %s", cfg.name, seed, e, exc_info=True
            )
            continue

    if results:
        fitnesses = [r["final_best_fitness"] for r in results]
        logger.info(
            "Config %s | mean=%.4f | std=%.4f | best=%.4f | seeds=%s",
            cfg.name,
            float(np.mean(fitnesses)),
            float(np.std(fitnesses)),
            float(np.min(fitnesses)),
            seeds,
        )

    return results


# ---------------------------------------------------------------------------
# Experiment plan
# ---------------------------------------------------------------------------

def build_experiment_plan() -> List[RunConfig]:
    """
    Define all configurations across phases.
    Each config is run once per seed in execute_run().
    """
    runs: List[RunConfig] = []

    # ------------------------------------------------------------------
    # Phase 1: Baseline
    # ------------------------------------------------------------------
    runs.append(RunConfig(
        name="p1_baseline",
        phase=1,
        description="Baseline - tournament k3, uniform xover, gaussian fixed, 5 elites",
        selection="tournament_k3",
        crossover="uniform",
        mutation="gaussian_fixed",
        n_elites=5,
    ))

    # ------------------------------------------------------------------
    # Phase 2: Elitism sweep
    # ------------------------------------------------------------------
    for n_elites, label in [(0, "0"), (1, "1"), (3, "3"), (5, "5"), (7, "7")]:
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
    # Phase 3: Selection sweep
    # ------------------------------------------------------------------
    for sel_name, label in [
        ("tournament_k2",  "tournament_k2"),
        ("tournament_k5",  "tournament_k5"),
        ("tournament_k10", "tournament_k10"),
        ("rank",           "rank"),
    ]:
        runs.append(RunConfig(
            name=f"p3_sel_{label}",
            phase=3,
            description=f"Selection sweep - {label}",
            selection=sel_name,
            crossover="uniform",
            mutation="gaussian_fixed",
            n_elites=5,
        ))

    # ------------------------------------------------------------------
    # Phase 4: Crossover sweep
    # ------------------------------------------------------------------
    for xover_name, label in [
        ("uniform",          "uniform"),
        ("single_point",     "single_point"),
        ("two_point",        "two_point"),
        ("segment_shuffle",  "segment_shuffle"),
        ("blend",            "blend"),
    ]:
        runs.append(RunConfig(
            name=f"p4_xover_{label}",
            phase=4,
            description=f"Crossover sweep - {label}",
            selection="tournament_k10",  # best from phase 3
            crossover=xover_name,
            mutation="gaussian_fixed",
            n_elites=5,
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
        "composite_gaussian_decay"
    ]:
        runs.append(RunConfig(
            name=f"p5_mut_{mut_name}",
            phase=5,
            description=f"Mutation sweep - {mut_name}",
            selection="tournament_k10",  # best from phase 3
            crossover="two_point",       # best from phase 4
            mutation=mut_name,
            n_elites=5,
        ))
    
    # ------------------------------------------------------------------
    # Phase 6: Best config with more generations
    # ------------------------------------------------------------------
    runs.append(RunConfig(
        name="p6_gaussian_decay_5k",
        phase=6,
        description="Best config with 20000 generations",
        selection="tournament_k10",  # best from phase 3
        crossover="two_point",       # best from phase 4
        mutation="gaussian_decay",   # best from phase 5
        n_elites=5,
        n_generations=20000,
    ))

    # ------------------------------------------------------------------
    # Phase 7: Init strategy sweep
    # ------------------------------------------------------------------
    for strategy in [
        "random", "image", "mixed", "grid",
        "semi_transparent", "small_random", "grid_random_color",
        "sorted_alpha", "quadrant",
    ]:
        runs.append(RunConfig(
            name=f"p7_init_{strategy}",
            phase=7,
            description=f"Init strategy sweep - {strategy}",
            selection="tournament_k10",
            crossover="two_point",
            mutation="gaussian_decay",
            n_elites=5,
            n_generations=3000,
            init_strategy=strategy,
        ))

    # ------------------------------------------------------------------
    # Phase 8: Elitism re-test with best operators
    # ------------------------------------------------------------------
    for n_elites in [1, 3, 7, 10]:  # 5 was already tested in phase 5
        runs.append(RunConfig(
            name=f"p8_elites_{n_elites}",
            phase=8,
            description=f"Elitism retest (best operators) - {n_elites} elites",
            selection="tournament_k10",
            crossover="blend",
            mutation="gaussian_decay",
            n_elites=n_elites,
            n_generations=3000,
        ))

    # ---------------------------------------------------------------------------
    # Phase 9: Interaction check — tournament_k5 with best mutations
    # ---------------------------------------------------------------------------
    # We do not test the best selection with the top 2 mutations as this was already done in phase 5
    # We do not test crossover because the spread is so small that any difference between crossovers
    # is likely within the random variation between seeds - not statistically significant.

    for mut in ["gaussian_decay", "creep_decay"]:
        runs.append(RunConfig(
            name=f"p9_k5_{mut}",
            phase=9,
            description=f"Interaction check - tournament_k5 + blend + {mut}",
            selection="tournament_k5",
            crossover="blend",
            mutation=mut,
            n_elites=5,  
            n_generations=3000,
        ))
    

    # ------------------------------------------------------------------
    # Phase 10: Grid search — mutation_rate & vertex_sigma_max
    # ------------------------------------------------------------------
    # Tests the interaction between perturbation frequency and magnitude

    for rate in [0.01, 0.05, 0.10, 0.20]:
        for sigma_max in [15.0, 40.0, 80.0]:

            if rate == 0.05 and sigma_max == 40.0:
                continue

            runs.append(RunConfig(
                name=f"p10_rate_{str(rate).replace('.','')}_sigmax_{int(sigma_max)}",
                phase=10,
                description=f"Grid search - mutation_rate={rate}, vertex_sigma_max={sigma_max}",
                selection="tournament_k10", 
                crossover="blend",
                mutation="gaussian_decay", 
                n_elites=7, # best of phase 8
                n_generations=3000,
                extra={
                    "mutation_rate": rate,
                    "vertex_sigma_max": sigma_max,
                    "color_sigma_max": sigma_max,
                },
            ))


    # ------------------------------------------------------------------
    # Phase 11: Testing population size
    # ------------------------------------------------------------------

    BEST_P10 = dict(
        selection="tournament_k10",
        crossover="blend",
        mutation="gaussian_decay", 
        n_elites=3, 
        n_generations=3000,
        init_strategy = "quadrant"
    )

    BEST_P10_EXTRA = {
    "mutation_rate": 0.01, 
    "vertex_sigma_max": 80.0, 
    "color_sigma_max": 80.0, 
    }

    for pop in [100, 150]:
        runs.append(RunConfig(
            name=f"p11_pop_{pop}",
            phase=11,
            description=f"Population_size={pop}",
            population_size=pop,
            extra=BEST_P10_EXTRA,
            **BEST_P10,
        ))
    

    # ------------------------------------------------------------------
    # Phase 12: Final run - best config with 20000 generations
    # ------------------------------------------------------------------
    runs.append(RunConfig(
        name="p12_final",
        phase=12,
        description="Final run - best config with 20000 generations",
        selection="tournament_k10",
        crossover="blend",
        mutation="gaussian_decay",
        n_elites=3,
        population_size=150,
        init_strategy="quadrant",
        n_generations=20000,
        extra={
            "mutation_rate": 0.01,
            "vertex_sigma_max": 80.0,
            "color_sigma_max": 80.0,
            "decay_n_generations": 5000,
        },
    ))

    runs.append(RunConfig(
        name="p12_final_300",
        phase=12,
        description="Final run - best config with 20000 generations, 300 population",
        selection="tournament_k10",
        crossover="blend",
        mutation="gaussian_decay",
        n_elites=3,
        population_size=300,
        init_strategy="quadrant",
        n_generations=20000,
        extra={
            "mutation_rate": 0.01,
            "vertex_sigma_max": 80.0,
            "color_sigma_max": 80.0,
            "decay_n_generations": 5000,
        },
    ))


    # ------------------------------------------------------------------
    # Phase 13: Challenge 1 — alternative fitness functions
    # ------------------------------------------------------------------

    BEST_P12 = dict(
        selection="tournament_k10",
        crossover="blend",
        mutation="gaussian_decay",
        n_elites=3,
        population_size=150,
        n_generations=3000,
        init_strategy="quadrant",
    )

    BEST_P12_EXTRA = {
        "mutation_rate":    0.01,
        "vertex_sigma_max": 80.0,
        "color_sigma_max":  80.0,
    }

    for fitness_name in ["ciede2000", "ssim"]:
        runs.append(RunConfig(
            name=f"p13_fitness_{fitness_name}",
            phase=13,
            description=f"Challenge 1 - {fitness_name} fitness function",
            extra={**BEST_P12_EXTRA, "fitness": fitness_name},
            **BEST_P12,
        ))

    # ------------------------------------------------------------------
    # Phase 14: Generalisation test on unseen images
    # ------------------------------------------------------------------
    runs.append(RunConfig(
        name="p14_mona_lisa",
        phase=14,
        description="Generalisation test - Mona Lisa",
        selection="tournament_k10",
        crossover="blend",
        mutation="gaussian_decay",
        n_elites=3,
        population_size=150,
        n_generations=3000,
        init_strategy= "quadrant",
        extra={
            "mutation_rate": 0.01,
            "vertex_sigma_max": 80.0,
            "color_sigma_max": 80.0,
        },
    ))

    runs.append(RunConfig(
        name="p14_nadir",
        phase=14,
        description="Generalisation test - Nadir Afonso Painting",
        selection="tournament_k10",
        crossover="blend",
        mutation="gaussian_decay",
        n_elites=3,
        population_size=150,
        n_generations=3000,
        init_strategy= "quadrant",
        extra={
            "mutation_rate": 0.01,
            "vertex_sigma_max": 80.0,
            "color_sigma_max": 80.0,
        },
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
        help="Path to the target image (default: data/girl_pearl.png).",
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=None,
        help="Phases to run (e.g. --phases 1 2). Runs all phases if omitted.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run a single named config (e.g. --run p1_baseline). Overrides --phases.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"Seeds to use (e.g. --seeds 42 43 44). Defaults to {SEEDS}.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available config names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_experiment_plan()

    if args.list:
        print(f"{'Name':<35} {'Phase':<8} Description")
        print("-" * 85)
        for cfg in plan:
            print(f"{cfg.name:<35} {cfg.phase:<8} {cfg.description}")
        return

    seeds = args.seeds if args.seeds is not None else SEEDS
    target = load_target(args.target)
    logger.info(
        "Target loaded from %s | shape=%s | seeds=%s",
        args.target, target.shape, seeds,
    )
    # ---------------------------------------------------------------------------
    # Filter configs
    # ---------------------------------------------------------------------------
    if args.run is not None:
        matching = [c for c in plan if c.name == args.run]
        if not matching:
            raise ValueError(
                f"No config named '{args.run}'. Use --list to see available configs."
            )
        selected = matching
    elif args.phases is not None:
        selected = [c for c in plan if c.phase in args.phases]
        if not selected:
            raise ValueError(f"No configs found for phases {args.phases}.")
    else:
        selected = plan

    logger.info(
        "Running %d config(s) x %d seed(s) = %d total runs.",
        len(selected), len(seeds), len(selected) * len(seeds),
    )

    all_results = []
    for cfg in selected:
        try:
            results = execute_run(cfg, seeds, target)
            all_results.extend(results)
        except Exception as e:
            logger.error("Config %s failed: %s", cfg.name, e, exc_info=True)
            continue

    logger.info("All runs complete. Global results in %s", RESULTS_CSV)

    # ---------------------------------------------------------------------------
    # Summary grouped by config
    # ---------------------------------------------------------------------------

    if all_results:
        print("\n--- Summary ---")
        print(f"{'Config':<35} {'Seeds':>5} {'Mean':>10} {'Std':>8} {'Best':>10}")
        print("-" * 72)

        grouped: Dict[str, List[float]] = defaultdict(list)
        for r in all_results:
            grouped[r["run_name"]].append(r["final_best_fitness"])

        for config_name, fitnesses in grouped.items():
            print(
                f"{config_name:<35} "
                f"{len(fitnesses):>5} "
                f"{float(np.mean(fitnesses)):>10.4f} "
                f"{float(np.std(fitnesses)):>8.4f} "
                f"{float(np.min(fitnesses)):>10.4f}"
            )


if __name__ == "__main__":
    main()