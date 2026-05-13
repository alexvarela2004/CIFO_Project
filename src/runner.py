"""
runner.py
---------
Single entry point for running GA experiments and saving results.

This module handles:
    - Building GA instances from config dicts
    - Running experiments and collecting all metrics
    - Saving results to pkl (one per run)
    - Saving snapshots (generation 0, middle, final) as PNG
    - Maintaining a master_index.csv across all runs
    - Cross-evaluating the best individual with all fitness functions

Usage (from terminal)
---------------------
    cd src
    python runner.py                          # run all experiment groups
    python runner.py --group phase1_mutation  # run one group only
    python runner.py --dry-run                # print configs without running

Usage (from Python / notebook)
-------------------------------
    from runner import run_single_experiment, build_experiment_configs
    configs = build_experiment_configs()
    result  = run_single_experiment(configs[0], target_array)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fitness import FitnessFunction, build_fitness, RMSEFitness, SSIMFitness, CIEDEFitness
from ga import GeneticAlgorithm, GAConfig, EarlyStopping
from ga_operators.selection import TournamentSelection, RankSelection, RouletteSelection
from ga_operators.crossover import SinglePointCrossover, UniformCrossover, KPointCrossover
from ga_operators.mutation import (
    GaussianMutation, ResetMutation, SwapMutation,
    CompositeMutation, SigmaDecayScheduler,
)
from individual import Individual
from population import Population
from triangle import Triangle
from utils import render, save_render, load_target_pil, IMG_WIDTH, IMG_HEIGHT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_PATH = os.path.join(PROJECT_ROOT, "data", "Girl_Pearl_Earing.png")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output_data")
MASTER_INDEX_PATH = os.path.join(OUTPUT_ROOT, "master_index.csv")


# ---------------------------------------------------------------------------
# Operator builders -- config dict -> operator instance
# ---------------------------------------------------------------------------

def build_selection(cfg: dict):
    """Build a SelectionOperator from a config dict."""
    name = cfg.get("selection_operator", "TournamentSelection")
    if name == "TournamentSelection":
        return TournamentSelection(tournament_size=cfg.get("tournament_size", 5))
    elif name == "RankSelection":
        return RankSelection(selection_pressure=cfg.get("selection_pressure", 1.5))
    elif name == "RouletteSelection":
        return RouletteSelection()
    else:
        raise ValueError(f"Unknown selection operator: {name}")


def build_crossover(cfg: dict):
    """Build a CrossoverOperator from a config dict."""
    name = cfg.get("crossover_operator", "SinglePointCrossover")
    if name == "SinglePointCrossover":
        return SinglePointCrossover()
    elif name == "UniformCrossover":
        return UniformCrossover(swap_prob=cfg.get("swap_prob", 0.5))
    elif name == "KPointCrossover":
        return KPointCrossover(k=cfg.get("k", 2))
    else:
        raise ValueError(f"Unknown crossover operator: {name}")


def build_mutation(cfg: dict) -> Tuple:
    """
    Build a MutationOperator from a config dict.

    Returns (mutation_operator, gaussian_ref_or_None).
    gaussian_ref is the GaussianMutation instance inside a CompositeMutation,
    needed by SigmaDecayScheduler to update sigma in place.
    """
    name = cfg.get("mutation_operator", "CompositeMutation")

    gaussian = GaussianMutation(
        mutation_rate=cfg.get("mutation_rate", 0.05),
        vertex_sigma=cfg.get("vertex_sigma", 15.0),
        color_sigma=cfg.get("color_sigma", 15.0),
    )

    if name == "GaussianMutation":
        return gaussian, gaussian

    elif name == "CompositeMutation":
        components = [gaussian]
        if cfg.get("reset_rate", 0.0) > 0:
            components.append(ResetMutation(mutation_rate=cfg["reset_rate"]))
        if cfg.get("swap_rate", 0.0) > 0:
            components.append(
                SwapMutation(
                    mutation_rate=cfg["swap_rate"],
                    n_swaps=cfg.get("swap_n", 1),
                )
            )
        return CompositeMutation(components), gaussian

    else:
        raise ValueError(f"Unknown mutation operator: {name}")


# ---------------------------------------------------------------------------
# Snapshot helper -- saves PNGs at key generations
# ---------------------------------------------------------------------------

def save_snapshots(
    best: Individual,
    run_dir: str,
    generation: int,
    target_array: np.ndarray,
) -> str:
    """Save a rendered PNG of the best individual. Returns the file path."""
    os.makedirs(run_dir, exist_ok=True)
    arr = render(best.triangles)
    path = os.path.join(run_dir, f"gen_{generation:05d}.png")
    save_render(arr, path)
    return path


# ---------------------------------------------------------------------------
# Cross-evaluation -- evaluate best with all three fitness functions
# ---------------------------------------------------------------------------

def cross_evaluate(best: Individual, target_pil: Image.Image) -> dict:
    """
    Evaluate the best individual with all three fitness functions.

    Returns a dict with keys: best_rmse, best_ssim_fitness, best_ciede2000.
    This allows comparing runs that used different primary fitness metrics
    on the same scale.
    """
    arr = render(best.triangles)
    target_array = np.array(target_pil.convert("RGB"), dtype=np.uint8)

    rmse_fn = build_fitness("rmse", target_pil)
    ssim_fn = build_fitness("ssim", target_pil)
    ciede_fn = build_fitness("ciede2000", target_pil)

    return {
        "best_rmse":         round(rmse_fn.evaluate(arr), 6),
        "best_ssim_fitness": round(ssim_fn.evaluate(arr), 6),
        "best_ciede2000":    round(ciede_fn.evaluate(arr), 6),
    }


# ---------------------------------------------------------------------------
# Compute run-level summary metrics
# ---------------------------------------------------------------------------

def compute_summary(
    generation_log: List[dict],
    config: dict,
    cross_eval: dict,
    elapsed_total: float,
    run_dir: str,
) -> dict:
    """
    Compute all run-level summary metrics from the generation log.

    This dict becomes one row in master_index.csv.
    """
    log_df = pd.DataFrame(generation_log)

    best_gen_idx = int(log_df["best"].idxmin())
    initial_best = float(log_df["best"].iloc[0])
    min_best = float(log_df["best"].min())
    n_gens_run = len(log_df) - 1  # generation 0 is init

    summary = {}

    # -- config metadata (all fields from the experiment config)
    summary.update(config)

    # -- per-run metrics
    summary["initial_best"] = round(initial_best, 6)
    summary["final_best"] = round(float(log_df["best"].iloc[-1]), 6)
    summary["min_best"] = round(min_best, 6)
    summary["final_mean"] = round(float(log_df["mean"].iloc[-1]), 6)
    summary["final_std"] = round(float(log_df["std"].iloc[-1]), 6)
    summary["final_worst"] = round(float(log_df["worst"].iloc[-1]), 6)
    summary["final_diversity"] = round(float(log_df["diversity"].iloc[-1]), 6)
    summary["best_generation"] = int(log_df.loc[best_gen_idx, "generation"])
    summary["total_elapsed_s"] = round(elapsed_total, 2)
    summary["absolute_improvement"] = round(initial_best - min_best, 6)
    summary["relative_improvement"] = round(
        (initial_best - min_best) / initial_best if initial_best > 0 else 0.0, 6
    )
    summary["converged_early"] = n_gens_run < config.get("n_generations", 0)
    summary["n_generations_run"] = n_gens_run

    # -- cross-evaluation metrics
    summary.update(cross_eval)

    # -- path to results
    summary["output_dir"] = run_dir

    # -- timestamp
    summary["timestamp"] = datetime.now().isoformat(timespec="seconds")

    return summary


# ---------------------------------------------------------------------------
# Core: run a single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: dict,
    target_array: np.ndarray,
    target_pil: Image.Image,
    output_root: str = OUTPUT_ROOT,
) -> dict:
    """
    Run a single GA experiment from a config dict.

    Saves:
        - result.pkl       (generation_log + best triangles + config + summary)
        - gen_00000.png    (initial best)
        - gen_XXXXX.png    (mid-run snapshot)
        - gen_final.png    (final best)

    Parameters
    ----------
    config : dict
        Full experiment configuration. Must contain at least
        'experiment_group' and 'experiment_name'.
    target_array : np.ndarray
        H x W x 3 uint8 target image.
    target_pil : PIL.Image.Image
        Target image as PIL (for fitness construction).
    output_root : str
        Root output directory.

    Returns
    -------
    dict
        Summary metrics dict (= one row in master_index.csv).
    """
    group = config.get("experiment_group", "default")
    name = config.get("experiment_name", "run")
    run_dir = os.path.join(output_root, group, name)
    os.makedirs(run_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s / %s", group, name)
    logger.info("=" * 60)

    # -- Build fitness function
    fitness_metric = config.get("fitness_metric", "rmse")
    fitness_fn = build_fitness(fitness_metric, target_pil)

    # -- Build operators
    selection = build_selection(config)
    crossover = build_crossover(config)
    mutation, gaussian_ref = build_mutation(config)

    # -- Build GA config
    ga_config = GAConfig(
        population_size=config.get("population_size", 50),
        n_generations=config.get("n_generations", 300),
        crossover_rate=config.get("crossover_rate", 0.8),
        n_elites=config.get("n_elites", 1),
        n_workers=config.get("n_workers", 1),
        early_stopping=EarlyStopping(
            patience=config.get("early_stopping_patience", 50),
            tolerance=config.get("early_stopping_tolerance", 1e-4),
        ),
        checkpoint_interval=0,  # we handle snapshots ourselves
        seed=config.get("seed", 42),
    )

    ga = GeneticAlgorithm(
        fitness_fn=fitness_fn,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        config=ga_config,
    )

    # -- Build sigma decay scheduler if enabled
    callback = None
    sigma_log = None
    if config.get("sigma_decay", False):
        scheduler = SigmaDecayScheduler(
            mutation=gaussian_ref,
            n_generations=ga_config.n_generations,
            vertex_sigma_max=config.get("sigma_decay_v_max", 40.0),
            vertex_sigma_min=config.get("sigma_decay_v_min", 2.0),
            color_sigma_max=config.get("sigma_decay_c_max", 40.0),
            color_sigma_min=config.get("sigma_decay_c_min", 2.0),
        )
        callback = scheduler

    # -- Snapshot tracking: save images at gen 0, mid-point and final
    n_gens = ga_config.n_generations
    snapshot_gens = {0, n_gens // 2, n_gens}
    snapshot_paths = {}

    def combined_callback(gen, best, stats):
        if gen in snapshot_gens:
            p = save_snapshots(best, run_dir, gen, target_array)
            snapshot_paths[gen] = p
        if callback:
            callback(gen, best, stats)

    # -- Run
    start = time.time()
    init_strategy = config.get("init_strategy", "image")
    image_ratio = config.get("image_ratio", 0.5)

    best = ga.run(
        target=target_array,
        init_strategy=init_strategy,
        image_ratio=image_ratio,
        callback=combined_callback,
    )
    elapsed_total = time.time() - start

    # -- Save final snapshot if not already saved (early stopping case)
    actual_final_gen = len(ga.generation_log) - 1
    if actual_final_gen not in snapshot_paths:
        p = save_snapshots(best, run_dir, actual_final_gen, target_array)
        snapshot_paths[actual_final_gen] = p

    # -- Cross-evaluate with all three fitness functions
    cross_eval = cross_evaluate(best, target_pil)

    # -- Compute summary
    summary = compute_summary(
        ga.generation_log, config, cross_eval, elapsed_total, run_dir
    )

    # -- Save result.pkl
    result_data = {
        "config": config,
        "generation_log": ga.generation_log,
        "summary": summary,
        "best_triangles": [t.to_dict() for t in best.triangles],
        "snapshot_paths": snapshot_paths,
        "cross_eval": cross_eval,
    }
    if sigma_log is not None:
        result_data["sigma_log"] = sigma_log

    pkl_path = os.path.join(run_dir, "result.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(result_data, f)

    # -- Save config as JSON for human inspection
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # -- Append to master index
    append_to_master_index(summary, output_root)

    logger.info("Run complete: %s | fitness=%.4f | elapsed=%.1fs",
                name, best.fitness, elapsed_total)
    logger.info("Results saved to: %s", run_dir)

    return summary


# ---------------------------------------------------------------------------
# Master index management
# ---------------------------------------------------------------------------

def append_to_master_index(summary: dict, output_root: str = OUTPUT_ROOT):
    """Append one row to the master CSV index. Creates the file if needed."""
    index_path = os.path.join(output_root, "master_index.csv")
    os.makedirs(output_root, exist_ok=True)

    row_df = pd.DataFrame([summary])

    if os.path.exists(index_path):
        existing = pd.read_csv(index_path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(index_path, index=False)


# ---------------------------------------------------------------------------
# Experiment config builders
# ---------------------------------------------------------------------------

def _base_config() -> dict:
    """Common defaults shared by all experiments."""
    return {
        "fitness_metric":       "rmse",
        "population_size":      50,
        "n_generations":        300,
        "crossover_rate":       0.8,
        "n_elites":             1,
        "n_workers":            1,
        "early_stopping_patience": 50,
        "early_stopping_tolerance": 1e-4,
        "selection_operator":   "TournamentSelection",
        "tournament_size":      5,
        "crossover_operator":   "SinglePointCrossover",
        "mutation_operator":    "CompositeMutation",
        "mutation_rate":        0.05,
        "vertex_sigma":         15.0,
        "color_sigma":          15.0,
        "reset_rate":           0.01,
        "swap_rate":            0.2,
        "swap_n":               1,
        "init_strategy":        "image",
        "image_ratio":          0.5,
        "sigma_decay":          False,
        "target_image":         "girl_pearl.png",
    }


def build_phase1_mutation_configs(seed: int = 42) -> List[dict]:
    """
    Phase 1: Coarse sweep of mutation parameters.

    Smaller population, shorter runs -- goal is to find promising
    mutation_rate x vertex_sigma x color_sigma combinations quickly.

    Grid:
        mutation_rate: [0.02, 0.05, 0.10]
        vertex_sigma:  [10, 20, 40]
        color_sigma:   [10, 20, 40]

    Total: 3 x 3 x 3 = 27 runs
    """
    configs = []
    rates = [0.02, 0.05, 0.10]
    v_sigmas = [10.0, 20.0, 40.0]
    c_sigmas = [10.0, 20.0, 40.0]

    for rate, v_sig, c_sig in product(rates, v_sigmas, c_sigmas):
        cfg = _base_config()
        cfg.update({
            "experiment_group": "phase1_mutation",
            "experiment_name":  f"rate{rate:.2f}_vsig{v_sig:.0f}_csig{c_sig:.0f}",
            "seed":             seed,
            "population_size":  30,
            "n_generations":    250,
            "mutation_rate":    rate,
            "vertex_sigma":     v_sig,
            "color_sigma":      c_sig,
            "early_stopping_patience": 0,  # disabled -- run full
        })
        configs.append(cfg)

    return configs


def build_phase2_selection_crossover_configs(
    best_mutation: dict,
    seed: int = 42,
) -> List[dict]:
    """
    Phase 2: Sweep selection and crossover using best mutation from phase 1.

    Grid:
        selection: Tournament(3), Tournament(5), Tournament(7), Rank(1.5)
        crossover: SinglePoint, KPoint(2), Uniform(0.5)
        n_elites:  [1, 2, 3]

    Total: 4 x 3 x 3 = 36 runs
    """
    configs = []

    selections = [
        {"selection_operator": "TournamentSelection", "tournament_size": 3},
        {"selection_operator": "TournamentSelection", "tournament_size": 5},
        {"selection_operator": "TournamentSelection", "tournament_size": 7},
        {"selection_operator": "RankSelection", "selection_pressure": 1.5},
    ]
    crossovers = [
        {"crossover_operator": "SinglePointCrossover"},
        {"crossover_operator": "KPointCrossover", "k": 2},
        {"crossover_operator": "UniformCrossover", "swap_prob": 0.5},
    ]
    elites = [1, 2, 3]

    for sel, cx, n_el in product(selections, crossovers, elites):
        cfg = _base_config()
        cfg.update(best_mutation)
        cfg.update(sel)
        cfg.update(cx)
        cfg.update({
            "experiment_group": "phase2_sel_cx",
            "experiment_name":  (
                f"{sel['selection_operator'][:4].lower()}"
                f"{sel.get('tournament_size', sel.get('selection_pressure', ''))}_"
                f"{cx['crossover_operator'][:6].lower()}_"
                f"el{n_el}"
            ),
            "seed":             seed,
            "population_size":  50,
            "n_generations":    300,
            "n_elites":         n_el,
            "early_stopping_patience": 0,
        })
        configs.append(cfg)

    return configs


def build_phase3_sigma_decay_configs(
    best_config: dict,
    seed: int = 42,
) -> List[dict]:
    """
    Phase 3: Compare fixed sigma vs decay sigma using best config from phase 2.

    Grid:
        sigma_decay:  [False, True]
        For True: (v_max, v_min) in [(40,2), (60,2), (30,3)]

    Total: 4 runs (1 fixed + 3 decay variants)
    """
    configs = []

    # baseline: no decay
    cfg = copy.deepcopy(best_config)
    cfg.update({
        "experiment_group": "phase3_sigma_decay",
        "experiment_name":  "fixed_sigma",
        "seed":             seed,
        "population_size":  80,
        "n_generations":    500,
        "sigma_decay":      False,
        "early_stopping_patience": 80,
    })
    configs.append(cfg)

    # decay variants
    decay_params = [
        {"v_max": 40, "v_min": 2, "c_max": 40, "c_min": 2},
        {"v_max": 60, "v_min": 2, "c_max": 60, "c_min": 2},
        {"v_max": 30, "v_min": 3, "c_max": 30, "c_min": 3},
    ]
    for dp in decay_params:
        cfg = copy.deepcopy(best_config)
        cfg.update({
            "experiment_group":   "phase3_sigma_decay",
            "experiment_name":    f"decay_v{dp['v_max']}to{dp['v_min']}_c{dp['c_max']}to{dp['c_min']}",
            "seed":               seed,
            "population_size":    80,
            "n_generations":      500,
            "sigma_decay":        True,
            "sigma_decay_v_max":  dp["v_max"],
            "sigma_decay_v_min":  dp["v_min"],
            "sigma_decay_c_max":  dp["c_max"],
            "sigma_decay_c_min":  dp["c_min"],
            "early_stopping_patience": 80,
        })
        configs.append(cfg)

    return configs


def build_phase4_final_run(
    best_config: dict,
    seed: int = 42,
) -> List[dict]:
    """
    Phase 4: Final production run with best settings.

    Large population, many generations, checkpoints for the report.
    """
    cfg = copy.deepcopy(best_config)
    cfg.update({
        "experiment_group":   "phase4_final",
        "experiment_name":    f"final_seed{seed}",
        "seed":               seed,
        "population_size":    100,
        "n_generations":      1000,
        "early_stopping_patience": 100,
        "early_stopping_tolerance": 1e-5,
    })
    return [cfg]


# ---------------------------------------------------------------------------
# Run a full experiment group
# ---------------------------------------------------------------------------

def run_experiment_group(
    configs: List[dict],
    target_array: np.ndarray,
    target_pil: Image.Image,
    output_root: str = OUTPUT_ROOT,
) -> pd.DataFrame:
    """
    Run all experiments in a group and return results as a DataFrame.

    Parameters
    ----------
    configs : list of dict
        Experiment configurations.
    target_array : np.ndarray
        Target image as numpy array.
    target_pil : PIL.Image.Image
        Target image as PIL.
    output_root : str
        Root output directory.

    Returns
    -------
    pd.DataFrame
        One row per run with all summary metrics.
    """
    summaries = []
    total = len(configs)

    for i, cfg in enumerate(configs, 1):
        group = cfg.get("experiment_group", "?")
        name = cfg.get("experiment_name", "?")
        print(f"\n[{i}/{total}] {group}/{name}")
        print("-" * 50)

        try:
            summary = run_single_experiment(cfg, target_array, target_pil, output_root)
            summaries.append(summary)
            print(f"  -> fitness={summary['min_best']:.4f}, "
                  f"elapsed={summary['total_elapsed_s']:.1f}s")
        except Exception as e:
            logger.error("FAILED: %s/%s -- %s", group, name, e)
            print(f"  -> FAILED: {e}")

    return pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GA experiment runner")
    parser.add_argument(
        "--group", type=str, default=None,
        help="Run only this experiment group (phase1_mutation, phase2_sel_cx, etc.)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configs and exit without running"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for all experiments"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load target image
    target_pil = load_target_pil(TARGET_PATH)
    target_array = np.array(target_pil)
    print(f"target image loaded: {target_pil.size}, {target_array.shape}")

    # Build phase 1 configs
    phase1_configs = build_phase1_mutation_configs(seed=args.seed)

    if args.group and args.group != "phase1_mutation":
        print(f"skipping phase 1 (--group={args.group})")
    else:
        if args.dry_run:
            print(f"\n--- phase1_mutation: {len(phase1_configs)} experiments ---")
            for cfg in phase1_configs:
                print(f"  {cfg['experiment_name']}")
        else:
            print(f"\n=== PHASE 1: Mutation sweep ({len(phase1_configs)} runs) ===")
            results_p1 = run_experiment_group(
                phase1_configs, target_array, target_pil
            )
            print("\n--- Phase 1 top 5 ---")
            print(results_p1.sort_values("min_best").head()[
                ["experiment_name", "min_best", "mutation_rate",
                 "vertex_sigma", "color_sigma", "total_elapsed_s"]
            ].to_string(index=False))

            # Extract best mutation params for phase 2
            best_p1_row = results_p1.sort_values("min_best").iloc[0]
            best_mutation = {
                "mutation_rate": best_p1_row["mutation_rate"],
                "vertex_sigma":  best_p1_row["vertex_sigma"],
                "color_sigma":   best_p1_row["color_sigma"],
            }

    # Phase 2 -- requires phase 1 results to determine best mutation
    # In practice you'd load from master_index.csv if running separately
    if args.group == "phase2_sel_cx" or (args.group is None and not args.dry_run):
        if "best_mutation" not in dir():
            # Load from master_index if running phase 2 standalone
            print("loading best mutation params from master_index.csv ...")
            idx = pd.read_csv(MASTER_INDEX_PATH)
            p1 = idx[idx["experiment_group"] == "phase1_mutation"]
            best_row = p1.sort_values("min_best").iloc[0]
            best_mutation = {
                "mutation_rate": best_row["mutation_rate"],
                "vertex_sigma":  best_row["vertex_sigma"],
                "color_sigma":   best_row["color_sigma"],
            }
            print(f"  best mutation: {best_mutation}")

        phase2_configs = build_phase2_selection_crossover_configs(
            best_mutation, seed=args.seed
        )

        if args.dry_run:
            print(f"\n--- phase2_sel_cx: {len(phase2_configs)} experiments ---")
            for cfg in phase2_configs:
                print(f"  {cfg['experiment_name']}")
        else:
            print(f"\n=== PHASE 2: Selection/Crossover sweep ({len(phase2_configs)} runs) ===")
            results_p2 = run_experiment_group(
                phase2_configs, target_array, target_pil
            )
            print("\n--- Phase 2 top 5 ---")
            print(results_p2.sort_values("min_best").head()[
                ["experiment_name", "min_best", "selection_operator",
                 "crossover_operator", "n_elites", "total_elapsed_s"]
            ].to_string(index=False))

    print("\ndone. results in:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()