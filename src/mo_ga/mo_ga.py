"""
mo_ga/mo_ga.py
--------------
NSGA-II Genetic Algorithm engine for multi-objective image approximation.

Orchestrates the full NSGA-II evolutionary loop:
    1. Initialise a random population of MOIndividuals.
    2. Evaluate all individuals (render + compute all objectives).
    3. Assign Pareto ranks and crowding distances.
    4. For each generation:
        a. Log generation statistics.
        b. Check early stopping condition.
        c. Select parents via MOTournamentSelection (crowded comparison).
        d. Apply crossover to produce offspring.
        e. Apply mutation to offspring.
        f. Combine parent + offspring populations (size 2N).
        g. Re-rank the combined pool.
        h. Select best N individuals (NSGA-II environmental selection).
    5. Return the complete Pareto front (all rank=1 individuals).

Design notes:
    - The engine depends on the same abstract operator interfaces as the
      original GA (CrossoverOperator, MutationOperator). All existing
      crossover and mutation operators work without modification because
      MOIndividual.copy_with() is interface-compatible with Individual.copy_with().
    - Early stopping monitors the best RMSE value (first objective) for
      stagnation. This is a simplification — a more sophisticated stopping
      criterion would monitor Pareto front convergence, but RMSE stagnation
      is a practical proxy that works well for this problem.
    - The generation log records both first-objective statistics (for
      compatibility with the original notebook plots) and Pareto front
      statistics (front size, per-objective min/mean/max).
    - Configuration reuses GAConfig from the original ga.py — all fields
      apply directly. n_elites is accepted but not used (NSGA-II provides
      implicit elitism via the combined population strategy).

References
----------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist
multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary
Computation, 6(2), 182–197.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from ga import GAConfig, EarlyStopping                  # reuse from original
from mo_ga.mo_individual import MOIndividual
from mo_ga.mo_population import MOPopulation
from mo_ga.mo_selection import MOTournamentSelection
from fitness import FitnessFunction
from ga_operators.crossover import CrossoverOperator
from ga_operators.mutation import MutationOperator
from utils import triangles_to_json, save_render, render

logger = logging.getLogger(__name__)


class MOGA:
    """
    NSGA-II Genetic Algorithm engine for triangle-based image approximation.

    Uses all existing crossover and mutation operators unchanged. Replaces
    the single-objective fitness evaluation with multi-objective evaluation
    and NSGA-II Pareto-based selection.

    Parameters
    ----------
    fitness_fns : list of FitnessFunction
        Objective functions. Evaluated in order; indices are preserved
        in MOIndividual.fitness_values. Typically [RMSEFitness, CIEDEFitness].
    crossover : CrossoverOperator
        Any crossover operator from ga_operators.crossover.
    mutation : MutationOperator
        Any mutation operator from ga_operators.mutation.
    config : GAConfig
        Hyperparameter configuration (population_size, n_generations, etc.)
        Reused directly from the original GA configuration dataclass.
    tournament_size : int
        Tournament size for MOTournamentSelection. Default 2 (NSGA-II standard).
    """

    def __init__(
        self,
        fitness_fns: List[FitnessFunction],
        crossover: CrossoverOperator,
        mutation: MutationOperator,
        config: GAConfig,
        tournament_size: int = 2,
    ) -> None:
        if not fitness_fns:
            raise ValueError("At least one fitness function required.")

        self._fitness_fns = fitness_fns
        self._crossover = crossover
        self._mutation = mutation
        self._config = config
        self._selection = MOTournamentSelection(tournament_size=tournament_size)
        self._rng = np.random.default_rng(config.seed)

        # State populated during run()
        self.generation_log: List[Dict] = []
        self.best_individual: Optional[MOIndividual] = None
        self.current_population: Optional[MOPopulation] = None

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        target: np.ndarray,
        init_strategy: str = "random",
        image_ratio: float = 0.5,
        callback: Optional[Callable] = None,
    ) -> List[MOIndividual]:
        """
        Execute the NSGA-II loop and return the final Pareto front.

        Parameters
        ----------
        target : np.ndarray
            H×W×3 uint8 RGB target image. Passed to image-based
            initialisation strategies if needed.
        init_strategy : str
            Initialisation strategy: 'random', 'image', or 'quadrant'.
        image_ratio : float
            Used only when init_strategy='mixed' (kept for interface
            compatibility with runner.py). Default 0.5.
        callback : callable, optional
            Called every generation as callback(generation, best_individual,
            stats_dict, pareto_front). Useful for checkpointing and logging.

        Returns
        -------
        list of MOIndividual
            All individuals on the final Pareto front (rank=1), sorted by
            crowding distance (most isolated first).
        """
        cfg = self._config
        self.generation_log = []

        # -- Initialise population
        population = MOPopulation.random(
            size=cfg.population_size,
            fitness_fns=self._fitness_fns,
            rng=self._rng,
            init_strategy=init_strategy,
            target=target,
        )

        # -- Evaluate and rank initial population
        population.evaluate(n_workers=cfg.n_workers)
        population.assign_pareto_ranks()
        self.current_population = population
        self.best_individual = population.best

        start_time = time.time()
        stats = self._log_generation(0, population, time.time() - start_time)
        logger.info(
            "gen    0 | best_rmse=%.4f | front_size=%d | %.1fs",
            stats["best"],
            stats.get("front_size", 0),
            time.time() - start_time,
        )

        if callback:
            callback(0, self.best_individual, stats, population.pareto_front)

        early_stop = EarlyStopping(
            patience=cfg.early_stopping.patience,
            tolerance=cfg.early_stopping.tolerance,
        )

        # -- Generational loop
        for gen in range(1, cfg.n_generations + 1):
            population = self._step(population)
            self.current_population = population

            # Update best individual (by NSGA-II crowded comparison)
            gen_best = population.best
            if self.best_individual is None or gen_best < self.best_individual:
                self.best_individual = gen_best

            elapsed = time.time() - start_time
            stats = self._log_generation(gen, population, elapsed)

            logger.info(
                "gen %4d | best_rmse=%.4f | front_size=%d | %.1fs",
                gen,
                stats["best"],
                stats.get("front_size", 0),
                elapsed,
            )

            self._maybe_checkpoint(gen, self.best_individual)

            if callback:
                callback(gen, self.best_individual, stats, population.pareto_front)

            if early_stop.update(stats["best"]):
                logger.info("Early stopping at generation %d.", gen)
                break

        final_front = sorted(population.pareto_front, reverse=True,
                             key=lambda ind: ind.crowding_distance)
        logger.info(
            "NSGA-II complete. Final Pareto front size: %d. "
            "Best RMSE: %.4f after %d generations.",
            len(final_front),
            min(ind.fitness_values[0] for ind in final_front) if final_front else float("inf"),
            len(self.generation_log) - 1,
        )
        return final_front

    # ------------------------------------------------------------------
    # Single generation step
    # ------------------------------------------------------------------

    def _step(self, population: MOPopulation) -> MOPopulation:
        """
        Produce the next generation using NSGA-II replacement.

        Steps:
            1. Select parents via crowded tournament selection.
            2. Apply crossover + mutation to produce offspring.
            3. Evaluate offspring.
            4. Combine parents + offspring (size 2N).
            5. Re-rank combined pool.
            6. Return best N by NSGA-II crowded comparison.
        """
        cfg = self._config
        n_offspring = cfg.population_size

        # Select 2 * ceil(n_offspring / 2) parents
        n_pairs = (n_offspring + 1) // 2
        parents = self._selection.select(
            population.individuals,
            n_parents=n_pairs * 2,
            rng=self._rng,
        )

        # Crossover + mutation
        offspring: List[MOIndividual] = []
        for i in range(0, len(parents) - 1, 2):
            parent_a = parents[i]
            parent_b = parents[i + 1]

            if self._rng.random() < cfg.crossover_rate:
                child_a, child_b = self._crossover.cross(parent_a, parent_b, self._rng)
            else:
                child_a = parent_a.copy_with()
                child_b = parent_b.copy_with()

            offspring.append(self._mutation.mutate(child_a, self._rng))
            offspring.append(self._mutation.mutate(child_b, self._rng))

        offspring = offspring[:n_offspring]

        # Evaluate offspring
        for child in offspring:
            child.evaluate()

        # NSGA-II replacement: combine + re-rank + select best N
        return population.replace(offspring, n_elites=cfg.n_elites)

    # ------------------------------------------------------------------
    # Logging and checkpointing
    # ------------------------------------------------------------------

    def _log_generation(
        self,
        generation: int,
        population: MOPopulation,
        elapsed_s: float,
    ) -> dict:
        """Record statistics for the current generation."""
        stats = population.stats()
        front_stats = population.pareto_front_stats()
        diversity = population.diversity()

        entry = {
            "generation":  generation,
            "elapsed_s":   round(elapsed_s, 2),
            "diversity":   round(diversity, 6),
            "front_size":  front_stats["front_size"],
            **{k: round(v, 6) for k, v in stats.items()},
        }

        # Per-objective Pareto front statistics
        obj_names = ["rmse", "ciede"][: len(self._fitness_fns)]
        for i, name in enumerate(obj_names):
            if i < len(front_stats.get("obj_mins", [])):
                entry[f"front_{name}_min"]  = round(front_stats["obj_mins"][i], 6)
                entry[f"front_{name}_mean"] = round(front_stats["obj_means"][i], 6)
                entry[f"front_{name}_max"]  = round(front_stats["obj_maxs"][i], 6)

        self.generation_log.append(entry)
        return entry

    def _maybe_checkpoint(self, generation: int, best: MOIndividual) -> None:
        """Save a checkpoint if the generation interval condition is met."""
        cfg = self._config
        if cfg.checkpoint_interval <= 0:
            return
        if generation % cfg.checkpoint_interval != 0:
            return

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        json_path = os.path.join(
            cfg.checkpoint_dir, f"best_gen_{generation:05d}.json"
        )
        png_path = os.path.join(
            cfg.checkpoint_dir, f"best_gen_{generation:05d}.png"
        )

        triangles_to_json(list(best.triangles), json_path)
        save_render(render(best.triangles), png_path)
        logger.info("Checkpoint saved -> %s", json_path)

    # ------------------------------------------------------------------
    # Log export
    # ------------------------------------------------------------------

    def save_log(self, path: str) -> None:
        """Save the generation log to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.generation_log, f, indent=2)
        logger.info("Generation log saved -> %s", path)

    def save_config(self, path: str) -> None:
        """Save the run configuration to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._config.to_dict(), f, indent=2)
        logger.info("Run config saved -> %s", path)

    def save_pareto_front(self, path: str, front: List[MOIndividual]) -> None:
        """
        Save all individuals on the Pareto front to a JSON file.

        Each entry includes the triangle chromosome, fitness values for all
        objectives, Pareto rank and crowding distance.

        Parameters
        ----------
        path : str
            Destination JSON file path.
        front : list of MOIndividual
            The Pareto front to save (typically the return value of run()).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = [ind.to_dict() for ind in front]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Pareto front (%d individuals) saved -> %s", len(front), path)

    def __repr__(self) -> str:
        return (
            f"MOGA(\n"
            f"  n_objectives={len(self._fitness_fns)},\n"
            f"  crossover={self._crossover!r},\n"
            f"  mutation={self._mutation!r},\n"
            f"  config={self._config!r}\n"
            f")"
        )
