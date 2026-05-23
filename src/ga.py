"""
ga.py
-----
The Genetic Algorithm engine for the image approximation problem.

The GeneticAlgorithm class orchestrates the full evolutionary loop:
    1. Initialise a random population.
    2. Evaluate all individuals.
    3. For each generation:
        a. Log generation statistics.
        b. Check early stopping condition.
        c. Select parents.
        d. Apply crossover to produce offspring.
        e. Apply mutation to offspring.
        f. Replace current population with offspring (with elitism).
        g. Evaluate new offspring.
    4. Return the best individual found.

Design notes:
    - The engine depends only on abstract operator interfaces
      (SelectionOperator, CrossoverOperator, MutationOperator). Concrete
      implementations are injected at construction time (dependency
      injection), so any combination of operators can be plugged in without
      touching this file. This is the core design property that makes the
      ablation study (challenge 3) straightforward.
    - Crossover probability (crossover_rate) is applied at the parent-pair
      level: with probability crossover_rate the two parents are crossed;
      otherwise the parents are copied directly into offspring unchanged.
      This allows the engine to interpolate between pure crossover and pure
      reproduction.
    - A generation log is maintained as a list of dicts, one per generation.
      Each dict contains the generation index, timestamp, fitness statistics
      and diversity. This feeds directly into the notebook's convergence
      plots with no post-processing required.
    - Checkpointing: the best individual is saved to disk every
      checkpoint_interval generations if a checkpoint_dir is provided.
      This protects against data loss during long runs.
    - Early stopping: the run terminates early if the best fitness does
      not improve by more than tolerance over patience generations.
      Controlled by the EarlyStopping helper class.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from fitness import FitnessFunction
from individual import Individual
from population import Population
from ga_operators.selection import SelectionOperator
from ga_operators.crossover import CrossoverOperator
from ga_operators.mutation import MutationOperator
from utils import triangles_to_json, save_render, render

# ---------------------------------------------------------------------------
# Module-level logger — callers can configure the logging level externally
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

@dataclass
class EarlyStopping:
    """
    Tracks improvement in best fitness and signals when to stop early.

    Stops the run if best fitness does not improve by more than `tolerance`
    over `patience` consecutive generations. This prevents wasting compute
    on a stagnated run.

    Parameters
    ----------
    patience : int
        Number of generations without improvement before stopping.
        Set to 0 or None to disable early stopping entirely.
    tolerance : float
        Minimum absolute improvement in best fitness to count as progress.
        Improvements smaller than tolerance are treated as no improvement.
    """

    patience: int = 50
    tolerance: float = 1e-4
    _best_fitness: float = field(default=float("inf"), init=False, repr=False)
    _generations_without_improvement: int = field(default=0, init=False, repr=False)

    def update(self, current_best: float) -> bool:
        """
        Update state with the current generation's best fitness.

        Parameters
        ----------
        current_best : float
            Best fitness value in the current generation.

        Returns
        -------
        bool
            True if the run should stop early, False otherwise.
        """
        if self.patience <= 0:
            return False

        improvement = self._best_fitness - current_best
        if improvement > self.tolerance:
            self._best_fitness = current_best
            self._generations_without_improvement = 0
        else:
            self._generations_without_improvement += 1

        if self._generations_without_improvement >= self.patience:
            logger.info(
                "Early stopping triggered: no improvement > %.6f "
                "for %d consecutive generations.",
                self.tolerance,
                self.patience,
            )
            return True
        return False

    def reset(self) -> None:
        """Reset state — used when restarting a run."""
        self._best_fitness = float("inf")
        self._generations_without_improvement = 0


# ---------------------------------------------------------------------------
# Diversity-aware early stopping
# ---------------------------------------------------------------------------

@dataclass
class DiversityAwareEarlyStopping:
    """
    Early stopping that requires both fitness stagnation AND diversity collapse
    before terminating the run.

    Motivation
    ----------
    The plain EarlyStopping class stops as soon as best fitness stops improving
    for `patience` generations, regardless of whether the population still has
    genetic diversity. This can kill a run prematurely: the population may be
    exploring diverse regions of the search space that have not yet produced
    a better best individual.

    This class adds a second condition: the run only stops when fitness has
    stagnated AND at least one diversity metric has fallen below its threshold.
    If fitness is stuck but the population is still diverse, the stagnation
    counter is reset — the algorithm is given more time to exploit its
    remaining diversity.

    The two diversity metrics used are:
        - phenotypic_variance : variance of fitness values across the population.
          Drops toward 0 when all individuals converge to similar fitness.
        - genotypic_variance  : variance of L2 distances from each individual's
          chromosome to the best individual's chromosome. Drops toward 0 when
          all chromosomes are genetically similar.

    Either metric falling below its threshold is treated as a diversity collapse
    signal. Requiring both to collapse simultaneously would be too conservative;
    requiring either is the safer choice for early stopping.

    Parameters
    ----------
    patience : int
        Consecutive generations of fitness stagnation required before checking
        diversity. Same semantics as EarlyStopping.patience. Default 50.
    tolerance : float
        Minimum absolute improvement in best fitness to count as progress.
        Same semantics as EarlyStopping.tolerance. Default 1e-4.
    phenotypic_variance_threshold : float
        Phenotypic variance value below which the population is considered
        phenotypically converged. Appropriate value depends on the fitness
        scale (RMSE values typically in [20, 80]); default 0.01 means
        fitness values are clustered within ~0.1 RMSE of each other.
    genotypic_variance_threshold : float
        Genotypic variance value below which the population is considered
        genotypically converged. This is variance of normalised L2 distances
        (chromosomes normalised to [0,1] per axis), so the scale is
        independent of image resolution. Default 1e-4.
    diversity_check_interval : int
        How often (in generations) to recompute genotypic_variance, which
        requires building the full chromosome matrix and is more expensive
        than fitness-only metrics. Default 10 (check every 10 generations).
        Set to 1 to check every generation (slower but more responsive).
    """

    patience: int = 50
    tolerance: float = 1e-4
    phenotypic_variance_threshold: float = 0.01
    genotypic_variance_threshold: float = 1e-4
    diversity_check_interval: int = 10

    # internal state — not constructor params
    _best_fitness: float = field(default=float("inf"), init=False, repr=False)
    _stagnation_counter: int = field(default=0, init=False, repr=False)
    _last_phenotypic_var: float = field(default=float("inf"), init=False, repr=False)
    _last_genotypic_var: float = field(default=float("inf"), init=False, repr=False)

    def update(
        self,
        generation: int,
        current_best: float,
        population: "Population",
    ) -> bool:
        """
        Update state and decide whether to stop.

        Parameters
        ----------
        generation : int
            Current generation index. Used to gate the expensive genotypic
            variance computation to every diversity_check_interval generations.
        current_best : float
            Best fitness in the current generation.
        population : Population
            The evaluated population. Used to compute diversity metrics when
            the diversity check interval fires.

        Returns
        -------
        bool
            True if the run should stop, False otherwise.
        """
        if self.patience <= 0:
            return False

        # -- Fitness stagnation check (same logic as EarlyStopping)
        improvement = self._best_fitness - current_best
        if improvement > self.tolerance:
            self._best_fitness = current_best
            self._stagnation_counter = 0
            return False  # fitness still improving - never stop regardless of diversity

        self._stagnation_counter += 1

        # not yet patient enough to consider stopping
        if self._stagnation_counter < self.patience:
            return False

        # -- Fitness has stagnated for `patience` gens - now check diversity
        # recompute on the check interval; reuse cached values otherwise
        if generation % self.diversity_check_interval == 0:
            self._last_phenotypic_var = population.phenotypic_variance()
            self._last_genotypic_var  = population.genotypic_variance()

            logger.debug(
                "Diversity check at gen %d: pheno_var=%.6f, geno_var=%.6f",
                generation,
                self._last_phenotypic_var,
                self._last_genotypic_var,
            )

        pheno_collapsed = self._last_phenotypic_var < self.phenotypic_variance_threshold
        geno_collapsed  = self._last_genotypic_var  < self.genotypic_variance_threshold

        if pheno_collapsed or geno_collapsed:
            logger.info(
                "DiversityAwareEarlyStopping triggered at gen %d: "
                "stagnation=%d gens, pheno_var=%.6f (threshold=%.6f), "
                "geno_var=%.6f (threshold=%.6f).",
                generation,
                self._stagnation_counter,
                self._last_phenotypic_var,
                self.phenotypic_variance_threshold,
                self._last_genotypic_var,
                self.genotypic_variance_threshold,
            )
            return True

        # fitness stagnated but population still diverse - reset counter and
        # give the algorithm more time to exploit remaining diversity
        logger.debug(
            "Fitness stagnated for %d gens but diversity is healthy "
            "(pheno_var=%.6f, geno_var=%.6f) - resetting stagnation counter.",
            self._stagnation_counter,
            self._last_phenotypic_var,
            self._last_genotypic_var,
        )
        self._stagnation_counter = 0
        return False

    def reset(self) -> None:
        """Reset all internal state — call before re-running the GA."""
        self._best_fitness = float("inf")
        self._stagnation_counter = 0
        self._last_phenotypic_var = float("inf")
        self._last_genotypic_var = float("inf")

    def __repr__(self) -> str:
        return (
            f"DiversityAwareEarlyStopping("
            f"patience={self.patience}, "
            f"tolerance={self.tolerance}, "
            f"pheno_var_threshold={self.phenotypic_variance_threshold}, "
            f"geno_var_threshold={self.genotypic_variance_threshold}, "
            f"check_interval={self.diversity_check_interval})"
        )


# ---------------------------------------------------------------------------
# GA configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    """
    Hyperparameter configuration for the GeneticAlgorithm.

    Centralising all hyperparameters in a dataclass makes it trivial to
    serialise experiment configurations to JSON, reproduce runs from a
    config file, and sweep parameters systematically in notebooks.

    Parameters
    ----------
    population_size : int
        Number of individuals per generation. Default 50.
    n_generations : int
        Maximum number of generations to run. Default 500.
    crossover_rate : float
        Probability of applying crossover to a parent pair. In [0, 1].
        If crossover does not apply, parents are passed directly to
        mutation unchanged. Default 0.8.
    n_elites : int
        Number of best individuals carried forward unchanged each
        generation (elitism). Default 1.
    n_workers : int
        Number of threads for parallel fitness evaluation. Default 1
        (serial). Increase for large populations on multi-core machines.
    early_stopping : EarlyStopping
        Early stopping configuration. Default: patience=50, tol=1e-4.
    checkpoint_interval : int
        Save a checkpoint every this many generations. 0 disables.
        Default 50.
    checkpoint_dir : str
        Directory for checkpoint files. Default 'outputs/checkpoints'.
    seed : int or None
        Random seed for reproducibility. None means non-deterministic.
    """

    population_size: int = 50
    n_generations: int = 500
    crossover_rate: float = 0.8
    n_elites: int = 1
    n_workers: int = 1
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)
    diversity_early_stopping: Optional[DiversityAwareEarlyStopping] = field(default=None)
    checkpoint_interval: int = 50
    checkpoint_dir: str = "outputs/checkpoints"
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialise config to a plain dict (for JSON logging)."""
        des = self.diversity_early_stopping
        return {
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "crossover_rate": self.crossover_rate,
            "n_elites": self.n_elites,
            "n_workers": self.n_workers,
            "early_stopping_patience": self.early_stopping.patience,
            "early_stopping_tolerance": self.early_stopping.tolerance,
            "diversity_early_stopping_enabled": des is not None,
            "diversity_early_stopping_patience": des.patience if des else None,
            "diversity_early_stopping_pheno_threshold": des.phenotypic_variance_threshold if des else None,
            "diversity_early_stopping_geno_threshold": des.genotypic_variance_threshold if des else None,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# GA engine
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """
    Genetic Algorithm engine for triangle-based image approximation.

    Orchestrates the full evolutionary loop using injected operator
    strategies. The engine is agnostic to which concrete operators are
    used — it only calls the abstract interfaces defined in operators/.

    Parameters
    ----------
    fitness_fn : FitnessFunction
        Fitness function shared across all individuals. Evaluated once
        per individual per generation.
    selection : SelectionOperator
        Strategy for selecting parents from the current population.
    crossover : CrossoverOperator
        Strategy for combining two parents into two offspring.
    mutation : MutationOperator
        Strategy for introducing variation into offspring.
    config : GAConfig
        Hyperparameter configuration. Uses defaults if not provided.

    Attributes
    ----------
    generation_log : list of dict
        Per-generation statistics accumulated during run(). Each entry
        contains keys: generation, elapsed_s, best, mean, std, worst,
        diversity.
    best_individual : Individual or None
        The best individual found across all generations. Updated every
        generation; available after run() completes.
    current_population : Population or None
        The most recently evaluated Population. Updated every generation
        so callbacks can access full diversity metrics via
        population.diversity_report() without the GA engine needing to
        recompute them independently.
    """

    def __init__(
        self,
        fitness_fn: FitnessFunction,
        selection: SelectionOperator,
        crossover: CrossoverOperator,
        mutation: MutationOperator,
        config: Optional[GAConfig] = None,
    ) -> None:
        self._fitness_fn = fitness_fn
        self._selection = selection
        self._crossover = crossover
        self._mutation = mutation
        self._config = config or GAConfig()

        # State initialised at run time
        self._rng: np.random.Generator = np.random.default_rng(self._config.seed)
        self.generation_log: List[dict] = []
        self.best_individual: Optional[Individual] = None
        self.current_population: Optional[Population] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        target: Optional[np.ndarray] = None,
        init_strategy: str = "random",
        image_ratio: float = 0.5,
        callback: Optional[Callable[[int, Individual, dict], None]] = None,
    ) -> Individual:
        """
        Execute the full evolutionary loop.

        Parameters
        ----------
        target : np.ndarray, optional
            H×W×3 uint8 RGB array of the target image. Required when
            init_strategy is 'image' or 'mixed'. Ignored for 'random'.
        init_strategy : str
            Population initialisation strategy. One of:
                'random' -- all triangles fully random (default).
                'image'  -- triangle colors sampled from target image.
                'mixed'  -- image_ratio fraction image-seeded, rest random.
        image_ratio : float
            Fraction of image-seeded individuals when init_strategy='mixed'.
            Ignored for other strategies. Default 0.5.
        callback : callable, optional
            Called at the end of every generation with signature:
                callback(generation: int, best: Individual, stats: dict)
            Useful for live progress display in notebooks.

        Returns
        -------
        Individual
            The best individual found across all generations.
        """
        cfg = self._config
        self._rng = np.random.default_rng(cfg.seed)
        self.generation_log = []
        self.best_individual = None
        cfg.early_stopping.reset()
        if cfg.diversity_early_stopping is not None:
            cfg.diversity_early_stopping.reset()

        if init_strategy in ("image", "mixed", "grid") and target is None:
            raise ValueError(
                f"init_strategy='{init_strategy}' requires a target array. "
                "Pass target=your_image_array to run()."
            )

        _KNOWN_STRATEGIES = {
            "random", "image", "mixed", "grid",
            "semi_transparent", "small_random", "grid_random_color",
            "sorted_alpha", "quadrant",
        }
        if init_strategy not in _KNOWN_STRATEGIES:
            raise ValueError(
                f"Unknown init_strategy='{init_strategy}'. "
                f"Must be one of: {sorted(_KNOWN_STRATEGIES)}"
            )

        start_time = time.time()

        logger.info(
            "Initialising population (size=%d, strategy=%s).",
            cfg.population_size, init_strategy,
        )

        if init_strategy == "image":
            population = Population.from_image(
                cfg.population_size, self._fitness_fn, self._rng, target
            )
        elif init_strategy == "mixed":
            population = Population.mixed(
                cfg.population_size, self._fitness_fn, self._rng, target,
                image_ratio=image_ratio,
            )
        elif init_strategy == "grid":
            population = Population.from_grid(
                cfg.population_size, self._fitness_fn, self._rng, target
            )
        elif init_strategy == "semi_transparent":
            population = Population.random_semitransparent(
                cfg.population_size, self._fitness_fn, self._rng
            )
        elif init_strategy == "small_random":
            population = Population.random_small(
                cfg.population_size, self._fitness_fn, self._rng
            )
        elif init_strategy == "grid_random_color":
            population = Population.from_grid_random_color(
                cfg.population_size, self._fitness_fn, self._rng
            )
        elif init_strategy == "sorted_alpha":
            population = Population.random_sorted_alpha(
                cfg.population_size, self._fitness_fn, self._rng
            )
        elif init_strategy == "quadrant":
            population = Population.random_quadrant(
                cfg.population_size, self._fitness_fn, self._rng
            )
        else:
            population = Population.random(
                cfg.population_size, self._fitness_fn, self._rng
            )
        population.evaluate(n_workers=cfg.n_workers)
        self.current_population = population

        self.best_individual = population.best
        self._log_generation(0, population, time.time() - start_time)
        self._maybe_checkpoint(0, population.best)

        if callback:
            callback(0, self.best_individual, self.generation_log[-1])

        # -- Main loop
        for gen in range(1, cfg.n_generations + 1):
            population = self._step(population)
            population.evaluate(n_workers=cfg.n_workers)
            self.current_population = population

            # Update global best across all generations
            if population.best < self.best_individual:
                self.best_individual = population.best

            elapsed = time.time() - start_time
            stats = self._log_generation(gen, population, elapsed)

            logger.info(
                "gen %4d | best=%.4f | mean=%.4f | std=%.4f | div=%.4f | %.1fs",
                gen,
                stats["best"],
                stats["mean"],
                stats["std"],
                stats["diversity"],
                elapsed,
            )

            self._maybe_checkpoint(gen, self.best_individual)

            if callback:
                callback(gen, self.best_individual, stats)

            if cfg.early_stopping.update(stats["best"]):
                logger.info("Run ended at generation %d by early stopping.", gen)
                break

            if cfg.diversity_early_stopping is not None:
                if cfg.diversity_early_stopping.update(gen, stats["best"], population):
                    logger.info(
                        "Run ended at generation %d by diversity-aware early stopping.", gen
                    )
                    break

        logger.info(
            "Run complete. Best fitness: %.4f after %d generations.",
            self.best_individual.fitness,
            len(self.generation_log) - 1,
        )
        return self.best_individual

    # ------------------------------------------------------------------
    # Single generation step
    # ------------------------------------------------------------------

    def _step(self, population: Population) -> Population:
        """
        Produce the next generation from the current population.

        Applies selection -> crossover -> mutation to produce a full set
        of offspring, then calls population.replace() with the configured
        elitism count.

        Parameters
        ----------
        population : Population
            The current (evaluated) generation.

        Returns
        -------
        Population
            The next (unevaluated) generation, ready for evaluate().
        """
        cfg = self._config
        n_offspring = cfg.population_size - cfg.n_elites

        # -- Selection: draw 2 * n_offspring parents (pairs for crossover)
        # We need n_offspring offspring total; each crossover call produces 2,
        # so we need ceil(n_offspring / 2) crossover operations -> n_parents pairs.
        n_pairs = (n_offspring + 1) // 2
        parents = self._selection.select(
            population.individuals,
            n_parents=n_pairs * 2,
            rng=self._rng,
        )

        # -- Crossover + mutation
        offspring: List[Individual] = []

        for i in range(0, len(parents) - 1, 2):
            parent_a = parents[i]
            parent_b = parents[i + 1]

            # Apply crossover with probability crossover_rate
            if self._rng.random() < cfg.crossover_rate:
                child_a, child_b = self._crossover.cross(
                    parent_a, parent_b, self._rng
                )
            else:
                # No crossover — pass parents forward as-is (copies)
                child_a = parent_a.copy_with()
                child_b = parent_b.copy_with()

            # Mutate both children
            offspring.append(self._mutation.mutate(child_a, self._rng))
            offspring.append(self._mutation.mutate(child_b, self._rng))

        # Trim to exactly n_offspring (handles odd n_offspring case)
        offspring = offspring[:n_offspring]

        # -- Replacement with elitism
        return population.replace(offspring, n_elites=cfg.n_elites)

    # ------------------------------------------------------------------
    # Logging and checkpointing
    # ------------------------------------------------------------------

    def _log_generation(
        self,
        generation: int,
        population: Population,
        elapsed_s: float,
    ) -> dict:
        """
        Record statistics for the current generation.

        Parameters
        ----------
        generation : int
            Current generation index (0 = initial population).
        population : Population
            The evaluated population for this generation.
        elapsed_s : float
            Wall-clock seconds elapsed since run() started.

        Returns
        -------
        dict
            The log entry appended to self.generation_log.
        """
        stats = population.stats()
        diversity = population.diversity()

        entry = {
            "generation": generation,
            "elapsed_s":  round(elapsed_s, 2),
            "diversity":  round(diversity, 6),
            **{k: round(v, 6) for k, v in stats.items()},
        }
        self.generation_log.append(entry)
        return entry

    def _maybe_checkpoint(self, generation: int, best: Individual) -> None:
        """
        Save a checkpoint if the generation interval condition is met.

        Saves both the triangle JSON (for reloading) and the rendered PNG
        (for visual inspection without re-running the GA).

        Parameters
        ----------
        generation : int
            Current generation index.
        best : Individual
            Best individual to checkpoint.
        """
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
        """
        Save the generation log to a JSON file.

        The log can be loaded into a pandas DataFrame in the notebook:

            import pandas as pd, json
            with open("outputs/run_log.json") as f:
                log = pd.DataFrame(json.load(f))

        Parameters
        ----------
        path : str
            Destination JSON file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.generation_log, f, indent=2)
        logger.info("Generation log saved -> %s", path)

    def save_config(self, path: str) -> None:
        """
        Save the run configuration to a JSON file.

        Useful for experiment reproducibility — pair with save_log() so
        every run's results are associated with the exact config used.

        Parameters
        ----------
        path : str
            Destination JSON file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._config.to_dict(), f, indent=2)
        logger.info("Run config saved -> %s", path)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GeneticAlgorithm(\n"
            f"  selection={self._selection!r},\n"
            f"  crossover={self._crossover!r},\n"
            f"  mutation={self._mutation!r},\n"
            f"  config={self._config!r}\n"
            f")"
        )