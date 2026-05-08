"""
population.py
-------------
Defines the Population class, which manages the collection of Individual
objects that form a single generation in the genetic algorithm.

Responsibilities
----------------
- Initialisation: create a population of N random individuals.
- Evaluation: batch-evaluate all unevaluated individuals (supports
  optional parallel evaluation via concurrent.futures).
- Statistics: expose per-generation metrics (best, mean, std fitness)
  consumed by the GA engine for logging and early stopping.
- Replacement: produce the next generation from a set of offspring,
  with configurable elitism.
- Diversity tracking: compute a simple genotypic diversity metric to
  detect premature convergence.

Design notes:
    - Population does not implement the GA loop — that is the GA engine's
      responsibility. Population is a data container with evaluation and
      replacement logic only.
    - Elitism is implemented here (not in the GA engine) because it is
      inherently a population-level operation: it requires knowing the
      global best individual before replacement.
    - Parallel evaluation is opt-in. Serial evaluation is the default
      because PIL rendering releases the GIL (I/O-bound), making
      ThreadPoolExecutor effective without the overhead of multiprocessing.
      Parallel evaluation is only beneficial for large populations (>= 50)
      due to thread spawn overhead.
    - The population maintains an internal sorted order (ascending fitness)
      after every evaluate() call. This makes best/worst access O(1) and
      avoids repeated sorting in the GA engine.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np

from individual import Individual
from fitness import FitnessFunction


class Population:
    """
    A generation of Individual objects in the genetic algorithm.

    Parameters
    ----------
    individuals : list of Individual
        The individuals comprising this generation. Length defines
        the population size, which stays constant across generations.
    """

    def __init__(self, individuals: List[Individual]) -> None:
        if not individuals:
            raise ValueError("Population must contain at least one individual.")
        self._individuals: List[Individual] = list(individuals)
        self._sorted: bool = False   # tracks whether _individuals is sorted by fitness
        self._size: int = len(individuals)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        size: int,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
    ) -> Population:
        """
        Create a Population of randomly initialised individuals.

        Parameters
        ----------
        size : int
            Number of individuals. Typical range: 20-100. Larger
            populations explore more broadly but cost more per generation.
        fitness_fn : FitnessFunction
            Shared fitness function — the same object is referenced by all
            individuals. No redundant copies of the target image are made.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.

        Returns
        -------
        Population
            Unevaluated population of `size` random individuals.
        """
        if size < 2:
            raise ValueError(f"Population size must be >= 2, got {size}.")

        individuals = [
            Individual.random(fitness_fn, rng)
            for _ in range(size)
        ]
        return cls(individuals)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of individuals in this population."""
        return self._size

    @property
    def individuals(self) -> List[Individual]:
        """
        The current list of individuals.

        Not guaranteed to be sorted unless evaluate() has been called.
        Use best, worst, or sorted_individuals for order-dependent access.
        """
        return self._individuals

    @property
    def best(self) -> Individual:
        """
        The individual with the lowest (best) fitness.

        Requires evaluate() to have been called at least once.

        Returns
        -------
        Individual
            Best individual in the current generation.
        """
        self._require_evaluated()
        if self._sorted:
            return self._individuals[0]
        return min(self._individuals)

    @property
    def worst(self) -> Individual:
        """
        The individual with the highest (worst) fitness.

        Requires evaluate() to have been called at least once.
        """
        self._require_evaluated()
        if self._sorted:
            return self._individuals[-1]
        return max(self._individuals)

    @property
    def sorted_individuals(self) -> List[Individual]:
        """
        Individuals sorted ascending by fitness (best first).

        Sorts in place on first access and caches the sorted order.
        Subsequent accesses are O(1).
        """
        self._require_evaluated()
        if not self._sorted:
            self._individuals.sort()   # uses Individual.__lt__
            self._sorted = True
        return self._individuals

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_workers: int = 1) -> None:
        """
        Evaluate all unevaluated individuals and sort by fitness.

        Already-evaluated individuals (fitness cached) are skipped,
        so this method is safe to call multiple times — e.g. after
        elitism carries evaluated individuals forward from the previous
        generation, only the new offspring are re-evaluated.

        Parameters
        ----------
        n_workers : int
            Number of threads for parallel evaluation.
            n_workers=1 (default) -> serial evaluation.
            n_workers>1 -> ThreadPoolExecutor with n_workers threads.
            PIL rendering releases the GIL, so thread parallelism is
            effective here without multiprocessing overhead. Recommended
            only for populations >= 50; for smaller populations serial
            evaluation avoids thread-spawn overhead.

        Notes
        -----
        After evaluate() returns, _individuals is sorted ascending by
        fitness and _sorted is set to True.
        """
        unevaluated = [ind for ind in self._individuals if not ind.is_evaluated()]

        if unevaluated:
            if n_workers > 1:
                self._evaluate_parallel(unevaluated, n_workers)
            else:
                self._evaluate_serial(unevaluated)

        # Sort after evaluation so best/worst access is O(1)
        self._individuals.sort()
        self._sorted = True

    @staticmethod
    def _evaluate_serial(individuals: List[Individual]) -> None:
        """Evaluate individuals one by one in the calling thread."""
        for ind in individuals:
            ind.evaluate()

    @staticmethod
    def _evaluate_parallel(
        individuals: List[Individual],
        n_workers: int,
    ) -> None:
        """
        Evaluate individuals concurrently using a thread pool.

        Each individual's evaluate() call is submitted as an independent
        task. Results are collected via as_completed() so exceptions
        surface immediately rather than being silently swallowed.

        Parameters
        ----------
        individuals : list of Individual
            Unevaluated individuals to process.
        n_workers : int
            Maximum number of concurrent threads.
        """
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(ind.evaluate): ind for ind in individuals}
            for future in as_completed(futures):
                # Re-raise any exception that occurred during evaluation
                future.result()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def best_fitness(self) -> float:
        """Return the lowest fitness value in the population."""
        return self.best.fitness

    def mean_fitness(self) -> float:
        """Return the mean fitness across all individuals."""
        self._require_evaluated()
        return float(np.mean([ind.fitness for ind in self._individuals]))

    def std_fitness(self) -> float:
        """Return the standard deviation of fitness across all individuals."""
        self._require_evaluated()
        return float(np.std([ind.fitness for ind in self._individuals]))

    def fitness_array(self) -> np.ndarray:
        """
        Return all fitness values as a numpy array (ascending order if sorted).

        Useful for plotting convergence curves in notebooks.

        Returns
        -------
        np.ndarray
            1-D float64 array of length self.size.
        """
        self._require_evaluated()
        return np.array([ind.fitness for ind in self._individuals], dtype=np.float64)

    def stats(self) -> dict:
        """
        Return a dictionary of per-generation statistics.

        Used by the GA engine to populate its generation log, which feeds
        into convergence plots and early-stopping checks.

        Returns
        -------
        dict
            Keys: 'best', 'mean', 'std', 'worst'.
            All values are Python floats.
        """
        self._require_evaluated()
        fitnesses = self.fitness_array()
        return {
            "best":  float(fitnesses.min()),
            "mean":  float(fitnesses.mean()),
            "std":   float(fitnesses.std()),
            "worst": float(fitnesses.max()),
        }

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    def diversity(self) -> float:
        """
        Compute a simple genotypic diversity metric.

        Diversity is defined as the mean pairwise standard deviation of
        fitness values across the population, normalised to [0, 1] by
        dividing by the fitness range. A value near 0 indicates the
        population has converged (all individuals have similar fitness);
        a value near 1 indicates high spread.

        This is a fitness-space proxy for genotypic diversity — computing
        true genotypic diversity (e.g. average Hamming distance between
        chromosomes) would require comparing 100 triangles × 10 parameters
        per individual pair, which is expensive and adds little insight
        for the purposes of convergence monitoring.

        Returns
        -------
        float
            Normalised diversity in [0, 1]. Returns 0.0 if all individuals
            have identical fitness or population size is 1.
        """
        self._require_evaluated()
        fitnesses = self.fitness_array()
        fitness_range = fitnesses.max() - fitnesses.min()
        if fitness_range < 1e-12:
            return 0.0
        return float(fitnesses.std() / fitness_range)

    # ------------------------------------------------------------------
    # Replacement
    # ------------------------------------------------------------------

    def replace(
        self,
        offspring: List[Individual],
        n_elites: int = 1,
    ) -> Population:
        """
        Produce the next generation by replacing the current population
        with offspring, preserving the top n_elites individuals (elitism).

        Elitism guarantees that the best solution found so far is never
        lost due to random variation. Even n_elites=1 (carrying forward
        only the single best individual) significantly stabilises convergence
        by preventing regression.

        Parameters
        ----------
        offspring : list of Individual
            New individuals produced by crossover + mutation. If
            len(offspring) + n_elites > self.size, offspring are truncated.
            If len(offspring) + n_elites < self.size, remaining slots are
            filled from the best of the current generation (steady-state
            fallback — rare in practice).
        n_elites : int
            Number of best individuals from the current generation to
            carry forward unchanged. Default is 1. Setting n_elites=0
            disables elitism entirely (not recommended).

        Returns
        -------
        Population
            A new Population instance representing the next generation.
            The current population is not modified.
        """
        if n_elites < 0:
            raise ValueError(f"n_elites must be >= 0, got {n_elites}.")
        if n_elites > self._size:
            raise ValueError(
                f"n_elites ({n_elites}) cannot exceed population size ({self._size})."
            )

        # Elites are the top n_elites individuals from the current generation
        elites = self.sorted_individuals[:n_elites]

        # Fill remaining slots with offspring, truncating if necessary
        n_offspring_needed = self._size - n_elites
        new_individuals = elites + offspring[:n_offspring_needed]

        # Steady-state fallback: if not enough offspring were provided,
        # fill remaining slots from the current generation's best
        if len(new_individuals) < self._size:
            shortfall = self._size - len(new_individuals)
            fillers = self.sorted_individuals[n_elites: n_elites + shortfall]
            new_individuals = new_individuals + fillers

        return Population(new_individuals)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_evaluated(self) -> None:
        """Raise RuntimeError if any individual has not been evaluated yet."""
        unevaluated_count = sum(
            1 for ind in self._individuals if not ind.is_evaluated()
        )
        if unevaluated_count > 0:
            raise RuntimeError(
                f"{unevaluated_count} individual(s) have not been evaluated. "
                "Call population.evaluate() before accessing fitness-dependent properties."
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        return iter(self._individuals)

    def __getitem__(self, index: int) -> Individual:
        return self._individuals[index]

    def __repr__(self) -> str:
        evaluated = sum(1 for ind in self._individuals if ind.is_evaluated())
        return (
            f"Population(size={self._size}, evaluated={evaluated}/{self._size})"
        )