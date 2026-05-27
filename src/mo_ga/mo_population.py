"""
mo_ga/mo_population.py
----------------------
Multi-objective Population implementing NSGA-II fast non-dominated sorting
and crowding distance assignment.

This module is the core of the multi-objective extension. It replaces the
fitness-sorted Population from population.py with one that:
    1. Assigns Pareto ranks (non-dominated fronts F1, F2, ...) via
       fast non-dominated sorting (Deb et al., 2002, Algorithm 1).
    2. Assigns crowding distances within each front to preserve diversity.
    3. Implements NSGA-II elitism: the combined parent+offspring population
       is sorted by (rank, -crowding_distance) and truncated to pop_size.

Design notes:
    - assign_pareto_ranks() is called once per generation after evaluation,
      before selection. It modifies individual.pareto_rank and
      individual.crowding_distance in-place.
    - The Pareto front F1 (rank=1) is the set of non-dominated solutions.
      These are the solutions the GA "wants to keep" — they are never
      dominated by any other individual in the population.
    - Crowding distance breaks ties within a front: individuals in sparse
      regions of objective space are preferred to maintain diversity along
      the Pareto front.
    - evaluate() shares the same ThreadPoolExecutor pattern as the original
      Population for optional parallel evaluation.
    - Statistics (best, mean, std) are computed on the first objective
      (RMSE) for logging compatibility with the original GA engine.

References
----------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and
elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on
Evolutionary Computation, 6(2), 182–197.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from mo_ga.mo_individual import MOIndividual
from fitness import FitnessFunction


class MOPopulation:
    """
    A generation of MOIndividual objects with Pareto ranking support.

    Parameters
    ----------
    individuals : list of MOIndividual
        The individuals comprising this generation.
    """

    def __init__(self, individuals: List[MOIndividual]) -> None:
        if not individuals:
            raise ValueError("MOPopulation must contain at least one individual.")
        self._individuals: List[MOIndividual] = list(individuals)
        self._size: int = len(individuals)
        self._ranked: bool = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        size: int,
        fitness_fns: List[FitnessFunction],
        rng: np.random.Generator,
        init_strategy: str = "random",
        target: Optional[np.ndarray] = None,
    ) -> MOPopulation:
        """
        Create an MOPopulation with the given initialisation strategy.

        Parameters
        ----------
        size : int
            Number of individuals. Must be >= 2.
        fitness_fns : list of FitnessFunction
            Objectives shared by all individuals.
        rng : np.random.Generator
            Caller-supplied random generator.
        init_strategy : str
            One of 'random', 'image', 'quadrant'. Default 'random'.
        target : np.ndarray, optional
            Required for 'image' and 'quadrant' strategies.

        Returns
        -------
        MOPopulation
            Unevaluated population of `size` individuals.
        """
        if size < 2:
            raise ValueError(f"Population size must be >= 2, got {size}.")

        if init_strategy == "image":
            if target is None:
                raise ValueError("init_strategy='image' requires target array.")
            individuals = [
                MOIndividual.from_image(fitness_fns, rng, target)
                for _ in range(size)
            ]
        elif init_strategy == "quadrant":
            individuals = [
                MOIndividual.random_quadrant(fitness_fns, rng)
                for _ in range(size)
            ]
        else:  
            individuals = [
                MOIndividual.random(fitness_fns, rng)
                for _ in range(size)
            ]

        return cls(individuals)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def individuals(self) -> List[MOIndividual]:
        """All individuals in this generation."""
        return self._individuals

    @property
    def size(self) -> int:
        """Number of individuals."""
        return self._size

    @property
    def best(self) -> MOIndividual:
        """
        Best individual by NSGA-II crowded comparison (rank=1, highest crowding).

        Returns the non-dominated individual with the highest crowding distance
        among those with rank=1. If ranking has not been assigned yet, returns
        the individual with the lowest first-objective (RMSE) fitness instead.
        """
        self._require_evaluated()
        return min(self._individuals)

    @property
    def pareto_front(self) -> List[MOIndividual]:
        """
        All non-dominated individuals (Pareto rank = 1).

        Returns
        -------
        list of MOIndividual
            The first Pareto front. Empty if ranking not yet assigned.
        """
        return [ind for ind in self._individuals if ind.pareto_rank == 1]

    @property
    def sorted_individuals(self) -> List[MOIndividual]:
        """
        Individuals sorted by NSGA-II crowded comparison operator.

        Sorted ascending by (pareto_rank, -crowding_distance), so index 0
        is always the "best" individual by the NSGA-II criterion.
        """
        return sorted(self._individuals)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_workers: int = 1) -> None:
        """
        Evaluate all unevaluated individuals.

        Each individual renders its image once and evaluates all objectives.
        Supports optional parallel evaluation via ThreadPoolExecutor.

        Parameters
        ----------
        n_workers : int
            Number of threads. Default 1 (serial). Set to > 1 for large
            populations on multi-core machines.
        """
        unevaluated = [ind for ind in self._individuals if not ind.is_evaluated()]
        if not unevaluated:
            return

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(ind.evaluate): ind for ind in unevaluated}
                for future in as_completed(futures):
                    future.result()  # re-raise any exceptions
        else:
            for ind in unevaluated:
                ind.evaluate()

    # ------------------------------------------------------------------
    # NSGA-II: Fast non-dominated sorting
    # ------------------------------------------------------------------

    def assign_pareto_ranks(self) -> None:
        """
        Assign Pareto ranks and crowding distances to all individuals.

        Implements NSGA-II Algorithm 1 (fast non-dominated sort) and
        crowding distance assignment. Must be called after evaluate().

        After this call:
            - individual.pareto_rank = 1 for the non-dominated front,
              2 for the second front, etc.
            - individual.crowding_distance is set for all individuals.

        Time complexity: O(M * N^2) where M = number of objectives,
        N = population size. For N=50 and M=3 this is very fast.
        """
        self._require_evaluated()

        n = len(self._individuals)

        S: List[List[int]] = [[] for _ in range(n)]
        n_dominated: List[int] = [0] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._individuals[i].dominates(self._individuals[j]):
                    S[i].append(j)
                elif self._individuals[j].dominates(self._individuals[i]):
                    n_dominated[i] += 1

        fronts: List[List[int]] = []
        current_front = [i for i in range(n) if n_dominated[i] == 0]
        fronts.append(current_front)

        rank = 1
        while current_front:
            for i in current_front:
                self._individuals[i].pareto_rank = rank

            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)

            rank += 1
            current_front = next_front
            if current_front:
                fronts.append(current_front)

        for front_indices in fronts:
            self._assign_crowding_distance(front_indices)

        self._ranked = True

    def _assign_crowding_distance(self, front_indices: List[int]) -> None:
        """
        Assign crowding distance to all individuals in a Pareto front.

        Crowding distance is the sum of normalised distances to the
        neighbouring individuals in objective space. Boundary individuals
        receive infinite distance (always preserved).

        Parameters
        ----------
        front_indices : list of int
            Indices into self._individuals for this front.
        """
        n = len(front_indices)
        if n == 0:
            return

        
        for i in front_indices:
            self._individuals[i].crowding_distance = 0.0

        
        if n <= 2:
            for i in front_indices:
                self._individuals[i].crowding_distance = float("inf")
            return

        n_obj = self._individuals[0].n_objectives

        for obj_idx in range(n_obj):
            sorted_indices = sorted(
                front_indices,
                key=lambda i: self._individuals[i].fitness_values[obj_idx],
            )

            
            self._individuals[sorted_indices[0]].crowding_distance = float("inf")
            self._individuals[sorted_indices[-1]].crowding_distance = float("inf")

            
            f_min = self._individuals[sorted_indices[0]].fitness_values[obj_idx]
            f_max = self._individuals[sorted_indices[-1]].fitness_values[obj_idx]
            f_range = f_max - f_min if f_max != f_min else 1.0

        
            for k in range(1, n - 1):
                prev_val = self._individuals[sorted_indices[k - 1]].fitness_values[obj_idx]
                next_val = self._individuals[sorted_indices[k + 1]].fitness_values[obj_idx]
                self._individuals[sorted_indices[k]].crowding_distance += (
                    (next_val - prev_val) / f_range
                )

    # ------------------------------------------------------------------
    # NSGA-II elitism: combine parent + offspring, keep best N
    # ------------------------------------------------------------------

    def replace(
        self,
        offspring: List[MOIndividual],
        n_elites: int = 1,
    ) -> MOPopulation:
        """
        NSGA-II replacement: combine current population with offspring,
        assign Pareto ranks to the combined pool, and return the best N.

        This implements the NSGA-II environmental selection step (Algorithm 1,
        lines 11-13 in Deb et al. 2002): the combined population R_t of size
        2N is sorted by (rank, -crowding_distance) and truncated to N.

        Note: n_elites is accepted for interface compatibility with the
        original GA engine but is not used — NSGA-II's rank-based selection
        already provides implicit elitism by always keeping the best front.

        Parameters
        ----------
        offspring : list of MOIndividual
            New individuals from crossover + mutation.
        n_elites : int
            Kept for interface compatibility. Not used in NSGA-II.

        Returns
        -------
        MOPopulation
            New population of size self._size, selected by NSGA-II criterion.
        """
        
        combined = self._individuals + list(offspring)
        combined_pop = MOPopulation(combined)

        
        combined_pop.evaluate()

    
        combined_pop.assign_pareto_ranks()

        selected = combined_pop.sorted_individuals[: self._size]
        return MOPopulation(selected)

    # ------------------------------------------------------------------
    # Statistics — computed on first objective (RMSE) for logging
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, float]:
        """
        Compute per-generation statistics on the first objective (RMSE).

        Returns a dict compatible with the original GA engine's logging format.
        """
        self._require_evaluated()
        first_obj = [ind.fitness_values[0] for ind in self._individuals]
        arr = np.array(first_obj, dtype=np.float64)
        return {
            "best":  float(arr.min()),
            "mean":  float(arr.mean()),
            "std":   float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            "worst": float(arr.max()),
        }

    def pareto_front_stats(self) -> Dict[str, object]:
        """
        Statistics about the Pareto front for detailed logging.

        Returns
        -------
        dict
            Keys: 'front_size', 'obj_means', 'obj_mins', 'obj_maxs'
            where obj_* are lists with one value per objective.
        """
        self._require_evaluated()
        front = self.pareto_front
        if not front:
            return {"front_size": 0, "obj_means": [], "obj_mins": [], "obj_maxs": []}

        n_obj = front[0].n_objectives
        obj_vals = [[ind.fitness_values[i] for ind in front] for i in range(n_obj)]

        return {
            "front_size": len(front),
            "obj_means":  [float(np.mean(v)) for v in obj_vals],
            "obj_mins":   [float(np.min(v))  for v in obj_vals],
            "obj_maxs":   [float(np.max(v))  for v in obj_vals],
        }

    def diversity(self) -> float:
        """
        Phenotypic diversity: variance of first-objective (RMSE) values.

        Returns 0.0 if fewer than 2 individuals are evaluated.
        """
        self._require_evaluated()
        first_obj = [ind.fitness_values[0] for ind in self._individuals]
        arr = np.array(first_obj, dtype=np.float64)
        return float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0

    def diversity_report(self) -> Dict[str, float]:
        """
        Compact diversity report for checkpoint logging.

        Compatible with the original Population.diversity_report() format.
        """
        self._require_evaluated()
        first_obj = [ind.fitness_values[0] for ind in self._individuals]
        arr = np.array(first_obj, dtype=np.float64)
        pheno_var = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0

        return {
            "phenotypic_entropy":  float(-np.sum(arr / arr.sum() * np.log(arr / arr.sum() + 1e-10))),
            "genotypic_entropy":   0.0, 
            "phenotypic_variance": pheno_var,
            "genotypic_variance":  0.0,
        }

    def fitness_array(self) -> np.ndarray:
        """First-objective fitness values as a numpy array."""
        self._require_evaluated()
        return np.array(
            [ind.fitness_values[0] for ind in self._individuals], dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_evaluated(self) -> None:
        """Raise RuntimeError if any individual has not been evaluated."""
        unevaluated = sum(1 for ind in self._individuals if not ind.is_evaluated())
        if unevaluated > 0:
            raise RuntimeError(
                f"{unevaluated} individual(s) not evaluated. "
                "Call evaluate() first."
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        return iter(self._individuals)

    def __getitem__(self, index: int) -> MOIndividual:
        return self._individuals[index]

    def __repr__(self) -> str:
        evaluated = sum(1 for ind in self._individuals if ind.is_evaluated())
        ranked = sum(1 for ind in self._individuals if ind.pareto_rank != float("inf"))
        return (
            f"MOPopulation(size={self._size}, "
            f"evaluated={evaluated}/{self._size}, "
            f"ranked={ranked}/{self._size})"
        )
