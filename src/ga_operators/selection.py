"""
operators/selection.py
----------------------
Selection operator hierarchy for the genetic algorithm.

Selection determines which individuals from the current generation are
chosen as parents for producing the next generation. The pressure applied
by selection directly governs the exploration/exploitation trade-off:
strong selection converges fast but risks premature convergence; weak
selection preserves diversity but slows convergence.

Three concrete strategies are provided:

    1. TournamentSelection  — runs k-way tournaments; standard choice for
                              GAs, robust and parameter-efficient.
    2. RankSelection        — selects based on fitness rank rather than raw
                              value; reduces dominance of very fit individuals.
    3. RouletteSelection    — fitness-proportionate selection; included for
                              completeness and ablation study (challenge 3).

All operators follow the minimisation convention (lower fitness = better),
consistent with Individual.__lt__ and the fitness functions.

Design notes:
    - SelectionOperator is an abstract base class. The GA engine depends
      only on the abstract select() interface, enabling strategy swapping
      without touching the engine.
    - All operators are stateless between calls — rng is the only mutable
      state and is supplied by the caller. This makes operators safe to
      reuse across generations and trivially testable.
    - select() always returns exactly n_parents individuals, drawn with
      replacement, so n_parents can exceed population size.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from individual import Individual


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SelectionOperator(ABC):
    """
    Abstract base class for all selection strategies.

    All concrete subclasses must implement select(), which draws
    n_parents individuals from the population according to the
    strategy's selection pressure logic.
    """

    @abstractmethod
    def select(
        self,
        population: List[Individual],
        n_parents: int,
        rng: np.random.Generator,
    ) -> List[Individual]:
        """
        Select n_parents individuals from the population.

        Parameters
        ----------
        population : list of Individual
            The current generation. All individuals must have been
            evaluated (fitness cached) before calling select().
        n_parents : int
            Number of individuals to select. May exceed len(population).
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.

        Returns
        -------
        list of Individual
            Selected individuals. References, not copies — the GA engine
            is responsible for passing these to crossover/mutation to
            produce new offspring rather than modifying them directly.
        """


# ---------------------------------------------------------------------------
# Concrete implementation 1: Tournament selection
# ---------------------------------------------------------------------------

class TournamentSelection(SelectionOperator):
    """
    K-way tournament selection.

    For each parent slot, randomly samples k individuals from the
    population and returns the one with the lowest fitness (best).
    Tournament size k controls selection pressure:
        k=2  : low pressure, preserves diversity
        k=5  : moderate pressure, good default
        k=10 : high pressure, fast convergence, risks premature convergence

    This is the recommended default for this project because:
        - It does not require fitness scaling (unlike roulette selection).
        - It is robust to outliers — one extremely fit individual does not
          dominate the selection pool.
        - Tournament size is an interpretable, easily tunable parameter.
        - It is O(k) per selection, computationally lightweight.

    Parameters
    ----------
    tournament_size : int
        Number of individuals competing in each tournament. Must be >= 2.
    """

    def __init__(self, tournament_size: int = 5) -> None:
        if tournament_size < 2:
            raise ValueError(
                f"Tournament size must be >= 2, got {tournament_size}."
            )
        self.tournament_size = tournament_size

    def select(
        self,
        population: List[Individual],
        n_parents: int,
        rng: np.random.Generator,
    ) -> List[Individual]:
        """
        Run n_parents independent tournaments and return the winners.

        Parameters
        ----------
        population : list of Individual
            Pool from which tournament participants are drawn with replacement.
        n_parents : int
            Number of parents to select (= number of tournaments to run).
        rng : np.random.Generator
            Random generator for sampling tournament participants.

        Returns
        -------
        list of Individual
            The winner (lowest fitness) of each tournament.
        """
        selected = []
        pop_size = len(population)

        for _ in range(n_parents):
            # Sample k indices with replacement — a single individual can
            # appear multiple times in the same tournament, which is fine
            # and avoids issues when tournament_size > pop_size.
            indices = rng.integers(0, pop_size, size=self.tournament_size)
            participants = [population[i] for i in indices]
            # min() uses Individual.__lt__ which compares by fitness
            winner = min(participants)
            selected.append(winner)

        return selected

    def __repr__(self) -> str:
        return f"TournamentSelection(tournament_size={self.tournament_size})"


# ---------------------------------------------------------------------------
# Concrete implementation 2: Rank selection
# ---------------------------------------------------------------------------

class RankSelection(SelectionOperator):
    """
    Linear rank-based selection.

    Individuals are ranked by fitness (rank 1 = best, rank N = worst).
    Selection probability is assigned linearly based on rank rather than
    raw fitness value:

        P(rank r) = (2 - sp) / N + 2 * (sp - 1) * (N - r) / (N * (N - 1))

    where sp (selection pressure) in [1.0, 2.0] controls the ratio of
    selection probability between the best and worst individual.

    Motivation
    ----------
    Rank selection maintains a consistent selection pressure regardless of
    the fitness distribution. This is particularly valuable in later generations
    when raw fitness values cluster tightly — a scenario where roulette selection
    degenerates to near-uniform sampling and loses all pressure, and where
    tournament selection's behaviour depends heavily on the chosen k. By
    decoupling selection probability from raw fitness magnitude and mapping it
    to rank instead, this operator avoids both the superindividual dominance
    problem of early roulette selection and the late-generation stagnation it
    causes.

    Parameters
    ----------
    selection_pressure : float
        Ratio of best-to-worst selection probability. Must be in (1.0, 2.0].
        sp=2.0 means the best individual is twice as likely to be selected
        as the worst. sp=1.0 degenerates to uniform selection.
    """

    def __init__(self, selection_pressure: float = 1.5) -> None:
        if not (1.0 < selection_pressure <= 2.0):
            raise ValueError(
                f"Selection pressure must be in (1.0, 2.0], got {selection_pressure}."
            )
        self.selection_pressure = selection_pressure

    def select(
        self,
        population: List[Individual],
        n_parents: int,
        rng: np.random.Generator,
    ) -> List[Individual]:
        """
        Assign rank-based probabilities and sample n_parents individuals.

        Parameters
        ----------
        population : list of Individual
            Will be ranked internally; does not need to be pre-sorted.
        n_parents : int
            Number of parents to select (with replacement).
        rng : np.random.Generator
            Random generator for the weighted sampling step.

        Returns
        -------
        list of Individual
            n_parents individuals sampled according to rank probabilities.
        """
        n = len(population)
        sp = self.selection_pressure

        # Sort ascending by fitness so index 0 = best (rank 1)
        ranked = sorted(population)

        # Linear rank probabilities — rank 1 gets highest probability
        # ranks array: rank[i] = i+1 for i in [0, N-1]
        ranks = np.arange(1, n + 1, dtype=np.float64)
        # Probability formula (Baker 1985, linear ranking)
        probs = (2.0 - sp) / n + 2.0 * (sp - 1.0) * (n - ranks) / (n * (n - 1.0))
        probs /= probs.sum()  # normalise to correct floating-point drift

        indices = rng.choice(n, size=n_parents, replace=True, p=probs)
        return [ranked[i] for i in indices]

    def __repr__(self) -> str:
        return f"RankSelection(selection_pressure={self.selection_pressure})"


# ---------------------------------------------------------------------------
# Concrete implementation 3: Roulette (fitness-proportionate) selection
# ---------------------------------------------------------------------------

class RouletteSelection(SelectionOperator):
    """
    Fitness-proportionate (roulette wheel) selection.

    Each individual's selection probability is proportional to its
    relative fitness. Because this project minimises fitness (lower =
    better), raw fitness values are inverted before computing proportions:

        weight_i = 1 / (fitness_i + epsilon)

    where epsilon prevents division by zero for perfect individuals.

    Limitations
    -----------
    - Sensitive to fitness scaling: if one individual is far better than
      all others, it will dominate the selection pool and cause premature
      convergence (the "superindividual" problem).
    - In late generations when fitnesses cluster tightly, selection becomes
      nearly uniform, losing all selection pressure.

    These limitations make roulette selection less robust than tournament
    or rank selection for this problem. It is included primarily for the
    ablation study (challenge 3) to quantify its effect on convergence.

    Parameters
    ----------
    epsilon : float
        Small constant added to fitness before inversion to avoid
        division by zero. Default is 1e-6.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def select(
        self,
        population: List[Individual],
        n_parents: int,
        rng: np.random.Generator,
    ) -> List[Individual]:
        """
        Sample n_parents individuals with probability proportional to 1/fitness.

        Parameters
        ----------
        population : list of Individual
            All individuals must be evaluated before calling this.
        n_parents : int
            Number of parents to sample (with replacement).
        rng : np.random.Generator
            Random generator for the weighted sampling step.

        Returns
        -------
        list of Individual
            n_parents individuals sampled by inverted fitness proportion.
        """
        fitnesses = np.array([ind.fitness for ind in population], dtype=np.float64)
        weights = 1.0 / (fitnesses + self.epsilon)
        probs = weights / weights.sum()

        indices = rng.choice(len(population), size=n_parents, replace=True, p=probs)
        return [population[i] for i in indices]

    def __repr__(self) -> str:
        return f"RouletteSelection(epsilon={self.epsilon})"
