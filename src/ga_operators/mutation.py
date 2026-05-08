"""
operators/mutation.py
---------------------
Mutation operator hierarchy for the genetic algorithm.

Mutation introduces random variation into offspring chromosomes. It is the
primary source of genetic diversity beyond what crossover can recombine,
and is essential for escaping local optima.

Four concrete strategies are provided:

    1. GaussianMutation     — perturbs vertices and/or color of randomly
                              selected triangles with Gaussian noise.
                              Primary mutation operator; fine-grained local search.
    2. ResetMutation        — replaces randomly selected triangles with
                              entirely new random ones. Coarse exploration;
                              useful for escaping local optima.
    3. SwapMutation         — swaps the draw-order positions of two triangles.
                              Explores the ordering space without changing
                              triangle geometry or color.
    4. CompositeMutation    — applies multiple mutation operators in sequence
                              with configurable probabilities. Recommended for
                              the main GA run as it combines fine and coarse
                              search in a single operator.

Design notes:
    - MutationOperator is an abstract base class. The GA engine calls
      mutate() uniformly regardless of which strategy is active.
    - mutation_rate controls per-gene application probability, not per-
      individual. At mutation_rate=0.05, each of the 100 triangles has
      a 5% chance of being mutated independently — on average 5 triangles
      per individual per generation.
    - All operators return new Individual instances (immutability contract).
    - Sigma parameters for Gaussian noise should be tuned alongside
      mutation_rate: high sigma + high rate = exploration; low sigma +
      low rate = exploitation. The GA engine may implement sigma decay
      (adaptive mutation) over generations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np

from individual import Individual, NUM_TRIANGLES
from triangle import Triangle
from utils import IMG_WIDTH, IMG_HEIGHT


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class MutationOperator(ABC):
    """
    Abstract base class for all mutation strategies.

    Concrete subclasses implement mutate(), which applies random
    perturbations to an individual's chromosome and returns a new
    Individual with the modifications applied.
    """

    @abstractmethod
    def mutate(
        self,
        individual: Individual,
        rng: np.random.Generator,
    ) -> Individual:
        """
        Apply mutation to an individual and return the mutated offspring.

        Parameters
        ----------
        individual : Individual
            The individual to mutate. Never modified in place.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.

        Returns
        -------
        Individual
            A new Individual with mutations applied. If no gene happened
            to be selected for mutation (all per-gene trials failed), the
            returned individual is a shallow copy with the same triangles
            but a fresh (unevaluated) fitness cache.
        """


# ---------------------------------------------------------------------------
# Concrete implementation 1: Gaussian mutation
# ---------------------------------------------------------------------------

class GaussianMutation(MutationOperator):
    """
    Gaussian perturbation of triangle vertices and/or color channels.

    For each triangle in the chromosome, independently decides whether to
    mutate it (Bernoulli trial with probability mutation_rate). Selected
    triangles are perturbed via Triangle.mutate_vertices(),
    Triangle.mutate_color(), or Triangle.mutate() depending on the
    mutate_vertices and mutate_color flags.

    This is the primary mutation operator. It implements local search:
    small sigma values refine an already-good solution; large sigma values
    allow larger jumps when the GA is stuck.

    Parameters
    ----------
    mutation_rate : float
        Per-triangle probability of applying mutation. In [0.0, 1.0].
        Typical range: 0.01 - 0.10. Default is 0.05 (5% per gene).
    vertex_sigma : float
        Std-dev of Gaussian noise applied to vertex coordinates (pixels).
        Default is 15.0.
    color_sigma : float
        Std-dev of Gaussian noise applied to RGBA channels.
        Default is 15.0.
    mutate_vertices : bool
        If True, vertex coordinates are perturbed. Default is True.
    mutate_color : bool
        If True, color channels are perturbed. Default is True.
    """

    def __init__(
        self,
        mutation_rate: float = 0.05,
        vertex_sigma: float = 15.0,
        color_sigma: float = 15.0,
        mutate_vertices: bool = True,
        mutate_color: bool = True,
    ) -> None:
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(
                f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}."
            )
        if not (mutate_vertices or mutate_color):
            raise ValueError(
                "At least one of mutate_vertices or mutate_color must be True."
            )
        self.mutation_rate = mutation_rate
        self.vertex_sigma = vertex_sigma
        self.color_sigma = color_sigma
        self.mutate_vertices = mutate_vertices
        self.mutate_color = mutate_color

    def mutate(
        self,
        individual: Individual,
        rng: np.random.Generator,
    ) -> Individual:
        """
        Apply per-gene Gaussian mutation and return the mutated individual.

        Parameters
        ----------
        individual : Individual
            Source individual (not modified).
        rng : np.random.Generator
            Random generator for Bernoulli trials and Gaussian noise.

        Returns
        -------
        Individual
            New individual with selected genes perturbed.
        """
        triangles = list(individual.triangles)
        # Draw all per-gene mutation decisions at once
        mutate_mask = rng.random(size=NUM_TRIANGLES) < self.mutation_rate

        for i, do_mutate in enumerate(mutate_mask):
            if not do_mutate:
                continue

            tri = triangles[i]

            if self.mutate_vertices and self.mutate_color:
                triangles[i] = tri.mutate(
                    IMG_WIDTH, IMG_HEIGHT, rng,
                    vertex_sigma=self.vertex_sigma,
                    color_sigma=self.color_sigma,
                )
            elif self.mutate_vertices:
                triangles[i] = tri.mutate_vertices(
                    IMG_WIDTH, IMG_HEIGHT, rng, sigma=self.vertex_sigma
                )
            else:
                triangles[i] = tri.mutate_color(rng, sigma=self.color_sigma)

        return individual.copy_with(triangles)

    def __repr__(self) -> str:
        return (
            f"GaussianMutation(rate={self.mutation_rate}, "
            f"v_sigma={self.vertex_sigma}, c_sigma={self.color_sigma})"
        )


# ---------------------------------------------------------------------------
# Concrete implementation 2: Reset mutation
# ---------------------------------------------------------------------------

class ResetMutation(MutationOperator):
    """
    Random gene reset mutation.

    For each triangle, independently decides (Bernoulli trial) whether to
    replace it with an entirely new randomly initialised triangle. This is
    a coarse-grained operator: it does not refine existing triangles but
    instead injects fresh random material into the chromosome.

    Use cases
    ---------
    - Early generations: high reset rate accelerates initial exploration.
    - Stagnation recovery: if the GA is stuck in a local optimum, a burst
      of reset mutations can perturb the population enough to escape.
    - Complementary to GaussianMutation in CompositeMutation: Gaussian
      handles fine-grained local search, reset handles coarse exploration.

    Parameters
    ----------
    mutation_rate : float
        Per-triangle probability of replacement. Default is 0.02.
        Lower than GaussianMutation because resets are destructive —
        they discard all information in the replaced gene.
    """

    def __init__(self, mutation_rate: float = 0.02) -> None:
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(
                f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}."
            )
        self.mutation_rate = mutation_rate

    def mutate(
        self,
        individual: Individual,
        rng: np.random.Generator,
    ) -> Individual:
        """
        Replace selected triangles with random ones.

        Parameters
        ----------
        individual : Individual
            Source individual (not modified).
        rng : np.random.Generator
            Used for Bernoulli trials and generating random triangles.

        Returns
        -------
        Individual
            New individual with selected genes replaced by random triangles.
        """
        triangles = list(individual.triangles)
        reset_mask = rng.random(size=NUM_TRIANGLES) < self.mutation_rate

        for i, do_reset in enumerate(reset_mask):
            if do_reset:
                triangles[i] = Triangle.random(IMG_WIDTH, IMG_HEIGHT, rng)

        return individual.copy_with(triangles)

    def __repr__(self) -> str:
        return f"ResetMutation(rate={self.mutation_rate})"


# ---------------------------------------------------------------------------
# Concrete implementation 3: Swap mutation
# ---------------------------------------------------------------------------

class SwapMutation(MutationOperator):
    """
    Draw-order swap mutation.

    Randomly selects n_swaps pairs of triangles and swaps their positions
    in the draw order. Geometry and color are unchanged — only the layering
    order is modified.

    Motivation
    ----------
    Draw order determines which triangles occlude which. A triangle that is
    currently being covered by others may produce much better results if
    moved to the top of the stack (or vice versa). This operator explores
    the ordering space without discarding any triangle's shape or color,
    making it non-destructive in terms of the genetic material it carries.

    Parameters
    ----------
    mutation_rate : float
        Probability of applying any swap at all to an individual.
        Default is 0.3 (swaps are applied to 30% of individuals).
    n_swaps : int
        Number of pair-swaps to perform when the operator fires.
        Default is 2.
    """

    def __init__(self, mutation_rate: float = 0.3, n_swaps: int = 2) -> None:
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(
                f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}."
            )
        if n_swaps < 1:
            raise ValueError(f"n_swaps must be >= 1, got {n_swaps}.")
        self.mutation_rate = mutation_rate
        self.n_swaps = n_swaps

    def mutate(
        self,
        individual: Individual,
        rng: np.random.Generator,
    ) -> Individual:
        """
        Apply n_swaps random draw-order swaps if the Bernoulli trial succeeds.

        Parameters
        ----------
        individual : Individual
            Source individual (not modified).
        rng : np.random.Generator
            Used for the overall trigger trial and swap index sampling.

        Returns
        -------
        Individual
            New individual with swapped draw order, or a copy of the
            original if the trigger trial did not fire.
        """
        # Per-individual trigger — swap mutation fires at the individual level
        if rng.random() >= self.mutation_rate:
            return individual.copy_with()

        triangles = list(individual.triangles)
        n = len(triangles)

        for _ in range(self.n_swaps):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                triangles[i], triangles[j] = triangles[j], triangles[i]

        return individual.copy_with(triangles)

    def __repr__(self) -> str:
        return f"SwapMutation(rate={self.mutation_rate}, n_swaps={self.n_swaps})"


# ---------------------------------------------------------------------------
# Concrete implementation 4: Composite mutation
# ---------------------------------------------------------------------------

class CompositeMutation(MutationOperator):
    """
    Composite mutation that applies multiple operators in sequence.

    Each registered operator is applied independently to the individual,
    in registration order. The output of operator i is fed as input to
    operator i+1, so effects accumulate.

    This is the recommended operator for the main GA run because it
    combines fine-grained local search (GaussianMutation) with occasional
    coarse perturbation (ResetMutation) and order exploration (SwapMutation)
    in a single unified operator — without requiring the GA engine to manage
    multiple operator calls.

    Example
    -------
    >>> composite = CompositeMutation([
    ...     GaussianMutation(mutation_rate=0.05, vertex_sigma=15.0),
    ...     ResetMutation(mutation_rate=0.01),
    ...     SwapMutation(mutation_rate=0.2, n_swaps=1),
    ... ])

    Parameters
    ----------
    operators : sequence of MutationOperator
        Operators applied in order. Must contain at least one operator.
    """

    def __init__(self, operators: Sequence[MutationOperator]) -> None:
        if not operators:
            raise ValueError("CompositeMutation requires at least one operator.")
        self.operators: List[MutationOperator] = list(operators)

    def mutate(
        self,
        individual: Individual,
        rng: np.random.Generator,
    ) -> Individual:
        """
        Apply all registered operators in sequence.

        Parameters
        ----------
        individual : Individual
            Source individual. Each operator receives the output of the
            previous one, so mutations accumulate across the chain.
        rng : np.random.Generator
            Shared random generator passed to all operators.

        Returns
        -------
        Individual
            Individual after all operators have been applied.
        """
        result = individual
        for operator in self.operators:
            result = operator.mutate(result, rng)
        return result

    def __repr__(self) -> str:
        ops_repr = ", ".join(repr(op) for op in self.operators)
        return f"CompositeMutation([{ops_repr}])"
