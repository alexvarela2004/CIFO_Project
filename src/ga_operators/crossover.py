"""
operators/crossover.py
----------------------
Crossover operator hierarchy for the genetic algorithm.

Crossover combines the chromosomes of two parent individuals to produce
one or two offspring. The intent is to preserve and recombine building
blocks (groups of well-placed triangles) discovered by the GA, producing
offspring that inherit beneficial traits from both parents.

Three concrete strategies are provided:

    1. SinglePointCrossover  — splits the chromosome at one random point
                               and swaps the tails. Simple and effective.
    2. UniformCrossover      — each gene independently inherited from either
                               parent with probability 0.5. High mixing rate.
    3. KPointCrossover       — generalisation of single-point with k cuts.
                               Allows contiguous segments from each parent.

All operators produce exactly two offspring per call (both orderings of
the split), which keeps population size stable when paired with the GA
engine's generational replacement.

Design notes:
    - CrossoverOperator is an abstract base class. The GA engine depends
      only on the abstract cross() interface.
    - Draw order of triangles is meaningful (bottom-to-top rendering), so
      crossover operators that preserve contiguous segments (single-point,
      k-point) are likely to produce more coherent offspring than uniform
      crossover, which breaks all spatial locality. This is a justifiable
      design choice for the report.
    - All operators are stateless between calls — safe to reuse across
      generations without resetting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from individual import Individual
from triangle import Triangle


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class CrossoverOperator(ABC):
    """
    Abstract base class for all crossover strategies.

    Concrete subclasses implement cross(), which combines two parent
    individuals and returns a pair of offspring.
    """

    @abstractmethod
    def cross(
        self,
        parent_a: Individual,
        parent_b: Individual,
        rng: np.random.Generator,
    ) -> Tuple[Individual, Individual]:
        """
        Combine two parents to produce two offspring.

        Parameters
        ----------
        parent_a, parent_b : Individual
            Parent individuals selected from the current generation.
            Neither parent is modified.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.

        Returns
        -------
        tuple of (Individual, Individual)
            Two offspring. Both share parent_a's fitness_fn reference.
        """


# ---------------------------------------------------------------------------
# Concrete implementation 1: Single-point crossover
# ---------------------------------------------------------------------------

class SinglePointCrossover(CrossoverOperator):
    """
    Single-point crossover.

    A random cut point is chosen uniformly in [1, NUM_TRIANGLES-1].
    Offspring A receives parent_a's triangles up to the cut point and
    parent_b's triangles from the cut point onward. Offspring B receives
    the complementary segments.

        parent_a: [A0, A1, A2 | A3, A4]
        parent_b: [B0, B1, B2 | B3, B4]
                          cut = 3
        child_1:  [A0, A1, A2 | B3, B4]
        child_2:  [B0, B1, B2 | A3, A4]

    Motivation
    ----------
    Preserves contiguous blocks of triangles from each parent. Since
    triangles interact visually (draw order, occlusion), keeping contiguous
    segments intact is more likely to preserve coherent visual regions than
    operators that shuffle genes independently.
    """

    def cross(
        self,
        parent_a: Individual,
        parent_b: Individual,
        rng: np.random.Generator,
    ) -> Tuple[Individual, Individual]:
        """
        Apply single-point crossover to produce two offspring.

        Parameters
        ----------
        parent_a, parent_b : Individual
            Parent individuals.
        rng : np.random.Generator
            Used to sample the cut point uniformly in [1, NUM_TRIANGLES-1].

        Returns
        -------
        tuple of (Individual, Individual)
            child_1 = parent_a[:cut] + parent_b[cut:]
            child_2 = parent_b[:cut] + parent_a[cut:]
        """
        n = len(parent_a.triangles)
        cut = int(rng.integers(1, n))  # cut in [1, n-1] — never trivial split

        tris_a = list(parent_a.triangles)
        tris_b = list(parent_b.triangles)

        child_tris_1 = tris_a[:cut] + tris_b[cut:]
        child_tris_2 = tris_b[:cut] + tris_a[cut:]

        child_1 = parent_a.copy_with(child_tris_1)
        child_2 = parent_a.copy_with(child_tris_2)
        return child_1, child_2

    def __repr__(self) -> str:
        return "SinglePointCrossover()"


# ---------------------------------------------------------------------------
# Concrete implementation 2: Uniform crossover
# ---------------------------------------------------------------------------

class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover.

    Each gene (triangle) is independently assigned to child_1 from either
    parent_a (with probability swap_prob) or parent_b (with probability
    1 - swap_prob). Child_2 receives the complementary assignment.

        for each gene i:
            if rng.random() < swap_prob:
                child_1[i] = parent_b[i], child_2[i] = parent_a[i]
            else:
                child_1[i] = parent_a[i], child_2[i] = parent_b[i]

    With swap_prob=0.5, each child is an equal mix of both parents.

    Trade-offs
    ----------
    - Highest mixing rate of all crossover operators — good for diversity.
    - Destroys all positional locality: a triangle at index 10 from parent_a
      may end up adjacent to a triangle from parent_b that was originally
      at index 90. For image approximation where draw order affects occlusion,
      this may disrupt visually coherent regions.
    - Worth including in the ablation study (challenge 3) to empirically
      measure whether locality-preserving crossover outperforms uniform.

    Parameters
    ----------
    swap_prob : float
        Per-gene probability of swapping from parent_b into child_1.
        Must be in (0.0, 1.0). Default is 0.5 (equal mixing).
    """

    def __init__(self, swap_prob: float = 0.5) -> None:
        if not (0.0 < swap_prob < 1.0):
            raise ValueError(
                f"swap_prob must be in (0.0, 1.0), got {swap_prob}."
            )
        self.swap_prob = swap_prob

    def cross(
        self,
        parent_a: Individual,
        parent_b: Individual,
        rng: np.random.Generator,
    ) -> Tuple[Individual, Individual]:
        """
        Apply uniform crossover to produce two complementary offspring.

        Parameters
        ----------
        parent_a, parent_b : Individual
            Parent individuals.
        rng : np.random.Generator
            Used to draw per-gene swap decisions.

        Returns
        -------
        tuple of (Individual, Individual)
            Two offspring with complementary gene assignments.
        """
        tris_a = list(parent_a.triangles)
        tris_b = list(parent_b.triangles)
        n = len(tris_a)

        # Draw all swap decisions at once — faster than n individual calls
        swap_mask = rng.random(size=n) < self.swap_prob

        child_tris_1: List[Triangle] = []
        child_tris_2: List[Triangle] = []

        for i, swap in enumerate(swap_mask):
            if swap:
                child_tris_1.append(tris_b[i])
                child_tris_2.append(tris_a[i])
            else:
                child_tris_1.append(tris_a[i])
                child_tris_2.append(tris_b[i])

        child_1 = parent_a.copy_with(child_tris_1)
        child_2 = parent_a.copy_with(child_tris_2)
        return child_1, child_2

    def __repr__(self) -> str:
        return f"UniformCrossover(swap_prob={self.swap_prob})"


# ---------------------------------------------------------------------------
# Concrete implementation 3: K-point crossover
# ---------------------------------------------------------------------------

class KPointCrossover(CrossoverOperator):
    """
    K-point crossover.

    Generalises single-point crossover by choosing k distinct cut points,
    splitting the chromosome into k+1 alternating segments. Odd-indexed
    segments are swapped between parents.

        k=1  -> equivalent to SinglePointCrossover
        k=2  -> two-point crossover (common in GA literature)
        k>2  -> progressively approaches uniform crossover behaviour

    Two-point crossover (k=2) is often preferred over single-point because
    it allows a middle segment to be inherited intact from one parent while
    both flanking segments come from the other — useful when important
    structural regions of the image cluster in the middle of the draw order.

    Parameters
    ----------
    k : int
        Number of crossover points. Must be in [1, NUM_TRIANGLES-1].
        Default is 2 (two-point crossover).
    """

    def __init__(self, k: int = 2) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        self.k = k

    def cross(
        self,
        parent_a: Individual,
        parent_b: Individual,
        rng: np.random.Generator,
    ) -> Tuple[Individual, Individual]:
        """
        Apply k-point crossover to produce two offspring.

        Parameters
        ----------
        parent_a, parent_b : Individual
            Parent individuals.
        rng : np.random.Generator
            Used to sample k distinct cut points.

        Returns
        -------
        tuple of (Individual, Individual)
            Two offspring produced by alternating segment exchange.
        """
        n = len(parent_a.triangles)
        k_actual = min(self.k, n - 1)  # guard against k >= n

        # Sample k distinct cut points and sort them
        cut_points = sorted(
            rng.choice(np.arange(1, n), size=k_actual, replace=False).tolist()
        )
        # Add sentinels for clean segment iteration
        boundaries = [0] + cut_points + [n]

        tris_a = list(parent_a.triangles)
        tris_b = list(parent_b.triangles)

        child_tris_1: List[Triangle] = []
        child_tris_2: List[Triangle] = []

        for seg_idx in range(len(boundaries) - 1):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1]
            # Even-indexed segments: child_1 gets parent_a, child_2 gets parent_b
            # Odd-indexed segments: swapped
            if seg_idx % 2 == 0:
                child_tris_1.extend(tris_a[start:end])
                child_tris_2.extend(tris_b[start:end])
            else:
                child_tris_1.extend(tris_b[start:end])
                child_tris_2.extend(tris_a[start:end])

        child_1 = parent_a.copy_with(child_tris_1)
        child_2 = parent_a.copy_with(child_tris_2)
        return child_1, child_2

    def __repr__(self) -> str:
        return f"KPointCrossover(k={self.k})"
