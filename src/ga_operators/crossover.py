"""
operators/crossover.py
----------------------
Crossover operator hierarchy for the genetic algorithm.

Crossover combines the chromosomes of two parent individuals to produce
one or two offspring. The intent is to preserve and recombine building
blocks (groups of well-placed triangles) discovered by the GA, producing
offspring that inherit beneficial traits from both parents.

Five concrete strategies are provided:

    1. SinglePointCrossover   -- splits the chromosome at one random point
                                 and swaps the tails. 
    2. UniformCrossover       -- each gene independently inherited from either
                                 parent with probability swap_prob. High mixing
                                 rate; likely to destroy positional locality.
    3. KPointCrossover        -- generalisation of single-point with k cuts.
                                 Allows contiguous segments from each parent;
                                 k=2 is two-point crossover.
    4. SegmentShuffleCrossover -- random number of cuts (k_min to k_max) with
                                  independent per-segment parent assignment,
                                  unlike the forced alternation of KPointCrossover.
    5. BlendCrossover         -- BLX-alpha (Eshelman & Schaffer, 1993); samples
                                 offspring vertices and colors from an extended
                                 interval around the two parents. Operates in
                                 continuous space rather than swapping whole genes.

All operators produce exactly two offspring per call, which keeps population
size stable when paired with the GA engine's generational replacement.

Design notes:
    - CrossoverOperator is an abstract base class. The GA engine depends
      only on the abstract cross() interface.
    - Draw order of triangles is meaningful (bottom-to-top rendering), so
      crossover operators that preserve contiguous segments (single-point,
      k-point) are likely to produce more coherent offspring than uniform
      crossover, which breaks all spatial locality.
    - All operators are stateless between calls -- safe to reuse across
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


# ---------------------------------------------------------------------------
# Concrete implementation 4: Segment shuffle crossover
# ---------------------------------------------------------------------------

class SegmentShuffleCrossover(CrossoverOperator):
    """
    Segment shuffle crossover.

    Divides the chromosome into a random number of segments (between
    k_min and k_max cuts) and for each segment independently decides
    (with probability 0.5) whether it comes from parent_a or parent_b.

    Differs from KPointCrossover in two ways:
        1. The number of cuts varies randomly each call, rather than
           being fixed — the operator explores different granularities
           of recombination within a single run.
        2. Segments are assigned independently rather than strictly
           alternating. KPointCrossover always gives exactly half the
           chromosome from each parent; here one parent may dominate
           if the coin flips go that way, which is useful when one
           parent is substantially fitter than the other.

    Parameters
    ----------
    k_min : int
        Minimum number of cut points. Must be >= 1. Default is 1.
    k_max : int
        Maximum number of cut points. Must be >= k_min. Default is 5.
    """

    def __init__(self, k_min: int = 1, k_max: int = 5) -> None:
        if k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {k_min}.")
        if k_max < k_min:
            raise ValueError(f"k_max must be >= k_min, got {k_max}.")
        self.k_min = k_min
        self.k_max = k_max

    def cross(
        self,
        parent_a: Individual,
        parent_b: Individual,
        rng: np.random.Generator,
    ) -> Tuple[Individual, Individual]:
        """
        Apply segment shuffle crossover to produce two offspring.

        Parameters
        ----------
        parent_a, parent_b : Individual
            Parent individuals.
        rng : np.random.Generator
            Used to sample the number of cuts, cut positions, and
            per-segment parent assignments.

        Returns
        -------
        tuple of (Individual, Individual)
            Two offspring. Segments not assigned to parent_a in child_1
            are assigned to parent_b, and vice versa for child_2.
        """
        n = len(parent_a.triangles)

        k = int(rng.integers(self.k_min, self.k_max + 1))
        k = min(k, n - 1)  # guard against k >= n

        cuts = sorted(
            rng.choice(np.arange(1, n), size=k, replace=False).tolist()
        )
        boundaries = [0] + cuts + [n]

        tris_a = list(parent_a.triangles)
        tris_b = list(parent_b.triangles)
        child_tris_1: List[Triangle] = []
        child_tris_2: List[Triangle] = []

        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            # independent coin flip per segment — not forced alternation
            if rng.random() < 0.5:
                child_tris_1.extend(tris_a[s:e])
                child_tris_2.extend(tris_b[s:e])
            else:
                child_tris_1.extend(tris_b[s:e])
                child_tris_2.extend(tris_a[s:e])

        child_1 = parent_a.copy_with(child_tris_1)
        child_2 = parent_b.copy_with(child_tris_2)
        return child_1, child_2

    def __repr__(self) -> str:
        return f"SegmentShuffleCrossover(k_min={self.k_min}, k_max={self.k_max})"


# ---------------------------------------------------------------------------
# Concrete implementation 5: Blend crossover
# ---------------------------------------------------------------------------

class BlendCrossover(CrossoverOperator):
    """
    Blend crossover BLX-alpha (Eshelman & Schaffer, 1993).

    For each position i, creates a new triangle whose vertices and color
    are sampled uniformly from the interval [min - alpha*I, max + alpha*I],
    where min and max are the per-gene minima and maxima across the two
    parents, and I = max - min is the distance between them.

    alpha=0.0 : offspring always lies between the two parents (conservative).
                Equivalent to convex interpolation.
    alpha=0.5 : offspring may extend up to 50% beyond each parent (default
                in the literature; balances exploitation and exploration).

    Parameters
    ----------
    alpha : float
        Extension factor for the sampling interval. Must be >= 0.0.
        Default is 0.5 following Eshelman & Schaffer (1993).
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0.0, got {alpha}.")
        self.alpha = alpha

    def cross(
        self,
        parent_a: Individual,
        parent_b: Individual,
        rng: np.random.Generator,
    ) -> Tuple[Individual, Individual]:
        from ga_utils import IMG_WIDTH, IMG_HEIGHT

        tris_a = list(parent_a.triangles)
        tris_b = list(parent_b.triangles)
        n = len(tris_a)

        child_tris_1: List[Triangle] = []
        child_tris_2: List[Triangle] = []

        for i in range(n):
            t_a, t_b = tris_a[i], tris_b[i]

            verts_a = np.array(t_a.vertices, dtype=np.float32)
            verts_b = np.array(t_b.vertices, dtype=np.float32)

            v_min = np.minimum(verts_a, verts_b)
            v_max = np.maximum(verts_a, verts_b)
            I_v   = v_max - v_min

            low_v  = v_min - self.alpha * I_v
            high_v = v_max + self.alpha * I_v

            verts_1 = rng.uniform(low_v, high_v).astype(np.float32)
            verts_2 = rng.uniform(low_v, high_v).astype(np.float32)

            verts_1[:, 0] = np.clip(verts_1[:, 0], 0, IMG_WIDTH - 1)
            verts_1[:, 1] = np.clip(verts_1[:, 1], 0, IMG_HEIGHT - 1)
            verts_2[:, 0] = np.clip(verts_2[:, 0], 0, IMG_WIDTH - 1)
            verts_2[:, 1] = np.clip(verts_2[:, 1], 0, IMG_HEIGHT - 1)

            # --- color ---
            color_a = np.array(t_a.color, dtype=np.float32)
            color_b = np.array(t_b.color, dtype=np.float32)

            c_min = np.minimum(color_a, color_b)
            c_max = np.maximum(color_a, color_b)
            I_c   = c_max - c_min

            low_c  = c_min - self.alpha * I_c
            high_c = c_max + self.alpha * I_c

            color_1 = tuple(
                int(c) for c in np.clip(rng.uniform(low_c, high_c), 0, 255)
            )
            color_2 = tuple(
                int(c) for c in np.clip(rng.uniform(low_c, high_c), 0, 255)
            )

            child_tris_1.append(Triangle(
                vertices=tuple(map(tuple, verts_1.tolist())),
                color=color_1,
            ))
            child_tris_2.append(Triangle(
                vertices=tuple(map(tuple, verts_2.tolist())),
                color=color_2,
            ))

        return parent_a.copy_with(child_tris_1), parent_b.copy_with(child_tris_2)

    def __repr__(self) -> str:
        return f"BlendCrossover(alpha={self.alpha})"