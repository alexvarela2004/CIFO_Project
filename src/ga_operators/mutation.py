"""
operators/mutation.py
---------------------
Mutation operator hierarchy for the genetic algorithm.

Mutation introduces random variation into offspring chromosomes. It is the
primary source of genetic diversity beyond what crossover can recombine,
and is essential for escaping local optima.

Five concrete strategies are provided:

    1. GaussianMutation     — perturbs vertices and/or color of randomly
                              selected triangles with Gaussian noise.
                              Primary mutation operator; fine-grained local search.
    2. CreepMutation        — perturbs vertices and/or color with bounded uniform
                              noise sampled from [-delta, +delta]. Unlike Gaussian,
                              large outlier jumps are structurally impossible.
                              Preferred over GaussianMutation in late generations
                              when fine refinement is the goal.
    3. ResetMutation        — replaces randomly selected triangles with
                              entirely new random ones. Coarse exploration;
                              useful for escaping local optima.
    4. SwapMutation         — swaps the draw-order positions of two triangles.
                              Explores the ordering space without changing
                              triangle geometry or color.
    5. CompositeMutation    — applies multiple mutation operators in sequence
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

    def set_sigma(self, vertex_sigma: float, color_sigma: float) -> None:
        """
        Update the noise sigma values in place.

        Called by SigmaDecayScheduler at the start of each generation to
        implement adaptive mutation strength. Mutating sigma in place avoids
        reconstructing the operator object every generation.

        Parameters
        ----------
        vertex_sigma : float
            New std-dev for vertex coordinate noise.
        color_sigma : float
            New std-dev for color channel noise.
        """
        self.vertex_sigma = vertex_sigma
        self.color_sigma  = color_sigma

    def __repr__(self) -> str:
        return (
            f"GaussianMutation(rate={self.mutation_rate}, "
            f"v_sigma={self.vertex_sigma:.2f}, c_sigma={self.color_sigma:.2f})"
        )


# ---------------------------------------------------------------------------
# Concrete implementation 1b: Creep mutation
# ---------------------------------------------------------------------------

class CreepMutation(MutationOperator):
    """
    Bounded uniform (creep) perturbation of triangle vertices and/or color channels.

    For each triangle in the chromosome, independently decides whether to
    mutate it (Bernoulli trial with probability mutation_rate). Selected
    triangles have their vertex coordinates and/or color channels perturbed
    by a value drawn uniformly from [-delta, +delta].

    Comparison with GaussianMutation
    ---------------------------------
    GaussianMutation samples noise from an unbounded normal distribution.
    Even with a moderate sigma, occasional large outlier jumps occur (e.g.
    a 50-pixel vertex displacement when sigma=15). CreepMutation hard-caps
    every perturbation at delta pixels/units, making large destructive jumps
    structurally impossible. This makes it preferable in late generations
    when triangles are already well-placed and the goal is fine refinement
    rather than broad exploration.

    The two operators can be combined in CompositeMutation: Gaussian for
    early exploration, creep for late-stage polish, or use DeltaDecayScheduler
    to shrink delta over time analogously to SigmaDecayScheduler.

    Parameters
    ----------
    mutation_rate : float
        Per-triangle probability of applying mutation. In [0.0, 1.0].
        Typical range: 0.01 - 0.10. Default is 0.05.
    vertex_delta : float
        Half-width of the uniform noise window for vertex coordinates (pixels).
        Perturbation is drawn from U(-vertex_delta, +vertex_delta).
        Default is 15.0.
    color_delta : float
        Half-width of the uniform noise window for color channels.
        Perturbation is drawn from U(-color_delta, +color_delta).
        Default is 15.0.
    mutate_vertices : bool
        If True, vertex coordinates are perturbed. Default is True.
    mutate_color : bool
        If True, color channels are perturbed. Default is True.
    """

    def __init__(
        self,
        mutation_rate: float = 0.05,
        vertex_delta: float = 15.0,
        color_delta: float = 15.0,
        mutate_vertices: bool = True,
        mutate_color: bool = True,
    ) -> None:
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(
                f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}."
            )
        if vertex_delta <= 0.0:
            raise ValueError(
                f"vertex_delta must be > 0, got {vertex_delta}."
            )
        if color_delta <= 0.0:
            raise ValueError(
                f"color_delta must be > 0, got {color_delta}."
            )
        if not (mutate_vertices or mutate_color):
            raise ValueError(
                "At least one of mutate_vertices or mutate_color must be True."
            )
        self.mutation_rate = mutation_rate
        self.vertex_delta = vertex_delta
        self.color_delta = color_delta
        self.mutate_vertices = mutate_vertices
        self.mutate_color = mutate_color

    def _perturb_vertices(
        self,
        tri: "Triangle",
        rng: np.random.Generator,
    ) -> "Triangle":
        """
        Return a new triangle with vertex coordinates perturbed by uniform noise.

        Each of the 6 coordinate values (x1,y1,x2,y2,x3,y3) is independently
        shifted by a value drawn from U(-vertex_delta, +vertex_delta), then
        clamped to the image bounds.
        """
        # Generate bounded uniform noise for x and y coordinates separately
        noise_x = rng.uniform(-self.vertex_delta, self.vertex_delta, size=3)
        noise_y = rng.uniform(-self.vertex_delta, self.vertex_delta, size=3)

        # Extract current vertex coordinates from the Triangle's vertices tuple
        xs = np.array([v[0] for v in tri.vertices], dtype=float)
        ys = np.array([v[1] for v in tri.vertices], dtype=float)

        # Apply noise and clamp to image bounds
        new_xs = np.clip(xs + noise_x, 0, IMG_WIDTH)
        new_ys = np.clip(ys + noise_y, 0, IMG_HEIGHT)

        # Reconstruct vertices as tuple of (x, y) pairs and return new Triangle
        new_vertices = tuple(zip(new_xs.tolist(), new_ys.tolist()))
        return Triangle(vertices=new_vertices, color=tri.color)

    def _perturb_color(
        self,
        tri: "Triangle",
        rng: np.random.Generator,
    ) -> "Triangle":
        """
        Return a new triangle with color channels perturbed by uniform noise.

        Each channel is independently shifted by U(-color_delta, +color_delta)
        then clamped to [0, 255].
        """
        noise = rng.uniform(-self.color_delta, self.color_delta, size=4)
        rgba = np.clip(np.array(tri.color, dtype=float) + noise, 0, 255).astype(int)
        new_color = tuple(int(c) for c in rgba)
        return Triangle(vertices=tri.vertices, color=new_color)

    def mutate(
        self,
        individual: Individual,
        rng: np.random.Generator,
    ) -> Individual:
        """
        Apply per-gene creep mutation and return the mutated individual.

        Parameters
        ----------
        individual : Individual
            Source individual (not modified).
        rng : np.random.Generator
            Random generator for Bernoulli trials and uniform noise.

        Returns
        -------
        Individual
            New individual with selected genes perturbed by bounded uniform noise.
        """
        triangles = list(individual.triangles)
        mutate_mask = rng.random(size=NUM_TRIANGLES) < self.mutation_rate

        for i, do_mutate in enumerate(mutate_mask):
            if not do_mutate:
                continue

            tri = triangles[i]

            if self.mutate_vertices:
                tri = self._perturb_vertices(tri, rng)
            if self.mutate_color:
                tri = self._perturb_color(tri, rng)

            triangles[i] = tri

        return individual.copy_with(triangles)

    def set_delta(self, vertex_delta: float, color_delta: float) -> None:
        """
        Update the noise delta values in place.

        Called by DeltaDecayScheduler each generation to implement adaptive
        mutation strength. Mutating delta in place avoids reconstructing the
        operator object every generation.

        Parameters
        ----------
        vertex_delta : float
            New half-width for vertex coordinate noise.
        color_delta : float
            New half-width for color channel noise.
        """
        self.vertex_delta = vertex_delta
        self.color_delta = color_delta

    def __repr__(self) -> str:
        return (
            f"CreepMutation(rate={self.mutation_rate}, "
            f"v_delta={self.vertex_delta:.2f}, c_delta={self.color_delta:.2f})"
        )


# ---------------------------------------------------------------------------
# Delta decay scheduler (mirrors SigmaDecayScheduler for CreepMutation)
# ---------------------------------------------------------------------------

class DeltaDecayScheduler:
    """
    Exponential decay schedule for CreepMutation delta parameters.

    Mirrors SigmaDecayScheduler exactly, but targets a CreepMutation instance
    instead of GaussianMutation. Starts with a large delta (broad bounded
    exploration) and anneals it toward a small value (fine refinement).

    The decay formula is identical:

        delta(t) = delta_max * (delta_min / delta_max) ** (t / T)

    At t=0: delta = delta_max. At t=T: delta = delta_min.

    Usage
    -----
        scheduler = DeltaDecayScheduler(
            mutation=creep_mut,
            n_generations=500,
            vertex_delta_max=30.0,
            vertex_delta_min=1.0,
            color_delta_max=30.0,
            color_delta_min=1.0,
        )
        ga.run(target=target_array, callback=scheduler)

    Parameters
    ----------
    mutation : CreepMutation
        The mutation operator whose delta values are updated each generation.
    n_generations : int
        Total number of generations (matches GAConfig.n_generations).
    vertex_delta_max : float
        Starting half-width for vertex coordinate noise. Default 30.0.
    vertex_delta_min : float
        Final half-width for vertex coordinate noise. Default 1.0.
    color_delta_max : float
        Starting half-width for color channel noise. Default 30.0.
    color_delta_min : float
        Final half-width for color channel noise. Default 1.0.
    extra_callback : callable, optional
        Additional callback chained after delta update, with signature
        (generation, best, stats).
    """

    def __init__(
        self,
        mutation: CreepMutation,
        n_generations: int,
        vertex_delta_max: float = 30.0,
        vertex_delta_min: float = 1.0,
        color_delta_max: float  = 30.0,
        color_delta_min: float  = 1.0,
        extra_callback: Optional[callable] = None,
    ) -> None:
        self._mutation       = mutation
        self._n_generations  = n_generations
        self._v_max          = vertex_delta_max
        self._v_min          = vertex_delta_min
        self._c_max          = color_delta_max
        self._c_min          = color_delta_min
        self._extra_callback = extra_callback
        self.delta_log: List[dict] = []

    def __call__(
        self,
        generation: int,
        best: "Individual",
        stats: dict,
    ) -> None:
        """
        Update delta for the current generation and log the values.

        Called automatically by the GA engine at the end of each generation.

        Parameters
        ----------
        generation : int
            Current generation index (0 = initial population).
        best : Individual
            Best individual in the current generation.
        stats : dict
            Generation statistics dict from the GA engine log.
        """
        T = max(self._n_generations, 1)
        t = min(generation, T)

        decay   = t / T
        v_delta = self._v_max * (self._v_min / self._v_max) ** decay
        c_delta = self._c_max * (self._c_min / self._c_max) ** decay

        self._mutation.set_delta(v_delta, c_delta)

        self.delta_log.append({
            "generation":   generation,
            "vertex_delta": round(v_delta, 4),
            "color_delta":  round(c_delta, 4),
        })

        if self._extra_callback is not None:
            self._extra_callback(generation, best, stats)

    def __repr__(self) -> str:
        return (
            f"DeltaDecayScheduler("
            f"v_delta={self._v_max}->{self._v_min}, "
            f"c_delta={self._c_max}->{self._c_min}, "
            f"n_generations={self._n_generations})"
        )


# ---------------------------------------------------------------------------
# Sigma decay scheduler
# ---------------------------------------------------------------------------

class SigmaDecayScheduler:
    """
    Exponential decay schedule for GaussianMutation sigma parameters.

    Motivation
    ----------
    A fixed sigma is a compromise: too large and late-generation refinement
    is noisy; too small and early-generation exploration is insufficient.
    Sigma decay resolves this by starting with large sigma (broad exploration)
    and annealing it exponentially toward a small value (fine refinement) as
    generations progress — analogous to simulated annealing's temperature schedule.

    The decay formula is:

        sigma(t) = sigma_max * (sigma_min / sigma_max) ** (t / T)

    where t is the current generation and T is the total number of generations.
    At t=0: sigma = sigma_max. At t=T: sigma = sigma_min.

    Usage
    -----
    Pass the scheduler as the callback to GeneticAlgorithm.run():

        scheduler = SigmaDecayScheduler(
            mutation=gaussian_mut,
            n_generations=500,
            vertex_sigma_max=40.0,
            vertex_sigma_min=2.0,
            color_sigma_max=40.0,
            color_sigma_min=2.0,
        )
        ga.run(target=target_array, init_strategy='image', callback=scheduler)

    The scheduler logs sigma values alongside generation stats so you can
    plot how sigma evolved over the run.

    Parameters
    ----------
    mutation : GaussianMutation
        The mutation operator whose sigma values are updated each generation.
        Must be the same object passed to GeneticAlgorithm.
    n_generations : int
        Total number of generations (matches GAConfig.n_generations).
    vertex_sigma_max : float
        Starting sigma for vertex coordinates. Default 40.0.
    vertex_sigma_min : float
        Final sigma for vertex coordinates. Default 2.0.
    color_sigma_max : float
        Starting sigma for color channels. Default 40.0.
    color_sigma_min : float
        Final sigma for color channels. Default 2.0.
    extra_callback : callable, optional
        Additional callback to chain after sigma update, with the same
        signature as the GA callback: (generation, best, stats).
    """

    def __init__(
        self,
        mutation: GaussianMutation,
        n_generations: int,
        vertex_sigma_max: float = 40.0,
        vertex_sigma_min: float = 2.0,
        color_sigma_max: float  = 40.0,
        color_sigma_min: float  = 2.0,
        extra_callback: Optional[callable] = None,
    ) -> None:
        self._mutation          = mutation
        self._n_generations     = n_generations
        self._v_max             = vertex_sigma_max
        self._v_min             = vertex_sigma_min
        self._c_max             = color_sigma_max
        self._c_min             = color_sigma_min
        self._extra_callback    = extra_callback
        self.sigma_log: List[dict] = []

    def __call__(
        self,
        generation: int,
        best: "Individual",
        stats: dict,
    ) -> None:
        """
        Update sigma for the current generation and log the values.

        Called automatically by the GA engine at the end of each generation.

        Parameters
        ----------
        generation : int
            Current generation index (0 = initial population).
        best : Individual
            Best individual in the current generation.
        stats : dict
            Generation statistics dict from the GA engine log.
        """
        T = max(self._n_generations, 1)
        t = min(generation, T)

        # Exponential decay: sigma(t) = max * (min/max)^(t/T)
        decay = t / T
        v_sigma = self._v_max * (self._v_min / self._v_max) ** decay
        c_sigma = self._c_max * (self._c_min / self._c_max) ** decay

        self._mutation.set_sigma(v_sigma, c_sigma)

        self.sigma_log.append({
            "generation":    generation,
            "vertex_sigma":  round(v_sigma, 4),
            "color_sigma":   round(c_sigma, 4),
        })

        if self._extra_callback is not None:
            self._extra_callback(generation, best, stats)

    def __repr__(self) -> str:
        return (
            f"SigmaDecayScheduler("
            f"v_sigma={self._v_max}->{self._v_min}, "
            f"c_sigma={self._c_max}->{self._c_min}, "
            f"n_generations={self._n_generations})"
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
    ...     CreepMutation(mutation_rate=0.05, vertex_delta=15.0),
    ...     ResetMutation(mutation_rate=0.01),
    ...     SwapMutation(mutation_rate=0.2, n_swaps=1),
    ... ])

    To benchmark Gaussian vs creep, swap CreepMutation for GaussianMutation
    keeping all other parameters fixed — the only difference is the noise
    distribution shape.

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