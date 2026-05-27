"""
mo_ga/mo_individual.py
----------------------
Multi-objective Individual for the NSGA-II genetic algorithm.

Extends the concept of the original Individual class but replaces the single
scalar fitness with a vector of fitness values — one per objective. Pareto
rank and crowding distance are assigned externally by MOPopulation after
each evaluation.

Design notes:
    - MOIndividual is intentionally immutable, matching the original Individual.
      All genetic operators (crossover, mutation) return new instances.
    - fitness_values stores raw scores from each fitness function. Lower is
      better for all objectives (same minimisation convention as the original).
    - pareto_rank and crowding_distance are mutable attributes assigned by
      MOPopulation.assign_pareto_ranks(). They are NOT part of the chromosome
      and are reset whenever a new population is formed.
    - copy_with() is interface-compatible with the original Individual so that
      all existing crossover and mutation operators work without modification.
    - __lt__ compares by pareto_rank first, then crowding_distance (higher
      crowding = better, more isolated in objective space). This is the NSGA-II
      selection criterion (Deb et al., 2002).

References
----------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist
multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary
Computation, 6(2), 182–197.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from triangle import Triangle
from fitness import FitnessFunction
from ga_utils import render, IMG_WIDTH, IMG_HEIGHT
from individual import NUM_TRIANGLES


class MOIndividual:
    """
    A candidate solution in the multi-objective genetic algorithm.

    Wraps an ordered list of Triangle objects (the chromosome) and stores
    a vector of fitness values, one per objective. Pareto rank and crowding
    distance are assigned externally by MOPopulation after evaluation.

    Parameters
    ----------
    triangles : sequence of Triangle
        Ordered list of exactly NUM_TRIANGLES triangles.
    fitness_fns : list of FitnessFunction
        One fitness function per objective. Each is evaluated independently.
        All use the minimisation convention (lower = better).
    cache_render : bool
        If True, the rendered array is cached after evaluation.

    Attributes
    ----------
    pareto_rank : int
        Pareto dominance rank (1 = non-dominated front, 2 = second front, ...).
        Assigned by MOPopulation.assign_pareto_ranks(). Default inf until set.
    crowding_distance : float
        Crowding distance within the Pareto front. Higher = more isolated =
        preferred when ranks are equal. Assigned by MOPopulation. Default 0.
    """

    __slots__ = (
        "_triangles",
        "_fitness_fns",
        "_fitness_values",
        "_rendered",
        "_cache_render",
        "pareto_rank",
        "crowding_distance",
    )

    def __init__(
        self,
        triangles: Sequence[Triangle],
        fitness_fns: List[FitnessFunction],
        cache_render: bool = False,
    ) -> None:
        if len(triangles) != NUM_TRIANGLES:
            raise ValueError(
                f"MOIndividual must have exactly {NUM_TRIANGLES} triangles, "
                f"got {len(triangles)}."
            )
        if not fitness_fns:
            raise ValueError("At least one fitness function must be provided.")

        self._triangles: Tuple[Triangle, ...] = tuple(triangles)
        self._fitness_fns: List[FitnessFunction] = fitness_fns
        self._cache_render: bool = cache_render

        self._fitness_values: Optional[List[float]] = None
        self._rendered: Optional[np.ndarray] = None

        self.pareto_rank: float = float("inf")
        self.crowding_distance: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def triangles(self) -> Tuple[Triangle, ...]:
        """Immutable tuple of Triangle objects forming the chromosome."""
        return self._triangles

    @property
    def fitness_values(self) -> List[float]:
        """
        Vector of fitness scores, one per objective (lower = better).

        Lazily evaluated on first access. Cached for all subsequent accesses.

        Returns
        -------
        list of float
            [rmse_score, ciede_score] (or however many objectives).
        """
        if self._fitness_values is None:
            self._evaluate()
        return self._fitness_values

    @property
    def fitness(self) -> float:
        """
        Primary fitness scalar for compatibility with existing operators.

        Returns the first objective (RMSE) as the primary fitness value.
        This allows crossover and mutation operators that reference
        individual.fitness to work without modification.

        Returns
        -------
        float
            fitness_values[0] — the first objective score.
        """
        return self.fitness_values[0]

    @property
    def n_objectives(self) -> int:
        """Number of objectives this individual is evaluated on."""
        return len(self._fitness_fns)

    @property
    def rendered(self) -> np.ndarray:
        """Rendered pixel array (H×W×3 uint8). Requires cache_render=True."""
        if self._rendered is None:
            if not self._cache_render:
                raise RuntimeError(
                    "Rendered array not cached. Use cache_render=True."
                )
            self._evaluate()
        return self._rendered

    def is_evaluated(self) -> bool:
        """Return True if fitness_values have been computed and cached."""
        return self._fitness_values is not None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> None:
        """
        Render the image once and evaluate all objectives.

        The image is rendered exactly once per individual — all fitness
        functions share the same rendered array. This keeps evaluation
        cost at O(1 render + k evaluations) rather than O(k renders).
        """
        array = render(self._triangles)
        self._fitness_values = [fn.evaluate(array) for fn in self._fitness_fns]

        if self._cache_render:
            self._rendered = array

    def evaluate(self) -> List[float]:
        """
        Public alias for triggering evaluation explicitly.

        Returns
        -------
        list of float
            Fitness values for all objectives.
        """
        if self._fitness_values is None:
            self._evaluate()
        return self._fitness_values

    # ------------------------------------------------------------------
    # Dominance
    # ------------------------------------------------------------------

    def dominates(self, other: MOIndividual) -> bool:
        """
        Return True if this individual Pareto-dominates other.

        Individual A dominates B if:
            - A is better than or equal to B in ALL objectives, AND
            - A is strictly better than B in AT LEAST ONE objective.

        Parameters
        ----------
        other : MOIndividual
            The individual to compare against.

        Returns
        -------
        bool
            True if self dominates other, False otherwise.
        """
        a = self.fitness_values
        b = other.fitness_values

        at_least_one_better = False
        for ai, bi in zip(a, b):
            if ai > bi:
                return False
            if ai < bi:
                at_least_one_better = True

        return at_least_one_better

    # ------------------------------------------------------------------
    # Factory methods — mirrors original Individual interface
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        fitness_fns: List[FitnessFunction],
        rng: np.random.Generator,
        cache_render: bool = False,
    ) -> MOIndividual:
        """Create an MOIndividual with NUM_TRIANGLES randomly initialised triangles."""
        triangles = [Triangle.random(IMG_WIDTH, IMG_HEIGHT, rng) for _ in range(NUM_TRIANGLES)]
        return cls(triangles, fitness_fns, cache_render=cache_render)

    @classmethod
    def from_image(
        cls,
        fitness_fns: List[FitnessFunction],
        rng: np.random.Generator,
        target: np.ndarray,
        cache_render: bool = False,
    ) -> MOIndividual:
        """Create an MOIndividual with image-seeded triangle colors."""
        triangles = [
            Triangle.from_image(IMG_WIDTH, IMG_HEIGHT, rng, target)
            for _ in range(NUM_TRIANGLES)
        ]
        return cls(triangles, fitness_fns, cache_render=cache_render)

    @classmethod
    def random_quadrant(
        cls,
        fitness_fns: List[FitnessFunction],
        rng: np.random.Generator,
        n_cols: int = 5,
        n_rows: int = 5,
        cache_render: bool = False,
    ) -> MOIndividual:
        """Random colors, vertices constrained per quadrant for guaranteed coverage."""
        n_cells = n_cols * n_rows
        per_cell = NUM_TRIANGLES // n_cells
        remainder = NUM_TRIANGLES % n_cells
        cell_w = IMG_WIDTH / n_cols
        cell_h = IMG_HEIGHT / n_rows
        triangles: List[Triangle] = []
        for row in range(n_rows):
            for col in range(n_cols):
                x0, y0 = col * cell_w, row * cell_h
                x1, y1 = x0 + cell_w, y0 + cell_h
                n = per_cell + (1 if row * n_cols + col < remainder else 0)
                for _ in range(n):
                    xs = rng.uniform(x0, x1, size=3)
                    ys = rng.uniform(y0, y1, size=3)
                    vertices = tuple(zip(xs.tolist(), ys.tolist()))
                    rgba = tuple(rng.integers(0, 256, size=4).tolist())
                    triangles.append(Triangle(vertices=vertices, color=rgba))
        rng.shuffle(triangles)
        return cls(triangles, fitness_fns, cache_render=cache_render)

    def copy_with(
        self,
        triangles: Optional[Sequence[Triangle]] = None,
        cache_render: bool = False,
    ) -> MOIndividual:
        """
        Return a new MOIndividual with optionally replaced triangles.

        Interface-compatible with Individual.copy_with() so all existing
        crossover and mutation operators work without modification.

        Parameters
        ----------
        triangles : sequence of Triangle, optional
            New chromosome. If None, copies the parent's triangles.
        cache_render : bool
            Whether the new individual should cache its rendered array.

        Returns
        -------
        MOIndividual
            New instance sharing the same fitness_fns reference.
            Pareto rank and crowding distance are reset (not inherited).
        """
        new_triangles = triangles if triangles is not None else list(self._triangles)
        return MOIndividual(new_triangles, self._fitness_fns, cache_render=cache_render)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON checkpointing."""
        return {
            "triangles":        [t.to_dict() for t in self._triangles],
            "fitness_values":   self._fitness_values,
            "pareto_rank":      self.pareto_rank if self.pareto_rank != float("inf") else None,
            "crowding_distance": self.crowding_distance,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        fitness_fns: List[FitnessFunction],
        cache_render: bool = False,
    ) -> MOIndividual:
        """Deserialise from a plain dict."""
        from triangle import Triangle
        triangles = [Triangle.from_dict(t) for t in data["triangles"]]
        ind = cls(triangles, fitness_fns, cache_render=cache_render)
        if data.get("fitness_values") is not None:
            ind._fitness_values = [float(v) for v in data["fitness_values"]]
        if data.get("pareto_rank") is not None:
            ind.pareto_rank = int(data["pareto_rank"])
        ind.crowding_distance = float(data.get("crowding_distance", 0.0))
        return ind

    # ------------------------------------------------------------------
    # Comparison — NSGA-II crowded comparison operator
    # ------------------------------------------------------------------

    def __lt__(self, other: MOIndividual) -> bool:
        """
        NSGA-II crowded comparison operator.

        Individual A is preferred over B if:
            1. A has a lower (better) Pareto rank, OR
            2. A and B have the same rank but A has higher crowding distance
               (A is more isolated in objective space — preserves diversity).

        References
        ----------
        Deb et al. (2002), Section III-B.
        """
        if self.pareto_rank != other.pareto_rank:
            return self.pareto_rank < other.pareto_rank
        return self.crowding_distance > other.crowding_distance

    def __le__(self, other: MOIndividual) -> bool:
        return self == other or self < other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MOIndividual):
            return NotImplemented
        return (
            self.pareto_rank == other.pareto_rank
            and self.crowding_distance == other.crowding_distance
        )

    def __repr__(self) -> str:
        if self._fitness_values is not None:
            fv = ", ".join(f"{v:.4f}" for v in self._fitness_values)
            return (
                f"MOIndividual(fitness=[{fv}], "
                f"rank={self.pareto_rank}, crowd={self.crowding_distance:.3f})"
            )
        return f"MOIndividual(not evaluated, rank={self.pareto_rank})"
