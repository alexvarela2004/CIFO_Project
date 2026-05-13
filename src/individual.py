"""
individual.py
-------------
Defines the Individual class, which represents a single candidate solution
(chromosome) in the genetic algorithm.

An Individual is an ordered list of NUM_TRIANGLES Triangle objects. The
draw order is significant: triangle at index 0 is rendered first (bottom
layer) and triangle at index 99 is rendered last (top layer, occludes all
others beneath it).

Design notes:
    - Fitness is lazily evaluated and cached. The first call to
      Individual.fitness triggers rendering + evaluation; subsequent calls
      return the cached scalar. This is critical for performance since the
      GA engine frequently accesses fitness for sorting and selection without
      needing to re-evaluate.
    - The cache is invalidated only when a new Individual is created —
      since Triangle and Individual are both immutable, a cached fitness
      value is always valid for the lifetime of that Individual instance.
    - Individual is intentionally immutable: all genetic operators (crossover,
      mutation) return new Individual instances rather than modifying in place.
      This eliminates a class of subtle bugs where a shared individual gets
      mutated mid-generation.
    - The rendered array is also optionally cached (cache_render=True) for
      cases where the caller needs both the fitness score and the pixel array
      (e.g. for visualisation or SSIM). Disabled by default to keep memory
      usage proportional to population size.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple

import numpy as np

from triangle import Triangle
from fitness import FitnessFunction
from utils import render, IMG_WIDTH, IMG_HEIGHT

# Number of triangles per individual — fixed by the project specification.
NUM_TRIANGLES: int = 100


class Individual:
    """
    A candidate solution in the genetic algorithm.

    Wraps an ordered list of Triangle objects (the chromosome) and provides
    lazy, cached fitness evaluation. All mutation and crossover operators
    produce new Individual instances; this one is never modified after init.

    Parameters
    ----------
    triangles : list of Triangle
        Ordered list of exactly NUM_TRIANGLES triangles. Index 0 is the
        bottom-most layer; index NUM_TRIANGLES-1 is the top-most layer.
    fitness_fn : FitnessFunction
        The fitness function used to evaluate this individual. Stored as a
        reference — not copied — so the same function object is shared across
        all individuals in a population (no redundant target storage).
    cache_render : bool
        If True, the rendered numpy array is cached alongside the fitness
        score. Useful when the caller needs the pixel array for display or
        further processing. Defaults to False to keep memory usage low.

    Attributes
    ----------
    triangles : tuple of Triangle
        Immutable view of the chromosome. Stored as a tuple to prevent
        accidental external mutation of the list.
    """

    __slots__ = (
        "_triangles",
        "_fitness_fn",
        "_fitness",
        "_rendered",
        "_cache_render",
    )

    def __init__(
        self,
        triangles: Sequence[Triangle],
        fitness_fn: FitnessFunction,
        cache_render: bool = False,
    ) -> None:
        if len(triangles) != NUM_TRIANGLES:
            raise ValueError(
                f"An Individual must have exactly {NUM_TRIANGLES} triangles, "
                f"got {len(triangles)}."
            )
        # Store as tuple to enforce immutability at the Individual level.
        # Triangle objects themselves are already frozen dataclasses.
        self._triangles: Tuple[Triangle, ...] = tuple(triangles)
        self._fitness_fn: FitnessFunction = fitness_fn
        self._cache_render: bool = cache_render

        # Lazy cache fields — None until first evaluation
        self._fitness: Optional[float] = None
        self._rendered: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def triangles(self) -> Tuple[Triangle, ...]:
        """Immutable tuple of Triangle objects forming the chromosome."""
        return self._triangles

    @property
    def fitness(self) -> float:
        """
        Fitness score for this individual (lower is better).

        Lazily evaluated on first access: renders the image and calls
        the fitness function. Result is cached for all subsequent accesses.

        Returns
        -------
        float
            Non-negative scalar fitness value as defined by the fitness
            function (RMSE, CIEDE2000, or 1-SSIM).
        """
        if self._fitness is None:
            self._evaluate()
        return self._fitness

    @property
    def rendered(self) -> np.ndarray:
        """
        The rendered pixel array for this individual (H×W×3 uint8).

        Only available if cache_render=True was passed at construction,
        or after fitness has been evaluated with cache_render=True.

        Returns
        -------
        np.ndarray
            H×W×3 uint8 RGB array.

        Raises
        ------
        RuntimeError
            If called when cache_render=False and fitness has not been
            evaluated yet (rendered array was discarded after evaluation).
        """
        if self._rendered is None:
            if not self._cache_render:
                raise RuntimeError(
                    "Rendered array is not cached. Construct Individual with "
                    "cache_render=True to retain the pixel array after evaluation."
                )
            # cache_render=True but evaluate() not called yet
            self._evaluate()
        return self._rendered

    def is_evaluated(self) -> bool:
        """Return True if fitness has already been computed and cached."""
        return self._fitness is not None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> None:
        """
        Render the image and compute fitness. Populates the internal cache.

        This method is called at most once per Individual instance.
        Subsequent accesses to self.fitness return the cached value.
        """
        array = render(self._triangles)

        self._fitness = self._fitness_fn.evaluate(array)

        if self._cache_render:
            self._rendered = array
        # If cache_render=False, array goes out of scope here and is GC'd.
        # This keeps peak memory proportional to population size rather
        # than population_size * image_size.

    def evaluate(self) -> float:
        """
        Public alias for triggering evaluation explicitly.

        Useful when the caller wants to force evaluation before entering
        a tight loop (e.g. parallelised fitness batch in the GA engine).

        Returns
        -------
        float
            Fitness score (same as self.fitness).
        """
        if self._fitness is None:
            self._evaluate()
        return self._fitness

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        cache_render: bool = False,
    ) -> "Individual":
        """
        Create an Individual with NUM_TRIANGLES randomly initialised triangles.

        All vertices and colors are drawn from uniform distributions with
        no reference to the target image. Use from_image() for a smarter
        initialisation that seeds colors from the target.

        Parameters
        ----------
        fitness_fn : FitnessFunction
            Fitness function to attach to this individual.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        cache_render : bool
            Whether to cache the rendered array after evaluation.

        Returns
        -------
        Individual
            A new randomly initialised Individual.
        """
        triangles = [
            Triangle.random(IMG_WIDTH, IMG_HEIGHT, rng)
            for _ in range(NUM_TRIANGLES)
        ]
        return cls(triangles, fitness_fn, cache_render=cache_render)

    @classmethod
    def from_image(
        cls,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        target: np.ndarray,
        cache_render: bool = False,
    ) -> "Individual":
        """
        Create an Individual with NUM_TRIANGLES image-seeded triangles.

        Vertices are random but each triangle's color is sampled from the
        target image at the triangle's centroid. This biases the initial
        population toward the correct color palette from generation 0,
        reducing the number of generations the GA needs to spend on basic
        color discovery before it can start refining shapes.

        Parameters
        ----------
        fitness_fn : FitnessFunction
            Fitness function to attach to this individual.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        target : np.ndarray
            H×W×3 uint8 RGB array of the target image. Passed through to
            Triangle.from_image() for centroid color sampling.
        cache_render : bool
            Whether to cache the rendered array after evaluation.

        Returns
        -------
        Individual
            A new image-seeded Individual.
        """
        triangles = [
            Triangle.from_image(IMG_WIDTH, IMG_HEIGHT, rng, target)
            for _ in range(NUM_TRIANGLES)
        ]
        return cls(triangles, fitness_fn, cache_render=cache_render)

    @classmethod
    def from_grid(
        cls,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        target: np.ndarray,
        n_cols: int = 10,
        n_rows: int = 10,
        vertex_noise_sigma: Optional[float] = None,
        cache_render: bool = False,
    ) -> "Individual":
        """
        Create an Individual with triangles anchored to a canvas grid.

        The canvas is divided into n_cols x n_rows cells. Each cell is
        split into two triangles (upper-left and lower-right), yielding
        n_cols * n_rows * 2 triangles total. If this exceeds NUM_TRIANGLES,
        triangles are sampled without replacement. If it falls short,
        the remaining slots are filled with random triangles.

        Triangle colors are sampled from the target image at each
        triangle's centroid, combining spatial coverage with local color.

        Vertex noise is applied per-individual to ensure population
        diversity — without it, all individuals in the population would
        start with identical chromosome structure.

        Parameters
        ----------
        fitness_fn : FitnessFunction
            Fitness function to attach to this individual.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        target : np.ndarray
            H×W×3 uint8 RGB array of the target image.
        n_cols : int
            Number of grid columns. Default 10.
        n_rows : int
            Number of grid rows. Default 10 -> 10*10*2 = 200 candidate
            triangles, from which NUM_TRIANGLES=100 are sampled.
        vertex_noise_sigma : float or None
            Std-dev of Gaussian noise applied to vertex coordinates.
            If None, defaults to 0.3 * min(cell_width, cell_height),
            which gives reasonable diversity while preserving locality.
        cache_render : bool
            Whether to cache the rendered array after evaluation.

        Returns
        -------
        Individual
            A new grid-anchored Individual.
        """
        cell_w = IMG_WIDTH  / n_cols
        cell_h = IMG_HEIGHT / n_rows

        if vertex_noise_sigma is None:
            vertex_noise_sigma = 0.3 * min(cell_w, cell_h)

        # Build all candidate triangles (2 per cell)
        candidates: List[Triangle] = []
        for row in range(n_rows):
            for col in range(n_cols):
                x0 = col * cell_w
                y0 = row * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h

                # Upper-left triangle: (x0,y0), (x1,y0), (x0,y1)
                candidates.append(Triangle.from_grid(
                    x0, y0, x1, y1, rng, target,
                    vertex_noise_sigma=vertex_noise_sigma,
                ))
                # Lower-right triangle: (x1,y0), (x1,y1), (x0,y1)
                candidates.append(Triangle.from_grid(
                    x1, y0, x1, y1, rng, target,
                    vertex_noise_sigma=vertex_noise_sigma,
                ))

        # Sample exactly NUM_TRIANGLES from the candidate pool
        if len(candidates) >= NUM_TRIANGLES:
            indices = rng.choice(len(candidates), size=NUM_TRIANGLES, replace=False)
            triangles = [candidates[i] for i in indices]
        else:
            # Pad with random triangles if grid produces too few
            triangles = candidates + [
                Triangle.random(IMG_WIDTH, IMG_HEIGHT, rng)
                for _ in range(NUM_TRIANGLES - len(candidates))
            ]

        # Shuffle draw order so no systematic bias in layering
        rng.shuffle(triangles)
        return cls(triangles, fitness_fn, cache_render=cache_render)

    def copy_with(
        self,
        triangles: Optional[Sequence[Triangle]] = None,
        cache_render: bool = False,
    ) -> Individual:
        """
        Return a new Individual with optionally replaced triangles.

        Shares the same fitness_fn reference as the parent. Fitness cache
        is not copied — the new individual must be evaluated independently.

        Parameters
        ----------
        triangles : sequence of Triangle, optional
            New chromosome. If None, copies the parent's triangles (deep copy
            of the list, not the Triangle objects which are immutable).
        cache_render : bool
            Whether the new individual should cache its rendered array.

        Returns
        -------
        Individual
            A new Individual instance (never the same object as self).
        """
        new_triangles = triangles if triangles is not None else list(self._triangles)
        return Individual(new_triangles, self._fitness_fn, cache_render=cache_render)

    # ------------------------------------------------------------------
    # Chromosome access helpers
    # ------------------------------------------------------------------

    def get_triangle(self, index: int) -> Triangle:
        """
        Return the triangle at the given draw-order index.

        Parameters
        ----------
        index : int
            Position in the draw order. 0 = bottom, NUM_TRIANGLES-1 = top.
            Negative indexing is supported.

        Returns
        -------
        Triangle
            The Triangle at the specified index.
        """
        return self._triangles[index]

    def with_triangle(self, index: int, triangle: Triangle) -> Individual:
        """
        Return a new Individual with one triangle replaced.

        Used by fine-grained mutation operators that modify a single gene.

        Parameters
        ----------
        index : int
            Index of the triangle to replace.
        triangle : Triangle
            Replacement triangle.

        Returns
        -------
        Individual
            New Individual with the replacement applied.
        """
        new_triangles = list(self._triangles)
        new_triangles[index] = triangle
        return Individual(new_triangles, self._fitness_fn)

    def with_triangles(
        self,
        replacements: Sequence[Tuple[int, Triangle]],
    ) -> Individual:
        """
        Return a new Individual with multiple triangles replaced at once.

        More efficient than chaining with_triangle() calls because only
        one new Individual (and one tuple copy) is created.

        Parameters
        ----------
        replacements : sequence of (index, Triangle) pairs
            Each pair specifies a position and the replacement triangle.

        Returns
        -------
        Individual
            New Individual with all replacements applied.
        """
        new_triangles = list(self._triangles)
        for idx, tri in replacements:
            new_triangles[idx] = tri
        return Individual(new_triangles, self._fitness_fn)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this Individual to a plain Python dictionary.

        The fitness value is included if already computed, allowing
        checkpoints to be reloaded without re-evaluating. The fitness_fn
        is not serialised (it is re-attached at load time by the caller).

        Returns
        -------
        dict
            Keys: 'triangles' (list of triangle dicts), 'fitness' (float or None).
        """
        return {
            "triangles": [t.to_dict() for t in self._triangles],
            "fitness": self._fitness,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        fitness_fn: FitnessFunction,
        cache_render: bool = False,
    ) -> Individual:
        """
        Deserialise an Individual from a plain Python dictionary.

        Parameters
        ----------
        data : dict
            Dictionary as produced by to_dict().
        fitness_fn : FitnessFunction
            Fitness function to attach to the restored individual.
        cache_render : bool
            Whether the restored individual should cache its rendered array.

        Returns
        -------
        Individual
            Reconstructed Individual. If the dict contains a cached fitness
            value it is restored directly, avoiding a redundant re-evaluation.
        """
        triangles = [Triangle.from_dict(t) for t in data["triangles"]]
        individual = cls(triangles, fitness_fn, cache_render=cache_render)

        # Restore cached fitness if available — avoids re-rendering
        if data.get("fitness") is not None:
            individual._fitness = float(data["fitness"])

        return individual

    # ------------------------------------------------------------------
    # Comparison — enables sorting by fitness directly
    # ------------------------------------------------------------------

    def __lt__(self, other: Individual) -> bool:
        """Less-than comparison by fitness (lower fitness = better)."""
        return self.fitness < other.fitness

    def __le__(self, other: Individual) -> bool:
        return self.fitness <= other.fitness

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.fitness == other.fitness

    def __repr__(self) -> str:
        fitness_str = f"{self._fitness:.4f}" if self._fitness is not None else "not evaluated"
        return f"Individual(triangles={NUM_TRIANGLES}, fitness={fitness_str})"