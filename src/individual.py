"""
individual.py
-------------
Defines the Individual class, which represents a single candidate solution
(chromosome) in the genetic algorithm.

An Individual is an ordered list of NUM_TRIANGLES Triangle objects. The
draw order is significant: triangle at index 0 is rendered first (bottom
layer) and triangle at index NUM_TRIANGLES-1 is rendered last (top layer,
occludes all others beneath it).

Initialisation strategies
-------------------------
Seven factory methods are provided to seed the initial population with
different structural biases:

    - random()                  -- uniformly random vertices and color.
    - random_semitransparent()  -- random with alpha restricted to a range,
                                   encouraging layering and blending from
                                   the start.
    - random_small()            -- random with vertex spread constrained to
                                   a fraction of the canvas, biasing toward
                                   fine-grained primitives.
    - from_grid_random_color()  -- grid-anchored placement with random color,
                                   guaranteeing spatial coverage without
                                   image-seeded color.
    - random_sorted_alpha()     -- random triangles sorted descending by alpha
                                   so opaque triangles anchor the bottom layers
                                   and transparent ones refine the top.
    - random_quadrant()         -- canvas divided into cells; triangles
                                   distributed proportionally across cells
                                   with vertices constrained to their cell.

Chromosome access helpers
-------------------------
    - get_triangle(index)             -- return triangle at a given draw-order index.
    - with_triangle(index, triangle)  -- return a new Individual with one gene replaced.
    - with_triangles(replacements)    -- return a new Individual with multiple genes
                                        replaced in a single operation.
    - copy_with(triangles)            -- return a new Individual optionally replacing
                                        the full chromosome; used by all genetic
                                        operators to produce offspring.

Serialisation
-------------
    - to_dict() / from_dict()  -- JSON-compatible roundtrip. Fitness is
                                  included if already cached, avoiding
                                  redundant re-evaluation on reload.

Comparison
----------
    __lt__, __le__, __eq__ compare by fitness directly, enabling sorting
    and min() calls on lists of individuals without a key function. The
    GA engine relies on this for selection and elitism.

Design notes
------------
    - Fitness is lazily evaluated and cached. The first call to
      Individual.fitness triggers rendering + evaluation; subsequent calls
      return the cached scalar. This is critical for performance since the
      GA engine frequently accesses fitness for sorting and selection without
      needing to re-evaluate.
    - The cache is invalidated only when a new Individual is created --
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

from typing import List, Optional, Sequence, Tuple

import numpy as np

from triangle import Triangle
from fitness import FitnessFunction
from ga_utils import render, IMG_WIDTH, IMG_HEIGHT

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
            function (RMSE, CIEDE2000 or 1-SSIM).
        """
        if self._fitness is None:
            self._evaluate()
        return self._fitness

    @property
    def rendered(self) -> np.ndarray:
        """
        The rendered pixel array for this individual (H×W×3 uint8).

        Available only when cache_render=True. 
        If evaluation has not yet occurred, it is computed lazily.

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
    # Initialization methods
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
    def random_semitransparent(
        cls,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        alpha_range: Tuple[int, int] = (30, 120),
        cache_render: bool = False,
    ) -> "Individual":
        """
        Create an Individual with randomly initialised semi-transparent triangles.

        Delegates to Triangle.random_semitransparent() for each gene, restricting
        the alpha channel to a bounded range. Fully opaque triangles tend to
        dominate lower layers and prevent them from contributing to the rendered
        image; constraining alpha from the start encourages layering and blending
        without requiring the GA to discover this through evolution.

        Parameters
        ----------
        fitness_fn : FitnessFunction
            Fitness function to attach to this individual.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        alpha_range : tuple of (int, int)
            (min_alpha, max_alpha) for all triangles. Default (30, 120).
        cache_render : bool
            Whether to cache the rendered array after evaluation.

        Returns
        -------
        Individual
            A new Individual with alpha-constrained triangles.
        """
        triangles = [
            Triangle.random_semitransparent(IMG_WIDTH, IMG_HEIGHT, rng, alpha_range=alpha_range)
            for _ in range(NUM_TRIANGLES)
        ]
        return cls(triangles, fitness_fn, cache_render=cache_render)

    @classmethod
    def random_small(
        cls,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        max_size_ratio: float = 0.15,
        cache_render: bool = False,
    ) -> "Individual":
        """
        Create an Individual whose triangles are spatially small.

        Delegates to Triangle.random_small() for each gene, constraining
        vertex spread to a fraction of the canvas dimensions. Unconstrained
        random triangles often cover large canvas regions, which is useful
        for coarse approximation early in evolution but may limit fine-grained
        detail later. This strategy biases the initial population toward
        smaller primitives, potentially accelerating convergence in
        high-detail regions.

        Parameters
        ----------
        fitness_fn : FitnessFunction
            Fitness function to attach to this individual.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        max_size_ratio : float
            Maximum triangle extent as a fraction of canvas dimensions.
            Default 0.15. Must be in (0, 1].
        cache_render : bool
            Whether to cache the rendered array after evaluation.

        Returns
        -------
        Individual
            A new Individual with size-constrained triangles.
        """
        triangles = [
            Triangle.random_small(IMG_WIDTH, IMG_HEIGHT, rng, max_size_ratio=max_size_ratio)
            for _ in range(NUM_TRIANGLES)
        ]
        return cls(triangles, fitness_fn, cache_render=cache_render)

    @classmethod
    def from_grid_random_color(
        cls,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        n_cols: int = 10,
        n_rows: int = 10,
        vertex_noise_sigma: Optional[float] = None,
        cache_render: bool = False,
    ) -> "Individual":
        """
        Create an Individual with grid-anchored triangles and random colours.

        Follows the same spatial partitioning logic as from_grid() — dividing
        the canvas into n_cols × n_rows cells and placing two candidate
        triangles per cell — but assigns fully random RGBA colours rather
        than sampling from the target image. This is designed as a controlled
        variant of from_grid() for ablation: isolating the contribution of
        image-seeded colour initialisation from the contribution of guaranteed
        spatial coverage.

        Parameters
        ----------
        fitness_fn : FitnessFunction
            Fitness function to attach to this individual.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        n_cols : int
            Number of grid columns. Default 10.
        n_rows : int
            Number of grid rows. Default 10.
        vertex_noise_sigma : float or None
            Std-dev of Gaussian noise applied to vertex coordinates.
            If None, defaults to 0.3 × min(cell_width, cell_height).
        cache_render : bool
            Whether to cache the rendered array after evaluation.

        Returns
        -------
        Individual
            A new grid-anchored Individual with random colours.
        """

        cell_w = IMG_WIDTH / n_cols
        cell_h = IMG_HEIGHT / n_rows
        if vertex_noise_sigma is None:
            vertex_noise_sigma = 0.3 * min(cell_w, cell_h)
        candidates: List[Triangle] = []
        for row in range(n_rows):
            for col in range(n_cols):
                x0 = col * cell_w
                y0 = row * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                candidates.append(Triangle.from_grid_random_color(
                    x0, y0, x1, y1, IMG_WIDTH, IMG_HEIGHT, rng,
                    vertex_noise_sigma=vertex_noise_sigma,
                ))
                candidates.append(Triangle.from_grid_random_color(
                    x1, y0, x1, y1, IMG_WIDTH, IMG_HEIGHT, rng,
                    vertex_noise_sigma=vertex_noise_sigma,
                ))
        if len(candidates) >= NUM_TRIANGLES:
            indices = rng.choice(len(candidates), size=NUM_TRIANGLES, replace=False)
            triangles = [candidates[i] for i in indices]
        else:
            triangles = candidates + [
                Triangle.random(IMG_WIDTH, IMG_HEIGHT, rng)
                for _ in range(NUM_TRIANGLES - len(candidates))
            ]
        rng.shuffle(triangles)
        return cls(triangles, fitness_fn, cache_render=cache_render)

    @classmethod
    def random_sorted_alpha(
        cls,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        cache_render: bool = False,
    ) -> "Individual":
        """
        Create an Individual with triangles sorted by alpha in descending order.

        Triangles are initialised randomly (equivalent to random()) and then
        sorted so that the most opaque triangles occupy the lowest draw-order
        indices (bottom layers) and the most transparent occupy the highest
        (top layers). This exploits the semantics of the draw order: opaque
        triangles at the bottom establish broad colour regions, while
        transparent triangles at the top refine and blend without fully
        occluding the layers beneath.

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
            A new randomly initialised Individual with alpha-sorted draw order.
        """
       
        triangles = [Triangle.random(IMG_WIDTH, IMG_HEIGHT, rng) for _ in range(NUM_TRIANGLES)]
        triangles.sort(key=lambda t: t.alpha, reverse=True)
        return cls(triangles, fitness_fn, cache_render=cache_render)

    @classmethod
    def random_quadrant(
        cls,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        n_cols: int = 5,
        n_rows: int = 5,
        cache_render: bool = False,
    ) -> "Individual":
        """
        Create an Individual with triangles spatially distributed across quadrants.

        The canvas is divided into n_cols × n_rows cells and NUM_TRIANGLES
        triangles are distributed proportionally across cells, with each
        triangle's vertices constrained to lie within its assigned cell.
        This guarantees uniform spatial coverage without requiring image
        information, unlike from_grid(). Colours are fully random.

        The distribution ensures every canvas region receives at least
        floor(NUM_TRIANGLES / (n_cols × n_rows)) triangles, with remainder
        triangles allocated to the first cells in row-major order.

        Parameters
        ----------
        fitness_fn : FitnessFunction
            Fitness function to attach to this individual.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        n_cols : int
            Number of grid columns. Default 5.
        n_rows : int
            Number of grid rows. Default 5 -> 25 cells,
            each receiving 4 triangles (100 / 25 = 4).
        cache_render : bool
            Whether to cache the rendered array after evaluation.

        Returns
        -------
        Individual
            A new Individual with quadrant-constrained triangle placement.
        """
        n_cells = n_cols * n_rows
        per_cell = NUM_TRIANGLES // n_cells
        remainder = NUM_TRIANGLES % n_cells
        cell_w = IMG_WIDTH / n_cols
        cell_h = IMG_HEIGHT / n_rows
        triangles: List[Triangle] = []
        for row in range(n_rows):
            for col in range(n_cols):
                x0 = col * cell_w
                y0 = row * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                n = per_cell + (1 if row * n_cols + col < remainder else 0)
                for _ in range(n):
                    xs = rng.uniform(x0, x1, size=3)
                    ys = rng.uniform(y0, y1, size=3)
                    vertices = tuple(zip(xs.tolist(), ys.tolist()))
                    rgba = tuple(rng.integers(0, 256, size=4).tolist())
                    triangles.append(Triangle(vertices=vertices, color=rgba))
        rng.shuffle(triangles)
        return cls(triangles, fitness_fn, cache_render=cache_render)

    # -------------------------------------------------------------------
    # Supporting Utilities
    # -------------------------------------------------------------------

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