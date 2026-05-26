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
    ) -> "Population":
        """
        Create a Population of randomly initialised individuals.

        All triangle vertices and colors are drawn from uniform distributions
        with no reference to the target image. Use from_image() or mixed()
        for smarter initialisation strategies.

        Parameters
        ----------
        size : int
            Number of individuals. Typical range: 20-100.
        fitness_fn : FitnessFunction
            Shared fitness function referenced by all individuals.
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

    @classmethod
    def random_semitransparent(
        cls,
        size: int,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        alpha_range: tuple = (30, 120),
    ) -> "Population":
        """
        Create a Population with random semi-transparent triangles.

        Equivalent to random() but with the alpha channel of every triangle
        restricted to alpha_range, preventing fully opaque triangles from
        dominating lower layers. This encourages layering and blending effects
        from generation 0 without requiring the GA to discover the benefit of
        transparency through evolution.

        Parameters
        ----------
        size : int
            Number of individuals. Must be >= 2.
        fitness_fn : FitnessFunction
            Shared fitness function referenced by all individuals.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        alpha_range : tuple of (int, int)
            (min_alpha, max_alpha) for all triangles. Default (30, 120).

        Returns
        -------
        Population
            Unevaluated population of `size` semi-transparent individuals.
        """
        if size < 2:
            raise ValueError(f"Population size must be >= 2, got {size}.")
        individuals = [
            Individual.random_semitransparent(fitness_fn, rng, alpha_range=alpha_range)
            for _ in range(size)
        ]
        return cls(individuals)


    @classmethod
    def random_small(
        cls,
        size: int,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        max_size_ratio: float = 0.15,
    ) -> "Population":
        """
        Create a Population with spatially small random triangles.

        Each triangle's vertices are clustered around a random centre, limiting
        their spatial extent to at most max_size_ratio of the canvas dimensions.
        Unconstrained random triangles often cover large canvas regions, which
        is useful for coarse approximation but limits fine-grained detail. This
        strategy biases the initial population toward smaller primitives.

        Parameters
        ----------
        size : int
            Number of individuals. Must be >= 2.
        fitness_fn : FitnessFunction
            Shared fitness function referenced by all individuals.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        max_size_ratio : float
            Maximum triangle extent as a fraction of canvas dimensions.
            Default 0.15. Must be in (0, 1].

        Returns
        -------
        Population
            Unevaluated population of `size` small-triangle individuals.
        """
        if size < 2:
            raise ValueError(f"Population size must be >= 2, got {size}.")
        individuals = [
            Individual.random_small(fitness_fn, rng, max_size_ratio=max_size_ratio)
            for _ in range(size)
        ]
        return cls(individuals)


    @classmethod
    def from_grid_random_color(
        cls,
        size: int,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        n_cols: int = 10,
        n_rows: int = 10,
        vertex_noise_sigma: Optional[float] = None,
    ) -> "Population":
        """
        Create a Population with grid-anchored triangles and random colours.

        Divides the canvas into an n_cols × n_rows grid and anchors triangles
        to grid cells, guaranteeing full spatial coverage from generation 0.
        Unlike from_grid(), colours are fully random rather than sampled from
        the target image — this serves as a controlled ablation variant to
        isolate the contribution of spatial coverage from image-seeded colour.
        Gaussian vertex noise (default σ = 0.3 × cell_size) is applied per
        individual to ensure population diversity.

        Parameters
        ----------
        size : int
            Number of individuals. Must be >= 2.
        fitness_fn : FitnessFunction
            Shared fitness function referenced by all individuals.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        n_cols : int
            Number of grid columns. Default 10.
        n_rows : int
            Number of grid rows. Default 10.
        vertex_noise_sigma : float or None
            Gaussian noise std-dev for vertex perturbation per individual.
            If None, defaults to 0.3 × min(cell_width, cell_height).

        Returns
        -------
        Population
            Unevaluated grid-initialised population with random colours.
        """
        if size < 2:
            raise ValueError(f"Population size must be >= 2, got {size}.")
        individuals = [
            Individual.from_grid_random_color(
                fitness_fn, rng,
                n_cols=n_cols, n_rows=n_rows,
                vertex_noise_sigma=vertex_noise_sigma,
            )
            for _ in range(size)
        ]
        return cls(individuals)


    @classmethod
    def random_sorted_alpha(
        cls,
        size: int,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
    ) -> "Population":
        """
        Create a Population with triangles sorted by alpha in descending order.

        Triangles are initialised randomly then sorted so the most opaque
        occupy the lowest draw-order indices (bottom layers) and the most
        transparent occupy the highest (top layers). This exploits draw-order
        semantics: opaque triangles at the bottom establish broad colour
        regions, while transparent triangles at the top refine and blend
        without fully occluding the layers beneath.

        Parameters
        ----------
        size : int
            Number of individuals. Must be >= 2.
        fitness_fn : FitnessFunction
            Shared fitness function referenced by all individuals.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.

        Returns
        -------
        Population
            Unevaluated population with alpha-sorted draw order.
        """
        if size < 2:
            raise ValueError(f"Population size must be >= 2, got {size}.")
        individuals = [
            Individual.random_sorted_alpha(fitness_fn, rng)
            for _ in range(size)
        ]
        return cls(individuals)


    @classmethod
    def random_quadrant(
        cls,
        size: int,
        fitness_fn: FitnessFunction,
        rng: np.random.Generator,
        n_cols: int = 5,
        n_rows: int = 5,
    ) -> "Population":
        """
        Create a Population with triangles distributed across canvas quadrants.

        The canvas is divided into n_cols × n_rows cells and triangles are
        distributed proportionally, with each triangle's vertices constrained
        to lie within its assigned cell. This guarantees uniform spatial
        coverage without requiring image information, unlike from_grid_random_color().
        Colours are fully random.

        Parameters
        ----------
        size : int
            Number of individuals. Must be >= 2.
        fitness_fn : FitnessFunction
            Shared fitness function referenced by all individuals.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        n_cols : int
            Number of grid columns. Default 5.
        n_rows : int
            Number of grid rows. Default 5 → 25 cells,
            each receiving 4 triangles (100 / 25 = 4).

        Returns
        -------
        Population
            Unevaluated population with quadrant-constrained placement.
        """
        if size < 2:
            raise ValueError(f"Population size must be >= 2, got {size}.")
        individuals = [
            Individual.random_quadrant(fitness_fn, rng, n_cols=n_cols, n_rows=n_rows)
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

    def _chromosome_matrix(self) -> np.ndarray:
        """
        Build an (m x 1000) float64 matrix where each row is the flattened
        chromosome of one individual.

        Each triangle contributes 10 values: 6 vertex coords (x1,y1,x2,y2,x3,y3)
        and 4 color channels (R,G,B,A). With 100 triangles per individual the
        chromosome is 1000-dimensional.

        Vertex coords are left in pixel space [0, IMG_WIDTH/HEIGHT]. Color
        channels are in [0, 255]. Both axes are on different scales, which
        matters for distance-based metrics (genotypic variance). Callers that
        need scale-invariant distances should normalise before use — this method
        returns raw values so the decision stays at the call site.

        Returns
        -------
        np.ndarray
            Shape (m, 1000), dtype float64.
        """
        rows = []
        for ind in self._individuals:
            gene = []
            for tri in ind.triangles:
                for x, y in tri.vertices:
                    gene.append(float(x))
                    gene.append(float(y))
                gene.extend(float(c) for c in tri.color)
            rows.append(gene)
        return np.array(rows, dtype=np.float64)

    def phenotypic_entropy(self) -> float:
        """
        Phenotypic entropy H(P) over the fitness distribution.

        From the course slides (slide 9):

            H(P) = sum_{j=1}^{N} F_j * log(F_j)

        where N is the number of distinct fitness values in the population
        and F_j is the fraction of individuals that share fitness value j.

        Because fitness values are real-valued floats, exact duplicates are
        rare. Each individual is therefore its own bucket in practice, giving
        maximum entropy almost always. This metric is more meaningful for
        discrete fitness landscapes; it is included here for completeness
        and comparability with the course formulation.

        Returns
        -------
        float
            Entropy value (<= 0 by convention; closer to 0 means lower entropy
            / more convergence). Returns 0.0 for a population of size 1.
        """
        self._require_evaluated()
        fitnesses = self.fitness_array()
        m = len(fitnesses)
        if m <= 1:
            return 0.0

        # Round to 6 decimal places so numerically near-identical fitnesses
        # are treated as the same bucket — avoids spuriously high entropy from
        # floating-point noise on what is effectively the same fitness value.
        rounded = np.round(fitnesses, decimals=6)
        unique, counts = np.unique(rounded, return_counts=True)
        fractions = counts / m  # F_j for each unique value

        # H = sum F_j * log(F_j); log of fractions in (0,1] is <= 0
        h = float(np.sum(fractions * np.log(fractions)))
        return h

    def genotypic_entropy(self) -> float:
        """
        Genotypic entropy H(P) over the chromosome distribution.

        Same formula as phenotypic_entropy() but N is the number of distinct
        genotypes (chromosome strings) and F_j is the fraction of individuals
        sharing a specific genotype.

        For real-valued chromosomes (float vertex coords, int color channels),
        exact genotype duplicates essentially never occur in a healthy
        population. This metric is therefore almost always equal to
        -log(1/m) = log(m) (maximum entropy, all individuals unique).

        It becomes informative only when the population has severely
        converged and multiple individuals are exact copies of each other
        — a sign that mutation rate is too low or elitism too aggressive.

        Returns
        -------
        float
            Entropy value (<= 0). Returns 0.0 for a population of size 1.
        """
        self._require_evaluated()
        m = len(self._individuals)
        if m <= 1:
            return 0.0

        # Represent each chromosome as a rounded tuple for hashing.
        # Vertex coords rounded to 2 decimal places; color channels are
        # already integers so rounding has no effect.
        def _key(ind) -> tuple:
            key = []
            for tri in ind.triangles:
                for coord in tri.vertices:
                    key.append(round(coord[0], 2))
                    key.append(round(coord[1], 2))
                key.extend(tri.color)
            return tuple(key)

        keys = [_key(ind) for ind in self._individuals]
        from collections import Counter
        counts = Counter(keys)
        fractions = np.array(list(counts.values()), dtype=np.float64) / m

        h = float(np.sum(fractions * np.log(fractions)))
        return h

    def phenotypic_variance(self) -> float:
        """
        Phenotypic variance V(P) of the fitness distribution.

        From the course slides (slide 10):

            V(P) = 1/(m-1) * sum_{i=1}^{m} (x_i - x_bar)^2

        where m is the number of individuals, x_i is the fitness of
        individual i and x_bar is the mean fitness of the population.

        This is the standard unbiased sample variance of the fitness values.
        A value near 0 indicates all individuals have similar fitness
        (converged population). A large value indicates high spread.

        Returns
        -------
        float
            Sample variance of fitness values. Returns 0.0 for m <= 1.
        """
        self._require_evaluated()
        fitnesses = self.fitness_array()
        m = len(fitnesses)
        if m <= 1:
            return 0.0
        # ddof=1 -> divides by (m-1) matching the course formula
        return float(np.var(fitnesses, ddof=1))

    def genotypic_variance(self) -> float:
        """
        Genotypic variance V(P) of the chromosome distribution.

        From the course slides (slide 10):

            V(P) = 1/(m-1) * sum_{i=1}^{m} (x_i - x_bar)^2

        where x_i is the L2 distance from individual i's chromosome to
        the best individual's chromosome, and x_bar is the mean of those
        distances across the population.

        The best individual is used as the origin (the slide notes that
        any individual can serve as origin; the best is the most
        informative choice because it measures how spread out the
        population is around the current solution).

        Chromosome vectors are normalised before computing distances so
        that vertex coordinates (range ~[0, 400]) and color channels
        (range [0, 255]) contribute on comparable scales:
            - vertex coords divided by max(IMG_WIDTH, IMG_HEIGHT)
            - color channels divided by 255

        Returns
        -------
        float
            Sample variance of distances to the best individual.
            Returns 0.0 for m <= 1 or a population of identical individuals.
        """
        self._require_evaluated()
        m = len(self._individuals)
        if m <= 1:
            return 0.0

        from ga_utils import IMG_WIDTH, IMG_HEIGHT
        coord_scale = float(max(IMG_WIDTH, IMG_HEIGHT))  # normalise vertex coords
        color_scale = 255.0                              # normalise color channels

        # Build chromosome matrix: shape (m, 1000)
        mat = self._chromosome_matrix()

        # Build normalisation vector: first 6 values per triangle are coords,
        # last 4 are color — repeat for all 100 triangles.
        # Pattern per triangle: [x, y, x, y, x, y, R, G, B, A] -> 10 values
        scales_per_tri = (
            [coord_scale, coord_scale] * 3  # 6 vertex coords
            + [color_scale] * 4             # 4 color channels
        )
        scale_vec = np.array(scales_per_tri * 100, dtype=np.float64)  # 1000-dim
        mat_norm = mat / scale_vec

        # Origin = best individual (index 0 after sort, population is kept sorted)
        origin = mat_norm[0]  # best individual is always at index 0 after evaluate()

        # L2 distance from each individual to the origin
        diffs = mat_norm - origin                    # (m, 1000)
        distances = np.sqrt((diffs ** 2).sum(axis=1))  # (m,)

        # Sample variance of distances
        if m <= 1:
            return 0.0
        return float(np.var(distances, ddof=1))

    def diversity(self) -> float:
        """
        Composite diversity metric returned by the GA engine logger.

        Returns phenotypic variance (sample variance of fitness values),
        matching the course slide 10 formula for V(P) with phenotypic
        interpretation. Logged every generation as the primary diversity
        signal.

        For full diversity analysis use the dedicated methods:
            - phenotypic_entropy()
            - genotypic_entropy()
            - phenotypic_variance()
            - genotypic_variance()

        Returns
        -------
        float
            Sample variance of fitness values across the population.
        """
        return self.phenotypic_variance()

    def diversity_report(self) -> dict:
        """
        Compute all four diversity metrics and return them as a dict.

        Intended for periodic logging (e.g. every 50 generations) rather
        than every generation, since genotypic_variance() builds a full
        chromosome matrix and is more expensive than the fitness-only metrics.

        Returns
        -------
        dict
            Keys: 'phenotypic_entropy', 'genotypic_entropy',
                  'phenotypic_variance', 'genotypic_variance'.
            All values are Python floats.
        """
        return {
            "phenotypic_entropy":  self.phenotypic_entropy(),
            "genotypic_entropy":   self.genotypic_entropy(),
            "phenotypic_variance": self.phenotypic_variance(),
            "genotypic_variance":  self.genotypic_variance(),
        }

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