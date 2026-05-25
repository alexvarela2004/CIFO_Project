"""
triangle.py
-----------
Defines the Triangle class, the atomic unit (gene) of the genetic algorithm.

Each Triangle encodes:
    - Three (x, y) vertices defining its shape and position on the canvas.
    - An RGBA color (R, G, B, A) where alpha controls transparency/blending.

Factory methods:
    - random()                  -- uniformly random vertices and color.
    - random_semitransparent()  -- random with alpha restricted to a range.
    - random_small()            -- random confined to a fraction of the canvas.
    - from_grid_random_color()  -- anchored to a grid cell with random color.

Mutation helpers:
    - mutate_vertices()  -- Gaussian perturbation of vertex coordinates.
    - mutate_color()     -- Gaussian perturbation of RGBA channels.
    - mutate()           -- convenience method combining both.

Geometry helpers:
    - area()          -- signed area via shoelace formula.
    - bounding_box()  -- axis-aligned bounding box.
    - is_degenerate() -- check for negligible-area triangles.

Serialisation:
    - to_dict() / from_dict()  -- JSON-compatible roundtrip.

Design notes:
    - Immutability is enforced via __slots__ and properties: once created,
      a Triangle's state cannot be mutated. Genetic operators always produce
      new Triangle instances, which makes reasoning about state straightforward
      and avoids accidental side effects.
    - The class is deliberately free of any fitness or population logic —
      it is purely a data container with rendering capability.
    - Numpy is used for all numerical operations for performance.
"""


from __future__ import annotations

import numpy as np
import numbers
from dataclasses import dataclass, field
from typing import Tuple

# ---------------------------------------------------------------------------
# Type aliases — kept local to this module for clarity
# ---------------------------------------------------------------------------
Vertex = Tuple[float, float]   # (x, y) in canvas pixel coordinates
RGBAColor = Tuple[int, int, int, int]  # (R, G, B, A) each in [0, 255]


@dataclass(frozen=True, slots= True)
class Triangle:
    """
    Immutable representation of a single triangle gene.

    Attributes
    ----------
    vertices : tuple of three (x, y) pairs
        Pixel coordinates of the three corners. Values are floats so that
        genetic operators (e.g. Gaussian mutation) can work in continuous
        space; they are rounded only at render time.
    color : tuple (R, G, B, A)
        Fill color with alpha channel. Alpha controls how this triangle
        blends with the layers beneath it, which is key for approximating
        smooth gradients in the target image.

    Canvas convention
    -----------------
    Origin (0, 0) is the top-left corner.
    x increases rightward (max = IMAGE_WIDTH - 1).
    y increases downward  (max = IMAGE_HEIGHT - 1).
    """

    vertices: Tuple[Vertex, Vertex, Vertex]
    color: RGBAColor

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate vertex and color ranges at construction time."""
        if len(self.vertices) != 3:
            raise ValueError(
                f"A triangle must have exactly 3 vertices, got {len(self.vertices)}."
            )

        for i, (x, y) in enumerate(self.vertices):
            if not (isinstance(x, numbers.Real) and isinstance(y, numbers.Real)):
                raise TypeError(
                    f"Vertex {i} coordinates must be numeric, got ({type(x)}, {type(y)})."
                )
        r, g, b, a = self.color
        for channel_name, value in zip("RGBA", (r, g, b, a)):
            if not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f"Color channel {channel_name} must be an integer, got {type(value)}."
                )
            if not (0 <= value <= 255):
                raise ValueError(
                    f"Color channel {channel_name} must be in [0, 255], got {value}."
                )
    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def xs(self) -> np.ndarray:
        """X-coordinates of the three vertices as a numpy array."""
        return np.array([v[0] for v in self.vertices], dtype=np.float32)

    @property
    def ys(self) -> np.ndarray:
        """Y-coordinates of the three vertices as a numpy array."""
        return np.array([v[1] for v in self.vertices], dtype=np.float32)

    @property
    def rgb(self) -> Tuple[int, int, int]:
        """RGB components of the color (alpha excluded)."""
        return self.color[:3]

    @property
    def alpha(self) -> int:
        """Alpha component of the color."""
        return self.color[3]

    @property
    def alpha_float(self) -> float:
        """Alpha normalised to [0.0, 1.0] for blending calculations."""
        return self.alpha / 255.0

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def area(self) -> float:
        """
        Compute the signed area of the triangle using the shoelace formula.

        Returns
        -------
        float
            Absolute area in square pixels. Returns 0.0 for degenerate
            (collinear) triangles.
        """
        (x1, y1), (x2, y2), (x3, y3) = self.vertices
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

    def is_degenerate(self, min_area: float = 1.0) -> bool:
        """
        Return True if the triangle has negligible area.

        Degenerate triangles contribute nothing to the rendered image and
        waste representational capacity in the chromosome. Mutation
        operators may use this check to avoid producing useless genes.

        Parameters
        ----------
        min_area : float
            Threshold below which a triangle is considered degenerate.
            Default is 1.0 square pixel.
        """
        return self.area() < min_area

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Return the axis-aligned bounding box of the triangle.

        Returns
        -------
        (x_min, y_min, x_max, y_max) : tuple of int
            Pixel-space bounding box, inclusive.
        """
        x_min = int(np.floor(self.xs.min()))
        x_max = int(np.ceil(self.xs.max()))
        y_min = int(np.floor(self.ys.min()))
        y_max = int(np.ceil(self.ys.max()))
        return x_min, y_min, x_max, y_max

    # ------------------------------------------------------------------
    # Factory / mutation helpers
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        img_width: int,
        img_height: int,
        rng: np.random.Generator,
    ) -> Triangle:
        """
        Create a Triangle with uniformly random vertices and color.

        Parameters
        ----------
        img_width : int
            Canvas width in pixels (x range: [0, img_width)).
        img_height : int
            Canvas height in pixels (y range: [0, img_height)).
        rng : np.random.Generator
            Caller-supplied random generator, enabling reproducibility
            and avoiding global state.

        Returns
        -------
        Triangle
            A new randomly initialised Triangle instance.
        """
        xs = rng.uniform(0, img_width, size=3)
        ys = rng.uniform(0, img_height, size=3)
        vertices = tuple(zip(xs.tolist(), ys.tolist()))

        rgba = tuple(rng.integers(0, 256, size=4).tolist())

        return cls(vertices=vertices, color=rgba)

    @classmethod
    def random_semitransparent(
        cls,
        img_width: int,
        img_height: int,
        rng: np.random.Generator,
        alpha_range: Tuple[int, int] = (30, 120),
    ) -> "Triangle":
        """
        Create a Triangle with random vertices and color, restricted to a
        semi-transparent alpha range.

        Motivation
        ----------
        Fully opaque triangles (alpha = 255) dominate lower layers and prevent
        them from contributing to the rendered image. This factory enforces a
        semi-transparent alpha range from the start, encouraging layering and
        blending effects without requiring the GA to discover this constraint
        through evolution.

        Parameters
        ----------
        img_width : int
            Canvas width in pixels.
        img_height : int
            Canvas height in pixels.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        alpha_range : tuple of (int, int)
            (min_alpha, max_alpha) for the alpha channel.
            Default (30, 120) keeps triangles clearly semi-transparent.

        Returns
        -------
        Triangle
            A new randomly initialised Triangle with constrained alpha.
        """
        
        xs = rng.uniform(0, img_width, size=3)
        ys = rng.uniform(0, img_height, size=3)
        vertices = tuple(zip(xs.tolist(), ys.tolist()))
        r, g, b = rng.integers(0, 256, size=3).tolist()
        alpha = int(rng.integers(alpha_range[0], alpha_range[1] + 1))
        return cls(vertices=vertices, color=(r, g, b, alpha))

    @classmethod
    def random_small(
        cls,
        img_width: int,
        img_height: int,
        rng: np.random.Generator,
        max_size_ratio: float = 0.15,
    ) -> "Triangle":
        """
        Create a Triangle with random color whose vertices are clustered
        around a random center, bounding its maximum size.

        Motivation
        ----------
        Unconstrained random triangles often cover large portions of the
        canvas, which is useful early in evolution for coarse approximation
        but limits fine-grained detail later. This factory biases initialisation
        toward smaller triangles (at most 15% of canvas width/height by default),
        which may be preferable when targeting regions of high spatial detail.

        Parameters
        ----------
        img_width : int
            Canvas width in pixels.
        img_height : int
            Canvas height in pixels.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        max_size_ratio : float
            Maximum triangle extent as a fraction of canvas dimensions.
            Default 0.15 limits each triangle to 15% of the canvas width
            and height. Must be in (0, 1].

        Returns
        -------
        Triangle
            A new randomly initialised small Triangle.
        """
        
        max_w = img_width * max_size_ratio
        max_h = img_height * max_size_ratio
        cx = rng.uniform(0, img_width)
        cy = rng.uniform(0, img_height)
        xs = np.clip(cx + rng.uniform(-max_w, max_w, size=3), 0, img_width - 1)
        ys = np.clip(cy + rng.uniform(-max_h, max_h, size=3), 0, img_height - 1)
        vertices = tuple(zip(xs.tolist(), ys.tolist()))
        rgba = tuple(rng.integers(0, 256, size=4).tolist())
        return cls(vertices=vertices, color=rgba)

    @classmethod
    def from_grid_random_color(
        cls,
        cell_x0: float,
        cell_y0: float,
        cell_x1: float,
        cell_y1: float,
        img_width: int,
        img_height: int,
        rng: np.random.Generator,
        vertex_noise_sigma: float = 0.0,
    ) -> "Triangle":
        """
        Create a Triangle anchored to a grid cell with fully random color.

        Motivation
        ----------
        Combines the spatial coverage guarantee of grid-based initialisation
        (see from_grid) with fully random color, rather than sampling from the
        target image. This is useful when testing whether image-seeded color
        initialisation provides a measurable advantage over random color —
        e.g. as a baseline in an ablation study on initialisation strategies.

        Parameters
        ----------
        cell_x0, cell_y0 : float
            Top-left corner of the grid cell (pixel coordinates).
        cell_x1, cell_y1 : float
            Bottom-right corner of the grid cell (pixel coordinates).
        img_width : int
            Canvas width in pixels, used for clipping vertices.
        img_height : int
            Canvas height in pixels, used for clipping vertices.
        rng : np.random.Generator
            Caller-supplied random generator for reproducibility.
        vertex_noise_sigma : float
            Std-dev of Gaussian noise applied to each vertex coordinate
            after grid placement. 0.0 means exact grid corners.
            Default 0.0 — pass a meaningful value (e.g. cell_width * 0.3)
            to ensure population diversity.

        Returns
        -------
        Triangle
            A new grid-anchored Triangle with random color.
        """
        base_xs = np.array([cell_x0, cell_x1, cell_x0], dtype=np.float32)
        base_ys = np.array([cell_y0, cell_y0, cell_y1], dtype=np.float32)
        if vertex_noise_sigma > 0.0:
            base_xs += rng.normal(0, vertex_noise_sigma, size=3)
            base_ys += rng.normal(0, vertex_noise_sigma, size=3)
        xs = np.clip(base_xs, 0, img_width - 1)
        ys = np.clip(base_ys, 0, img_height - 1)
        vertices = tuple(zip(xs.tolist(), ys.tolist()))
        rgba = tuple(rng.integers(0, 256, size=4).tolist())
        return cls(vertices=vertices, color=rgba)

    # ------------------------------------------------------------------
    # Triangle Mutation Helpers
    # ------------------------------------------------------------------
    def mutate_vertices(
        self,
        img_width: int,
        img_height: int,
        rng: np.random.Generator,
        sigma: float = 20.0,
    ) -> "Triangle":
        """
        Return a new Triangle with Gaussian-perturbed vertices.

        Vertices are clipped to the canvas boundaries after perturbation
        so the triangle always remains (at least partially) visible.

        Parameters
        ----------
        img_width, img_height : int
            Canvas dimensions used for clipping.
        rng : np.random.Generator
            Random generator (caller-supplied).
        sigma : float
            Standard deviation of the Gaussian noise in pixels.
            Smaller values produce fine-grained local search;
            larger values allow larger structural jumps.

        Returns
        -------
        Triangle
            A new Triangle instance with perturbed vertices.
        """
        noise_x = rng.normal(0, sigma, size=3)
        noise_y = rng.normal(0, sigma, size=3)

        new_xs = np.clip(self.xs + noise_x, 0, img_width - 1)
        new_ys = np.clip(self.ys + noise_y, 0, img_height - 1)

        new_vertices = tuple(zip(new_xs.tolist(), new_ys.tolist()))
        return Triangle(vertices=new_vertices, color=self.color)

    def mutate_color(
        self,
        rng: np.random.Generator,
        sigma: float = 20.0,
    ) -> Triangle:
        """
        Return a new Triangle with Gaussian-perturbed RGBA color.

        Each channel is perturbed independently and clipped to [0, 255].

        Parameters
        ----------
        rng : np.random.Generator
            Random generator (caller-supplied).
        sigma : float
            Standard deviation of the Gaussian noise per channel.

        Returns
        -------
        Triangle
            A new Triangle instance with perturbed color.
        """
        noise = rng.normal(0, sigma, size=4)
        new_color = tuple(
            int(np.clip(c + n, 0, 255))
            for c, n in zip(self.color, noise)
        )
        return Triangle(vertices=self.vertices, color=new_color)

    def mutate(
        self,
        img_width: int,
        img_height: int,
        rng: np.random.Generator,
        vertex_sigma: float = 20.0,
        color_sigma: float = 20.0,
    ) -> Triangle:
        """
        Return a new Triangle with both vertices and color perturbed.

        Convenience method combining mutate_vertices and mutate_color.

        Parameters
        ----------
        img_width, img_height : int
            Canvas dimensions.
        rng : np.random.Generator
            Random generator (caller-supplied).
        vertex_sigma : float
            Noise std-dev for vertex coordinates.
        color_sigma : float
            Noise std-dev for color channels.

        Returns
        -------
        Triangle
            A fully mutated new Triangle instance.
        """
        return (
            self
            .mutate_vertices(img_width, img_height, rng, sigma=vertex_sigma)
            .mutate_color(rng, sigma=color_sigma)
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the Triangle to a plain Python dictionary.

        Useful for saving/loading individuals to JSON.

        Returns
        -------
        dict
            Keys: 'vertices' (list of [x, y] pairs), 'color' (list [R,G,B,A]).
        """
        return {
            "vertices": [list(v) for v in self.vertices],
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Triangle:
        """
        Deserialise a Triangle from a plain Python dictionary.

        Parameters
        ----------
        data : dict
            Must contain 'vertices' and 'color' keys, as produced by to_dict().

        Returns
        -------
        Triangle
            Reconstructed Triangle instance.
        """
        vertices = tuple(tuple(v) for v in data["vertices"])
        color = tuple(data["color"])
        return cls(vertices=vertices, color=color)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        v = ", ".join(f"({x:.1f},{y:.1f})" for x, y in self.vertices)
        r, g, b, a = self.color
        return f"Triangle(vertices=[{v}], color=RGBA({r},{g},{b},{a}))"