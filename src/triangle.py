"""
triangle.py
-----------
Defines the Triangle class, the atomic unit (gene) of the genetic algorithm.

Each Triangle encodes:
    - Three (x, y) vertices defining its shape and position on the canvas.
    - An RGBA color (R, G, B, A) where alpha controls transparency/blending.

Design notes:
    - Immutability is enforced via frozen and properties: once created,
      a Triangle's state cannot be mutated. Genetic operators always produce
      new Triangle instances, which makes reasoning about state straightforward
      and avoids accidental side effects.
    - The class is deliberately free of any fitness or population logic —
      it is purely a data container with rendering capability.
    - Numpy is used for all numerical operations for performance.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Vertex = Tuple[float, float]   # (x, y) in canvas pixel coordinates
RGBAColor = Tuple[int, int, int, int]  # (R, G, B, A) each in [0, 255]


@dataclass(frozen=True)
class Triangle:
    """
    Immutable representation of a single triangle gene.

    Attributes
    ----------
    vertices : tuple of three (x, y) pairs
        Pixel coordinates of the three corners. Values are floats so that
        genetic operators can work in continuous space; 
        they are rounded only at render time.
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
                f"A triangle must have exactly 3 vertices, got {len(self.vertices)} instead."
            )
        for i, (x, y) in enumerate(self.vertices):
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                raise TypeError(
                    f"Vertex {i} coordinates must be numeric, got ({type(x)}, {type(y)})."
                )

        r, g, b, a = self.color
        for channel_name, value in zip("RGBA", (r, g, b, a)):
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
        Compute the absolute area of the triangle using the shoelace formula.

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

    def mutate_vertices(
        self,
        img_width: int,
        img_height: int,
        rng: np.random.Generator,
        sigma: float = 20.0,
    ) -> Triangle:
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