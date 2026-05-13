"""
utils.py
--------
Rendering pipeline and I/O helpers for the GA image approximation.

The central function is render(), which takes a list of Triangle objects
and produces an HxWx3 numpy array representing the composed image. This
array is what gets passed to FitnessFunction.evaluate().

Rendering pipeline
------------------
1. Start with a solid black RGBA canvas (PIL Image, mode RGBA).
2. For each triangle in draw order (index 0 = bottom, index 99 = top):
   a. Create a blank RGBA layer the same size as the canvas.
   b. Draw the filled triangle onto that layer using PIL ImageDraw.
   c. Alpha-composite the layer onto the canvas using PIL's
      Image.alpha_composite(), which implements the standard
      Porter-Duff "over" operation:

          out_RGB = src_alpha * src_RGB + (1 - src_alpha) * dst_RGB
          out_A   = src_alpha + (1 - src_alpha) * dst_A

3. Convert the final RGBA canvas to RGB (drop alpha) and return as
   a uint8 numpy array.

Why a per-triangle layer?
    PIL's ImageDraw.polygon() does not support per-polygon alpha blending
    directly — it only paints flat colours. To get true alpha compositing
    for each triangle independently, we draw each one onto its own
    transparent layer and composite it onto the canvas. This is slightly
    slower than drawing directly but is the only correct approach.

Why black background?
    The target image (Vermeer's painting) has large dark/black areas
    (the background). Starting from black means those regions require
    fewer triangles to approximate, leaving more representational budget
    for the subject.

Performance note:
    render() is called once per fitness evaluation. With a population of
    N individuals over G generations, it runs N*G times. PIL is not the
    fastest renderer available, but it is correct, dependency-light, and
    fast enough for this problem scale. If runtime becomes a bottleneck,
    consider replacing the per-triangle layer approach with a numpy-based
    rasteriser (see REFERENCE below), but only after profiling.
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from triangle import Triangle

# ---------------------------------------------------------------------------
# Canvas constants — single source of truth for image dimensions.
# All other modules import from here rather than hardcoding values.
# ---------------------------------------------------------------------------

IMG_WIDTH: int = 300
IMG_HEIGHT: int = 400
IMG_SIZE: Tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT)   # (width, height) PIL convention

# Background colour for the rendered canvas (RGBA).
# Black matches the dominant background of the target painting.
BACKGROUND_COLOR: Tuple[int, int, int, int] = (0, 0, 0, 255)


# ---------------------------------------------------------------------------
# Core rendering function
# ---------------------------------------------------------------------------

def render(triangles: Sequence[Triangle]) -> np.ndarray:
    """
    Render a sequence of triangles onto a canvas and return the pixel array.

    Triangles are drawn in list order: index 0 is at the bottom (drawn
    first), index -1 is at the top (drawn last, occludes all others).
    Each triangle is alpha-composited onto the canvas using PIL's
    Porter-Duff "over" operator, so semi-transparent triangles blend
    correctly with the layers beneath them.

    Parameters
    ----------
    triangles : sequence of Triangle
        The 100 (or fewer, for testing) triangles that compose the image.
        Order determines draw priority.

    Returns
    -------
    np.ndarray
        HxWx3 uint8 array (RGB, no alpha) representing the rendered image.
        Shape is (IMG_HEIGHT, IMG_WIDTH, 3).

    Notes
    -----
    The returned array has no alpha channel because fitness functions
    compare against the RGB target image. Alpha is internal to the
    rendering process only.
    """
    # Start with a solid background layer
    canvas = Image.new("RGBA", IMG_SIZE, BACKGROUND_COLOR)

    for tri in triangles:
        # Create a fully transparent layer for this triangle
        layer = Image.new("RGBA", IMG_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # Round float vertices to integer pixel coordinates at draw time.
        # Keeping floats in Triangle allows continuous-space mutation;
        # rounding happens only here, once per render.
        pixel_vertices = [
            (int(round(x)), int(round(y)))
            for x, y in tri.vertices
        ]

        # Draw filled polygon — color is RGBA so alpha is respected
        draw.polygon(pixel_vertices, fill=tri.color)

        # Porter-Duff "over": composite layer onto canvas
        canvas = Image.alpha_composite(canvas, layer)

    # Drop alpha channel — fitness functions operate on RGB only
    canvas_rgb = canvas.convert("RGB")
    return np.array(canvas_rgb, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_target(path: str) -> np.ndarray:
    """
    Load the target image from disk and return it as an RGB numpy array.

    Converts any input mode (RGBA, L, P, etc.) to RGB so the array
    is always HxWx3 uint8, regardless of the source file format.

    Parameters
    ----------
    path : str
        Path to the target image file (PNG, JPEG, etc.).

    Returns
    -------
    np.ndarray
        HxWx3 uint8 RGB array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Target image not found at: {path}")

    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_target_pil(path: str) -> Image.Image:
    """
    Load the target image as a PIL Image (RGB mode).

    Used by build_fitness() in fitness.py, which accepts PIL Images
    directly to centralise the numpy conversion in one place.

    Parameters
    ----------
    path : str
        Path to the target image file.

    Returns
    -------
    PIL.Image.Image
        RGB-mode PIL Image.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Target image not found at: {path}")

    return Image.open(path).convert("RGB")


def save_render(array: np.ndarray, path: str) -> None:
    """
    Save a rendered numpy array to disk as a PNG file.

    Creates any missing parent directories automatically.

    Parameters
    ----------
    array : np.ndarray
        HxWx3 uint8 RGB array as returned by render().
    path : str
        Destination file path. Extension determines format (.png recommended).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


# ---------------------------------------------------------------------------
# Individual serialisation helpers
# ---------------------------------------------------------------------------

def triangles_to_json(triangles: List[Triangle], path: str) -> None:
    """
    Serialise a list of Triangle objects to a JSON file.

    Useful for checkpointing the best individual during a long GA run
    so progress is not lost if execution is interrupted.

    Parameters
    ----------
    triangles : list of Triangle
        The triangles to serialise (typically Individual.triangles).
    path : str
        Destination JSON file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = [t.to_dict() for t in triangles]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def triangles_from_json(path: str) -> List[Triangle]:
    """
    Deserialise a list of Triangle objects from a JSON file.

    Parameters
    ----------
    path : str
        Path to a JSON file previously written by triangles_to_json().

    Returns
    -------
    list of Triangle
        Reconstructed Triangle instances in original order.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Triangle.from_dict(d) for d in data]


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def side_by_side(
    target: np.ndarray,
    candidate: np.ndarray,
) -> Image.Image:
    """
    Produce a side-by-side PIL Image of target and candidate for display.

    Intended for use in notebooks to visually inspect how close a
    candidate individual is to the target at any point during the run.

    Parameters
    ----------
    target : np.ndarray
        HxWx3 uint8 RGB array of the original painting.
    candidate : np.ndarray
        HxWx3 uint8 RGB array of the rendered candidate.

    Returns
    -------
    PIL.Image.Image
        A single image with target on the left and candidate on the right,
        separated by a 4-pixel white divider.
    """
    h, w = target.shape[:2]
    divider_width = 4

    combined = Image.new("RGB", (w * 2 + divider_width, h), (255, 255, 255))
    combined.paste(Image.fromarray(target, mode="RGB"), (0, 0))
    combined.paste(Image.fromarray(candidate, mode="RGB"), (w + divider_width, 0))
    return combined


def render_triangles_pil(triangles: Sequence[Triangle]) -> Image.Image:
    """
    Render triangles and return a PIL Image instead of a numpy array.

    Convenience wrapper around render() for cases where a PIL Image is
    more practical (e.g. direct display in a notebook via IPython.display).

    Parameters
    ----------
    triangles : sequence of Triangle
        Triangles to render.

    Returns
    -------
    PIL.Image.Image
        RGB-mode PIL Image of the rendered canvas.
    """
    return Image.fromarray(render(triangles), mode="RGB")