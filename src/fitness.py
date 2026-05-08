"""
fitness.py
----------
Defines the fitness evaluation strategy hierarchy for the GA image approximation.

Three concrete implementations are provided:

    1. RMSEFitness       — pixel-wise Root Mean Squared Error in RGB space.
                           Required by the project specification.

    2. CIEDEFitness      — perceptual colour difference using the CIEDE2000
                           metric, which better models how the human
                           visual system perceives colour differences.
                           Implements the additional challenge (option 1).

    3. SSIMFitness        — Structural Similarity Index (SSIM), which jointly
                           captures luminance, contrast and local structure.
                           Goes beyond colour distance to model how humans
                           perceive image quality holistically.

Design notes:
    - FitnessFunction is an abstract base class (ABC). All concrete fitness
      classes must implement evaluate(). The GA engine depends only on the
      abstract interface, making it trivially easy to swap metrics.
    - All implementations operate on pre-rendered numpy arrays (HxWx3,
      dtype uint8) rather than PIL Images, keeping the hot path fast.
    - Lower fitness = better (minimisation convention). SSIM is natively a
      similarity metric in [-1, 1] (higher = better), so SSIMFitness returns
      1 - SSIM to conform to the minimisation convention.
    - The CIEDE2000 implementation converts sRGB -> CIELAB per pixel before
      computing ΔE₀₀. This is the perceptually uniform colour space that
      models human vision far better than raw RGB distance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class FitnessFunction(ABC):
    """
    Abstract base class for all fitness evaluation strategies.

    Concrete subclasses implement evaluate() to return a scalar fitness
    value for a candidate image array. Lower values indicate better
    approximations of the target image.

    Parameters
    ----------
    target : np.ndarray
        The reference image as an HxWx3 uint8 array (RGB, no alpha).
        Stored once at construction time; shared across all evaluations.
    """

    def __init__(self, target: np.ndarray) -> None:
        if target.ndim != 3 or target.shape[2] != 3:
            raise ValueError(
                f"Target must be an HxWx3 RGB array, got shape {target.shape}."
            )
        if target.dtype != np.uint8:
            raise ValueError(
                f"Target must be dtype uint8, got {target.dtype}."
            )
        # Store as float32 once — avoids repeated casting in evaluate()
        self._target_f32: np.ndarray = target.astype(np.float32)
        self._target: np.ndarray = target
        self._height, self._width = target.shape[:2]

    @property
    def target(self) -> np.ndarray:
        """The reference image (uint8, HxWx3)."""
        return self._target

    @abstractmethod
    def evaluate(self, candidate: np.ndarray) -> float:
        """
        Compute the fitness (error) between candidate and target images.

        Parameters
        ----------
        candidate : np.ndarray
            Rendered candidate image, HxWx3 uint8, same shape as target.

        Returns
        -------
        float
            Non-negative scalar error. Lower is better.
        """

    def _validate_candidate(self, candidate: np.ndarray) -> None:
        """Shared validation for candidate arrays."""
        if candidate.shape != self._target.shape:
            raise ValueError(
                f"Candidate shape {candidate.shape} does not match "
                f"target shape {self._target.shape}."
            )


# ---------------------------------------------------------------------------
# Concrete implementation 1: RMSE (required by specification)
# ---------------------------------------------------------------------------

class RMSEFitness(FitnessFunction):
    """
    Pixel-wise Root Mean Squared Error in RGB colour space.

    This is the fitness function required by the project specification.
    It computes the Euclidean distance per channel across all pixels,
    averaged and square-rooted to give a single scalar in [0, 255].

    Formula
    -------
        RMSE = sqrt( mean( (target - candidate)^2 ) )

    where the mean is taken over all pixels and all three colour channels.

    Limitations
    -----------
    RGB distance is not perceptually uniform: a difference of 10 units in
    blue is not perceived the same as 10 units in green. The CIEDEFitness
    class addresses this limitation.
    """

    def evaluate(self, candidate: np.ndarray) -> float:
        """
        Compute RMSE between candidate and target.

        Parameters
        ----------
        candidate : np.ndarray
            Rendered candidate image, HxWx3 uint8.

        Returns
        -------
        float
            RMSE value in [0.0, 255.0]. Lower is better.
        """
        self._validate_candidate(candidate)
        diff = self._target_f32 - candidate.astype(np.float32)
        return float(np.sqrt(np.mean(diff ** 2)))

    def __repr__(self) -> str:
        return f"RMSEFitness(target_shape={self._target.shape})"


# ---------------------------------------------------------------------------
# Concrete implementation 2: CIEDE2000 (additional challenge — option 1)
# ---------------------------------------------------------------------------

class CIEDEFitness(FitnessFunction):
    """
    Perceptual colour fitness using the CIEDE2000 (ΔE₀₀) metric.

    Motivation
    ----------
    The human visual system is not equally sensitive to all RGB differences.
    Two pairs of colours with the same Euclidean RGB distance may look very
    different in terms of perceived colour shift. CIEDE2000 corrects for
    this by operating in CIELAB (L*a*b*) colour space, which is designed
    to be perceptually uniform, and applying additional correction terms for
    lightness, chroma and hue weighting.

    Pipeline
    --------
    1. sRGB (uint8) -> linear RGB (float, [0,1]) via gamma expansion
    2. Linear RGB -> CIE XYZ (D65 illuminant)
    3. CIE XYZ -> CIELAB (L*, a*, b*)
    4. ΔE₀₀ computed per pixel pair
    5. Mean ΔE₀₀ returned as the fitness scalar

    A ΔE₀₀ value of 1.0 is considered the threshold of just-noticeable
    difference (JND) for the human eye under standardised conditions.

    References
    ----------
    Sharma, G., Wu, W., & Dalal, E. N. (2005). The CIEDE2000 color‐
    difference formula. Color Research & Application, 30(1), 21–30.
    """

    # D65 illuminant reference white point (used in XYZ -> LAB conversion)
    _D65_WHITE: np.ndarray = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    # sRGB -> CIE XYZ (D65) linear transformation matrix
    _RGB_TO_XYZ: np.ndarray = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)

    def __init__(self, target: np.ndarray) -> None:
        super().__init__(target)
        # Pre-compute target LAB once — avoids repeating the expensive
        # conversion for every fitness evaluation during the GA run.
        self._target_lab: np.ndarray = self._rgb_to_lab(self._target_f32)

    # ------------------------------------------------------------------
    # Colour space conversion helpers (vectorised, operate on HxWx3 arrays)
    # ------------------------------------------------------------------

    @staticmethod
    def _gamma_expand(rgb_uint8: np.ndarray) -> np.ndarray:
        """
        Convert sRGB uint8 [0,255] to linear RGB float [0,1].

        Applies the standard IEC 61966-2-1 sRGB gamma expansion curve.
        This step is required before the linear RGB->XYZ matrix transform.

        Parameters
        ----------
        rgb_uint8 : np.ndarray
            HxWx3 float32 array with values in [0, 255].

        Returns
        -------
        np.ndarray
            HxWx3 float32 array with linear RGB values in [0, 1].
        """
        c = rgb_uint8 / 255.0
        # Piecewise sRGB transfer function (vectorised)
        linear = np.where(
            c <= 0.04045,
            c / 12.92,
            ((c + 0.055) / 1.055) ** 2.4,
        )
        return linear.astype(np.float32)

    @classmethod
    def _rgb_to_xyz(cls, rgb_linear: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB (HxWx3) to CIE XYZ using the D65 matrix.

        Parameters
        ----------
        rgb_linear : np.ndarray
            HxWx3 float32, linear RGB values in [0, 1].

        Returns
        -------
        np.ndarray
            HxWx3 float32, CIE XYZ values.
        """
        h, w, _ = rgb_linear.shape
        flat = rgb_linear.reshape(-1, 3)               # (N, 3)
        xyz_flat = flat @ cls._RGB_TO_XYZ.T            # (N, 3)
        return xyz_flat.reshape(h, w, 3)

    @classmethod
    def _xyz_to_lab(cls, xyz: np.ndarray) -> np.ndarray:
        """
        Convert CIE XYZ (HxWx3) to CIELAB (L*, a*, b*).

        Parameters
        ----------
        xyz : np.ndarray
            HxWx3 float32, normalised relative to D65 white point.

        Returns
        -------
        np.ndarray
            HxWx3 float32 with channels [L*, a*, b*].
        """
        # Normalise by D65 white point
        xyz_norm = xyz / cls._D65_WHITE

        # CIE standard f() function
        delta = 6.0 / 29.0
        delta_cubed = delta ** 3  # ≈ 0.008856

        f = np.where(
            xyz_norm > delta_cubed,
            np.cbrt(xyz_norm),
            xyz_norm / (3 * delta ** 2) + 4.0 / 29.0,
        )

        L = 116.0 * f[..., 1] - 16.0
        a = 500.0 * (f[..., 0] - f[..., 1])
        b = 200.0 * (f[..., 1] - f[..., 2])

        return np.stack([L, a, b], axis=-1).astype(np.float32)

    @classmethod
    def _rgb_to_lab(cls, rgb_f32: np.ndarray) -> np.ndarray:
        """
        Full pipeline: sRGB float32 [0,255] -> CIELAB.

        Parameters
        ----------
        rgb_f32 : np.ndarray
            HxWx3 float32 with values in [0, 255].

        Returns
        -------
        np.ndarray
            HxWx3 float32 with CIELAB values.
        """
        linear = cls._gamma_expand(rgb_f32)
        xyz = cls._rgb_to_xyz(linear)
        return cls._xyz_to_lab(xyz)

    # ------------------------------------------------------------------
    # CIEDE2000 per-pixel computation
    # ------------------------------------------------------------------

    @staticmethod
    def _ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
        """
        Compute CIEDE2000 ΔE₀₀ for every pixel pair.

        Implements the full CIEDE2000 formula including the correction
        terms for chroma (S_C), hue (S_H) and the rotation term (R_T)
        that accounts for the blue-region hue-chroma interaction.

        Parameters
        ----------
        lab1, lab2 : np.ndarray
            HxWx3 float32 arrays with CIELAB values [L*, a*, b*].

        Returns
        -------
        np.ndarray
            HxW float32 array of per-pixel ΔE₀₀ values.

        References
        ----------
        Sharma et al. (2005), equations (1)–(21).
        """
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

        # Step 1: C*ab and mean C*ab
        C1 = np.sqrt(a1 ** 2 + b1 ** 2)
        C2 = np.sqrt(a2 ** 2 + b2 ** 2)
        C_avg = (C1 + C2) / 2.0

        # G factor — adjusts a* for chroma-dependent weighting
        C_avg_7 = C_avg ** 7
        G = 0.5 * (1.0 - np.sqrt(C_avg_7 / (C_avg_7 + 25.0 ** 7)))

        a1p = a1 * (1.0 + G)
        a2p = a2 * (1.0 + G)

        # Step 2: C' and h'
        C1p = np.sqrt(a1p ** 2 + b1 ** 2)
        C2p = np.sqrt(a2p ** 2 + b2 ** 2)

        h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
        h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

        # Step 3: ΔL', ΔC', Δh'
        dLp = L2 - L1
        dCp = C2p - C1p

        # Δh' — must handle wrap-around at 360°
        dhp = np.where(
            C1p * C2p == 0, 0.0,
            np.where(
                np.abs(h2p - h1p) <= 180.0, h2p - h1p,
                np.where(h2p - h1p > 180.0, h2p - h1p - 360.0, h2p - h1p + 360.0)
            )
        )
        dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

        # Step 4: CIEDE2000 weighting functions
        Lp_avg = (L1 + L2) / 2.0
        Cp_avg = (C1p + C2p) / 2.0

        # Mean h' — again handle wrap-around
        hp_avg = np.where(
            C1p * C2p == 0, h1p + h2p,
            np.where(
                np.abs(h1p - h2p) <= 180.0, (h1p + h2p) / 2.0,
                np.where(
                    h1p + h2p < 360.0,
                    (h1p + h2p + 360.0) / 2.0,
                    (h1p + h2p - 360.0) / 2.0,
                )
            )
        )

        T = (
            1.0
            - 0.17 * np.cos(np.radians(hp_avg - 30.0))
            + 0.24 * np.cos(np.radians(2.0 * hp_avg))
            + 0.32 * np.cos(np.radians(3.0 * hp_avg + 6.0))
            - 0.20 * np.cos(np.radians(4.0 * hp_avg - 63.0))
        )

        SL = 1.0 + 0.015 * (Lp_avg - 50.0) ** 2 / np.sqrt(20.0 + (Lp_avg - 50.0) ** 2)
        SC = 1.0 + 0.045 * Cp_avg
        SH = 1.0 + 0.015 * Cp_avg * T

        # Rotation term (blue-purple hue region correction)
        Cp_avg_7 = Cp_avg ** 7
        RC = 2.0 * np.sqrt(Cp_avg_7 / (Cp_avg_7 + 25.0 ** 7))
        d_theta = 30.0 * np.exp(-((hp_avg - 275.0) / 25.0) ** 2)
        RT = -np.sin(np.radians(2.0 * d_theta)) * RC

        # Step 5: Final ΔE₀₀
        de00 = np.sqrt(
            (dLp / SL) ** 2
            + (dCp / SC) ** 2
            + (dHp / SH) ** 2
            + RT * (dCp / SC) * (dHp / SH)
        )
        return de00.astype(np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, candidate: np.ndarray) -> float:
        """
        Compute mean CIEDE2000 between candidate and target.

        Parameters
        ----------
        candidate : np.ndarray
            Rendered candidate image, HxWx3 uint8.

        Returns
        -------
        float
            Mean ΔE₀₀ across all pixels. Lower is better.
            Values below 1.0 are imperceptible to the human eye;
            values above ~10 represent clearly visible differences.
        """
        self._validate_candidate(candidate)
        candidate_lab = self._rgb_to_lab(candidate.astype(np.float32))
        de00_map = self._ciede2000(self._target_lab, candidate_lab)
        return float(np.mean(de00_map))

    def __repr__(self) -> str:
        return f"CIEDEFitness(target_shape={self._target.shape})"


# ---------------------------------------------------------------------------
# Concrete implementation 3: SSIM (Structural Similarity Index)
# ---------------------------------------------------------------------------

class SSIMFitness(FitnessFunction):
    """
    Structural Similarity Index (SSIM) fitness function.

    Motivation
    ----------
    Both RMSE and CIEDE2000 are pixel-wise metrics — they compare colours
    at each location independently. The human visual system, however, is
    highly sensitive to local structure: edges, textures and contrast
    patterns. SSIM models this by jointly measuring three components:

        - Luminance  : mean intensity comparison
        - Contrast   : standard deviation comparison
        - Structure  : normalised cross-correlation of local patches

    This makes SSIM particularly effective at preserving the perceived
    sharpness and structural fidelity of the approximation, beyond what
    colour metrics alone can capture.

    Convention
    ----------
    SSIM is natively a *similarity* metric in [-1, 1] where 1 means
    identical images. To conform to the project's minimisation convention
    (lower = better), this class returns ``1 - SSIM``, mapping the range
    to [0, 2] where 0 means perfect match.

    Implementation
    --------------
    Uses scikit-image's ``structural_similarity`` with ``channel_axis=-1``
    to compute SSIM jointly across all three RGB channels, which is more
    representative than averaging per-channel SSIM scores independently.
    The ``data_range`` is fixed at 255 (uint8 images).

    References
    ----------
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: from error visibility to structural
    similarity. IEEE Transactions on Image Processing, 13(4), 600–612.
    """

    def evaluate(self, candidate: np.ndarray) -> float:
        """
        Compute 1 - SSIM between candidate and target.

        Parameters
        ----------
        candidate : np.ndarray
            Rendered candidate image, HxWx3 uint8.

        Returns
        -------
        float
            Value in [0, 2]. Lower is better. 0 means perfect structural
            and luminance match; values above ~0.5 indicate poor similarity.
        """
        self._validate_candidate(candidate)
        score = ssim(
            self._target,
            candidate,
            data_range=255,
            channel_axis=-1,    # treat last axis as RGB channels
        )
        # Invert: SSIM=1 (identical) -> fitness=0 (best)
        return float(1.0 - score)

    def __repr__(self) -> str:
        return f"SSIMFitness(target_shape={self._target.shape})"


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_fitness(
    metric: str,
    target_image: Image.Image,
) -> FitnessFunction:
    """
    Factory function to instantiate a FitnessFunction by name.

    Converts the PIL Image to the required numpy format and returns
    the requested fitness object. Centralises image pre-processing so
    callers (notebooks, GA engine) don't need to handle it.

    Parameters
    ----------
    metric : str
        One of ``'rmse'``, ``'ciede2000'``, or ``'ssim'``.
    target_image : PIL.Image.Image
        The target painting. Will be converted to RGB if necessary.

    Returns
    -------
    FitnessFunction
        Instantiated fitness object ready for evaluation.

    Raises
    ------
    ValueError
        If an unknown metric name is provided.

    Examples
    --------
    >>> from PIL import Image
    >>> img = Image.open("data/girl_pearl.png")
    >>> fitness = build_fitness("rmse", img)
    >>> fitness
    RMSEFitness(target_shape=(400, 300, 3))
    """
    target_rgb = target_image.convert("RGB")
    target_array = np.array(target_rgb, dtype=np.uint8)

    registry: dict[str, type[FitnessFunction]] = {
        "rmse": RMSEFitness,
        "ciede2000": CIEDEFitness,
        "ssim": SSIMFitness,
    }

    metric_lower = metric.lower().strip()
    if metric_lower not in registry:
        raise ValueError(
            f"Unknown fitness metric '{metric}'. "
            f"Available options: {list(registry.keys())}."
        )

    return registry[metric_lower](target_array)