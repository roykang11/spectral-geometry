"""Domain generation utilities for Spectral Geometry Explorer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from scipy import ndimage

from .utils import Grid, compute_mass, create_grid

MaskFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class Domain:
    """Representation of a planar domain on a structured grid."""

    name: str
    grid: Grid
    mask: np.ndarray

    def __post_init__(self) -> None:
        if self.mask.shape != self.grid.shape:
            raise ValueError("mask and grid must have matching shapes")
        self.mask = self.mask.astype(bool)
        self.boundary_mask = _compute_boundary(self.mask)
        self.interior_mask = np.logical_and(self.mask, ~self.boundary_mask)
        self.area = compute_mass(self.mask, self.grid.h)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def interior_indices(self) -> np.ndarray:
        return np.flatnonzero(self.interior_mask.ravel())

    def all_indices(self) -> np.ndarray:
        return np.flatnonzero(self.mask.ravel())

    def summary(self) -> str:
        return (
            f"Domain(name={self.name}, shape={self.shape}, h={self.grid.h:.4f}, "
            f"area≈{self.area:.3f})"
        )


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def mask_rectangle(grid: Grid, width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """Axis-aligned rectangle centered at the origin."""

    return (np.abs(grid.X) <= width / 2) & (np.abs(grid.Y) <= height / 2)


def mask_circle(grid: Grid, radius: float = 1.0) -> np.ndarray:
    """Disk centered at the origin."""

    return grid.X**2 + grid.Y**2 <= radius**2


def mask_annulus(grid: Grid, inner_radius: float, outer_radius: float) -> np.ndarray:
    r2 = grid.X**2 + grid.Y**2
    return (r2 <= outer_radius**2) & (r2 >= inner_radius**2)


def mask_l_shape(grid: Grid, arm: float = 1.0, thickness: float = 0.5) -> np.ndarray:
    rect1 = mask_rectangle(grid, width=arm, height=thickness)
    rect2 = mask_rectangle(grid, width=thickness, height=arm)
    return np.logical_or(rect1, rect2)


def mask_polygon(grid: Grid, vertices: Sequence[tuple[float, float]]) -> np.ndarray:
    """Rasterize a polygon defined by vertices in counter-clockwise order."""

    from matplotlib.path import Path as MplPath

    path = MplPath(vertices)
    points = np.stack([grid.X.ravel(), grid.Y.ravel()], axis=1)
    mask = path.contains_points(points)
    return mask.reshape(grid.shape)


def mask_from_callable(grid: Grid, fn: MaskFunction) -> np.ndarray:
    mask = fn(grid.X, grid.Y)
    return np.asarray(mask, dtype=bool)


def mask_from_image(file_path: str | Path, grid: Grid, threshold: float = 0.5) -> np.ndarray:
    """Load a binary mask from an image file (white = inside)."""

    from PIL import Image

    img = Image.open(file_path).convert("L")
    arr = np.array(img, dtype=float)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
    arr = arr >= threshold
    arr = arr[::-1, :]  # flip y-axis
    arr = ndimage.zoom(arr, (grid.shape[0] / arr.shape[0], grid.shape[1] / arr.shape[1]), order=1)
    return arr > 0.5


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_domain(
    resolution: int = 120,
    bounds: Sequence[float] = (-1.0, 1.0, -1.0, 1.0),
    shape: Literal["circle", "square", "rectangle", "l-shape", "annulus"] | None = "circle",
    *,
    mask: np.ndarray | None = None,
    mask_fn: MaskFunction | None = None,
    params: dict | None = None,
    name: str | None = None,
) -> Domain:
    """Create a domain using a built-in shape or custom mask."""

    grid = create_grid(bounds=bounds, resolution=resolution)
    params = params or {}

    if mask is not None:
        domain_mask = np.asarray(mask, dtype=bool)
    elif mask_fn is not None:
        domain_mask = mask_from_callable(grid, mask_fn)
    elif shape == "circle":
        domain_mask = mask_circle(grid, radius=params.get("radius", 0.9))
    elif shape in {"square", "rectangle"}:
        w = params.get("width", 1.5)
        h = params.get("height", 1.5 if shape == "rectangle" else w)
        domain_mask = mask_rectangle(grid, width=w, height=h)
    elif shape == "l-shape":
        domain_mask = mask_l_shape(
            grid,
            arm=params.get("arm", 1.6),
            thickness=params.get("thickness", 0.8),
        )
    elif shape == "annulus":
        domain_mask = mask_annulus(
            grid,
            inner_radius=params.get("inner_radius", 0.4),
            outer_radius=params.get("outer_radius", 0.9),
        )
    else:
        raise ValueError(f"Unsupported shape '{shape}'. Provide mask or mask_fn instead.")

    domain_name = name or (shape or "custom")
    return Domain(name=domain_name, grid=grid, mask=domain_mask)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=int)
    eroded = ndimage.binary_erosion(mask, structure=structure, iterations=1, border_value=0)
    return np.logical_and(mask, ~eroded)
