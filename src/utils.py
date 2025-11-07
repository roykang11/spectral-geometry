"""Utility helpers for the Spectral Geometry Explorer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Structured Cartesian grid in 2D."""

    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    h: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

    @property
    def spacing(self) -> tuple[float, float]:
        return (self.h, self.h)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (float(self.x.min()), float(self.x.max()), float(self.y.min()), float(self.y.max()))


def create_grid(bounds: Sequence[float] = (-1.0, 1.0, -1.0, 1.0), resolution: int = 80) -> Grid:
    """Generate a square grid covering the given bounds.

    Parameters
    ----------
    bounds:
        Tuple (xmin, xmax, ymin, ymax). The spacing is uniform in x and y.
    resolution:
        Number of interior points along each axis (including boundary points).
    """

    xmin, xmax, ymin, ymax = map(float, bounds)
    if resolution < 5:
        raise ValueError("resolution must be at least 5 to form a meaningful grid")

    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y, indexing="ij")
    h = float((xmax - xmin) / (resolution - 1))
    return Grid(x=x, y=y, X=X, Y=Y, h=h)


def masked_indices(mask: np.ndarray) -> np.ndarray:
    """Return flattened indices for True entries of a boolean mask."""

    if mask.dtype != bool:
        mask = mask.astype(bool)
    return np.flatnonzero(mask.ravel())


def flatten_fields(*fields: np.ndarray) -> np.ndarray:
    """Flatten multiple fields and concatenate column-wise."""

    if not fields:
        raise ValueError("At least one field must be provided")
    flat = [np.asarray(field).reshape(field.shape[0], -1) for field in fields]
    return np.concatenate(flat, axis=1)


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Normalize columns of a 2D array to unit 2-norm."""

    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def sort_eigenspectrum(values: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort eigenpairs by ascending eigenvalue and ensure deterministic orientation."""

    order = np.argsort(values)
    vals = np.asarray(values)[order]
    vecs = np.asarray(vectors)[:, order]
    # Flip signs deterministically so that the maximum component is positive
    max_indices = np.argmax(np.abs(vecs), axis=0)
    signs = np.sign(vecs[max_indices, range(vecs.shape[1])])
    signs[signs == 0.0] = 1.0
    vecs = vecs * signs
    return vals, vecs


def compute_mass(mask: np.ndarray, h: float) -> float:
    """Approximate domain area using cell counting."""

    return float(mask.astype(float).sum()) * (h ** 2)


def enumerate_interior(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    """Yield (i, j) pairs for True entries in the mask."""

    mask = np.asarray(mask, dtype=bool)
    indices = np.argwhere(mask)
    for i, j in indices:
        yield int(i), int(j)
