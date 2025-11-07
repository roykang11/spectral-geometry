"""Sparse Laplacian assembly for masked grids."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

from .geometry import Domain

Neighbor = tuple[int, int]


def _neighbor_offsets() -> Iterable[Neighbor]:
    yield (1, 0)
    yield (-1, 0)
    yield (0, 1)
    yield (0, -1)


def assemble_dirichlet_laplacian(domain: Domain, dtype=np.float64) -> csc_matrix:
    """Construct the Laplace operator with zero Dirichlet boundary conditions."""

    interior_mask = domain.interior_mask
    if interior_mask.sum() == 0:
        raise ValueError("Domain mask leaves no interior points for solving")

    grid_shape = domain.mask.shape
    idx_map = -np.ones(grid_shape, dtype=int)
    interior_points = np.argwhere(interior_mask)
    for linear_index, (i, j) in enumerate(interior_points):
        idx_map[i, j] = linear_index

    n = interior_points.shape[0]
    matrix = lil_matrix((n, n), dtype=dtype)
    h2 = domain.grid.h ** 2

    for (row_idx, (i, j)) in enumerate(interior_points):
        matrix[row_idx, row_idx] = -4.0 / h2
        for di, dj in _neighbor_offsets():
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_shape[0] and 0 <= nj < grid_shape[1]:
                if interior_mask[ni, nj]:
                    col_idx = idx_map[ni, nj]
                    matrix[row_idx, col_idx] = matrix[row_idx, col_idx] + 1.0 / h2
            # Outside domain contributes zero via Dirichlet BC

    return matrix.tocsc()


def points_to_grid(domain: Domain, interior_vector: np.ndarray) -> np.ndarray:
    """Lift an interior vector back to the full grid with zeros on boundary."""

    field = np.zeros(domain.mask.size, dtype=interior_vector.dtype)
    field[domain.interior_indices()] = interior_vector
    return field.reshape(domain.mask.shape)
