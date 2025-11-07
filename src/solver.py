"""Eigenvalue problem solvers for the Spectral Geometry Explorer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.sparse.linalg import eigsh

from .geometry import Domain
from .laplacian import assemble_dirichlet_laplacian, points_to_grid
from .utils import normalize_columns, sort_eigenspectrum


@dataclass
class EigenResult:
    """Container holding eigenvalues and eigenfunctions on the full grid."""

    values: np.ndarray
    vectors: np.ndarray  # shape = (n_interior, k)
    grid_modes: np.ndarray  # shape = (k, *domain.shape)
    domain: Domain

    def fundamental_frequency(self, wave_speed: float = 1.0) -> float:
        return wave_speed * np.sqrt(float(self.values[0]))

    def mode(self, index: int) -> np.ndarray:
        return self.grid_modes[index]

    def spectra(self) -> Iterable[tuple[int, float]]:
        for i, value in enumerate(self.values, start=1):
            yield i, float(value)


SolverMode = Literal["SM", "SA", "LM", "LA"]


def solve_eigenmodes(
    domain: Domain,
    k: int = 10,
    which: SolverMode = "SM",
    shift_invert: bool = False,
    sigma: float | None = None,
    wave_speed: float = 1.0,
) -> EigenResult:
    """Compute the lowest `k` eigenmodes of the Laplacian on `domain`.

    Parameters
    ----------
    domain:
        Domain instance containing mask and grid information.
    k:
        Number of eigenvalues/eigenvectors to compute.
    which:
        Selection rule passed to `scipy.sparse.linalg.eigsh`.
    shift_invert:
        Enable shift-invert mode to target eigenvalues near `sigma`.
    sigma:
        Shift for inverse iteration when `shift_invert` is True.
    wave_speed:
        Physical wave speed. Only used for derived quantities (e.g., audio pitch).
    """

    laplacian = assemble_dirichlet_laplacian(domain)
    n = laplacian.shape[0]
    if k >= n:
        k = n - 2
        if k < 1:
            raise ValueError("Domain too small to compute requested number of eigenmodes")

    op_kwargs = {}
    if shift_invert:
        op_kwargs["sigma"] = 0.0 if sigma is None else sigma
        op_kwargs["which"] = "LM"
        op_kwargs["mode"] = "buckling"
        which = "LM"

    values, vectors = eigsh(laplacian, k=k, which=which, **op_kwargs)
    values, vectors = sort_eigenspectrum(values, vectors)
    vectors = normalize_columns(vectors)

    grid_modes = np.stack(
        [points_to_grid(domain, vectors[:, idx]) for idx in range(vectors.shape[1])],
        axis=0,
    )

    # Normalize modes with respect to L2 norm over domain
    for idx in range(grid_modes.shape[0]):
        mode = grid_modes[idx]
        norm = np.linalg.norm(mode[domain.mask])
        if norm != 0.0:
            grid_modes[idx] = mode / norm
            vectors[:, idx] = vectors[:, idx] / norm

    return EigenResult(values=values, vectors=vectors, grid_modes=grid_modes, domain=domain)


def estimate_weyl_constant(domain: Domain) -> float:
    """Return the Weyl constant |Ω|/(4π)."""

    return domain.area / (4.0 * np.pi)


def modes_to_frequencies(values: np.ndarray, wave_speed: float = 340.0) -> np.ndarray:
    """Map eigenvalues to audible frequencies given a wave speed in m/s."""

    values = np.asarray(values)
    return wave_speed * np.sqrt(np.clip(values, 0.0, None)) / (2.0 * np.pi)
