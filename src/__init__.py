"""Spectral Geometry Explorer core package."""

from .geometry import Domain
from .solver import EigenResult, solve_eigenmodes

__all__ = ["Domain", "EigenResult", "solve_eigenmodes"]
