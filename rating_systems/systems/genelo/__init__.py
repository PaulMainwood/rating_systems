"""
GenElo — Generalised Elo from a Bayesian perspective.

Implements the rating system framework from:
    Ingram, M. (2021). How to extend Elo: a Bayesian perspective.
    Journal of Quantitative Analysis in Sports.

Two main classes:
- GenElo: Scalar version (basic Bayesian Elo + optional margin of victory)
- GenEloSurface: Multivariate version (correlated surface skills + margin +
  tournament effects + best-of-5 format modifier)
"""

from .genelo import GenElo, GenEloConfig
from .genelo_surface import (
    GenEloSurface,
    GenEloSurfaceConfig,
    SURFACE_NAMES,
    TOURNAMENT_NAMES,
)
from ._numba_surface import (
    SURFACE_HARD,
    SURFACE_CLAY,
    SURFACE_GRASS,
    SURFACE_INDOOR_HARD,
    TOURNAMENT_250_500,
    TOURNAMENT_MASTERS,
    TOURNAMENT_SLAM,
    MARGIN_MISSING,
)

__all__ = [
    "GenElo",
    "GenEloConfig",
    "GenEloSurface",
    "GenEloSurfaceConfig",
    "SURFACE_HARD",
    "SURFACE_CLAY",
    "SURFACE_GRASS",
    "SURFACE_INDOOR_HARD",
    "SURFACE_NAMES",
    "TOURNAMENT_250_500",
    "TOURNAMENT_MASTERS",
    "TOURNAMENT_SLAM",
    "TOURNAMENT_NAMES",
    "MARGIN_MISSING",
]
