"""Rating system implementations.

Online systems (incremental updates):
- Elo / WElo: Classic Elo rating system
- Glicko / WGlicko: Glicko rating system with rating deviation
- Glicko2 / WGlicko2: Glicko-2 rating system with volatility
- Stephenson / WStephenson: Extended Glicko with neighbourhood and bonus parameters
- TrueSkill / WeightedTrueSkill: Bayesian skill estimation with Gaussian beliefs
- Yuksel / WYuksel: Adaptive rating system with uncertainty tracking

Batch systems (refit on full history):
- WHR / WeightedWHR: Whole History Rating
- TrueSkillThroughTime / WeightedTTT: TrueSkill Through Time
- SurfaceTTT: Surface-specific TrueSkill Through Time

Analysis:
- HodgeIntransitivity: Hodge decomposition for measuring intransitivity

W* (weighted) variants accept per-game weights for boosted rating updates.
All implementations use Numba for high performance.
"""

# Default Numba implementations
from .elo import Elo, EloConfig
from .glicko import Glicko, GlickoConfig
from .glicko2 import Glicko2, Glicko2Config
from .stephenson import Stephenson, StephensonConfig
from .trueskill import TrueSkill, TrueSkillConfig
from .yuksel import Yuksel, YukselConfig

# Batch systems
from .whr import WHR
from .welo import WElo, WEloConfig
from .trueskill_through_time import TrueSkillThroughTime, SurfaceTTT
from .trueskill_through_time import (
    SURFACE_HARD,
    SURFACE_CLAY,
    SURFACE_GRASS,
    SURFACE_NON_CLAY,
    SURFACE_NAMES,
)
from .wttt import WeightedTTT, WTTTConfig
from .wglicko import WGlicko, WGlickoConfig
from .wglicko2 import WGlicko2, WGlicko2Config
from .wwhr import WeightedWHR, WWHRConfig
from .wstephenson import WStephenson, WStephensonConfig
from .wyuksel import WYuksel, WYukselConfig
from .wtrueskill import WeightedTrueSkill, WTrueSkillConfig
from .hodge import HodgeIntransitivity, HodgeConfig
from .melo import MElo, MEloConfig
from .wmelo import WMElo, WMEloConfig
from .matern_ttt import MaternTTT, MaternTTTConfig
from .wmatern_ttt import WeightedMaternTTT, WMaternTTTConfig
from .margin_ttt import MarginTTT, MarginTTTConfig
from .wmargin_ttt import WMarginTTT, WMarginTTTConfig
from .eigenvector import EigenvectorCentrality, EigenvectorConfig
from .weigenvector import WEigenvectorCentrality, WEigenvectorConfig
from .gelo import GElo, GEloConfig
from .wgelo import WGElo, WGEloConfig
from .disc import Disc, DiscConfig
from .wdisc import WDisc, WDiscConfig
from .kickscore_rating import KickScoreRating, KickScoreConfig
from .serve_return import ServeReturnGlicko, ServeReturnElo, ServeReturnGlickoPoints
from .genelo import GenElo, GenEloConfig, GenEloSurface, GenEloSurfaceConfig, MARGIN_MISSING
from .surface_factor_glicko import WSurfaceFactorGlicko, SurfaceFactorGlickoConfig
from .surface_factor_whr import (
    SurfaceFactorWHR,
    SurfaceFactorWHRConfig,
    WSurfaceFactorWHR,
)

__all__ = [
    # Default (Numba) implementations
    "Elo",
    "EloConfig",
    "WElo",
    "WEloConfig",
    "Glicko",
    "GlickoConfig",
    "Glicko2",
    "Glicko2Config",
    "Stephenson",
    "StephensonConfig",
    "TrueSkill",
    "TrueSkillConfig",
    "Yuksel",
    "YukselConfig",
    # Batch systems
    "WHR",
    "TrueSkillThroughTime",
    "SurfaceTTT",
    # Surface constants (input values)
    "SURFACE_HARD",
    "SURFACE_CLAY",
    "SURFACE_GRASS",
    # Surface constants (internal)
    "SURFACE_NON_CLAY",
    "SURFACE_NAMES",
    # Weighted TTT
    "WeightedTTT",
    "WTTTConfig",
    # Weighted Glicko / Glicko-2
    "WGlicko",
    "WGlickoConfig",
    "WGlicko2",
    "WGlicko2Config",
    # Weighted WHR
    "WeightedWHR",
    "WWHRConfig",
    # Weighted Stephenson
    "WStephenson",
    "WStephensonConfig",
    # Weighted Yuksel
    "WYuksel",
    "WYukselConfig",
    # Weighted TrueSkill
    "WeightedTrueSkill",
    "WTrueSkillConfig",
    # Hodge intransitivity
    "HodgeIntransitivity",
    "HodgeConfig",
    # mElo (multidimensional Elo with intransitivity)
    "MElo",
    "MEloConfig",
    # Weighted mElo
    "WMElo",
    "WMEloConfig",
    # Matérn 3/2 TTT
    "MaternTTT",
    "MaternTTTConfig",
    # Weighted Matérn 3/2 TTT
    "WeightedMaternTTT",
    "WMaternTTTConfig",
    # Margin TTT (Gaussian margin-of-victory)
    "MarginTTT",
    "MarginTTTConfig",
    # Weighted Margin TTT
    "WMarginTTT",
    "WMarginTTTConfig",
    # Eigenvector centrality (B-Score)
    "EigenvectorCentrality",
    "EigenvectorConfig",
    # Weighted Eigenvector centrality
    "WEigenvectorCentrality",
    "WEigenvectorConfig",
    # G-Elo (ordinal margin-of-victory)
    "GElo",
    "GEloConfig",
    # Weighted G-Elo
    "WGElo",
    "WGEloConfig",
    # Disc decomposition (skill + consistency)
    "Disc",
    "DiscConfig",
    # Weighted Disc decomposition
    "WDisc",
    "WDiscConfig",
    # KickScore (GP paired comparisons)
    "KickScoreRating",
    "KickScoreConfig",
    # GenElo (Bayesian Elo, Ingram 2021)
    "GenElo",
    "GenEloConfig",
    "GenEloSurface",
    "GenEloSurfaceConfig",
    # Surface-Factor Weighted Glicko (TrueSkill 2 cross-surface model)
    "WSurfaceFactorGlicko",
    "SurfaceFactorGlickoConfig",
    # Surface-Factor WHR (batch joint MAP via coordinate descent)
    "SurfaceFactorWHR",
    "SurfaceFactorWHRConfig",
    "WSurfaceFactorWHR",
]
