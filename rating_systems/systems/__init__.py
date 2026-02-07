"""Rating system implementations.

Online systems (incremental updates):
- Elo: Classic Elo rating system
- Glicko: Glicko rating system with rating deviation
- Glicko2: Glicko-2 rating system with volatility
- Stephenson: Extended Glicko with neighbourhood and bonus parameters
- TrueSkill: Bayesian skill estimation with Gaussian beliefs
- Yuksel: Adaptive rating system with uncertainty tracking

Batch systems (refit on full history):
- WHR: Whole History Rating
- TrueSkillThroughTime: TrueSkill Through Time
- SurfaceTTT: Surface-specific TrueSkill Through Time

All implementations use Numba for high performance.
"""

# Default Numba implementations
from .elo import Elo, EloConfig
from .glicko import Glicko, GlickoConfig
from .glicko2 import Glicko2, Glicko2Config
from .stephenson import Stephenson, StephensonConfig
from .trueskill import TrueSkill, TrueSkillConfig
from .yuksel import Yuksel, YukselConfig

# Batch systems (keep existing implementations)
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
]
