"""Rating system implementations.

Default implementations (Numba-based, high performance):
- Elo: Classic Elo rating system
- Glicko: Glicko rating system with rating deviation
- Glicko2: Glicko-2 rating system with volatility
- TrueSkill: Bayesian skill estimation with Gaussian beliefs
- Yuksel: Adaptive rating system with uncertainty tracking (Yuksel 2024)

Batch systems (refit on full history):
- WHR: Whole History Rating
- TrueSkillThroughTime: TrueSkill Through Time

PyTorch implementations (GPU acceleration) are available via explicit imports:
- from rating_systems.systems.elo.elo_torch import EloTorch
- from rating_systems.systems.glicko.glicko_torch import GlickoTorch
- from rating_systems.systems.glicko2.glicko2_torch import Glicko2Torch
"""

# Default Numba implementations
from .elo import Elo, EloConfig
from .glicko import Glicko, GlickoConfig
from .glicko2 import Glicko2, Glicko2Config
from .trueskill import TrueSkill, TrueSkillConfig
from .yuksel import Yuksel, YukselConfig

# Batch systems (keep existing implementations)
from .whr import WHR
from .trueskill_through_time import TrueSkillThroughTime

__all__ = [
    # Default (Numba) implementations
    "Elo",
    "EloConfig",
    "Glicko",
    "GlickoConfig",
    "Glicko2",
    "Glicko2Config",
    "TrueSkill",
    "TrueSkillConfig",
    "Yuksel",
    "YukselConfig",
    # Batch systems
    "WHR",
    "TrueSkillThroughTime",
]
