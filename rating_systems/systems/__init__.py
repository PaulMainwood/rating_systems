"""Rating system implementations.

Default implementations (Numba-based, high performance):
- Elo: Classic Elo rating system
- Glicko: Glicko rating system with rating deviation
- Glicko2: Glicko-2 rating system with volatility

PyTorch implementations (GPU acceleration for large datasets):
- EloTorch: Elo with PyTorch backend
- GlickoTorch: Glicko with PyTorch backend
- Glicko2Torch: Glicko-2 with PyTorch backend

Batch systems (refit on full history):
- WHR: Whole History Rating
- TrueSkillThroughTime: TrueSkill Through Time
"""

# Default Numba implementations
from .elo import Elo, EloConfig, EloTorch
from .glicko import Glicko, GlickoConfig, GlickoTorch
from .glicko2 import Glicko2, Glicko2Config, Glicko2Torch

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
    # PyTorch implementations
    "EloTorch",
    "GlickoTorch",
    "Glicko2Torch",
    # Batch systems
    "WHR",
    "TrueSkillThroughTime",
]
