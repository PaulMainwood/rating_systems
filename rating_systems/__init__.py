"""
Rating Systems - High-performance Numba-accelerated implementations of rating systems.

This package provides modular, Numba-accelerated implementations of popular
rating systems with support for backtesting and evaluation.

Available systems:
- Elo: Classic Elo rating system
- Glicko: Glicko rating system with rating deviation
- Glicko2: Glicko-2 rating system with volatility
- Stephenson: Extended Glicko with neighbourhood and bonus parameters
- TrueSkill: Bayesian skill estimation with Gaussian beliefs
- Yuksel: Adaptive rating system with uncertainty tracking
- WHR: Whole History Rating (batch)
- TrueSkillThroughTime: TrueSkill Through Time (batch)

Quick Start:
    from rating_systems import GameDataset, Elo, Glicko, Glicko2, TrueSkill, Backtester

    # Load data
    dataset = GameDataset.from_parquet("games.parquet")

    # Create and fit a rating system
    elo = Elo(k_factor=32)
    elo.fit(dataset)

    # Get queryable fitted ratings
    fitted = elo.get_fitted_ratings()
    print(fitted.top(10))  # Top 10 players
    print(fitted.predict(0, 1))  # P(player 0 beats player 1)

    # TrueSkill with uncertainty tracking
    ts = TrueSkill()
    ts.fit(dataset)
    fitted_ts = ts.get_fitted_ratings()
    print(fitted_ts.conservative_top(10))  # Top 10 by mu - 3*sigma

    # Backtest
    backtester = Backtester(elo, dataset)
    results = backtester.run()

Command-line interface:
    python -m rating_systems fit data.parquet --top 20
    python -m rating_systems predict data.parquet 0 1
    python -m rating_systems backtest data.parquet
"""

from .data import GameDataset, GameBatch, PredictionResult
from .base import RatingSystem, RatingSystemType, PlayerRatings
from .systems import (
    # Default (Numba) implementations
    Elo,
    EloConfig,
    WElo,
    WEloConfig,
    Glicko,
    GlickoConfig,
    Glicko2,
    Glicko2Config,
    Stephenson,
    StephensonConfig,
    TrueSkill,
    TrueSkillConfig,
    Yuksel,
    YukselConfig,
    # Batch systems
    WHR,
    TrueSkillThroughTime,
    SurfaceTTT,
    # Surface constants
    SURFACE_HARD,
    SURFACE_CLAY,
    SURFACE_GRASS,
    SURFACE_NON_CLAY,
    SURFACE_NAMES,
)
from .results import FittedEloRatings, FittedGlickoRatings, FittedTrueSkillRatings, FittedYukselRatings
from .evaluation import (
    Backtester,
    BacktestResult,
    DayResult,
    brier_score,
    log_loss,
    accuracy,
    calibration_error,
    compare_systems,
)

__version__ = "0.1.0"

__all__ = [
    # Data
    "GameDataset",
    "GameBatch",
    "PredictionResult",
    # Base
    "RatingSystem",
    "RatingSystemType",
    "PlayerRatings",
    # Rating systems
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
    "WHR",
    "TrueSkillThroughTime",
    "SurfaceTTT",
    # Fitted ratings (queryable results)
    "FittedEloRatings",
    "FittedGlickoRatings",
    "FittedTrueSkillRatings",
    "FittedYukselRatings",
    # Surface constants
    "SURFACE_HARD",
    "SURFACE_CLAY",
    "SURFACE_GRASS",
    "SURFACE_NON_CLAY",
    "SURFACE_NAMES",
    # Evaluation
    "Backtester",
    "BacktestResult",
    "DayResult",
    "brier_score",
    "log_loss",
    "accuracy",
    "calibration_error",
    "compare_systems",
]
