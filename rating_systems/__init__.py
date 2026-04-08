"""
Rating Systems - High-performance Numba-accelerated implementations of rating systems.

This package provides modular, Numba-accelerated implementations of popular
rating systems with support for backtesting, evaluation, and checkpointing.

Available systems (online — incremental updates):
- Elo / WElo: Classic Elo rating system (weighted variant supports per-game weights)
- Glicko / WGlicko: Glicko with rating deviation
- Glicko2 / WGlicko2: Glicko-2 with volatility
- Stephenson / WStephenson: Extended Glicko with neighbourhood and bonus parameters
- TrueSkill / WeightedTrueSkill: Bayesian skill estimation with Gaussian beliefs
- Yuksel / WYuksel: Adaptive rating system with uncertainty tracking

Available systems (batch — full history refit):
- WHR / WeightedWHR: Whole History Rating
- TrueSkillThroughTime / WeightedTTT: TrueSkill Through Time
- SurfaceTTT: Surface-specific TTT (e.g. clay vs non-clay)

Analysis:
- HodgeIntransitivity: Hodge decomposition for measuring intransitivity

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
    # Weighted TTT
    WeightedTTT,
    WTTTConfig,
    # Weighted Glicko / Glicko-2
    WGlicko,
    WGlickoConfig,
    WGlicko2,
    WGlicko2Config,
    # Weighted WHR
    WeightedWHR,
    WWHRConfig,
    # Weighted Stephenson
    WStephenson,
    WStephensonConfig,
    # Weighted Yuksel
    WYuksel,
    WYukselConfig,
    # Weighted TrueSkill
    WeightedTrueSkill,
    WTrueSkillConfig,
    # Hodge intransitivity
    HodgeIntransitivity,
    HodgeConfig,
    # mElo (multidimensional Elo with intransitivity)
    MElo,
    MEloConfig,
    # Weighted mElo
    WMElo,
    WMEloConfig,
    # Matérn 3/2 TTT
    MaternTTT,
    MaternTTTConfig,
    WeightedMaternTTT,
    WMaternTTTConfig,
    # Margin TTT (Gaussian margin-of-victory)
    MarginTTT,
    MarginTTTConfig,
    WMarginTTT,
    WMarginTTTConfig,
    # Eigenvector centrality (B-Score)
    EigenvectorCentrality,
    EigenvectorConfig,
    WEigenvectorCentrality,
    WEigenvectorConfig,
    # G-Elo (ordinal margin-of-victory)
    GElo,
    GEloConfig,
    WGElo,
    WGEloConfig,
    # Disc decomposition (skill + consistency)
    Disc,
    DiscConfig,
    WDisc,
    WDiscConfig,
    # KickScore (GP paired comparisons)
    KickScoreRating,
    KickScoreConfig,
    # Serve/Return
    ServeReturnGlicko,
    ServeReturnElo,
    ServeReturnGlickoPoints,
    # GenElo (Bayesian Elo, Ingram 2021)
    GenElo,
    GenEloConfig,
    GenEloSurface,
    GenEloSurfaceConfig,
)
from .results import (
    FittedEloRatings,
    FittedGlickoRatings,
    FittedGlicko2Ratings,
    FittedWHRRatings,
    FittedTTTRatings,
    FittedTrueSkillRatings,
    FittedYukselRatings,
)
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
    "WeightedTTT",
    "WTTTConfig",
    "WGlicko",
    "WGlickoConfig",
    "WGlicko2",
    "WGlicko2Config",
    "WeightedWHR",
    "WWHRConfig",
    "WStephenson",
    "WStephensonConfig",
    "WYuksel",
    "WYukselConfig",
    "WeightedTrueSkill",
    "WTrueSkillConfig",
    "HodgeIntransitivity",
    "HodgeConfig",
    "MElo",
    "MEloConfig",
    "WMElo",
    "WMEloConfig",
    "MaternTTT",
    "MaternTTTConfig",
    "WeightedMaternTTT",
    "WMaternTTTConfig",
    "MarginTTT",
    "MarginTTTConfig",
    "WMarginTTT",
    "WMarginTTTConfig",
    "EigenvectorCentrality",
    "EigenvectorConfig",
    "WEigenvectorCentrality",
    "WEigenvectorConfig",
    "GElo",
    "GEloConfig",
    "WGElo",
    "WGEloConfig",
    "Disc",
    "DiscConfig",
    "WDisc",
    "WDiscConfig",
    "KickScoreRating",
    "KickScoreConfig",
    # GenElo (Bayesian Elo, Ingram 2021)
    "GenElo",
    "GenEloConfig",
    "GenEloSurface",
    "GenEloSurfaceConfig",
    # Fitted ratings (queryable results)
    "FittedEloRatings",
    "FittedGlickoRatings",
    "FittedGlicko2Ratings",
    "FittedWHRRatings",
    "FittedTTTRatings",
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
