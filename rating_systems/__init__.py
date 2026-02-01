"""
Rating Systems - High-performance implementations of Elo, Glicko, Glicko-2, and TrueSkill.

This package provides modular, Numba-accelerated implementations of popular
rating systems with support for backtesting and evaluation.

Default implementations use Numba for high performance on CPU. PyTorch-based
implementations are available for GPU acceleration on very large datasets.

Quick Start:
    from rating_systems import GameDataset, Elo, Glicko, Glicko2, TrueSkill, Backtester

    # Load data
    dataset = GameDataset.from_parquet("games.parquet")

    # Create and fit a rating system (uses fast Numba backend)
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

For GPU acceleration (requires PyTorch, import explicitly):
    from rating_systems.systems.elo.elo_torch import EloTorch
    from rating_systems.systems.glicko.glicko_torch import GlickoTorch
    from rating_systems.systems.glicko2.glicko2_torch import Glicko2Torch

    elo_gpu = EloTorch(k_factor=32)  # Uses CUDA if available
    elo_gpu.fit(dataset)
"""

from .data import GameDataset, GameBatch, TorchGameBatch, PredictionResult
from .base import RatingSystem, RatingSystemType, PlayerRatings, TorchPlayerRatings
from .systems import (
    # Default (Numba) implementations
    Elo,
    EloConfig,
    Glicko,
    GlickoConfig,
    Glicko2,
    Glicko2Config,
    TrueSkill,
    TrueSkillConfig,
    Yuksel,
    YukselConfig,
    # Batch systems
    WHR,
    TrueSkillThroughTime,
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
from .utils import get_device, is_torch_available

__version__ = "0.1.0"

__all__ = [
    # Data
    "GameDataset",
    "GameBatch",
    "TorchGameBatch",
    "PredictionResult",
    # Base
    "RatingSystem",
    "RatingSystemType",
    "PlayerRatings",
    "TorchPlayerRatings",
    # Default (Numba) systems
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
    # Fitted ratings (queryable results)
    "FittedEloRatings",
    "FittedGlickoRatings",
    "FittedTrueSkillRatings",
    "FittedYukselRatings",
    # Batch systems
    "WHR",
    "TrueSkillThroughTime",
    # Evaluation
    "Backtester",
    "BacktestResult",
    "DayResult",
    "brier_score",
    "log_loss",
    "accuracy",
    "calibration_error",
    "compare_systems",
    # Utils
    "get_device",
    "is_torch_available",
]
