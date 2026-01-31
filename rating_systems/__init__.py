"""
Rating Systems - High-performance implementations of Elo, Glicko, and Glicko-2.

This package provides modular, Numba-accelerated implementations of popular
rating systems with support for backtesting and evaluation.

Default implementations use Numba for high performance on CPU. PyTorch-based
implementations are available for GPU acceleration on very large datasets.

Quick Start:
    from rating_systems import GameDataset, Elo, Glicko, Glicko2, Backtester

    # Load data
    dataset = GameDataset.from_parquet("games.parquet")

    # Create and fit a rating system (uses fast Numba backend)
    elo = Elo(k_factor=32)
    elo.fit(dataset)

    # Get queryable fitted ratings
    fitted = elo.get_fitted_ratings()
    print(fitted.top(10))  # Top 10 players
    print(fitted.predict(0, 1))  # P(player 0 beats player 1)

    # Backtest
    backtester = Backtester(elo, dataset)
    results = backtester.run()

Command-line interface:
    python -m rating_systems fit data.parquet --top 20
    python -m rating_systems predict data.parquet 0 1
    python -m rating_systems backtest data.parquet

For GPU acceleration (requires PyTorch):
    from rating_systems import EloTorch, GlickoTorch, Glicko2Torch

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
    # PyTorch implementations
    EloTorch,
    GlickoTorch,
    Glicko2Torch,
    # Batch systems
    WHR,
    TrueSkillThroughTime,
)
from .results import FittedEloRatings, FittedGlickoRatings
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
    # Fitted ratings (queryable results)
    "FittedEloRatings",
    "FittedGlickoRatings",
    # PyTorch systems
    "EloTorch",
    "GlickoTorch",
    "Glicko2Torch",
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
