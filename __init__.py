"""
Rating Systems - PyTorch-based implementations of Elo, Glicko, and Glicko-2.

This package provides modular, GPU-accelerated implementations of popular
rating systems with support for backtesting and evaluation.

Quick Start:
    from rating_systems import GameDataset, Elo, Glicko, Glicko2, Backtester

    # Load data
    dataset = GameDataset.from_parquet("games.parquet")

    # Create and fit a rating system
    elo = Elo(k_factor=32)
    elo.fit(dataset)

    # Get ratings
    ratings = elo.get_ratings()
    print(ratings.to_dataframe())

    # Backtest
    backtester = Backtester(elo, dataset)
    results = backtester.run()
"""

from .data import GameDataset, GameBatch
from .base import RatingSystem, RatingSystemType, PlayerRatings
from .systems import Elo, Glicko, Glicko2, WHR, TrueSkillThroughTime
from .evaluation import (
    Backtester,
    BacktestResult,
    DayResult,
    brier_score,
    log_loss,
    accuracy,
    compare_systems,
)
from .utils import get_device

__version__ = "0.1.0"

__all__ = [
    # Data
    "GameDataset",
    "GameBatch",
    # Base
    "RatingSystem",
    "RatingSystemType",
    "PlayerRatings",
    # Systems
    "Elo",
    "Glicko",
    "Glicko2",
    "WHR",
    "TrueSkillThroughTime",
    # Evaluation
    "Backtester",
    "BacktestResult",
    "DayResult",
    "brier_score",
    "log_loss",
    "accuracy",
    "compare_systems",
    # Utils
    "get_device",
]
