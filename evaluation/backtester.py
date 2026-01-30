"""Backtesting framework for rating systems."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import torch

from ..base import RatingSystem
from ..data import GameDataset
from .metrics import accuracy, brier_score, log_loss


@dataclass
class DayResult:
    """Results for a single day of backtesting."""

    day: int
    num_games: int
    brier: float
    log_loss: float
    accuracy: float

    def to_dict(self) -> Dict:
        return {
            "day": self.day,
            "num_games": self.num_games,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "accuracy": self.accuracy,
        }


@dataclass
class BacktestResult:
    """Complete backtesting results."""

    system_name: str
    train_days: int
    test_days: int
    total_games: int

    # Aggregate metrics (weighted by number of games per day)
    brier: float
    log_loss: float
    accuracy: float

    # Per-day results
    daily_results: List[DayResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert daily results to DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.daily_results])

    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"{self.system_name} Backtest Results:\n"
            f"  Train days: {self.train_days}, Test days: {self.test_days}\n"
            f"  Total games: {self.total_games}\n"
            f"  Brier Score: {self.brier:.4f}\n"
            f"  Log Loss: {self.log_loss:.4f}\n"
            f"  Accuracy: {self.accuracy:.4f}"
        )


class Backtester:
    """
    Backtesting framework for rating systems.

    Performs walk-forward validation:
    1. Train on historical data up to day T-1
    2. Predict outcomes for day T
    3. Evaluate predictions against actual results
    4. Update model with day T data
    5. Move to day T+1 and repeat

    This simulates real-world usage where predictions are made
    before seeing the actual game results.
    """

    def __init__(self, system: RatingSystem, dataset: GameDataset):
        """
        Initialize backtester.

        Args:
            system: Rating system to backtest
            dataset: Full dataset containing all games
        """
        self.system = system
        self.dataset = dataset

    def run(
        self,
        train_end_day: Optional[int] = None,
        test_start_day: Optional[int] = None,
        test_end_day: Optional[int] = None,
        verbose: bool = True,
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            train_end_day: Last day of initial training (inclusive).
                          If None, uses first 70% of days.
            test_start_day: First day of testing. If None, uses train_end_day + 1.
            test_end_day: Last day of testing (inclusive). If None, uses last day.
            verbose: Whether to print progress.

        Returns:
            BacktestResult with aggregate and per-day metrics
        """
        days = self.dataset.days

        # Determine train/test split
        if train_end_day is None:
            split_idx = int(len(days) * 0.7)
            train_end_day = days[split_idx - 1] if split_idx > 0 else days[0]

        if test_start_day is None:
            test_start_day = train_end_day + 1

        if test_end_day is None:
            test_end_day = days[-1]

        # Get test days
        test_days = [d for d in days if test_start_day <= d <= test_end_day]

        if len(test_days) == 0:
            raise ValueError(f"No test days in range [{test_start_day}, {test_end_day}]")

        # Initial training
        train_dataset = self.dataset.filter_days(end_day=train_end_day)
        train_days_count = len([d for d in days if d <= train_end_day])

        if verbose:
            print(f"Training on days up to {train_end_day} ({train_dataset.num_games} games)...")

        self.system.reset()
        self.system.fit(train_dataset)

        # Walk-forward testing
        daily_results: List[DayResult] = []
        all_predictions: List[torch.Tensor] = []
        all_actuals: List[torch.Tensor] = []

        if verbose:
            print(f"Testing on {len(test_days)} days [{test_start_day} to {test_end_day}]...")

        for i, day in enumerate(test_days):
            try:
                batch = self.dataset.get_day(day)
            except ValueError:
                continue  # Skip days with no games

            # Predict before updating
            predictions = self.system.predict_proba(batch.player1, batch.player2)

            # Calculate metrics
            day_brier = brier_score(predictions, batch.scores).item()
            day_log_loss = log_loss(predictions, batch.scores).item()
            day_accuracy = accuracy(predictions, batch.scores).item()

            daily_results.append(DayResult(
                day=day,
                num_games=len(batch),
                brier=day_brier,
                log_loss=day_log_loss,
                accuracy=day_accuracy,
            ))

            all_predictions.append(predictions)
            all_actuals.append(batch.scores)

            # Update model with this day's results
            self.system.update(batch)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed day {i + 1}/{len(test_days)}")

        # Calculate aggregate metrics
        all_preds = torch.cat(all_predictions)
        all_acts = torch.cat(all_actuals)

        total_games = len(all_preds)
        agg_brier = brier_score(all_preds, all_acts).item()
        agg_log_loss = log_loss(all_preds, all_acts).item()
        agg_accuracy = accuracy(all_preds, all_acts).item()

        result = BacktestResult(
            system_name=self.system.__class__.__name__,
            train_days=train_days_count,
            test_days=len(test_days),
            total_games=total_games,
            brier=agg_brier,
            log_loss=agg_log_loss,
            accuracy=agg_accuracy,
            daily_results=daily_results,
        )

        if verbose:
            print(result.summary())

        return result


def compare_systems(
    systems: List[RatingSystem],
    dataset: GameDataset,
    train_end_day: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare multiple rating systems on the same dataset.

    Args:
        systems: List of rating systems to compare
        dataset: Game dataset
        train_end_day: Last day of initial training
        verbose: Whether to print progress

    Returns:
        DataFrame with comparison results
    """
    results = []

    for system in systems:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Backtesting {system.__class__.__name__}...")
            print('='*50)

        backtester = Backtester(system, dataset)
        result = backtester.run(train_end_day=train_end_day, verbose=verbose)

        results.append({
            "system": result.system_name,
            "brier": result.brier,
            "log_loss": result.log_loss,
            "accuracy": result.accuracy,
            "train_days": result.train_days,
            "test_days": result.test_days,
            "test_games": result.total_games,
        })

    return pd.DataFrame(results)
