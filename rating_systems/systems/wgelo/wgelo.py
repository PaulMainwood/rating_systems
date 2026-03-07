"""
Weighted G-Elo (WGElo) — G-Elo with per-game weights and handicaps.

When w=1 and handicap=0 for all games, this is identical to standard GElo.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ..gelo._numba_core import (
    update_ratings_weighted_sequential,
    predict_proba_batch,
    predict_single,
    predict_proba_with_handicap_batch,
    predict_single_with_handicap,
    fit_all_days_weighted,
    walk_forward_predict_update,
)


@dataclass
class WGEloConfig:
    """Configuration for Weighted G-Elo."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0
    threshold_spread: float = 0.5


class WGElo(RatingSystem):
    """
    Weighted G-Elo rating system with Numba acceleration.

    Like standard GElo, but each game has an associated weight that scales
    the update, plus an optional handicap that shifts the expected score
    for player 1.

    Parameters:
        initial_rating: Starting rating (default: 1500)
        k_factor: Maximum rating change per game (default: 32)
        scale: Logistic scale (default: 400)
        threshold_spread: Distance between ordinal thresholds (default: 0.5)
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        scale: float = 400.0,
        threshold_spread: float = 0.5,
        num_players: Optional[int] = None,
    ):
        self.config = WGEloConfig(
            initial_rating=initial_rating,
            k_factor=k_factor,
            scale=scale,
            threshold_spread=threshold_spread,
        )
        self._thresholds = np.array(
            [-threshold_spread, 0.0, threshold_spread], dtype=np.float64)
        self._margins: Optional[np.ndarray] = None
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial ratings."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "wgelo", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update with uniform weights (standard G-Elo behaviour)."""
        if len(batch) == 0:
            return

        n = len(batch)
        start = self._num_games_fitted
        end = start + n
        if self._margins is not None and end <= len(self._margins):
            margins = self._margins[start:end]
        else:
            margins = np.where(batch.scores > 0.5, 2.0, 1.0)

        weights = np.ones(n, dtype=np.float64)
        handicaps = np.zeros(n, dtype=np.float64)
        update_ratings_weighted_sequential(
            batch.player1,
            batch.player2,
            margins,
            weights,
            handicaps,
            ratings.ratings,
            self.config.k_factor,
            self.config.scale,
            self._thresholds,
        )
        self._num_games_fitted += n

    def update_weighted(
        self,
        batch: GameBatch,
        weights: np.ndarray,
        handicaps: Optional[np.ndarray] = None,
        margins: Optional[np.ndarray] = None,
    ) -> "WGElo":
        """
        Incrementally update ratings with per-game weights and handicaps.

        Args:
            batch: Games to process
            weights: Per-game weights
            handicaps: Per-game handicap for player 1 in Elo points
            margins: Per-game ordinal margin categories. If None,
                     infers from binary scores.

        Returns:
            self (for method chaining)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before updating. Call fit() first.")

        if len(batch) == 0:
            return self

        n = len(batch)
        w = np.ascontiguousarray(weights, dtype=np.float64)
        h = np.zeros(n, dtype=np.float64) if handicaps is None else \
            np.ascontiguousarray(handicaps, dtype=np.float64)

        if margins is not None:
            m = np.ascontiguousarray(margins, dtype=np.float64)
        else:
            m = np.where(batch.scores > 0.5, 2.0, 1.0)

        update_ratings_weighted_sequential(
            batch.player1,
            batch.player2,
            m,
            w,
            h,
            self._ratings.ratings,
            self.config.k_factor,
            self.config.scale,
            self._thresholds,
        )
        self._num_games_fitted += n
        self._current_day = batch.day
        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        handicaps: Optional[Union[float, np.ndarray]] = None,
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict P(player1 wins)."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            if handicaps is not None:
                return predict_single_with_handicap(
                    self._ratings.ratings[int(player1)],
                    self._ratings.ratings[int(player2)],
                    self.config.scale,
                    float(handicaps),
                )
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
                self.config.scale,
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        if handicaps is not None:
            h = np.ascontiguousarray(handicaps, dtype=np.float64)
            return predict_proba_with_handicap_batch(
                p1, p2, self._ratings.ratings, self.config.scale, h)
        return predict_proba_batch(p1, p2, self._ratings.ratings, self.config.scale)

    def fit(
        self,
        dataset: GameDataset,
        margins: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        handicaps: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WGElo":
        """
        Fit WGElo on a dataset with optional weights and handicaps.

        Args:
            dataset: Game dataset to fit on
            margins: Per-game ordinal margin categories. If None, infers
                     from binary scores (middle-margin).
            weights: Per-game weights. If None, all 1.0.
            handicaps: Per-game handicaps. If None, all 0.0.
            end_day: Last day to include.
            player_names: Optional player names.

        Returns:
            self (for method chaining)
        """
        self._player_names = player_names

        if end_day is not None and weights is not None:
            raise ValueError(
                "Cannot use both end_day and weights. Pre-filter the dataset "
                "and pass the corresponding weights instead."
            )

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is None or len(player1) == 0:
            self._num_games_fitted = 0
            self._fitted = True
            return self

        n = len(player1)

        # Margins
        if margins is not None:
            m = np.ascontiguousarray(margins, dtype=np.float64)
        else:
            m = np.where(scores > 0.5, 2.0, 1.0)
        self._margins = m

        # Weights and handicaps
        w = np.ones(n, dtype=np.float64) if weights is None else \
            np.ascontiguousarray(weights, dtype=np.float64)
        h = np.zeros(n, dtype=np.float64) if handicaps is None else \
            np.ascontiguousarray(handicaps, dtype=np.float64)

        fit_all_days_weighted(
            player1, player2, m, w, h, day_offsets,
            self._ratings.ratings,
            self.config.k_factor,
            self.config.scale,
            self._thresholds,
        )
        self._num_games_fitted = n
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None

        self._fitted = True
        return self

    def fused_walk_forward(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        day_indices: np.ndarray,
        day_offsets: np.ndarray,
        weights: np.ndarray,
        handicaps: np.ndarray,
        n_train_days: int,
    ) -> np.ndarray:
        """Fused predict+update walk-forward in a single Numba call.

        Eliminates Python per-day overhead. Modifies ratings in-place.
        Returns predictions array (NaN for training period).
        """
        margins = np.where(scores > 0.5, 2.0, 1.0)
        return walk_forward_predict_update(
            player1, player2, scores, margins, day_offsets,
            weights, handicaps,
            self._ratings.ratings,
            self.config.k_factor, self.config.scale,
            self._thresholds,
            n_train_days,
        )

    def reset(self) -> "WGElo":
        """Reset to initial state."""
        self._margins = None
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WGElo(k_factor={self.config.k_factor}, "
            f"threshold_spread={self.config.threshold_spread}, "
            f"players={players}, {status})"
        )
