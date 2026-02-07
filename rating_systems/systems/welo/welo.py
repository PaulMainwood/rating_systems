"""
Weighted Elo (WElo) rating system - Elo with per-game weights and handicaps.

The update rule is:
    expected = sigmoid((r1 - r2 + handicap) * log10/scale)
    delta = K * w * (score - expected)

where w is a per-game weight and handicap is a per-game advantage for
player 1 (in Elo points). When w=1 and handicap=0 for all games, this
is identical to standard Elo.

This is a general-purpose weighted Elo. It knows nothing about surfaces
or any other domain-specific concept - it simply accepts a weight and
handicap for each game.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ._numba_core import (
    update_ratings_weighted_sequential,
    predict_proba_batch,
    predict_single,
    fit_all_days_weighted,
)


@dataclass
class WEloConfig:
    """Configuration for Weighted Elo rating system."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0


class WElo(RatingSystem):
    """
    Weighted Elo rating system with Numba acceleration.

    Like standard Elo, but each game has an associated weight that scales
    the rating update. This allows certain games to have more or less
    influence on the rating.

    Update rule: delta = K * w * (score - expected)

    When weights are not provided, all default to 1.0 (standard Elo).

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        k_factor: Maximum rating change per game (default: 32)
        scale: Rating difference where one player is 10x stronger (default: 400)

    Example:
        >>> welo = WElo(k_factor=47)
        >>> weights = np.array([1.0, 0.8, 0.8, 1.0, ...])
        >>> welo.fit(dataset, weights=weights)
        >>> print(welo.predict_proba(0, 1))
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        scale: float = 400.0,
        num_players: Optional[int] = None,
    ):
        self.config = WEloConfig(
            initial_rating=initial_rating,
            k_factor=k_factor,
            scale=scale,
        )
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial WElo ratings for all players."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "welo", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with uniform weights and no handicaps (standard Elo behaviour)."""
        if len(batch) == 0:
            return

        n = len(batch)
        weights = np.ones(n, dtype=np.float64)
        handicaps = np.zeros(n, dtype=np.float64)
        update_ratings_weighted_sequential(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
            handicaps,
            ratings.ratings,
            self.config.k_factor,
            self.config.scale,
        )
        self._num_games_fitted += n

    def update_weighted(
        self,
        batch: GameBatch,
        weights: np.ndarray,
        handicaps: Optional[np.ndarray] = None,
    ) -> "WElo":
        """
        Incrementally update ratings with per-game weights and handicaps.

        Args:
            batch: Games to process
            weights: Per-game weights array (same length as batch)
            handicaps: Per-game handicap for player 1 in Elo points.
                       Positive = player 1 advantage. If None, all zero.

        Returns:
            self (for method chaining)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before updating. Call fit() first.")

        if len(batch) == 0:
            return self

        n = len(batch)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if handicaps is None:
            h = np.zeros(n, dtype=np.float64)
        else:
            h = np.ascontiguousarray(handicaps, dtype=np.float64)
        update_ratings_weighted_sequential(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
            h,
            self._ratings.ratings,
            self.config.k_factor,
            self.config.scale,
        )
        self._num_games_fitted += n
        self._current_day = batch.day
        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        handicaps: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        Args:
            player1: Single player ID or array of player IDs
            player2: Single player ID or array of player IDs
            handicaps: Per-game handicap for player 1 in Elo points.
                       Single float for single prediction, array for batch.
                       Positive = player 1 advantage. If None, all zero.

        Returns:
            Single probability or array of probabilities
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            h = 0.0 if handicaps is None else float(handicaps)
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
                self.config.scale,
                h,
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        if handicaps is None:
            h = np.zeros(len(p1), dtype=np.float64)
        else:
            h = np.ascontiguousarray(handicaps, dtype=np.float64)
        return predict_proba_batch(p1, p2, self._ratings.ratings, self.config.scale, h)

    def fit(
        self,
        dataset: GameDataset,
        weights: Optional[np.ndarray] = None,
        handicaps: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WElo":
        """
        Fit the rating system on a dataset with optional per-game weights and handicaps.

        Uses optimised single Numba call to process all days without
        Python iteration overhead.

        Args:
            dataset: Game dataset to fit on
            weights: Per-game weights array. Must match the number of games
                     in the dataset (after any end_day filtering). If None,
                     all weights default to 1.0.
            handicaps: Per-game handicap for player 1 in Elo points.
                       Positive = player 1 advantage. If None, all zero.
            end_day: Last day to include (inclusive). Cannot be used together
                     with weights - pre-filter the dataset instead.
            player_names: Optional mapping of player_id -> name

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

        # Initialize if needed
        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        # Get pre-batched arrays for direct Numba processing
        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is not None and len(player1) > 0:
            n = len(player1)
            if weights is None:
                w = np.ones(n, dtype=np.float64)
            else:
                w = np.ascontiguousarray(weights, dtype=np.float64)

            if handicaps is None:
                h = np.zeros(n, dtype=np.float64)
            else:
                h = np.ascontiguousarray(handicaps, dtype=np.float64)

            fit_all_days_weighted(
                player1,
                player2,
                scores,
                w,
                h,
                day_offsets,
                self._ratings.ratings,
                self.config.k_factor,
                self.config.scale,
            )
            self._num_games_fitted = n
            self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        else:
            self._num_games_fitted = 0

        self._fitted = True
        return self

    def reset(self) -> "WElo":
        """Reset the rating system to initial state."""
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WElo(k_factor={self.config.k_factor}, "
            f"initial_rating={self.config.initial_rating}, "
            f"players={players}, {status})"
        )
