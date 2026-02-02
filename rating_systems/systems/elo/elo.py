"""
Elo rating system - high-performance Numba implementation.

This implementation prioritizes efficiency through:
1. Numba JIT compilation of all hot paths
2. Contiguous numpy arrays throughout
3. Zero allocation in update loops
4. Parallel prediction via prange
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union, List

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...results.fitted_ratings import FittedEloRatings
from ._numba_core import (
    update_ratings_sequential,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
    fit_all_days,
)


@dataclass
class EloConfig:
    """Configuration for Elo rating system."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0  # Rating difference for 10x expected score


class Elo(RatingSystem):
    """
    Elo rating system with Numba acceleration.

    The classic rating system developed by Arpad Elo. Uses logistic
    expected score and simple update rule.

    Performance characteristics:
    - Update: O(n) sequential (required for correctness)
    - Predict: O(n) parallel across matchups
    - Memory: O(num_players) for ratings array

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        k_factor: Maximum rating change per game (default: 32)
        scale: Rating difference where one player is 10x stronger (default: 400)

    Example:
        >>> elo = Elo(k_factor=32)
        >>> elo.fit(dataset)
        >>> fitted = elo.get_fitted_ratings()
        >>> print(fitted.top(10))  # Top 10 players
        >>> print(fitted.predict(0, 1))  # P(player 0 beats player 1)
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        scale: float = 400.0,
        num_players: Optional[int] = None,
    ):
        self.config = EloConfig(
            initial_rating=initial_rating,
            k_factor=k_factor,
            scale=scale,
        )
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Elo ratings for all players."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "elo", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update Elo ratings for a batch of games."""
        if len(batch) == 0:
            return

        update_ratings_sequential(
            batch.player1,
            batch.player2,
            batch.scores,
            ratings.ratings,
            self.config.k_factor,
            self.config.scale,
        )
        self._num_games_fitted += len(batch)

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        Args:
            player1: Single player ID or array of player IDs
            player2: Single player ID or array of player IDs

        Returns:
            Single probability or array of probabilities
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle single prediction
        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
                self.config.scale,
            )

        # Batch prediction
        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(p1, p2, self._ratings.ratings, self.config.scale)

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "Elo":
        """
        Fit the rating system on a dataset.

        Uses optimized single Numba call to process all days without
        Python iteration overhead.

        Args:
            dataset: Game dataset to fit on
            end_day: Last day to include (inclusive)
            player_names: Optional mapping of player_id -> name

        Returns:
            self (for method chaining)
        """
        self._player_names = player_names

        # Filter dataset if end_day specified
        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        # Initialize if needed
        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        # Get pre-batched arrays for direct Numba processing
        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is not None and len(player1) > 0:
            # Process ALL days in single Numba call - no Python iteration!
            fit_all_days(
                player1,
                player2,
                scores,
                day_offsets,
                self._ratings.ratings,
                self.config.k_factor,
                self.config.scale,
            )
            self._num_games_fitted = len(player1)
            self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        else:
            self._num_games_fitted = 0

        self._fitted = True
        return self

    def get_fitted_ratings(self) -> FittedEloRatings:
        """
        Get a queryable fitted ratings object.

        Returns:
            FittedEloRatings with methods for querying results
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return FittedEloRatings(
            ratings=self._ratings.ratings.copy(),
            scale=self.config.scale,
            initial_rating=self.config.initial_rating,
            k_factor=self.config.k_factor,
            num_games_fitted=self._num_games_fitted,
            last_day=self._current_day,
            player_names=self._player_names,
        )

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players (convenience method)."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return get_top_n_indices(self._ratings.ratings, n)

    def reset(self) -> "Elo":
        """Reset the rating system to initial state."""
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"Elo(k_factor={self.config.k_factor}, "
            f"initial_rating={self.config.initial_rating}, "
            f"players={players}, {status})"
        )
