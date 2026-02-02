"""
Glicko-2 rating system - high-performance Numba implementation.

Extension of Glicko that adds a volatility parameter to model
rating stability. Uses internal Glicko-2 scale for calculations.

This implementation prioritizes efficiency through:
1. Numba JIT compilation of all hot paths
2. Single Numba call to process all days (no Python iteration)
3. Contiguous numpy arrays throughout
4. Parallel prediction via prange
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...results.fitted_ratings import FittedGlicko2Ratings
from ._numba_core import (
    fit_all_days,
    update_ratings_batch,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
)


@dataclass
class Glicko2Config:
    """Configuration for Glicko-2 rating system."""

    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    initial_volatility: float = 0.06
    tau: float = 0.5  # System constant (typically 0.3 to 1.2)
    epsilon: float = 0.000001  # Convergence tolerance
    scale: float = 173.7178  # Conversion factor from Glicko to Glicko-2 scale
    max_rd: float = 350.0


class Glicko2(RatingSystem):
    """
    Glicko-2 rating system with Numba acceleration.

    Extension of Glicko that adds a volatility parameter to model
    rating stability. Uses internal Glicko-2 scale for calculations.

    Performance characteristics:
    - Fit: O(games * players_per_day) with single Numba call
    - Predict: O(n) parallel across matchups
    - Memory: O(num_players) for ratings arrays

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        initial_rd: Starting rating deviation (default: 350)
        initial_volatility: Starting volatility (default: 0.06)
        tau: System constant controlling volatility change (default: 0.5)

    Example:
        >>> glicko2 = Glicko2(tau=0.5)
        >>> glicko2.fit(dataset)
        >>> fitted = glicko2.get_fitted_ratings()
        >>> print(fitted.top(10))  # Top 10 players with RD and volatility
        >>> print(fitted.predict(0, 1))  # P(player 0 beats player 1)
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        initial_volatility: float = 0.06,
        tau: float = 0.5,
        num_players: Optional[int] = None,
    ):
        self.config = Glicko2Config(
            initial_rating=initial_rating,
            initial_rd=initial_rd,
            initial_volatility=initial_volatility,
            tau=tau,
        )
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Glicko-2 ratings for all players."""
        # Store internally in Glicko-2 scale (mu, phi)
        return PlayerRatings(
            ratings=np.zeros(num_players, dtype=np.float64),  # mu
            rd=np.full(
                num_players,
                self.config.initial_rd / self.config.scale,  # phi
                dtype=np.float64,
            ),
            volatility=np.full(
                num_players,
                self.config.initial_volatility,
                dtype=np.float64,
            ),
            last_played=np.zeros(num_players, dtype=np.int32),
            metadata={"system": "glicko2", "config": self.config},
        )

    def _to_glicko_scale(self, mu: np.ndarray) -> np.ndarray:
        """Convert mu from Glicko-2 scale to Glicko scale."""
        return mu * self.config.scale + self.config.initial_rating

    def _to_glicko_rd(self, phi: np.ndarray) -> np.ndarray:
        """Convert phi from Glicko-2 scale to Glicko RD."""
        return phi * self.config.scale

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update Glicko-2 ratings for a rating period."""
        if len(batch) == 0:
            return

        update_ratings_batch(
            batch.player1,
            batch.player2,
            batch.scores,
            ratings.ratings,  # mu
            ratings.rd,       # phi
            ratings.volatility,
            ratings.last_played,
            batch.day,
            self.config.tau,
            self.config.epsilon,
            self.config.max_rd / self.config.scale,  # max_phi
        )
        self._num_games_fitted += len(batch)

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "Glicko2":
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
                day_indices,
                day_offsets,
                self._ratings.ratings,      # mu
                self._ratings.rd,           # phi
                self._ratings.volatility,
                self._ratings.last_played,
                self.config.tau,
                self.config.epsilon,
                self.config.max_rd / self.config.scale,  # max_phi
            )
            self._num_games_fitted = len(player1)
            self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        else:
            self._num_games_fitted = 0

        self._fitted = True
        return self

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
            p1, p2 = int(player1), int(player2)
            return predict_single(
                self._ratings.ratings[p1],
                self._ratings.rd[p1],
                self._ratings.ratings[p2],
                self._ratings.rd[p2],
            )

        # Batch prediction
        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(
            p1, p2, self._ratings.ratings, self._ratings.rd
        )

    def get_fitted_ratings(self) -> FittedGlicko2Ratings:
        """
        Get a queryable fitted ratings object.

        Returns:
            FittedGlicko2Ratings with methods for querying results
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return FittedGlicko2Ratings(
            ratings=self._to_glicko_scale(self._ratings.ratings),
            rd=self._to_glicko_rd(self._ratings.rd),
            volatility=self._ratings.volatility.copy(),
            last_played=self._ratings.last_played.copy(),
            scale=self.config.scale,
            initial_rating=self.config.initial_rating,
            initial_rd=self.config.initial_rd,
            initial_volatility=self.config.initial_volatility,
            tau=self.config.tau,
            num_games_fitted=self._num_games_fitted,
            last_day=self._current_day,
            player_names=self._player_names,
        )

    def get_ratings(self) -> PlayerRatings:
        """Get current player ratings in Glicko scale."""
        if self._ratings is None:
            raise ValueError("No ratings available. Call fit() first.")

        # Convert to Glicko scale for output
        return PlayerRatings(
            ratings=self._to_glicko_scale(self._ratings.ratings),
            rd=self._to_glicko_rd(self._ratings.rd),
            volatility=self._ratings.volatility.copy(),
            last_played=self._ratings.last_played.copy(),
            metadata={"system": "glicko2", "config": self.config},
        )

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players (convenience method)."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return get_top_n_indices(self._ratings.ratings, n)

    def reset(self) -> "Glicko2":
        """Reset the rating system to initial state."""
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"Glicko2(tau={self.config.tau}, "
            f"initial_volatility={self.config.initial_volatility}, "
            f"players={players}, {status})"
        )
