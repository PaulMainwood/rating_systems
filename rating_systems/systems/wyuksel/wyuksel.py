"""
Weighted Yuksel (WYuksel) rating system - Yuksel with per-game weights.

Per-game weights w scale the forces and curvature contribution of each game.
A game with higher weight has more influence on the rating update.
When weights=None, all default to 1.0 (identical to standard Yuksel).

This is a general-purpose weighted Yuksel. It knows nothing about surfaces
or any other domain-specific concept - it simply accepts a weight per game.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...results.fitted_ratings import FittedYukselRatings
from ._numba_core import (
    update_ratings_sequential_weighted,
    fit_all_days_weighted,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
    compute_phi,
)


@dataclass
class WYukselConfig:
    """Configuration for Weighted Yuksel rating system."""

    initial_rating: float = 1500.0
    delta_r_max: float = 350.0
    alpha: float = 2.0
    scaling_factor: float = 0.9
    initial_weight: float = 0.01


class WYuksel(RatingSystem):
    """
    Weighted Yuksel rating system with Numba acceleration.

    Like standard Yuksel, but each game has a weight w that scales its
    forces and curvature contribution. Higher weights make games more
    influential on the rating update.

    With weights=None or all 1.0, this is identical to standard Yuksel.

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        delta_r_max: Maximum rating change per game (default: 350)
        alpha: Uncertainty decay factor (default: 2.0)
        scaling_factor: Update scaling (default: 0.9)
        initial_weight: Initial value for W accumulator (default: 0.01)

    Example:
        >>> wyuksel = WYuksel(delta_r_max=499.9, alpha=1.67)
        >>> weights = np.array([1.0, 1.5, 0.8, ...])
        >>> wyuksel.fit(dataset, weights=weights)
        >>> print(wyuksel.predict_proba(0, 1))
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        delta_r_max: float = 350.0,
        alpha: float = 2.0,
        scaling_factor: float = 0.9,
        initial_weight: float = 0.01,
        num_players: Optional[int] = None,
    ):
        self.config = WYukselConfig(
            initial_rating=initial_rating,
            delta_r_max=delta_r_max,
            alpha=alpha,
            scaling_factor=scaling_factor,
            initial_weight=initial_weight,
        )
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None

        # Additional state arrays for Yuksel algorithm
        self._R: Optional[np.ndarray] = None
        self._W: Optional[np.ndarray] = None
        self._V: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Yuksel ratings and state arrays for all players."""
        self._R = np.full(num_players, self.config.initial_rating, dtype=np.float64)
        self._W = np.full(num_players, self.config.initial_weight, dtype=np.float64)
        self._V = np.zeros(num_players, dtype=np.float64)
        self._D = np.zeros(num_players, dtype=np.float64)

        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "wyuksel", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with uniform weights (standard Yuksel behaviour)."""
        if len(batch) == 0:
            return

        weights = np.ones(len(batch), dtype=np.float64)
        update_ratings_sequential_weighted(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
            ratings.ratings,
            self._R,
            self._W,
            self._V,
            self._D,
            self.config.delta_r_max,
            self.config.alpha,
            self.config.scaling_factor,
        )
        self._num_games_fitted += len(batch)

    def update_weighted(self, batch: GameBatch, weights: np.ndarray) -> "WYuksel":
        """
        Incrementally update ratings with per-game weights.

        Args:
            batch: Games to process
            weights: Per-game weights array (same length as batch)

        Returns:
            self (for method chaining)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before updating. Call fit() first.")

        if len(batch) == 0:
            return self

        weights = np.ascontiguousarray(weights, dtype=np.float64)
        update_ratings_sequential_weighted(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
            self._ratings.ratings,
            self._R,
            self._W,
            self._V,
            self._D,
            self.config.delta_r_max,
            self.config.alpha,
            self.config.scaling_factor,
        )
        self._num_games_fitted += len(batch)
        self._current_day = batch.day
        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        Prediction is identical to standard Yuksel - weights only affect updates.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(p1, p2, self._ratings.ratings)

    def fit(
        self,
        dataset: GameDataset,
        weights: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WYuksel":
        """
        Fit the rating system on a dataset with optional per-game weights.

        Args:
            dataset: Game dataset to fit on
            weights: Per-game weights array. If None, all default to 1.0.
            end_day: Last day to include (inclusive). Cannot be used with weights.
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

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is not None and len(player1) > 0:
            if weights is None:
                w = np.ones(len(player1), dtype=np.float64)
            else:
                w = np.ascontiguousarray(weights, dtype=np.float64)

            fit_all_days_weighted(
                player1,
                player2,
                scores,
                w,
                day_offsets,
                self._ratings.ratings,
                self._R,
                self._W,
                self._V,
                self._D,
                self.config.delta_r_max,
                self.config.alpha,
                self.config.scaling_factor,
            )
            self._num_games_fitted = len(player1)
            self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        else:
            self._num_games_fitted = 0

        self._fitted = True
        return self

    def get_uncertainty(self, player_id: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Get uncertainty (phi) for a player or all players.

        phi = sqrt(V / W) is the standard deviation of the rating history.
        """
        if self._W is None or self._V is None:
            raise ValueError("Model not fitted. Call fit() first.")

        phi = compute_phi(self._V, self._W)

        if player_id is not None:
            return float(phi[player_id])
        return phi

    def get_fitted_ratings(self) -> FittedYukselRatings:
        """Get a queryable fitted ratings object."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        phi = compute_phi(self._V, self._W)

        return FittedYukselRatings(
            ratings=self._ratings.ratings.copy(),
            phi=phi,
            delta_r_max=self.config.delta_r_max,
            alpha=self.config.alpha,
            scaling_factor=self.config.scaling_factor,
            initial_rating=self.config.initial_rating,
            num_games_fitted=self._num_games_fitted,
            last_day=self._current_day,
            player_names=self._player_names,
        )

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return get_top_n_indices(self._ratings.ratings, n)

    def reset(self) -> "WYuksel":
        """Reset the rating system to initial state."""
        self._num_games_fitted = 0
        self._R = None
        self._W = None
        self._V = None
        self._D = None
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WYuksel(delta_r_max={self.config.delta_r_max}, "
            f"alpha={self.config.alpha}, "
            f"players={players}, {status})"
        )
