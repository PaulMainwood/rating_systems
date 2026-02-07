"""
Weighted Glicko-2 (WGlicko2) rating system - Glicko-2 with per-game weights.

Per-game weights w_j scale the Fisher information contribution of each game.
The volatility update is unchanged in form but receives weighted v and delta.
When weights=None, all default to 1.0 (identical to standard Glicko-2).

This is a general-purpose weighted Glicko-2. It knows nothing about surfaces
or any other domain-specific concept - it simply accepts a weight per game.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...results.fitted_ratings import FittedGlicko2Ratings
from ._numba_core import (
    update_ratings_batch_weighted,
    fit_all_days_weighted,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
)


@dataclass
class WGlicko2Config:
    """Configuration for Weighted Glicko-2 rating system."""

    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    initial_volatility: float = 0.06
    tau: float = 0.5
    epsilon: float = 0.000001
    scale: float = 173.7178
    max_rd: float = 350.0


class WGlicko2(RatingSystem):
    """
    Weighted Glicko-2 rating system with Numba acceleration.

    Like standard Glicko-2, but each game has a weight w_j that scales its
    Fisher information contribution. Higher weights make games more
    informative. The volatility update receives weighted v and delta.

    With weights=None or all 1.0, this is identical to standard Glicko-2.

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        initial_rd: Starting rating deviation (default: 350)
        initial_volatility: Starting volatility (default: 0.06)
        tau: System constant controlling volatility change (default: 0.5)

    Example:
        >>> wglicko2 = WGlicko2(tau=1.17, initial_rd=491.4)
        >>> weights = np.array([1.0, 1.5, 0.8, ...])
        >>> wglicko2.fit(dataset, weights=weights)
        >>> print(wglicko2.predict_proba(0, 1))
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
        self.config = WGlicko2Config(
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
            metadata={"system": "wglicko2", "config": self.config},
        )

    def _to_glicko_scale(self, mu: np.ndarray) -> np.ndarray:
        """Convert mu from Glicko-2 scale to Glicko scale."""
        return mu * self.config.scale + self.config.initial_rating

    def _to_glicko_rd(self, phi: np.ndarray) -> np.ndarray:
        """Convert phi from Glicko-2 scale to Glicko RD."""
        return phi * self.config.scale

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with uniform weights (standard Glicko-2 behaviour)."""
        if len(batch) == 0:
            return

        weights = np.ones(len(batch), dtype=np.float64)
        update_ratings_batch_weighted(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
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

    def update_weighted(self, batch: GameBatch, weights: np.ndarray) -> "WGlicko2":
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
        update_ratings_batch_weighted(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
            self._ratings.ratings,  # mu
            self._ratings.rd,       # phi
            self._ratings.volatility,
            self._ratings.last_played,
            batch.day,
            self.config.tau,
            self.config.epsilon,
            self.config.max_rd / self.config.scale,  # max_phi
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

        Prediction is identical to standard Glicko-2 - weights only affect updates.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            p1, p2 = int(player1), int(player2)
            return predict_single(
                self._ratings.ratings[p1],
                self._ratings.rd[p1],
                self._ratings.ratings[p2],
                self._ratings.rd[p2],
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(
            p1, p2, self._ratings.ratings, self._ratings.rd
        )

    def fit(
        self,
        dataset: GameDataset,
        weights: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WGlicko2":
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

    def get_fitted_ratings(self) -> FittedGlicko2Ratings:
        """Get a queryable fitted ratings object."""
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

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return get_top_n_indices(self._ratings.ratings, n)

    def reset(self) -> "WGlicko2":
        """Reset the rating system to initial state."""
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WGlicko2(tau={self.config.tau}, "
            f"initial_volatility={self.config.initial_volatility}, "
            f"players={players}, {status})"
        )
