"""
Weighted Glicko (WGlicko) rating system - Glicko with per-game weights.

Per-game weights w_j scale the Fisher information contribution of each game.
A game with w_j=3 reduces RD by the same amount as 3 unweighted games
(Remark 3 from the paper). When weights=None, all default to 1.0
(identical to standard Glicko).

This is a general-purpose weighted Glicko. It knows nothing about surfaces
or any other domain-specific concept - it simply accepts a weight per game.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import math
import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...results.fitted_ratings import FittedGlickoRatings
from ._numba_core import (
    update_ratings_batch_weighted,
    fit_all_days_weighted,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
)


@dataclass
class WGlickoConfig:
    """Configuration for Weighted Glicko rating system."""

    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    min_rd: float = 30.0
    max_rd: float = 350.0
    c: float = 34.6
    q: float = math.log(10) / 400


class WGlicko(RatingSystem):
    """
    Weighted Glicko rating system with Numba acceleration.

    Like standard Glicko, but each game has a weight w_j that scales its
    Fisher information contribution. Higher weights make games more
    informative (reduce RD more and shift rating further).

    With weights=None or all 1.0, this is identical to standard Glicko.

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        initial_rd: Starting rating deviation (default: 350)
        min_rd: Minimum rating deviation (default: 30)
        max_rd: Maximum rating deviation (default: 350)
        c: RD increase per period of inactivity (default: 34.6)

    Example:
        >>> wglicko = WGlicko(initial_rd=439.2, c=13.4)
        >>> weights = np.array([1.0, 1.5, 0.8, ...])
        >>> wglicko.fit(dataset, weights=weights)
        >>> print(wglicko.predict_proba(0, 1))
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        min_rd: float = 30.0,
        max_rd: float = 350.0,
        c: float = 34.6,
        num_players: Optional[int] = None,
    ):
        self.config = WGlickoConfig(
            initial_rating=initial_rating,
            initial_rd=initial_rd,
            min_rd=min_rd,
            max_rd=max_rd,
            c=c,
        )
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Glicko ratings for all players."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            rd=np.full(num_players, self.config.initial_rd, dtype=np.float64),
            last_played=np.zeros(num_players, dtype=np.int32),
            metadata={"system": "wglicko", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with uniform weights (standard Glicko behaviour)."""
        if len(batch) == 0:
            return

        weights = np.ones(len(batch), dtype=np.float64)
        update_ratings_batch_weighted(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
            ratings.ratings,
            ratings.rd,
            ratings.last_played,
            batch.day,
            self.config.c,
            self.config.min_rd,
            self.config.max_rd,
        )
        self._num_games_fitted += len(batch)

    def update_weighted(self, batch: GameBatch, weights: np.ndarray) -> "WGlicko":
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
            self._ratings.ratings,
            self._ratings.rd,
            self._ratings.last_played,
            batch.day,
            self.config.c,
            self.config.min_rd,
            self.config.max_rd,
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

        Prediction is identical to standard Glicko - weights only affect updates.
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
    ) -> "WGlicko":
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
                self._ratings.ratings,
                self._ratings.rd,
                self._ratings.last_played,
                self.config.c,
                self.config.min_rd,
                self.config.max_rd,
            )
            self._num_games_fitted = len(player1)
            self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        else:
            self._num_games_fitted = 0

        self._fitted = True
        return self

    def get_fitted_ratings(self) -> FittedGlickoRatings:
        """Get a queryable fitted ratings object."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return FittedGlickoRatings(
            ratings=self._ratings.ratings.copy(),
            rd=self._ratings.rd.copy(),
            last_played=self._ratings.last_played.copy() if self._ratings.last_played is not None else None,
            q=self.config.q,
            initial_rating=self.config.initial_rating,
            initial_rd=self.config.initial_rd,
            num_games_fitted=self._num_games_fitted,
            last_day=self._current_day,
            player_names=self._player_names,
        )

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return get_top_n_indices(self._ratings.ratings, n)

    def get_rating(self, player_id: int) -> Tuple[float, float]:
        """Get (rating, rd) for a player."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return (
            float(self._ratings.ratings[player_id]),
            float(self._ratings.rd[player_id]),
        )

    def reset(self) -> "WGlicko":
        """Reset to initial state."""
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WGlicko(initial_rating={self.config.initial_rating}, "
            f"initial_rd={self.config.initial_rd}, players={players}, {status})"
        )
