"""
Weighted Stephenson (WStephenson) rating system - Stephenson with per-game weights.

Per-game weights w_j scale the Fisher information contribution of each game,
following the same pattern as WGlicko (Proposition 2). When weights=None,
all default to 1.0 (identical to standard Stephenson).

This is a general-purpose weighted Stephenson. It knows nothing about surfaces
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
class WStephensonConfig:
    """Configuration for Weighted Stephenson rating system."""

    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    min_rd: float = 30.0
    max_rd: float = 350.0
    cval: float = 10.0
    hval: float = 10.0
    bval: float = 0.0
    lambda_param: float = 2.0
    gamma: float = 0.0
    q: float = math.log(10) / 400


class WStephenson(RatingSystem):
    """
    Weighted Stephenson rating system with Numba acceleration.

    Like standard Stephenson, but each game has a weight w_j that scales its
    Fisher information contribution. Higher weights make games more
    informative (reduce RD more and shift rating further).

    With weights=None or all 1.0, this is identical to standard Stephenson.

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        initial_rd: Starting rating deviation (default: 350)
        min_rd: Minimum rating deviation (default: 30)
        max_rd: Maximum rating deviation (default: 350)
        cval: RD increase per period of inactivity (default: 10)
        hval: Additional RD increase per game (default: 10)
        bval: Per-game bonus added to actual score (default: 0)
        lambda_param: Neighbourhood shrinkage parameter (default: 2)
        gamma: First-player advantage (default: 0)

    Example:
        >>> wsteph = WStephenson(initial_rd=331.1, cval=13.0)
        >>> weights = np.array([1.0, 1.5, 0.8, ...])
        >>> wsteph.fit(dataset, weights=weights)
        >>> print(wsteph.predict_proba(0, 1))
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        min_rd: float = 30.0,
        max_rd: float = 350.0,
        cval: float = 10.0,
        hval: float = 10.0,
        bval: float = 0.0,
        lambda_param: float = 2.0,
        gamma: float = 0.0,
        num_players: Optional[int] = None,
    ):
        self.config = WStephensonConfig(
            initial_rating=initial_rating,
            initial_rd=initial_rd,
            min_rd=min_rd,
            max_rd=max_rd,
            cval=cval,
            hval=hval,
            bval=bval,
            lambda_param=lambda_param,
            gamma=gamma,
        )
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Stephenson ratings for all players."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            rd=np.full(num_players, self.config.initial_rd, dtype=np.float64),
            last_played=np.zeros(num_players, dtype=np.int32),
            metadata={"system": "wstephenson", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with uniform weights (standard Stephenson behaviour)."""
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
            self.config.cval,
            self.config.hval,
            self.config.bval,
            self.config.lambda_param,
            self.config.gamma,
            self.config.min_rd,
            self.config.max_rd,
        )
        self._num_games_fitted += len(batch)

    def update_weighted(self, batch: GameBatch, weights: np.ndarray) -> "WStephenson":
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
            self.config.cval,
            self.config.hval,
            self.config.bval,
            self.config.lambda_param,
            self.config.gamma,
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

        Prediction is identical to standard Stephenson - weights only affect updates.
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
                self.config.gamma,
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(
            p1, p2, self._ratings.ratings, self._ratings.rd, self.config.gamma
        )

    def fit(
        self,
        dataset: GameDataset,
        weights: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WStephenson":
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
                self.config.cval,
                self.config.hval,
                self.config.bval,
                self.config.lambda_param,
                self.config.gamma,
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

    def reset(self) -> "WStephenson":
        """Reset to initial state."""
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WStephenson(initial_rating={self.config.initial_rating}, "
            f"initial_rd={self.config.initial_rd}, players={players}, {status})"
        )
