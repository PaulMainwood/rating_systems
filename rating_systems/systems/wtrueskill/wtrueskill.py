"""
Weighted TrueSkill (WTrueSkill) rating system - TrueSkill with per-game weights.

Per-game weights w scale the mean shift and uncertainty reduction of each game.
A game with higher weight has more influence on the rating update.
When weights=None, all default to 1.0 (identical to standard TrueSkill).

This is a general-purpose weighted TrueSkill. It knows nothing about surfaces
or any other domain-specific concept - it simply accepts a weight per game.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ._numba_core import (
    update_ratings_sequential_weighted,
    fit_all_days_weighted,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
    compute_conservative_rating,
)


@dataclass
class WTrueSkillConfig:
    """Configuration for Weighted TrueSkill rating system."""

    initial_mu: float = 25.0
    initial_sigma: float = 8.333333333
    beta: float = 4.166666667
    tau: float = 0.0
    display_scale: float = 400.0 / 3.0
    display_offset: float = 1500.0
    min_sigma: float = 0.01


class WeightedTrueSkill(RatingSystem):
    """
    Weighted TrueSkill rating system with Numba acceleration.

    Like standard TrueSkill, but each game has a weight w that scales the
    mean shift and uncertainty reduction. Higher weights make games more
    influential on the rating update.

    With weights=None or all 1.0, this is identical to standard TrueSkill.

    Parameters:
        initial_mu: Starting skill estimate (default: 25)
        initial_sigma: Starting uncertainty (default: 25/3)
        beta: Performance variability per game (default: sigma/2)
        tau: Skill drift rate per rating period (default: 0, static skills)
        display_scale: Scale factor for Elo-like display (default: 400/3)
        display_offset: Offset for Elo-like display (default: 1500)
        min_sigma: Minimum sigma to prevent overconfidence (default: 0.01)

    Example:
        >>> wts = WeightedTrueSkill(initial_mu=20.08, initial_sigma=6.59, beta=0.62)
        >>> weights = np.array([1.0, 1.5, 0.8, ...])
        >>> wts.fit(dataset, weights=weights)
        >>> print(wts.predict_proba(0, 1))
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_mu: float = 25.0,
        initial_sigma: float = 8.333333333,
        beta: float = 4.166666667,
        tau: float = 0.0,
        display_scale: float = 400.0 / 3.0,
        display_offset: float = 1500.0,
        min_sigma: float = 0.01,
        num_players: Optional[int] = None,
    ):
        self.config = WTrueSkillConfig(
            initial_mu=initial_mu,
            initial_sigma=initial_sigma,
            beta=beta,
            tau=tau,
            display_scale=display_scale,
            display_offset=display_offset,
            min_sigma=min_sigma,
        )
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial TrueSkill ratings for all players."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_mu, dtype=np.float64),
            rd=np.full(num_players, self.config.initial_sigma, dtype=np.float64),
            metadata={"system": "wtrueskill", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with uniform weights (standard TrueSkill behaviour)."""
        if len(batch) == 0:
            return

        if self.config.tau > 0:
            ratings.rd[:] = np.sqrt(
                ratings.rd ** 2 + self.config.tau ** 2
            )

        weights = np.ones(len(batch), dtype=np.float64)
        update_ratings_sequential_weighted(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
            ratings.ratings,
            ratings.rd,
            self.config.beta,
            self.config.min_sigma,
        )

        np.maximum(ratings.rd, self.config.min_sigma, out=ratings.rd)
        self._num_games_fitted += len(batch)

    def update_weighted(self, batch: GameBatch, weights: np.ndarray) -> "WeightedTrueSkill":
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

        if self.config.tau > 0:
            self._ratings.rd[:] = np.sqrt(
                self._ratings.rd ** 2 + self.config.tau ** 2
            )

        weights = np.ascontiguousarray(weights, dtype=np.float64)
        update_ratings_sequential_weighted(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
            self._ratings.ratings,
            self._ratings.rd,
            self.config.beta,
            self.config.min_sigma,
        )

        np.maximum(self._ratings.rd, self.config.min_sigma, out=self._ratings.rd)
        self._num_games_fitted += len(batch)
        self._current_day = batch.day
        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        Prediction is identical to standard TrueSkill - weights only affect updates.
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
                self.config.beta,
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(
            p1, p2,
            self._ratings.ratings,
            self._ratings.rd,
            self.config.beta,
        )

    def fit(
        self,
        dataset: GameDataset,
        weights: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WeightedTrueSkill":
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
                self._ratings.rd,
                self.config.beta,
                self.config.min_sigma,
            )

            np.maximum(self._ratings.rd, self.config.min_sigma, out=self._ratings.rd)

            self._num_games_fitted = len(player1)
            self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        else:
            self._num_games_fitted = 0

        self._fitted = True
        return self

    def get_fitted_ratings(self) -> "FittedTrueSkillRatings":
        """Get a queryable fitted ratings object."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        from ...results.fitted_ratings import FittedTrueSkillRatings

        return FittedTrueSkillRatings(
            mu=self._ratings.ratings.copy(),
            sigma=self._ratings.rd.copy(),
            beta=self.config.beta,
            initial_mu=self.config.initial_mu,
            initial_sigma=self.config.initial_sigma,
            display_scale=self.config.display_scale,
            display_offset=self.config.display_offset,
            num_games_fitted=self._num_games_fitted,
            last_day=self._current_day,
            player_names=self._player_names,
        )

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players (by mu)."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return get_top_n_indices(self._ratings.ratings, n)

    def conservative_top(self, n: int = 10, k: float = 3.0) -> np.ndarray:
        """Get indices of top N players by conservative rating (mu - k*sigma)."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        conservative = compute_conservative_rating(
            self._ratings.ratings,
            self._ratings.rd,
            k,
        )
        return get_top_n_indices(conservative, n)

    def get_rating(self, player_id: int) -> Tuple[float, float]:
        """Get (mu, sigma) for a player in internal scale."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return (
            float(self._ratings.ratings[player_id]),
            float(self._ratings.rd[player_id]),
        )

    def reset(self) -> "WeightedTrueSkill":
        """Reset to initial state."""
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WeightedTrueSkill(mu={self.config.initial_mu}, "
            f"sigma={self.config.initial_sigma:.2f}, "
            f"beta={self.config.beta:.2f}, players={players}, {status})"
        )
