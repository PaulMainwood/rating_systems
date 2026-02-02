"""
TrueSkill rating system - high-performance Numba implementation.

TrueSkill models player skill as a Gaussian distribution N(mu, sigma^2),
where mu is the estimated skill and sigma represents uncertainty.

This is the "vanilla" TrueSkill algorithm for 1v1 games, as described in:
Herbrich, Minka, Graepel (2006). "TrueSkill: A Bayesian Skill Rating System"

Key features:
- Skill represented as Gaussian belief N(mu, sigma^2)
- Uncertainty decreases with more games played
- Predictions account for both players' uncertainties
- Conservative rating (mu - k*sigma) for ranking
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ._numba_core import (
    update_ratings_sequential,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
    fit_all_days,
    compute_conservative_rating,
)


@dataclass
class TrueSkillConfig:
    """Configuration for TrueSkill rating system.

    Default values follow the original TrueSkill paper:
    - mu = 25 (initial skill estimate)
    - sigma = 25/3 ≈ 8.333 (initial uncertainty)
    - beta = sigma/2 ≈ 4.167 (performance variability)

    For display purposes, ratings are often scaled to familiar ranges
    (e.g., Elo-like 1500 scale) using display_scale and display_offset.
    """

    # Core TrueSkill parameters (internal scale)
    initial_mu: float = 25.0
    initial_sigma: float = 8.333333333
    beta: float = 4.166666667  # Performance variability (sigma/2)
    tau: float = 0.0  # Dynamics factor (skill drift per period, 0 = static)

    # Display scaling (to convert to Elo-like scale for readability)
    display_scale: float = 400.0 / 3.0  # ~133.33, maps 3 sigma to ~400 points
    display_offset: float = 1500.0  # Center point for display

    # Minimum sigma to prevent overconfidence
    min_sigma: float = 0.01


class TrueSkill(RatingSystem):
    """
    TrueSkill rating system with Numba acceleration.

    TrueSkill improves on Elo by:
    1. Explicitly modeling uncertainty (sigma) for each player
    2. Using Bayesian updates with truncated Gaussians
    3. Accounting for both players' uncertainties in predictions
    4. Providing principled uncertainty reduction with more games

    The "conservative rating" (mu - k*sigma) can be used for ranking,
    representing a lower bound on a player's true skill.

    Parameters:
        initial_mu: Starting skill estimate (default: 25)
        initial_sigma: Starting uncertainty (default: 25/3)
        beta: Performance variability per game (default: sigma/2)
        tau: Skill drift rate per rating period (default: 0, static skills)
        display_scale: Scale factor for Elo-like display (default: 400/3)
        display_offset: Offset for Elo-like display (default: 1500)

    Example:
        >>> ts = TrueSkill()
        >>> ts.fit(dataset)
        >>> fitted = ts.get_fitted_ratings()
        >>> print(fitted.top(10))  # Top 10 players
        >>> mu, sigma = fitted.get_rating(player_id)
        >>> print(fitted.conservative_top(10))  # Top 10 by mu - 3*sigma
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
        self.config = TrueSkillConfig(
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
            metadata={"system": "trueskill", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update TrueSkill ratings for a batch of games."""
        if len(batch) == 0:
            return

        # Apply dynamics factor (skill drift) if enabled
        if self.config.tau > 0:
            # Increase uncertainty over time: sigma_new = sqrt(sigma^2 + tau^2)
            ratings.rd[:] = np.sqrt(
                ratings.rd ** 2 + self.config.tau ** 2
            )

        update_ratings_sequential(
            batch.player1,
            batch.player2,
            batch.scores,
            ratings.ratings,  # mu
            ratings.rd,  # sigma
            self.config.beta,
        )

        # Enforce minimum sigma
        np.maximum(ratings.rd, self.config.min_sigma, out=ratings.rd)

        self._num_games_fitted += len(batch)

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        Uses both players' mu and sigma in the calculation:
        P(p1 wins) = Phi((mu1 - mu2) / sqrt(2*beta^2 + sigma1^2 + sigma2^2))
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle single prediction
        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            p1, p2 = int(player1), int(player2)
            return predict_single(
                self._ratings.ratings[p1],  # mu1
                self._ratings.rd[p1],  # sigma1
                self._ratings.ratings[p2],  # mu2
                self._ratings.rd[p2],  # sigma2
                self.config.beta,
            )

        # Batch prediction
        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(
            p1, p2,
            self._ratings.ratings,  # mu
            self._ratings.rd,  # sigma
            self.config.beta,
        )

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "TrueSkill":
        """
        Fit the rating system on a dataset.

        Uses optimized single Numba call to process all days without
        Python iteration overhead.

        Args:
            dataset: Game dataset to fit on
            end_day: Last day to include (inclusive)
            player_names: Optional mapping of player_id -> name
        """
        self._player_names = player_names

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        # Get pre-batched arrays for direct Numba processing
        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is not None and len(player1) > 0:
            # Process ALL days in single Numba call
            fit_all_days(
                player1,
                player2,
                scores,
                day_offsets,
                self._ratings.ratings,  # mu
                self._ratings.rd,  # sigma
                self.config.beta,
            )

            # Enforce minimum sigma
            np.maximum(self._ratings.rd, self.config.min_sigma, out=self._ratings.rd)

            self._num_games_fitted = len(player1)
            self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        else:
            self._num_games_fitted = 0

        self._fitted = True
        return self

    def get_fitted_ratings(self) -> "FittedTrueSkillRatings":
        """
        Get a queryable fitted ratings object.

        Returns:
            FittedTrueSkillRatings with methods for querying results
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Import here to avoid circular import
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
        """
        Get indices of top N players by conservative rating (mu - k*sigma).

        Args:
            n: Number of players to return
            k: Number of standard deviations for conservative estimate

        Returns:
            Array of player indices sorted by conservative rating
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        conservative = compute_conservative_rating(
            self._ratings.ratings,
            self._ratings.rd,
            k,
        )
        return get_top_n_indices(conservative, n)

    def get_rating(self, player_id: int) -> Tuple[float, float]:
        """
        Get (mu, sigma) for a player in internal scale.

        For display-scale values, use get_display_rating().
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return (
            float(self._ratings.ratings[player_id]),
            float(self._ratings.rd[player_id]),
        )

    def get_display_rating(self, player_id: int) -> Tuple[float, float]:
        """
        Get (rating, rd) for a player in display scale (Elo-like).

        rating = mu * display_scale + display_offset
        rd = sigma * display_scale
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        mu = self._ratings.ratings[player_id]
        sigma = self._ratings.rd[player_id]
        return (
            mu * self.config.display_scale + self.config.display_offset,
            sigma * self.config.display_scale,
        )

    def reset(self) -> "TrueSkill":
        """Reset to initial state."""
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"TrueSkill(mu={self.config.initial_mu}, sigma={self.config.initial_sigma:.2f}, "
            f"beta={self.config.beta:.2f}, players={players}, {status})"
        )
