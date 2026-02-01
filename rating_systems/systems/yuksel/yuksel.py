"""
Yuksel rating system - high-performance Numba implementation.

Based on: Cem Yuksel, "Skill-Based Matchmaking for Competitive Two-Player Games"
Proceedings of the ACM on Computer Graphics and Interactive Techniques, 2024.
https://www.cemyuksel.com/research/matchmaking/

The Yuksel method is an online rating system that combines ideas from Elo and Glicko
with adaptive step-size control similar to modern optimizers like Adam/RMSprop.

Key features:
- Tracks rating variance to estimate uncertainty without storing history
- Uses Glicko's g function to downweight updates when uncertainty is high
- Maintains a "direction" term for adaptive step sizes (like Adam's second moment)
- Zero-sum updates that respect both players' uncertainties
- Fast convergence with bounded updates for stability

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
from ...results.fitted_ratings import FittedYukselRatings
from ._numba_core import (
    update_ratings_sequential,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
    fit_all_days,
    compute_phi,
)


@dataclass
class YukselConfig:
    """
    Configuration for Yuksel rating system.

    Parameters:
        initial_rating: Starting rating for new players (default: 1500.0)
            Standard Elo-scale value.

        delta_r_max: Maximum rating change per game (default: 350.0)
            Acts as a clipping bound on updates. Larger values allow faster
            convergence but can cause instability. 350 is ~1.5 sigma on the
            standard Elo scale.

        alpha: Uncertainty decay factor (default: 2.0)
            Controls how quickly uncertainty estimates decay with new games.
            Higher values mean uncertainty decreases faster. The g function
            is applied with alpha*phi to determine the decay rate.

        scaling_factor: Update scaling (default: 0.9)
            Scales the final update by this factor. Values < 1 improve
            numerical stability in edge cases. Use 1.0 for standard updates.

        initial_weight: Initial value for W (weight accumulator) (default: 0.01)
            Small positive value to ensure W > 0 for new players.
            Larger values make initial uncertainty lower.
    """

    initial_rating: float = 1500.0
    delta_r_max: float = 350.0
    alpha: float = 2.0
    scaling_factor: float = 0.9
    initial_weight: float = 0.01


class Yuksel(RatingSystem):
    """
    Yuksel rating system with Numba acceleration.

    The Yuksel method combines several innovations:

    1. **Uncertainty Tracking**: Uses Welford's online algorithm to track
       rating variance without storing history. Uncertainty (phi) is estimated
       as sqrt(V/W), the standard deviation of the player's rating history.

    2. **Adaptive Step Sizes**: Maintains a "direction" term D that accumulates
       curvature information (like Adam's second moment). Updates are computed as:
       delta_r = (D1*F1 - D2*F2) / (D1² + D2²)
       This naturally adapts step sizes based on accumulated gradient history.

    3. **Glicko-style Weighting**: Uses the g(phi) function to downweight updates
       when uncertainty is high. This means:
       - Well-established players have stable ratings
       - New players can move quickly
       - Updates respect both players' certainty levels

    4. **Zero-Sum Updates**: Rating changes are symmetric (p1 gains what p2 loses),
       but the magnitude is determined by both players' uncertainties.

    Performance characteristics:
    - Update: O(n) sequential (required for correctness)
    - Predict: O(n) parallel across matchups
    - Memory: O(5 * num_players) for ratings + state arrays

    Example:
        >>> yuksel = Yuksel(delta_r_max=350, alpha=2.0)
        >>> yuksel.fit(dataset)
        >>> fitted = yuksel.get_fitted_ratings()
        >>> print(fitted.top(10))  # Top 10 players
        >>> print(fitted.predict(0, 1))  # P(player 0 beats player 1)
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
        self.config = YukselConfig(
            initial_rating=initial_rating,
            delta_r_max=delta_r_max,
            alpha=alpha,
            scaling_factor=scaling_factor,
            initial_weight=initial_weight,
        )
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None

        # Additional state arrays for Yuksel algorithm
        self._R: Optional[np.ndarray] = None  # Weighted mean of ratings
        self._W: Optional[np.ndarray] = None  # Accumulated weights
        self._V: Optional[np.ndarray] = None  # Weighted variance
        self._D: Optional[np.ndarray] = None  # Direction/curvature term

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Yuksel ratings and state arrays for all players."""
        # Initialize additional state arrays
        self._R = np.full(num_players, self.config.initial_rating, dtype=np.float64)
        self._W = np.full(num_players, self.config.initial_weight, dtype=np.float64)
        self._V = np.zeros(num_players, dtype=np.float64)
        self._D = np.zeros(num_players, dtype=np.float64)

        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "yuksel", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update Yuksel ratings for a batch of games."""
        if len(batch) == 0:
            return

        update_ratings_sequential(
            batch.player1,
            batch.player2,
            batch.scores,
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

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        Uses simple Elo-style prediction since the ratings already incorporate
        uncertainty through the adaptive update mechanism.

        Args:
            player1: Single player ID or array of player IDs
            player2: Single player ID or array of player IDs

        Returns:
            Single probability or array of probabilities
        """
        if self._ratings is None:
            raise ValueError("Model not fitted")

        # Handle single prediction
        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
            )

        # Batch prediction
        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(p1, p2, self._ratings.ratings)

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "Yuksel":
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

        phi = sqrt(V/W) is the estimated standard deviation of the rating.
        Higher values indicate less certainty about the player's true skill.

        Args:
            player_id: Specific player (returns float) or None (returns all)

        Returns:
            Single phi value or array of phi values
        """
        if self._W is None or self._V is None:
            raise ValueError("Model not fitted")

        phi = compute_phi(self._V, self._W)

        if player_id is not None:
            return float(phi[player_id])
        return phi

    def get_fitted_ratings(self) -> "FittedYukselRatings":
        """
        Get a queryable fitted ratings object.

        Returns:
            FittedYukselRatings with methods for querying results
        """
        if self._ratings is None:
            raise ValueError("Model not fitted")

        # Compute phi (uncertainty) from V and W
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
        """Get indices of top N rated players (convenience method)."""
        if self._ratings is None:
            raise ValueError("Model not fitted")
        return get_top_n_indices(self._ratings.ratings, n)

    def reset(self) -> "Yuksel":
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
            f"Yuksel(delta_r_max={self.config.delta_r_max}, "
            f"alpha={self.config.alpha}, "
            f"players={players}, {status})"
        )
