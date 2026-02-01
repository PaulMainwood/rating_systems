"""
Whole History Rating (WHR) - High-performance Numba implementation.

Based on:
- Rémi Coulom, "Whole-History Rating: A Bayesian Rating System for Players
  of Time-Varying Strength" (2008)
- https://www.remi-coulom.fr/WHR/WHR.pdf

WHR models player ratings as a Wiener process (Brownian motion) over time,
using all historical game data to estimate ratings at any point in time.

This implementation prioritizes efficiency through:
1. Numba JIT compilation of all hot paths
2. CSR-like data structures for player timelines
3. Pre-computed game adjacency for fast gradient/Hessian computation
4. Parallel prediction via prange
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...results.fitted_ratings import FittedWHRRatings
from ._numba_core import (
    LN10_400,
    compute_uncertainties,
    extract_current_ratings,
    get_top_n_indices,
    predict_proba_batch,
    predict_single,
    run_all_iterations,
)


@dataclass
class WHRConfig:
    """Configuration for WHR rating system."""

    w2: float = 300.0  # Wiener variance per time unit (Elo² per day)
    initial_rating: float = 1500.0  # Initial Elo-scale rating
    max_iterations: int = 50  # Maximum Newton-Raphson iterations for initial fit
    update_max_iterations: int = None  # Max iterations for incremental updates (default: same as max_iterations)
    convergence_threshold: float = 1e-6  # Convergence threshold

    def __post_init__(self):
        if self.update_max_iterations is None:
            self.update_max_iterations = self.max_iterations


class WHR(RatingSystem):
    """
    Whole History Rating system with Numba acceleration.

    WHR is a Bayesian rating system that:
    1. Models player strength as a Wiener process (random walk) over time
    2. Uses Bradley-Terry model for game outcomes
    3. Finds MAP estimates via Newton-Raphson optimization
    4. Computes uncertainty from the Hessian

    This is a BATCH system - it must refit on all historical data.

    Performance characteristics:
    - Fit: O(iterations * players * avg_days_per_player) with Numba acceleration
    - Predict: O(n) parallel across matchups
    - Memory: O(total_player_days + total_games)

    Parameters:
        w2: Wiener variance per time unit (default: 300.0)
            Higher values allow ratings to change more quickly.
            In Elo² units per day.
        initial_rating: Starting rating in Elo scale (default: 1500)
        max_iterations: Maximum Newton-Raphson iterations (default: 50)
        convergence_threshold: Stop when max rating change < threshold

    Example:
        >>> whr = WHR(w2=300.0, max_iterations=50)
        >>> whr.fit(dataset)
        >>> fitted = whr.get_fitted_ratings()
        >>> print(fitted.top(10))
        >>> print(fitted.predict(0, 1))
    """

    system_type = RatingSystemType.BATCH

    def __init__(
        self,
        w2: float = 300.0,
        initial_rating: float = 1500.0,
        max_iterations: int = 50,
        update_max_iterations: Optional[int] = None,
        convergence_threshold: float = 1e-6,
        num_players: Optional[int] = None,
    ):
        self.config = WHRConfig(
            w2=w2,
            initial_rating=initial_rating,
            max_iterations=max_iterations,
            update_max_iterations=update_max_iterations,
            convergence_threshold=convergence_threshold,
        )

        # w2 in log-gamma scale
        self._w2_r = w2 * (LN10_400**2)

        # Data structures (CSR-like format for Numba)
        self._player_offsets: Optional[np.ndarray] = None  # [num_players + 1]
        self._pd_days: Optional[np.ndarray] = None  # [total_player_days]
        self._pd_r: Optional[np.ndarray] = None  # [total_player_days] - ratings
        self._pd_uncertainty: Optional[np.ndarray] = None  # [total_player_days]
        self._pd_game_offsets: Optional[np.ndarray] = None  # [total_player_days + 1]
        self._pd_game_opp_pd: Optional[np.ndarray] = None  # [total_games * 2]
        self._pd_game_score: Optional[np.ndarray] = None  # [total_games * 2]

        # Metadata
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._player_names: Optional[Dict[int, str]] = None

        # Store raw data for refitting
        self._stored_player1: Optional[np.ndarray] = None
        self._stored_player2: Optional[np.ndarray] = None
        self._stored_scores: Optional[np.ndarray] = None
        self._stored_days: Optional[np.ndarray] = None

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial WHR ratings."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            rd=np.full(num_players, 350.0, dtype=np.float64),
            metadata={"system": "whr", "config": self.config},
        )

    def _build_data_structures(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        days: np.ndarray,
        num_players: int,
    ) -> None:
        """
        Build CSR-like data structures from game arrays.

        This converts the raw game data into the format needed by Numba:
        - Player timelines with their active days
        - Games per player-day with opponent references
        """
        n_games = len(player1)

        # Step 1: Identify unique (player, day) pairs and build mapping
        # Use a dict to track (player_id, day) -> player_day_index
        player_day_map: Dict[Tuple[int, int], int] = {}
        player_day_list: List[Tuple[int, int]] = []  # (player_id, day)

        for i in range(n_games):
            p1, p2, day = int(player1[i]), int(player2[i]), int(days[i])

            key1 = (p1, day)
            if key1 not in player_day_map:
                player_day_map[key1] = len(player_day_list)
                player_day_list.append(key1)

            key2 = (p2, day)
            if key2 not in player_day_map:
                player_day_map[key2] = len(player_day_list)
                player_day_list.append(key2)

        total_pd = len(player_day_list)

        # Step 2: Sort player-days by (player_id, day) and build offsets
        player_day_list.sort(key=lambda x: (x[0], x[1]))

        # Rebuild mapping with sorted indices
        sorted_map: Dict[Tuple[int, int], int] = {}
        for new_idx, (pid, day) in enumerate(player_day_list):
            sorted_map[(pid, day)] = new_idx

        # Build player_offsets
        self._player_offsets = np.zeros(num_players + 1, dtype=np.int64)
        for pid, _ in player_day_list:
            self._player_offsets[pid + 1] += 1
        np.cumsum(self._player_offsets, out=self._player_offsets)

        # Build pd_days array
        self._pd_days = np.empty(total_pd, dtype=np.int32)
        for idx, (_, day) in enumerate(player_day_list):
            self._pd_days[idx] = day

        # Initialize ratings to 0 (log-gamma scale, equals initial_rating in Elo)
        self._pd_r = np.zeros(total_pd, dtype=np.float64)
        self._pd_uncertainty = np.full(total_pd, 350.0, dtype=np.float64)

        # Step 3: Count games per player-day (each game appears twice - once per player)
        pd_game_counts = np.zeros(total_pd, dtype=np.int64)
        for i in range(n_games):
            p1, p2, day = int(player1[i]), int(player2[i]), int(days[i])
            pd1 = sorted_map[(p1, day)]
            pd2 = sorted_map[(p2, day)]
            pd_game_counts[pd1] += 1
            pd_game_counts[pd2] += 1

        # Build game offsets
        self._pd_game_offsets = np.zeros(total_pd + 1, dtype=np.int64)
        np.cumsum(pd_game_counts, out=self._pd_game_offsets[1:])

        total_game_refs = self._pd_game_offsets[-1]  # = 2 * n_games

        # Step 4: Fill game arrays
        self._pd_game_opp_pd = np.empty(total_game_refs, dtype=np.int64)
        self._pd_game_score = np.empty(total_game_refs, dtype=np.float64)

        # Track current position for each player-day's games
        pd_game_pos = self._pd_game_offsets[:-1].copy()

        for i in range(n_games):
            p1, p2, day = int(player1[i]), int(player2[i]), int(days[i])
            score = float(scores[i])

            pd1 = sorted_map[(p1, day)]
            pd2 = sorted_map[(p2, day)]

            # Add game from player1's perspective
            pos1 = pd_game_pos[pd1]
            self._pd_game_opp_pd[pos1] = pd2
            self._pd_game_score[pos1] = score
            pd_game_pos[pd1] += 1

            # Add game from player2's perspective
            pos2 = pd_game_pos[pd2]
            self._pd_game_opp_pd[pos2] = pd1
            self._pd_game_score[pos2] = 1.0 - score
            pd_game_pos[pd2] += 1

    def _run_optimization(self, max_iterations: Optional[int] = None) -> None:
        """Run Newton-Raphson optimization.

        Args:
            max_iterations: Override max iterations (uses config value if None)
        """
        if self._player_offsets is None or self._num_players is None:
            return

        if max_iterations is None:
            max_iterations = self.config.max_iterations

        self._num_iterations = run_all_iterations(
            self._num_players,
            self._player_offsets,
            self._pd_days,
            self._pd_r,
            self._pd_game_offsets,
            self._pd_game_opp_pd,
            self._pd_game_score,
            self._w2_r,
            max_iterations,
            self.config.convergence_threshold,
        )

        # Compute uncertainties
        compute_uncertainties(
            self._num_players,
            self._player_offsets,
            self._pd_days,
            self._pd_r,
            self._pd_uncertainty,
            self._pd_game_offsets,
            self._pd_game_opp_pd,
            self._pd_game_score,
            self._w2_r,
        )

    def _extract_current_ratings(self) -> None:
        """Extract most recent ratings for each player."""
        if self._num_players is None or self._player_offsets is None:
            return

        ratings = np.empty(self._num_players, dtype=np.float64)
        rd = np.empty(self._num_players, dtype=np.float64)

        extract_current_ratings(
            self._num_players,
            self._player_offsets,
            self._pd_r,
            self._pd_uncertainty,
            ratings,
            rd,
            self.config.initial_rating,
        )

        self._ratings = PlayerRatings(
            ratings=ratings,
            rd=rd,
            metadata={"system": "whr", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with a new batch (refits on all data)."""
        # Append new data and refit
        if self._stored_player1 is None:
            self._stored_player1 = batch.player1.copy()
            self._stored_player2 = batch.player2.copy()
            self._stored_scores = batch.scores.copy()
            self._stored_days = np.full(len(batch), batch.day, dtype=np.int32)
        else:
            self._stored_player1 = np.concatenate([self._stored_player1, batch.player1])
            self._stored_player2 = np.concatenate([self._stored_player2, batch.player2])
            self._stored_scores = np.concatenate([self._stored_scores, batch.scores])
            self._stored_days = np.concatenate(
                [self._stored_days, np.full(len(batch), batch.day, dtype=np.int32)]
            )

        self._refit()

    def _refit(self, max_iterations: Optional[int] = None) -> None:
        """Refit on all stored data.

        Args:
            max_iterations: Override max iterations (uses update_max_iterations if None)
        """
        if self._stored_player1 is None:
            return

        if max_iterations is None:
            max_iterations = self.config.update_max_iterations

        self._build_data_structures(
            self._stored_player1,
            self._stored_player2,
            self._stored_scores,
            self._stored_days,
            self._num_players,
        )
        self._run_optimization(max_iterations)
        self._extract_current_ratings()
        self._num_games_fitted = len(self._stored_player1)

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WHR":
        """
        Fit WHR on a dataset.

        Args:
            dataset: Game dataset to fit on
            end_day: Last day to include (inclusive). Useful for backtesting.
            player_names: Optional mapping of player_id -> name

        Returns:
            self (for method chaining)
        """
        self._player_names = player_names

        # Filter dataset if end_day specified
        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        # Get batched arrays
        player1, player2, scores, day_indices, day_offsets = (
            dataset.get_batched_arrays()
        )

        if player1 is None or len(player1) == 0:
            self._fitted = True
            return self

        # Create per-game day array from day_indices and day_offsets
        n_games = len(player1)
        days = np.empty(n_games, dtype=np.int32)
        n_days = len(day_indices)
        for d in range(n_days):
            start = day_offsets[d]
            end = day_offsets[d + 1]
            days[start:end] = day_indices[d]

        # Store for potential refitting
        self._stored_player1 = player1.copy()
        self._stored_player2 = player2.copy()
        self._stored_scores = scores.copy()
        self._stored_days = days

        # Build data structures and optimize
        self._build_data_structures(player1, player2, scores, days, self._num_players)
        self._run_optimization()
        self._extract_current_ratings()

        self._num_games_fitted = n_games
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None

        return self

    def update(self, batch: GameBatch) -> "WHR":
        """Update with new games by refitting on all data."""
        if not self._fitted:
            raise ValueError("Model must be fitted before updating")

        self._update_ratings(batch, self._ratings)
        self._current_day = batch.day

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
            raise ValueError("Model not fitted")

        # Handle single prediction
        if isinstance(player1, (int, np.integer)) and isinstance(
            player2, (int, np.integer)
        ):
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
            )

        # Batch prediction
        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(p1, p2, self._ratings.ratings)

    def get_fitted_ratings(self) -> FittedWHRRatings:
        """
        Get a queryable fitted ratings object.

        Returns:
            FittedWHRRatings with methods for querying results
        """
        if self._ratings is None:
            raise ValueError("Model not fitted")

        # Build rating history per player
        rating_history = {}
        if self._player_offsets is not None:
            for pid in range(self._num_players):
                pd_start = self._player_offsets[pid]
                pd_end = self._player_offsets[pid + 1]
                if pd_end > pd_start:
                    rating_history[pid] = {
                        "days": self._pd_days[pd_start:pd_end].copy(),
                        "ratings": (
                            self._pd_r[pd_start:pd_end] / LN10_400
                            + self.config.initial_rating
                        ).copy(),
                        "uncertainties": self._pd_uncertainty[pd_start:pd_end].copy(),
                    }

        return FittedWHRRatings(
            ratings=self._ratings.ratings.copy(),
            rd=self._ratings.rd.copy(),
            w2=self.config.w2,
            initial_rating=self.config.initial_rating,
            num_games_fitted=self._num_games_fitted,
            num_iterations=self._num_iterations,
            last_day=self._current_day,
            player_names=self._player_names,
            rating_history=rating_history,
        )

    def get_rating_history(self, player_id: int) -> Optional[Dict]:
        """Get the full rating history for a player."""
        if self._player_offsets is None:
            return None

        pd_start = self._player_offsets[player_id]
        pd_end = self._player_offsets[player_id + 1]

        if pd_end <= pd_start:
            return None

        return {
            "days": self._pd_days[pd_start:pd_end].tolist(),
            "ratings": (
                self._pd_r[pd_start:pd_end] / LN10_400 + self.config.initial_rating
            ).tolist(),
            "uncertainties": self._pd_uncertainty[pd_start:pd_end].tolist(),
        }

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players (convenience method)."""
        if self._ratings is None:
            raise ValueError("Model not fitted")
        return get_top_n_indices(self._ratings.ratings, n)

    def reset(self) -> "WHR":
        """Reset the rating system."""
        self._player_offsets = None
        self._pd_days = None
        self._pd_r = None
        self._pd_uncertainty = None
        self._pd_game_offsets = None
        self._pd_game_opp_pd = None
        self._pd_game_score = None
        self._stored_player1 = None
        self._stored_player2 = None
        self._stored_scores = None
        self._stored_days = None
        self._num_games_fitted = 0
        self._num_iterations = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WHR(w2={self.config.w2}, "
            f"max_iter={self.config.max_iterations}, "
            f"players={players}, {status})"
        )
