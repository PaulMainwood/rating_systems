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
from ...data.checkpoint import compute_fingerprint, verify_fingerprint, save_checkpoint, load_checkpoint
from ...results.fitted_ratings import FittedWHRRatings
from ._numba_core import (
    LN10_400,
    build_player_day_indices,
    compute_uncertainties,
    extract_current_ratings,
    extract_player_last_day,
    fill_game_arrays,
    get_top_n_indices,
    predict_proba_batch,
    predict_proba_batch_at_day,
    predict_single,
    predict_single_at_day,
    run_all_iterations,
    run_all_iterations_accelerated,
    warm_start_ratings,
)


@dataclass
class WHRConfig:
    """Configuration for WHR rating system."""

    w2: float = 300.0  # Wiener variance per time unit (Elo² per day)
    initial_rating: float = 1500.0  # Initial Elo-scale rating
    initial_rd: float = 350.0  # Initial rating deviation (uncertainty)
    max_iterations: int = 50  # Maximum Newton-Raphson iterations for initial fit
    refit_max_iterations: int = 5  # Max iterations for refits
    refit_interval: int = 1  # Days between refits during walk-forward (1 = every day)
    convergence_threshold: float = 1e-6  # Convergence threshold
    warm_start: bool = True  # Use previous solution as starting point for refits
    use_active_set: bool = True  # Skip converged players during iteration
    anderson_window: int = 5  # Anderson acceleration window (0 = disabled)
    use_jacobi: bool = True  # Jacobi parallel iteration (vs sequential Gauss-Seidel)

    def __post_init__(self):
        if self.refit_max_iterations is None:
            self.refit_max_iterations = self.max_iterations


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
        initial_rd: float = 350.0,
        max_iterations: int = 50,
        refit_max_iterations: int = 5,
        refit_interval: int = 1,
        convergence_threshold: float = 1e-6,
        warm_start: bool = True,
        use_active_set: bool = True,
        anderson_window: int = 5,
        use_jacobi: bool = True,
        num_players: Optional[int] = None,
    ):
        self.config = WHRConfig(
            w2=w2,
            initial_rating=initial_rating,
            initial_rd=initial_rd,
            max_iterations=max_iterations,
            refit_max_iterations=refit_max_iterations,
            refit_interval=refit_interval,
            convergence_threshold=convergence_threshold,
            warm_start=warm_start,
            use_active_set=use_active_set,
            anderson_window=anderson_window,
            use_jacobi=use_jacobi,
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
        self._pd_to_player: Optional[np.ndarray] = None  # [total_player_days]
        self._player_last_day: Optional[np.ndarray] = None  # [num_players] last active day

        # Metadata
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._player_names: Optional[Dict[int, str]] = None

        # Store raw data for refitting
        self._stored_player1: Optional[np.ndarray] = None
        self._stored_player2: Optional[np.ndarray] = None
        self._stored_scores: Optional[np.ndarray] = None
        self._stored_days: Optional[np.ndarray] = None
        self._last_refit_day: Optional[int] = None

        # Loaded checkpoint state (used by fit() for warm-start)
        self._checkpoint_loaded = False

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial WHR ratings."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            rd=np.full(num_players, self.config.initial_rd, dtype=np.float64),
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

        Uses NumPy vectorized operations for ~4.5x speedup over pure Python.
        """
        n_games = len(player1)

        # Step 1+2: Build player-day CSR structure in O(n_games)
        # Exploits day-sorted input to avoid O(n log n) np.unique
        (self._player_offsets, self._pd_days, pd1_indices, pd2_indices, total_pd
        ) = build_player_day_indices(player1, player2, days, num_players, n_games)

        # Reverse mapping: player-day index -> player_id (needed by active-set)
        self._pd_to_player = np.repeat(
            np.arange(num_players, dtype=np.int64),
            np.diff(self._player_offsets),
        )

        # Initialize ratings to 0 (log-gamma scale, equals initial_rating in Elo)
        self._pd_r = np.zeros(total_pd, dtype=np.float64)
        self._pd_uncertainty = np.full(total_pd, self.config.initial_rd, dtype=np.float64)

        # Step 3: Count games per player-day using scatter-add
        pd_game_counts = np.zeros(total_pd, dtype=np.int64)
        np.add.at(pd_game_counts, pd1_indices, 1)
        np.add.at(pd_game_counts, pd2_indices, 1)

        # Build game offsets
        self._pd_game_offsets = np.zeros(total_pd + 1, dtype=np.int64)
        np.cumsum(pd_game_counts, out=self._pd_game_offsets[1:])

        total_game_refs = self._pd_game_offsets[-1]  # = 2 * n_games

        # Step 4: Fill game arrays using Numba helper
        self._pd_game_opp_pd = np.empty(total_game_refs, dtype=np.int64)
        self._pd_game_score = np.empty(total_game_refs, dtype=np.float64)

        fill_game_arrays(
            n_games,
            pd1_indices,
            pd2_indices,
            scores,
            self._pd_game_offsets,
            self._pd_game_opp_pd,
            self._pd_game_score,
        )

    def _run_optimization(self, max_iterations: Optional[int] = None) -> None:
        """Run Newton-Raphson optimization.

        Args:
            max_iterations: Override max iterations (uses config value if None)
        """
        if self._player_offsets is None or self._num_players is None:
            return

        if max_iterations is None:
            max_iterations = self.config.max_iterations

        if self.config.use_active_set or self.config.anderson_window > 0 or self.config.use_jacobi:
            self._num_iterations = run_all_iterations_accelerated(
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
                self.config.anderson_window,
                self.config.use_active_set,
                self._pd_to_player,
                self.config.use_jacobi,
            )
        else:
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

        # Cache last active day per player for time-aware predictions
        self._player_last_day = extract_player_last_day(
            self._num_players, self._player_offsets, self._pd_days,
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with a new batch (refits based on refit_interval)."""
        # Append new data
        if self._stored_player1 is None:
            self._stored_player1 = batch.player1.copy()
            self._stored_player2 = batch.player2.copy()
            self._stored_scores = batch.scores.copy()
            self._stored_days = np.full(len(batch), batch.day, dtype=np.int32)
            self._last_refit_day = batch.day
        else:
            self._stored_player1 = np.concatenate([self._stored_player1, batch.player1])
            self._stored_player2 = np.concatenate([self._stored_player2, batch.player2])
            self._stored_scores = np.concatenate([self._stored_scores, batch.scores])
            self._stored_days = np.concatenate(
                [self._stored_days, np.full(len(batch), batch.day, dtype=np.int32)]
            )

        # Check if it's time to refit based on refit_interval
        if self._last_refit_day is None:
            self._last_refit_day = batch.day

        if batch.day - self._last_refit_day >= self.config.refit_interval:
            self._refit()
            self._last_refit_day = batch.day

    def _refit(self, max_iterations: Optional[int] = None) -> None:
        """Refit on all stored data.

        Args:
            max_iterations: Override max iterations (uses refit_max_iterations if None)
        """
        if self._stored_player1 is None:
            return

        if max_iterations is None:
            max_iterations = self.config.refit_max_iterations

        # Save old state for warm start
        old_pd_r = None
        old_player_offsets = None
        old_pd_days = None
        if (
            self.config.warm_start
            and self._pd_r is not None
            and self._player_offsets is not None
            and self._pd_days is not None
        ):
            old_pd_r = self._pd_r
            old_player_offsets = self._player_offsets
            old_pd_days = self._pd_days

        self._build_data_structures(
            self._stored_player1,
            self._stored_player2,
            self._stored_scores,
            self._stored_days,
            self._num_players,
        )

        # Apply warm start: transfer old converged ratings to new structure
        if old_pd_r is not None:
            warm_start_ratings(
                self._num_players,
                old_player_offsets,
                old_pd_days,
                old_pd_r,
                self._player_offsets,
                self._pd_days,
                self._pd_r,
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

        # Save old checkpoint state for warm-start (if loaded)
        old_pd_r = None
        old_player_offsets = None
        old_pd_days = None
        if self._checkpoint_loaded and self._pd_r is not None:
            old_pd_r = self._pd_r.copy()
            old_player_offsets = self._player_offsets.copy()
            old_pd_days = self._pd_days.copy()
            self._checkpoint_loaded = False

        # Build data structures and optimize
        self._build_data_structures(player1, player2, scores, days, self._num_players)

        # Warm-start from checkpoint if available
        if old_pd_r is not None:
            warm_start_ratings(
                self._num_players,
                old_player_offsets,
                old_pd_days,
                old_pd_r,
                self._player_offsets,
                self._pd_days,
                self._pd_r,
            )

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

        if self._player_last_day is not None:
            players = np.unique(np.concatenate([batch.player1, batch.player2]))
            self._player_last_day[players] = batch.day

        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        When day is provided, uses damped sigmoid to account for Wiener
        process drift since each player's last active day.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if day is not None and self._player_last_day is not None:
            if isinstance(player1, (int, np.integer)) and isinstance(
                player2, (int, np.integer)
            ):
                return predict_single_at_day(
                    self._ratings.ratings[int(player1)],
                    self._ratings.ratings[int(player2)],
                    self._player_last_day[int(player1)],
                    self._player_last_day[int(player2)],
                    day, self.config.w2,
                )
            p1 = np.ascontiguousarray(player1, dtype=np.int64)
            p2 = np.ascontiguousarray(player2, dtype=np.int64)
            return predict_proba_batch_at_day(
                p1, p2, self._ratings.ratings,
                self._player_last_day, day, self.config.w2,
            )

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
            raise ValueError("Model not fitted. Call fit() first.")

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
            raise ValueError("Model not fitted. Call fit() first.")
        return get_top_n_indices(self._ratings.ratings, n)

    def save_state(self, path: str) -> None:
        """Save fitted WHR state to .npz file.

        Saves CSR structures and stored game arrays needed for prediction
        and warm-start refitting.
        """
        if not self._fitted or self._pd_r is None:
            raise ValueError("Model must be fitted before saving state.")

        arrays = {
            "ratings": self._ratings.ratings,
            "rd": self._ratings.rd,
            "pd_r": self._pd_r,
            "pd_days": self._pd_days,
            "player_offsets": self._player_offsets,
            "pd_uncertainty": self._pd_uncertainty,
            "pd_game_offsets": self._pd_game_offsets,
            "pd_game_opp_pd": self._pd_game_opp_pd,
            "pd_game_score": self._pd_game_score,
            "pd_to_player": self._pd_to_player,
        }
        if self._stored_player1 is not None:
            arrays["stored_player1"] = self._stored_player1
            arrays["stored_player2"] = self._stored_player2
            arrays["stored_scores"] = self._stored_scores
            arrays["stored_days"] = self._stored_days

        metadata = {
            "system_class": "WHR",
            "num_players": self._num_players,
            "current_day": self._current_day,
            "num_games_fitted": self._num_games_fitted,
            "num_iterations": self._num_iterations,
            "config": {
                "w2": self.config.w2,
                "initial_rating": self.config.initial_rating,
                "initial_rd": self.config.initial_rd,
            },
        }
        save_checkpoint(path, arrays, metadata)

    def load_state(self, path: str) -> None:
        """Restore fitted WHR state from .npz file."""
        arrays, metadata = load_checkpoint(path)

        self._num_players = metadata["num_players"]
        self._current_day = metadata["current_day"]
        self._num_games_fitted = metadata.get("num_games_fitted", 0)
        self._num_iterations = metadata.get("num_iterations", 0)
        self._fitted = True

        self._ratings = PlayerRatings(
            ratings=arrays["ratings"],
            rd=arrays["rd"],
            metadata={"system": "whr", "config": self.config},
        )

        self._pd_r = arrays["pd_r"]
        self._pd_days = arrays["pd_days"]
        self._player_offsets = arrays["player_offsets"]
        self._pd_uncertainty = arrays.get("pd_uncertainty")
        self._pd_game_offsets = arrays.get("pd_game_offsets")
        self._pd_game_opp_pd = arrays.get("pd_game_opp_pd")
        self._pd_game_score = arrays.get("pd_game_score")
        self._pd_to_player = arrays.get("pd_to_player")

        if "stored_player1" in arrays:
            self._stored_player1 = arrays["stored_player1"]
            self._stored_player2 = arrays["stored_player2"]
            self._stored_scores = arrays["stored_scores"]
            self._stored_days = arrays["stored_days"]

        # Cache last active day per player for time-aware predictions
        self._player_last_day = extract_player_last_day(
            self._num_players, self._player_offsets, self._pd_days,
        )

    def save_checkpoint(self, path: str, player_to_idx: Optional[Dict[int, int]] = None) -> None:
        """Save fitted state to .npz for later warm-start.

        Args:
            path: Output file path (should end in .npz).
            player_to_idx: Optional player ID → index mapping to store.
        """
        if not self._fitted or self._pd_r is None:
            raise ValueError("Model must be fitted before saving checkpoint.")

        arrays = {
            "pd_r": self._pd_r,
            "pd_days": self._pd_days,
            "player_offsets": self._player_offsets,
        }

        fingerprint = compute_fingerprint(
            self._stored_player1,
            self._stored_player2,
            self._stored_scores,
            self._num_players,
            int(self._stored_days.max()),
        )

        metadata = {
            "system": "whr",
            "config": {
                "w2": self.config.w2,
                "initial_rating": self.config.initial_rating,
                "initial_rd": self.config.initial_rd,
            },
            "fingerprint": fingerprint,
            "num_iterations": self._num_iterations,
        }
        if player_to_idx is not None:
            metadata["player_to_idx"] = {str(k): v for k, v in player_to_idx.items()}

        save_checkpoint(path, arrays, metadata)

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint state for warm-starting the next fit().

        Sets internal arrays so the next call to fit() will warm-start
        from the saved converged ratings instead of from zeros.

        Args:
            path: Path to .npz checkpoint file.

        Returns:
            Metadata dict (contains fingerprint, config, player_to_idx).
        """
        arrays, metadata = load_checkpoint(path)

        self._pd_r = arrays["pd_r"]
        self._pd_days = arrays["pd_days"]
        self._player_offsets = arrays["player_offsets"]
        self._checkpoint_loaded = True

        return metadata

    def reset(self) -> "WHR":
        """Reset the rating system."""
        self._player_offsets = None
        self._pd_days = None
        self._pd_r = None
        self._pd_uncertainty = None
        self._pd_game_offsets = None
        self._pd_game_opp_pd = None
        self._pd_game_score = None
        self._pd_to_player = None
        self._stored_player1 = None
        self._stored_player2 = None
        self._stored_scores = None
        self._stored_days = None
        self._last_refit_day = None
        self._checkpoint_loaded = False
        self._num_games_fitted = 0
        self._num_iterations = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WHR(w2={self.config.w2}, "
            f"max_iterations={self.config.max_iterations}, "
            f"warm_start={self.config.warm_start}, "
            f"active_set={self.config.use_active_set}, "
            f"anderson={self.config.anderson_window}, "
            f"players={players}, {status})"
        )
