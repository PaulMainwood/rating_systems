"""
Weighted Whole History Rating (WWHR) - WHR with per-game weights.

Extends WHR by allowing per-game weights that scale the gradient and Hessian
contributions from each game. Decisive wins (higher weight) become more
informative for rating estimation.

Virtual game priors and Wiener process priors are NOT weighted — they are
structural components, not data.

With weights=1.0, this produces identical results to standard WHR.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...data.checkpoint import save_checkpoint, load_checkpoint
from ...results.fitted_ratings import FittedWHRRatings
from ..whr._numba_core import (
    LN10_400,
    build_player_day_indices,
    extract_current_ratings,
    extract_player_last_day,
    predict_proba_batch,
    predict_proba_batch_at_day,
    predict_single,
    predict_single_at_day,
    get_top_n_indices,
    warm_start_ratings,
)
from ._numba_core import (
    fill_game_arrays_weighted,
    fill_game_arrays_weighted_h,
    compute_uncertainties_weighted,
    compute_uncertainties_weighted_h,
    run_all_iterations_weighted,
    run_all_iterations_accelerated_weighted,
    run_all_iterations_accelerated_weighted_h,
)

from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def _predict_proba_batch_h(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
    handicaps: np.ndarray,
) -> np.ndarray:
    """Predict probabilities with per-game handicaps (log-gamma scale)."""
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)
    for i in prange(n):
        r1 = (ratings[player1[i]] - 1500.0) * LN10_400
        r2 = (ratings[player2[i]] - 1500.0) * LN10_400
        x = r1 - r2 + handicaps[i]
        if x >= 0:
            proba[i] = 1.0 / (1.0 + np.exp(-x))
        else:
            ex = np.exp(x)
            proba[i] = ex / (1.0 + ex)
    return proba


@dataclass
class WWHRConfig:
    """Configuration for Weighted WHR rating system."""

    w2: float = 300.0
    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    max_iterations: int = 50
    refit_max_iterations: int = 5
    refit_interval: int = 1
    convergence_threshold: float = 1e-6
    warm_start: bool = True
    use_active_set: bool = True
    anderson_window: int = 5
    use_jacobi: bool = True  # Jacobi parallel iteration (vs sequential Gauss-Seidel)

    def __post_init__(self):
        if self.refit_max_iterations is None:
            self.refit_max_iterations = self.max_iterations


class WeightedWHR(RatingSystem):
    """
    Weighted Whole History Rating system with Numba acceleration.

    Extends WHR with per-game weights that scale game contributions to the
    gradient and Hessian. This makes decisive wins more informative for
    rating estimation.

    With all weights = 1.0, this is identical to standard WHR.

    Parameters:
        w2: Wiener variance per time unit (default: 300.0)
        initial_rating: Starting rating in Elo scale (default: 1500)
        max_iterations: Maximum Newton-Raphson iterations (default: 50)
        convergence_threshold: Stop when max rating change < threshold

    Example:
        >>> wwhr = WeightedWHR(w2=300.0, max_iterations=50)
        >>> wwhr.fit(dataset, weights=weights)
        >>> fitted = wwhr.get_fitted_ratings()
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
        self.config = WWHRConfig(
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
        self._player_offsets: Optional[np.ndarray] = None
        self._pd_days: Optional[np.ndarray] = None
        self._pd_r: Optional[np.ndarray] = None
        self._pd_uncertainty: Optional[np.ndarray] = None
        self._pd_game_offsets: Optional[np.ndarray] = None
        self._pd_game_opp_pd: Optional[np.ndarray] = None
        self._pd_game_score: Optional[np.ndarray] = None
        self._pd_game_weights: Optional[np.ndarray] = None
        self._pd_game_handicaps: Optional[np.ndarray] = None
        self._pd_to_player: Optional[np.ndarray] = None
        self._pd_w2_r: Optional[np.ndarray] = None
        self._player_last_day: Optional[np.ndarray] = None

        # Metadata
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._player_names: Optional[Dict[int, str]] = None

        # Store raw data for refitting
        self._stored_player1: Optional[np.ndarray] = None
        self._stored_player2: Optional[np.ndarray] = None
        self._stored_scores: Optional[np.ndarray] = None
        self._stored_days: Optional[np.ndarray] = None
        self._stored_weights: Optional[np.ndarray] = None
        self._stored_handicaps: Optional[np.ndarray] = None
        self._last_refit_day: Optional[int] = None

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial WWHR ratings."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            rd=np.full(num_players, self.config.initial_rd, dtype=np.float64),
            metadata={"system": "wwhr", "config": self.config},
        )

    def _build_data_structures(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        days: np.ndarray,
        weights: np.ndarray,
        num_players: int,
        handicaps: Optional[np.ndarray] = None,
    ) -> None:
        """
        Build CSR-like data structures from game arrays with weights and optional handicaps.

        Same as WHR but also allocates and fills pd_game_weights (and pd_game_handicaps).
        """
        n_games = len(player1)

        # Step 1+2: Build player-day CSR structure in O(n_games)
        # Exploits day-sorted input to avoid O(n log n) np.unique
        (self._player_offsets, self._pd_days, pd1_indices, pd2_indices, total_pd
        ) = build_player_day_indices(player1, player2, days, num_players, n_games)

        # Reverse mapping: player-day index -> player_id
        self._pd_to_player = np.repeat(
            np.arange(num_players, dtype=np.int64),
            np.diff(self._player_offsets),
        )

        # Initialize ratings to 0 (log-gamma scale)
        self._pd_r = np.zeros(total_pd, dtype=np.float64)
        self._pd_uncertainty = np.full(total_pd, self.config.initial_rd, dtype=np.float64)

        # Step 3: Count games per player-day
        pd_game_counts = np.zeros(total_pd, dtype=np.int64)
        np.add.at(pd_game_counts, pd1_indices, 1)
        np.add.at(pd_game_counts, pd2_indices, 1)

        self._pd_game_offsets = np.zeros(total_pd + 1, dtype=np.int64)
        np.cumsum(pd_game_counts, out=self._pd_game_offsets[1:])

        total_game_refs = self._pd_game_offsets[-1]

        # Step 4: Fill game arrays with weights (and handicaps if provided)
        self._pd_game_opp_pd = np.empty(total_game_refs, dtype=np.int64)
        self._pd_game_score = np.empty(total_game_refs, dtype=np.float64)
        self._pd_game_weights = np.empty(total_game_refs, dtype=np.float64)

        if handicaps is not None:
            self._pd_game_handicaps = np.empty(total_game_refs, dtype=np.float64)
            fill_game_arrays_weighted_h(
                n_games,
                pd1_indices,
                pd2_indices,
                scores,
                weights,
                handicaps,
                self._pd_game_offsets,
                self._pd_game_opp_pd,
                self._pd_game_score,
                self._pd_game_weights,
                self._pd_game_handicaps,
            )
        else:
            self._pd_game_handicaps = None
            fill_game_arrays_weighted(
                n_games,
                pd1_indices,
                pd2_indices,
                scores,
                weights,
                self._pd_game_offsets,
                self._pd_game_opp_pd,
                self._pd_game_score,
                self._pd_game_weights,
            )

        self._pd_w2_r = np.full(total_pd, self._w2_r, dtype=np.float64)

    def _run_optimization(self, max_iterations: Optional[int] = None) -> None:
        """Run weighted Newton-Raphson optimization."""
        if self._player_offsets is None or self._num_players is None:
            return

        if max_iterations is None:
            max_iterations = self.config.max_iterations

        has_handicaps = self._pd_game_handicaps is not None

        if has_handicaps:
            # Use handicap-aware solver
            self._num_iterations = run_all_iterations_accelerated_weighted_h(
                self._num_players,
                self._player_offsets,
                self._pd_days,
                self._pd_r,
                self._pd_game_offsets,
                self._pd_game_opp_pd,
                self._pd_game_score,
                self._pd_game_weights,
                self._pd_game_handicaps,
                self._pd_w2_r,
                max_iterations,
                self.config.convergence_threshold,
                self.config.anderson_window,
                self.config.use_active_set,
                self._pd_to_player,
                self.config.use_jacobi,
            )
            compute_uncertainties_weighted_h(
                self._num_players,
                self._player_offsets,
                self._pd_days,
                self._pd_r,
                self._pd_uncertainty,
                self._pd_game_offsets,
                self._pd_game_opp_pd,
                self._pd_game_score,
                self._pd_game_weights,
                self._pd_game_handicaps,
                self._pd_w2_r,
            )
        elif self.config.use_active_set or self.config.anderson_window > 0 or self.config.use_jacobi:
            self._num_iterations = run_all_iterations_accelerated_weighted(
                self._num_players,
                self._player_offsets,
                self._pd_days,
                self._pd_r,
                self._pd_game_offsets,
                self._pd_game_opp_pd,
                self._pd_game_score,
                self._pd_game_weights,
                self._pd_w2_r,
                max_iterations,
                self.config.convergence_threshold,
                self.config.anderson_window,
                self.config.use_active_set,
                self._pd_to_player,
                self.config.use_jacobi,
            )
            compute_uncertainties_weighted(
                self._num_players,
                self._player_offsets,
                self._pd_days,
                self._pd_r,
                self._pd_uncertainty,
                self._pd_game_offsets,
                self._pd_game_opp_pd,
                self._pd_game_score,
                self._pd_game_weights,
                self._pd_w2_r,
            )
        else:
            self._num_iterations = run_all_iterations_weighted(
                self._num_players,
                self._player_offsets,
                self._pd_days,
                self._pd_r,
                self._pd_game_offsets,
                self._pd_game_opp_pd,
                self._pd_game_score,
                self._pd_game_weights,
                self._pd_w2_r,
                max_iterations,
                self.config.convergence_threshold,
            )
            compute_uncertainties_weighted(
                self._num_players,
                self._player_offsets,
                self._pd_days,
                self._pd_r,
                self._pd_uncertainty,
                self._pd_game_offsets,
                self._pd_game_opp_pd,
                self._pd_game_score,
                self._pd_game_weights,
                self._pd_w2_r,
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
            metadata={"system": "wwhr", "config": self.config},
        )

        # Cache last active day per player for time-aware predictions
        self._player_last_day = extract_player_last_day(
            self._num_players, self._player_offsets, self._pd_days,
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with a new batch (refits based on refit_interval)."""
        raise NotImplementedError("Use update_weighted() or update() instead")

    def _refit(self, max_iterations: Optional[int] = None) -> None:
        """Refit on all stored data."""
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
            self._stored_weights,
            self._num_players,
            handicaps=self._stored_handicaps,
        )

        # Apply warm start
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
        weights: Optional[np.ndarray] = None,
        handicaps: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WeightedWHR":
        """
        Fit WWHR on a dataset.

        Args:
            dataset: Game dataset to fit on
            weights: Per-game weights (None = all 1.0)
            handicaps: Per-game handicap for player 1 (log-gamma scale).
                       Positive = player 1 advantage. If None, no handicaps.
            end_day: Last day to include (inclusive)
            player_names: Optional mapping of player_id -> name

        Returns:
            self (for method chaining)
        """
        self._player_names = player_names

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

        # Create per-game day array
        n_games = len(player1)
        days = np.empty(n_games, dtype=np.int32)
        n_days = len(day_indices)
        for d in range(n_days):
            start = day_offsets[d]
            end = day_offsets[d + 1]
            days[start:end] = day_indices[d]

        # Default weights
        if weights is None:
            weights = np.ones(n_games, dtype=np.float64)

        # Store for potential refitting
        self._stored_player1 = player1.copy()
        self._stored_player2 = player2.copy()
        self._stored_scores = scores.copy()
        self._stored_days = days
        self._stored_weights = weights.copy()
        self._stored_handicaps = handicaps.copy() if handicaps is not None else None

        # Build data structures and optimize
        self._build_data_structures(
            player1, player2, scores, days, weights, self._num_players,
            handicaps=handicaps,
        )
        self._run_optimization()
        self._extract_current_ratings()

        self._num_games_fitted = n_games
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None

        return self

    def update_weighted(
        self,
        batch: GameBatch,
        weights: np.ndarray,
        handicaps: Optional[np.ndarray] = None,
    ) -> "WeightedWHR":
        """Update with new weighted games (and optional handicaps) by refitting on all data."""
        if not self._fitted:
            raise ValueError("Model must be fitted before updating")

        n = len(batch)
        h = handicaps if handicaps is not None else np.zeros(n, dtype=np.float64)

        # Append new data
        if self._stored_player1 is None:
            self._stored_player1 = batch.player1.copy()
            self._stored_player2 = batch.player2.copy()
            self._stored_scores = batch.scores.copy()
            self._stored_days = np.full(n, batch.day, dtype=np.int32)
            self._stored_weights = weights.copy()
            self._stored_handicaps = h.copy()
            self._last_refit_day = batch.day
        else:
            self._stored_player1 = np.concatenate([self._stored_player1, batch.player1])
            self._stored_player2 = np.concatenate([self._stored_player2, batch.player2])
            self._stored_scores = np.concatenate([self._stored_scores, batch.scores])
            self._stored_days = np.concatenate(
                [self._stored_days, np.full(n, batch.day, dtype=np.int32)]
            )
            self._stored_weights = np.concatenate([self._stored_weights, weights])
            if self._stored_handicaps is not None:
                self._stored_handicaps = np.concatenate([self._stored_handicaps, h])
            else:
                # fit() was called without handicaps — pad zeros for fit data
                n_prev = len(self._stored_player1) - n
                self._stored_handicaps = np.concatenate([
                    np.zeros(n_prev, dtype=np.float64), h
                ])

        # Check if it's time to refit
        if self._last_refit_day is None:
            self._last_refit_day = batch.day

        if batch.day - self._last_refit_day >= self.config.refit_interval:
            self._refit()
            self._last_refit_day = batch.day

        self._current_day = batch.day

        if self._player_last_day is not None:
            players = np.unique(np.concatenate([batch.player1, batch.player2]))
            self._player_last_day[players] = batch.day

        return self

    def update(self, batch: GameBatch) -> "WeightedWHR":
        """Update with new games (unit weights) by refitting on all data."""
        return self.update_weighted(batch, np.ones(len(batch), dtype=np.float64))

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        handicaps: Optional[Union[float, np.ndarray]] = None,
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict probability that player1 beats player2.

        When handicaps are provided, they shift the Bradley-Terry probability
        on the log-gamma scale (positive = player1 advantage).
        When day is provided (without handicaps), uses damped sigmoid for Wiener drift.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if day is not None and handicaps is None and self._player_last_day is not None:
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

        if handicaps is not None:
            if isinstance(player1, (int, np.integer)) and isinstance(
                player2, (int, np.integer)
            ):
                r1 = (self._ratings.ratings[int(player1)] - 1500.0) * LN10_400
                r2 = (self._ratings.ratings[int(player2)] - 1500.0) * LN10_400
                h = float(handicaps)
                return 1.0 / (1.0 + np.exp(-(r1 - r2 + h)))
            p1 = np.ascontiguousarray(player1, dtype=np.int64)
            p2 = np.ascontiguousarray(player2, dtype=np.int64)
            h = np.ascontiguousarray(handicaps, dtype=np.float64)
            return _predict_proba_batch_h(
                p1, p2, self._ratings.ratings, h,
            )

        if isinstance(player1, (int, np.integer)) and isinstance(
            player2, (int, np.integer)
        ):
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(p1, p2, self._ratings.ratings)

    def get_fitted_ratings(self) -> FittedWHRRatings:
        """Get a queryable fitted ratings object."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

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

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return get_top_n_indices(self._ratings.ratings, n)

    def snapshot(self) -> dict:
        """Snapshot full WWHR state including CSR structures.

        The base class snapshot only saves ratings/num_players/current_day,
        losing the CSR player-day structures that WWHR needs for correct
        warm-start refits during walk-forward.  Without these, update_weighted()
        refits without training history, producing divergent predictions.
        """
        if not self._fitted or self._ratings is None:
            raise ValueError("Model must be fitted before snapshotting.")
        state = {
            "ratings": self._ratings.clone(),
            "num_players": self._num_players,
            "current_day": self._current_day,
            "num_games_fitted": self._num_games_fitted,
            "num_iterations": self._num_iterations,
            "last_refit_day": self._last_refit_day,
        }
        for attr in (
            "_pd_r", "_pd_days", "_player_offsets", "_pd_uncertainty",
            "_pd_game_offsets", "_pd_game_opp_pd", "_pd_game_score",
            "_pd_game_weights", "_pd_game_handicaps",
            "_pd_to_player", "_pd_w2_r", "_player_last_day",
            "_stored_player1", "_stored_player2",
            "_stored_scores", "_stored_days",
            "_stored_weights", "_stored_handicaps",
        ):
            val = getattr(self, attr)
            state[attr] = val.copy() if val is not None else None
        return state

    def restore(self, state: dict) -> None:
        """Restore full WWHR state from snapshot."""
        self._num_players = state["num_players"]
        self._current_day = state["current_day"]
        self._num_games_fitted = state["num_games_fitted"]
        self._num_iterations = state["num_iterations"]
        self._last_refit_day = state["last_refit_day"]
        self._fitted = True
        self._ratings = state["ratings"].clone()
        for attr in (
            "_pd_r", "_pd_days", "_player_offsets", "_pd_uncertainty",
            "_pd_game_offsets", "_pd_game_opp_pd", "_pd_game_score",
            "_pd_game_weights", "_pd_game_handicaps",
            "_pd_to_player", "_pd_w2_r", "_player_last_day",
            "_stored_player1", "_stored_player2",
            "_stored_scores", "_stored_days",
            "_stored_weights", "_stored_handicaps",
        ):
            val = state[attr]
            setattr(self, attr, val.copy() if val is not None else None)

    def save_state(self, path: str) -> None:
        """Save fitted WWHR state to .npz file."""
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
            "pd_game_weights": self._pd_game_weights,
            "pd_to_player": self._pd_to_player,
            "pd_w2_r": self._pd_w2_r,
        }
        if self._pd_game_handicaps is not None:
            arrays["pd_game_handicaps"] = self._pd_game_handicaps
        if self._stored_player1 is not None:
            arrays["stored_player1"] = self._stored_player1
            arrays["stored_player2"] = self._stored_player2
            arrays["stored_scores"] = self._stored_scores
            arrays["stored_days"] = self._stored_days
            arrays["stored_weights"] = self._stored_weights
        if self._stored_handicaps is not None:
            arrays["stored_handicaps"] = self._stored_handicaps

        metadata = {
            "system_class": "WeightedWHR",
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
        """Restore fitted WWHR state from .npz file."""
        arrays, metadata = load_checkpoint(path)

        self._num_players = metadata["num_players"]
        self._current_day = metadata["current_day"]
        self._num_games_fitted = metadata.get("num_games_fitted", 0)
        self._num_iterations = metadata.get("num_iterations", 0)
        self._fitted = True

        self._ratings = PlayerRatings(
            ratings=arrays["ratings"],
            rd=arrays["rd"],
            metadata={"system": "wwhr", "config": self.config},
        )

        self._pd_r = arrays["pd_r"]
        self._pd_days = arrays["pd_days"]
        self._player_offsets = arrays["player_offsets"]
        self._pd_uncertainty = arrays.get("pd_uncertainty")
        self._pd_game_offsets = arrays.get("pd_game_offsets")
        self._pd_game_opp_pd = arrays.get("pd_game_opp_pd")
        self._pd_game_score = arrays.get("pd_game_score")
        self._pd_game_weights = arrays.get("pd_game_weights")
        self._pd_game_handicaps = arrays.get("pd_game_handicaps")
        self._pd_to_player = arrays.get("pd_to_player")
        self._pd_w2_r = arrays.get("pd_w2_r")
        if self._pd_w2_r is None and self._pd_r is not None:
            # Backward compat: old checkpoint without pd_w2_r
            self._pd_w2_r = np.full(len(self._pd_r), self._w2_r, dtype=np.float64)

        if "stored_player1" in arrays:
            self._stored_player1 = arrays["stored_player1"]
            self._stored_player2 = arrays["stored_player2"]
            self._stored_scores = arrays["stored_scores"]
            self._stored_days = arrays["stored_days"]
            self._stored_weights = arrays.get("stored_weights")
        self._stored_handicaps = arrays.get("stored_handicaps")

        # Cache last active day per player for time-aware predictions
        self._player_last_day = extract_player_last_day(
            self._num_players, self._player_offsets, self._pd_days,
        )

    def reset(self) -> "WeightedWHR":
        """Reset the rating system."""
        self._player_offsets = None
        self._pd_days = None
        self._pd_r = None
        self._pd_uncertainty = None
        self._pd_game_offsets = None
        self._pd_game_opp_pd = None
        self._pd_game_score = None
        self._pd_game_weights = None
        self._pd_game_handicaps = None
        self._pd_to_player = None
        self._pd_w2_r = None
        self._stored_player1 = None
        self._stored_player2 = None
        self._stored_scores = None
        self._stored_days = None
        self._stored_weights = None
        self._stored_handicaps = None
        self._last_refit_day = None
        self._num_games_fitted = 0
        self._num_iterations = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WeightedWHR(w2={self.config.w2}, "
            f"max_iterations={self.config.max_iterations}, "
            f"warm_start={self.config.warm_start}, "
            f"active_set={self.config.use_active_set}, "
            f"anderson={self.config.anderson_window}, "
            f"players={players}, {status})"
        )
