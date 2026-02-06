"""
TrueSkill Through Time (TTT) - Numba implementation with sparse state storage.

Based on:
- Dangauthier et al., "TrueSkill Through Time: Revisiting the History of Chess" (2007)
- Glandfried's Python implementation: https://github.com/glandfried/TrueSkillThroughTime.py

Uses sparse appearance-based state storage instead of dense (batches × players)
arrays. Memory scales with actual player appearances (~500K) rather than the
dense product (~119M), giving ~240x memory reduction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...results.fitted_ratings import FittedTTTRatings
from ._numba_core import (
    build_appearance_structure,
    initial_forward_pass_sparse,
    run_convergence_sparse,
    extract_final_ratings_sparse,
    predict_proba_batch,
    predict_single,
    INF_SIGMA,
)


@dataclass
class TTTConfig:
    """Configuration for TrueSkill Through Time."""

    mu: float = 0.0  # Prior mean skill (internal scale, 0 = average)
    sigma: float = 6.0  # Prior skill std dev (internal scale) - reference default
    beta: float = 1.0  # Performance std dev (within-game noise) - reference default
    gamma: float = 0.03  # Skill dynamics per time unit - reference default
    max_iterations: int = 30  # Max forward-backward iterations for initial fit
    refit_max_iterations: int = 2  # Max iterations for periodic refits (faster)
    convergence_threshold: float = 1e-6  # Convergence threshold
    refit_interval: int = 1  # Days between refits (1 = refit daily)


class TrueSkillThroughTime(RatingSystem):
    """
    TrueSkill Through Time rating system.

    Uses forward-backward belief propagation with sparse state storage
    for globally consistent player skill estimates over time.

    Parameters:
        mu: Prior mean skill (default: 0.0, displayed as 1500)
        sigma: Prior skill std dev (default: 6.0, reference default)
        beta: Performance variability within games (default: 1.0)
        gamma: Skill drift rate per time unit (default: 0.03)
        max_iterations: Max belief propagation iterations (default: 30)
        convergence_threshold: Stop when max change < threshold

    Example:
        >>> ttt = TrueSkillThroughTime(sigma=6.0, beta=1.0)
        >>> ttt.fit(dataset)
        >>> fitted = ttt.get_fitted_ratings()
        >>> print(fitted.top(10))
    """

    system_type = RatingSystemType.BATCH

    # Scale factor for display (internal 0 = display 1500, like Elo)
    DISPLAY_OFFSET = 1500.0
    DISPLAY_SCALE = 400.0 / 6.0  # Map internal sigma=6 to ~67 Elo points

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 6.0,
        beta: float = 1.0,
        gamma: float = 0.03,
        max_iterations: int = 30,
        refit_max_iterations: int = 2,
        convergence_threshold: float = 1e-6,
        refit_interval: int = 1,
        num_players: Optional[int] = None,
    ):
        self.config = TTTConfig(
            mu=mu,
            sigma=sigma,
            beta=beta,
            gamma=gamma,
            max_iterations=max_iterations,
            refit_max_iterations=refit_max_iterations,
            convergence_threshold=convergence_threshold,
            refit_interval=refit_interval,
        )

        # Game data arrays
        self._num_batches = 0
        self._batch_offsets: Optional[np.ndarray] = None
        self._batch_times: Optional[np.ndarray] = None
        self._game_p1: Optional[np.ndarray] = None
        self._game_p2: Optional[np.ndarray] = None
        self._game_scores: Optional[np.ndarray] = None

        # Sparse appearance structures
        self._num_appearances = 0
        self._app_offsets: Optional[np.ndarray] = None
        self._app_player: Optional[np.ndarray] = None
        self._app_prev: Optional[np.ndarray] = None
        self._app_next: Optional[np.ndarray] = None
        self._app_batch: Optional[np.ndarray] = None
        self._player_last_app: Optional[np.ndarray] = None

        # Sparse state arrays (indexed by appearance, not batch*player)
        self._state_forward_mu: Optional[np.ndarray] = None
        self._state_forward_sigma: Optional[np.ndarray] = None
        self._state_backward_mu: Optional[np.ndarray] = None
        self._state_backward_sigma: Optional[np.ndarray] = None
        self._state_likelihood_mu: Optional[np.ndarray] = None
        self._state_likelihood_sigma: Optional[np.ndarray] = None

        # Pre-allocated temp arrays (size num_players, reused across sweeps)
        self._temp_fwd_mu: Optional[np.ndarray] = None
        self._temp_fwd_sigma: Optional[np.ndarray] = None
        self._temp_bwd_mu: Optional[np.ndarray] = None
        self._temp_bwd_sigma: Optional[np.ndarray] = None
        self._temp_lik_mu: Optional[np.ndarray] = None
        self._temp_lik_sigma: Optional[np.ndarray] = None

        # Metadata
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._player_names: Optional[Dict[int, str]] = None

        # Accumulated data for periodic refitting
        self._accum_p1: List[np.ndarray] = []
        self._accum_p2: List[np.ndarray] = []
        self._accum_scores: List[np.ndarray] = []
        self._accum_days: List[np.ndarray] = []
        self._last_refit_day: Optional[int] = None

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial TTT ratings."""
        return PlayerRatings(
            ratings=np.full(num_players, self.DISPLAY_OFFSET, dtype=np.float64),
            rd=np.full(num_players, self.config.sigma * self.DISPLAY_SCALE, dtype=np.float64),
            metadata={"system": "ttt", "config": self.config},
        )

    def _ensure_temp_arrays(self) -> None:
        """Ensure temp arrays are allocated at the right size."""
        if self._temp_fwd_mu is None or len(self._temp_fwd_mu) < self._num_players:
            self._temp_fwd_mu = np.zeros(self._num_players, dtype=np.float64)
            self._temp_fwd_sigma = np.full(self._num_players, INF_SIGMA, dtype=np.float64)
            self._temp_bwd_mu = np.zeros(self._num_players, dtype=np.float64)
            self._temp_bwd_sigma = np.full(self._num_players, INF_SIGMA, dtype=np.float64)
            self._temp_lik_mu = np.zeros(self._num_players, dtype=np.float64)
            self._temp_lik_sigma = np.full(self._num_players, INF_SIGMA, dtype=np.float64)

    def _build_batch_structure(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        days: np.ndarray,
    ) -> None:
        """
        Build batch and sparse appearance structures from game arrays.
        """
        n_games = len(player1)

        # Get unique days and sort games by day
        unique_days = np.unique(days)
        self._num_batches = len(unique_days)

        day_to_batch = {int(d): i for i, d in enumerate(unique_days)}

        sort_order = np.argsort(days)
        self._game_p1 = player1[sort_order].astype(np.int64)
        self._game_p2 = player2[sort_order].astype(np.int64)
        self._game_scores = scores[sort_order].astype(np.float64)

        sorted_days = days[sort_order]

        # Build batch offsets
        self._batch_offsets = np.zeros(self._num_batches + 1, dtype=np.int64)
        self._batch_times = np.zeros(self._num_batches, dtype=np.float64)

        current_batch = -1
        for i, d in enumerate(sorted_days):
            batch_idx = day_to_batch[int(d)]
            if batch_idx != current_batch:
                self._batch_offsets[batch_idx] = i
                self._batch_times[batch_idx] = float(d)
                current_batch = batch_idx
        self._batch_offsets[self._num_batches] = n_games

        # Build sparse appearance structure
        (self._app_offsets, self._app_player, self._app_prev,
         self._app_next, self._app_batch, self._player_last_app) = \
            build_appearance_structure(
                self._num_batches,
                self._batch_offsets,
                self._game_p1,
                self._game_p2,
                self._num_players,
            )

        self._num_appearances = len(self._app_player)

        # Allocate sparse state arrays
        na = self._num_appearances
        self._state_forward_mu = np.zeros(na, dtype=np.float64)
        self._state_forward_sigma = np.full(na, INF_SIGMA, dtype=np.float64)
        self._state_backward_mu = np.zeros(na, dtype=np.float64)
        self._state_backward_sigma = np.full(na, INF_SIGMA, dtype=np.float64)
        self._state_likelihood_mu = np.zeros(na, dtype=np.float64)
        self._state_likelihood_sigma = np.full(na, INF_SIGMA, dtype=np.float64)

        # Ensure temp arrays
        self._ensure_temp_arrays()

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "TrueSkillThroughTime":
        """
        Fit TTT on a dataset.

        Args:
            dataset: Game dataset to fit on
            end_day: Last day to include (inclusive). Useful for backtesting.
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
        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

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

        # Build data structures (sparse)
        self._build_batch_structure(player1, player2, scores, days)

        # Initial forward pass (all batches)
        initial_forward_pass_sparse(
            self._num_batches,
            self._batch_offsets,
            self._batch_times,
            self._game_p1,
            self._game_p2,
            self._game_scores,
            self._num_players,
            self._app_offsets,
            self._app_player,
            self._app_prev,
            self._app_batch,
            self._state_forward_mu,
            self._state_forward_sigma,
            self._state_backward_mu,
            self._state_backward_sigma,
            self._state_likelihood_mu,
            self._state_likelihood_sigma,
            self._temp_fwd_mu,
            self._temp_fwd_sigma,
            self._temp_bwd_mu,
            self._temp_bwd_sigma,
            self._temp_lik_mu,
            self._temp_lik_sigma,
            self.config.mu,
            self.config.sigma,
            self.config.beta,
            self.config.gamma,
            0,  # start_batch
        )

        # Run convergence
        self._num_iterations = run_convergence_sparse(
            self._num_batches,
            self._batch_offsets,
            self._batch_times,
            self._game_p1,
            self._game_p2,
            self._game_scores,
            self._num_players,
            self._app_offsets,
            self._app_player,
            self._app_prev,
            self._app_next,
            self._app_batch,
            self._state_forward_mu,
            self._state_forward_sigma,
            self._state_backward_mu,
            self._state_backward_sigma,
            self._state_likelihood_mu,
            self._state_likelihood_sigma,
            self._temp_fwd_mu,
            self._temp_fwd_sigma,
            self._temp_bwd_mu,
            self._temp_bwd_sigma,
            self._temp_lik_mu,
            self._temp_lik_sigma,
            self.config.mu,
            self.config.sigma,
            self.config.beta,
            self.config.gamma,
            self.config.max_iterations,
            self.config.convergence_threshold,
        )

        # Extract final ratings
        ratings = np.empty(self._num_players, dtype=np.float64)
        rd = np.empty(self._num_players, dtype=np.float64)

        extract_final_ratings_sparse(
            self._num_players,
            self._player_last_app,
            self._state_forward_mu,
            self._state_forward_sigma,
            self._state_backward_mu,
            self._state_backward_sigma,
            self._state_likelihood_mu,
            self._state_likelihood_sigma,
            ratings,
            rd,
            self.config.mu,
            self.config.sigma,
            self.DISPLAY_SCALE,
            self.DISPLAY_OFFSET,
        )

        self._ratings = PlayerRatings(
            ratings=ratings,
            rd=rd,
            metadata={"system": "ttt", "config": self.config},
        )

        self._num_games_fitted = n_games
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None

        # Store data for potential refitting
        if self.config.refit_interval > 0:
            self._accum_p1 = [player1.copy()]
            self._accum_p2 = [player2.copy()]
            self._accum_scores = [scores.copy()]
            self._accum_days = [days.copy()]
            self._last_refit_day = self._current_day

        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
                self._ratings.rd[int(player1)],
                self._ratings.rd[int(player2)],
                self.config.beta,
                self.DISPLAY_SCALE,
                self.DISPLAY_OFFSET,
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(
            p1, p2,
            self._ratings.ratings,
            self._ratings.rd,
            self.config.beta,
            self.DISPLAY_SCALE,
            self.DISPLAY_OFFSET,
        )

    def get_fitted_ratings(self) -> FittedTTTRatings:
        """Get a queryable fitted ratings object."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return FittedTTTRatings(
            ratings=self._ratings.ratings.copy(),
            rd=self._ratings.rd.copy(),
            sigma=self.config.sigma,
            beta=self.config.beta,
            gamma=self.config.gamma,
            display_scale=self.DISPLAY_SCALE,
            display_offset=self.DISPLAY_OFFSET,
            num_games_fitted=self._num_games_fitted,
            num_iterations=self._num_iterations,
            last_day=self._current_day,
            player_names=self._player_names,
            rating_history={},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with a new batch - for TTT this is a no-op.
        TTT is a batch algorithm that requires full refit for updates.
        """
        pass

    def update(self, batch: GameBatch) -> "TrueSkillThroughTime":
        """
        Incrementally update ratings with a new batch of games.

        Accumulates data and refits periodically based on refit_interval.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before updating. Call fit() first.")

        if self.config.refit_interval <= 0:
            return self

        # Accumulate new data
        n_games = len(batch)
        days_array = np.full(n_games, batch.day, dtype=np.int32)
        self._accum_p1.append(batch.player1.copy())
        self._accum_p2.append(batch.player2.copy())
        self._accum_scores.append(batch.scores.copy())
        self._accum_days.append(days_array)

        # Check if it's time to refit
        if self._last_refit_day is None:
            self._last_refit_day = batch.day

        if batch.day - self._last_refit_day >= self.config.refit_interval:
            self._refit_from_accumulated()
            self._last_refit_day = batch.day

        self._current_day = batch.day
        return self

    def _refit_from_accumulated(self) -> None:
        """Refit the model on all accumulated data using warm-start.

        Rebuilds the sparse structure from all data but preserves converged
        state from previous batches. Only runs the forward pass on new
        batches, then convergence on the full graph (which converges quickly
        since most messages are already stable).
        """
        if not self._accum_p1:
            return

        player1 = np.concatenate(self._accum_p1)
        player2 = np.concatenate(self._accum_p2)
        scores = np.concatenate(self._accum_scores)
        days = np.concatenate(self._accum_days)

        max_player = max(player1.max(), player2.max())
        if self._num_players is None or max_player >= self._num_players:
            self._num_players = int(max_player) + 1
            self._ratings = self._initialize_ratings(self._num_players)

        # Save old state for warm-start
        old_num_batches = self._num_batches
        old_num_appearances = self._num_appearances
        old_fwd_mu = self._state_forward_mu
        old_fwd_sigma = self._state_forward_sigma
        old_bwd_mu = self._state_backward_mu
        old_bwd_sigma = self._state_backward_sigma
        old_lik_mu = self._state_likelihood_mu
        old_lik_sigma = self._state_likelihood_sigma

        # Rebuild sparse structures (allocates fresh arrays)
        self._build_batch_structure(player1, player2, scores, days)

        # Restore old state into new arrays for previously-computed appearances.
        # build_appearance_structure is deterministic: old games in the same
        # order produce the same appearance indices, so indices 0..old-1 match.
        if old_fwd_mu is not None and old_num_appearances > 0:
            n = old_num_appearances
            self._state_forward_mu[:n] = old_fwd_mu[:n]
            self._state_forward_sigma[:n] = old_fwd_sigma[:n]
            self._state_backward_mu[:n] = old_bwd_mu[:n]
            self._state_backward_sigma[:n] = old_bwd_sigma[:n]
            self._state_likelihood_mu[:n] = old_lik_mu[:n]
            self._state_likelihood_sigma[:n] = old_lik_sigma[:n]

        # Forward pass only for new batches (warm-start)
        initial_forward_pass_sparse(
            self._num_batches,
            self._batch_offsets,
            self._batch_times,
            self._game_p1,
            self._game_p2,
            self._game_scores,
            self._num_players,
            self._app_offsets,
            self._app_player,
            self._app_prev,
            self._app_batch,
            self._state_forward_mu,
            self._state_forward_sigma,
            self._state_backward_mu,
            self._state_backward_sigma,
            self._state_likelihood_mu,
            self._state_likelihood_sigma,
            self._temp_fwd_mu,
            self._temp_fwd_sigma,
            self._temp_bwd_mu,
            self._temp_bwd_sigma,
            self._temp_lik_mu,
            self._temp_lik_sigma,
            self.config.mu,
            self.config.sigma,
            self.config.beta,
            self.config.gamma,
            old_num_batches,  # start_batch: skip already-computed batches
        )

        # Run convergence (fewer iterations needed thanks to warm-start)
        self._num_iterations = run_convergence_sparse(
            self._num_batches,
            self._batch_offsets,
            self._batch_times,
            self._game_p1,
            self._game_p2,
            self._game_scores,
            self._num_players,
            self._app_offsets,
            self._app_player,
            self._app_prev,
            self._app_next,
            self._app_batch,
            self._state_forward_mu,
            self._state_forward_sigma,
            self._state_backward_mu,
            self._state_backward_sigma,
            self._state_likelihood_mu,
            self._state_likelihood_sigma,
            self._temp_fwd_mu,
            self._temp_fwd_sigma,
            self._temp_bwd_mu,
            self._temp_bwd_sigma,
            self._temp_lik_mu,
            self._temp_lik_sigma,
            self.config.mu,
            self.config.sigma,
            self.config.beta,
            self.config.gamma,
            self.config.refit_max_iterations,
            self.config.convergence_threshold,
        )

        # Extract final ratings
        ratings = np.empty(self._num_players, dtype=np.float64)
        rd = np.empty(self._num_players, dtype=np.float64)

        extract_final_ratings_sparse(
            self._num_players,
            self._player_last_app,
            self._state_forward_mu,
            self._state_forward_sigma,
            self._state_backward_mu,
            self._state_backward_sigma,
            self._state_likelihood_mu,
            self._state_likelihood_sigma,
            ratings,
            rd,
            self.config.mu,
            self.config.sigma,
            self.DISPLAY_SCALE,
            self.DISPLAY_OFFSET,
        )

        self._ratings = PlayerRatings(
            ratings=ratings,
            rd=rd,
            metadata={"system": "ttt", "config": self.config},
        )

        self._num_games_fitted = len(player1)

    def reset(self) -> "TrueSkillThroughTime":
        """Reset the rating system."""
        self._num_batches = 0
        self._batch_offsets = None
        self._batch_times = None
        self._game_p1 = None
        self._game_p2 = None
        self._game_scores = None
        self._num_appearances = 0
        self._app_offsets = None
        self._app_player = None
        self._app_prev = None
        self._app_next = None
        self._app_batch = None
        self._player_last_app = None
        self._state_forward_mu = None
        self._state_forward_sigma = None
        self._state_backward_mu = None
        self._state_backward_sigma = None
        self._state_likelihood_mu = None
        self._state_likelihood_sigma = None
        self._temp_fwd_mu = None
        self._temp_fwd_sigma = None
        self._temp_bwd_mu = None
        self._temp_bwd_sigma = None
        self._temp_lik_mu = None
        self._temp_lik_sigma = None
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._accum_p1 = []
        self._accum_p2 = []
        self._accum_scores = []
        self._accum_days = []
        self._last_refit_day = None
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        apps = self._num_appearances or 0
        return (
            f"TrueSkillThroughTime(sigma={self.config.sigma:.2f}, "
            f"beta={self.config.beta:.2f}, gamma={self.config.gamma:.3f}, "
            f"players={players}, appearances={apps}, {status})"
        )
