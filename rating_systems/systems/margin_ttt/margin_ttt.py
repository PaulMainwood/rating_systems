"""
MarginTTT — Gaussian margin-of-victory TrueSkill Through Time.

Replaces binary win/loss observations with Gaussian margin observations:
    margin ~ N(skill_i - skill_j, sigma_margin²)

This extracts richer signal from match scores. Prediction remains binary:
    P(win) = Φ(diff_mu / diff_sigma), identical to TTT.

Parameters:
    mu, sigma, beta, gamma: Same as TTT
    margin_scale: Converts normalised [0,1] margin → skill units
    sigma_margin: Observation noise in skill units
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...data.checkpoint import save_checkpoint, load_checkpoint
from ...results.fitted_ratings import FittedTTTRatings

from ..trueskill_through_time._numba_core import (
    build_appearance_structure,
    extract_final_ratings_sparse,
    predict_proba_batch,
    predict_proba_batch_at_day,
    predict_single,
    predict_single_at_day,
    extract_player_last_day_ttt,
    INF_SIGMA,
)

from ._numba_core import (
    initial_forward_pass_margin,
    run_convergence_margin,
)


# Default margin used when actual score is missing (modest default with correct sign)
DEFAULT_MARGIN = 0.15


@dataclass
class MarginTTTConfig:
    """Configuration for MarginTTT."""

    mu: float = 0.0
    sigma: float = 4.0
    beta: float = 2.3
    gamma: float = 0.075
    margin_scale: float = 10.0     # normalised margin → skill units
    sigma_margin: float = 3.0      # observation noise (skill units)
    max_iterations: int = 50
    refit_max_iterations: int = 1
    convergence_threshold: float = 1e-6
    refit_interval: int = 1


class MarginTTT(RatingSystem):
    """
    MarginTTT rating system.

    Uses Gaussian margin observations instead of binary outcomes.
    Forward-backward belief propagation with sparse state storage.

    Parameters:
        mu: Prior mean skill (default: 0.0, displayed as 1500)
        sigma: Prior skill std dev
        beta: Performance variability within games
        gamma: Skill drift rate per time unit
        margin_scale: Converts normalised margin to skill units
        sigma_margin: Observation noise for margin (skill units)
        max_iterations: Max belief propagation iterations
        convergence_threshold: Stop when max change < threshold

    Example:
        >>> mttt = MarginTTT(sigma=4.0, beta=2.3, margin_scale=10.0)
        >>> mttt.fit(dataset, margins=margin_array)
        >>> fitted = mttt.get_fitted_ratings()
    """

    system_type = RatingSystemType.BATCH

    DISPLAY_OFFSET = 1500.0
    DISPLAY_SCALE = 400.0 / 6.0

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 4.0,
        beta: float = 2.3,
        gamma: float = 0.075,
        margin_scale: float = 10.0,
        sigma_margin: float = 3.0,
        max_iterations: int = 50,
        refit_max_iterations: int = 1,
        convergence_threshold: float = 1e-6,
        refit_interval: int = 1,
        num_players: Optional[int] = None,
    ):
        self.config = MarginTTTConfig(
            mu=mu,
            sigma=sigma,
            beta=beta,
            gamma=gamma,
            margin_scale=margin_scale,
            sigma_margin=sigma_margin,
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
        self._game_margins: Optional[np.ndarray] = None  # signed + scaled

        # Sparse appearance structures
        self._num_appearances = 0
        self._app_offsets: Optional[np.ndarray] = None
        self._app_player: Optional[np.ndarray] = None
        self._app_prev: Optional[np.ndarray] = None
        self._app_next: Optional[np.ndarray] = None
        self._app_batch: Optional[np.ndarray] = None
        self._player_last_app: Optional[np.ndarray] = None
        self._player_last_day: Optional[np.ndarray] = None

        # Sparse state arrays
        self._state_forward_mu: Optional[np.ndarray] = None
        self._state_forward_sigma: Optional[np.ndarray] = None
        self._state_backward_mu: Optional[np.ndarray] = None
        self._state_backward_sigma: Optional[np.ndarray] = None
        self._state_likelihood_mu: Optional[np.ndarray] = None
        self._state_likelihood_sigma: Optional[np.ndarray] = None

        # Pre-allocated temp arrays
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
        self._accum_margins: List[np.ndarray] = []
        self._last_refit_day: Optional[int] = None

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        return PlayerRatings(
            ratings=np.full(num_players, self.DISPLAY_OFFSET, dtype=np.float64),
            rd=np.full(num_players, self.config.sigma * self.DISPLAY_SCALE, dtype=np.float64),
            metadata={"system": "margin_ttt", "config": self.config},
        )

    def _ensure_temp_arrays(self) -> None:
        if self._temp_fwd_mu is None or len(self._temp_fwd_mu) < self._num_players:
            self._temp_fwd_mu = np.zeros(self._num_players, dtype=np.float64)
            self._temp_fwd_sigma = np.full(self._num_players, INF_SIGMA, dtype=np.float64)
            self._temp_bwd_mu = np.zeros(self._num_players, dtype=np.float64)
            self._temp_bwd_sigma = np.full(self._num_players, INF_SIGMA, dtype=np.float64)
            self._temp_lik_mu = np.zeros(self._num_players, dtype=np.float64)
            self._temp_lik_sigma = np.full(self._num_players, INF_SIGMA, dtype=np.float64)

    def _prepare_margins(self, scores: np.ndarray, margins: Optional[np.ndarray]) -> np.ndarray:
        """Convert unsigned margins + scores → signed, scaled margin array.

        Args:
            scores: Binary scores (1.0 = P1 won, 0.0 = P2 won).
            margins: Unsigned margins in [0, 1] or None.

        Returns:
            Float64 array of signed, scaled margins.
        """
        n = len(scores)
        if margins is None:
            margins = np.full(n, DEFAULT_MARGIN, dtype=np.float64)
        else:
            margins = np.asarray(margins, dtype=np.float64)

        # Sign: positive when P1 wins, negative when P2 wins
        sign = 2.0 * scores - 1.0  # +1 or -1
        return sign * margins * self.config.margin_scale

    def _build_batch_structure(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        days: np.ndarray,
        margins: Optional[np.ndarray] = None,
    ) -> None:
        """Build batch and sparse appearance structures from game arrays."""
        n_games = len(player1)

        unique_days = np.unique(days)
        self._num_batches = len(unique_days)

        day_to_batch = {int(d): i for i, d in enumerate(unique_days)}

        sort_order = np.argsort(days)
        self._game_p1 = player1[sort_order].astype(np.int64)
        self._game_p2 = player2[sort_order].astype(np.int64)
        self._game_scores = scores[sort_order].astype(np.float64)

        # Prepare and sort margins
        prepared_margins = self._prepare_margins(scores, margins)
        self._game_margins = prepared_margins[sort_order]

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

        self._ensure_temp_arrays()

    def fit(
        self,
        dataset: GameDataset,
        margins: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "MarginTTT":
        """
        Fit MarginTTT on a dataset.

        Args:
            dataset: Game dataset to fit on
            margins: Unsigned per-game margins in [0, 1]. If None, uses
                default margin (0.15) with correct sign from scores.
            end_day: Last day to include (inclusive).
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

        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is None or len(player1) == 0:
            self._fitted = True
            return self

        n_games = len(player1)

        # Create per-game day array
        days = np.empty(n_games, dtype=np.int32)
        n_days = len(day_indices)
        for d in range(n_days):
            start = day_offsets[d]
            end = day_offsets[d + 1]
            days[start:end] = day_indices[d]

        # Build data structures
        self._build_batch_structure(player1, player2, scores, days, margins)

        # Initial forward pass
        initial_forward_pass_margin(
            self._num_batches,
            self._batch_offsets,
            self._batch_times,
            self._game_p1,
            self._game_p2,
            self._game_margins,
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
            self.config.sigma_margin,
            self.config.gamma,
            0,  # start_batch
        )

        # Run convergence
        self._num_iterations = run_convergence_margin(
            self._num_batches,
            self._batch_offsets,
            self._batch_times,
            self._game_p1,
            self._game_p2,
            self._game_margins,
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
            self.config.sigma_margin,
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
            metadata={"system": "margin_ttt", "config": self.config},
        )

        self._num_games_fitted = n_games
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        self._compute_player_last_day()

        # Store data for potential refitting
        if self.config.refit_interval > 0:
            self._accum_p1 = [player1.copy()]
            self._accum_p2 = [player2.copy()]
            self._accum_scores = [scores.copy()]
            self._accum_days = [days.copy()]
            self._accum_margins = [margins.copy() if margins is not None
                                   else np.full(n_games, DEFAULT_MARGIN, dtype=np.float64)]
            self._last_refit_day = self._current_day

        return self

    def _compute_player_last_day(self) -> None:
        if (self._player_last_app is not None and
                self._app_batch is not None and
                self._batch_times is not None):
            self._player_last_day = extract_player_last_day_ttt(
                self._num_players, self._player_last_app,
                self._app_batch, self._batch_times,
            )

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict probability that player1 beats player2 (binary, like TTT)."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if day is not None and self._player_last_day is not None:
            if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
                p1, p2 = int(player1), int(player2)
                return predict_single_at_day(
                    self._ratings.ratings[p1], self._ratings.ratings[p2],
                    self._ratings.rd[p1], self._ratings.rd[p2],
                    self._player_last_day[p1], self._player_last_day[p2],
                    day, self.config.beta, self.config.gamma,
                    self.DISPLAY_SCALE, self.DISPLAY_OFFSET,
                )
            p1 = np.ascontiguousarray(player1, dtype=np.int64)
            p2 = np.ascontiguousarray(player2, dtype=np.int64)
            return predict_proba_batch_at_day(
                p1, p2, self._ratings.ratings, self._ratings.rd,
                self._player_last_day, day,
                self.config.beta, self.config.gamma,
                self.DISPLAY_SCALE, self.DISPLAY_OFFSET,
            )

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
        """No-op. MarginTTT is a batch algorithm."""
        pass

    def update(self, batch: GameBatch) -> "MarginTTT":
        """Update with default margins."""
        return self.update_margin(batch)

    def update_margin(
        self,
        batch: GameBatch,
        margins: Optional[np.ndarray] = None,
    ) -> "MarginTTT":
        """
        Incrementally update with new games and optional margins.

        Args:
            batch: Game batch.
            margins: Unsigned margins in [0, 1] for this batch.
                If None, uses default margin (0.15).
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before updating. Call fit() first.")

        if self.config.refit_interval <= 0:
            return self

        n_games = len(batch)
        days_array = np.full(n_games, batch.day, dtype=np.int32)
        self._accum_p1.append(batch.player1.copy())
        self._accum_p2.append(batch.player2.copy())
        self._accum_scores.append(batch.scores.copy())
        self._accum_days.append(days_array)

        if margins is not None:
            self._accum_margins.append(np.asarray(margins, dtype=np.float64).copy())
        else:
            self._accum_margins.append(np.full(n_games, DEFAULT_MARGIN, dtype=np.float64))

        if self._last_refit_day is None:
            self._last_refit_day = batch.day

        if batch.day - self._last_refit_day >= self.config.refit_interval:
            self._refit_from_accumulated()
            self._last_refit_day = batch.day

        self._current_day = batch.day

        if self._player_last_day is not None:
            players = np.unique(np.concatenate([batch.player1, batch.player2]))
            self._player_last_day[players] = batch.day

        return self

    def _refit_from_accumulated(self) -> None:
        """Refit on all accumulated data with warm-start."""
        if not self._accum_p1:
            return

        player1 = np.concatenate(self._accum_p1)
        player2 = np.concatenate(self._accum_p2)
        scores = np.concatenate(self._accum_scores)
        days = np.concatenate(self._accum_days)
        margins = np.concatenate(self._accum_margins)

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

        # Rebuild sparse structures
        self._build_batch_structure(player1, player2, scores, days, margins)

        # Restore old state
        if old_fwd_mu is not None and old_num_appearances > 0:
            n = old_num_appearances
            self._state_forward_mu[:n] = old_fwd_mu[:n]
            self._state_forward_sigma[:n] = old_fwd_sigma[:n]
            self._state_backward_mu[:n] = old_bwd_mu[:n]
            self._state_backward_sigma[:n] = old_bwd_sigma[:n]
            self._state_likelihood_mu[:n] = old_lik_mu[:n]
            self._state_likelihood_sigma[:n] = old_lik_sigma[:n]

        # Forward pass only for new batches
        initial_forward_pass_margin(
            self._num_batches,
            self._batch_offsets,
            self._batch_times,
            self._game_p1,
            self._game_p2,
            self._game_margins,
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
            self.config.sigma_margin,
            self.config.gamma,
            old_num_batches,
        )

        # Run convergence
        self._num_iterations = run_convergence_margin(
            self._num_batches,
            self._batch_offsets,
            self._batch_times,
            self._game_p1,
            self._game_p2,
            self._game_margins,
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
            self.config.sigma_margin,
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
            metadata={"system": "margin_ttt", "config": self.config},
        )
        self._compute_player_last_day()
        self._num_games_fitted = len(player1)

    def snapshot(self) -> dict:
        if not self._fitted or self._ratings is None:
            raise ValueError("Model must be fitted before snapshotting.")
        state = {
            "ratings": self._ratings.clone(),
            "num_players": self._num_players,
            "current_day": self._current_day,
            "num_batches": self._num_batches,
            "num_appearances": self._num_appearances,
            "num_games_fitted": self._num_games_fitted,
            "num_iterations": self._num_iterations,
            "last_refit_day": self._last_refit_day,
        }
        for attr in (
            "_state_forward_mu", "_state_forward_sigma",
            "_state_backward_mu", "_state_backward_sigma",
            "_state_likelihood_mu", "_state_likelihood_sigma",
            "_app_offsets", "_app_player", "_app_prev", "_app_next",
            "_app_batch", "_player_last_app", "_player_last_day",
            "_batch_offsets", "_batch_times",
            "_game_p1", "_game_p2", "_game_scores", "_game_margins",
        ):
            val = getattr(self, attr)
            state[attr] = val.copy() if val is not None else None
        state["_accum_p1"] = [a.copy() for a in self._accum_p1]
        state["_accum_p2"] = [a.copy() for a in self._accum_p2]
        state["_accum_scores"] = [a.copy() for a in self._accum_scores]
        state["_accum_days"] = [a.copy() for a in self._accum_days]
        state["_accum_margins"] = [a.copy() for a in self._accum_margins]
        return state

    def restore(self, state: dict) -> None:
        self._num_players = state["num_players"]
        self._current_day = state["current_day"]
        self._num_batches = state["num_batches"]
        self._num_appearances = state["num_appearances"]
        self._num_games_fitted = state["num_games_fitted"]
        self._num_iterations = state["num_iterations"]
        self._last_refit_day = state["last_refit_day"]
        self._fitted = True
        self._ratings = state["ratings"].clone()
        for attr in (
            "_state_forward_mu", "_state_forward_sigma",
            "_state_backward_mu", "_state_backward_sigma",
            "_state_likelihood_mu", "_state_likelihood_sigma",
            "_app_offsets", "_app_player", "_app_prev", "_app_next",
            "_app_batch", "_player_last_app", "_player_last_day",
            "_batch_offsets", "_batch_times",
            "_game_p1", "_game_p2", "_game_scores", "_game_margins",
        ):
            val = state[attr]
            setattr(self, attr, val.copy() if val is not None else None)
        self._accum_p1 = [a.copy() for a in state["_accum_p1"]]
        self._accum_p2 = [a.copy() for a in state["_accum_p2"]]
        self._accum_scores = [a.copy() for a in state["_accum_scores"]]
        self._accum_days = [a.copy() for a in state["_accum_days"]]
        self._accum_margins = [a.copy() for a in state["_accum_margins"]]

    def save_state(self, path: str) -> None:
        if not self._fitted or self._state_forward_mu is None:
            raise ValueError("Model must be fitted before saving state.")

        arrays = {
            "ratings": self._ratings.ratings,
            "rd": self._ratings.rd,
            "state_forward_mu": self._state_forward_mu,
            "state_forward_sigma": self._state_forward_sigma,
            "state_backward_mu": self._state_backward_mu,
            "state_backward_sigma": self._state_backward_sigma,
            "state_likelihood_mu": self._state_likelihood_mu,
            "state_likelihood_sigma": self._state_likelihood_sigma,
            "app_offsets": self._app_offsets,
            "app_player": self._app_player,
            "app_prev": self._app_prev,
            "app_next": self._app_next,
            "app_batch": self._app_batch,
            "player_last_app": self._player_last_app,
            "batch_offsets": self._batch_offsets,
            "batch_times": self._batch_times,
            "game_p1": self._game_p1,
            "game_p2": self._game_p2,
            "game_scores": self._game_scores,
            "game_margins": self._game_margins,
        }

        metadata = {
            "system_class": "MarginTTT",
            "num_players": self._num_players,
            "current_day": self._current_day,
            "num_batches": self._num_batches,
            "num_appearances": self._num_appearances,
            "num_games_fitted": self._num_games_fitted,
            "num_iterations": self._num_iterations,
            "config": {
                "mu": self.config.mu,
                "sigma": self.config.sigma,
                "beta": self.config.beta,
                "gamma": self.config.gamma,
                "margin_scale": self.config.margin_scale,
                "sigma_margin": self.config.sigma_margin,
            },
        }
        save_checkpoint(path, arrays, metadata)

    def load_state(self, path: str) -> None:
        arrays, metadata = load_checkpoint(path)

        self._num_players = metadata["num_players"]
        self._current_day = metadata["current_day"]
        self._num_batches = metadata.get("num_batches", 0)
        self._num_appearances = metadata.get("num_appearances", 0)
        self._num_games_fitted = metadata.get("num_games_fitted", 0)
        self._num_iterations = metadata.get("num_iterations", 0)
        self._fitted = True

        self._ratings = PlayerRatings(
            ratings=arrays["ratings"],
            rd=arrays["rd"],
            metadata={"system": "margin_ttt", "config": self.config},
        )

        self._state_forward_mu = arrays["state_forward_mu"]
        self._state_forward_sigma = arrays["state_forward_sigma"]
        self._state_backward_mu = arrays["state_backward_mu"]
        self._state_backward_sigma = arrays["state_backward_sigma"]
        self._state_likelihood_mu = arrays["state_likelihood_mu"]
        self._state_likelihood_sigma = arrays["state_likelihood_sigma"]
        self._app_offsets = arrays["app_offsets"]
        self._app_player = arrays["app_player"]
        self._app_prev = arrays["app_prev"]
        self._app_next = arrays["app_next"]
        self._app_batch = arrays["app_batch"]
        self._player_last_app = arrays["player_last_app"]
        self._batch_offsets = arrays["batch_offsets"]
        self._batch_times = arrays["batch_times"]
        self._game_p1 = arrays["game_p1"]
        self._game_p2 = arrays["game_p2"]
        self._game_scores = arrays["game_scores"]
        self._game_margins = arrays["game_margins"]

        # Reconstruct per-game days from batch structure
        days = np.empty(len(self._game_p1), dtype=np.int32)
        for b in range(self._num_batches):
            start = self._batch_offsets[b]
            end = self._batch_offsets[b + 1] if b + 1 < self._num_batches else len(self._game_p1)
            days[start:end] = self._batch_times[b]

        # Initialize accumulation lists
        self._accum_p1 = [self._game_p1.copy()]
        self._accum_p2 = [self._game_p2.copy()]
        self._accum_scores = [self._game_scores.copy()]
        self._accum_days = [days]
        self._accum_margins = [np.full(len(self._game_p1), DEFAULT_MARGIN, dtype=np.float64)]
        self._last_refit_day = self._current_day
        self._compute_player_last_day()

    def reset(self) -> "MarginTTT":
        self._num_batches = 0
        self._batch_offsets = None
        self._batch_times = None
        self._game_p1 = None
        self._game_p2 = None
        self._game_scores = None
        self._game_margins = None
        self._num_appearances = 0
        self._app_offsets = None
        self._app_player = None
        self._app_prev = None
        self._app_next = None
        self._app_batch = None
        self._player_last_app = None
        self._player_last_day = None
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
        self._accum_margins = []
        self._last_refit_day = None
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        apps = self._num_appearances or 0
        return (
            f"MarginTTT(sigma={self.config.sigma:.2f}, "
            f"beta={self.config.beta:.2f}, gamma={self.config.gamma:.3f}, "
            f"margin_scale={self.config.margin_scale:.1f}, "
            f"sigma_margin={self.config.sigma_margin:.1f}, "
            f"players={players}, appearances={apps}, {status})"
        )
