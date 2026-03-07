"""
Matérn 3/2 TrueSkill Through Time — Numba implementation with sparse state.

Replaces the Wiener process (Matérn 1/2) dynamics of standard TTT with
Matérn 3/2 dynamics. The state is 2D [skill, velocity], giving
autocorrelated skill trajectories that can capture form streaks.

Parameters:
    sigma: Marginal std dev of the GP (stationary prior skill uncertainty)
    beta: Performance noise (same as TTT)
    lengthscale: Temporal correlation length in days. Smaller → faster-changing
        form (approaches Wiener). Larger → slow, smooth trajectories.
    max_iterations: Max EP iterations for initial fit
    refit_max_iterations: Max iterations for periodic refits
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import math
import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...data.checkpoint import compute_fingerprint, save_checkpoint, load_checkpoint
from ...results.fitted_ratings import FittedTTTRatings
from ..trueskill_through_time._numba_core import (
    build_appearance_structure,
    predict_proba_batch,
    predict_single,
    extract_player_last_day_ttt,
    INF_SIGMA,
)
from ._numba_core import (
    INF_COV,
    initial_forward_pass_matern,
    run_convergence_matern,
    extract_final_ratings_matern,
    extract_player_posteriors_matern,
    predict_proba_batch_at_day_matern,
    predict_single_at_day_matern,
)


@dataclass
class MaternTTTConfig:
    """Configuration for Matérn 3/2 TrueSkill Through Time."""

    mu: float = 0.0             # Prior mean (internal scale)
    sigma: float = 6.0          # Marginal std dev (stationary)
    beta: float = 1.0           # Performance noise
    lengthscale: float = 365.0  # Temporal correlation length (days)
    max_iterations: int = 30
    refit_max_iterations: int = 2
    convergence_threshold: float = 1e-6
    refit_interval: int = 1


class MaternTTT(RatingSystem):
    """
    Matérn 3/2 TrueSkill Through Time rating system.

    Uses a 2D state [skill, velocity/λ] with Matérn 3/2 dynamics instead of
    the Wiener process of standard TTT. This gives autocorrelated skill
    trajectories — a player on an upswing is predicted to continue rising.
    The velocity component is rescaled by 1/λ for numerical stability.

    Parameters:
        mu: Prior mean skill (default: 0.0, displayed as 1500)
        sigma: Marginal std dev of the GP (default: 6.0)
        beta: Performance variability within games (default: 1.0)
        lengthscale: Temporal correlation length in days (default: 365)
        max_iterations: Max belief propagation iterations (default: 30)
    """

    system_type = RatingSystemType.BATCH

    DISPLAY_OFFSET = 1500.0
    DISPLAY_SCALE = 400.0 / 6.0

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 6.0,
        beta: float = 1.0,
        lengthscale: float = 365.0,
        max_iterations: int = 30,
        refit_max_iterations: int = 2,
        convergence_threshold: float = 1e-6,
        refit_interval: int = 1,
        num_players: Optional[int] = None,
    ):
        self.config = MaternTTTConfig(
            mu=mu,
            sigma=sigma,
            beta=beta,
            lengthscale=lengthscale,
            max_iterations=max_iterations,
            refit_max_iterations=refit_max_iterations,
            convergence_threshold=convergence_threshold,
            refit_interval=refit_interval,
        )

        # Derived constants
        self._lambda = math.sqrt(3.0) / lengthscale
        self._sigma_sq = sigma * sigma

        # Stationary covariance P∞_z = σ² × [[1, 0], [0, 1]]
        # (rescaled velocity state: component 1 = velocity/λ)
        self._prior_c00 = self._sigma_sq
        self._prior_c01 = 0.0
        self._prior_c11 = self._sigma_sq

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
        self._player_last_day: Optional[np.ndarray] = None

        # 2D forward state (per appearance): 5 arrays
        self._fwd_mu0: Optional[np.ndarray] = None
        self._fwd_mu1: Optional[np.ndarray] = None
        self._fwd_c00: Optional[np.ndarray] = None
        self._fwd_c01: Optional[np.ndarray] = None
        self._fwd_c11: Optional[np.ndarray] = None

        # 2D backward state (per appearance): 5 arrays
        self._bwd_mu0: Optional[np.ndarray] = None
        self._bwd_mu1: Optional[np.ndarray] = None
        self._bwd_c00: Optional[np.ndarray] = None
        self._bwd_c01: Optional[np.ndarray] = None
        self._bwd_c11: Optional[np.ndarray] = None

        # Scalar likelihood (per appearance): 2 arrays
        self._lik_mu: Optional[np.ndarray] = None
        self._lik_sigma: Optional[np.ndarray] = None

        # Temp arrays (per player): 12 arrays
        self._t_fwd_mu0: Optional[np.ndarray] = None
        self._t_fwd_mu1: Optional[np.ndarray] = None
        self._t_fwd_c00: Optional[np.ndarray] = None
        self._t_fwd_c01: Optional[np.ndarray] = None
        self._t_fwd_c11: Optional[np.ndarray] = None
        self._t_bwd_mu0: Optional[np.ndarray] = None
        self._t_bwd_mu1: Optional[np.ndarray] = None
        self._t_bwd_c00: Optional[np.ndarray] = None
        self._t_bwd_c01: Optional[np.ndarray] = None
        self._t_bwd_c11: Optional[np.ndarray] = None
        self._t_lik_mu: Optional[np.ndarray] = None
        self._t_lik_sigma: Optional[np.ndarray] = None

        # Per-player 2D posterior (for time-aware prediction)
        self._p_mu0: Optional[np.ndarray] = None
        self._p_mu1: Optional[np.ndarray] = None
        self._p_c00: Optional[np.ndarray] = None
        self._p_c01: Optional[np.ndarray] = None
        self._p_c11: Optional[np.ndarray] = None

        # Metadata
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._player_names: Optional[Dict[int, str]] = None

        # Stored batched arrays for checkpointing
        self._stored_player1: Optional[np.ndarray] = None
        self._stored_player2: Optional[np.ndarray] = None
        self._stored_scores: Optional[np.ndarray] = None

        # Accumulated data for periodic refitting
        self._accum_p1: List[np.ndarray] = []
        self._accum_p2: List[np.ndarray] = []
        self._accum_scores: List[np.ndarray] = []
        self._accum_days: List[np.ndarray] = []
        self._last_refit_day: Optional[int] = None

        # Checkpoint warm-start
        self._checkpoint_loaded = False
        self._checkpoint_num_batches = 0
        self._checkpoint_num_appearances = 0

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        return PlayerRatings(
            ratings=np.full(num_players, self.DISPLAY_OFFSET, dtype=np.float64),
            rd=np.full(num_players, self.config.sigma * self.DISPLAY_SCALE, dtype=np.float64),
            metadata={"system": "matern_ttt", "config": self.config},
        )

    def _ensure_temp_arrays(self) -> None:
        if self._t_fwd_mu0 is None or len(self._t_fwd_mu0) < self._num_players:
            n = self._num_players
            self._t_fwd_mu0 = np.zeros(n, dtype=np.float64)
            self._t_fwd_mu1 = np.zeros(n, dtype=np.float64)
            self._t_fwd_c00 = np.full(n, INF_COV, dtype=np.float64)
            self._t_fwd_c01 = np.zeros(n, dtype=np.float64)
            self._t_fwd_c11 = np.full(n, INF_COV, dtype=np.float64)
            self._t_bwd_mu0 = np.zeros(n, dtype=np.float64)
            self._t_bwd_mu1 = np.zeros(n, dtype=np.float64)
            self._t_bwd_c00 = np.full(n, INF_COV, dtype=np.float64)
            self._t_bwd_c01 = np.zeros(n, dtype=np.float64)
            self._t_bwd_c11 = np.full(n, INF_COV, dtype=np.float64)
            self._t_lik_mu = np.zeros(n, dtype=np.float64)
            self._t_lik_sigma = np.full(n, INF_SIGMA, dtype=np.float64)

    def _build_batch_structure(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        days: np.ndarray,
    ) -> None:
        n_games = len(player1)

        unique_days = np.unique(days)
        self._num_batches = len(unique_days)

        day_to_batch = {int(d): i for i, d in enumerate(unique_days)}

        sort_order = np.argsort(days)
        self._game_p1 = player1[sort_order].astype(np.int64)
        self._game_p2 = player2[sort_order].astype(np.int64)
        self._game_scores = scores[sort_order].astype(np.float64)

        sorted_days = days[sort_order]

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

        # Allocate 2D state arrays (12 total per appearance)
        na = self._num_appearances
        self._fwd_mu0 = np.zeros(na, dtype=np.float64)
        self._fwd_mu1 = np.zeros(na, dtype=np.float64)
        self._fwd_c00 = np.full(na, INF_COV, dtype=np.float64)
        self._fwd_c01 = np.zeros(na, dtype=np.float64)
        self._fwd_c11 = np.full(na, INF_COV, dtype=np.float64)
        self._bwd_mu0 = np.zeros(na, dtype=np.float64)
        self._bwd_mu1 = np.zeros(na, dtype=np.float64)
        self._bwd_c00 = np.full(na, INF_COV, dtype=np.float64)
        self._bwd_c01 = np.zeros(na, dtype=np.float64)
        self._bwd_c11 = np.full(na, INF_COV, dtype=np.float64)
        self._lik_mu = np.zeros(na, dtype=np.float64)
        self._lik_sigma = np.full(na, INF_SIGMA, dtype=np.float64)

        self._ensure_temp_arrays()

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "MaternTTT":
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

        self._stored_player1 = player1.copy()
        self._stored_player2 = player2.copy()
        self._stored_scores = scores.copy()

        n_games = len(player1)
        days = np.empty(n_games, dtype=np.int32)
        n_days = len(day_indices)
        for d in range(n_days):
            start = day_offsets[d]
            end = day_offsets[d + 1]
            days[start:end] = day_indices[d]

        # Save old state for warm-start
        old_num_batches = 0
        old_num_appearances = 0
        old_state = None
        if self._checkpoint_loaded:
            old_num_batches = self._checkpoint_num_batches
            old_num_appearances = self._checkpoint_num_appearances
            old_state = self._save_state_arrays()
            self._checkpoint_loaded = False

        self._build_batch_structure(player1, player2, scores, days)

        # Restore old state for warm-start
        if old_state is not None and old_num_appearances > 0:
            self._restore_state_arrays(old_state, old_num_appearances)

        start_batch = old_num_batches if old_state is not None else 0

        # Initial forward pass
        initial_forward_pass_matern(
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
            self._fwd_mu0, self._fwd_mu1,
            self._fwd_c00, self._fwd_c01, self._fwd_c11,
            self._bwd_mu0, self._bwd_mu1,
            self._bwd_c00, self._bwd_c01, self._bwd_c11,
            self._lik_mu, self._lik_sigma,
            self._t_fwd_mu0, self._t_fwd_mu1,
            self._t_fwd_c00, self._t_fwd_c01, self._t_fwd_c11,
            self._t_bwd_mu0, self._t_bwd_mu1,
            self._t_bwd_c00, self._t_bwd_c01, self._t_bwd_c11,
            self._t_lik_mu, self._t_lik_sigma,
            self.config.mu,
            self.config.sigma,
            self._prior_c00, self._prior_c01, self._prior_c11,
            self.config.beta,
            self._lambda,
            self._sigma_sq,
            start_batch,
        )

        # Run convergence
        self._num_iterations = run_convergence_matern(
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
            self._fwd_mu0, self._fwd_mu1,
            self._fwd_c00, self._fwd_c01, self._fwd_c11,
            self._bwd_mu0, self._bwd_mu1,
            self._bwd_c00, self._bwd_c01, self._bwd_c11,
            self._lik_mu, self._lik_sigma,
            self._t_fwd_mu0, self._t_fwd_mu1,
            self._t_fwd_c00, self._t_fwd_c01, self._t_fwd_c11,
            self._t_bwd_mu0, self._t_bwd_mu1,
            self._t_bwd_c00, self._t_bwd_c01, self._t_bwd_c11,
            self._t_lik_mu, self._t_lik_sigma,
            self.config.mu,
            self.config.sigma,
            self._prior_c00, self._prior_c01, self._prior_c11,
            self.config.beta,
            self._lambda,
            self._sigma_sq,
            self.config.max_iterations,
            self.config.convergence_threshold,
        )

        # Extract ratings
        self._extract_ratings()

        self._num_games_fitted = n_games
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        self._compute_player_last_day()
        self._compute_player_posteriors()

        if self.config.refit_interval > 0:
            self._accum_p1 = [player1.copy()]
            self._accum_p2 = [player2.copy()]
            self._accum_scores = [scores.copy()]
            self._accum_days = [days.copy()]
            self._last_refit_day = self._current_day

        return self

    def _extract_ratings(self) -> None:
        ratings = np.empty(self._num_players, dtype=np.float64)
        rd = np.empty(self._num_players, dtype=np.float64)

        extract_final_ratings_matern(
            self._num_players,
            self._player_last_app,
            self._fwd_mu0, self._fwd_mu1,
            self._fwd_c00, self._fwd_c01, self._fwd_c11,
            self._bwd_mu0, self._bwd_mu1,
            self._bwd_c00, self._bwd_c01, self._bwd_c11,
            self._lik_mu, self._lik_sigma,
            ratings, rd,
            self.config.mu,
            self.config.sigma,
            self.DISPLAY_SCALE,
            self.DISPLAY_OFFSET,
        )

        self._ratings = PlayerRatings(
            ratings=ratings,
            rd=rd,
            metadata={"system": "matern_ttt", "config": self.config},
        )

    def _compute_player_last_day(self) -> None:
        if (self._player_last_app is not None and
                self._app_batch is not None and
                self._batch_times is not None):
            self._player_last_day = extract_player_last_day_ttt(
                self._num_players, self._player_last_app,
                self._app_batch, self._batch_times,
            )

    def _compute_player_posteriors(self) -> None:
        """Compute per-player 2D posterior for time-aware prediction."""
        n = self._num_players
        self._p_mu0 = np.zeros(n, dtype=np.float64)
        self._p_mu1 = np.zeros(n, dtype=np.float64)
        self._p_c00 = np.full(n, self._prior_c00, dtype=np.float64)
        self._p_c01 = np.full(n, self._prior_c01, dtype=np.float64)
        self._p_c11 = np.full(n, self._prior_c11, dtype=np.float64)

        extract_player_posteriors_matern(
            self._num_players,
            self._player_last_app,
            self._fwd_mu0, self._fwd_mu1,
            self._fwd_c00, self._fwd_c01, self._fwd_c11,
            self._bwd_mu0, self._bwd_mu1,
            self._bwd_c00, self._bwd_c01, self._bwd_c11,
            self._lik_mu, self._lik_sigma,
            self._p_mu0, self._p_mu1,
            self._p_c00, self._p_c01, self._p_c11,
            self.config.mu,
            self._prior_c00, self._prior_c01, self._prior_c11,
        )

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Time-aware prediction using 2D state propagation
        if day is not None and self._player_last_day is not None and self._p_mu0 is not None:
            if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
                p1, p2 = int(player1), int(player2)
                return predict_single_at_day_matern(
                    self._p_mu0[p1], self._p_mu1[p1],
                    self._p_c00[p1], self._p_c01[p1], self._p_c11[p1],
                    self._p_mu0[p2], self._p_mu1[p2],
                    self._p_c00[p2], self._p_c01[p2], self._p_c11[p2],
                    self._player_last_day[p1], self._player_last_day[p2],
                    day, self.config.beta,
                    self._lambda, self._sigma_sq,
                )
            p1 = np.ascontiguousarray(player1, dtype=np.int64)
            p2 = np.ascontiguousarray(player2, dtype=np.int64)
            return predict_proba_batch_at_day_matern(
                p1, p2,
                self._p_mu0, self._p_mu1,
                self._p_c00, self._p_c01, self._p_c11,
                self._player_last_day, day,
                self.config.beta,
                self._lambda, self._sigma_sq,
                self.DISPLAY_SCALE, self.DISPLAY_OFFSET,
            )

        # Static prediction using extracted ratings (same as TTT)
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

        # Compute equivalent gamma for FittedTTTRatings compatibility
        # For Matérn 3/2, the short-time variance growth is ~σ²·(λ·dt)²
        # which doesn't map cleanly to γ²·dt, but we provide a rough
        # equivalent for display purposes.
        equiv_gamma = self.config.sigma * self._lambda

        return FittedTTTRatings(
            ratings=self._ratings.ratings.copy(),
            rd=self._ratings.rd.copy(),
            sigma=self.config.sigma,
            beta=self.config.beta,
            gamma=equiv_gamma,
            display_scale=self.DISPLAY_SCALE,
            display_offset=self.DISPLAY_OFFSET,
            num_games_fitted=self._num_games_fitted,
            num_iterations=self._num_iterations,
            last_day=self._current_day,
            player_names=self._player_names,
            rating_history={},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        pass  # Batch system — uses accumulate+refit pattern

    def update(self, batch: GameBatch) -> "MaternTTT":
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

    def _save_state_arrays(self) -> dict:
        """Save state arrays for warm-start."""
        return {
            'fwd_mu0': self._fwd_mu0.copy(), 'fwd_mu1': self._fwd_mu1.copy(),
            'fwd_c00': self._fwd_c00.copy(), 'fwd_c01': self._fwd_c01.copy(),
            'fwd_c11': self._fwd_c11.copy(),
            'bwd_mu0': self._bwd_mu0.copy(), 'bwd_mu1': self._bwd_mu1.copy(),
            'bwd_c00': self._bwd_c00.copy(), 'bwd_c01': self._bwd_c01.copy(),
            'bwd_c11': self._bwd_c11.copy(),
            'lik_mu': self._lik_mu.copy(), 'lik_sigma': self._lik_sigma.copy(),
        }

    def _restore_state_arrays(self, state: dict, n: int) -> None:
        """Restore state arrays from warm-start snapshot."""
        for key in ('fwd_mu0', 'fwd_mu1', 'fwd_c00', 'fwd_c01', 'fwd_c11',
                     'bwd_mu0', 'bwd_mu1', 'bwd_c00', 'bwd_c01', 'bwd_c11',
                     'lik_mu', 'lik_sigma'):
            attr = f'_{key}'
            getattr(self, attr)[:n] = state[key][:n]

    def _refit_from_accumulated(self) -> None:
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

        old_num_batches = self._num_batches
        old_num_appearances = self._num_appearances
        old_state = self._save_state_arrays()

        self._build_batch_structure(player1, player2, scores, days)

        if old_state is not None and old_num_appearances > 0:
            self._restore_state_arrays(old_state, old_num_appearances)

        initial_forward_pass_matern(
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
            self._fwd_mu0, self._fwd_mu1,
            self._fwd_c00, self._fwd_c01, self._fwd_c11,
            self._bwd_mu0, self._bwd_mu1,
            self._bwd_c00, self._bwd_c01, self._bwd_c11,
            self._lik_mu, self._lik_sigma,
            self._t_fwd_mu0, self._t_fwd_mu1,
            self._t_fwd_c00, self._t_fwd_c01, self._t_fwd_c11,
            self._t_bwd_mu0, self._t_bwd_mu1,
            self._t_bwd_c00, self._t_bwd_c01, self._t_bwd_c11,
            self._t_lik_mu, self._t_lik_sigma,
            self.config.mu,
            self.config.sigma,
            self._prior_c00, self._prior_c01, self._prior_c11,
            self.config.beta,
            self._lambda,
            self._sigma_sq,
            old_num_batches,
        )

        self._num_iterations = run_convergence_matern(
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
            self._fwd_mu0, self._fwd_mu1,
            self._fwd_c00, self._fwd_c01, self._fwd_c11,
            self._bwd_mu0, self._bwd_mu1,
            self._bwd_c00, self._bwd_c01, self._bwd_c11,
            self._lik_mu, self._lik_sigma,
            self._t_fwd_mu0, self._t_fwd_mu1,
            self._t_fwd_c00, self._t_fwd_c01, self._t_fwd_c11,
            self._t_bwd_mu0, self._t_bwd_mu1,
            self._t_bwd_c00, self._t_bwd_c01, self._t_bwd_c11,
            self._t_lik_mu, self._t_lik_sigma,
            self.config.mu,
            self.config.sigma,
            self._prior_c00, self._prior_c01, self._prior_c11,
            self.config.beta,
            self._lambda,
            self._sigma_sq,
            self.config.refit_max_iterations,
            self.config.convergence_threshold,
        )

        self._extract_ratings()
        self._compute_player_last_day()
        self._compute_player_posteriors()
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
        # State arrays (12 per appearance)
        for attr in (
            "_fwd_mu0", "_fwd_mu1", "_fwd_c00", "_fwd_c01", "_fwd_c11",
            "_bwd_mu0", "_bwd_mu1", "_bwd_c00", "_bwd_c01", "_bwd_c11",
            "_lik_mu", "_lik_sigma",
            # Structure arrays
            "_app_offsets", "_app_player", "_app_prev", "_app_next",
            "_app_batch", "_player_last_app", "_player_last_day",
            "_batch_offsets", "_batch_times",
            "_game_p1", "_game_p2", "_game_scores",
            "_stored_player1", "_stored_player2", "_stored_scores",
            # Per-player posteriors
            "_p_mu0", "_p_mu1", "_p_c00", "_p_c01", "_p_c11",
        ):
            val = getattr(self, attr)
            state[attr] = val.copy() if val is not None else None
        state["_accum_p1"] = [a.copy() for a in self._accum_p1]
        state["_accum_p2"] = [a.copy() for a in self._accum_p2]
        state["_accum_scores"] = [a.copy() for a in self._accum_scores]
        state["_accum_days"] = [a.copy() for a in self._accum_days]
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
            "_fwd_mu0", "_fwd_mu1", "_fwd_c00", "_fwd_c01", "_fwd_c11",
            "_bwd_mu0", "_bwd_mu1", "_bwd_c00", "_bwd_c01", "_bwd_c11",
            "_lik_mu", "_lik_sigma",
            "_app_offsets", "_app_player", "_app_prev", "_app_next",
            "_app_batch", "_player_last_app", "_player_last_day",
            "_batch_offsets", "_batch_times",
            "_game_p1", "_game_p2", "_game_scores",
            "_stored_player1", "_stored_player2", "_stored_scores",
            "_p_mu0", "_p_mu1", "_p_c00", "_p_c01", "_p_c11",
        ):
            val = state[attr]
            setattr(self, attr, val.copy() if val is not None else None)
        self._accum_p1 = [a.copy() for a in state["_accum_p1"]]
        self._accum_p2 = [a.copy() for a in state["_accum_p2"]]
        self._accum_scores = [a.copy() for a in state["_accum_scores"]]
        self._accum_days = [a.copy() for a in state["_accum_days"]]

    def save_state(self, path: str) -> None:
        if not self._fitted or self._fwd_mu0 is None:
            raise ValueError("Model must be fitted before saving state.")

        arrays = {
            "ratings": self._ratings.ratings,
            "rd": self._ratings.rd,
            "fwd_mu0": self._fwd_mu0, "fwd_mu1": self._fwd_mu1,
            "fwd_c00": self._fwd_c00, "fwd_c01": self._fwd_c01, "fwd_c11": self._fwd_c11,
            "bwd_mu0": self._bwd_mu0, "bwd_mu1": self._bwd_mu1,
            "bwd_c00": self._bwd_c00, "bwd_c01": self._bwd_c01, "bwd_c11": self._bwd_c11,
            "lik_mu": self._lik_mu, "lik_sigma": self._lik_sigma,
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
        }

        metadata = {
            "system_class": "MaternTTT",
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
                "lengthscale": self.config.lengthscale,
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
            metadata={"system": "matern_ttt", "config": self.config},
        )

        # State arrays
        self._fwd_mu0 = arrays["fwd_mu0"]
        self._fwd_mu1 = arrays["fwd_mu1"]
        self._fwd_c00 = arrays["fwd_c00"]
        self._fwd_c01 = arrays["fwd_c01"]
        self._fwd_c11 = arrays["fwd_c11"]
        self._bwd_mu0 = arrays["bwd_mu0"]
        self._bwd_mu1 = arrays["bwd_mu1"]
        self._bwd_c00 = arrays["bwd_c00"]
        self._bwd_c01 = arrays["bwd_c01"]
        self._bwd_c11 = arrays["bwd_c11"]
        self._lik_mu = arrays["lik_mu"]
        self._lik_sigma = arrays["lik_sigma"]

        # Structure arrays
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

        self._stored_player1 = arrays["game_p1"]
        self._stored_player2 = arrays["game_p2"]
        self._stored_scores = arrays["game_scores"]

        # Reconstruct per-game days
        days = np.empty(len(self._game_p1), dtype=np.int32)
        for b in range(self._num_batches):
            start = self._batch_offsets[b]
            end = self._batch_offsets[b + 1] if b + 1 < self._num_batches else len(self._game_p1)
            days[start:end] = self._batch_times[b]

        self._accum_p1 = [self._game_p1.copy()]
        self._accum_p2 = [self._game_p2.copy()]
        self._accum_scores = [self._game_scores.copy()]
        self._accum_days = [days]
        self._last_refit_day = self._current_day
        self._compute_player_last_day()
        self._compute_player_posteriors()

    def save_checkpoint(self, path: str, player_to_idx: Optional[Dict[int, int]] = None) -> None:
        if not self._fitted or self._fwd_mu0 is None:
            raise ValueError("Model must be fitted before saving checkpoint.")

        arrays = {
            "fwd_mu0": self._fwd_mu0, "fwd_mu1": self._fwd_mu1,
            "fwd_c00": self._fwd_c00, "fwd_c01": self._fwd_c01, "fwd_c11": self._fwd_c11,
            "bwd_mu0": self._bwd_mu0, "bwd_mu1": self._bwd_mu1,
            "bwd_c00": self._bwd_c00, "bwd_c01": self._bwd_c01, "bwd_c11": self._bwd_c11,
            "lik_mu": self._lik_mu, "lik_sigma": self._lik_sigma,
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
        }

        fingerprint = compute_fingerprint(
            self._stored_player1,
            self._stored_player2,
            self._stored_scores,
            self._num_players,
            int(self._batch_times[-1]) if self._num_batches > 0 else 0,
        )

        metadata = {
            "system": "matern_ttt",
            "config": {
                "mu": self.config.mu,
                "sigma": self.config.sigma,
                "beta": self.config.beta,
                "lengthscale": self.config.lengthscale,
            },
            "fingerprint": fingerprint,
            "num_batches": self._num_batches,
            "num_appearances": self._num_appearances,
            "num_players": self._num_players,
            "num_iterations": self._num_iterations,
        }
        if player_to_idx is not None:
            metadata["player_to_idx"] = {str(k): v for k, v in player_to_idx.items()}

        save_checkpoint(path, arrays, metadata)

    def load_checkpoint(self, path: str) -> dict:
        arrays, metadata = load_checkpoint(path)

        self._fwd_mu0 = arrays["fwd_mu0"]
        self._fwd_mu1 = arrays["fwd_mu1"]
        self._fwd_c00 = arrays["fwd_c00"]
        self._fwd_c01 = arrays["fwd_c01"]
        self._fwd_c11 = arrays["fwd_c11"]
        self._bwd_mu0 = arrays["bwd_mu0"]
        self._bwd_mu1 = arrays["bwd_mu1"]
        self._bwd_c00 = arrays["bwd_c00"]
        self._bwd_c01 = arrays["bwd_c01"]
        self._bwd_c11 = arrays["bwd_c11"]
        self._lik_mu = arrays["lik_mu"]
        self._lik_sigma = arrays["lik_sigma"]

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

        self._checkpoint_loaded = True
        self._checkpoint_num_batches = metadata["num_batches"]
        self._checkpoint_num_appearances = metadata["num_appearances"]
        self._num_players = metadata.get("num_players", self._num_players)

        return metadata

    def reset(self) -> "MaternTTT":
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
        self._fwd_mu0 = None
        self._fwd_mu1 = None
        self._fwd_c00 = None
        self._fwd_c01 = None
        self._fwd_c11 = None
        self._bwd_mu0 = None
        self._bwd_mu1 = None
        self._bwd_c00 = None
        self._bwd_c01 = None
        self._bwd_c11 = None
        self._lik_mu = None
        self._lik_sigma = None
        self._t_fwd_mu0 = None
        self._t_fwd_mu1 = None
        self._t_fwd_c00 = None
        self._t_fwd_c01 = None
        self._t_fwd_c11 = None
        self._t_bwd_mu0 = None
        self._t_bwd_mu1 = None
        self._t_bwd_c00 = None
        self._t_bwd_c01 = None
        self._t_bwd_c11 = None
        self._t_lik_mu = None
        self._t_lik_sigma = None
        self._p_mu0 = None
        self._p_mu1 = None
        self._p_c00 = None
        self._p_c01 = None
        self._p_c11 = None
        self._stored_player1 = None
        self._stored_player2 = None
        self._stored_scores = None
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._accum_p1 = []
        self._accum_p2 = []
        self._accum_scores = []
        self._accum_days = []
        self._last_refit_day = None
        self._checkpoint_loaded = False
        self._checkpoint_num_batches = 0
        self._checkpoint_num_appearances = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        apps = self._num_appearances or 0
        return (
            f"MaternTTT(sigma={self.config.sigma:.2f}, "
            f"beta={self.config.beta:.2f}, "
            f"lengthscale={self.config.lengthscale:.0f}, "
            f"players={players}, appearances={apps}, {status})"
        )
