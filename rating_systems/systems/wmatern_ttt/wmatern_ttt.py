"""
Weighted Matérn 3/2 TrueSkill Through Time (WMaternTTT).

Extends MaternTTT by allowing per-game weights that control performance noise:
β_eff = β / √w. Higher weight → more informative game → tighter update.
With all weights = 1.0, this is identical to standard MaternTTT.

Uses a 2D state [skill, velocity] with Matérn 3/2 dynamics instead of
the Wiener process of standard WTTT. This gives autocorrelated skill
trajectories — a player on an upswing is predicted to continue rising.

Parameters:
    sigma: Marginal std dev of the GP (stationary prior skill uncertainty)
    beta: Base performance noise (same as MaternTTT)
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
from ...data.checkpoint import save_checkpoint, load_checkpoint
from ...results.fitted_ratings import FittedTTTRatings

# Reuse structural/utility functions from TTT
from ..trueskill_through_time._numba_core import (
    build_appearance_structure,
    predict_proba_batch,
    predict_proba_batch_h,
    predict_single,
    predict_single_h,
    extract_player_last_day_ttt,
    INF_SIGMA,
)

# Matérn 2D-specific functions
from ..matern_ttt._numba_core import (
    INF_COV,
    extract_final_ratings_matern,
    extract_player_posteriors_matern,
    predict_proba_batch_at_day_matern,
    predict_single_at_day_matern,
)

# Weighted Matérn variants
from ._numba_core import (
    initial_forward_pass_matern_weighted,
    run_convergence_matern_weighted,
    initial_forward_pass_matern_weighted_h,
    run_convergence_matern_weighted_h,
)


@dataclass
class WMaternTTTConfig:
    """Configuration for Weighted Matérn 3/2 TrueSkill Through Time."""

    mu: float = 0.0             # Prior mean (internal scale)
    sigma: float = 6.0          # Marginal std dev (stationary)
    beta: float = 1.0           # Base performance noise
    lengthscale: float = 365.0  # Temporal correlation length (days)
    max_iterations: int = 30
    refit_max_iterations: int = 2
    convergence_threshold: float = 1e-6
    refit_interval: int = 1


class WeightedMaternTTT(RatingSystem):
    """
    Weighted Matérn 3/2 TrueSkill Through Time rating system.

    Extends MaternTTT with per-game weights that modulate performance noise.
    β_eff[g] = β / √(weight[g]), so higher-weight games are more
    informative.

    Uses a 2D state [skill, velocity] with Matérn 3/2 dynamics instead of
    the Wiener process of standard WTTT. This gives autocorrelated skill
    trajectories — a player on an upswing is predicted to continue rising.

    With weights=None (all 1.0), produces identical results to standard MaternTTT.

    Parameters:
        mu: Prior mean skill (default: 0.0, displayed as 1500)
        sigma: Marginal std dev of the GP (default: 6.0)
        beta: Base performance variability (default: 1.0)
        lengthscale: Temporal correlation length in days (default: 365)
        max_iterations: Max belief propagation iterations (default: 30)
        convergence_threshold: Stop when max change < threshold

    Example:
        >>> wmttt = WeightedMaternTTT(sigma=6.0, beta=1.0, lengthscale=365)
        >>> wmttt.fit(dataset, weights=mov_weights)
        >>> fitted = wmttt.get_fitted_ratings()
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
        self.config = WMaternTTTConfig(
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

        # Per-game effective beta
        self._game_beta_eff: Optional[np.ndarray] = None
        # Per-game handicaps (mu-scale, positive = p1 advantage)
        self._game_handicaps: Optional[np.ndarray] = None

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

        # Accumulated data for periodic refitting
        self._accum_p1: List[np.ndarray] = []
        self._accum_p2: List[np.ndarray] = []
        self._accum_scores: List[np.ndarray] = []
        self._accum_days: List[np.ndarray] = []
        self._accum_weights: List[np.ndarray] = []
        self._accum_handicaps: List[np.ndarray] = []
        self._last_refit_day: Optional[int] = None

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial WMaternTTT ratings."""
        return PlayerRatings(
            ratings=np.full(num_players, self.DISPLAY_OFFSET, dtype=np.float64),
            rd=np.full(num_players, self.config.sigma * self.DISPLAY_SCALE, dtype=np.float64),
            metadata={"system": "wmatern_ttt", "config": self.config},
        )

    def _ensure_temp_arrays(self) -> None:
        """Ensure temp arrays are allocated at the right size."""
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
        weights: np.ndarray,
        handicaps: Optional[np.ndarray] = None,
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

        # Compute per-game beta_eff from weights: β_eff = β / √w
        sorted_weights = weights[sort_order].astype(np.float64)
        self._game_beta_eff = self.config.beta / np.sqrt(np.maximum(sorted_weights, 1e-10))

        # Store sorted handicaps if provided
        if handicaps is not None:
            self._game_handicaps = handicaps[sort_order].astype(np.float64)
        else:
            self._game_handicaps = None

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
        weights: Optional[np.ndarray] = None,
        handicaps: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WeightedMaternTTT":
        """
        Fit WMaternTTT on a dataset.

        Args:
            dataset: Game dataset to fit on
            weights: Per-game weights (length = num_games). Higher weight
                means more informative game (β_eff = β / √w). If None,
                defaults to all 1.0 (identical to standard MaternTTT).
            handicaps: Per-game handicaps in mu-scale (positive = p1
                advantage). If None, no handicaps are applied.
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

        # Default weights = all 1.0
        if weights is None:
            weights = np.ones(n_games, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)

        if handicaps is not None:
            handicaps = np.asarray(handicaps, dtype=np.float64)

        # Create per-game day array
        days = np.empty(n_games, dtype=np.int32)
        n_days = len(day_indices)
        for d in range(n_days):
            start = day_offsets[d]
            end = day_offsets[d + 1]
            days[start:end] = day_indices[d]

        # Build data structures
        self._build_batch_structure(player1, player2, scores, days, weights,
                                    handicaps=handicaps)

        has_handicaps = self._game_handicaps is not None

        # Initial forward pass
        if has_handicaps:
            initial_forward_pass_matern_weighted_h(
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
                self._game_beta_eff,
                self._game_handicaps,
                self._lambda,
                self._sigma_sq,
                0,  # start_batch
            )
        else:
            initial_forward_pass_matern_weighted(
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
                self._game_beta_eff,
                self._lambda,
                self._sigma_sq,
                0,  # start_batch
            )

        # Run convergence
        if has_handicaps:
            self._num_iterations = run_convergence_matern_weighted_h(
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
                self._game_beta_eff,
                self._game_handicaps,
                self._lambda,
                self._sigma_sq,
                self.config.max_iterations,
                self.config.convergence_threshold,
            )
        else:
            self._num_iterations = run_convergence_matern_weighted(
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
                self._game_beta_eff,
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

        # Store data for potential refitting
        if self.config.refit_interval > 0:
            self._accum_p1 = [player1.copy()]
            self._accum_p2 = [player2.copy()]
            self._accum_scores = [scores.copy()]
            self._accum_days = [days.copy()]
            self._accum_weights = [weights.copy()]
            if handicaps is not None:
                self._accum_handicaps = [handicaps.copy()]
            else:
                self._accum_handicaps = []
            self._last_refit_day = self._current_day

        return self

    def _extract_ratings(self) -> None:
        """Extract display-scale ratings from 2D state arrays."""
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
            metadata={"system": "wmatern_ttt", "config": self.config},
        )

    def _compute_player_last_day(self) -> None:
        """Cache last active day per player for time-aware predictions."""
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
        handicaps: Optional[Union[float, np.ndarray]] = None,
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict probability that player1 beats player2.

        Args:
            player1: Player 1 index/indices.
            player2: Player 2 index/indices.
            handicaps: Per-game handicaps in mu-scale (positive = p1
                advantage). Scalar for single prediction, array for batch.
            day: If provided, propagates 2D state forward using Matérn
                dynamics for time-aware prediction.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if handicaps is not None:
            # Handicap-aware prediction — uses scalar TTT predict functions
            # (handicap is applied to skill difference, which is scalar)
            if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
                p1, p2 = int(player1), int(player2)
                return predict_single_h(
                    self._ratings.ratings[p1], self._ratings.ratings[p2],
                    self._ratings.rd[p1], self._ratings.rd[p2],
                    float(handicaps),
                    self.config.beta,
                    self.DISPLAY_SCALE,
                    self.DISPLAY_OFFSET,
                )

            p1 = np.ascontiguousarray(player1, dtype=np.int64)
            p2 = np.ascontiguousarray(player2, dtype=np.int64)
            h = np.ascontiguousarray(handicaps, dtype=np.float64)
            return predict_proba_batch_h(
                p1, p2,
                self._ratings.ratings,
                self._ratings.rd,
                h,
                self.config.beta,
                self.DISPLAY_SCALE,
                self.DISPLAY_OFFSET,
            )

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
        """Get a queryable fitted ratings object."""
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
        """No-op. WMaternTTT is a batch algorithm that requires full refit."""
        pass

    def update(self, batch: GameBatch) -> "WeightedMaternTTT":
        """Update with default weights (all 1.0)."""
        return self.update_weighted(batch, np.ones(len(batch), dtype=np.float64))

    def update_weighted(
        self,
        batch: GameBatch,
        weights: np.ndarray,
        handicaps: Optional[np.ndarray] = None,
    ) -> "WeightedMaternTTT":
        """
        Incrementally update with new games, weights, and optional handicaps.

        Accumulates data and refits periodically based on refit_interval.
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
        self._accum_weights.append(np.asarray(weights, dtype=np.float64).copy())

        if handicaps is not None:
            if not self._accum_handicaps:
                # fit() was called without handicaps — pad zeros for prior batches
                for p1_arr in self._accum_p1[:-1]:
                    self._accum_handicaps.append(np.zeros(len(p1_arr), dtype=np.float64))
            self._accum_handicaps.append(np.asarray(handicaps, dtype=np.float64).copy())
        elif self._accum_handicaps:
            # Previous batches had handicaps — fill zeros for consistency
            self._accum_handicaps.append(np.zeros(n_games, dtype=np.float64))

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
        """Refit on all accumulated data with warm-start."""
        if not self._accum_p1:
            return

        player1 = np.concatenate(self._accum_p1)
        player2 = np.concatenate(self._accum_p2)
        scores = np.concatenate(self._accum_scores)
        days = np.concatenate(self._accum_days)
        weights = np.concatenate(self._accum_weights)
        handicaps_arr = np.concatenate(self._accum_handicaps) if self._accum_handicaps else None

        max_player = max(player1.max(), player2.max())
        if self._num_players is None or max_player >= self._num_players:
            self._num_players = int(max_player) + 1
            self._ratings = self._initialize_ratings(self._num_players)

        # Save old state for warm-start
        old_num_batches = self._num_batches
        old_num_appearances = self._num_appearances
        old_state = self._save_state_arrays()

        # Rebuild sparse structures
        self._build_batch_structure(player1, player2, scores, days, weights,
                                    handicaps=handicaps_arr)

        # Restore old state for previously-computed appearances
        if old_state is not None and old_num_appearances > 0:
            self._restore_state_arrays(old_state, old_num_appearances)

        has_handicaps = self._game_handicaps is not None

        # Forward pass only for new batches
        if has_handicaps:
            initial_forward_pass_matern_weighted_h(
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
                self._game_beta_eff,
                self._game_handicaps,
                self._lambda,
                self._sigma_sq,
                old_num_batches,  # start_batch
            )
        else:
            initial_forward_pass_matern_weighted(
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
                self._game_beta_eff,
                self._lambda,
                self._sigma_sq,
                old_num_batches,  # start_batch
            )

        # Run convergence
        if has_handicaps:
            self._num_iterations = run_convergence_matern_weighted_h(
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
                self._game_beta_eff,
                self._game_handicaps,
                self._lambda,
                self._sigma_sq,
                self.config.refit_max_iterations,
                self.config.convergence_threshold,
            )
        else:
            self._num_iterations = run_convergence_matern_weighted(
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
                self._game_beta_eff,
                self._lambda,
                self._sigma_sq,
                self.config.refit_max_iterations,
                self.config.convergence_threshold,
            )

        # Extract final ratings and posteriors
        self._extract_ratings()
        self._compute_player_last_day()
        self._compute_player_posteriors()
        self._num_games_fitted = len(player1)

    def snapshot(self) -> dict:
        """Snapshot full WMaternTTT state including message-passing structures."""
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
        # State arrays (12 per appearance) + structure + game arrays + posteriors
        for attr in (
            "_fwd_mu0", "_fwd_mu1", "_fwd_c00", "_fwd_c01", "_fwd_c11",
            "_bwd_mu0", "_bwd_mu1", "_bwd_c00", "_bwd_c01", "_bwd_c11",
            "_lik_mu", "_lik_sigma",
            # Structure arrays
            "_app_offsets", "_app_player", "_app_prev", "_app_next",
            "_app_batch", "_player_last_app", "_player_last_day",
            "_batch_offsets", "_batch_times",
            "_game_p1", "_game_p2", "_game_scores",
            "_game_beta_eff", "_game_handicaps",
            # Per-player posteriors
            "_p_mu0", "_p_mu1", "_p_c00", "_p_c01", "_p_c11",
        ):
            val = getattr(self, attr)
            state[attr] = val.copy() if val is not None else None
        # Accumulated data (lists of arrays)
        state["_accum_p1"] = [a.copy() for a in self._accum_p1]
        state["_accum_p2"] = [a.copy() for a in self._accum_p2]
        state["_accum_scores"] = [a.copy() for a in self._accum_scores]
        state["_accum_days"] = [a.copy() for a in self._accum_days]
        state["_accum_weights"] = [a.copy() for a in self._accum_weights]
        state["_accum_handicaps"] = [a.copy() for a in self._accum_handicaps]
        return state

    def restore(self, state: dict) -> None:
        """Restore full WMaternTTT state from snapshot."""
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
            "_game_beta_eff", "_game_handicaps",
            "_p_mu0", "_p_mu1", "_p_c00", "_p_c01", "_p_c11",
        ):
            val = state[attr]
            setattr(self, attr, val.copy() if val is not None else None)
        self._accum_p1 = [a.copy() for a in state["_accum_p1"]]
        self._accum_p2 = [a.copy() for a in state["_accum_p2"]]
        self._accum_scores = [a.copy() for a in state["_accum_scores"]]
        self._accum_days = [a.copy() for a in state["_accum_days"]]
        self._accum_weights = [a.copy() for a in state["_accum_weights"]]
        self._accum_handicaps = [a.copy() for a in state["_accum_handicaps"]]

    def save_state(self, path: str) -> None:
        """Save fitted WMaternTTT state to .npz file."""
        if not self._fitted or self._fwd_mu0 is None:
            raise ValueError("Model must be fitted before saving state.")

        arrays = {
            "ratings": self._ratings.ratings,
            "rd": self._ratings.rd,
            # 2D state arrays
            "fwd_mu0": self._fwd_mu0, "fwd_mu1": self._fwd_mu1,
            "fwd_c00": self._fwd_c00, "fwd_c01": self._fwd_c01, "fwd_c11": self._fwd_c11,
            "bwd_mu0": self._bwd_mu0, "bwd_mu1": self._bwd_mu1,
            "bwd_c00": self._bwd_c00, "bwd_c01": self._bwd_c01, "bwd_c11": self._bwd_c11,
            "lik_mu": self._lik_mu, "lik_sigma": self._lik_sigma,
            # Structure arrays
            "app_offsets": self._app_offsets,
            "app_player": self._app_player,
            "app_prev": self._app_prev,
            "app_next": self._app_next,
            "app_batch": self._app_batch,
            "player_last_app": self._player_last_app,
            # Batch arrays
            "batch_offsets": self._batch_offsets,
            "batch_times": self._batch_times,
            # Game arrays
            "game_p1": self._game_p1,
            "game_p2": self._game_p2,
            "game_scores": self._game_scores,
            "game_beta_eff": self._game_beta_eff,
        }
        if self._game_handicaps is not None:
            arrays["game_handicaps"] = self._game_handicaps

        metadata = {
            "system_class": "WeightedMaternTTT",
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
        """Restore fitted WMaternTTT state from .npz file."""
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
            metadata={"system": "wmatern_ttt", "config": self.config},
        )

        # 2D state arrays
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

        # Batch arrays
        self._batch_offsets = arrays["batch_offsets"]
        self._batch_times = arrays["batch_times"]

        # Game arrays
        self._game_p1 = arrays["game_p1"]
        self._game_p2 = arrays["game_p2"]
        self._game_scores = arrays["game_scores"]
        self._game_beta_eff = arrays.get("game_beta_eff")
        self._game_handicaps = arrays.get("game_handicaps")

        # Reconstruct per-game days from batch structure
        days = np.empty(len(self._game_p1), dtype=np.int32)
        for b in range(self._num_batches):
            start = self._batch_offsets[b]
            end = self._batch_offsets[b + 1] if b + 1 < self._num_batches else len(self._game_p1)
            days[start:end] = self._batch_times[b]

        # Recover weights from game_beta_eff: beta_eff = beta / sqrt(w) -> w = (beta / beta_eff)^2
        if self._game_beta_eff is not None:
            weights = (self.config.beta / np.maximum(self._game_beta_eff, 1e-10)) ** 2
        else:
            weights = np.ones(len(self._game_p1), dtype=np.float64)

        # Initialize accumulation lists for update() / update_weighted() compatibility
        self._accum_p1 = [self._game_p1.copy()]
        self._accum_p2 = [self._game_p2.copy()]
        self._accum_scores = [self._game_scores.copy()]
        self._accum_days = [days]
        self._accum_weights = [weights]
        if self._game_handicaps is not None:
            self._accum_handicaps = [self._game_handicaps.copy()]
        else:
            self._accum_handicaps = []
        self._last_refit_day = self._current_day
        self._compute_player_last_day()
        self._compute_player_posteriors()

    def reset(self) -> "WeightedMaternTTT":
        """Reset the rating system."""
        self._num_batches = 0
        self._batch_offsets = None
        self._batch_times = None
        self._game_p1 = None
        self._game_p2 = None
        self._game_scores = None
        self._game_beta_eff = None
        self._game_handicaps = None
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
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._accum_p1 = []
        self._accum_p2 = []
        self._accum_scores = []
        self._accum_days = []
        self._accum_weights = []
        self._accum_handicaps = []
        self._last_refit_day = None
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        apps = self._num_appearances or 0
        return (
            f"WeightedMaternTTT(sigma={self.config.sigma:.2f}, "
            f"beta={self.config.beta:.2f}, "
            f"lengthscale={self.config.lengthscale:.0f}, "
            f"players={players}, appearances={apps}, {status})"
        )
