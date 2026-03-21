"""
Weighted TrueSkill Through Time (WTTT).

Extends TTT by allowing per-game weights that control performance noise:
β_eff = β / √w. Higher weight → more informative game → tighter update.
With all weights = 1.0, this is identical to standard TTT.

Learned-α handicaps: when handicap_features are provided (an n_games ×
n_features matrix of raw covariate values), the system learns scaling
factors α_k for each feature type via Bayesian linear regression
integrated into the message-passing forward pass.

The effective handicap for each game is:
    h_g = Σ_k α_k * features[g, k]

The α posterior is maintained as a Gaussian N(Λ⁻¹η, Λ⁻¹) with prior
N(0, σ²_prior · I).  During each forward-pass batch, the truncated-
Gaussian moments (v, w) from belief propagation are used to update the
precision matrix Λ and natural parameter η via an online Laplace
approximation.  This naturally regularises α and avoids the divergence
problems of the previous Newton outer-loop approach.
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
    extract_final_ratings_sparse,
    predict_proba_batch,
    predict_proba_batch_at_day,
    predict_proba_batch_h,
    predict_proba_batch_at_day_h,
    predict_single,
    predict_single_at_day,
    predict_single_h,
    predict_single_at_day_h,
    extract_player_last_day_ttt,
    INF_SIGMA,
)

# Weighted variants
from ._numba_core import (
    initial_forward_pass_weighted,
    run_convergence_weighted,
    initial_forward_pass_weighted_h,
    run_convergence_weighted_h,
    initial_forward_pass_weighted_h_bayesian,
)


@dataclass
class WTTTConfig:
    """Configuration for Weighted TrueSkill Through Time."""

    mu: float = 0.0
    sigma: float = 3.8735      # Optimised TTT value
    beta: float = 2.3747       # Optimised TTT value
    gamma: float = 0.0804      # Optimised TTT value
    max_iterations: int = 50   # Canonical TTT default
    refit_max_iterations: int = 1
    convergence_threshold: float = 1e-6
    refit_interval: int = 1
    # Bayesian α learning (prior variance for N(0, σ²_prior · I))
    alpha_prior_variance: float = 10.0
    # Time-warp margin penalty (days): decisive wins stay at real time,
    # close wins are pushed into the past by up to this many days.
    margin_time_penalty: float = 0.0


class WeightedTTT(RatingSystem):
    """
    Weighted TrueSkill Through Time rating system.

    Extends TTT with per-game weights that modulate performance noise.
    β_eff[g] = β / √(weight[g]), so higher-weight games are more
    informative.

    When handicap_features are provided, the system learns scaling factors
    (α) for each feature via Bayesian linear regression integrated into
    the forward pass.  The α posterior N(Λ⁻¹η, Λ⁻¹) is updated online
    using truncated-Gaussian moments from belief propagation.

    With weights=None (all 1.0) and no features, produces identical results
    to standard TTT.

    Parameters:
        mu: Prior mean skill (default: 0.0, displayed as 1500)
        sigma: Prior skill std dev
        beta: Base performance variability
        gamma: Skill drift rate per time unit
        max_iterations: Max belief propagation iterations
        convergence_threshold: Stop when max change < threshold
        alpha_prior_variance: Prior variance for α ~ N(0, σ²_prior · I)

    Example:
        >>> wttt = WeightedTTT(sigma=3.87, beta=2.37, gamma=0.08)
        >>> wttt.fit(dataset, weights=mov_weights, handicap_features=features)
        >>> fitted = wttt.get_fitted_ratings()
    """

    system_type = RatingSystemType.BATCH

    DISPLAY_OFFSET = 1500.0
    DISPLAY_SCALE = 400.0 / 6.0

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 3.8735,
        beta: float = 2.3747,
        gamma: float = 0.0804,
        max_iterations: int = 50,
        refit_max_iterations: int = 1,
        convergence_threshold: float = 1e-6,
        refit_interval: int = 1,
        alpha_prior_variance: float = 10.0,
        margin_time_penalty: float = 0.0,
        num_players: Optional[int] = None,
    ):
        self.config = WTTTConfig(
            mu=mu,
            sigma=sigma,
            beta=beta,
            gamma=gamma,
            max_iterations=max_iterations,
            refit_max_iterations=refit_max_iterations,
            convergence_threshold=convergence_threshold,
            refit_interval=refit_interval,
            alpha_prior_variance=alpha_prior_variance,
            margin_time_penalty=margin_time_penalty,
        )

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
        # Per-appearance gamma for volatility modulation
        self._app_gamma: Optional[np.ndarray] = None

        # Learned-α handicap state
        self._handicap_features: Optional[np.ndarray] = None
        self._handicap_feature_names: Optional[List[str]] = None
        self._alpha: Optional[np.ndarray] = None
        # Bayesian α posterior: precision matrix Λ and precision-weighted mean η
        self._alpha_Lambda: Optional[np.ndarray] = None  # K×K
        self._alpha_eta: Optional[np.ndarray] = None      # K-vector

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
        self._accum_weights: List[np.ndarray] = []
        self._accum_handicap_features: List[np.ndarray] = []
        self._accum_margin_sq: List[np.ndarray] = []
        self._last_refit_day: Optional[int] = None

        super().__init__(num_players=num_players)

    @property
    def alpha(self) -> Optional[np.ndarray]:
        """Current learned handicap scaling factors (read-only copy)."""
        return self._alpha.copy() if self._alpha is not None else None

    @property
    def handicap_feature_names(self) -> Optional[List[str]]:
        """Names of handicap features, if provided."""
        return self._handicap_feature_names

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial WTTT ratings."""
        return PlayerRatings(
            ratings=np.full(num_players, self.DISPLAY_OFFSET, dtype=np.float64),
            rd=np.full(num_players, self.config.sigma * self.DISPLAY_SCALE, dtype=np.float64),
            metadata={"system": "wttt", "config": self.config},
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
        weights: np.ndarray,
        handicap_features: Optional[np.ndarray] = None,
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

        # Handle handicap features
        if handicap_features is not None:
            sorted_features = handicap_features[sort_order]
            n_features = sorted_features.shape[1]

            # Initialise α to zero if not already set (or wrong size)
            if self._alpha is None or len(self._alpha) != n_features:
                self._alpha = np.zeros(n_features, dtype=np.float64)

            # Initialise Bayesian posterior if needed
            if (self._alpha_Lambda is None or
                    self._alpha_Lambda.shape[0] != n_features):
                # Prior: Λ₀ = (1/σ²_prior) · I, η₀ = 0
                prior_prec = 1.0 / self.config.alpha_prior_variance
                self._alpha_Lambda = np.eye(n_features, dtype=np.float64) * prior_prec
                self._alpha_eta = np.zeros(n_features, dtype=np.float64)

            # Compute initial handicaps from current α
            self._game_handicaps = sorted_features @ self._alpha
            self._handicap_features = sorted_features
        else:
            self._game_handicaps = None
            self._handicap_features = None
            self._alpha = None
            self._alpha_Lambda = None
            self._alpha_eta = None

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

        self._app_gamma = np.full(self._num_appearances, self.config.gamma,
                                   dtype=np.float64)

    def _run_initial_forward_pass(self, start_batch: int = 0) -> None:
        """Run initial forward pass (weighted or weighted+h)."""
        has_h = self._game_handicaps is not None

        if has_h:
            initial_forward_pass_weighted_h(
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
                self._game_beta_eff,
                self._game_handicaps,
                self._app_gamma,
                start_batch,
            )
        else:
            initial_forward_pass_weighted(
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
                self._game_beta_eff,
                self._app_gamma,
                start_batch,
            )

    def _run_bayesian_forward_pass(self, start_batch: int = 0) -> None:
        """Run Bayesian inline α forward pass.

        Updates α posterior (Λ, η) incrementally during the forward pass,
        solving for α_mean after each batch and recomputing handicaps for
        subsequent batches. This replaces the Newton outer-loop approach.
        """
        n_features = len(self._alpha)

        # Make working copies of Λ and η (the Numba function modifies in-place)
        Lambda_work = self._alpha_Lambda.copy()
        eta_work = self._alpha_eta.copy()

        alpha_result = initial_forward_pass_weighted_h_bayesian(
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
            self._game_beta_eff,
            self._game_handicaps,
            self._handicap_features,
            n_features,
            Lambda_work,
            eta_work,
            self._app_gamma,
            start_batch,
        )

        # Store the updated posterior
        self._alpha_Lambda = Lambda_work
        self._alpha_eta = eta_work
        self._alpha = alpha_result.copy()

    def _run_convergence(self, max_iterations: int) -> int:
        """Run forward-backward convergence (weighted or weighted+h)."""
        has_h = self._game_handicaps is not None

        if has_h:
            return run_convergence_weighted_h(
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
                self._game_beta_eff,
                self._game_handicaps,
                self._app_gamma,
                max_iterations,
                self.config.convergence_threshold,
            )
        else:
            return run_convergence_weighted(
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
                self._game_beta_eff,
                self._app_gamma,
                max_iterations,
                self.config.convergence_threshold,
            )

    def _extract_final_ratings(self) -> None:
        """Extract final ratings from converged beliefs."""
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
            metadata={"system": "wttt", "config": self.config},
        )

    def fit(
        self,
        dataset: GameDataset,
        weights: Optional[np.ndarray] = None,
        handicap_features: Optional[np.ndarray] = None,
        handicap_feature_names: Optional[List[str]] = None,
        margin_sq: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WeightedTTT":
        """
        Fit WTTT on a dataset.

        Args:
            dataset: Game dataset to fit on
            weights: Per-game weights (length = num_games). Higher weight
                means more informative game (β_eff = β / √w). If None,
                defaults to all 1.0 (identical to standard TTT).
            handicap_features: Raw feature matrix (n_games, n_features).
                Each column is a covariate (surface bias, rest diff, etc.).
                The system learns scaling factors α internally.
                If None, no handicaps are applied.
            handicap_feature_names: Optional names for each feature column.
            margin_sq: Per-game squared margin in [0, 1] for time-warp.
                When margin_time_penalty > 0, games are shifted into the
                past: effective_day = real_day - penalty * (1 - margin_sq).
            end_day: Last day to include (inclusive).
            player_names: Optional mapping of player_id -> name

        Returns:
            self (for method chaining)
        """
        self._player_names = player_names
        self._handicap_feature_names = handicap_feature_names

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

        if handicap_features is not None:
            handicap_features = np.ascontiguousarray(handicap_features, dtype=np.float64)
            if handicap_features.shape[0] != n_games:
                raise ValueError(
                    f"handicap_features has {handicap_features.shape[0]} rows "
                    f"but dataset has {n_games} games"
                )

        # Create per-game day array
        days = np.empty(n_games, dtype=np.int32)
        n_days = len(day_indices)
        for d in range(n_days):
            start = day_offsets[d]
            end = day_offsets[d + 1]
            days[start:end] = day_indices[d]

        # Apply time-warp: shift close wins into the past
        if margin_sq is not None and self.config.margin_time_penalty > 0:
            days = np.round(
                days.astype(np.float64)
                - self.config.margin_time_penalty * (1.0 - margin_sq)
            ).astype(np.int32)

        # Build data structures
        self._build_batch_structure(player1, player2, scores, days, weights,
                                    handicap_features=handicap_features)

        has_learned_h = self._handicap_features is not None and self._alpha is not None

        # Initial forward pass (Bayesian α when features present, else standard)
        if has_learned_h:
            self._run_bayesian_forward_pass(start_batch=0)
        else:
            self._run_initial_forward_pass(start_batch=0)

        # Convergence sweeps with fixed handicaps (Bayesian α already
        # learned during forward pass; _run_convergence dispatches to _h
        # variant when _game_handicaps is set)
        self._num_iterations = self._run_convergence(self.config.max_iterations)

        # Extract final ratings
        self._extract_final_ratings()

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
            self._accum_weights = [weights.copy()]
            if handicap_features is not None:
                self._accum_handicap_features = [handicap_features.copy()]
            else:
                self._accum_handicap_features = []
            if margin_sq is not None:
                self._accum_margin_sq = [margin_sq.copy()]
            else:
                self._accum_margin_sq = []
            self._last_refit_day = self._current_day

        return self

    def _compute_player_last_day(self) -> None:
        """Cache last active day per player for time-aware predictions."""
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
        handicap_features: Optional[np.ndarray] = None,
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict probability that player1 beats player2.

        When handicap_features are provided and α has been learned, computes
        per-game handicaps h = features @ α (in mu-scale).

        Args:
            player1: Player 1 index/indices.
            player2: Player 2 index/indices.
            handicap_features: Raw feature matrix (n_pred, n_features).
                The learned α is applied to compute handicaps.
            day: If provided, grows sigma forward using gamma for inactivity.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Compute handicaps from features and learned α
        if handicap_features is not None and self._alpha is not None:
            handicap_features = np.ascontiguousarray(handicap_features, dtype=np.float64)
            handicaps = handicap_features @ self._alpha

            if day is not None and self._player_last_day is not None:
                if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
                    p1, p2 = int(player1), int(player2)
                    return predict_single_at_day_h(
                        self._ratings.ratings[p1], self._ratings.ratings[p2],
                        self._ratings.rd[p1], self._ratings.rd[p2],
                        float(handicaps) if handicaps.ndim == 0 else float(handicaps[0]),
                        self._player_last_day[p1], self._player_last_day[p2],
                        day, self.config.beta, self.config.gamma,
                        self.DISPLAY_SCALE, self.DISPLAY_OFFSET,
                    )
                p1 = np.ascontiguousarray(player1, dtype=np.int64)
                p2 = np.ascontiguousarray(player2, dtype=np.int64)
                h = np.ascontiguousarray(handicaps, dtype=np.float64)
                return predict_proba_batch_at_day_h(
                    p1, p2, self._ratings.ratings, self._ratings.rd,
                    h, self._player_last_day, day,
                    self.config.beta, self.config.gamma,
                    self.DISPLAY_SCALE, self.DISPLAY_OFFSET,
                )

            if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
                p1, p2 = int(player1), int(player2)
                return predict_single_h(
                    self._ratings.ratings[p1], self._ratings.ratings[p2],
                    self._ratings.rd[p1], self._ratings.rd[p2],
                    float(handicaps) if handicaps.ndim == 0 else float(handicaps[0]),
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

        # No handicaps — original path
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
        """No-op. WTTT is a batch algorithm that requires full refit."""
        pass

    def update(self, batch: GameBatch) -> "WeightedTTT":
        """Update with default weights (all 1.0)."""
        return self.update_weighted(batch, np.ones(len(batch), dtype=np.float64))

    def update_weighted(
        self,
        batch: GameBatch,
        weights: np.ndarray,
        handicap_features: Optional[np.ndarray] = None,
        margin_sq: Optional[np.ndarray] = None,
    ) -> "WeightedTTT":
        """
        Incrementally update with new games, weights, and optional features.

        Accumulates data and refits periodically based on refit_interval.

        Args:
            batch: New game batch
            weights: Per-game weights for the batch
            handicap_features: Raw feature matrix (n_batch, n_features)
            margin_sq: Per-game squared margin in [0, 1] for time-warp.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before updating. Call fit() first.")

        if self.config.refit_interval <= 0:
            return self

        n_games = len(batch)

        # Compute effective days with time-warp
        if margin_sq is not None and self.config.margin_time_penalty > 0:
            days_array = np.round(
                float(batch.day)
                - self.config.margin_time_penalty * (1.0 - margin_sq)
            ).astype(np.int32)
        else:
            days_array = np.full(n_games, batch.day, dtype=np.int32)

        self._accum_p1.append(batch.player1.copy())
        self._accum_p2.append(batch.player2.copy())
        self._accum_scores.append(batch.scores.copy())
        self._accum_days.append(days_array)

        # Store margin_sq for future refitting
        if margin_sq is not None:
            if not self._accum_margin_sq:
                # fit() had no margin_sq — pad zeros for prior batches
                for p1_arr in self._accum_p1[:-1]:
                    self._accum_margin_sq.append(
                        np.zeros(len(p1_arr), dtype=np.float64)
                    )
            self._accum_margin_sq.append(margin_sq.copy())
        elif self._accum_margin_sq:
            self._accum_margin_sq.append(
                np.zeros(n_games, dtype=np.float64)
            )
        self._accum_weights.append(np.asarray(weights, dtype=np.float64).copy())

        if handicap_features is not None:
            handicap_features = np.ascontiguousarray(handicap_features, dtype=np.float64)
            if not self._accum_handicap_features:
                # fit() was called without features — pad zeros for prior batches
                for p1_arr in self._accum_p1[:-1]:
                    n_feat = handicap_features.shape[1]
                    self._accum_handicap_features.append(
                        np.zeros((len(p1_arr), n_feat), dtype=np.float64)
                    )
            self._accum_handicap_features.append(handicap_features.copy())
        elif self._accum_handicap_features:
            # Previous batches had features — fill zeros for consistency
            n_feat = self._accum_handicap_features[0].shape[1]
            self._accum_handicap_features.append(
                np.zeros((n_games, n_feat), dtype=np.float64)
            )

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
        weights = np.concatenate(self._accum_weights)
        features_arr = (
            np.concatenate(self._accum_handicap_features)
            if self._accum_handicap_features else None
        )

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
        self._build_batch_structure(player1, player2, scores, days, weights,
                                    handicap_features=features_arr)

        # Restore old state for previously-computed appearances
        if old_fwd_mu is not None and old_num_appearances > 0:
            n = old_num_appearances
            self._state_forward_mu[:n] = old_fwd_mu[:n]
            self._state_forward_sigma[:n] = old_fwd_sigma[:n]
            self._state_backward_mu[:n] = old_bwd_mu[:n]
            self._state_backward_sigma[:n] = old_bwd_sigma[:n]
            self._state_likelihood_mu[:n] = old_lik_mu[:n]
            self._state_likelihood_sigma[:n] = old_lik_sigma[:n]

        has_learned_h = self._handicap_features is not None and self._alpha is not None

        # Forward pass only for new batches
        if has_learned_h:
            self._run_bayesian_forward_pass(start_batch=old_num_batches)
        else:
            self._run_initial_forward_pass(start_batch=old_num_batches)

        # Run convergence
        self._num_iterations = self._run_convergence(
            self.config.refit_max_iterations
        )

        # Extract final ratings
        self._extract_final_ratings()
        self._compute_player_last_day()

        self._num_games_fitted = len(player1)

    def snapshot(self) -> dict:
        """Snapshot full WTTT state including message-passing structures and learned α."""
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
            "_game_p1", "_game_p2", "_game_scores",
            "_game_beta_eff", "_game_handicaps", "_app_gamma",
            "_alpha", "_handicap_features",
            "_alpha_Lambda", "_alpha_eta",
        ):
            val = getattr(self, attr)
            state[attr] = val.copy() if val is not None else None
        # Accumulated data (lists of arrays)
        state["_accum_p1"] = [a.copy() for a in self._accum_p1]
        state["_accum_p2"] = [a.copy() for a in self._accum_p2]
        state["_accum_scores"] = [a.copy() for a in self._accum_scores]
        state["_accum_days"] = [a.copy() for a in self._accum_days]
        state["_accum_weights"] = [a.copy() for a in self._accum_weights]
        state["_accum_handicap_features"] = [a.copy() for a in self._accum_handicap_features]
        state["_accum_margin_sq"] = [a.copy() for a in self._accum_margin_sq]
        state["_handicap_feature_names"] = (
            list(self._handicap_feature_names) if self._handicap_feature_names else None
        )
        return state

    def restore(self, state: dict) -> None:
        """Restore full WTTT state from snapshot."""
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
            "_game_p1", "_game_p2", "_game_scores",
            "_game_beta_eff", "_game_handicaps", "_app_gamma",
            "_alpha", "_handicap_features",
            "_alpha_Lambda", "_alpha_eta",
        ):
            val = state.get(attr)
            setattr(self, attr, val.copy() if val is not None else None)
        self._accum_p1 = [a.copy() for a in state["_accum_p1"]]
        self._accum_p2 = [a.copy() for a in state["_accum_p2"]]
        self._accum_scores = [a.copy() for a in state["_accum_scores"]]
        self._accum_days = [a.copy() for a in state["_accum_days"]]
        self._accum_weights = [a.copy() for a in state["_accum_weights"]]
        self._accum_handicap_features = [a.copy() for a in state["_accum_handicap_features"]]
        self._accum_margin_sq = [a.copy() for a in state.get("_accum_margin_sq", [])]
        self._handicap_feature_names = state.get("_handicap_feature_names")

    def save_state(self, path: str) -> None:
        """Save fitted WTTT state to .npz file."""
        if not self._fitted or self._state_forward_mu is None:
            raise ValueError("Model must be fitted before saving state.")

        arrays = {
            "ratings": self._ratings.ratings,
            "rd": self._ratings.rd,
            # State arrays
            "state_forward_mu": self._state_forward_mu,
            "state_forward_sigma": self._state_forward_sigma,
            "state_backward_mu": self._state_backward_mu,
            "state_backward_sigma": self._state_backward_sigma,
            "state_likelihood_mu": self._state_likelihood_mu,
            "state_likelihood_sigma": self._state_likelihood_sigma,
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
        if self._app_gamma is not None:
            arrays["app_gamma"] = self._app_gamma
        if self._game_handicaps is not None:
            arrays["game_handicaps"] = self._game_handicaps
        if self._alpha is not None:
            arrays["alpha"] = self._alpha
        if self._handicap_features is not None:
            arrays["handicap_features"] = self._handicap_features
        if self._alpha_Lambda is not None:
            arrays["alpha_Lambda"] = self._alpha_Lambda
        if self._alpha_eta is not None:
            arrays["alpha_eta"] = self._alpha_eta

        metadata = {
            "system_class": "WeightedTTT",
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
                "margin_time_penalty": self.config.margin_time_penalty,
            },
        }
        if self._handicap_feature_names:
            metadata["handicap_feature_names"] = self._handicap_feature_names
        save_checkpoint(path, arrays, metadata)

    def load_state(self, path: str) -> None:
        """Restore fitted WTTT state from .npz file."""
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
            metadata={"system": "wttt", "config": self.config},
        )

        # State arrays
        self._state_forward_mu = arrays["state_forward_mu"]
        self._state_forward_sigma = arrays["state_forward_sigma"]
        self._state_backward_mu = arrays["state_backward_mu"]
        self._state_backward_sigma = arrays["state_backward_sigma"]
        self._state_likelihood_mu = arrays["state_likelihood_mu"]
        self._state_likelihood_sigma = arrays["state_likelihood_sigma"]

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
        self._app_gamma = arrays.get("app_gamma")
        if self._app_gamma is None and self._num_appearances > 0:
            self._app_gamma = np.full(self._num_appearances, self.config.gamma,
                                       dtype=np.float64)

        # Learned-α state
        self._alpha = arrays.get("alpha")
        self._handicap_features = arrays.get("handicap_features")
        self._handicap_feature_names = metadata.get("handicap_feature_names")
        self._alpha_Lambda = arrays.get("alpha_Lambda")
        self._alpha_eta = arrays.get("alpha_eta")

        # Reconstruct per-game days from batch structure
        days = np.empty(len(self._game_p1), dtype=np.int32)
        for b in range(self._num_batches):
            start = self._batch_offsets[b]
            end = self._batch_offsets[b + 1] if b + 1 < self._num_batches else len(self._game_p1)
            days[start:end] = self._batch_times[b]

        # Recover weights from game_beta_eff: beta_eff = beta / sqrt(w) -> w = (beta / beta_eff)^2
        weights = (self.config.beta / np.maximum(self._game_beta_eff, 1e-10)) ** 2

        # Initialise accumulation lists for update() / update_weighted() compatibility
        self._accum_p1 = [self._game_p1.copy()]
        self._accum_p2 = [self._game_p2.copy()]
        self._accum_scores = [self._game_scores.copy()]
        self._accum_days = [days]
        self._accum_weights = [weights]
        if self._handicap_features is not None:
            self._accum_handicap_features = [self._handicap_features.copy()]
        else:
            self._accum_handicap_features = []
        self._last_refit_day = self._current_day
        self._compute_player_last_day()

    def reset(self) -> "WeightedTTT":
        """Reset the rating system."""
        self._num_batches = 0
        self._batch_offsets = None
        self._batch_times = None
        self._game_p1 = None
        self._game_p2 = None
        self._game_scores = None
        self._game_beta_eff = None
        self._game_handicaps = None
        self._app_gamma = None
        self._alpha = None
        self._alpha_Lambda = None
        self._alpha_eta = None
        self._handicap_features = None
        self._handicap_feature_names = None
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
        self._accum_weights = []
        self._accum_handicap_features = []
        self._accum_margin_sq = []
        self._last_refit_day = None
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        alpha_str = ""
        if self._alpha is not None:
            alpha_str = f", alpha={self._alpha}"
        return (
            f"WeightedTTT(sigma={self.config.sigma:.4f}, "
            f"beta={self.config.beta:.4f}, "
            f"gamma={self.config.gamma:.4f}, "
            f"players={players}, {status}{alpha_str})"
        )
