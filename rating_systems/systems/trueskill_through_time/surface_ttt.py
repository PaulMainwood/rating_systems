"""
Surface-specific TrueSkill Through Time (SurfaceTTT).

Models player skill using weighted teams:
    Team = [Player_base, Player_surface]
    Team_skill = w_base × base_skill + w_surf × surface_skill

Each match becomes a team game:
    Team1 = [P1_base, P1_surface] vs Team2 = [P2_base, P2_surface]

Uses SPARSE appearance-based storage: state arrays are indexed by
appearance ID (unique player-batch combinations), not by the full
batch × num_players product. Memory scales with actual appearances
(~1.5M) rather than the dense product (~1B), giving ~600× reduction.

Surface Model:
    - Non-Clay (surface=0): Hard and Grass courts
    - Clay (surface=1): Clay courts
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ._team_numba_core import (
    predict_proba_batch,
    predict_team_match,
    predict_team_match_at_day,
    INF_SIGMA,
)
from ._sparse_team_numba_core import (
    build_team_appearance_structure,
    sparse_initial_forward_pass,
    sparse_run_convergence,
    sparse_extract_final_ratings,
)


# =============================================================================
# Surface Constants
# =============================================================================

SURFACE_HARD = 0
SURFACE_CLAY = 1
SURFACE_GRASS = 2

SURFACE_NON_CLAY = 0
SURFACE_CLAY_INTERNAL = 1

NUM_SURFACES = 2

SURFACE_NAMES = ["Non-Clay", "Clay"]

SURFACE_MAP = {
    SURFACE_HARD: SURFACE_NON_CLAY,
    SURFACE_CLAY: SURFACE_CLAY_INTERNAL,
    SURFACE_GRASS: SURFACE_NON_CLAY,
}


def map_surface(surface: int) -> int:
    """Map from 3-surface (Hard/Clay/Grass) to 2-surface (Non-Clay/Clay)."""
    return SURFACE_MAP.get(surface, SURFACE_NON_CLAY)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SurfaceTTTConfig:
    """Configuration for Surface-specific TrueSkill Through Time."""

    base_weight: float = 0.9
    mu: float = 0.0
    sigma: float = 6.0
    gamma: float = 0.03
    surface_sigma: float = 3.0
    surface_gamma: float = 0.01
    beta: float = 1.0
    max_iterations: int = 30
    refit_max_iterations: int = 2
    convergence_threshold: float = 1e-6
    refit_interval: int = 1


# =============================================================================
# Main Class
# =============================================================================

class SurfaceTTT(RatingSystem):
    """
    Surface-specific TrueSkill Through Time rating system.

    Uses weighted team games where each player is modeled as a team of
    [base_player, surface_player]. Sparse appearance-based storage keeps
    memory proportional to actual player appearances, not the full
    batch × player product.

    Player Space Layout (for N real players):
        [0, N):    Base players
        [N, 2N):   Non-Clay surface components
        [2N, 3N):  Clay surface components
    """

    system_type = RatingSystemType.BATCH

    DISPLAY_OFFSET = 1500.0
    DISPLAY_SCALE = 400.0 / 6.0

    def __init__(
        self,
        base_weight: float = 0.9,
        mu: float = 0.0,
        sigma: float = 6.0,
        beta: float = 1.0,
        gamma: float = 0.03,
        max_iterations: int = 30,
        refit_max_iterations: int = 2,
        convergence_threshold: float = 1e-6,
        refit_interval: int = 1,
        surface_sigma: float = 3.0,
        surface_gamma: float = 0.01,
        num_players: Optional[int] = None,
    ):
        self.config = SurfaceTTTConfig(
            base_weight=base_weight, mu=mu, sigma=sigma, beta=beta,
            gamma=gamma, max_iterations=max_iterations,
            refit_max_iterations=refit_max_iterations,
            convergence_threshold=convergence_threshold,
            refit_interval=refit_interval,
            surface_sigma=surface_sigma, surface_gamma=surface_gamma,
        )

        self._surface_weight = 1.0 - base_weight
        self._num_real_players: Optional[int] = None

        # Game data arrays (team game format)
        self._num_batches = 0
        self._batch_offsets: Optional[np.ndarray] = None
        self._batch_times: Optional[np.ndarray] = None
        self._game_t1_base: Optional[np.ndarray] = None
        self._game_t1_surf: Optional[np.ndarray] = None
        self._game_t2_base: Optional[np.ndarray] = None
        self._game_t2_surf: Optional[np.ndarray] = None
        self._game_scores: Optional[np.ndarray] = None

        # Sparse appearance structure
        self._num_appearances = 0
        self._app_offsets: Optional[np.ndarray] = None
        self._app_player: Optional[np.ndarray] = None
        self._app_prev: Optional[np.ndarray] = None
        self._app_next: Optional[np.ndarray] = None
        self._app_batch: Optional[np.ndarray] = None
        self._player_last_app: Optional[np.ndarray] = None

        # Sparse state arrays [num_appearances]
        self._state_forward_mu: Optional[np.ndarray] = None
        self._state_forward_sigma: Optional[np.ndarray] = None
        self._state_backward_mu: Optional[np.ndarray] = None
        self._state_backward_sigma: Optional[np.ndarray] = None
        self._state_likelihood_mu: Optional[np.ndarray] = None
        self._state_likelihood_sigma: Optional[np.ndarray] = None

        # Temp arrays [num_expanded] — reused per batch during sweeps
        self._tmp_fwd_mu: Optional[np.ndarray] = None
        self._tmp_fwd_sigma: Optional[np.ndarray] = None
        self._tmp_bwd_mu: Optional[np.ndarray] = None
        self._tmp_bwd_sigma: Optional[np.ndarray] = None
        self._tmp_lik_mu: Optional[np.ndarray] = None
        self._tmp_lik_sigma: Optional[np.ndarray] = None

        self._player_last_day: Optional[np.ndarray] = None

        # Accumulated data for incremental refit
        self._accum_p1: List[np.ndarray] = []
        self._accum_p2: List[np.ndarray] = []
        self._accum_scores: List[np.ndarray] = []
        self._accum_days: List[np.ndarray] = []
        self._accum_surfaces: List[np.ndarray] = []
        self._last_refit_day: Optional[int] = None

        # Metadata
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._player_names: Optional[Dict[int, str]] = None

        super().__init__(num_players=num_players)

    # =========================================================================
    # Player Space Management
    # =========================================================================

    def _get_num_expanded_players(self) -> int:
        return self._num_real_players * (1 + NUM_SURFACES)

    def _surface_player_id(self, player_id: int, surface: int) -> int:
        return self._num_real_players * (1 + surface) + player_id

    # =========================================================================
    # Initialization
    # =========================================================================

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        self._num_real_players = num_players
        num_expanded = self._get_num_expanded_players()

        ratings = np.full(num_expanded, self.DISPLAY_OFFSET, dtype=np.float32)
        rd = np.empty(num_expanded, dtype=np.float32)
        rd[:num_players] = np.float32(self.config.sigma * self.DISPLAY_SCALE)
        rd[num_players:] = np.float32(self.config.surface_sigma * self.DISPLAY_SCALE)

        return PlayerRatings(
            ratings=ratings, rd=rd,
            metadata={"system": "surface_ttt", "config": self.config},
        )

    def _ensure_temp_arrays(self):
        """Allocate or resize temp arrays for per-batch processing."""
        num_expanded = self._get_num_expanded_players()
        if self._tmp_fwd_mu is None or len(self._tmp_fwd_mu) < num_expanded:
            self._tmp_fwd_mu = np.zeros(num_expanded, dtype=np.float32)
            self._tmp_fwd_sigma = np.full(num_expanded, INF_SIGMA, dtype=np.float32)
            self._tmp_bwd_mu = np.zeros(num_expanded, dtype=np.float32)
            self._tmp_bwd_sigma = np.full(num_expanded, INF_SIGMA, dtype=np.float32)
            self._tmp_lik_mu = np.zeros(num_expanded, dtype=np.float32)
            self._tmp_lik_sigma = np.full(num_expanded, INF_SIGMA, dtype=np.float32)

    def _build_batch_structure(
        self, player1, player2, scores, days, surfaces,
    ) -> None:
        """Build batch + sparse appearance structures from game arrays."""
        n_games = len(player1)

        unique_days = np.unique(days)
        self._num_batches = len(unique_days)
        day_to_batch = {int(d): i for i, d in enumerate(unique_days)}

        sort_order = np.argsort(days)
        self._game_t1_base = player1[sort_order].astype(np.int32)
        self._game_t2_base = player2[sort_order].astype(np.int32)
        self._game_scores = scores[sort_order].astype(np.float32)

        sorted_surfaces = surfaces[sort_order]
        self._game_t1_surf = np.empty(n_games, dtype=np.int32)
        self._game_t2_surf = np.empty(n_games, dtype=np.int32)
        for i in range(n_games):
            surf = map_surface(sorted_surfaces[i])
            self._game_t1_surf[i] = self._surface_player_id(self._game_t1_base[i], surf)
            self._game_t2_surf[i] = self._surface_player_id(self._game_t2_base[i], surf)

        sorted_days = days[sort_order]

        self._batch_offsets = np.zeros(self._num_batches + 1, dtype=np.int64)
        self._batch_times = np.zeros(self._num_batches, dtype=np.float32)
        current_batch = -1
        for i, d in enumerate(sorted_days):
            batch_idx = day_to_batch[int(d)]
            if batch_idx != current_batch:
                self._batch_offsets[batch_idx] = i
                self._batch_times[batch_idx] = np.float32(d)
                current_batch = batch_idx
        self._batch_offsets[self._num_batches] = n_games

        # Build sparse appearance structure
        num_expanded = self._get_num_expanded_players()
        (self._app_offsets, self._app_player, self._app_prev,
         self._app_next, self._app_batch, self._player_last_app) = \
            build_team_appearance_structure(
                self._num_batches, self._batch_offsets,
                self._game_t1_base, self._game_t1_surf,
                self._game_t2_base, self._game_t2_surf,
                num_expanded,
            )
        self._num_appearances = len(self._app_player)

        # Allocate sparse state arrays
        na = self._num_appearances
        self._state_forward_mu = np.zeros(na, dtype=np.float32)
        self._state_forward_sigma = np.full(na, INF_SIGMA, dtype=np.float32)
        self._state_backward_mu = np.zeros(na, dtype=np.float32)
        self._state_backward_sigma = np.full(na, INF_SIGMA, dtype=np.float32)
        self._state_likelihood_mu = np.zeros(na, dtype=np.float32)
        self._state_likelihood_sigma = np.full(na, INF_SIGMA, dtype=np.float32)

        self._ensure_temp_arrays()

    # =========================================================================
    # Fitting
    # =========================================================================

    def fit(
        self, dataset, surfaces=None, end_day=None, player_names=None,
    ) -> "SurfaceTTT":
        """Fit SurfaceTTT on a dataset with surface information.

        Args:
            dataset: GameDataset to fit on.
            surfaces: int array of surface types (0=Hard, 1=Clay, 2=Grass).
            end_day: Last day to include (for backtesting).
            player_names: Optional player_id -> name mapping.
        """
        if surfaces is None:
            raise ValueError("surfaces array is required for SurfaceTTT")

        self._player_names = player_names

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is None or len(player1) == 0:
            self._fitted = True
            return self

        n_games = len(player1)
        if len(surfaces) != n_games:
            raise ValueError(f"surfaces length ({len(surfaces)}) must match games ({n_games})")

        # Create per-game day array
        days = np.empty(n_games, dtype=np.int32)
        for d in range(len(day_indices)):
            days[day_offsets[d]:day_offsets[d + 1]] = day_indices[d]

        # Initialize expanded player space
        required_players = max(self._num_players or 0, dataset.num_players)
        if self._num_players is None or self._num_players < required_players:
            self._num_players = required_players
            self._num_real_players = required_players
            self._ratings = self._initialize_ratings(self._num_players)

        num_expanded = self._get_num_expanded_players()

        # Build batch + sparse structures
        self._build_batch_structure(player1, player2, scores, days, surfaces)

        # Initial forward pass
        sparse_initial_forward_pass(
            self._num_batches, self._batch_offsets, self._batch_times,
            self._game_t1_base, self._game_t1_surf,
            self._game_t2_base, self._game_t2_surf, self._game_scores,
            num_expanded, self._num_real_players,
            self._state_forward_mu, self._state_forward_sigma,
            self._state_backward_mu, self._state_backward_sigma,
            self._state_likelihood_mu, self._state_likelihood_sigma,
            self._app_offsets, self._app_player, self._app_prev, self._app_batch,
            self._tmp_fwd_mu, self._tmp_fwd_sigma,
            self._tmp_bwd_mu, self._tmp_bwd_sigma,
            self._tmp_lik_mu, self._tmp_lik_sigma,
            np.float32(self.config.mu), np.float32(self.config.sigma),
            np.float32(self.config.surface_sigma),
            np.float32(self.config.base_weight), np.float32(self._surface_weight),
            np.float32(self.config.beta),
            np.float32(self.config.gamma), np.float32(self.config.surface_gamma),
            0,  # start_batch
        )

        # Convergence
        self._num_iterations = sparse_run_convergence(
            self._num_batches, self._batch_offsets, self._batch_times,
            self._game_t1_base, self._game_t1_surf,
            self._game_t2_base, self._game_t2_surf, self._game_scores,
            num_expanded, self._num_real_players,
            self._state_forward_mu, self._state_forward_sigma,
            self._state_backward_mu, self._state_backward_sigma,
            self._state_likelihood_mu, self._state_likelihood_sigma,
            self._app_offsets, self._app_player, self._app_prev,
            self._app_next, self._app_batch,
            self._tmp_fwd_mu, self._tmp_fwd_sigma,
            self._tmp_bwd_mu, self._tmp_bwd_sigma,
            self._tmp_lik_mu, self._tmp_lik_sigma,
            np.float32(self.config.mu), np.float32(self.config.sigma),
            np.float32(self.config.surface_sigma),
            np.float32(self.config.base_weight), np.float32(self._surface_weight),
            np.float32(self.config.beta),
            np.float32(self.config.gamma), np.float32(self.config.surface_gamma),
            self.config.max_iterations,
            np.float32(self.config.convergence_threshold),
        )

        # Extract ratings
        ratings = np.empty(num_expanded, dtype=np.float32)
        rd = np.empty(num_expanded, dtype=np.float32)
        sparse_extract_final_ratings(
            num_expanded, self._num_real_players,
            self._state_forward_mu, self._state_forward_sigma,
            self._state_backward_mu, self._state_backward_sigma,
            self._state_likelihood_mu, self._state_likelihood_sigma,
            self._player_last_app, ratings, rd,
            np.float32(self.config.mu), np.float32(self.config.sigma),
            np.float32(self.config.surface_sigma),
            np.float32(self.DISPLAY_SCALE), np.float32(self.DISPLAY_OFFSET),
        )

        self._ratings = PlayerRatings(
            ratings=ratings, rd=rd,
            metadata={"system": "surface_ttt", "config": self.config},
        )

        self._num_games_fitted = n_games
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        self._compute_player_last_day()

        # Store accumulated data for incremental updates
        self._accum_p1 = [player1.copy()]
        self._accum_p2 = [player2.copy()]
        self._accum_scores = [scores.copy()]
        self._accum_days = [days.copy()]
        self._accum_surfaces = [surfaces[:n_games].copy()]
        self._last_refit_day = self._current_day

        return self

    # =========================================================================
    # Incremental Update
    # =========================================================================

    def update(self, batch, surfaces=None) -> "SurfaceTTT":
        """Incrementally update with new games.

        Accumulates data and periodically refits using warm-started
        sparse message passing.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before updating. Call fit() first.")
        if self.config.refit_interval <= 0:
            return self

        n_games = len(batch)
        if surfaces is None:
            surfaces = np.zeros(n_games, dtype=np.int32)

        days_array = np.full(n_games, batch.day, dtype=np.int32)

        self._accum_p1.append(batch.player1.copy())
        self._accum_p2.append(batch.player2.copy())
        self._accum_scores.append(batch.scores.copy())
        self._accum_days.append(days_array)
        self._accum_surfaces.append(np.asarray(surfaces, dtype=np.int32).copy())

        if self._last_refit_day is None:
            self._last_refit_day = batch.day

        if batch.day - self._last_refit_day >= self.config.refit_interval:
            self._refit_from_accumulated()
            self._last_refit_day = batch.day

        self._current_day = batch.day

        # Update player_last_day for time-aware predictions
        if self._player_last_day is not None:
            for i in range(n_games):
                p1, p2 = int(batch.player1[i]), int(batch.player2[i])
                surf = map_surface(int(surfaces[i]))
                self._player_last_day[p1] = batch.day
                self._player_last_day[p2] = batch.day
                s1 = self._surface_player_id(p1, surf)
                s2 = self._surface_player_id(p2, surf)
                if s1 < len(self._player_last_day):
                    self._player_last_day[s1] = batch.day
                if s2 < len(self._player_last_day):
                    self._player_last_day[s2] = batch.day

        return self

    def _refit_from_accumulated(self) -> None:
        """Refit on all accumulated data."""
        if not self._accum_p1:
            return

        player1 = np.concatenate(self._accum_p1)
        player2 = np.concatenate(self._accum_p2)
        scores = np.concatenate(self._accum_scores)
        days = np.concatenate(self._accum_days)
        surfaces = np.concatenate(self._accum_surfaces)

        # Expand player space if needed
        max_player = max(int(player1.max()), int(player2.max()))
        required = max_player + 1
        if self._num_players is None or required > self._num_players:
            self._num_players = required
            self._num_real_players = required
            self._ratings = self._initialize_ratings(self._num_players)

        num_expanded = self._get_num_expanded_players()

        # Save old appearance count for warm-start
        old_num_appearances = self._num_appearances

        # Rebuild sparse structures (this is cheap — O(games), not O(batches×players))
        self._build_batch_structure(player1, player2, scores, days, surfaces)

        # Forward pass on all batches (sparse — cost proportional to appearances)
        sparse_initial_forward_pass(
            self._num_batches, self._batch_offsets, self._batch_times,
            self._game_t1_base, self._game_t1_surf,
            self._game_t2_base, self._game_t2_surf, self._game_scores,
            num_expanded, self._num_real_players,
            self._state_forward_mu, self._state_forward_sigma,
            self._state_backward_mu, self._state_backward_sigma,
            self._state_likelihood_mu, self._state_likelihood_sigma,
            self._app_offsets, self._app_player, self._app_prev, self._app_batch,
            self._tmp_fwd_mu, self._tmp_fwd_sigma,
            self._tmp_bwd_mu, self._tmp_bwd_sigma,
            self._tmp_lik_mu, self._tmp_lik_sigma,
            np.float32(self.config.mu), np.float32(self.config.sigma),
            np.float32(self.config.surface_sigma),
            np.float32(self.config.base_weight), np.float32(self._surface_weight),
            np.float32(self.config.beta),
            np.float32(self.config.gamma), np.float32(self.config.surface_gamma),
            0,  # start from batch 0
        )

        # Limited convergence iterations for refit
        self._num_iterations = sparse_run_convergence(
            self._num_batches, self._batch_offsets, self._batch_times,
            self._game_t1_base, self._game_t1_surf,
            self._game_t2_base, self._game_t2_surf, self._game_scores,
            num_expanded, self._num_real_players,
            self._state_forward_mu, self._state_forward_sigma,
            self._state_backward_mu, self._state_backward_sigma,
            self._state_likelihood_mu, self._state_likelihood_sigma,
            self._app_offsets, self._app_player, self._app_prev,
            self._app_next, self._app_batch,
            self._tmp_fwd_mu, self._tmp_fwd_sigma,
            self._tmp_bwd_mu, self._tmp_bwd_sigma,
            self._tmp_lik_mu, self._tmp_lik_sigma,
            np.float32(self.config.mu), np.float32(self.config.sigma),
            np.float32(self.config.surface_sigma),
            np.float32(self.config.base_weight), np.float32(self._surface_weight),
            np.float32(self.config.beta),
            np.float32(self.config.gamma), np.float32(self.config.surface_gamma),
            self.config.refit_max_iterations,
            np.float32(self.config.convergence_threshold),
        )

        # Extract ratings
        ratings = np.empty(num_expanded, dtype=np.float32)
        rd = np.empty(num_expanded, dtype=np.float32)
        sparse_extract_final_ratings(
            num_expanded, self._num_real_players,
            self._state_forward_mu, self._state_forward_sigma,
            self._state_backward_mu, self._state_backward_sigma,
            self._state_likelihood_mu, self._state_likelihood_sigma,
            self._player_last_app, ratings, rd,
            np.float32(self.config.mu), np.float32(self.config.sigma),
            np.float32(self.config.surface_sigma),
            np.float32(self.DISPLAY_SCALE), np.float32(self.DISPLAY_OFFSET),
        )

        self._ratings = PlayerRatings(
            ratings=ratings, rd=rd,
            metadata={"system": "surface_ttt", "config": self.config},
        )
        self._num_games_fitted = len(player1)
        self._compute_player_last_day()

    def _compute_player_last_day(self) -> None:
        """Cache last active day per expanded player for time-aware predictions."""
        if self._player_last_app is not None and self._batch_times is not None:
            num_expanded = self._get_num_expanded_players()
            last_day = np.zeros(num_expanded, dtype=np.int32)
            for p in range(num_expanded):
                a = self._player_last_app[p]
                if a >= 0:
                    last_day[p] = int(self._batch_times[self._app_batch[a]])
            self._player_last_day = last_day

    # =========================================================================
    # Prediction
    # =========================================================================

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        surface: Union[int, np.ndarray, List[int]] = SURFACE_NON_CLAY,
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict probability that player1 beats player2 on given surface."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)):
            surf = map_surface(int(surface))
            return self._predict_single(int(player1), int(player2), surf, day)

        p1 = np.asarray(player1, dtype=np.int32)
        p2 = np.asarray(player2, dtype=np.int32)
        surf = np.asarray(surface, dtype=np.int32)

        if surf.ndim == 0:
            surf = np.full(len(p1), map_surface(int(surf)), dtype=np.int32)
        else:
            surf = np.array([map_surface(s) for s in surf], dtype=np.int32)

        t1_surf = np.array([self._surface_player_id(p, s) for p, s in zip(p1, surf)], dtype=np.int32)
        t2_surf = np.array([self._surface_player_id(p, s) for p, s in zip(p2, surf)], dtype=np.int32)

        if day is not None and self._player_last_day is not None:
            n = len(p1)
            result = np.empty(n, dtype=np.float64)
            scale = np.float32(self.DISPLAY_SCALE)
            offset = np.float32(self.DISPLAY_OFFSET)

            for i in range(n):
                _p1, _p2 = int(p1[i]), int(p2[i])
                _t1s, _t2s = int(t1_surf[i]), int(t2_surf[i])
                result[i] = predict_team_match_at_day(
                    (self._ratings.ratings[_p1] - offset) / scale,
                    self._ratings.rd[_p1] / scale,
                    (self._ratings.ratings[_t1s] - offset) / scale,
                    self._ratings.rd[_t1s] / scale,
                    (self._ratings.ratings[_p2] - offset) / scale,
                    self._ratings.rd[_p2] / scale,
                    (self._ratings.ratings[_t2s] - offset) / scale,
                    self._ratings.rd[_t2s] / scale,
                    np.float32(self.config.base_weight),
                    np.float32(self._surface_weight),
                    np.float32(self.config.beta),
                    max(0, day - self._player_last_day[_p1]),
                    max(0, day - self._player_last_day[_t1s]),
                    max(0, day - self._player_last_day[_p2]),
                    max(0, day - self._player_last_day[_t2s]),
                    np.float32(self.config.gamma),
                    np.float32(self.config.surface_gamma),
                )
            return result

        return predict_proba_batch(
            p1, t1_surf, p2, t2_surf,
            self._ratings.ratings, self._ratings.rd,
            np.float32(self.config.base_weight),
            np.float32(self._surface_weight),
            np.float32(self.config.beta),
            np.float32(self.DISPLAY_SCALE),
            np.float32(self.DISPLAY_OFFSET),
        )

    def _predict_single(self, p1, p2, surface, day=None):
        t1_surf_id = self._surface_player_id(p1, surface)
        t2_surf_id = self._surface_player_id(p2, surface)
        scale = np.float32(self.DISPLAY_SCALE)
        offset = np.float32(self.DISPLAY_OFFSET)

        t1bm = (self._ratings.ratings[p1] - offset) / scale
        t1bs = self._ratings.rd[p1] / scale
        t1sm = (self._ratings.ratings[t1_surf_id] - offset) / scale
        t1ss = self._ratings.rd[t1_surf_id] / scale
        t2bm = (self._ratings.ratings[p2] - offset) / scale
        t2bs = self._ratings.rd[p2] / scale
        t2sm = (self._ratings.ratings[t2_surf_id] - offset) / scale
        t2ss = self._ratings.rd[t2_surf_id] / scale

        if day is not None and self._player_last_day is not None:
            return predict_team_match_at_day(
                t1bm, t1bs, t1sm, t1ss, t2bm, t2bs, t2sm, t2ss,
                np.float32(self.config.base_weight),
                np.float32(self._surface_weight),
                np.float32(self.config.beta),
                max(0, day - self._player_last_day[p1]),
                max(0, day - self._player_last_day[t1_surf_id]),
                max(0, day - self._player_last_day[p2]),
                max(0, day - self._player_last_day[t2_surf_id]),
                np.float32(self.config.gamma),
                np.float32(self.config.surface_gamma),
            )

        return predict_team_match(
            t1bm, t1bs, t1sm, t1ss, t2bm, t2bs, t2sm, t2ss,
            np.float32(self.config.base_weight),
            np.float32(self._surface_weight),
            np.float32(self.config.beta),
        )

    # =========================================================================
    # Housekeeping
    # =========================================================================

    def _update_ratings(self, batch, ratings):
        pass

    def reset(self) -> "SurfaceTTT":
        self._num_batches = 0
        self._num_appearances = 0
        self._batch_offsets = None
        self._batch_times = None
        self._game_t1_base = None
        self._game_t1_surf = None
        self._game_t2_base = None
        self._game_t2_surf = None
        self._game_scores = None
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
        self._tmp_fwd_mu = None
        self._tmp_fwd_sigma = None
        self._tmp_bwd_mu = None
        self._tmp_bwd_sigma = None
        self._tmp_lik_mu = None
        self._tmp_lik_sigma = None
        self._player_last_day = None
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._num_real_players = None
        self._accum_p1 = []
        self._accum_p2 = []
        self._accum_scores = []
        self._accum_days = []
        self._accum_surfaces = []
        self._last_refit_day = None
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_real_players or "?"
        apps = self._num_appearances or 0
        return (
            f"SurfaceTTT(base_weight={self.config.base_weight:.2f}, "
            f"sigma={self.config.sigma:.2f}, "
            f"surface_sigma={self.config.surface_sigma:.2f}, "
            f"players={players}, appearances={apps:,}, {status})"
        )
