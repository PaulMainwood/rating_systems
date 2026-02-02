"""
Surface-specific TrueSkill Through Time (SurfaceTTT).

Models player skill using weighted teams:
    Team = [Player_base, Player_surface]
    Team_skill = w_base × base_skill + w_surf × surface_skill

Each match becomes a team game:
    Team1 = [P1_base, P1_surface] vs Team2 = [P2_base, P2_surface]

This uses a single TTT fit on an expanded player space (3N players for N real
players with 2 surfaces: Clay and Non-Clay).

Surface Model:
    - Non-Clay (surface=0): Hard and Grass courts
    - Clay (surface=1): Clay courts

Memory optimized with float32 arrays (~12GB vs ~24GB with float64).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ._team_numba_core import (
    initial_forward_pass,
    run_convergence,
    extract_final_ratings,
    predict_proba_batch,
    predict_team_match,
    INF_SIGMA,
)


# =============================================================================
# Surface Constants
# =============================================================================

# Input surface values (what users pass from datasets)
SURFACE_HARD = 0
SURFACE_CLAY = 1
SURFACE_GRASS = 2

# Internal surface indices (what the model uses)
SURFACE_NON_CLAY = 0  # Maps Hard and Grass
SURFACE_CLAY_INTERNAL = 1

NUM_SURFACES = 2  # Only 2 internal surfaces: Non-Clay and Clay

SURFACE_NAMES = ["Non-Clay", "Clay"]

# Mapping from input (3-surface) to internal (2-surface)
# Hard(0) -> Non-Clay(0), Clay(1) -> Clay(1), Grass(2) -> Non-Clay(0)
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

    # Weighting
    base_weight: float = 0.9  # Weight for base skill (surface_weight = 1 - base_weight)

    # Base player parameters
    mu: float = 0.0           # Prior mean skill (internal scale)
    sigma: float = 6.0        # Prior skill std dev for base players
    gamma: float = 0.03       # Skill drift per time unit for base players

    # Surface component parameters
    surface_sigma: float = 3.0   # Prior std dev for surface adjustments
    surface_gamma: float = 0.01  # Skill drift for surface components

    # Performance noise
    beta: float = 1.0

    # Convergence
    max_iterations: int = 30
    refit_max_iterations: int = 2
    convergence_threshold: float = 1e-6
    refit_interval: int = 0


# =============================================================================
# Main Class
# =============================================================================

class SurfaceTTT(RatingSystem):
    """
    Surface-specific TrueSkill Through Time rating system.

    Uses weighted team games where each player is modeled as a team of
    [base_player, surface_player]. A single TTT is fit on the expanded
    player space (3N players for N real players and 2 surfaces).

    Player Space Layout (for N real players):
        [0, N):    Base players
        [N, 2N):   Non-Clay surface components
        [2N, 3N):  Clay surface components

    Example for a match on Clay between players 5 and 12:
        Team 1: [5, 2N+5]  (player 5's base + clay component)
        Team 2: [12, 2N+12] (player 12's base + clay component)

    Memory Usage:
        ~12GB for typical dataset (vs ~24GB with float64)
        State arrays: num_batches × 3N players × 6 arrays × 4 bytes

    Parameters:
        base_weight: Weight for base skill (default 0.9)
        sigma: Prior skill std dev for base players (default 6.0)
        surface_sigma: Prior std dev for surface components (default 3.0)
        beta: Performance variability (default 1.0)
        gamma: Base skill drift rate (default 0.03)
        surface_gamma: Surface skill drift rate (default 0.01)
    """

    system_type = RatingSystemType.BATCH

    # Display scale: internal 0 = display 1500
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
        refit_interval: int = 0,
        surface_sigma: float = 3.0,
        surface_gamma: float = 0.01,
        num_players: Optional[int] = None,
    ):
        self.config = SurfaceTTTConfig(
            base_weight=base_weight,
            mu=mu,
            sigma=sigma,
            beta=beta,
            gamma=gamma,
            max_iterations=max_iterations,
            refit_max_iterations=refit_max_iterations,
            convergence_threshold=convergence_threshold,
            refit_interval=refit_interval,
            surface_sigma=surface_sigma,
            surface_gamma=surface_gamma,
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

        # State arrays (float32 for memory efficiency)
        self._state_forward_mu: Optional[np.ndarray] = None
        self._state_forward_sigma: Optional[np.ndarray] = None
        self._state_backward_mu: Optional[np.ndarray] = None
        self._state_backward_sigma: Optional[np.ndarray] = None
        self._state_likelihood_mu: Optional[np.ndarray] = None
        self._state_likelihood_sigma: Optional[np.ndarray] = None

        # Agent state
        self._agent_message_mu: Optional[np.ndarray] = None
        self._agent_message_sigma: Optional[np.ndarray] = None
        self._agent_last_time: Optional[np.ndarray] = None
        self._player_last_batch: Optional[np.ndarray] = None

        # Metadata
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._player_names: Optional[Dict[int, str]] = None

        super().__init__(num_players=num_players)

    # =========================================================================
    # Player Space Management
    # =========================================================================

    def _get_num_expanded_players(self) -> int:
        """Total players in expanded space: N base + 2N surface = 3N."""
        return self._num_real_players * (1 + NUM_SURFACES)

    def _surface_player_id(self, player_id: int, surface: int) -> int:
        """
        Get surface component ID for a player.

        Layout: [0,N) base, [N,2N) non-clay, [2N,3N) clay
        """
        return self._num_real_players * (1 + surface) + player_id

    def _is_surface_player(self, player_id: int) -> bool:
        """Check if player ID is a surface component (not base)."""
        return player_id >= self._num_real_players

    # =========================================================================
    # Initialization
    # =========================================================================

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Initialize ratings for expanded player space."""
        self._num_real_players = num_players
        num_expanded = self._get_num_expanded_players()

        ratings = np.full(num_expanded, self.DISPLAY_OFFSET, dtype=np.float32)
        rd = np.empty(num_expanded, dtype=np.float32)

        # Base players: [0, N) - larger prior uncertainty
        rd[:num_players] = np.float32(self.config.sigma * self.DISPLAY_SCALE)

        # Surface components: [N, 3N) - smaller prior uncertainty
        rd[num_players:] = np.float32(self.config.surface_sigma * self.DISPLAY_SCALE)

        return PlayerRatings(
            ratings=ratings,
            rd=rd,
            metadata={"system": "surface_ttt", "config": self.config},
        )

    def _build_batch_structure(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        days: np.ndarray,
        surfaces: np.ndarray,
    ) -> None:
        """
        Build batch data structures from game arrays.

        Transforms each game into a team game with 4 player IDs:
            [t1_base, t1_surf, t2_base, t2_surf]
        """
        n_games = len(player1)

        # Get unique days and create batch mapping
        unique_days = np.unique(days)
        self._num_batches = len(unique_days)
        day_to_batch = {int(d): i for i, d in enumerate(unique_days)}

        # Sort games by day
        sort_order = np.argsort(days)

        # Create team game arrays
        self._game_t1_base = player1[sort_order].astype(np.int32)
        self._game_t2_base = player2[sort_order].astype(np.int32)
        self._game_scores = scores[sort_order].astype(np.float32)

        sorted_surfaces = surfaces[sort_order]

        # Compute surface component IDs (mapping 3-surface to 2-surface)
        self._game_t1_surf = np.empty(n_games, dtype=np.int32)
        self._game_t2_surf = np.empty(n_games, dtype=np.int32)

        for i in range(n_games):
            surf = map_surface(sorted_surfaces[i])
            self._game_t1_surf[i] = self._surface_player_id(self._game_t1_base[i], surf)
            self._game_t2_surf[i] = self._surface_player_id(self._game_t2_base[i], surf)

        sorted_days = days[sort_order]

        # Build batch offsets and times
        self._batch_offsets = np.zeros(self._num_batches + 1, dtype=np.int32)
        self._batch_times = np.zeros(self._num_batches, dtype=np.float32)

        current_batch = -1
        for i, d in enumerate(sorted_days):
            batch_idx = day_to_batch[int(d)]
            if batch_idx != current_batch:
                self._batch_offsets[batch_idx] = i
                self._batch_times[batch_idx] = np.float32(d)
                current_batch = batch_idx
        self._batch_offsets[self._num_batches] = n_games

        # Initialize state arrays (float32 for ~50% memory reduction)
        num_expanded = self._get_num_expanded_players()
        total_state = self._num_batches * num_expanded

        self._state_forward_mu = np.zeros(total_state, dtype=np.float32)
        self._state_forward_sigma = np.full(total_state, INF_SIGMA, dtype=np.float32)
        self._state_backward_mu = np.zeros(total_state, dtype=np.float32)
        self._state_backward_sigma = np.full(total_state, INF_SIGMA, dtype=np.float32)
        self._state_likelihood_mu = np.zeros(total_state, dtype=np.float32)
        self._state_likelihood_sigma = np.full(total_state, INF_SIGMA, dtype=np.float32)

        # Agent state
        self._agent_message_mu = np.zeros(num_expanded, dtype=np.float32)
        self._agent_message_sigma = np.full(num_expanded, INF_SIGMA, dtype=np.float32)
        self._agent_last_time = np.full(num_expanded, np.float32(-1e10), dtype=np.float32)
        self._player_last_batch = np.full(num_expanded, -1, dtype=np.int32)

    # =========================================================================
    # Fitting
    # =========================================================================

    def fit(
        self,
        dataset: "GameDataset",
        surfaces: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "SurfaceTTT":
        """
        Fit SurfaceTTT on a dataset with surface information.

        Args:
            dataset: Game dataset to fit on
            surfaces: Array of surface types (0=Hard, 1=Clay, 2=Grass) per game.
                      Hard and Grass are mapped to Non-Clay internally.
            end_day: Last day to include (for backtesting)
            player_names: Optional mapping of player_id -> name

        Returns:
            self (for method chaining)
        """
        if surfaces is None:
            raise ValueError("surfaces array is required for SurfaceTTT")

        self._player_names = player_names

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        # Get game arrays
        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is None or len(player1) == 0:
            self._fitted = True
            return self

        n_games = len(player1)
        if len(surfaces) != n_games:
            raise ValueError(f"surfaces length ({len(surfaces)}) must match games ({n_games})")

        # Create per-game day array
        days = np.empty(n_games, dtype=np.int32)
        n_days = len(day_indices)
        for d in range(n_days):
            start = day_offsets[d]
            end = day_offsets[d + 1]
            days[start:end] = day_indices[d]

        # Initialize expanded player space
        required_players = max(self._num_players or 0, dataset.num_players)
        if self._num_players is None or self._num_players < required_players:
            self._num_players = required_players
            self._num_real_players = required_players
            self._ratings = self._initialize_ratings(self._num_players)

        num_expanded = self._get_num_expanded_players()

        # Build batch structure with team game data
        self._build_batch_structure(player1, player2, scores, days, surfaces)

        # Initial forward pass
        initial_forward_pass(
            self._num_batches,
            self._batch_offsets,
            self._batch_times,
            self._game_t1_base,
            self._game_t1_surf,
            self._game_t2_base,
            self._game_t2_surf,
            self._game_scores,
            num_expanded,
            self._num_real_players,
            self._state_forward_mu,
            self._state_forward_sigma,
            self._state_backward_mu,
            self._state_backward_sigma,
            self._state_likelihood_mu,
            self._state_likelihood_sigma,
            self._agent_message_mu,
            self._agent_message_sigma,
            self._agent_last_time,
            np.float32(self.config.mu),
            np.float32(self.config.sigma),
            np.float32(self.config.surface_sigma),
            np.float32(self.config.base_weight),
            np.float32(self._surface_weight),
            np.float32(self.config.beta),
            np.float32(self.config.gamma),
            np.float32(self.config.surface_gamma),
        )

        # Run convergence
        self._num_iterations = run_convergence(
            self._num_batches,
            self._batch_offsets,
            self._batch_times,
            self._game_t1_base,
            self._game_t1_surf,
            self._game_t2_base,
            self._game_t2_surf,
            self._game_scores,
            num_expanded,
            self._num_real_players,
            self._state_forward_mu,
            self._state_forward_sigma,
            self._state_backward_mu,
            self._state_backward_sigma,
            self._state_likelihood_mu,
            self._state_likelihood_sigma,
            self._agent_message_mu,
            self._agent_message_sigma,
            np.float32(self.config.base_weight),
            np.float32(self._surface_weight),
            np.float32(self.config.beta),
            np.float32(self.config.gamma),
            np.float32(self.config.surface_gamma),
            self.config.max_iterations,
            np.float32(self.config.convergence_threshold),
        )

        # Extract final ratings
        ratings = np.empty(num_expanded, dtype=np.float32)
        rd = np.empty(num_expanded, dtype=np.float32)

        extract_final_ratings(
            self._num_batches,
            self._batch_offsets,
            self._game_t1_base,
            self._game_t1_surf,
            self._game_t2_base,
            self._game_t2_surf,
            num_expanded,
            self._num_real_players,
            self._state_forward_mu,
            self._state_forward_sigma,
            self._state_backward_mu,
            self._state_backward_sigma,
            self._state_likelihood_mu,
            self._state_likelihood_sigma,
            self._player_last_batch,
            ratings,
            rd,
            np.float32(self.config.mu),
            np.float32(self.config.sigma),
            np.float32(self.config.surface_sigma),
            np.float32(self.DISPLAY_SCALE),
            np.float32(self.DISPLAY_OFFSET),
        )

        self._ratings = PlayerRatings(
            ratings=ratings,
            rd=rd,
            metadata={"system": "surface_ttt", "config": self.config},
        )

        self._num_games_fitted = n_games
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None

        return self

    # =========================================================================
    # Prediction
    # =========================================================================

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        surface: Union[int, np.ndarray, List[int]] = SURFACE_NON_CLAY,
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2 on given surface.

        Args:
            player1: Single player ID or array of player IDs
            player2: Single player ID or array of player IDs
            surface: Surface type (0=Non-Clay, 1=Clay) or array.
                     Original 3-surface values (0=Hard, 1=Clay, 2=Grass) are
                     automatically mapped.

        Returns:
            Single probability or array of probabilities
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle single prediction
        if isinstance(player1, (int, np.integer)):
            surf = map_surface(int(surface))
            return self._predict_single(int(player1), int(player2), surf)

        # Batch prediction
        p1 = np.asarray(player1, dtype=np.int32)
        p2 = np.asarray(player2, dtype=np.int32)
        surf = np.asarray(surface, dtype=np.int32)

        if surf.ndim == 0:
            surf = np.full(len(p1), map_surface(int(surf)), dtype=np.int32)
        else:
            surf = np.array([map_surface(s) for s in surf], dtype=np.int32)

        # Build team arrays
        t1_surf = np.array([self._surface_player_id(p, s) for p, s in zip(p1, surf)], dtype=np.int32)
        t2_surf = np.array([self._surface_player_id(p, s) for p, s in zip(p2, surf)], dtype=np.int32)

        return predict_proba_batch(
            p1, t1_surf, p2, t2_surf,
            self._ratings.ratings, self._ratings.rd,
            np.float32(self.config.base_weight),
            np.float32(self._surface_weight),
            np.float32(self.config.beta),
            np.float32(self.DISPLAY_SCALE),
            np.float32(self.DISPLAY_OFFSET),
        )

    def _predict_single(self, p1: int, p2: int, surface: int) -> float:
        """Predict single game outcome."""
        t1_surf_id = self._surface_player_id(p1, surface)
        t2_surf_id = self._surface_player_id(p2, surface)

        # Convert to internal scale
        scale = np.float32(self.DISPLAY_SCALE)
        offset = np.float32(self.DISPLAY_OFFSET)

        t1_base_mu = (self._ratings.ratings[p1] - offset) / scale
        t1_base_sigma = self._ratings.rd[p1] / scale
        t1_surf_mu = (self._ratings.ratings[t1_surf_id] - offset) / scale
        t1_surf_sigma = self._ratings.rd[t1_surf_id] / scale

        t2_base_mu = (self._ratings.ratings[p2] - offset) / scale
        t2_base_sigma = self._ratings.rd[p2] / scale
        t2_surf_mu = (self._ratings.ratings[t2_surf_id] - offset) / scale
        t2_surf_sigma = self._ratings.rd[t2_surf_id] / scale

        return predict_team_match(
            t1_base_mu, t1_base_sigma,
            t1_surf_mu, t1_surf_sigma,
            t2_base_mu, t2_base_sigma,
            t2_surf_mu, t2_surf_sigma,
            np.float32(self.config.base_weight),
            np.float32(self._surface_weight),
            np.float32(self.config.beta),
        )

    # =========================================================================
    # Rating Queries
    # =========================================================================

    def get_player_rating(
        self,
        player_id: int,
        surface: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Get a player's rating.

        Args:
            player_id: Player ID
            surface: If None, returns base rating. Otherwise returns
                    effective rating for that surface (0=Non-Clay, 1=Clay).

        Returns:
            (rating, rd) tuple in display scale
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if surface is None:
            return (float(self._ratings.ratings[player_id]),
                    float(self._ratings.rd[player_id]))

        # Compute effective rating for surface
        surf = map_surface(surface)
        base_rating = self._ratings.ratings[player_id]
        surf_id = self._surface_player_id(player_id, surf)
        surf_rating = self._ratings.ratings[surf_id]

        # Weighted combination
        base_adj = base_rating - self.DISPLAY_OFFSET
        surf_adj = surf_rating - self.DISPLAY_OFFSET
        effective = self.DISPLAY_OFFSET + self.config.base_weight * base_adj + self._surface_weight * surf_adj

        return float(effective), float(self._ratings.rd[player_id])

    def get_surface_adjustment(self, player_id: int, surface: int) -> Tuple[float, float]:
        """
        Get a player's surface-specific adjustment.

        Args:
            player_id: Player ID
            surface: Surface type (0=Non-Clay, 1=Clay)

        Returns:
            (adjustment, rd) where adjustment is in display scale points
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        surf = map_surface(surface)
        surf_id = self._surface_player_id(player_id, surf)
        adjustment = self._ratings.ratings[surf_id] - self.DISPLAY_OFFSET
        rd = self._ratings.rd[surf_id]

        return float(adjustment), float(rd)

    def get_fitted_ratings(self) -> "FittedSurfaceTTTRatings":
        """Get a queryable fitted ratings object."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return FittedSurfaceTTTRatings(
            ratings=self._ratings.ratings.copy(),
            rd=self._ratings.rd.copy(),
            num_real_players=self._num_real_players,
            base_weight=self.config.base_weight,
            surface_weight=self._surface_weight,
            display_scale=self.DISPLAY_SCALE,
            display_offset=self.DISPLAY_OFFSET,
            player_names=self._player_names,
        )

    # =========================================================================
    # Housekeeping
    # =========================================================================

    def _update_ratings(self, batch: "GameBatch", ratings: PlayerRatings) -> None:
        """Update ratings - for SurfaceTTT this is handled via refit."""
        pass

    def reset(self) -> "SurfaceTTT":
        """Reset the rating system to initial state."""
        self._num_batches = 0
        self._batch_offsets = None
        self._batch_times = None
        self._game_t1_base = None
        self._game_t1_surf = None
        self._game_t2_base = None
        self._game_t2_surf = None
        self._game_scores = None
        self._state_forward_mu = None
        self._state_forward_sigma = None
        self._state_backward_mu = None
        self._state_backward_sigma = None
        self._state_likelihood_mu = None
        self._state_likelihood_sigma = None
        self._agent_message_mu = None
        self._agent_message_sigma = None
        self._agent_last_time = None
        self._player_last_batch = None
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._num_real_players = None
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_real_players or "?"
        return (
            f"SurfaceTTT(base_weight={self.config.base_weight:.2f}, "
            f"sigma={self.config.sigma:.2f}, "
            f"surface_sigma={self.config.surface_sigma:.2f}, "
            f"players={players}, {status})"
        )


# =============================================================================
# Fitted Ratings Helper
# =============================================================================

class FittedSurfaceTTTRatings:
    """Queryable fitted ratings from SurfaceTTT."""

    def __init__(
        self,
        ratings: np.ndarray,
        rd: np.ndarray,
        num_real_players: int,
        base_weight: float,
        surface_weight: float,
        display_scale: float,
        display_offset: float,
        player_names: Optional[Dict[int, str]] = None,
    ):
        self.ratings = ratings[:num_real_players].copy()
        self.rd = rd[:num_real_players].copy()
        self._full_ratings = ratings
        self._full_rd = rd
        self._num_real_players = num_real_players
        self._base_weight = base_weight
        self._surface_weight = surface_weight
        self._display_scale = display_scale
        self._display_offset = display_offset
        self._player_names = player_names or {}

    def get_effective_rating(self, player_id: int, surface: int) -> float:
        """Get effective rating for player on surface (0=Non-Clay, 1=Clay)."""
        surf = map_surface(surface)
        base_adj = self._full_ratings[player_id] - self._display_offset
        surf_offset = self._num_real_players * (1 + surf)
        surf_adj = self._full_ratings[surf_offset + player_id] - self._display_offset
        return float(self._display_offset + self._base_weight * base_adj + self._surface_weight * surf_adj)

    def get_surface_adjustment(self, player_id: int, surface: int) -> float:
        """Get surface adjustment for player (0=Non-Clay, 1=Clay)."""
        surf = map_surface(surface)
        surf_offset = self._num_real_players * (1 + surf)
        return float(self._full_ratings[surf_offset + player_id] - self._display_offset)

    def top_by_surface(self, surface: int, n: int = 10) -> List[Tuple[int, str, float, float]]:
        """
        Get top players by effective rating on a surface.

        Returns list of (player_id, name, effective_rating, surface_adjustment)
        """
        effective = np.array([
            self.get_effective_rating(p, surface)
            for p in range(self._num_real_players)
        ])
        top_indices = np.argsort(effective)[::-1][:n]

        results = []
        for idx in top_indices:
            name = self._player_names.get(idx, f"Player {idx}")
            eff = effective[idx]
            adj = self.get_surface_adjustment(idx, surface)
            results.append((int(idx), name, float(eff), float(adj)))
        return results

    def surface_specialists(self, surface: int, n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get players with highest surface adjustment (specialists).

        Returns list of (player_id, name, adjustment)
        """
        surf = map_surface(surface)
        surf_offset = self._num_real_players * (1 + surf)
        adjustments = self._full_ratings[surf_offset:surf_offset + self._num_real_players] - self._display_offset

        top_indices = np.argsort(adjustments)[::-1][:n]

        results = []
        for idx in top_indices:
            name = self._player_names.get(idx, f"Player {idx}")
            adj = adjustments[idx]
            results.append((int(idx), name, float(adj)))
        return results

    def to_dataframe(self) -> "pl.DataFrame":
        """Convert to a polars DataFrame with base ratings."""
        data = {
            "player_id": list(range(self._num_real_players)),
            "rating": [float(r) for r in self.ratings],
            "rd": [float(r) for r in self.rd],
        }

        if self._player_names:
            data["name"] = [self._player_names.get(i, f"Player {i}") for i in range(self._num_real_players)]

        return pl.DataFrame(data)
