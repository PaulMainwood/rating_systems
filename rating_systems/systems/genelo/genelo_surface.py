"""
GenEloSurface — Multivariate GenElo with correlated surface skills.

Implements the full model from Ingram (2021), Sections 2.5-2.6:
- Correlated surface skills (clay, grass, hard, indoor hard)
- Optional margin of victory
- Optional tournament level effects (Masters 1000, Grand Slam)
- Optional best-of-5 format modifier

Each player has a vector of n_skills ratings. A match on a given surface
updates all skill dimensions proportionally to the covariance structure.
For example, a win on clay updates the grass rating by ρ_{cg} × (grass_σ/clay_σ)
of the clay update.

The update uses the Sherman-Morrison identity to avoid matrix inversion,
making it efficient and Numba-compilable.

Reference:
    Ingram, M. (2021). How to extend Elo: a Bayesian perspective.
    Journal of Quantitative Analysis in Sports.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ._numba_surface import (
    NUM_SURFACES,
    NUM_TOURNAMENT_ADDITIONS,
    SURFACE_HARD,
    SURFACE_CLAY,
    SURFACE_GRASS,
    SURFACE_INDOOR_HARD,
    TOURNAMENT_250_500,
    TOURNAMENT_MASTERS,
    TOURNAMENT_SLAM,
    fit_all_days_surface,
    predict_single_surface,
    predict_proba_surface_batch,
    predict_proba_surface_batch_scalar,
)

SURFACE_NAMES = {
    SURFACE_HARD: "Hard",
    SURFACE_CLAY: "Clay",
    SURFACE_GRASS: "Grass",
    SURFACE_INDOOR_HARD: "Indoor Hard",
}

TOURNAMENT_NAMES = {
    TOURNAMENT_250_500: "250/500",
    TOURNAMENT_MASTERS: "Masters 1000",
    TOURNAMENT_SLAM: "Grand Slam",
}


def _build_cov_matrix(
    surface_sigmas: Tuple[float, ...],
    surface_correlations: Dict[Tuple[int, int], float],
    tournament_sigmas: Tuple[float, ...],
) -> np.ndarray:
    """
    Build the full covariance matrix Σ from marginal sigmas and correlations.

    The matrix is (n_skills × n_skills) where n_skills = 4 surfaces + 2 tournaments.
    Tournament additions are modelled as independent from surface skills
    and from each other (zeros in off-diagonal blocks).
    """
    n_surfaces = len(surface_sigmas)
    n_tournaments = len(tournament_sigmas)
    n_skills = n_surfaces + n_tournaments

    cov = np.zeros((n_skills, n_skills), dtype=np.float64)

    # Surface block: Σ[i,j] = ρ_ij × σ_i × σ_j
    for i in range(n_surfaces):
        cov[i, i] = surface_sigmas[i] ** 2
        for j in range(i + 1, n_surfaces):
            key = (i, j)
            rho = surface_correlations.get(key, surface_correlations.get((j, i), 0.0))
            cov[i, j] = rho * surface_sigmas[i] * surface_sigmas[j]
            cov[j, i] = cov[i, j]

    # Tournament block: diagonal only (independent)
    for t in range(n_tournaments):
        idx = n_surfaces + t
        cov[idx, idx] = tournament_sigmas[t] ** 2

    return cov


@dataclass
class GenEloSurfaceConfig:
    """Configuration for GenEloSurface.

    Default values are from the paper's parameter estimates (Tables 3 & 4,
    Section 3.1), fitted on 2010-2017 ATP men's tennis matches.

    Attributes:
        initial_rating: Starting rating for each surface skill.
        scale: Elo scale factor. b = log(10)/scale.
        surface_sigmas: Prior std devs per surface [hard, clay, grass, indoor_hard].
        surface_correlations: Pairwise correlations between surfaces.
            Keys are (i, j) tuples with i < j.
        tournament_sigmas: Prior std devs for tournament additions
            [masters_1000, grand_slam]. Set to (0, 0) to disable.
        c1: Margin slope. Set to None to disable margin updates.
        c2: Margin offset (winner's expected advantage).
        sigma_obs: Margin noise std for bo3 matches.
        sigma_obs_bo5: Margin noise std for bo5 matches.
        bo5_factor: Best-of-5 multiplier m (Eq. 32). Set to 0 to disable.
    """

    initial_rating: float = 1500.0
    scale: float = 400.0
    surface_sigmas: Tuple[float, ...] = (82.2, 90.6, 95.5, 86.5)
    surface_correlations: Dict[Tuple[int, int], float] = field(
        default_factory=lambda: {
            (0, 1): 0.72,  # hard-clay
            (0, 2): 0.82,  # hard-grass
            (0, 3): 0.86,  # hard-indoor_hard
            (1, 2): 0.41,  # clay-grass
            (1, 3): 0.65,  # clay-indoor_hard
            (2, 3): 0.75,  # grass-indoor_hard
        }
    )
    tournament_sigmas: Tuple[float, ...] = (0.0, 23.7)
    c1: Optional[float] = 0.000144
    c2: Optional[float] = 0.0998
    sigma_obs: Optional[float] = 0.087
    sigma_obs_bo5: Optional[float] = 0.071
    bo5_factor: float = 0.432


class GenEloSurface(RatingSystem):
    """
    GenEloSurface — Multivariate GenElo with correlated skills.

    Each player has ratings for each surface (and optionally each tournament
    level). A match on one surface updates all skills proportionally to the
    covariance structure.

    Examples:
        Full model (surface + margin + tournament + format):
            >>> model = GenEloSurface()
            >>> model.fit(dataset, surfaces=surface_arr, margins=margin_arr,
            ...           tournament_levels=tourn_arr, is_bo5=bo5_arr)

        Surface only (no margin):
            >>> model = GenEloSurface(c1=None)
            >>> model.fit(dataset, surfaces=surface_arr)

        Get surface-specific ratings:
            >>> ratings = model.get_surface_ratings(player_id)
            >>> print(ratings)
            {'Hard': 1623.4, 'Clay': 1589.2, 'Grass': 1601.8, ...}
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        scale: float = 400.0,
        surface_sigmas: Tuple[float, ...] = (82.2, 90.6, 95.5, 86.5),
        surface_correlations: Optional[Dict[Tuple[int, int], float]] = None,
        tournament_sigmas: Tuple[float, ...] = (0.0, 23.7),
        c1: Optional[float] = None,
        c2: Optional[float] = None,
        sigma_obs: Optional[float] = None,
        sigma_obs_bo5: Optional[float] = None,
        bo5_factor: float = 0.432,
        num_players: Optional[int] = None,
    ):
        if surface_correlations is None:
            surface_correlations = {
                (0, 1): 0.72, (0, 2): 0.82, (0, 3): 0.86,
                (1, 2): 0.41, (1, 3): 0.65, (2, 3): 0.75,
            }

        self.config = GenEloSurfaceConfig(
            initial_rating=initial_rating,
            scale=scale,
            surface_sigmas=surface_sigmas,
            surface_correlations=surface_correlations,
            tournament_sigmas=tournament_sigmas,
            c1=c1,
            c2=c2,
            sigma_obs=sigma_obs,
            sigma_obs_bo5=sigma_obs_bo5,
            bo5_factor=bo5_factor,
        )

        # Precompute constants
        self._b = np.log(10.0) / scale
        self._n_skills = len(surface_sigmas) + len(tournament_sigmas)
        self._cov_matrix = _build_cov_matrix(
            surface_sigmas, surface_correlations, tournament_sigmas
        )

        self._use_margin = c1 is not None and c2 is not None and sigma_obs is not None
        self._c1 = c1 if c1 is not None else 0.0
        self._c2 = c2 if c2 is not None else 0.0
        self._sigma_obs_sq = sigma_obs ** 2 if sigma_obs is not None else 1.0
        self._sigma_obs_bo5_sq = sigma_obs_bo5 ** 2 if sigma_obs_bo5 is not None else 1.0

        # Multi-dimensional skill ratings stored separately
        self._skill_ratings: Optional[np.ndarray] = None

        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial (scalar) ratings for interface compatibility."""
        # The "overall" rating is just the initial_rating
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "genelo_surface", "config": self.config},
        )

    def _initialize_skill_ratings(self, num_players: int) -> np.ndarray:
        """Create initial multi-dimensional skill ratings."""
        skills = np.full(
            (num_players, self._n_skills),
            self.config.initial_rating,
            dtype=np.float64,
        )
        # Tournament additions start at 0 (they're additive adjustments)
        for t in range(NUM_TOURNAMENT_ADDITIONS):
            skills[:, NUM_SURFACES + t] = 0.0
        return skills

    def reset(self) -> "GenEloSurface":
        """Reset ratings (including multi-dimensional skills) to initial state."""
        super().reset()
        if self._num_players is not None:
            self._skill_ratings = self._initialize_skill_ratings(self._num_players)
        self._num_games_fitted = 0
        return self

    def _sync_overall_ratings(self) -> None:
        """Update the scalar ratings array from skill ratings.

        Uses the hard court rating as the "overall" rating for compatibility
        (hard court is the most common surface).
        """
        if self._skill_ratings is not None and self._ratings is not None:
            self._ratings.ratings[:] = self._skill_ratings[:, SURFACE_HARD]

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Not used directly — use update() with surface kwargs instead."""
        raise NotImplementedError(
            "GenEloSurface requires surface data. Use update() with surfaces kwarg."
        )

    def update(
        self,
        batch: GameBatch,
        surfaces: Optional[np.ndarray] = None,
        tournament_levels: Optional[np.ndarray] = None,
        is_bo5: Optional[np.ndarray] = None,
        margins: Optional[np.ndarray] = None,
    ) -> "GenEloSurface":
        """
        Incrementally update ratings with a new batch of games.

        Args:
            batch: New games to process.
            surfaces: Surface ID per game (0=hard, 1=clay, 2=grass, 3=indoor_hard).
                Defaults to SURFACE_HARD if not provided.
            tournament_levels: Tournament level per game. Defaults to 0.
            is_bo5: Best-of-5 indicator per game. Defaults to 0.
            margins: Continuous margin per game. Ignored if model has no margin params.

        Returns:
            self (for method chaining).
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before updating. Call fit() first.")

        n = len(batch)

        if surfaces is None:
            surfaces = np.zeros(n, dtype=np.int64)
        else:
            surfaces = np.ascontiguousarray(surfaces, dtype=np.int64)

        if tournament_levels is None:
            tournament_levels = np.zeros(n, dtype=np.int64)
        else:
            tournament_levels = np.ascontiguousarray(tournament_levels, dtype=np.int64)

        if is_bo5 is None:
            is_bo5 = np.zeros(n, dtype=np.int64)
        else:
            is_bo5 = np.ascontiguousarray(is_bo5, dtype=np.int64)

        if margins is None:
            margins = np.zeros(n, dtype=np.float64)
        else:
            margins = np.ascontiguousarray(margins, dtype=np.float64)

        # Single-day offset: [0, n]
        day_offsets = np.array([0, n], dtype=np.int64)

        fit_all_days_surface(
            np.ascontiguousarray(batch.player1, dtype=np.int64),
            np.ascontiguousarray(batch.player2, dtype=np.int64),
            np.ascontiguousarray(batch.scores, dtype=np.float64),
            margins, surfaces, tournament_levels, is_bo5,
            day_offsets, self._skill_ratings, self._cov_matrix,
            self._b, self._c1, self._c2,
            self._sigma_obs_sq, self._sigma_obs_bo5_sq,
            self.config.bo5_factor, self._use_margin,
        )

        self._sync_overall_ratings()
        self._num_games_fitted += n
        self._current_day = batch.day
        return self

    def fit(
        self,
        dataset: GameDataset,
        surfaces: Optional[np.ndarray] = None,
        margins: Optional[np.ndarray] = None,
        tournament_levels: Optional[np.ndarray] = None,
        is_bo5: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "GenEloSurface":
        """
        Fit GenEloSurface on a dataset.

        Args:
            dataset: GameDataset to fit on.
            surfaces: Surface ID per game (0=hard, 1=clay, 2=grass, 3=indoor_hard).
                Required.
            margins: Optional continuous margin per game (from P1's perspective).
                Required if margin parameters are set.
            tournament_levels: Optional tournament level per game
                (0=250/500, 1=Masters, 2=Slam). Defaults to 0 for all.
            is_bo5: Optional best-of-5 indicator per game (0 or 1).
                Defaults to 0 for all.
            end_day: Optional last day to include.
            player_names: Optional mapping of player IDs to names.

        Returns:
            self (for method chaining).
        """
        self._player_names = player_names

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)
            self._skill_ratings = self._initialize_skill_ratings(self._num_players)

        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()
        if player1 is None or len(player1) == 0:
            self._fitted = True
            return self

        n_games = len(player1)

        # Validate and prepare surfaces
        if surfaces is None:
            raise ValueError("surfaces array is required for GenEloSurface.")
        surfaces = np.ascontiguousarray(surfaces, dtype=np.int64)
        if len(surfaces) != n_games:
            raise ValueError(
                f"surfaces length ({len(surfaces)}) must match games ({n_games})"
            )

        # Validate margins
        if self._use_margin:
            if margins is None:
                raise ValueError(
                    "Margin parameters are set but no margins array provided."
                )
            margins = np.ascontiguousarray(margins, dtype=np.float64)
            if len(margins) != n_games:
                raise ValueError(
                    f"margins length ({len(margins)}) must match games ({n_games})"
                )
        else:
            # Dummy array (won't be used)
            margins = np.zeros(n_games, dtype=np.float64)

        # Default tournament levels and bo5
        if tournament_levels is None:
            tournament_levels = np.zeros(n_games, dtype=np.int64)
        else:
            tournament_levels = np.ascontiguousarray(tournament_levels, dtype=np.int64)
            if len(tournament_levels) != n_games:
                raise ValueError(
                    f"tournament_levels length ({len(tournament_levels)}) "
                    f"must match games ({n_games})"
                )

        if is_bo5 is None:
            is_bo5 = np.zeros(n_games, dtype=np.int64)
        else:
            is_bo5 = np.ascontiguousarray(is_bo5, dtype=np.int64)
            if len(is_bo5) != n_games:
                raise ValueError(
                    f"is_bo5 length ({len(is_bo5)}) must match games ({n_games})"
                )

        # Run the Numba fit
        fit_all_days_surface(
            player1, player2, scores, margins,
            surfaces, tournament_levels, is_bo5,
            day_offsets, self._skill_ratings, self._cov_matrix,
            self._b, self._c1, self._c2,
            self._sigma_obs_sq, self._sigma_obs_bo5_sq,
            self.config.bo5_factor, self._use_margin,
        )

        # Sync overall ratings
        self._sync_overall_ratings()

        self._num_games_fitted = n_games
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        self._fitted = True
        return self

    def predict_proba(
        self,
        player1: Union[int, np.integer, np.ndarray],
        player2: Union[int, np.integer, np.ndarray],
        surface: int = SURFACE_HARD,
        tournament_level: int = TOURNAMENT_250_500,
        is_bo5: Union[bool, int] = False,
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """
        Predict P(player1 wins) on a given surface/tournament/format.

        Uses the approximate marginal likelihood accounting for the
        covariance structure.

        Args:
            player1: Player 1 ID(s).
            player2: Player 2 ID(s).
            surface: Surface ID (0=hard, 1=clay, 2=grass, 3=indoor_hard).
            tournament_level: Tournament level (0=250/500, 1=Masters, 2=Slam).
            is_bo5: Whether the match is best-of-5.
            day: Unused (for interface compatibility).

        Returns:
            Win probability (float or array).
        """
        if self._skill_ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        bo5_int = int(is_bo5)

        if isinstance(player1, (int, np.integer)):
            return predict_single_surface(
                self._skill_ratings,
                int(player1), int(player2),
                surface, tournament_level, bo5_int,
                self._cov_matrix, self._b, self.config.bo5_factor,
            )

        player1 = np.asarray(player1, dtype=np.int64)
        player2 = np.asarray(player2, dtype=np.int64)
        return predict_proba_surface_batch_scalar(
            player1, player2,
            surface, tournament_level, bo5_int,
            self._skill_ratings, self._cov_matrix,
            self._b, self.config.bo5_factor,
        )

    def predict_proba_with_covariates(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        surfaces: np.ndarray,
        tournament_levels: np.ndarray,
        is_bo5: np.ndarray,
    ) -> np.ndarray:
        """
        Predict win probabilities for matches with varying covariates.

        Use this when predicting a batch of matches on different surfaces
        or tournament levels.

        Args:
            player1: Player 1 IDs.
            player2: Player 2 IDs.
            surfaces: Surface ID per match.
            tournament_levels: Tournament level per match.
            is_bo5: Bo5 indicator per match.

        Returns:
            Array of win probabilities.
        """
        if self._skill_ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        player1 = np.ascontiguousarray(player1, dtype=np.int64)
        player2 = np.ascontiguousarray(player2, dtype=np.int64)
        surfaces = np.ascontiguousarray(surfaces, dtype=np.int64)
        tournament_levels = np.ascontiguousarray(tournament_levels, dtype=np.int64)
        is_bo5 = np.ascontiguousarray(is_bo5, dtype=np.int64)

        return predict_proba_surface_batch(
            player1, player2,
            surfaces, tournament_levels, is_bo5,
            self._skill_ratings, self._cov_matrix,
            self._b, self.config.bo5_factor,
        )

    def get_surface_ratings(self, player_id: int) -> Dict[str, float]:
        """Get all surface ratings for a player.

        Returns:
            Dictionary mapping surface names to ratings.
        """
        if self._skill_ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        result = {}
        for sid, name in SURFACE_NAMES.items():
            result[name] = float(self._skill_ratings[player_id, sid])
        return result

    def get_tournament_additions(self, player_id: int) -> Dict[str, float]:
        """Get tournament-level additions for a player.

        Returns:
            Dictionary mapping tournament names to additive adjustments.
        """
        if self._skill_ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        result = {}
        for tid, name in TOURNAMENT_NAMES.items():
            if tid == TOURNAMENT_250_500:
                result[name] = 0.0  # Baseline
            else:
                idx = NUM_SURFACES + (tid - TOURNAMENT_MASTERS)
                result[name] = float(self._skill_ratings[player_id, idx])
        return result

    def get_effective_rating(
        self,
        player_id: int,
        surface: int = SURFACE_HARD,
        tournament_level: int = TOURNAMENT_250_500,
    ) -> float:
        """Get a player's effective rating for a specific surface + tournament.

        This is the sum of the surface skill and any tournament addition.
        """
        if self._skill_ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        rating = float(self._skill_ratings[player_id, surface])
        if tournament_level >= TOURNAMENT_MASTERS:
            tourn_idx = NUM_SURFACES + (tournament_level - TOURNAMENT_MASTERS)
            rating += float(self._skill_ratings[player_id, tourn_idx])
        return rating

    @property
    def covariance_matrix(self) -> np.ndarray:
        """The prior covariance matrix Σ."""
        return self._cov_matrix.copy()

    @property
    def skill_ratings(self) -> Optional[np.ndarray]:
        """The full skill ratings matrix (num_players, n_skills). Read-only copy."""
        if self._skill_ratings is None:
            return None
        return self._skill_ratings.copy()

    def save_state(self, path: str) -> None:
        """Save state including skill ratings."""
        from ...data.checkpoint import save_checkpoint

        if self._skill_ratings is None:
            raise ValueError("No state to save — model not fitted.")

        arrays = {
            "ratings": self._ratings.ratings,
            "skill_ratings": self._skill_ratings,
            "cov_matrix": self._cov_matrix,
        }
        metadata = {
            "system": "genelo_surface",
            "num_players": self._num_players,
            "current_day": self._current_day,
            "num_games_fitted": self._num_games_fitted,
            "n_skills": self._n_skills,
        }
        save_checkpoint(path, arrays, metadata)

    def load_state(self, path: str) -> None:
        """Load state including skill ratings."""
        from ...data.checkpoint import load_checkpoint

        arrays, metadata = load_checkpoint(path)
        self._num_players = metadata["num_players"]
        self._current_day = metadata.get("current_day")
        self._num_games_fitted = metadata.get("num_games_fitted", 0)
        self._n_skills = metadata.get("n_skills", self._n_skills)

        self._ratings = PlayerRatings(
            ratings=arrays["ratings"],
            metadata={"system": "genelo_surface", "config": self.config},
        )
        self._skill_ratings = arrays["skill_ratings"]
        if "cov_matrix" in arrays:
            self._cov_matrix = arrays["cov_matrix"]
        self._fitted = True
