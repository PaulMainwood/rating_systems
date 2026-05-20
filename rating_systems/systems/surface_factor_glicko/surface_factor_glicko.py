"""Surface-Factor Weighted Glicko — rank-1 factor rating over 3 tennis surfaces.

Each player has one base skill plus three surface-specific offsets (Hard, Clay,
Grass). The effective rating used in the Bradley-Terry / Glicko logit on
surface ``s`` is

    R_eff_s = w_s * R_base + R_offset_s,

with per-surface coupling weights ``w_s``. When ``w_s = 0`` for all surfaces
this degenerates to three independent per-surface Glicko ratings; when all
offsets are zero and ``w_s = 1`` it reduces to standard WGlicko on the base
rating. Mid-range ``w_s`` values share information across surfaces, which is
the TrueSkill 2 cross-mode-correlation mechanism translated to tennis.

Surface convention: 0 = Hard, 1 = Clay, 2 = Grass (matches SurfaceTTT).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import math
import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameDataset
from ._numba_core import (
    N_SURFACES,
    fit_all_days_factor,
    predict_proba_batch_factor,
    predict_single_factor,
    update_ratings_batch_factor,
    walk_forward_predict_update_factor,
)

SURFACE_HARD = 0
SURFACE_CLAY = 1
SURFACE_GRASS = 2
SURFACE_NAMES = ("Hard", "Clay", "Grass")


@dataclass
class SurfaceFactorGlickoConfig:
    """Configuration for Surface-Factor Weighted Glicko.

    Defaults are deliberately close to the WGlicko defaults so a researcher
    can run the new system as a near drop-in and only the cross-surface
    parameters are different. The per-surface coupling weights start at 1.0
    (full sharing); tighter offset RD and smaller offset drift reflect the
    expectation that surface-specific skill is more stable than overall form.
    """

    initial_base_rating: float = 1500.0
    initial_base_rd: float = 350.0
    initial_offset_rating: float = 0.0
    initial_offset_rd: float = 100.0
    min_rd: float = 30.0
    max_rd: float = 350.0
    c_base: float = 34.6
    c_offset: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    w_surfaces: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    q: float = math.log(10) / 400


class WSurfaceFactorGlicko(RatingSystem):
    """Weighted Glicko with rank-1 cross-surface skill correlation.

    See module docstring for the underlying factor model. Per-game weights
    scale Fisher information exactly as in :class:`WGlicko` — defaulting to
    1.0 reduces to plain Surface-Factor Glicko.
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_base_rating: float = 1500.0,
        initial_base_rd: float = 350.0,
        initial_offset_rating: float = 0.0,
        initial_offset_rd: float = 100.0,
        min_rd: float = 30.0,
        max_rd: float = 350.0,
        c_base: float = 34.6,
        c_offset: Union[float, Tuple[float, float, float]] = 10.0,
        w_surfaces: Union[float, Tuple[float, float, float]] = 1.0,
        num_players: Optional[int] = None,
    ):
        if isinstance(c_offset, (int, float)):
            c_offset_tuple = (float(c_offset),) * N_SURFACES
        else:
            if len(c_offset) != N_SURFACES:
                raise ValueError(f"c_offset must have length {N_SURFACES} (Hard, Clay, Grass)")
            c_offset_tuple = tuple(float(x) for x in c_offset)

        if isinstance(w_surfaces, (int, float)):
            w_tuple = (float(w_surfaces),) * N_SURFACES
        else:
            if len(w_surfaces) != N_SURFACES:
                raise ValueError(f"w_surfaces must have length {N_SURFACES} (Hard, Clay, Grass)")
            w_tuple = tuple(float(x) for x in w_surfaces)
            for w in w_tuple:
                if w < 0.0:
                    raise ValueError("w_surfaces entries must be non-negative")

        self.config = SurfaceFactorGlickoConfig(
            initial_base_rating=initial_base_rating,
            initial_base_rd=initial_base_rd,
            initial_offset_rating=initial_offset_rating,
            initial_offset_rd=initial_offset_rd,
            min_rd=min_rd,
            max_rd=max_rd,
            c_base=c_base,
            c_offset=c_offset_tuple,
            w_surfaces=w_tuple,
        )
        self._w_surfaces_arr = np.asarray(w_tuple, dtype=np.float64)
        self._c_offset_arr = np.asarray(c_offset_tuple, dtype=np.float64)
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        self._offset_rating: Optional[np.ndarray] = None
        self._offset_rd: Optional[np.ndarray] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Allocate base ratings (in PlayerRatings) plus per-surface offsets."""
        self._offset_rating = np.full(
            (num_players, N_SURFACES),
            self.config.initial_offset_rating,
            dtype=np.float64,
        )
        self._offset_rd = np.full(
            (num_players, N_SURFACES),
            self.config.initial_offset_rd,
            dtype=np.float64,
        )
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_base_rating, dtype=np.float64),
            rd=np.full(num_players, self.config.initial_base_rd, dtype=np.float64),
            last_played=np.zeros(num_players, dtype=np.int32),
            metadata={
                "system": "wsurface_factor_glicko",
                "config": self.config,
                "offset_rating_ref": "use get_offsets()",
            },
        )

    def _ensure_capacity(self, num_players: int) -> None:
        """Grow internal arrays when the dataset has more players than allocated."""
        if self._num_players is None or self._num_players < num_players:
            self._num_players = num_players
            self._ratings = self._initialize_ratings(num_players)

    def _update_ratings(self, batch, ratings) -> None:
        """Default RatingSystem hook is not used; SurfaceFactorGlicko needs surfaces."""
        raise NotImplementedError(
            "WSurfaceFactorGlicko requires a per-game surface array; "
            "use fit(...) or update_weighted(...) with surfaces=...",
        )

    # ------------------------------------------------------------------ fit / update

    def fit(
        self,
        dataset: GameDataset,
        surfaces: np.ndarray,
        weights: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WSurfaceFactorGlicko":
        """Fit on a full dataset.

        Parameters
        ----------
        dataset: GameDataset
        surfaces: int array, one entry per game, values in {0,1,2} (Hard, Clay, Grass).
        weights: optional per-game weights (default 1.0 each).
        end_day: optional last day (inclusive). Cannot be combined with weights.
        player_names: optional id → name mapping for output.
        """
        if end_day is not None and weights is not None:
            raise ValueError("Cannot use both end_day and weights — pre-filter the dataset.")
        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        self._player_names = player_names

        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()
        if player1 is None or len(player1) == 0:
            self._fitted = True
            return self

        n_games = len(player1)
        surfaces = np.ascontiguousarray(surfaces, dtype=np.int32)
        if surfaces.shape[0] != n_games:
            raise ValueError(
                f"surfaces length {surfaces.shape[0]} != games {n_games}"
            )
        if (surfaces < 0).any() or (surfaces >= N_SURFACES).any():
            raise ValueError(f"surfaces must be in [0, {N_SURFACES})")

        self._ensure_capacity(dataset.num_players)

        if weights is None:
            w = np.ones(n_games, dtype=np.float64)
        else:
            w = np.ascontiguousarray(weights, dtype=np.float64)

        fit_all_days_factor(
            player1, player2, scores, w, surfaces, self._w_surfaces_arr,
            day_indices, day_offsets,
            self._ratings.ratings, self._ratings.rd,
            self._offset_rating, self._offset_rd,
            self._ratings.last_played,
            self.config.c_base, self._c_offset_arr,
            self.config.min_rd, self.config.max_rd,
        )

        self._num_games_fitted = n_games
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        self._fitted = True
        return self

    def update_weighted(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        surfaces: np.ndarray,
        day: int,
        weights: Optional[np.ndarray] = None,
    ) -> "WSurfaceFactorGlicko":
        """Apply a single rating-period update for the given games."""
        if not self._fitted:
            raise ValueError("Model must be fitted before update_weighted(). Call fit() first.")
        n = len(player1)
        if n == 0:
            return self
        if weights is None:
            w = np.ones(n, dtype=np.float64)
        else:
            w = np.ascontiguousarray(weights, dtype=np.float64)
        update_ratings_batch_factor(
            np.ascontiguousarray(player1, dtype=np.int64),
            np.ascontiguousarray(player2, dtype=np.int64),
            np.ascontiguousarray(scores, dtype=np.float64),
            w,
            np.ascontiguousarray(surfaces, dtype=np.int32),
            self._w_surfaces_arr,
            self._ratings.ratings, self._ratings.rd,
            self._offset_rating, self._offset_rd,
            self._ratings.last_played,
            int(day),
            self.config.c_base, self._c_offset_arr,
            self.config.min_rd, self.config.max_rd,
        )
        self._num_games_fitted += n
        self._current_day = int(day)
        return self

    # ------------------------------------------------------------------ predict

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        surfaces: Union[int, np.ndarray, List[int]],
    ) -> Union[float, np.ndarray]:
        """Predict P(player1 beats player2) on the supplied surface."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            i = int(player1)
            j = int(player2)
            s = int(surfaces)
            return predict_single_factor(
                self._ratings.ratings[i], self._ratings.rd[i],
                self._offset_rating[i, s], self._offset_rd[i, s],
                self._ratings.ratings[j], self._ratings.rd[j],
                self._offset_rating[j, s], self._offset_rd[j, s],
                self._w_surfaces_arr[s],
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        srf = np.ascontiguousarray(surfaces, dtype=np.int32)
        return predict_proba_batch_factor(
            p1, p2, srf, self._w_surfaces_arr,
            self._ratings.ratings, self._ratings.rd,
            self._offset_rating, self._offset_rd,
        )

    # ------------------------------------------------------------------ inspect

    def get_rating(self, player_id: int, surface: Optional[int] = None) -> Tuple[float, float]:
        """Return (rating, rd). With ``surface=None``, returns the base; otherwise
        the effective surface rating ``w_s · R_base + R_offset_s`` and the
        combined RD ``sqrt((w_s·RD_base)^2 + RD_offset_s^2)``.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if surface is None:
            return float(self._ratings.ratings[player_id]), float(self._ratings.rd[player_id])
        s = int(surface)
        w = float(self._w_surfaces_arr[s])
        r_base = float(self._ratings.ratings[player_id])
        rd_base = float(self._ratings.rd[player_id])
        r_off = float(self._offset_rating[player_id, s])
        rd_off = float(self._offset_rd[player_id, s])
        eff = w * r_base + r_off
        eff_rd = math.sqrt((w * rd_base) ** 2 + rd_off ** 2)
        return eff, eff_rd

    def get_offsets(self, player_id: int) -> np.ndarray:
        """Return the ``(N_SURFACES, 2)`` array of (rating, rd) per surface."""
        if self._offset_rating is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.stack(
            [self._offset_rating[player_id], self._offset_rd[player_id]], axis=1,
        )

    def top(self, n: int = 10, surface: Optional[int] = None) -> np.ndarray:
        """Indices of top-N players by base rating (or by effective surface rating)."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if surface is None:
            scores = self._ratings.ratings
        else:
            s = int(surface)
            w = float(self._w_surfaces_arr[s])
            scores = w * self._ratings.ratings + self._offset_rating[:, s]
        return np.argsort(-scores)[:n]

    # ------------------------------------------------------------------ walk-forward

    def fused_walk_forward(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        surfaces: np.ndarray,
        day_indices: np.ndarray,
        day_offsets: np.ndarray,
        weights: np.ndarray,
        n_train_days: int,
    ) -> np.ndarray:
        """Fused predict+update walk-forward, identical contract to WGlicko's."""
        return walk_forward_predict_update_factor(
            np.ascontiguousarray(player1, dtype=np.int64),
            np.ascontiguousarray(player2, dtype=np.int64),
            np.ascontiguousarray(scores, dtype=np.float64),
            np.ascontiguousarray(surfaces, dtype=np.int32),
            self._w_surfaces_arr,
            day_indices, day_offsets,
            np.ascontiguousarray(weights, dtype=np.float64),
            self._ratings.ratings, self._ratings.rd,
            self._offset_rating, self._offset_rd,
            self._ratings.last_played,
            self.config.c_base, self._c_offset_arr,
            self.config.min_rd, self.config.max_rd,
            int(n_train_days),
        )

    # ------------------------------------------------------------------ housekeeping

    def reset(self) -> "WSurfaceFactorGlicko":
        self._num_games_fitted = 0
        if self._num_players is not None:
            self._offset_rating = None
            self._offset_rd = None
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        w = self.config.w_surfaces
        return (
            f"WSurfaceFactorGlicko(base_rating={self.config.initial_base_rating}, "
            f"base_rd={self.config.initial_base_rd}, "
            f"w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f}), "
            f"players={players}, {status})"
        )
