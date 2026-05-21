"""Surface-Factor Whole History Rating.

A WHR batch model with a TrueSkill-2-style rank-1 cross-surface factor:

    base_i(t)          ~ Wiener(w2_base)
    offset_{i, s}(t)   ~ Wiener(w2_off[s])              s ∈ {Hard, Clay, Grass}
    skill_{i, s}(t)    = w_s * base_i(t) + offset_{i, s}(t)

Inference is by coordinate descent over the four trajectories per player,
each step reusing WHR's tridiagonal Newton solver from the plain ``WHR``
implementation. Surface convention: 0=Hard, 1=Clay, 2=Grass (matches
SurfaceTTT and WSurfaceFactorGlicko).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ..whr._numba_core import (
    LN10_400,
    build_player_day_indices,
    fill_game_arrays,
)
from ._numba_core import (
    N_SURFACES,
    extract_current_ratings,
    predict_proba_batch_factor,
    predict_single_factor,
    run_all_iterations,
)


SURFACE_HARD = 0
SURFACE_CLAY = 1
SURFACE_GRASS = 2
SURFACE_NAMES = ("Hard", "Clay", "Grass")


@dataclass
class SurfaceFactorWHRConfig:
    """Configuration for Surface-Factor WHR.

    w2_base: Wiener variance per day for the base trajectory (Elo^2 / day).
    w2_off: Wiener variance per day for each surface offset (Hard, Clay, Grass).
    w_surfaces: per-surface coupling weights w_s ≥ 0. ``(1, 1, 1)`` shares
        information fully; smaller values isolate the surfaces.
    offset_anchor: precision of a zero-mean Gaussian anchor on each offset
        trajectory. Needed because shifting base by ``c/w_s`` and offset by
        ``-c`` leaves all effective ratings unchanged for any constant ``c``
        — the anchor removes this near-degeneracy.
    """

    w2_base: float = 300.0
    w2_off: Tuple[float, float, float] = (50.0, 50.0, 50.0)
    w_surfaces: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    initial_rating: float = 1500.0
    initial_rd: float = 600.0
    max_iterations: int = 50
    refit_max_iterations: int = 5
    refit_interval: int = 1
    convergence_threshold: float = 1e-6
    offset_anchor: float = 1e-3


class SurfaceFactorWHR(RatingSystem):
    """Whole-History Rating with a rank-1 cross-surface factor model."""

    system_type = RatingSystemType.BATCH

    def __init__(
        self,
        w2_base: float = 300.0,
        w2_off: Union[float, Tuple[float, float, float]] = 50.0,
        w_surfaces: Union[float, Tuple[float, float, float]] = 1.0,
        initial_rating: float = 1500.0,
        initial_rd: float = 600.0,
        max_iterations: int = 50,
        refit_max_iterations: int = 5,
        refit_interval: int = 1,
        convergence_threshold: float = 1e-6,
        offset_anchor: float = 1e-3,
        num_players: Optional[int] = None,
    ):
        w2_off_tuple = self._broadcast(w2_off, "w2_off")
        w_tuple = self._broadcast(w_surfaces, "w_surfaces")
        for w in w_tuple:
            if w < 0.0:
                raise ValueError("w_surfaces entries must be non-negative")

        self.config = SurfaceFactorWHRConfig(
            w2_base=w2_base,
            w2_off=w2_off_tuple,
            w_surfaces=w_tuple,
            initial_rating=initial_rating,
            initial_rd=initial_rd,
            max_iterations=max_iterations,
            refit_max_iterations=refit_max_iterations,
            refit_interval=refit_interval,
            convergence_threshold=convergence_threshold,
            offset_anchor=offset_anchor,
        )

        # Convert hyperparameters to log-gamma scale (consistent with WHR).
        ln10_400_sq = LN10_400 * LN10_400
        self._w2_base_r = w2_base * ln10_400_sq
        self._w2_off_r = np.asarray(
            [w * ln10_400_sq for w in w2_off_tuple], dtype=np.float64,
        )
        self._w_surfaces_arr = np.asarray(w_tuple, dtype=np.float64)
        self._offset_anchor = float(offset_anchor)

        # CSR-like data structures (built lazily in fit/update).
        self._player_offsets: Optional[np.ndarray] = None
        self._pd_days: Optional[np.ndarray] = None
        self._pd_r_base: Optional[np.ndarray] = None
        self._pd_r_off: Optional[np.ndarray] = None
        self._pd_game_offsets: Optional[np.ndarray] = None
        self._pd_game_opp_pd: Optional[np.ndarray] = None
        self._pd_game_score: Optional[np.ndarray] = None
        self._pd_game_surface: Optional[np.ndarray] = None
        self._pd_game_weight: Optional[np.ndarray] = None

        # Per-player current values.
        self._base_rating: Optional[np.ndarray] = None
        self._offset_rating: Optional[np.ndarray] = None
        self._num_iterations = 0
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None

        # Stored data for refits.
        self._stored_player1: Optional[np.ndarray] = None
        self._stored_player2: Optional[np.ndarray] = None
        self._stored_scores: Optional[np.ndarray] = None
        self._stored_days: Optional[np.ndarray] = None
        self._stored_surfaces: Optional[np.ndarray] = None
        self._stored_weights: Optional[np.ndarray] = None
        self._last_refit_day: Optional[int] = None

        super().__init__(num_players=num_players)

    @staticmethod
    def _broadcast(value, name: str) -> Tuple[float, float, float]:
        if isinstance(value, (int, float)):
            return (float(value),) * N_SURFACES
        if len(value) != N_SURFACES:
            raise ValueError(f"{name} must have length {N_SURFACES}")
        return tuple(float(x) for x in value)

    # ------------------------------------------------------------------ init

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        self._base_rating = np.full(num_players, self.config.initial_rating, dtype=np.float64)
        self._offset_rating = np.zeros((num_players, N_SURFACES), dtype=np.float64)
        return PlayerRatings(
            ratings=self._base_rating,
            rd=np.full(num_players, self.config.initial_rd, dtype=np.float64),
            metadata={"system": "surface_factor_whr", "config": self.config},
        )

    # ------------------------------------------------------------------ structures

    def _build_data_structures(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        days: np.ndarray,
        surfaces: np.ndarray,
        weights: np.ndarray,
        num_players: int,
    ) -> None:
        n_games = len(player1)
        (self._player_offsets, self._pd_days, pd1, pd2, total_pd
         ) = build_player_day_indices(player1, player2, days, num_players, n_games)

        self._pd_r_base = np.zeros(total_pd, dtype=np.float64)
        self._pd_r_off = np.zeros((total_pd, N_SURFACES), dtype=np.float64)

        pd_game_counts = np.zeros(total_pd, dtype=np.int64)
        np.add.at(pd_game_counts, pd1, 1)
        np.add.at(pd_game_counts, pd2, 1)
        self._pd_game_offsets = np.zeros(total_pd + 1, dtype=np.int64)
        np.cumsum(pd_game_counts, out=self._pd_game_offsets[1:])

        total_refs = int(self._pd_game_offsets[-1])
        self._pd_game_opp_pd = np.empty(total_refs, dtype=np.int64)
        self._pd_game_score = np.empty(total_refs, dtype=np.float64)
        fill_game_arrays(
            n_games, pd1, pd2, scores,
            self._pd_game_offsets, self._pd_game_opp_pd, self._pd_game_score,
        )

        # Parallel arrays to pd_game_opp_pd: surface and weight for each (pd, game) ref.
        self._pd_game_surface = np.empty(total_refs, dtype=np.int32)
        self._pd_game_weight = np.empty(total_refs, dtype=np.float64)
        write_ptr = self._pd_game_offsets[:-1].copy()
        for i in range(n_games):
            s = surfaces[i]
            w = weights[i]
            self._pd_game_surface[write_ptr[pd1[i]]] = s
            self._pd_game_weight[write_ptr[pd1[i]]] = w
            write_ptr[pd1[i]] += 1
            self._pd_game_surface[write_ptr[pd2[i]]] = s
            self._pd_game_weight[write_ptr[pd2[i]]] = w
            write_ptr[pd2[i]] += 1

    # ------------------------------------------------------------------ optimisation

    def _run_optimisation(self, max_iterations: Optional[int] = None) -> None:
        if self._player_offsets is None or self._num_players is None:
            return
        if max_iterations is None:
            max_iterations = self.config.max_iterations

        self._num_iterations = run_all_iterations(
            self._num_players,
            self._player_offsets,
            self._pd_days,
            self._pd_r_base,
            self._pd_r_off,
            self._pd_game_offsets,
            self._pd_game_opp_pd,
            self._pd_game_score,
            self._pd_game_surface,
            self._pd_game_weight,
            self._w_surfaces_arr,
            self._w2_base_r,
            self._w2_off_r,
            int(max_iterations),
            float(self.config.convergence_threshold),
            self._offset_anchor,
        )

    def _extract_current(self) -> None:
        if self._num_players is None or self._player_offsets is None:
            return
        base = np.empty(self._num_players, dtype=np.float64)
        off = np.zeros((self._num_players, N_SURFACES), dtype=np.float64)
        # Anchor at log-gamma scale equivalent to ``initial_rating`` Elo.
        initial_logged = 0.0  # log-gamma origin
        extract_current_ratings(
            self._num_players, self._player_offsets,
            self._pd_r_base, self._pd_r_off, base, off, initial_logged,
        )

        # Convert log-gamma back to Elo display scale.
        base_elo = self.config.initial_rating + base / LN10_400
        off_elo = off / LN10_400

        self._base_rating = base_elo
        self._offset_rating = off_elo
        self._ratings = PlayerRatings(
            ratings=base_elo,
            rd=np.full(self._num_players, self.config.initial_rd, dtype=np.float64),
            metadata={"system": "surface_factor_whr", "config": self.config},
        )

    def _update_ratings(self, batch, ratings) -> None:
        """Required by the abstract base; SurfaceFactorWHR needs surfaces, so
        the public path is :meth:`update` (with a ``surfaces`` arg)."""
        raise NotImplementedError(
            "SurfaceFactorWHR requires a per-game surface array; use update(batch, surfaces=...)",
        )

    # ------------------------------------------------------------------ fit / update

    def fit(
        self,
        dataset: GameDataset,
        surfaces: np.ndarray,
        weights: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "SurfaceFactorWHR":
        """Fit on a full dataset.

        ``weights``: optional per-game weights (default 1.0 each). Weights
        scale Fisher information exactly as in WWHR — a game with ``w=3`` has
        the same effect on the posterior as three unit-weight games.
        """
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

        if weights is None:
            w_arr = np.ones(n_games, dtype=np.float64)
        else:
            w_arr = np.ascontiguousarray(weights, dtype=np.float64)
            if w_arr.shape[0] != n_games:
                raise ValueError(
                    f"weights length {w_arr.shape[0]} != games {n_games}"
                )

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        days = np.empty(n_games, dtype=np.int32)
        for d in range(len(day_indices)):
            days[day_offsets[d]:day_offsets[d + 1]] = day_indices[d]

        self._stored_player1 = player1.copy()
        self._stored_player2 = player2.copy()
        self._stored_scores = scores.copy()
        self._stored_days = days
        self._stored_surfaces = surfaces.copy()
        self._stored_weights = w_arr.copy()

        self._build_data_structures(
            player1, player2, scores, days, surfaces, w_arr, self._num_players,
        )
        self._run_optimisation()
        self._extract_current()

        self._num_games_fitted = n_games
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        # Force the very next update() to refit, by pretending the last refit
        # happened ``refit_interval`` days ago.
        self._last_refit_day = (
            (self._current_day or 0) - self.config.refit_interval
        )
        return self

    def update(
        self,
        batch: GameBatch,
        surfaces: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "SurfaceFactorWHR":
        if not self._fitted:
            raise ValueError("Model must be fitted before update()")
        surfaces = np.ascontiguousarray(surfaces, dtype=np.int32)
        if len(surfaces) != len(batch):
            raise ValueError("surfaces length mismatch with batch")
        if weights is None:
            w_arr = np.ones(len(batch), dtype=np.float64)
        else:
            w_arr = np.ascontiguousarray(weights, dtype=np.float64)
            if len(w_arr) != len(batch):
                raise ValueError("weights length mismatch with batch")

        new_days = np.full(len(batch), batch.day, dtype=np.int32)
        if self._stored_player1 is None:
            self._stored_player1 = batch.player1.copy()
            self._stored_player2 = batch.player2.copy()
            self._stored_scores = batch.scores.copy()
            self._stored_days = new_days
            self._stored_surfaces = surfaces.copy()
            self._stored_weights = w_arr.copy()
            self._last_refit_day = batch.day
        else:
            self._stored_player1 = np.concatenate([self._stored_player1, batch.player1])
            self._stored_player2 = np.concatenate([self._stored_player2, batch.player2])
            self._stored_scores = np.concatenate([self._stored_scores, batch.scores])
            self._stored_days = np.concatenate([self._stored_days, new_days])
            self._stored_surfaces = np.concatenate([self._stored_surfaces, surfaces])
            self._stored_weights = np.concatenate([self._stored_weights, w_arr])

        if self._last_refit_day is None:
            self._last_refit_day = batch.day

        if batch.day - self._last_refit_day >= self.config.refit_interval:
            self._build_data_structures(
                self._stored_player1, self._stored_player2,
                self._stored_scores, self._stored_days,
                self._stored_surfaces, self._stored_weights, self._num_players,
            )
            self._run_optimisation(self.config.refit_max_iterations)
            self._extract_current()
            self._last_refit_day = batch.day

        self._current_day = batch.day
        return self

    # ------------------------------------------------------------------ predict

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        surfaces: Union[int, np.ndarray, List[int]],
    ) -> Union[float, np.ndarray]:
        if self._base_rating is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert per-player ratings back to log-gamma scale for the
        # internal sigmoid: dropping the initial-rating offset because it
        # cancels in the difference.
        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            i, j, s = int(player1), int(player2), int(surfaces)
            return predict_single_factor(
                (self._base_rating[i] - self.config.initial_rating) * LN10_400,
                self._offset_rating[i, s] * LN10_400,
                (self._base_rating[j] - self.config.initial_rating) * LN10_400,
                self._offset_rating[j, s] * LN10_400,
                self._w_surfaces_arr[s],
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        srf = np.ascontiguousarray(surfaces, dtype=np.int32)

        base_log = (self._base_rating - self.config.initial_rating) * LN10_400
        off_log = self._offset_rating * LN10_400
        return predict_proba_batch_factor(
            p1, p2, srf, base_log, off_log, self._w_surfaces_arr,
        )

    # ------------------------------------------------------------------ inspect

    def get_rating(
        self, player_id: int, surface: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Return (rating, rd). With ``surface=None`` returns the base; with
        a surface index returns the effective Elo rating
        ``w_s · (base - 1500) + offset + 1500``.

        RD is currently the constant ``initial_rd`` — a follow-up could add a
        Hessian-derived per-day uncertainty.
        """
        if self._base_rating is None:
            raise ValueError("Model not fitted. Call fit() first.")
        b = float(self._base_rating[player_id])
        if surface is None:
            return b, float(self.config.initial_rd)
        return self.get_effective_rating(player_id, int(surface)), float(self.config.initial_rd)

    def get_effective_rating(self, player_id: int, surface: int) -> float:
        """Effective Elo rating ``w_s * (base - 1500) + offset_s + 1500``."""
        if self._base_rating is None:
            raise ValueError("Model not fitted. Call fit() first.")
        s = int(surface)
        w = float(self._w_surfaces_arr[s])
        return float(
            w * (self._base_rating[player_id] - self.config.initial_rating)
            + self._offset_rating[player_id, s]
            + self.config.initial_rating
        )

    def get_offsets(self, player_id: int) -> np.ndarray:
        if self._offset_rating is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._offset_rating[player_id].copy()

    def top(self, n: int = 10, surface: Optional[int] = None) -> np.ndarray:
        if self._base_rating is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if surface is None:
            scores = self._base_rating
        else:
            s = int(surface)
            w = float(self._w_surfaces_arr[s])
            scores = w * (self._base_rating - self.config.initial_rating) + self._offset_rating[:, s]
        return np.argsort(-scores)[:n]

    # ------------------------------------------------------------------ housekeeping

    def reset(self) -> "SurfaceFactorWHR":
        self._player_offsets = None
        self._pd_days = None
        self._pd_r_base = None
        self._pd_r_off = None
        self._pd_game_offsets = None
        self._pd_game_opp_pd = None
        self._pd_game_score = None
        self._pd_game_surface = None
        self._pd_game_weight = None
        self._base_rating = None
        self._offset_rating = None
        self._stored_player1 = None
        self._stored_player2 = None
        self._stored_scores = None
        self._stored_days = None
        self._stored_surfaces = None
        self._stored_weights = None
        self._last_refit_day = None
        self._num_iterations = 0
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        w = self.config.w_surfaces
        return (
            f"{type(self).__name__}(w2_base={self.config.w2_base}, "
            f"w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f}), "
            f"players={players}, {status})"
        )


class WSurfaceFactorWHR(SurfaceFactorWHR):
    """Weighted variant of :class:`SurfaceFactorWHR`.

    Identical underlying model and inference; named separately to match the
    package convention (``WHR`` / ``WeightedWHR``) and to signal that the
    expected use is with per-game weights. ``fit()`` and ``update()`` accept
    a ``weights`` array exactly as in WWHR — when omitted, behaviour is
    identical to :class:`SurfaceFactorWHR`.

    Example:
        >>> sys = WSurfaceFactorWHR()
        >>> sys.fit(dataset, surfaces=surfaces, weights=weights)
    """

    pass
