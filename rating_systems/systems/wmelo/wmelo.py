"""
Weighted mElo (WMElo) rating system — mElo with per-game weights and handicaps.

The update rule is:
    expected = sigmoid((r1 - r2 + c_i^T Ω c_j + handicap) * log10/scale)
    delta = score - expected
    r_i += k_factor * w * delta
    c_i += c_factor * w * delta * (Ω @ c_j)

where w is a per-game weight and handicap is a per-game advantage for
player 1 (in Elo points). When w=1 and handicap=0 for all games, this
is identical to standard mElo.

This is a general-purpose weighted mElo. It knows nothing about surfaces
or any other domain-specific concept — it simply accepts a weight and
handicap for each game.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...data.checkpoint import save_checkpoint, load_checkpoint
from ._numba_core import (
    update_ratings_weighted_sequential,
    predict_proba_batch,
    predict_single,
    fit_all_days_weighted,
    walk_forward_predict_update,
)


@dataclass
class WMEloConfig:
    """Configuration for Weighted mElo rating system."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0
    k_dim: int = 1
    c_factor: float = 16.0
    initial_c_std: float = 0.01
    max_c_norm: float = 5.0


class WMElo(RatingSystem):
    """
    Weighted mElo rating system with Numba acceleration.

    Like standard mElo, but each game has an associated weight that scales
    both the rating and style vector updates, plus an optional handicap that
    shifts the expected score for player 1.

    Update rule:
        delta = score - expected
        r_i += k_factor * w * delta
        c_i += c_factor * w * delta * (Omega @ c_j)

    When weights are not provided, all default to 1.0 (standard mElo).

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        k_factor: Maximum scalar rating change per game (default: 32)
        scale: Rating difference where one player is 10x stronger (default: 400)
        k_dim: Number of intransitivity dimensions (default: 1)
        c_factor: Style vector learning rate (default: 16)
        initial_c_std: Std for initial style vectors (default: 0.01)
        max_c_norm: Max L2 norm for style vectors (default: 5.0)
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        scale: float = 400.0,
        k_dim: int = 1,
        c_factor: float = 16.0,
        initial_c_std: float = 0.01,
        max_c_norm: float = 5.0,
        num_players: Optional[int] = None,
    ):
        self.config = WMEloConfig(
            initial_rating=initial_rating,
            k_factor=k_factor,
            scale=scale,
            k_dim=k_dim,
            c_factor=c_factor,
            initial_c_std=initial_c_std,
            max_c_norm=max_c_norm,
        )
        self._c_vectors: Optional[np.ndarray] = None
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial WMElo ratings and style vectors for all players."""
        rng = np.random.RandomState(42)
        self._c_vectors = np.ascontiguousarray(
            rng.randn(num_players, 2 * self.config.k_dim).astype(np.float64)
            * self.config.initial_c_std
        )
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "wmelo", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with uniform weights and no handicaps (standard mElo behaviour)."""
        if len(batch) == 0:
            return

        n = len(batch)
        weights = np.ones(n, dtype=np.float64)
        handicaps = np.zeros(n, dtype=np.float64)
        update_ratings_weighted_sequential(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
            handicaps,
            ratings.ratings,
            self._c_vectors,
            self.config.k_factor,
            self.config.c_factor,
            self.config.scale,
            self.config.k_dim,
            self.config.max_c_norm,
        )
        self._num_games_fitted += n

    def update_weighted(
        self,
        batch: GameBatch,
        weights: np.ndarray,
        handicaps: Optional[np.ndarray] = None,
    ) -> "WMElo":
        """
        Incrementally update ratings with per-game weights and handicaps.

        Args:
            batch: Games to process
            weights: Per-game weights array (same length as batch)
            handicaps: Per-game handicap for player 1 in Elo points.
                       Positive = player 1 advantage. If None, all zero.

        Returns:
            self (for method chaining)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before updating. Call fit() first.")

        if len(batch) == 0:
            return self

        n = len(batch)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if handicaps is None:
            h = np.zeros(n, dtype=np.float64)
        else:
            h = np.ascontiguousarray(handicaps, dtype=np.float64)
        update_ratings_weighted_sequential(
            batch.player1,
            batch.player2,
            batch.scores,
            weights,
            h,
            self._ratings.ratings,
            self._c_vectors,
            self.config.k_factor,
            self.config.c_factor,
            self.config.scale,
            self.config.k_dim,
            self.config.max_c_norm,
        )
        self._num_games_fitted += n
        self._current_day = batch.day
        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        handicaps: Optional[Union[float, np.ndarray]] = None,
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        Includes both the scalar rating difference, the intransitive
        style-vector matchup component, and an optional handicap.

        Args:
            player1: Single player ID or array of player IDs
            player2: Single player ID or array of player IDs
            handicaps: Per-game handicap for player 1 in Elo points.
                       Single float for single prediction, array for batch.
                       Positive = player 1 advantage. If None, all zero.
            day: Ignored (online system, no time decay)

        Returns:
            Single probability or array of probabilities
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            h = 0.0 if handicaps is None else float(handicaps)
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
                self._c_vectors[int(player1)],
                self._c_vectors[int(player2)],
                self.config.scale,
                self.config.k_dim,
                h,
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        if handicaps is None:
            h = np.zeros(len(p1), dtype=np.float64)
        else:
            h = np.ascontiguousarray(handicaps, dtype=np.float64)
        return predict_proba_batch(
            p1, p2, self._ratings.ratings, self._c_vectors,
            self.config.scale, self.config.k_dim, h,
        )

    def fit(
        self,
        dataset: GameDataset,
        weights: Optional[np.ndarray] = None,
        handicaps: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WMElo":
        """
        Fit the rating system on a dataset with optional per-game weights and handicaps.

        Args:
            dataset: Game dataset to fit on
            weights: Per-game weights array. If None, all weights default to 1.0.
            handicaps: Per-game handicap for player 1 in Elo points.
                       Positive = player 1 advantage. If None, all zero.
            end_day: Last day to include (inclusive). Cannot be used together
                     with weights — pre-filter the dataset instead.
            player_names: Optional mapping of player_id -> name

        Returns:
            self (for method chaining)
        """
        self._player_names = player_names

        if end_day is not None and weights is not None:
            raise ValueError(
                "Cannot use both end_day and weights. Pre-filter the dataset "
                "and pass the corresponding weights instead."
            )

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        # Initialize if needed
        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        # Get pre-batched arrays for direct Numba processing
        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is not None and len(player1) > 0:
            n = len(player1)
            if weights is None:
                w = np.ones(n, dtype=np.float64)
            else:
                w = np.ascontiguousarray(weights, dtype=np.float64)

            if handicaps is None:
                h = np.zeros(n, dtype=np.float64)
            else:
                h = np.ascontiguousarray(handicaps, dtype=np.float64)

            fit_all_days_weighted(
                player1,
                player2,
                scores,
                w,
                h,
                day_offsets,
                self._ratings.ratings,
                self._c_vectors,
                self.config.k_factor,
                self.config.c_factor,
                self.config.scale,
                self.config.k_dim,
                self.config.max_c_norm,
            )
            self._num_games_fitted = n
            self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        else:
            self._num_games_fitted = 0

        self._fitted = True
        return self

    def get_c_vectors(self) -> Optional[np.ndarray]:
        """Get a copy of the style vectors (num_players, 2*k_dim)."""
        if self._c_vectors is None:
            return None
        return self._c_vectors.copy()

    def snapshot(self) -> dict:
        """Lightweight in-memory copy of fitted state."""
        state = super().snapshot()
        if self._c_vectors is not None:
            state["c_vectors"] = self._c_vectors.copy()
        return state

    def restore(self, state: dict) -> None:
        """Restore from in-memory snapshot."""
        super().restore(state)
        if "c_vectors" in state:
            self._c_vectors = state["c_vectors"].copy()

    def save_state(self, path: str) -> None:
        """Save fitted state to .npz file."""
        if not self._fitted or self._ratings is None:
            raise ValueError("Model must be fitted before saving state.")

        arrays = {
            "ratings": self._ratings.ratings,
            "c_vectors": self._c_vectors,
        }
        metadata = {
            "system_class": "WMElo",
            "num_players": self._num_players,
            "current_day": self._current_day,
            "k_dim": self.config.k_dim,
        }
        save_checkpoint(path, arrays, metadata)

    def load_state(self, path: str) -> None:
        """Restore fitted state from .npz file."""
        arrays, metadata = load_checkpoint(path)

        self._num_players = metadata["num_players"]
        self._current_day = metadata["current_day"]
        self._fitted = True
        self._ratings = PlayerRatings(ratings=arrays["ratings"])
        self._c_vectors = arrays["c_vectors"]

    def fused_walk_forward(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        day_indices: np.ndarray,
        day_offsets: np.ndarray,
        weights: np.ndarray,
        handicaps: np.ndarray,
        n_train_days: int,
    ) -> np.ndarray:
        """Fused predict+update walk-forward in a single Numba call.

        Eliminates Python per-day overhead. Modifies ratings and style
        vectors in-place. Returns predictions array (NaN for training period).
        """
        return walk_forward_predict_update(
            player1, player2, scores, day_offsets,
            weights, handicaps,
            self._ratings.ratings, self._c_vectors,
            self.config.k_factor, self.config.c_factor,
            self.config.scale, self.config.k_dim,
            self.config.max_c_norm,
            n_train_days,
        )

    def reset(self) -> "WMElo":
        """Reset to initial state."""
        self._c_vectors = None
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WMElo(k_factor={self.config.k_factor}, "
            f"k_dim={self.config.k_dim}, "
            f"c_factor={self.config.c_factor}, "
            f"players={players}, {status})"
        )
