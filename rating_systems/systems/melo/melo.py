"""
mElo (multidimensional Elo) rating system — Numba-accelerated implementation.

Extends standard Elo with per-player style vectors that capture intransitive
matchup effects (rock-paper-scissors dynamics). Each player has:
  - A scalar rating r_i (same as standard Elo)
  - A 2k-dimensional style vector c_i

Win probability:
    P(i beats j) = σ((r_i - r_j + c_i^T Ω c_j) · log10 / scale)

where Ω is block-diagonal with k copies of [[0,1],[-1,0]].

The intransitive term c_i^T Ω c_j is antisymmetric (changes sign when
players swap), so it genuinely captures directional matchup advantages
that standard Elo misses.

Style vectors are clipped to max_c_norm after each update to prevent
divergence from the gradient feedback loop (update ∝ ||c_opponent||).

Reference: Balduzzi et al., "Re-evaluating Evaluation", NeurIPS 2018.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...data.checkpoint import save_checkpoint, load_checkpoint
from ._numba_core import (
    update_ratings_sequential,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
    fit_all_days,
)


@dataclass
class MEloConfig:
    """Configuration for mElo rating system."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0
    k_dim: int = 1           # intransitivity dimensions (style vector is 2*k_dim)
    c_factor: float = 16.0   # learning rate for style vector updates
    initial_c_std: float = 0.01  # std of initial style vector values
    max_c_norm: float = 5.0  # max L2 norm for style vectors (prevents divergence)


class MElo(RatingSystem):
    """
    mElo rating system with Numba acceleration.

    Extends Elo with per-player style vectors that model intransitive
    matchup effects. With k_dim=1, each player gets 2 extra parameters
    capturing a single "rock-paper-scissors" dimension.

    When c_factor=0, reduces to standard Elo.

    The max intransitive effect is bounded: |c_i^T Ω c_j| ≤ k_dim * max_c_norm²
    (by Cauchy-Schwarz). With defaults (k_dim=1, max_c_norm=5), this is at most
    25 rating points — meaningful but not dominant.

    Performance characteristics:
    - Update: O(n * k_dim) sequential
    - Predict: O(n * k_dim) parallel across matchups
    - Memory: O(num_players * (1 + 2*k_dim))

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
        self.config = MEloConfig(
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
        """Create initial mElo ratings and style vectors for all players."""
        # Style vectors need non-zero init to break symmetry
        # (zero init → zero gradients → vectors never update)
        rng = np.random.RandomState(42)
        self._c_vectors = np.ascontiguousarray(
            rng.randn(num_players, 2 * self.config.k_dim).astype(np.float64)
            * self.config.initial_c_std
        )
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "melo", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update mElo ratings and style vectors for a batch of games."""
        if len(batch) == 0:
            return

        update_ratings_sequential(
            batch.player1,
            batch.player2,
            batch.scores,
            ratings.ratings,
            self._c_vectors,
            self.config.k_factor,
            self.config.c_factor,
            self.config.scale,
            self.config.k_dim,
            self.config.max_c_norm,
        )
        self._num_games_fitted += len(batch)

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        Includes both the scalar rating difference and the intransitive
        style-vector matchup component.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
                self._c_vectors[int(player1)],
                self._c_vectors[int(player2)],
                self.config.scale,
                self.config.k_dim,
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(
            p1, p2, self._ratings.ratings, self._c_vectors,
            self.config.scale, self.config.k_dim,
        )

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "MElo":
        """
        Fit mElo on a dataset using optimised single Numba call.

        Args:
            dataset: Game dataset to fit on
            end_day: Last day to include (inclusive)
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

        if player1 is not None and len(player1) > 0:
            fit_all_days(
                player1,
                player2,
                scores,
                day_offsets,
                self._ratings.ratings,
                self._c_vectors,
                self.config.k_factor,
                self.config.c_factor,
                self.config.scale,
                self.config.k_dim,
                self.config.max_c_norm,
            )
            self._num_games_fitted = len(player1)
            self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        else:
            self._num_games_fitted = 0

        self._fitted = True
        return self

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players by scalar rating."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return get_top_n_indices(self._ratings.ratings, n)

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
            "system_class": "MElo",
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

    def reset(self) -> "MElo":
        """Reset to initial state."""
        self._c_vectors = None
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"MElo(k_factor={self.config.k_factor}, "
            f"k_dim={self.config.k_dim}, "
            f"c_factor={self.config.c_factor}, "
            f"players={players}, {status})"
        )
