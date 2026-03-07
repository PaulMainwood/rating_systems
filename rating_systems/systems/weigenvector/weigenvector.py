"""
Weighted Eigenvector Centrality — with per-game weights and handicaps.

The update rule scales the edge weight added to the adjacency matrix by the
per-game weight from the boosted ensemble.  Handicaps shift the prediction
logistic for player 1.

Uses the same array-based edge storage + power-iteration with damping as
the standard EigenvectorCentrality.

When w=1 and handicap=0 for all games, this is identical to standard
EigenvectorCentrality.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigs

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...data.checkpoint import save_checkpoint, load_checkpoint
from ..eigenvector._numba_core import (
    predict_proba_batch,
    predict_single,
    decay_edges_inplace,
)
from ._numba_core import (
    predict_proba_with_handicap_batch,
    predict_single_with_handicap,
)

# Internal constants (shared with eigenvector)
_INITIAL_CAPACITY = 4096
_PRUNE_THRESHOLD = 1e-4
_POWER_MAX_ITER = 100
_POWER_TOL = 1e-6


@dataclass
class WEigenvectorConfig:
    """Configuration for Weighted Eigenvector Centrality."""

    half_life: float = 365.0
    scale: float = 200.0
    centrality_scale: float = 10000.0
    recompute_interval: int = 7
    min_edges: int = 10
    damping: float = 1e-6


class WEigenvectorCentrality(RatingSystem):
    """
    Weighted Eigenvector Centrality rating system.

    Like standard EigenvectorCentrality, but each game has an associated
    weight that scales the edge added to the adjacency matrix, plus an
    optional handicap that shifts the prediction for player 1.

    Parameters:
        half_life: Exponential decay half-life in days (default: 365)
        scale: Logistic scale for predictions (default: 200)
        centrality_scale: Multiplier to map centrality to Elo-like range
        recompute_interval: Days between eigenvector recomputations (default: 7)
        min_edges: Minimum non-zero edges before first centrality computation
        damping: Uniform perturbation strength for irreducibility (default: 1e-6)
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        half_life: float = 365.0,
        scale: float = 200.0,
        centrality_scale: float = 10000.0,
        recompute_interval: int = 7,
        min_edges: int = 10,
        damping: float = 1e-6,
        num_players: Optional[int] = None,
    ):
        self.config = WEigenvectorConfig(
            half_life=half_life,
            scale=scale,
            centrality_scale=centrality_scale,
            recompute_interval=recompute_interval,
            min_edges=min_edges,
            damping=damping,
        )
        self._decay_rate = np.log(2.0) / half_life

        # Array-based edge storage with dict index for O(1) lookup
        self._edge_src = np.empty(_INITIAL_CAPACITY, dtype=np.int32)
        self._edge_dst = np.empty(_INITIAL_CAPACITY, dtype=np.int32)
        self._edge_weight = np.empty(_INITIAL_CAPACITY, dtype=np.float64)
        self._edge_day = np.empty(_INITIAL_CAPACITY, dtype=np.int32)
        self._edge_index: Dict[tuple, int] = {}
        self._n_edges = 0
        self._edge_capacity = _INITIAL_CAPACITY

        self._centrality: Optional[np.ndarray] = None
        self._last_eigenvector_day: int = -1
        self._days_since_recompute: int = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial ratings (uniform centrality)."""
        n = num_players
        self._edge_index.clear()
        self._n_edges = 0
        self._centrality = np.ones(n, dtype=np.float64) / np.sqrt(n)
        return PlayerRatings(
            ratings=np.full(n, self.config.centrality_scale / np.sqrt(n),
                            dtype=np.float64),
            metadata={"system": "weigenvector"},
        )

    # ------------------------------------------------------------------
    # Edge storage helpers
    # ------------------------------------------------------------------

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._edge_capacity:
            return
        new_cap = max(needed, self._edge_capacity * 2)
        for attr in ('_edge_src', '_edge_dst', '_edge_day'):
            old = getattr(self, attr)
            new = np.empty(new_cap, dtype=old.dtype)
            new[:self._n_edges] = old[:self._n_edges]
            setattr(self, attr, new)
        old_w = self._edge_weight
        new_w = np.empty(new_cap, dtype=np.float64)
        new_w[:self._n_edges] = old_w[:self._n_edges]
        self._edge_weight = new_w
        self._edge_capacity = new_cap

    def _init_edge_arrays(self, capacity: int) -> None:
        self._edge_capacity = capacity
        self._edge_src = np.empty(capacity, dtype=np.int32)
        self._edge_dst = np.empty(capacity, dtype=np.int32)
        self._edge_weight = np.empty(capacity, dtype=np.float64)
        self._edge_day = np.empty(capacity, dtype=np.int32)

    def _rebuild_edge_index(self) -> None:
        self._edge_index = {}
        src = self._edge_src
        dst = self._edge_dst
        for i in range(self._n_edges):
            self._edge_index[(int(src[i]), int(dst[i]))] = i

    # ------------------------------------------------------------------
    # Adjacency matrix
    # ------------------------------------------------------------------

    def _update_adjacency_weighted(self, batch: GameBatch,
                                    weights: np.ndarray) -> None:
        """Update edge arrays with weighted games."""
        day = batch.day
        dr = self._decay_rate
        for g in range(len(batch)):
            p1 = int(batch.player1[g])
            p2 = int(batch.player2[g])
            s = batch.scores[g]
            gw = float(weights[g])

            if s > 0.5:
                loser, winner = p2, p1
            else:
                loser, winner = p1, p2

            key = (loser, winner)
            idx = self._edge_index.get(key)
            if idx is not None:
                dt = day - self._edge_day[idx]
                if dt > 0:
                    self._edge_weight[idx] *= np.exp(-dr * dt)
                    self._edge_day[idx] = day
                self._edge_weight[idx] += gw
            else:
                idx = self._n_edges
                self._ensure_capacity(idx + 1)
                self._edge_src[idx] = loser
                self._edge_dst[idx] = winner
                self._edge_weight[idx] = gw
                self._edge_day[idx] = day
                self._edge_index[key] = idx
                self._n_edges += 1

    def _build_sparse_matrix(self, day: int) -> csr_matrix:
        """Build CSR from edge arrays, decaying weights to ``day`` in-place."""
        n = self._n_edges
        if n == 0:
            return csr_matrix((self._num_players, self._num_players),
                              dtype=np.float64)

        decay_edges_inplace(
            self._edge_weight, self._edge_day, n, day, self._decay_rate,
        )

        w = self._edge_weight[:n]
        alive = w >= _PRUNE_THRESHOLD

        return csr_matrix(
            (w[alive].copy(),
             (self._edge_src[:n][alive], self._edge_dst[:n][alive])),
            shape=(self._num_players, self._num_players),
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Centrality computation
    # ------------------------------------------------------------------

    def _recompute_centrality(self, day: int) -> None:
        """Compute centrality via ARPACK with damping for Perron-Frobenius."""
        A_T = self._build_sparse_matrix(day).T.tocsr()

        if A_T.nnz < self.config.min_edges:
            return

        n = A_T.shape[0]
        eps = self.config.damping

        def matvec(v):
            return A_T @ v + eps * v.sum()

        op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

        try:
            _, eigvecs = eigs(op, k=1, which='LR', v0=self._centrality,
                              maxiter=_POWER_MAX_ITER, tol=_POWER_TOL)
            ev = np.real(eigvecs[:, 0])
            if ev.sum() < 0:
                ev = -ev
            norm = np.linalg.norm(ev)
            if norm > 0:
                ev /= norm
            self._centrality = np.ascontiguousarray(ev, dtype=np.float64)
        except Exception:
            return

        self._ratings.ratings[:] = self._centrality * self.config.centrality_scale
        self._last_eigenvector_day = day

    # ------------------------------------------------------------------
    # Rating system interface
    # ------------------------------------------------------------------

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update with uniform weights (standard behaviour)."""
        if len(batch) == 0:
            return

        n = len(batch)
        weights = np.ones(n, dtype=np.float64)
        self._update_adjacency_weighted(batch, weights)
        self._days_since_recompute += 1

        if self._days_since_recompute >= self.config.recompute_interval:
            self._recompute_centrality(batch.day)
            self._days_since_recompute = 0

    def update_weighted(
        self,
        batch: GameBatch,
        weights: np.ndarray,
        handicaps: Optional[np.ndarray] = None,
    ) -> "WEigenvectorCentrality":
        """Incrementally update with per-game weights."""
        if not self._fitted:
            raise ValueError("Model must be fitted before updating.")

        if len(batch) == 0:
            return self

        w = np.ascontiguousarray(weights, dtype=np.float64)
        self._update_adjacency_weighted(batch, w)
        self._days_since_recompute += 1
        self._current_day = batch.day

        if self._days_since_recompute >= self.config.recompute_interval:
            self._recompute_centrality(batch.day)
            self._days_since_recompute = 0

        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        handicaps: Optional[Union[float, np.ndarray]] = None,
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict probability that player1 beats player2."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            if handicaps is not None:
                return predict_single_with_handicap(
                    self._ratings.ratings[int(player1)],
                    self._ratings.ratings[int(player2)],
                    self.config.scale,
                    float(handicaps),
                )
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
                self.config.scale,
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        if handicaps is not None:
            h = np.ascontiguousarray(handicaps, dtype=np.float64)
            return predict_proba_with_handicap_batch(
                p1, p2, self._ratings.ratings, self.config.scale, h,
            )
        return predict_proba_batch(p1, p2, self._ratings.ratings, self.config.scale)

    def fit(
        self,
        dataset: GameDataset,
        weights: Optional[np.ndarray] = None,
        handicaps: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "WEigenvectorCentrality":
        """Fit on a dataset with optional per-game weights."""
        self._player_names = player_names

        if end_day is not None and weights is not None:
            raise ValueError(
                "Cannot use both end_day and weights. Pre-filter the dataset "
                "and pass the corresponding weights instead."
            )

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        # Build edges only (skip expensive eigenvector recomputes during fit)
        game_idx = 0
        for batch in dataset.iter_days():
            n = len(batch)
            if n == 0:
                continue

            if weights is not None:
                w = np.ascontiguousarray(
                    weights[game_idx:game_idx + n], dtype=np.float64)
            else:
                w = np.ones(n, dtype=np.float64)

            self._update_adjacency_weighted(batch, w)
            self._current_day = batch.day
            game_idx += n

        # Single eigenvector computation at the end
        self._recompute_centrality(self._current_day)
        self._days_since_recompute = 0

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        state = super().snapshot()
        n = self._n_edges
        state["edge_src"] = self._edge_src[:n].copy()
        state["edge_dst"] = self._edge_dst[:n].copy()
        state["edge_weight"] = self._edge_weight[:n].copy()
        state["edge_day"] = self._edge_day[:n].copy()
        state["centrality"] = self._centrality.copy()
        state["last_eigenvector_day"] = self._last_eigenvector_day
        state["days_since_recompute"] = self._days_since_recompute
        return state

    def restore(self, state: dict) -> None:
        super().restore(state)
        src = state["edge_src"]
        n = len(src)
        self._n_edges = n
        cap = max(n, _INITIAL_CAPACITY)
        self._init_edge_arrays(cap)
        self._edge_src[:n] = src
        self._edge_dst[:n] = state["edge_dst"]
        self._edge_weight[:n] = state["edge_weight"]
        self._edge_day[:n] = state["edge_day"]
        self._rebuild_edge_index()
        self._centrality = state["centrality"].copy()
        self._last_eigenvector_day = state["last_eigenvector_day"]
        self._days_since_recompute = state["days_since_recompute"]

    def save_state(self, path: str) -> None:
        if not self._fitted or self._ratings is None:
            raise ValueError("Model must be fitted before saving state.")

        n = self._n_edges
        arrays = {
            "ratings": self._ratings.ratings,
            "centrality": self._centrality,
            "edge_src": self._edge_src[:n],
            "edge_dst": self._edge_dst[:n],
            "edge_weight": self._edge_weight[:n],
            "edge_day": self._edge_day[:n],
        }
        metadata = {
            "system_class": "WEigenvectorCentrality",
            "num_players": self._num_players,
            "current_day": self._current_day,
            "last_eigenvector_day": self._last_eigenvector_day,
            "days_since_recompute": self._days_since_recompute,
        }
        save_checkpoint(path, arrays, metadata)

    def load_state(self, path: str) -> None:
        arrays, metadata = load_checkpoint(path)

        self._num_players = metadata["num_players"]
        self._current_day = metadata["current_day"]
        self._last_eigenvector_day = metadata.get("last_eigenvector_day", -1)
        self._days_since_recompute = metadata.get("days_since_recompute", 0)
        self._fitted = True
        self._ratings = PlayerRatings(ratings=arrays["ratings"])
        self._centrality = arrays["centrality"]
        self._decay_rate = np.log(2.0) / self.config.half_life

        src = arrays.get("edge_src")
        if src is not None:
            n = len(src)
            self._n_edges = n
            cap = max(n, _INITIAL_CAPACITY)
            self._init_edge_arrays(cap)
            self._edge_src[:n] = src
            self._edge_dst[:n] = arrays["edge_dst"]
            self._edge_weight[:n] = arrays["edge_weight"]
            self._edge_day[:n] = arrays["edge_day"].astype(np.int32)
        else:
            edge_keys = arrays["edge_keys"]
            edge_vals = arrays["edge_vals"]
            n = len(edge_keys)
            self._n_edges = n
            cap = max(n, _INITIAL_CAPACITY)
            self._init_edge_arrays(cap)
            if n > 0:
                self._edge_src[:n] = edge_keys[:, 0]
                self._edge_dst[:n] = edge_keys[:, 1]
                self._edge_weight[:n] = edge_vals[:, 0]
                self._edge_day[:n] = edge_vals[:, 1].astype(np.int32)

        self._rebuild_edge_index()

    def reset(self) -> "WEigenvectorCentrality":
        self._edge_index = {}
        self._n_edges = 0
        self._init_edge_arrays(_INITIAL_CAPACITY)
        self._centrality = None
        self._last_eigenvector_day = -1
        self._days_since_recompute = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WEigenvectorCentrality(half_life={self.config.half_life}, "
            f"scale={self.config.scale}, "
            f"players={players}, {status})"
        )
