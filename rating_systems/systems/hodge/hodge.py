"""Hodge intransitivity feature system.

Maintains a directed match graph and computes intransitivity scores
via Hodge decomposition for each matchup. These scores are features
for the XGBoost ensemble, not win probabilities.

See Clegg & Cartlidge (2025, arXiv:2510.20454) for the approach.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch
from ...data.checkpoint import save_checkpoint, load_checkpoint
from .graph import MatchGraph
from .decomposition import compute_i_star


@dataclass
class HodgeConfig:
    """Configuration for Hodge intransitivity system."""

    half_life: float = 365.0   # days for exponential decay
    min_common: int = 2        # minimum common opponents to produce a score


class HodgeIntransitivity(RatingSystem):
    """Hodge intransitivity feature system.

    Produces I* scores (intransitivity × √evidence) rather than win
    probabilities. These are intended as features for an ensemble model.

    The system is ONLINE: the match graph is incrementally updated
    match-by-match with no need for full refit.
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        half_life: float = 365.0,
        min_common: int = 2,
        max_nodes: int = 50,
        num_players: Optional[int] = None,
    ):
        self._half_life = half_life
        self._min_common = min_common
        self._max_nodes = max_nodes
        self._graph = MatchGraph()
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create minimal dummy ratings (not used for predictions)."""
        return PlayerRatings(ratings=np.zeros(num_players, dtype=np.float64))

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Add match results to the graph."""
        for i in range(len(batch)):
            p1 = int(batch.player1[i])
            p2 = int(batch.player2[i])
            score = float(batch.scores[i])
            if score >= 0.5:
                self._graph.add_result(p1, p2, batch.day)
            else:
                self._graph.add_result(p2, p1, batch.day)

    def predict_proba(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        day: Optional[int] = None,
    ) -> np.ndarray:
        """Compute I* intransitivity scores for each matchup.

        Returns an array of I* scores (NOT probabilities).
        Values are typically in [0, ~10+] range, or NaN if insufficient data.
        """
        n = len(player1)
        scores = np.full(n, np.nan, dtype=np.float64)

        graph = self._graph
        current_day = self._current_day
        half_life = self._half_life
        min_common = self._min_common
        max_nodes = self._max_nodes

        for i in range(n):
            scores[i] = compute_i_star(
                int(player1[i]),
                int(player2[i]),
                graph,
                current_day,
                half_life,
                min_common,
                max_nodes,
            )

        return scores

    def save_state(self, path: str) -> None:
        """Save graph state to .npz checkpoint."""
        if not self._fitted:
            raise ValueError("Model must be fitted before saving state.")

        arrays = self._graph.save_arrays()
        metadata = {
            "system_class": self.__class__.__name__,
            "num_players": self._num_players,
            "current_day": self._current_day,
            "half_life": self._half_life,
            "min_common": self._min_common,
            "max_nodes": self._max_nodes,
        }
        save_checkpoint(path, arrays, metadata)

    def load_state(self, path: str) -> None:
        """Restore graph state from .npz checkpoint."""
        arrays, metadata = load_checkpoint(path)

        self._num_players = metadata["num_players"]
        self._current_day = metadata["current_day"]
        self._half_life = metadata.get("half_life", 365.0)
        self._min_common = metadata.get("min_common", 2)
        self._max_nodes = metadata.get("max_nodes", 50)
        self._fitted = True

        self._graph = MatchGraph()
        self._graph.load_arrays(arrays)

        # Initialize dummy ratings
        self._ratings = self._initialize_ratings(self._num_players)
