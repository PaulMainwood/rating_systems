"""
Glicko rating system - high-performance Numba implementation.

Glicko extends Elo by tracking Rating Deviation (RD), which represents
uncertainty in a player's rating. RD decreases with more games and
increases during periods of inactivity.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...results.fitted_ratings import FittedGlickoRatings
from ._numba_core import (
    update_ratings_batch,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
    fit_all_days,
)


@dataclass
class GlickoConfig:
    """Configuration for Glicko rating system."""

    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    min_rd: float = 30.0
    max_rd: float = 350.0
    c: float = 34.6  # RD increase per rating period of inactivity
    q: float = math.log(10) / 400  # System constant: ln(10)/400


class Glicko(RatingSystem):
    """
    Glicko rating system with Numba acceleration.

    Glicko improves on Elo by:
    1. Tracking rating deviation (RD) - uncertainty in the rating
    2. RD decreases when playing games (more certainty)
    3. RD increases during inactivity (less certainty over time)
    4. Using RD in prediction (uncertain ratings matter less)

    Within a rating period (day), all games are treated as simultaneous.
    Updates use pre-period ratings, then RD is adjusted based on games played.

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        initial_rd: Starting rating deviation (default: 350)
        min_rd: Minimum rating deviation (default: 30)
        max_rd: Maximum rating deviation (default: 350)
        c: RD increase per period of inactivity (default: 34.6)

    Example:
        >>> glicko = Glicko()
        >>> glicko.fit(dataset)
        >>> fitted = glicko.get_fitted_ratings()
        >>> print(fitted.top(10))  # Top 10 players with RD
        >>> r, rd = fitted.get_rating(player_id)
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        min_rd: float = 30.0,
        max_rd: float = 350.0,
        c: float = 34.6,
        num_players: Optional[int] = None,
    ):
        self.config = GlickoConfig(
            initial_rating=initial_rating,
            initial_rd=initial_rd,
            min_rd=min_rd,
            max_rd=max_rd,
            c=c,
        )
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Glicko ratings for all players."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            rd=np.full(num_players, self.config.initial_rd, dtype=np.float64),
            last_played=np.zeros(num_players, dtype=np.int32),
            metadata={"system": "glicko", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update Glicko ratings for a rating period."""
        if len(batch) == 0:
            return

        n_updated = update_ratings_batch(
            batch.player1,
            batch.player2,
            batch.scores,
            ratings.ratings,
            ratings.rd,
            ratings.last_played,
            batch.day,
            self.config.c,
            self.config.min_rd,
            self.config.max_rd,
        )
        self._num_games_fitted += len(batch)

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        Uses combined RD of both players in the calculation.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle single prediction
        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            p1, p2 = int(player1), int(player2)
            return predict_single(
                self._ratings.ratings[p1],
                self._ratings.rd[p1],
                self._ratings.ratings[p2],
                self._ratings.rd[p2],
            )

        # Batch prediction
        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(
            p1, p2, self._ratings.ratings, self._ratings.rd
        )

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "Glicko":
        """
        Fit the rating system on a dataset.

        Uses optimized single Numba call to process all days without
        Python iteration overhead.

        Args:
            dataset: Game dataset to fit on
            end_day: Last day to include (inclusive)
            player_names: Optional mapping of player_id -> name
        """
        self._player_names = player_names

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        # Get pre-batched arrays for direct Numba processing
        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is not None and len(player1) > 0:
            # Process ALL days in single Numba call - no Python iteration!
            fit_all_days(
                player1,
                player2,
                scores,
                day_indices,
                day_offsets,
                self._ratings.ratings,
                self._ratings.rd,
                self._ratings.last_played,
                self.config.c,
                self.config.min_rd,
                self.config.max_rd,
            )
            self._num_games_fitted = len(player1)
            self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        else:
            self._num_games_fitted = 0

        self._fitted = True
        return self

    def get_fitted_ratings(self) -> FittedGlickoRatings:
        """
        Get a queryable fitted ratings object.

        Returns:
            FittedGlickoRatings with methods for querying results
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return FittedGlickoRatings(
            ratings=self._ratings.ratings.copy(),
            rd=self._ratings.rd.copy(),
            last_played=self._ratings.last_played.copy() if self._ratings.last_played is not None else None,
            q=self.config.q,
            initial_rating=self.config.initial_rating,
            initial_rd=self.config.initial_rd,
            num_games_fitted=self._num_games_fitted,
            last_day=self._current_day,
            player_names=self._player_names,
        )

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return get_top_n_indices(self._ratings.ratings, n)

    def get_rating(self, player_id: int) -> Tuple[float, float]:
        """Get (rating, rd) for a player."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return (
            float(self._ratings.ratings[player_id]),
            float(self._ratings.rd[player_id]),
        )

    def reset(self) -> "Glicko":
        """Reset to initial state."""
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"Glicko(initial_rating={self.config.initial_rating}, "
            f"initial_rd={self.config.initial_rd}, players={players}, {status})"
        )
