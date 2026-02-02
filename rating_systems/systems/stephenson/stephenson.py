"""
Stephenson rating system - high-performance Numba implementation.

The Stephenson system extends Glicko with additional parameters for
improved predictive accuracy. It was developed by Alec Stephenson as
the winning entry in the 2012 FIDE/Deloitte Chess Rating Challenge.

Key extensions from Glicko:
1. Opponent's RD directly weights the expected score calculation
2. hval parameter adds uncertainty proportional to games played
3. bval bonus rewards frequent play (creates rating inflation)
4. lambda neighbourhood parameter shrinks ratings toward opponents
5. gamma first-player advantage parameter (e.g., white pieces in chess)

Glicko is obtained as a special case with hval=0, bval=0, lambda=0.

Reference:
- Stephenson & Sonas (2012), PlayerRatings R package
- https://cran.r-project.org/package=PlayerRatings
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
class StephensonConfig:
    """Configuration for Stephenson rating system."""

    initial_rating: float = 2200.0  # Starting rating
    initial_rd: float = 300.0       # Starting rating deviation
    min_rd: float = 30.0            # Minimum RD
    max_rd: float = 350.0           # Maximum RD
    cval: float = 10.0              # RD increase per period of inactivity
    hval: float = 10.0              # Additional RD increase per game played
    bval: float = 0.0               # Per-game bonus (creates inflation if > 0)
    lambda_param: float = 2.0       # Neighbourhood parameter (shrinks toward opponents)
    gamma: float = 0.0              # First-player advantage (e.g., white pieces)
    q: float = math.log(10) / 400   # System constant: ln(10)/400


class Stephenson(RatingSystem):
    """
    Stephenson rating system with Numba acceleration.

    The Stephenson system extends Glicko by including:
    1. A second parameter (hval) controlling player deviation across time
    2. A bonus parameter (bval) rewarding frequent players
    3. A neighbourhood parameter (lambda) shrinking ratings toward opponents
    4. A first-player advantage parameter (gamma)

    Glicko is obtained as a special case with hval=0, bval=0, lambda=0.

    The system improves predictive accuracy over Glicko by ~1% and over
    Elo by ~6.8% on chess data (per the original Kaggle competition).

    Parameters:
        initial_rating: Starting rating for new players (default: 2200)
        initial_rd: Starting rating deviation (default: 300)
        min_rd: Minimum rating deviation (default: 30)
        max_rd: Maximum rating deviation (default: 350)
        cval: RD increase per period of inactivity (default: 10)
        hval: Additional RD increase per game (default: 10)
        bval: Per-game bonus added to actual score (default: 0)
        lambda_param: Neighbourhood shrinkage parameter (default: 2)
        gamma: First-player advantage (default: 0)

    Example:
        >>> steph = Stephenson()
        >>> steph.fit(dataset)
        >>> fitted = steph.get_fitted_ratings()
        >>> print(fitted.top(10))  # Top 10 players with RD
        >>> r, rd = steph.get_rating(player_id)

    Note:
        Setting hval=0, bval=0, lambda_param=0 gives equivalent results
        to the Glicko system (with cval instead of c for RD decay).
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 2200.0,
        initial_rd: float = 300.0,
        min_rd: float = 30.0,
        max_rd: float = 350.0,
        cval: float = 10.0,
        hval: float = 10.0,
        bval: float = 0.0,
        lambda_param: float = 2.0,
        gamma: float = 0.0,
        num_players: Optional[int] = None,
    ):
        self.config = StephensonConfig(
            initial_rating=initial_rating,
            initial_rd=initial_rd,
            min_rd=min_rd,
            max_rd=max_rd,
            cval=cval,
            hval=hval,
            bval=bval,
            lambda_param=lambda_param,
            gamma=gamma,
        )
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Stephenson ratings for all players."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            rd=np.full(num_players, self.config.initial_rd, dtype=np.float64),
            last_played=np.zeros(num_players, dtype=np.int32),
            metadata={"system": "stephenson", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update Stephenson ratings for a rating period."""
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
            self.config.cval,
            self.config.hval,
            self.config.bval,
            self.config.lambda_param,
            self.config.gamma,
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

        Uses Stephenson expected score formula with opponent's RD.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted")

        # Handle single prediction
        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            p1, p2 = int(player1), int(player2)
            return predict_single(
                self._ratings.ratings[p1],
                self._ratings.rd[p1],
                self._ratings.ratings[p2],
                self._ratings.rd[p2],
                self.config.gamma,
            )

        # Batch prediction
        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(
            p1, p2, self._ratings.ratings, self._ratings.rd, self.config.gamma
        )

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "Stephenson":
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
                self.config.cval,
                self.config.hval,
                self.config.bval,
                self.config.lambda_param,
                self.config.gamma,
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

        Returns FittedGlickoRatings (compatible interface) with methods
        for querying results.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted")

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
            raise ValueError("Model not fitted")
        return get_top_n_indices(self._ratings.ratings, n)

    def get_rating(self, player_id: int) -> Tuple[float, float]:
        """Get (rating, rd) for a player."""
        if self._ratings is None:
            raise ValueError("Model not fitted")
        return (
            float(self._ratings.ratings[player_id]),
            float(self._ratings.rd[player_id]),
        )

    def reset(self) -> "Stephenson":
        """Reset to initial state."""
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"Stephenson(initial_rating={self.config.initial_rating}, "
            f"initial_rd={self.config.initial_rd}, cval={self.config.cval}, "
            f"hval={self.config.hval}, lambda={self.config.lambda_param}, "
            f"players={players}, {status})"
        )
