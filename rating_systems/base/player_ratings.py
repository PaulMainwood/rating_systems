"""Container for player ratings."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import polars as pl


@dataclass
class PlayerRatings:
    """
    Container for player ratings from a rating system.

    This provides a uniform interface for accessing ratings across
    different systems, regardless of whether they track just a rating
    (Elo) or additional parameters like RD and volatility (Glicko-2).

    Uses numpy arrays for broad compatibility with Numba and PyTorch backends.
    """

    ratings: np.ndarray  # (num_players,) Primary rating - float64

    # Optional additional parameters (system-specific)
    rd: Optional[np.ndarray] = None          # Rating deviation (Glicko, Glicko-2)
    volatility: Optional[np.ndarray] = None  # Volatility (Glicko-2)
    last_played: Optional[np.ndarray] = None # Last day played

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Ensure arrays are contiguous and correct dtype for Numba compatibility."""
        self.ratings = np.ascontiguousarray(self.ratings, dtype=np.float64)
        if self.rd is not None:
            self.rd = np.ascontiguousarray(self.rd, dtype=np.float64)
        if self.volatility is not None:
            self.volatility = np.ascontiguousarray(self.volatility, dtype=np.float64)
        if self.last_played is not None:
            self.last_played = np.ascontiguousarray(self.last_played, dtype=np.int32)

    @property
    def num_players(self) -> int:
        return len(self.ratings)

    def get_rating(self, player_id: int) -> float:
        """Get rating for a single player."""
        return float(self.ratings[player_id])

    def get_ratings_batch(self, player_ids: np.ndarray) -> np.ndarray:
        """Get ratings for a batch of players."""
        return self.ratings[player_ids]

    def to_dataframe(self) -> pl.DataFrame:
        """Convert ratings to a Polars DataFrame."""
        data = {
            "player_id": np.arange(self.num_players),
            "rating": self.ratings,
        }

        if self.rd is not None:
            data["rd"] = self.rd
        if self.volatility is not None:
            data["volatility"] = self.volatility
        if self.last_played is not None:
            data["last_played"] = self.last_played

        return pl.DataFrame(data)

    def clone(self) -> "PlayerRatings":
        """Create a deep copy of the ratings."""
        return PlayerRatings(
            ratings=self.ratings.copy(),
            rd=self.rd.copy() if self.rd is not None else None,
            volatility=self.volatility.copy() if self.volatility is not None else None,
            last_played=self.last_played.copy() if self.last_played is not None else None,
            metadata=self.metadata.copy(),
        )
