"""Container for player ratings (numpy-based for broad compatibility)."""

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import torch


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


@dataclass
class TorchPlayerRatings:
    """
    Container for player ratings using PyTorch tensors (for torch-based systems).
    """

    ratings: "torch.Tensor"  # (num_players,) Primary rating
    device: "torch.device" = field(default_factory=lambda: __import__("torch").device("cpu"))

    # Optional additional parameters
    rd: Optional["torch.Tensor"] = None
    volatility: Optional["torch.Tensor"] = None
    last_played: Optional["torch.Tensor"] = None

    # Metadata
    metadata: Dict = field(default_factory=dict)

    @property
    def num_players(self) -> int:
        return len(self.ratings)

    def get_rating(self, player_id: int) -> float:
        """Get rating for a single player."""
        return self.ratings[player_id].item()

    def get_ratings_batch(self, player_ids: "torch.Tensor") -> "torch.Tensor":
        """Get ratings for a batch of players."""
        return self.ratings[player_ids]

    def to_dataframe(self) -> pl.DataFrame:
        """Convert ratings to a Polars DataFrame."""
        data = {
            "player_id": list(range(self.num_players)),
            "rating": self.ratings.cpu().numpy(),
        }

        if self.rd is not None:
            data["rd"] = self.rd.cpu().numpy()
        if self.volatility is not None:
            data["volatility"] = self.volatility.cpu().numpy()
        if self.last_played is not None:
            data["last_played"] = self.last_played.cpu().numpy()

        return pl.DataFrame(data)

    def clone(self) -> "TorchPlayerRatings":
        """Create a deep copy of the ratings."""
        return TorchPlayerRatings(
            ratings=self.ratings.clone(),
            device=self.device,
            rd=self.rd.clone() if self.rd is not None else None,
            volatility=self.volatility.clone() if self.volatility is not None else None,
            last_played=self.last_played.clone() if self.last_played is not None else None,
            metadata=self.metadata.copy(),
        )

    def to(self, device: "torch.device") -> "TorchPlayerRatings":
        """Move ratings to specified device."""
        return TorchPlayerRatings(
            ratings=self.ratings.to(device),
            device=device,
            rd=self.rd.to(device) if self.rd is not None else None,
            volatility=self.volatility.to(device) if self.volatility is not None else None,
            last_played=self.last_played.to(device) if self.last_played is not None else None,
            metadata=self.metadata.copy(),
        )

    def to_numpy(self) -> PlayerRatings:
        """Convert to numpy-based PlayerRatings."""
        return PlayerRatings(
            ratings=self.ratings.cpu().numpy(),
            rd=self.rd.cpu().numpy() if self.rd is not None else None,
            volatility=self.volatility.cpu().numpy() if self.volatility is not None else None,
            last_played=self.last_played.cpu().numpy() if self.last_played is not None else None,
            metadata=self.metadata.copy(),
        )
