"""Container for player ratings."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd
import torch


@dataclass
class PlayerRatings:
    """
    Container for player ratings from a rating system.

    This provides a uniform interface for accessing ratings across
    different systems, regardless of whether they track just a rating
    (Elo) or additional parameters like RD and volatility (Glicko-2).
    """

    ratings: torch.Tensor  # (num_players,) Primary rating
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Optional additional parameters (system-specific)
    rd: Optional[torch.Tensor] = None          # Rating deviation (Glicko, Glicko-2)
    volatility: Optional[torch.Tensor] = None  # Volatility (Glicko-2)
    last_played: Optional[torch.Tensor] = None # Last day played

    # Metadata
    metadata: Dict = field(default_factory=dict)

    @property
    def num_players(self) -> int:
        return len(self.ratings)

    def get_rating(self, player_id: int) -> float:
        """Get rating for a single player."""
        return self.ratings[player_id].item()

    def get_ratings_batch(self, player_ids: torch.Tensor) -> torch.Tensor:
        """Get ratings for a batch of players."""
        return self.ratings[player_ids]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert ratings to a pandas DataFrame."""
        data = {
            "player_id": range(self.num_players),
            "rating": self.ratings.cpu().numpy(),
        }

        if self.rd is not None:
            data["rd"] = self.rd.cpu().numpy()
        if self.volatility is not None:
            data["volatility"] = self.volatility.cpu().numpy()
        if self.last_played is not None:
            data["last_played"] = self.last_played.cpu().numpy()

        return pd.DataFrame(data)

    def clone(self) -> "PlayerRatings":
        """Create a deep copy of the ratings."""
        return PlayerRatings(
            ratings=self.ratings.clone(),
            device=self.device,
            rd=self.rd.clone() if self.rd is not None else None,
            volatility=self.volatility.clone() if self.volatility is not None else None,
            last_played=self.last_played.clone() if self.last_played is not None else None,
            metadata=self.metadata.copy(),
        )

    def to(self, device: torch.device) -> "PlayerRatings":
        """Move ratings to specified device."""
        return PlayerRatings(
            ratings=self.ratings.to(device),
            device=device,
            rd=self.rd.to(device) if self.rd is not None else None,
            volatility=self.volatility.to(device) if self.volatility is not None else None,
            last_played=self.last_played.to(device) if self.last_played is not None else None,
            metadata=self.metadata.copy(),
        )
