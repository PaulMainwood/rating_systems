"""Data types for rating systems."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GameBatch:
    """A batch of games, typically from a single day/rating period."""

    player1: torch.Tensor  # (N,) int64 - Player 1 IDs
    player2: torch.Tensor  # (N,) int64 - Player 2 IDs
    scores: torch.Tensor   # (N,) float32 - Scores (1.0 = P1 wins, 0.0 = P2 wins)
    day: int               # The day/rating period for this batch

    def __len__(self) -> int:
        return len(self.player1)

    def to(self, device: torch.device) -> "GameBatch":
        """Move batch to specified device."""
        return GameBatch(
            player1=self.player1.to(device),
            player2=self.player2.to(device),
            scores=self.scores.to(device),
            day=self.day,
        )


@dataclass
class PredictionResult:
    """Result of predicting game outcomes."""

    player1: torch.Tensor      # (N,) int64 - Player 1 IDs
    player2: torch.Tensor      # (N,) int64 - Player 2 IDs
    predicted_proba: torch.Tensor  # (N,) float32 - P(player1 wins)
    actual_scores: Optional[torch.Tensor] = None  # (N,) float32 - Actual outcomes

    def __len__(self) -> int:
        return len(self.player1)
