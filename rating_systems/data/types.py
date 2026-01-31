"""Data types for rating systems (numpy-based for broad compatibility)."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

# Optional torch support for torch-based systems
if TYPE_CHECKING:
    import torch


@dataclass
class GameBatch:
    """A batch of games, typically from a single day/rating period."""

    player1: np.ndarray  # (N,) int64 - Player 1 IDs
    player2: np.ndarray  # (N,) int64 - Player 2 IDs
    scores: np.ndarray   # (N,) float64 - Scores (1.0 = P1 wins, 0.0 = P2 wins)
    day: int             # The day/rating period for this batch

    def __len__(self) -> int:
        return len(self.player1)

    def __post_init__(self):
        """Ensure arrays are contiguous and correct dtype for Numba compatibility."""
        self.player1 = np.ascontiguousarray(self.player1, dtype=np.int64)
        self.player2 = np.ascontiguousarray(self.player2, dtype=np.int64)
        self.scores = np.ascontiguousarray(self.scores, dtype=np.float64)

    def to_torch(self, device: Optional["torch.device"] = None) -> "TorchGameBatch":
        """Convert to PyTorch tensors for torch-based systems."""
        import torch
        return TorchGameBatch(
            player1=torch.from_numpy(self.player1).to(device or torch.device("cpu")),
            player2=torch.from_numpy(self.player2).to(device or torch.device("cpu")),
            scores=torch.from_numpy(self.scores).float().to(device or torch.device("cpu")),
            day=self.day,
        )


@dataclass
class TorchGameBatch:
    """A batch of games using PyTorch tensors (for torch-based systems)."""

    player1: "torch.Tensor"  # (N,) int64 - Player 1 IDs
    player2: "torch.Tensor"  # (N,) int64 - Player 2 IDs
    scores: "torch.Tensor"   # (N,) float32 - Scores
    day: int

    def __len__(self) -> int:
        return len(self.player1)

    def to(self, device: "torch.device") -> "TorchGameBatch":
        """Move batch to specified device."""
        return TorchGameBatch(
            player1=self.player1.to(device),
            player2=self.player2.to(device),
            scores=self.scores.to(device),
            day=self.day,
        )

    def to_numpy(self) -> GameBatch:
        """Convert to numpy arrays."""
        return GameBatch(
            player1=self.player1.cpu().numpy(),
            player2=self.player2.cpu().numpy(),
            scores=self.scores.cpu().numpy(),
            day=self.day,
        )


@dataclass
class PredictionResult:
    """Result of predicting game outcomes."""

    player1: np.ndarray      # (N,) int64 - Player 1 IDs
    player2: np.ndarray      # (N,) int64 - Player 2 IDs
    predicted_proba: np.ndarray  # (N,) float64 - P(player1 wins)
    actual_scores: Optional[np.ndarray] = None  # (N,) float64 - Actual outcomes

    def __len__(self) -> int:
        return len(self.player1)
