"""Data types for rating systems."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


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


@dataclass
class PredictionResult:
    """Result of predicting game outcomes."""

    player1: np.ndarray      # (N,) int64 - Player 1 IDs
    player2: np.ndarray      # (N,) int64 - Player 2 IDs
    predicted_proba: np.ndarray  # (N,) float64 - P(player1 wins)
    actual_scores: Optional[np.ndarray] = None  # (N,) float64 - Actual outcomes

    def __len__(self) -> int:
        return len(self.player1)
