"""Elo rating system implementation using PyTorch for GPU acceleration."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...base.player_ratings import TorchPlayerRatings
from ...data import GameBatch
from ...data.types import TorchGameBatch, PredictionResult


def get_device() -> torch.device:
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EloConfig:
    """Configuration for Elo rating system."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0  # Rating difference for 10x expected score


class EloTorch(RatingSystem):
    """
    Elo rating system with PyTorch GPU acceleration.

    The classic rating system developed by Arpad Elo. Uses a simple
    update rule based on expected vs actual game outcomes.

    This implementation uses PyTorch for GPU acceleration on large datasets.

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        k_factor: Maximum rating change per game (default: 32)
        scale: Rating difference where one player is 10x stronger (default: 400)
        device: PyTorch device for computations (default: auto-detect)
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        scale: float = 400.0,
        num_players: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = EloConfig(
            initial_rating=initial_rating,
            k_factor=k_factor,
            scale=scale,
        )
        self.device = device or get_device()
        self._torch_ratings: Optional[TorchPlayerRatings] = None

        # Call parent init (but we override rating storage)
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Elo ratings for all players (numpy for interface)."""
        # Create torch ratings internally
        self._torch_ratings = TorchPlayerRatings(
            ratings=torch.full(
                (num_players,),
                self.config.initial_rating,
                dtype=torch.float32,
                device=self.device,
            ),
            device=self.device,
            metadata={"system": "elo_torch", "config": self.config},
        )
        # Return numpy version for interface compatibility
        return self._torch_ratings.to_numpy()

    def _expected_score(
        self,
        rating_a: torch.Tensor,
        rating_b: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + torch.pow(10, (rating_b - rating_a) / self.config.scale))

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """
        Update Elo ratings for a batch of games.

        Games are processed sequentially within a batch to maintain
        rating consistency when players have multiple games.
        """
        if len(batch) == 0:
            return

        k = self.config.k_factor
        torch_ratings = self._torch_ratings.ratings

        # Convert batch to torch if needed
        player1 = torch.from_numpy(batch.player1).to(self.device)
        player2 = torch.from_numpy(batch.player2).to(self.device)
        scores = torch.from_numpy(batch.scores).float().to(self.device)

        for i in range(len(batch)):
            p1_idx = player1[i].item()
            p2_idx = player2[i].item()
            score = scores[i].item()

            r1 = torch_ratings[p1_idx]
            r2 = torch_ratings[p2_idx]

            e1 = self._expected_score(r1, r2)

            torch_ratings[p1_idx] = r1 + k * (score - e1)
            torch_ratings[p2_idx] = r2 + k * ((1.0 - score) - (1.0 - e1))

        # Sync back to numpy ratings
        ratings.ratings = torch_ratings.cpu().numpy()

    def predict_proba(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
    ) -> np.ndarray:
        """Predict probability that player1 beats player2."""
        if self._torch_ratings is None:
            raise ValueError("Model not fitted")

        p1 = torch.from_numpy(player1).to(self.device)
        p2 = torch.from_numpy(player2).to(self.device)

        r1 = self._torch_ratings.ratings[p1]
        r2 = self._torch_ratings.ratings[p2]

        proba = self._expected_score(r1, r2)
        return proba.cpu().numpy()

    def get_ratings(self) -> PlayerRatings:
        """Get current player ratings."""
        if self._torch_ratings is None:
            raise ValueError("No ratings available. Call fit() first.")
        return self._torch_ratings.to_numpy()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"EloTorch(k={self.config.k_factor}, initial={self.config.initial_rating}, "
            f"device={self.device}, players={players}, {status})"
        )
