"""Elo rating system implementation."""

from dataclasses import dataclass
from typing import Optional

import torch

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch


@dataclass
class EloConfig:
    """Configuration for Elo rating system."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0  # Rating difference for 10x expected score


class Elo(RatingSystem):
    """
    Elo rating system.

    The classic rating system developed by Arpad Elo. Uses a simple
    update rule based on expected vs actual game outcomes.

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        k_factor: Maximum rating change per game (default: 32)
        scale: Rating difference where one player is 10x stronger (default: 400)
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
        super().__init__(num_players=num_players, device=device)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Elo ratings for all players."""
        return PlayerRatings(
            ratings=torch.full(
                (num_players,),
                self.config.initial_rating,
                dtype=torch.float32,
                device=self.device,
            ),
            device=self.device,
            metadata={"system": "elo", "config": self.config},
        )

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
        k = self.config.k_factor

        for i in range(len(batch)):
            p1_idx = batch.player1[i].item()
            p2_idx = batch.player2[i].item()
            score = batch.scores[i].item()

            r1 = ratings.ratings[p1_idx]
            r2 = ratings.ratings[p2_idx]

            e1 = self._expected_score(r1, r2)

            ratings.ratings[p1_idx] = r1 + k * (score - e1)
            ratings.ratings[p2_idx] = r2 + k * ((1.0 - score) - (1.0 - e1))

    def predict_proba(
        self,
        player1: torch.Tensor,
        player2: torch.Tensor,
    ) -> torch.Tensor:
        """Predict probability that player1 beats player2."""
        if self._ratings is None:
            raise ValueError("Model not fitted")

        player1 = player1.to(self.device)
        player2 = player2.to(self.device)

        r1 = self._ratings.ratings[player1]
        r2 = self._ratings.ratings[player2]

        return self._expected_score(r1, r2)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"Elo(k={self.config.k_factor}, initial={self.config.initial_rating}, "
            f"players={players}, {status})"
        )
