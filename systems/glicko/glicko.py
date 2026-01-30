"""Glicko rating system implementation."""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch


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
    Glicko rating system.

    Extension of Elo that adds Rating Deviation (RD) to model uncertainty.
    RD decreases when playing games and increases during inactivity.

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        initial_rd: Starting rating deviation (default: 350)
        min_rd: Minimum rating deviation (default: 30)
        max_rd: Maximum rating deviation (default: 350)
        c: RD increase per period of inactivity (default: 34.6)
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
        device: Optional[torch.device] = None,
    ):
        self.config = GlickoConfig(
            initial_rating=initial_rating,
            initial_rd=initial_rd,
            min_rd=min_rd,
            max_rd=max_rd,
            c=c,
        )
        super().__init__(num_players=num_players, device=device)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Glicko ratings for all players."""
        return PlayerRatings(
            ratings=torch.full(
                (num_players,),
                self.config.initial_rating,
                dtype=torch.float32,
                device=self.device,
            ),
            rd=torch.full(
                (num_players,),
                self.config.initial_rd,
                dtype=torch.float32,
                device=self.device,
            ),
            last_played=torch.zeros(num_players, dtype=torch.int32, device=self.device),
            device=self.device,
            metadata={"system": "glicko", "config": self.config},
        )

    def _g(self, rd: torch.Tensor) -> torch.Tensor:
        """Calculate g(RD) function."""
        q = self.config.q
        return 1.0 / torch.sqrt(1.0 + 3.0 * (q ** 2) * (rd ** 2) / (math.pi ** 2))

    def _expected_score(
        self,
        rating: torch.Tensor,
        opp_rating: torch.Tensor,
        opp_rd: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate expected score accounting for opponent's RD."""
        g_rd = self._g(opp_rd)
        exponent = -g_rd * (rating - opp_rating) / 400.0
        return 1.0 / (1.0 + torch.pow(10, exponent))

    def _update_rd_for_inactivity(
        self,
        ratings: PlayerRatings,
        player_indices: torch.Tensor,
        current_day: int,
    ) -> None:
        """Increase RD for players based on time since last game."""
        days_inactive = current_day - ratings.last_played[player_indices]
        days_inactive = days_inactive.float().clamp(min=0)

        current_rd = ratings.rd[player_indices]
        new_rd = torch.sqrt(current_rd ** 2 + (self.config.c ** 2) * days_inactive)
        new_rd = new_rd.clamp(min=self.config.min_rd, max=self.config.max_rd)

        ratings.rd[player_indices] = new_rd

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """
        Update Glicko ratings for a rating period (batch of games).

        All games in a batch are considered simultaneous. Ratings are
        updated based on pre-period values, then RD is reduced.
        """
        if len(batch) == 0:
            return

        current_day = batch.day
        q = self.config.q

        # Get unique players in this period
        all_players = torch.cat([batch.player1, batch.player2]).unique()

        # Update RD for inactivity before processing
        self._update_rd_for_inactivity(ratings, all_players, current_day)

        # Store pre-period ratings and RDs
        pre_ratings = ratings.ratings.clone()
        pre_rd = ratings.rd.clone()

        # Process each player's games in this period
        for player in all_players:
            player_idx = player.item()

            # Find games where this player participated
            as_p1 = batch.player1 == player
            as_p2 = batch.player2 == player

            if not (as_p1.any() or as_p2.any()):
                continue

            # Collect opponents and scores
            opponents = []
            player_scores = []

            if as_p1.any():
                opponents.append(batch.player2[as_p1])
                player_scores.append(batch.scores[as_p1])

            if as_p2.any():
                opponents.append(batch.player1[as_p2])
                player_scores.append(1.0 - batch.scores[as_p2])

            opponents = torch.cat(opponents)
            player_scores = torch.cat(player_scores)

            # Get opponent ratings and RDs (pre-period values)
            opp_ratings = pre_ratings[opponents]
            opp_rds = pre_rd[opponents]

            # Calculate d^2 (variance)
            g_vals = self._g(opp_rds)
            e_vals = self._expected_score(
                pre_ratings[player_idx].expand(len(opponents)),
                opp_ratings,
                opp_rds,
            )

            d_squared_inv = (q ** 2) * torch.sum(g_vals ** 2 * e_vals * (1 - e_vals))

            if d_squared_inv > 0:
                d_squared = 1.0 / d_squared_inv
            else:
                d_squared = torch.tensor(1e10, device=self.device)

            # Calculate new rating
            rd_squared = pre_rd[player_idx] ** 2
            new_rd_squared = 1.0 / (1.0 / rd_squared + 1.0 / d_squared)

            sum_term = torch.sum(g_vals * (player_scores - e_vals))
            rating_change = q * new_rd_squared * sum_term

            # Update state
            ratings.ratings[player_idx] = pre_ratings[player_idx] + rating_change
            ratings.rd[player_idx] = torch.sqrt(new_rd_squared).clamp(
                min=self.config.min_rd, max=self.config.max_rd
            )
            ratings.last_played[player_idx] = current_day

    def predict_proba(
        self,
        player1: torch.Tensor,
        player2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict probability that player1 beats player2.

        Uses combined RD of both players in the calculation.
        """
        if self._ratings is None:
            raise ValueError("Model not fitted")

        player1 = player1.to(self.device)
        player2 = player2.to(self.device)

        r1 = self._ratings.ratings[player1]
        r2 = self._ratings.ratings[player2]
        rd1 = self._ratings.rd[player1]
        rd2 = self._ratings.rd[player2]

        # Combined RD for prediction
        combined_rd = torch.sqrt(rd1 ** 2 + rd2 ** 2)
        g_combined = self._g(combined_rd)

        exponent = -g_combined * (r1 - r2) / 400.0
        return 1.0 / (1.0 + torch.pow(10, exponent))

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"Glicko(initial_rating={self.config.initial_rating}, "
            f"initial_rd={self.config.initial_rd}, players={players}, {status})"
        )
