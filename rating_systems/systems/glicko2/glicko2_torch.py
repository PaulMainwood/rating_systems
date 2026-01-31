"""Glicko-2 rating system implementation using PyTorch for GPU acceleration."""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...base.player_ratings import TorchPlayerRatings
from ...data import GameBatch


def get_device() -> torch.device:
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Glicko2Config:
    """Configuration for Glicko-2 rating system."""

    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    initial_volatility: float = 0.06
    tau: float = 0.5  # System constant (typically 0.3 to 1.2)
    epsilon: float = 0.000001  # Convergence tolerance
    scale: float = 173.7178  # Conversion factor from Glicko to Glicko-2 scale
    max_rd: float = 350.0


class Glicko2Torch(RatingSystem):
    """
    Glicko-2 rating system with PyTorch GPU acceleration.

    Extension of Glicko that adds a volatility parameter to model
    rating stability. Uses internal Glicko-2 scale for calculations.

    This implementation uses PyTorch for GPU acceleration on large datasets.

    Parameters:
        initial_rating: Starting rating for new players (default: 1500)
        initial_rd: Starting rating deviation (default: 350)
        initial_volatility: Starting volatility (default: 0.06)
        tau: System constant controlling volatility change (default: 0.5)
        device: PyTorch device for computations (default: auto-detect)
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        initial_volatility: float = 0.06,
        tau: float = 0.5,
        num_players: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = Glicko2Config(
            initial_rating=initial_rating,
            initial_rd=initial_rd,
            initial_volatility=initial_volatility,
            tau=tau,
        )
        self.device = device or get_device()
        self._torch_ratings: Optional[TorchPlayerRatings] = None

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial Glicko-2 ratings for all players."""
        # Store internally in Glicko-2 scale (mu, phi)
        self._torch_ratings = TorchPlayerRatings(
            ratings=torch.zeros(num_players, dtype=torch.float32, device=self.device),  # mu
            rd=torch.full(
                (num_players,),
                self.config.initial_rd / self.config.scale,  # phi
                dtype=torch.float32,
                device=self.device,
            ),
            volatility=torch.full(
                (num_players,),
                self.config.initial_volatility,
                dtype=torch.float32,
                device=self.device,
            ),
            last_played=torch.zeros(num_players, dtype=torch.int32, device=self.device),
            device=self.device,
            metadata={"system": "glicko2_torch", "config": self.config},
        )
        return self._torch_ratings.to_numpy()

    def _to_glicko_scale(self, mu: torch.Tensor) -> torch.Tensor:
        """Convert mu from Glicko-2 scale to Glicko scale."""
        return mu * self.config.scale + self.config.initial_rating

    def _to_glicko_rd(self, phi: torch.Tensor) -> torch.Tensor:
        """Convert phi from Glicko-2 scale to Glicko RD."""
        return phi * self.config.scale

    def _g(self, phi: torch.Tensor) -> torch.Tensor:
        """Calculate g(phi) function."""
        return 1.0 / torch.sqrt(1.0 + 3.0 * (phi ** 2) / (math.pi ** 2))

    def _expected_score(
        self,
        mu: torch.Tensor,
        opp_mu: torch.Tensor,
        opp_phi: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate expected score in Glicko-2 scale."""
        g_phi = self._g(opp_phi)
        return 1.0 / (1.0 + torch.exp(-g_phi * (mu - opp_mu)))

    def _update_volatility(
        self,
        sigma: float,
        phi: float,
        v: float,
        delta: float,
    ) -> float:
        """Update volatility using iterative algorithm (Step 5)."""
        tau = self.config.tau
        epsilon = self.config.epsilon

        a = math.log(sigma ** 2)
        phi_sq = phi ** 2
        delta_sq = delta ** 2

        def f(x: float) -> float:
            ex = math.exp(x)
            num1 = ex * (delta_sq - phi_sq - v - ex)
            den1 = 2 * ((phi_sq + v + ex) ** 2)
            return num1 / den1 - (x - a) / (tau ** 2)

        # Set initial bounds
        A = a
        if delta_sq > phi_sq + v:
            B = math.log(delta_sq - phi_sq - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
                if k > 100:  # Safety limit
                    break
            B = a - k * tau

        # Iterative algorithm
        f_A = f(A)
        f_B = f(B)

        iterations = 0
        max_iterations = 100

        while abs(B - A) > epsilon and iterations < max_iterations:
            C = A + (A - B) * f_A / (f_B - f_A)
            f_C = f(C)

            if f_C * f_B <= 0:
                A = B
                f_A = f_B
            else:
                f_A = f_A / 2

            B = C
            f_B = f_C
            iterations += 1

        return math.exp(A / 2)

    def _update_phi_for_inactivity(
        self,
        player_indices: torch.Tensor,
        current_day: int,
    ) -> None:
        """Update phi (RD) for rating period decay."""
        days_inactive = current_day - self._torch_ratings.last_played[player_indices]
        days_inactive = days_inactive.float().clamp(min=0)

        current_phi = self._torch_ratings.rd[player_indices]
        current_sigma = self._torch_ratings.volatility[player_indices]

        # phi* = sqrt(phi^2 + sigma^2 * days)
        new_phi = torch.sqrt(current_phi ** 2 + (current_sigma ** 2) * days_inactive)
        max_phi = self.config.max_rd / self.config.scale
        new_phi = new_phi.clamp(max=max_phi)

        self._torch_ratings.rd[player_indices] = new_phi

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update Glicko-2 ratings for a rating period."""
        if len(batch) == 0:
            return

        current_day = batch.day

        # Convert batch to torch
        player1 = torch.from_numpy(batch.player1).to(self.device)
        player2 = torch.from_numpy(batch.player2).to(self.device)
        scores = torch.from_numpy(batch.scores).float().to(self.device)

        # Get unique players
        all_players = torch.cat([player1, player2]).unique()

        # Update phi for inactivity (Step 1)
        self._update_phi_for_inactivity(all_players, current_day)

        # Store pre-period values
        pre_mu = self._torch_ratings.ratings.clone()
        pre_phi = self._torch_ratings.rd.clone()
        pre_sigma = self._torch_ratings.volatility.clone()

        # Process each player
        for player in all_players:
            player_idx = player.item()

            # Find games
            as_p1 = player1 == player
            as_p2 = player2 == player

            if not (as_p1.any() or as_p2.any()):
                continue

            # Collect opponents and scores
            opponents = []
            player_scores = []

            if as_p1.any():
                opponents.append(player2[as_p1])
                player_scores.append(scores[as_p1])

            if as_p2.any():
                opponents.append(player1[as_p2])
                player_scores.append(1.0 - scores[as_p2])

            opponents = torch.cat(opponents)
            player_scores = torch.cat(player_scores)

            # Get opponent values
            opp_mus = pre_mu[opponents]
            opp_phis = pre_phi[opponents]

            # Step 3: Compute variance v
            g_vals = self._g(opp_phis)
            e_vals = self._expected_score(
                pre_mu[player_idx].expand(len(opponents)),
                opp_mus,
                opp_phis,
            )

            v_inv = torch.sum(g_vals ** 2 * e_vals * (1 - e_vals))
            v = (1.0 / v_inv).item() if v_inv > 0 else 1e10

            # Step 4: Compute delta
            delta = v * torch.sum(g_vals * (player_scores - e_vals)).item()

            # Step 5: Update volatility
            sigma = pre_sigma[player_idx].item()
            phi = pre_phi[player_idx].item()
            new_sigma = self._update_volatility(sigma, phi, v, delta)

            # Step 6: Update phi*
            phi_star = math.sqrt(phi ** 2 + new_sigma ** 2)

            # Step 7: Update rating and RD
            new_phi = 1.0 / math.sqrt(1.0 / (phi_star ** 2) + 1.0 / v)
            delta_sum = torch.sum(g_vals * (player_scores - e_vals)).item()
            new_mu = pre_mu[player_idx].item() + (new_phi ** 2) * delta_sum

            # Update state
            self._torch_ratings.ratings[player_idx] = new_mu
            self._torch_ratings.rd[player_idx] = new_phi
            self._torch_ratings.volatility[player_idx] = new_sigma
            self._torch_ratings.last_played[player_idx] = current_day

        # Sync back to numpy ratings
        ratings.ratings = self._torch_ratings.ratings.cpu().numpy()
        ratings.rd = self._torch_ratings.rd.cpu().numpy()
        ratings.volatility = self._torch_ratings.volatility.cpu().numpy()
        ratings.last_played = self._torch_ratings.last_played.cpu().numpy()

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

        mu1 = self._torch_ratings.ratings[p1]
        mu2 = self._torch_ratings.ratings[p2]
        phi1 = self._torch_ratings.rd[p1]
        phi2 = self._torch_ratings.rd[p2]

        # Combined phi for prediction
        combined_phi = torch.sqrt(phi1 ** 2 + phi2 ** 2)
        g_combined = self._g(combined_phi)

        proba = 1.0 / (1.0 + torch.exp(-g_combined * (mu1 - mu2)))
        return proba.cpu().numpy()

    def get_ratings(self) -> PlayerRatings:
        """Get current player ratings in Glicko scale."""
        if self._torch_ratings is None:
            raise ValueError("No ratings available. Call fit() first.")

        # Convert to Glicko scale for output
        return PlayerRatings(
            ratings=self._to_glicko_scale(self._torch_ratings.ratings).cpu().numpy(),
            rd=self._to_glicko_rd(self._torch_ratings.rd).cpu().numpy(),
            volatility=self._torch_ratings.volatility.cpu().numpy(),
            last_played=self._torch_ratings.last_played.cpu().numpy(),
            metadata={"system": "glicko2_torch", "config": self.config},
        )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"Glicko2Torch(tau={self.config.tau}, "
            f"initial_volatility={self.config.initial_volatility}, "
            f"device={self.device}, players={players}, {status})"
        )
