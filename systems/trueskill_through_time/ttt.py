"""
TrueSkill Through Time (TTT) implementation in PyTorch.

Based on:
- Dangauthier et al., "TrueSkill Through Time: Revisiting the History of Chess" (2007)
- Glandfried's Python implementation: https://github.com/glandfried/TrueSkillThroughTime.py

TTT extends TrueSkill by modeling player skill as evolving over time using
Gaussian belief propagation with forward-backward message passing.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from .gaussian import cdf


@dataclass
class TTTConfig:
    """Configuration for TrueSkill Through Time."""

    mu: float = 0.0  # Prior mean skill (internal scale, 0 = average)
    sigma: float = 1.5  # Prior skill std dev (internal scale)
    beta: float = 0.5  # Performance std dev (within-game noise)
    gamma: float = 0.01  # Skill dynamics per time unit
    p_draw: float = 0.0  # Draw probability
    max_iterations: int = 30  # Max forward-backward iterations
    convergence_threshold: float = 1e-4  # Convergence threshold


@dataclass
class PlayerSkill:
    """Skill state for a player at a specific time."""

    day: int
    day_idx: int

    # Gaussian parameters (mu, sigma form for stability)
    prior_mu: float = 0.0
    prior_sigma: float = 6.0

    # Forward message (from past)
    forward_mu: float = 0.0
    forward_sigma: float = 1e6  # Large = uninformative

    # Backward message (from future)
    backward_mu: float = 0.0
    backward_sigma: float = 1e6

    # Likelihood from games (in precision form)
    likelihood_pi: float = 0.0  # Precision from games
    likelihood_tau: float = 0.0  # Precision-weighted mean from games

    # Games at this time: list of (opponent_id, opp_day_idx, score)
    games: List[Tuple[int, int, float]] = field(default_factory=list)

    def _get_prior_prec_tau(self) -> Tuple[float, float]:
        """Get precision and tau from forward + backward only (no likelihood)."""
        fwd_prec = 1.0 / (self.forward_sigma ** 2) if self.forward_sigma < 1e5 else 0.0
        bwd_prec = 1.0 / (self.backward_sigma ** 2) if self.backward_sigma < 1e5 else 0.0

        total_prec = fwd_prec + bwd_prec
        if total_prec < 1e-10:
            return 1.0 / (self.prior_sigma ** 2), self.prior_mu / (self.prior_sigma ** 2)

        total_tau = fwd_prec * self.forward_mu + bwd_prec * self.backward_mu
        return total_prec, total_tau

    @property
    def prior_from_messages_mu(self) -> float:
        """Mean from forward+backward messages only."""
        prec, tau = self._get_prior_prec_tau()
        return tau / prec if prec > 1e-10 else self.prior_mu

    @property
    def prior_from_messages_sigma(self) -> float:
        """Sigma from forward+backward messages only."""
        prec, _ = self._get_prior_prec_tau()
        return 1.0 / math.sqrt(prec) if prec > 1e-10 else self.prior_sigma

    @property
    def mu(self) -> float:
        """Posterior mean (messages + likelihood)."""
        prec, tau = self._get_prior_prec_tau()

        # Add likelihood contribution
        tau += self.likelihood_tau
        prec += self.likelihood_pi

        if prec < 1e-10:
            return self.prior_mu
        return tau / prec

    @property
    def sigma(self) -> float:
        """Posterior std dev (messages + likelihood)."""
        prec, _ = self._get_prior_prec_tau()
        prec += self.likelihood_pi

        if prec < 1e-10:
            return self.prior_sigma
        return 1.0 / math.sqrt(prec)


class TrueSkillThroughTime(RatingSystem):
    """
    TrueSkill Through Time rating system.

    TTT models player skill as a Gaussian that evolves over time:
    - Skill at time t: N(mu_t, sigma_t^2)
    - Skill drift: sigma increases by gamma per time unit
    - Game outcomes update beliefs via Gaussian message passing

    Uses forward-backward belief propagation for globally consistent estimates.

    Parameters:
        mu: Prior mean skill (default: 0.0, displayed as 25.0)
        sigma: Prior skill std dev (default: 6.0)
        beta: Performance variability within games (default: 1.0)
        gamma: Skill drift rate per time unit (default: 0.03)
        p_draw: Draw probability (default: 0.0)
        max_iterations: Max belief propagation iterations (default: 30)
    """

    system_type = RatingSystemType.BATCH

    # Scale factor for display (internal 0 = display 1500, like Elo)
    DISPLAY_OFFSET = 1500.0
    DISPLAY_SCALE = 400.0 / 1.5  # Map 1.5 internal sigma to ~267 Elo points

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.5,
        beta: float = 0.5,
        gamma: float = 0.01,
        p_draw: float = 0.0,
        max_iterations: int = 30,
        convergence_threshold: float = 1e-4,
        num_players: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = TTTConfig(
            mu=mu,
            sigma=sigma,
            beta=beta,
            gamma=gamma,
            p_draw=p_draw,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )

        # Player skill timelines: player_id -> list of PlayerSkill
        self._player_skills: Dict[int, List[PlayerSkill]] = {}

        # Index: (player_id, day) -> day_idx
        self._day_index: Dict[Tuple[int, int], int] = {}

        # All games for refitting
        self._all_games: List[GameBatch] = []

        super().__init__(num_players=num_players, device=device)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial TTT ratings."""
        return PlayerRatings(
            ratings=torch.full(
                (num_players,),
                self.DISPLAY_OFFSET,
                dtype=torch.float32,
                device=self.device,
            ),
            rd=torch.full(
                (num_players,),
                self.config.sigma * self.DISPLAY_SCALE,
                dtype=torch.float32,
                device=self.device,
            ),
            device=self.device,
            metadata={"system": "ttt", "config": self.config},
        )

    def _get_or_create_skill(self, player_id: int, day: int) -> PlayerSkill:
        """Get or create a PlayerSkill for a player on a specific day."""
        key = (player_id, day)
        if key in self._day_index:
            idx = self._day_index[key]
            return self._player_skills[player_id][idx]

        if player_id not in self._player_skills:
            self._player_skills[player_id] = []

        # Find insertion point
        skills_list = self._player_skills[player_id]
        insert_idx = 0
        for i, sk in enumerate(skills_list):
            if sk.day > day:
                break
            insert_idx = i + 1

        # Create new skill with prior
        new_skill = PlayerSkill(
            day=day,
            day_idx=insert_idx,
            prior_mu=self.config.mu,
            prior_sigma=self.config.sigma,
        )
        skills_list.insert(insert_idx, new_skill)

        # Update indices
        for i in range(insert_idx, len(skills_list)):
            skills_list[i].day_idx = i
            self._day_index[(player_id, skills_list[i].day)] = i

        return new_skill

    def _build_graph(self, batches: List[GameBatch]) -> None:
        """Build the game graph from batches."""
        self._player_skills.clear()
        self._day_index.clear()

        # First pass: create skill nodes
        for batch in batches:
            day = batch.day
            for i in range(len(batch)):
                p1 = batch.player1[i].item()
                p2 = batch.player2[i].item()
                self._get_or_create_skill(p1, day)
                self._get_or_create_skill(p2, day)

        # Second pass: add games
        for batch in batches:
            day = batch.day
            for i in range(len(batch)):
                p1 = batch.player1[i].item()
                p2 = batch.player2[i].item()
                score = batch.scores[i].item()

                p1_idx = self._day_index[(p1, day)]
                p2_idx = self._day_index[(p2, day)]

                self._player_skills[p1][p1_idx].games.append((p2, p2_idx, score))
                self._player_skills[p2][p2_idx].games.append((p1, p1_idx, 1.0 - score))

    def _compute_drift_sigma(self, elapsed: int) -> float:
        """Compute additional sigma due to skill drift."""
        if elapsed <= 0:
            return 0.0
        gamma = self.config.gamma
        sigma = self.config.sigma
        # Cap drift at 1.67 * sigma
        return min(math.sqrt(elapsed) * gamma, 1.67 * sigma)

    def _add_variances(self, sigma1: float, sigma2: float) -> float:
        """Add two independent Gaussian variances."""
        return math.sqrt(sigma1 ** 2 + sigma2 ** 2)

    def _forward_pass(self) -> float:
        """Run forward pass: propagate beliefs from past to future."""
        max_change = 0.0

        for player_id, skills_list in self._player_skills.items():
            # Start with prior
            prev_mu = self.config.mu
            prev_sigma = self.config.sigma
            prev_day = skills_list[0].day if skills_list else 0

            for skill in skills_list:
                # Apply drift
                elapsed = skill.day - prev_day
                drift_sigma = self._compute_drift_sigma(elapsed)
                msg_sigma = self._add_variances(prev_sigma, drift_sigma)

                # Update forward message
                old_mu = skill.forward_mu
                skill.forward_mu = prev_mu
                skill.forward_sigma = msg_sigma

                max_change = max(max_change, abs(skill.forward_mu - old_mu))

                # Prepare for next: use current posterior
                prev_mu = skill.mu
                prev_sigma = skill.sigma
                prev_day = skill.day

        return max_change

    def _backward_pass(self) -> float:
        """Run backward pass: propagate beliefs from future to past."""
        max_change = 0.0

        for player_id, skills_list in self._player_skills.items():
            # Start uninformative
            prev_mu = 0.0
            prev_sigma = 1e6
            prev_day = skills_list[-1].day if skills_list else 0

            for skill in reversed(skills_list):
                # Apply drift
                elapsed = prev_day - skill.day
                drift_sigma = self._compute_drift_sigma(elapsed)
                msg_sigma = self._add_variances(prev_sigma, drift_sigma)

                # Update backward message
                old_mu = skill.backward_mu
                skill.backward_mu = prev_mu
                skill.backward_sigma = msg_sigma

                max_change = max(max_change, abs(skill.backward_mu - old_mu))

                # Prepare for next
                prev_mu = skill.mu
                prev_sigma = skill.sigma
                prev_day = skill.day

        return max_change

    def _v_function(self, t: float, eps: float = 1e-10) -> float:
        """Compute v = pdf(t) / cdf(-t) for truncated Gaussian."""
        from scipy.stats import norm
        denom = norm.cdf(-t)
        if denom < eps:
            return -t + 1.0 / (-t) if t < -5 else 10.0
        return norm.pdf(t) / denom

    def _w_function(self, t: float, v: float = None) -> float:
        """Compute w = v * (v + t)."""
        if v is None:
            v = self._v_function(t)
        w = v * (v + t)
        return max(0.0, min(w, 1.0 - 1e-6))

    def _update_likelihoods(self) -> float:
        """Update likelihood messages from games."""
        max_change = 0.0
        beta = self.config.beta

        for player_id, skills_list in self._player_skills.items():
            for skill in skills_list:
                if not skill.games:
                    skill.likelihood_pi = 0.0
                    skill.likelihood_tau = 0.0
                    continue

                # Use prior from messages only (not including current likelihood)
                my_mu = skill.prior_from_messages_mu
                my_sigma = skill.prior_from_messages_sigma
                my_var = my_sigma ** 2

                # Aggregate likelihood updates
                total_pi = 0.0
                total_tau = 0.0

                for opp_id, opp_day_idx, score in skill.games:
                    opp_skill = self._player_skills[opp_id][opp_day_idx]
                    opp_mu = opp_skill.prior_from_messages_mu
                    opp_var = opp_skill.prior_from_messages_sigma ** 2

                    # Performance difference: d ~ N(mu1-mu2, var1+var2+2*beta^2)
                    diff_mu = my_mu - opp_mu
                    diff_var = my_var + opp_var + 2 * beta**2
                    diff_sigma = math.sqrt(diff_var)

                    if diff_sigma < 1e-10:
                        continue

                    t = diff_mu / diff_sigma

                    if score > 0.5:  # Win
                        v = self._v_function(t)
                        w = self._w_function(t, v)
                    else:  # Loss
                        v = -self._v_function(-t)
                        w = self._w_function(-t)

                    # Clamp w for stability
                    w = max(1e-6, min(w, 1.0 - 1e-6))

                    # My contribution factor
                    c = (my_var + beta**2) / diff_var

                    # Truncated Gaussian update
                    new_mu = my_mu + c * diff_sigma * v
                    new_var = my_var * (1 - c * w)
                    new_var = max(new_var, 1e-6)

                    # Convert to precision form for this game's contribution
                    game_pi = 1.0 / new_var - 1.0 / my_var
                    game_tau = new_mu / new_var - my_mu / my_var

                    # Clamp for stability
                    game_pi = max(0.0, min(game_pi, 10.0))
                    game_tau = max(-10.0, min(game_tau, 10.0))

                    total_pi += game_pi
                    total_tau += game_tau

                old_pi = skill.likelihood_pi
                skill.likelihood_pi = total_pi
                skill.likelihood_tau = total_tau

                max_change = max(max_change, abs(skill.likelihood_pi - old_pi))

        return max_change

    def _run_iterations(self) -> None:
        """Run forward-backward iterations until convergence."""
        for iteration in range(self.config.max_iterations):
            fwd_change = self._forward_pass()
            lik_change = self._update_likelihoods()
            bwd_change = self._backward_pass()
            lik_change2 = self._update_likelihoods()

            max_change = max(fwd_change, bwd_change, lik_change, lik_change2)

            if max_change < self.config.convergence_threshold:
                break

    def _extract_current_ratings(self) -> None:
        """Extract the most recent rating for each player."""
        if self._num_players is None:
            return

        ratings = torch.full(
            (self._num_players,),
            self.DISPLAY_OFFSET,
            dtype=torch.float32,
            device=self.device,
        )
        uncertainties = torch.full(
            (self._num_players,),
            self.config.sigma * self.DISPLAY_SCALE,
            dtype=torch.float32,
            device=self.device,
        )

        for player_id, skills_list in self._player_skills.items():
            if skills_list:
                last_skill = skills_list[-1]
                # Convert to display scale
                ratings[player_id] = last_skill.mu * self.DISPLAY_SCALE + self.DISPLAY_OFFSET
                uncertainties[player_id] = last_skill.sigma * self.DISPLAY_SCALE

        self._ratings = PlayerRatings(
            ratings=ratings,
            rd=uncertainties,
            device=self.device,
            metadata={"system": "ttt", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with a new batch (refits on all data)."""
        self._all_games.append(batch)
        self._refit()

    def _refit(self) -> None:
        """Refit on all stored games."""
        self._build_graph(self._all_games)
        self._run_iterations()
        self._extract_current_ratings()

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
    ) -> "TrueSkillThroughTime":
        """Fit TTT on a dataset."""
        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        self._all_games = list(dataset.iter_days())
        self._refit()

        self._fitted = True
        if self._all_games:
            self._current_day = max(b.day for b in self._all_games)

        return self

    def update(self, batch: GameBatch) -> "TrueSkillThroughTime":
        """Update with new games by refitting on all data."""
        if not self._fitted:
            raise ValueError("Model must be fitted before updating")

        batch = batch.to(self.device)
        self._all_games.append(batch)
        self._refit()
        self._current_day = batch.day

        return self

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

        # Ratings are in display scale, convert back for prediction
        mu1 = (self._ratings.ratings[player1] - self.DISPLAY_OFFSET) / self.DISPLAY_SCALE
        mu2 = (self._ratings.ratings[player2] - self.DISPLAY_OFFSET) / self.DISPLAY_SCALE
        sigma1 = self._ratings.rd[player1] / self.DISPLAY_SCALE
        sigma2 = self._ratings.rd[player2] / self.DISPLAY_SCALE

        beta = self.config.beta

        # diff ~ N(mu1 - mu2, sigma1^2 + sigma2^2 + 2*beta^2)
        diff_mu = mu1 - mu2
        diff_sigma = torch.sqrt(sigma1**2 + sigma2**2 + 2 * beta**2)

        # P(player1 wins) = P(diff > 0)
        return cdf(diff_mu / diff_sigma)

    def get_rating_history(self, player_id: int) -> Optional[Dict]:
        """Get the full rating history for a player."""
        if player_id not in self._player_skills:
            return None

        skills_list = self._player_skills[player_id]
        if not skills_list:
            return None

        return {
            "days": [sk.day for sk in skills_list],
            "ratings": [sk.mu * self.DISPLAY_SCALE + self.DISPLAY_OFFSET for sk in skills_list],
            "uncertainties": [sk.sigma * self.DISPLAY_SCALE for sk in skills_list],
        }

    def reset(self) -> "TrueSkillThroughTime":
        """Reset the rating system."""
        self._player_skills.clear()
        self._day_index.clear()
        self._all_games.clear()
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"TrueSkillThroughTime(sigma={self.config.sigma:.2f}, "
            f"beta={self.config.beta:.2f}, gamma={self.config.gamma:.3f}, "
            f"players={players}, {status})"
        )
