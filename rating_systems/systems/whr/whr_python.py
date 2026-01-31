"""
Whole History Rating (WHR) - Pure Python implementation.

This is the original pure Python implementation, preserved for reference
and comparison. For high-performance use, see whr.py which uses Numba.

Based on:
- Rémi Coulom, "Whole-History Rating: A Bayesian Rating System for Players
  of Time-Varying Strength" (2008)
- https://www.remi-coulom.fr/WHR/WHR.pdf

WHR models player ratings as a Wiener process (Brownian motion) over time,
using all historical game data to estimate ratings at any point in time.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset


@dataclass
class WHRConfig:
    """Configuration for WHR rating system."""

    w2: float = 300.0  # Wiener variance per time unit (Elo² per day)
    initial_rating: float = 1500.0  # Initial Elo-scale rating
    max_iterations: int = 50  # Maximum Newton-Raphson iterations
    convergence_threshold: float = 1e-6  # Convergence threshold


@dataclass
class PlayerDay:
    """A single day in a player's timeline."""

    day: int
    day_idx: int  # Index in player's timeline
    r: float = 0.0  # Log-gamma rating
    uncertainty: float = 350.0

    # Games: list of (opponent_id, opponent_day_idx, score)
    games: List[Tuple[int, int, float]] = field(default_factory=list)


class WHRPython(RatingSystem):
    """
    Whole History Rating system - Pure Python implementation.

    This is the reference implementation using pure Python data structures.
    For high-performance use, see WHR in whr.py which uses Numba.

    WHR is a Bayesian rating system that:
    1. Models player strength as a Wiener process (random walk) over time
    2. Uses Bradley-Terry model for game outcomes
    3. Finds MAP estimates via Newton-Raphson optimization
    4. Computes uncertainty from the Hessian

    This is a BATCH system - it must refit on all historical data.

    Parameters:
        w2: Wiener variance per time unit (default: 300.0)
            Higher values allow ratings to change more quickly.
            In Elo² units per day.
        initial_rating: Starting rating in Elo scale (default: 1500)
        max_iterations: Maximum Newton-Raphson iterations (default: 50)
        convergence_threshold: Stop when max rating change < threshold
    """

    system_type = RatingSystemType.BATCH

    # Conversion constant: r = elo * LN10_400
    LN10_400 = math.log(10) / 400.0

    def __init__(
        self,
        w2: float = 300.0,
        initial_rating: float = 1500.0,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-6,
        num_players: Optional[int] = None,
    ):
        self.config = WHRConfig(
            w2=w2,
            initial_rating=initial_rating,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )

        # Player timelines: player_id -> list of PlayerDay
        self._player_days: Dict[int, List[PlayerDay]] = {}

        # Quick lookup: (player_id, day) -> day_idx
        self._day_index: Dict[Tuple[int, int], int] = {}

        # Store all games for refitting
        self._all_games: List[GameBatch] = []

        # w2 in log-gamma scale
        self._w2_r = w2 * (self.LN10_400 ** 2)

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial WHR ratings."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            rd=np.full(num_players, 350.0, dtype=np.float64),
            metadata={"system": "whr", "config": self.config},
        )

    def _get_or_create_player_day(self, player_id: int, day: int) -> PlayerDay:
        """Get or create a PlayerDay for a player on a specific day."""
        key = (player_id, day)
        if key in self._day_index:
            idx = self._day_index[key]
            return self._player_days[player_id][idx]

        # Create player's day list if needed
        if player_id not in self._player_days:
            self._player_days[player_id] = []

        # Find insertion point to keep days sorted
        days_list = self._player_days[player_id]
        insert_idx = 0
        for i, pd in enumerate(days_list):
            if pd.day > day:
                break
            insert_idx = i + 1

        # Create new PlayerDay
        new_pd = PlayerDay(day=day, day_idx=insert_idx, r=0.0)
        days_list.insert(insert_idx, new_pd)

        # Update indices for all days after insertion
        for i in range(insert_idx, len(days_list)):
            days_list[i].day_idx = i
            self._day_index[(player_id, days_list[i].day)] = i

        return new_pd

    def _build_game_graph(self, batches: List[GameBatch]) -> None:
        """Build the game graph from batches."""
        self._player_days.clear()
        self._day_index.clear()

        # First pass: create all player days
        for batch in batches:
            day = batch.day
            for i in range(len(batch)):
                p1 = int(batch.player1[i])
                p2 = int(batch.player2[i])
                self._get_or_create_player_day(p1, day)
                self._get_or_create_player_day(p2, day)

        # Second pass: add games
        for batch in batches:
            day = batch.day
            for i in range(len(batch)):
                p1 = int(batch.player1[i])
                p2 = int(batch.player2[i])
                score = float(batch.scores[i])

                p1_day_idx = self._day_index[(p1, day)]
                p2_day_idx = self._day_index[(p2, day)]

                # Add game from both perspectives
                self._player_days[p1][p1_day_idx].games.append((p2, p2_day_idx, score))
                self._player_days[p2][p2_day_idx].games.append((p1, p1_day_idx, 1.0 - score))

    def _get_opponent_r(self, opp_id: int, opp_day_idx: int) -> float:
        """Get opponent's rating at a specific day."""
        return self._player_days[opp_id][opp_day_idx].r

    def _run_newton_iteration(self) -> float:
        """
        Run one Newton-Raphson iteration for all players.

        Returns maximum rating change.
        """
        max_change = 0.0

        for player_id, days_list in self._player_days.items():
            change = self._update_player(player_id, days_list)
            max_change = max(max_change, change)

        return max_change

    def _update_player(self, player_id: int, days_list: List[PlayerDay]) -> float:
        """
        Update a single player's ratings using Newton-Raphson.

        Uses the tridiagonal structure for efficiency.
        """
        n = len(days_list)
        if n == 0:
            return 0.0

        # Get current ratings
        r = [pd.r for pd in days_list]

        # Compute sigma² between consecutive days
        sigma2 = []
        for i in range(n - 1):
            day_diff = max(1, days_list[i + 1].day - days_list[i].day)
            sigma2.append(self._w2_r * day_diff)

        # Build gradient and Hessian
        gradient = [0.0] * n
        hessian_diag = [0.0] * n
        hessian_off = [0.0] * (n - 1)

        for i, pd in enumerate(days_list):
            # Game contributions
            for opp_id, opp_day_idx, score in pd.games:
                opp_r = self._get_opponent_r(opp_id, opp_day_idx)

                # Compute probability using numerically stable sigmoid
                diff = r[i] - opp_r
                if diff > 20:
                    p_win = 1.0 - 1e-9
                elif diff < -20:
                    p_win = 1e-9
                else:
                    p_win = 1.0 / (1.0 + math.exp(-diff))

                # Gradient: sum of (score - p_win)
                gradient[i] += score - p_win

                # Hessian diagonal: sum of -p_win * (1 - p_win)
                hessian_diag[i] -= p_win * (1.0 - p_win)

            # Prior contributions (Wiener process)
            if i > 0:
                # Connection to previous day
                inv_sigma2 = 1.0 / sigma2[i - 1]
                gradient[i] -= (r[i] - r[i - 1]) * inv_sigma2
                hessian_diag[i] -= inv_sigma2
                hessian_off[i - 1] = inv_sigma2

            if i < n - 1:
                # Connection to next day
                inv_sigma2 = 1.0 / sigma2[i]
                gradient[i] -= (r[i] - r[i + 1]) * inv_sigma2
                hessian_diag[i] -= inv_sigma2

        # Add small regularization for numerical stability
        for i in range(n):
            if hessian_diag[i] > -1e-10:
                hessian_diag[i] = -1e-10

        # Solve tridiagonal system: H * delta = -gradient
        # (H is negative definite, so we solve for the Newton step)
        delta = self._solve_tridiagonal(hessian_diag, hessian_off, gradient, n)

        # Update ratings
        max_change = 0.0
        for i, pd in enumerate(days_list):
            old_r = pd.r
            pd.r = old_r + delta[i]
            max_change = max(max_change, abs(delta[i]))

        return max_change

    def _solve_tridiagonal(
        self,
        diag: List[float],
        off: List[float],
        rhs: List[float],
        n: int,
    ) -> List[float]:
        """
        Solve tridiagonal system Ax = b using Thomas algorithm.

        A has:
        - diag[i] on main diagonal (negative)
        - off[i] on both sub and super diagonals (positive, symmetric)

        We solve: A * x = -rhs (Newton step direction)
        """
        if n == 0:
            return []
        if n == 1:
            return [-rhs[0] / diag[0]] if abs(diag[0]) > 1e-15 else [0.0]

        # Forward elimination (modified for symmetric tridiagonal)
        c = [0.0] * (n - 1)
        d = [0.0] * n

        # First row
        c[0] = off[0] / diag[0]
        d[0] = -rhs[0] / diag[0]

        # Middle rows
        for i in range(1, n - 1):
            denom = diag[i] - off[i - 1] * c[i - 1]
            if abs(denom) < 1e-15:
                denom = -1e-15 if denom <= 0 else 1e-15
            c[i] = off[i] / denom
            d[i] = (-rhs[i] - off[i - 1] * d[i - 1]) / denom

        # Last row
        denom = diag[n - 1] - off[n - 2] * c[n - 2]
        if abs(denom) < 1e-15:
            denom = -1e-15 if denom <= 0 else 1e-15
        d[n - 1] = (-rhs[n - 1] - off[n - 2] * d[n - 2]) / denom

        # Back substitution
        x = [0.0] * n
        x[n - 1] = d[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]

        return x

    def _compute_uncertainties(self) -> None:
        """Compute rating uncertainties from Hessian diagonal."""
        for player_id, days_list in self._player_days.items():
            n = len(days_list)
            if n == 0:
                continue

            r = [pd.r for pd in days_list]

            # Compute sigma² between consecutive days
            sigma2 = []
            for i in range(n - 1):
                day_diff = max(1, days_list[i + 1].day - days_list[i].day)
                sigma2.append(self._w2_r * day_diff)

            for i, pd in enumerate(days_list):
                hess = 0.0

                # Game contributions
                for opp_id, opp_day_idx, score in pd.games:
                    opp_r = self._get_opponent_r(opp_id, opp_day_idx)
                    diff = r[i] - opp_r

                    if diff > 20:
                        p_win = 1.0 - 1e-9
                    elif diff < -20:
                        p_win = 1e-9
                    else:
                        p_win = 1.0 / (1.0 + math.exp(-diff))

                    hess -= p_win * (1.0 - p_win)

                # Prior contributions
                if i > 0:
                    hess -= 1.0 / sigma2[i - 1]
                if i < n - 1:
                    hess -= 1.0 / sigma2[i]

                # Uncertainty from inverse Hessian
                if hess < -1e-10:
                    var_r = -1.0 / hess
                    pd.uncertainty = math.sqrt(var_r) / self.LN10_400
                else:
                    pd.uncertainty = 350.0

    def _run_iterations(self) -> None:
        """Run Newton-Raphson iterations until convergence."""
        for iteration in range(self.config.max_iterations):
            max_change = self._run_newton_iteration()

            if max_change < self.config.convergence_threshold:
                break

        self._compute_uncertainties()

    def _extract_current_ratings(self) -> None:
        """Extract the most recent rating for each player."""
        if self._num_players is None:
            return

        ratings = np.full(self._num_players, self.config.initial_rating, dtype=np.float64)
        uncertainties = np.full(self._num_players, 350.0, dtype=np.float64)

        for player_id, days_list in self._player_days.items():
            if days_list:
                # Get most recent rating (add initial_rating offset)
                last_pd = days_list[-1]
                ratings[player_id] = last_pd.r / self.LN10_400 + self.config.initial_rating
                uncertainties[player_id] = last_pd.uncertainty

        self._ratings = PlayerRatings(
            ratings=ratings,
            rd=uncertainties,
            metadata={"system": "whr", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with a new batch (refits on all data)."""
        self._all_games.append(batch)
        self._refit()

    def _refit(self) -> None:
        """Refit on all stored games."""
        self._build_game_graph(self._all_games)
        self._run_iterations()
        self._extract_current_ratings()

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
    ) -> "WHRPython":
        """Fit WHR on a dataset."""
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

    def update(self, batch: GameBatch) -> "WHRPython":
        """Update with new games by refitting on all data."""
        if not self._fitted:
            raise ValueError("Model must be fitted before updating")

        self._all_games.append(batch)
        self._refit()
        self._current_day = batch.day

        return self

    def predict_proba(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
    ) -> np.ndarray:
        """Predict probability that player1 beats player2."""
        if self._ratings is None:
            raise ValueError("Model not fitted")

        player1 = np.asarray(player1, dtype=np.int64)
        player2 = np.asarray(player2, dtype=np.int64)

        r1 = self._ratings.ratings[player1] * self.LN10_400
        r2 = self._ratings.ratings[player2] * self.LN10_400

        return 1.0 / (1.0 + np.exp(-(r1 - r2)))

    def get_rating_history(self, player_id: int) -> Optional[Dict]:
        """Get the full rating history for a player."""
        if player_id not in self._player_days:
            return None

        days_list = self._player_days[player_id]
        if not days_list:
            return None

        return {
            "days": [pd.day for pd in days_list],
            "ratings": [pd.r / self.LN10_400 + self.config.initial_rating for pd in days_list],
            "uncertainties": [pd.uncertainty for pd in days_list],
        }

    def reset(self) -> "WHRPython":
        """Reset the rating system."""
        self._player_days.clear()
        self._day_index.clear()
        self._all_games.clear()
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"WHRPython(w2={self.config.w2}, "
            f"max_iter={self.config.max_iterations}, "
            f"players={players}, {status})"
        )
