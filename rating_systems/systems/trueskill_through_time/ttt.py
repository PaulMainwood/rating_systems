"""
TrueSkill Through Time (TTT) - High-performance Numba implementation.

Based on:
- Dangauthier et al., "TrueSkill Through Time: Revisiting the History of Chess" (2007)
- Glandfried's Python implementation: https://github.com/glandfried/TrueSkillThroughTime.py

TTT extends TrueSkill by modeling player skill as evolving over time using
Gaussian belief propagation with forward-backward message passing.

This implementation prioritizes efficiency through:
1. Numba JIT compilation of all hot paths
2. CSR-like data structures for player timelines
3. Custom norm_cdf/norm_pdf (no scipy dependency)
4. Parallel prediction via prange
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ...results.fitted_ratings import FittedTTTRatings
from ._numba_core import (
    run_all_iterations,
    extract_ratings,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
)


@dataclass
class TTTConfig:
    """Configuration for TrueSkill Through Time."""

    mu: float = 0.0  # Prior mean skill (internal scale, 0 = average)
    sigma: float = 1.5  # Prior skill std dev (internal scale)
    beta: float = 0.5  # Performance std dev (within-game noise)
    gamma: float = 0.01  # Skill dynamics per time unit
    max_iterations: int = 30  # Max forward-backward iterations for initial fit
    update_max_iterations: int = None  # Max iterations for incremental updates (default: same as max_iterations)
    convergence_threshold: float = 1e-4  # Convergence threshold

    def __post_init__(self):
        if self.update_max_iterations is None:
            self.update_max_iterations = self.max_iterations


class TrueSkillThroughTime(RatingSystem):
    """
    TrueSkill Through Time rating system with Numba acceleration.

    TTT models player skill as a Gaussian that evolves over time:
    - Skill at time t: N(μ_t, σ_t²)
    - Skill drift: σ increases by γ√t per time unit
    - Game outcomes update beliefs via Gaussian message passing

    Uses forward-backward belief propagation for globally consistent estimates.

    Performance characteristics:
    - Fit: O(iterations * games) with Numba acceleration
    - Predict: O(n) parallel across matchups
    - Memory: O(total_player_days + total_games)

    Parameters:
        mu: Prior mean skill (default: 0.0, displayed as 1500)
        sigma: Prior skill std dev (default: 1.5)
        beta: Performance variability within games (default: 0.5)
        gamma: Skill drift rate per time unit (default: 0.01)
        max_iterations: Max belief propagation iterations (default: 30)
        convergence_threshold: Stop when max change < threshold

    Example:
        >>> ttt = TrueSkillThroughTime(sigma=1.5, beta=0.5)
        >>> ttt.fit(dataset)
        >>> fitted = ttt.get_fitted_ratings()
        >>> print(fitted.top(10))
        >>> print(fitted.predict(0, 1))
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
        max_iterations: int = 30,
        update_max_iterations: Optional[int] = None,
        convergence_threshold: float = 1e-4,
        num_players: Optional[int] = None,
    ):
        self.config = TTTConfig(
            mu=mu,
            sigma=sigma,
            beta=beta,
            gamma=gamma,
            max_iterations=max_iterations,
            update_max_iterations=update_max_iterations,
            convergence_threshold=convergence_threshold,
        )

        # Data structures (CSR-like format for Numba)
        self._player_offsets: Optional[np.ndarray] = None
        self._pd_days: Optional[np.ndarray] = None
        self._pd_forward_mu: Optional[np.ndarray] = None
        self._pd_forward_sigma: Optional[np.ndarray] = None
        self._pd_backward_mu: Optional[np.ndarray] = None
        self._pd_backward_sigma: Optional[np.ndarray] = None
        self._pd_likelihood_pi: Optional[np.ndarray] = None
        self._pd_likelihood_tau: Optional[np.ndarray] = None
        self._pd_game_offsets: Optional[np.ndarray] = None
        self._pd_game_opp_pd: Optional[np.ndarray] = None
        self._pd_game_score: Optional[np.ndarray] = None

        # Metadata
        self._num_games_fitted = 0
        self._num_iterations = 0
        self._player_names: Optional[Dict[int, str]] = None

        # Store raw data for refitting
        self._stored_player1: Optional[np.ndarray] = None
        self._stored_player2: Optional[np.ndarray] = None
        self._stored_scores: Optional[np.ndarray] = None
        self._stored_days: Optional[np.ndarray] = None

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial TTT ratings."""
        return PlayerRatings(
            ratings=np.full(num_players, self.DISPLAY_OFFSET, dtype=np.float64),
            rd=np.full(num_players, self.config.sigma * self.DISPLAY_SCALE, dtype=np.float64),
            metadata={"system": "ttt", "config": self.config},
        )

    def _build_data_structures(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        days: np.ndarray,
        num_players: int,
    ) -> None:
        """
        Build CSR-like data structures from game arrays.

        Converts raw game data into the format needed by Numba:
        - Player timelines with their active days
        - Games per player-day with opponent references
        """
        n_games = len(player1)

        # Step 1: Identify unique (player, day) pairs
        player_day_map: Dict[Tuple[int, int], int] = {}
        player_day_list: List[Tuple[int, int]] = []

        for i in range(n_games):
            p1, p2, day = int(player1[i]), int(player2[i]), int(days[i])

            key1 = (p1, day)
            if key1 not in player_day_map:
                player_day_map[key1] = len(player_day_list)
                player_day_list.append(key1)

            key2 = (p2, day)
            if key2 not in player_day_map:
                player_day_map[key2] = len(player_day_list)
                player_day_list.append(key2)

        total_pd = len(player_day_list)

        # Step 2: Sort player-days by (player_id, day) and build offsets
        player_day_list.sort(key=lambda x: (x[0], x[1]))

        # Rebuild mapping with sorted indices
        sorted_map: Dict[Tuple[int, int], int] = {}
        for new_idx, (pid, day) in enumerate(player_day_list):
            sorted_map[(pid, day)] = new_idx

        # Build player_offsets
        self._player_offsets = np.zeros(num_players + 1, dtype=np.int64)
        for pid, _ in player_day_list:
            self._player_offsets[pid + 1] += 1
        np.cumsum(self._player_offsets, out=self._player_offsets)

        # Build pd_days array
        self._pd_days = np.empty(total_pd, dtype=np.int32)
        for idx, (_, day) in enumerate(player_day_list):
            self._pd_days[idx] = day

        # Initialize message arrays
        self._pd_forward_mu = np.zeros(total_pd, dtype=np.float64)
        self._pd_forward_sigma = np.full(total_pd, 1e6, dtype=np.float64)
        self._pd_backward_mu = np.zeros(total_pd, dtype=np.float64)
        self._pd_backward_sigma = np.full(total_pd, 1e6, dtype=np.float64)
        self._pd_likelihood_pi = np.zeros(total_pd, dtype=np.float64)
        self._pd_likelihood_tau = np.zeros(total_pd, dtype=np.float64)

        # Step 3: Count games per player-day
        pd_game_counts = np.zeros(total_pd, dtype=np.int64)
        for i in range(n_games):
            p1, p2, day = int(player1[i]), int(player2[i]), int(days[i])
            pd1 = sorted_map[(p1, day)]
            pd2 = sorted_map[(p2, day)]
            pd_game_counts[pd1] += 1
            pd_game_counts[pd2] += 1

        # Build game offsets
        self._pd_game_offsets = np.zeros(total_pd + 1, dtype=np.int64)
        np.cumsum(pd_game_counts, out=self._pd_game_offsets[1:])

        total_game_refs = self._pd_game_offsets[-1]

        # Step 4: Fill game arrays
        self._pd_game_opp_pd = np.empty(total_game_refs, dtype=np.int64)
        self._pd_game_score = np.empty(total_game_refs, dtype=np.float64)

        pd_game_pos = self._pd_game_offsets[:-1].copy()

        for i in range(n_games):
            p1, p2, day = int(player1[i]), int(player2[i]), int(days[i])
            score = float(scores[i])

            pd1 = sorted_map[(p1, day)]
            pd2 = sorted_map[(p2, day)]

            # Add game from player1's perspective
            pos1 = pd_game_pos[pd1]
            self._pd_game_opp_pd[pos1] = pd2
            self._pd_game_score[pos1] = score
            pd_game_pos[pd1] += 1

            # Add game from player2's perspective
            pos2 = pd_game_pos[pd2]
            self._pd_game_opp_pd[pos2] = pd1
            self._pd_game_score[pos2] = 1.0 - score
            pd_game_pos[pd2] += 1

    def _run_optimization(self, max_iterations: Optional[int] = None) -> None:
        """Run forward-backward belief propagation.

        Args:
            max_iterations: Override max iterations (uses config value if None)
        """
        if self._player_offsets is None or self._num_players is None:
            return

        if max_iterations is None:
            max_iterations = self.config.max_iterations

        self._num_iterations = run_all_iterations(
            self._num_players,
            self._player_offsets,
            self._pd_days,
            self._pd_forward_mu,
            self._pd_forward_sigma,
            self._pd_backward_mu,
            self._pd_backward_sigma,
            self._pd_likelihood_pi,
            self._pd_likelihood_tau,
            self._pd_game_offsets,
            self._pd_game_opp_pd,
            self._pd_game_score,
            self.config.mu,
            self.config.sigma,
            self.config.beta,
            self.config.gamma,
            max_iterations,
            self.config.convergence_threshold,
        )

    def _extract_current_ratings(self) -> None:
        """Extract most recent ratings for each player."""
        if self._num_players is None or self._player_offsets is None:
            return

        ratings = np.empty(self._num_players, dtype=np.float64)
        rd = np.empty(self._num_players, dtype=np.float64)

        extract_ratings(
            self._num_players,
            self._player_offsets,
            self._pd_forward_mu,
            self._pd_forward_sigma,
            self._pd_backward_mu,
            self._pd_backward_sigma,
            self._pd_likelihood_pi,
            self._pd_likelihood_tau,
            ratings,
            rd,
            self.config.mu,
            self.config.sigma,
            self.DISPLAY_SCALE,
            self.DISPLAY_OFFSET,
        )

        self._ratings = PlayerRatings(
            ratings=ratings,
            rd=rd,
            metadata={"system": "ttt", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with a new batch (refits on all data)."""
        if self._stored_player1 is None:
            self._stored_player1 = batch.player1.copy()
            self._stored_player2 = batch.player2.copy()
            self._stored_scores = batch.scores.copy()
            self._stored_days = np.full(len(batch), batch.day, dtype=np.int32)
        else:
            self._stored_player1 = np.concatenate([self._stored_player1, batch.player1])
            self._stored_player2 = np.concatenate([self._stored_player2, batch.player2])
            self._stored_scores = np.concatenate([self._stored_scores, batch.scores])
            self._stored_days = np.concatenate([
                self._stored_days,
                np.full(len(batch), batch.day, dtype=np.int32)
            ])

        self._refit()

    def _refit(self, max_iterations: Optional[int] = None) -> None:
        """Refit on all stored data.

        Args:
            max_iterations: Override max iterations (uses update_max_iterations if None)
        """
        if self._stored_player1 is None:
            return

        if max_iterations is None:
            max_iterations = self.config.update_max_iterations

        self._build_data_structures(
            self._stored_player1,
            self._stored_player2,
            self._stored_scores,
            self._stored_days,
            self._num_players,
        )
        self._run_optimization(max_iterations)
        self._extract_current_ratings()
        self._num_games_fitted = len(self._stored_player1)

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "TrueSkillThroughTime":
        """
        Fit TTT on a dataset.

        Args:
            dataset: Game dataset to fit on
            end_day: Last day to include (inclusive). Useful for backtesting.
            player_names: Optional mapping of player_id -> name

        Returns:
            self (for method chaining)
        """
        self._player_names = player_names

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        # Get batched arrays
        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is None or len(player1) == 0:
            self._fitted = True
            return self

        # Create per-game day array
        n_games = len(player1)
        days = np.empty(n_games, dtype=np.int32)
        n_days = len(day_indices)
        for d in range(n_days):
            start = day_offsets[d]
            end = day_offsets[d + 1]
            days[start:end] = day_indices[d]

        # Store for potential refitting
        self._stored_player1 = player1.copy()
        self._stored_player2 = player2.copy()
        self._stored_scores = scores.copy()
        self._stored_days = days

        # Build data structures and optimize
        self._build_data_structures(
            player1, player2, scores, days, self._num_players
        )
        self._run_optimization()
        self._extract_current_ratings()

        self._num_games_fitted = n_games
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None

        return self

    def update(self, batch: GameBatch) -> "TrueSkillThroughTime":
        """Update with new games by refitting on all data."""
        if not self._fitted:
            raise ValueError("Model must be fitted before updating")

        self._update_ratings(batch, self._ratings)
        self._current_day = batch.day

        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
    ) -> Union[float, np.ndarray]:
        """
        Predict probability that player1 beats player2.

        Args:
            player1: Single player ID or array of player IDs
            player2: Single player ID or array of player IDs

        Returns:
            Single probability or array of probabilities
        """
        if self._ratings is None:
            raise ValueError("Model not fitted")

        # Handle single prediction
        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
                self._ratings.rd[int(player1)],
                self._ratings.rd[int(player2)],
                self.config.beta,
                self.DISPLAY_SCALE,
                self.DISPLAY_OFFSET,
            )

        # Batch prediction
        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(
            p1, p2,
            self._ratings.ratings,
            self._ratings.rd,
            self.config.beta,
            self.DISPLAY_SCALE,
            self.DISPLAY_OFFSET,
        )

    def get_fitted_ratings(self) -> FittedTTTRatings:
        """
        Get a queryable fitted ratings object.

        Returns:
            FittedTTTRatings with methods for querying results
        """
        if self._ratings is None:
            raise ValueError("Model not fitted")

        # Build rating history per player
        rating_history = {}
        if self._player_offsets is not None:
            for pid in range(self._num_players):
                pd_start = self._player_offsets[pid]
                pd_end = self._player_offsets[pid + 1]
                if pd_end > pd_start:
                    # Compute posteriors for each player-day
                    n = pd_end - pd_start
                    ratings = np.empty(n, dtype=np.float64)
                    uncertainties = np.empty(n, dtype=np.float64)

                    for i in range(n):
                        pd_idx = pd_start + i
                        # Get posterior
                        fwd_prec = 1.0 / (self._pd_forward_sigma[pd_idx] ** 2) if self._pd_forward_sigma[pd_idx] < 1e5 else 0.0
                        bwd_prec = 1.0 / (self._pd_backward_sigma[pd_idx] ** 2) if self._pd_backward_sigma[pd_idx] < 1e5 else 0.0
                        total_prec = fwd_prec + bwd_prec + self._pd_likelihood_pi[pd_idx]

                        if total_prec > 1e-10:
                            total_tau = (fwd_prec * self._pd_forward_mu[pd_idx] +
                                        bwd_prec * self._pd_backward_mu[pd_idx] +
                                        self._pd_likelihood_tau[pd_idx])
                            mu = total_tau / total_prec
                            sigma = 1.0 / np.sqrt(total_prec)
                        else:
                            mu = self.config.mu
                            sigma = self.config.sigma

                        ratings[i] = mu * self.DISPLAY_SCALE + self.DISPLAY_OFFSET
                        uncertainties[i] = sigma * self.DISPLAY_SCALE

                    rating_history[pid] = {
                        "days": self._pd_days[pd_start:pd_end].copy(),
                        "ratings": ratings,
                        "uncertainties": uncertainties,
                    }

        return FittedTTTRatings(
            ratings=self._ratings.ratings.copy(),
            rd=self._ratings.rd.copy(),
            sigma=self.config.sigma,
            beta=self.config.beta,
            gamma=self.config.gamma,
            display_scale=self.DISPLAY_SCALE,
            display_offset=self.DISPLAY_OFFSET,
            num_games_fitted=self._num_games_fitted,
            num_iterations=self._num_iterations,
            last_day=self._current_day,
            player_names=self._player_names,
            rating_history=rating_history,
        )

    def get_rating_history(self, player_id: int) -> Optional[Dict]:
        """Get the full rating history for a player."""
        if self._player_offsets is None:
            return None

        pd_start = self._player_offsets[player_id]
        pd_end = self._player_offsets[player_id + 1]

        if pd_end <= pd_start:
            return None

        n = pd_end - pd_start
        ratings = []
        uncertainties = []

        for i in range(n):
            pd_idx = pd_start + i
            fwd_prec = 1.0 / (self._pd_forward_sigma[pd_idx] ** 2) if self._pd_forward_sigma[pd_idx] < 1e5 else 0.0
            bwd_prec = 1.0 / (self._pd_backward_sigma[pd_idx] ** 2) if self._pd_backward_sigma[pd_idx] < 1e5 else 0.0
            total_prec = fwd_prec + bwd_prec + self._pd_likelihood_pi[pd_idx]

            if total_prec > 1e-10:
                total_tau = (fwd_prec * self._pd_forward_mu[pd_idx] +
                            bwd_prec * self._pd_backward_mu[pd_idx] +
                            self._pd_likelihood_tau[pd_idx])
                mu = total_tau / total_prec
                sigma = 1.0 / np.sqrt(total_prec)
            else:
                mu = self.config.mu
                sigma = self.config.sigma

            ratings.append(mu * self.DISPLAY_SCALE + self.DISPLAY_OFFSET)
            uncertainties.append(sigma * self.DISPLAY_SCALE)

        return {
            "days": self._pd_days[pd_start:pd_end].tolist(),
            "ratings": ratings,
            "uncertainties": uncertainties,
        }

    def top(self, n: int = 10) -> np.ndarray:
        """Get indices of top N rated players (convenience method)."""
        if self._ratings is None:
            raise ValueError("Model not fitted")
        return get_top_n_indices(self._ratings.ratings, n)

    def reset(self) -> "TrueSkillThroughTime":
        """Reset the rating system."""
        self._player_offsets = None
        self._pd_days = None
        self._pd_forward_mu = None
        self._pd_forward_sigma = None
        self._pd_backward_mu = None
        self._pd_backward_sigma = None
        self._pd_likelihood_pi = None
        self._pd_likelihood_tau = None
        self._pd_game_offsets = None
        self._pd_game_opp_pd = None
        self._pd_game_score = None
        self._stored_player1 = None
        self._stored_player2 = None
        self._stored_scores = None
        self._stored_days = None
        self._num_games_fitted = 0
        self._num_iterations = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"TrueSkillThroughTime(sigma={self.config.sigma:.2f}, "
            f"beta={self.config.beta:.2f}, gamma={self.config.gamma:.3f}, "
            f"players={players}, {status})"
        )
