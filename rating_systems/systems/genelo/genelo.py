"""
GenElo (Generalised Elo) rating system.

Implements the Bayesian Elo framework from Ingram (2021), "How to extend
Elo: a Bayesian perspective", Journal of Quantitative Analysis in Sports.

GenElo derives Elo as approximate Bayesian posterior mode estimation
(a steady-state extended Kalman filter). The k-factor becomes a function
of the prior skill difference rather than a constant:

    k = (b/2) / (1/σ²_δ + b² γ(bμ_δ)(1 - γ(bμ_δ)))

Optionally incorporates margin of victory via a joint additive update
(Eq. 24-27): the update decomposes into a win component and a margin
component, each weighted by how well the outcome/margin was predicted.

Prediction uses the approximate marginal likelihood (Appendix E):
    P(win) ≈ γ(b μ_δ / α)  where α = sqrt(1 + π σ²_δ b² / 8)

Reference:
    Ingram, M. (2021). How to extend Elo: a Bayesian perspective.
    Journal of Quantitative Analysis in Sports.
    Code: https://github.com/martiningram/jax_elo
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ._numba_core import (
    fit_all_days_genelo,
    fit_all_days_genelo_margin,
    predict_proba_genelo_batch,
    predict_single_genelo,
)


@dataclass
class GenEloConfig:
    """Configuration for GenElo rating system.

    Attributes:
        initial_rating: Starting rating for all players.
        sigma: Prior standard deviation per player. The prior variance of the
            skill difference is 2σ² (since both players contribute σ²).
            Paper estimate: ~84 (Table in Section 3.1).
        scale: Elo scale factor. b = log(10)/scale.
        use_taylor_k: If True, use the Bayesian k-factor (Eq. 17) that depends
            on the skill difference. If False, use a constant k = bσ²_δ/2,
            which recovers standard Elo.
        c1: Margin slope — scales rating difference to expected margin.
            Set to None to disable margin updates. Paper: 0.000131.
        c2: Margin offset — expected margin advantage for the winner.
            Paper: 0.102 (for serve % won difference in tennis).
        sigma_obs: Observation noise std for the margin. Paper: 0.085.
    """

    initial_rating: float = 1500.0
    sigma: float = 84.0
    scale: float = 400.0
    use_taylor_k: bool = True
    c1: Optional[float] = None
    c2: Optional[float] = None
    sigma_obs: Optional[float] = None


class GenElo(RatingSystem):
    """
    GenElo rating system with Numba acceleration.

    Extends standard Elo with a Bayesian k-factor and optional
    margin-of-victory update. Processes matches sequentially (ONLINE).

    Examples:
        Basic usage (win/loss only):
            >>> model = GenElo(sigma=84.0)
            >>> model.fit(dataset)

        With margin of victory:
            >>> model = GenElo(sigma=84.0, c1=0.000131, c2=0.102, sigma_obs=0.085)
            >>> model.fit(dataset, margins=margin_array)

        Constant-k mode (recovers standard Elo):
            >>> model = GenElo(sigma=84.0, use_taylor_k=False)
            >>> model.fit(dataset)
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        sigma: float = 84.0,
        scale: float = 400.0,
        use_taylor_k: bool = True,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
        sigma_obs: Optional[float] = None,
        num_players: Optional[int] = None,
    ):
        self.config = GenEloConfig(
            initial_rating=initial_rating,
            sigma=sigma,
            scale=scale,
            use_taylor_k=use_taylor_k,
            c1=c1,
            c2=c2,
            sigma_obs=sigma_obs,
        )
        # Precompute constants
        self._b = np.log(10.0) / scale
        self._sigma_delta_sq = 2.0 * sigma * sigma

        self._use_margin = c1 is not None and c2 is not None and sigma_obs is not None
        if self._use_margin:
            self._sigma_obs_sq = sigma_obs * sigma_obs
        else:
            self._sigma_obs_sq = 0.0

        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        self._margins: Optional[np.ndarray] = None

        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial ratings for all players."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "genelo", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings for a single day batch."""
        if len(batch) == 0:
            return

        score_arr = batch.scores
        p1_arr = batch.player1
        p2_arr = batch.player2

        if self._use_margin and self._margins is not None:
            # We need to figure out which slice of margins corresponds to this batch.
            # In the day-by-day iteration, this is called per day.
            # For simplicity, use the fit() fast path instead.
            raise NotImplementedError(
                "Use fit() directly for margin updates; "
                "_update_ratings is for win/loss-only day iteration."
            )

        # Win/loss only: process this batch
        day_offsets = np.array([0, len(p1_arr)], dtype=np.int64)
        fit_all_days_genelo(
            p1_arr, p2_arr, score_arr, day_offsets,
            ratings.ratings,
            self._sigma_delta_sq,
            self._b,
            self.config.use_taylor_k,
        )
        self._num_games_fitted += len(batch)

    def fit(
        self,
        dataset: GameDataset,
        margins: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "GenElo":
        """
        Fit GenElo on a dataset.

        Args:
            dataset: GameDataset to fit on.
            margins: Optional array of continuous margin values per game,
                aligned with the dataset's game order. Required if c1/c2/sigma_obs
                are set. For tennis, typically the difference in serve % won.
            end_day: Optional last day to include.
            player_names: Optional mapping of player IDs to names.

        Returns:
            self (for method chaining).
        """
        self._player_names = player_names

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()
        if player1 is None or len(player1) == 0:
            self._fitted = True
            return self

        if self._use_margin:
            if margins is None:
                raise ValueError(
                    "Margin parameters (c1, c2, sigma_obs) are set but no "
                    "margins array was provided to fit()."
                )
            margins = np.ascontiguousarray(margins, dtype=np.float64)
            if len(margins) != len(player1):
                raise ValueError(
                    f"margins length ({len(margins)}) must match number of "
                    f"games ({len(player1)})"
                )
            self._margins = margins

            fit_all_days_genelo_margin(
                player1, player2, scores, margins, day_offsets,
                self._ratings.ratings,
                self._sigma_delta_sq,
                self._b,
                self.config.c1,
                self.config.c2,
                self._sigma_obs_sq,
            )
        else:
            fit_all_days_genelo(
                player1, player2, scores, day_offsets,
                self._ratings.ratings,
                self._sigma_delta_sq,
                self._b,
                self.config.use_taylor_k,
            )

        self._num_games_fitted = len(player1)
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        self._fitted = True
        return self

    def predict_proba(
        self,
        player1: Union[int, np.integer, np.ndarray],
        player2: Union[int, np.integer, np.ndarray],
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """
        Predict P(player1 wins).

        Uses the approximate marginal likelihood (Eq. 33 / Appendix E):
            P(win) ≈ γ(b μ_δ / α) where α = sqrt(1 + π σ²_δ b² / 8)

        This accounts for uncertainty in the skill difference, giving
        slightly more conservative predictions than plugging in point estimates.

        Args:
            player1: Player 1 ID(s).
            player2: Player 2 ID(s).
            day: Unused (for interface compatibility).

        Returns:
            Win probability (float or array).
        """
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)):
            return predict_single_genelo(
                self._ratings.ratings[player1],
                self._ratings.ratings[player2],
                self._b,
                self._sigma_delta_sq,
            )

        player1 = np.asarray(player1, dtype=np.int64)
        player2 = np.asarray(player2, dtype=np.int64)
        return predict_proba_genelo_batch(
            player1, player2, self._ratings.ratings,
            self._b, self._sigma_delta_sq,
        )

    @property
    def equivalent_k(self) -> float:
        """The equivalent constant k-factor for evenly matched players.

        For an evenly matched contest (γ=0.5), Eq. 17 gives:
            k = bσ²_δ / (2 + b²σ²_δ/2)
        which simplifies to approximately bσ² for typical σ values.
        """
        b = self._b
        s = self._sigma_delta_sq
        eta = 0.25  # γ(1-γ) when γ=0.5
        return (b / 2.0) / (1.0 / s + b * b * eta)
