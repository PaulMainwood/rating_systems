"""Serve/Return Elo rating system.

Same architecture as ServeReturnGlicko but using Elo sub-systems:
  - ServeElo: tracks each player's serve strength
  - ReturnElo: tracks each player's return strength

Each match with serve stats produces two Elo observations:
  1. P1_serve vs P2_return: score = P1 serve point win %
  2. P2_serve vs P1_return: score = P2 serve point win %

Predictions use a probit link:
  serve_point_prob = Phi(mu + (serve_rating - return_rating) / scale)

where Phi is the standard normal CDF. This has lighter tails than the
logistic (sigmoid) used in ServeReturnGlicko, naturally compressing
extreme predictions without requiring hard clamps.

The point probabilities are then propagated through the exact tennis
scoring model (point -> game -> set -> match).

Compared to ServeReturnGlicko:
  - Probit link (normal CDF) instead of logit (sigmoid)
  - No RD (rating deviation) — simpler, faster
  - k_factor controls update speed (analogous to Glicko's update magnitude)
  - No uncertainty-based weighting; all observations weighted equally
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from numba import njit

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameDataset, GameBatch
from ..elo.elo import Elo
from ._scoring import prob_win_match, prob_win_match_batch


@dataclass
class ServeReturnEloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    elo_scale: float = 400.0      # Elo sub-system scale
    # Probit-with-intercept:
    #   serve_pct = Phi(mu + (r_serve - r_return) / scale)
    # where Phi is the standard normal CDF. Lighter tails than logistic,
    # naturally compresses extreme predictions without hard clamps.
    mu: float = 0.49              # server advantage in probit space
    scale: float = 400.0          # rating units per probit unit


@njit(cache=True)
def _ndtr(x):
    """Standard normal CDF (Phi), Numba-compatible.

    Uses the Abramowitz & Stegun rational approximation (eq. 26.2.17),
    accurate to ~1.5e-7 across the entire real line.
    """
    # Constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1.0
    if x < 0:
        sign = -1.0
        x = -x

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2.0)

    return 0.5 * (1.0 + sign * y)


@njit(cache=True)
def _ratings_to_serve_pct_elo(
    serve_ratings, return_ratings,
    p1_ids, p2_ids, mu, scale,
):
    """Convert serve/return Elo ratings to serve point win probabilities.

    Uses probit link: serve_pct = Phi(mu + (r_serve - r_return) / scale).
    """
    n = len(p1_ids)
    p_serve_1 = np.empty(n, dtype=np.float64)
    p_serve_2 = np.empty(n, dtype=np.float64)

    for i in range(n):
        p1 = p1_ids[i]
        p2 = p2_ids[i]

        z1 = mu + (serve_ratings[p1] - return_ratings[p2]) / scale
        z2 = mu + (serve_ratings[p2] - return_ratings[p1]) / scale

        p_serve_1[i] = _ndtr(z1)
        p_serve_2[i] = _ndtr(z2)

    return p_serve_1, p_serve_2


class ServeReturnElo(RatingSystem):
    """Serve/Return Elo: two Elo instances + scoring model.

    Each match with serve stats produces two observations:
      1. P1_serve vs P2_return: score = p1_serve_won / p1_serve_total
      2. P2_serve vs P1_return: score = p2_serve_won / p2_serve_total

    Prediction uses probit-with-intercept:
      serve_pct = Phi(mu + (serve_rating - return_rating) / scale)
    then exact scoring model for match probability.
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        elo_scale: float = 400.0,
        mu: float = 0.49,
        scale: float = 400.0,
        num_players: Optional[int] = None,
    ):
        self.config = ServeReturnEloConfig(
            initial_rating=initial_rating,
            k_factor=k_factor,
            elo_scale=elo_scale,
            mu=mu,
            scale=scale,
        )

        self._serve_elo = Elo(
            initial_rating=initial_rating,
            k_factor=k_factor,
            scale=elo_scale,
            num_players=num_players,
        )
        self._return_elo = Elo(
            initial_rating=initial_rating,
            k_factor=k_factor,
            scale=elo_scale,
            num_players=num_players,
        )

        self._num_games_fitted = 0
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        self._serve_elo._num_players = num_players
        self._serve_elo._ratings = self._serve_elo._initialize_ratings(num_players)
        self._serve_elo._fitted = True
        self._return_elo._num_players = num_players
        self._return_elo._ratings = self._return_elo._initialize_ratings(num_players)
        self._return_elo._fitted = True
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "serve_return_elo"},
        )

    def fit(
        self,
        dataset: GameDataset,
        serve_stats: Optional[dict] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "ServeReturnElo":
        """Fit on a dataset with serve statistics."""
        if serve_stats is None:
            raise ValueError("serve_stats dict is required")

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()
        if player1 is None or len(player1) == 0:
            self._fitted = True
            return self

        n_games = len(player1)

        required = max(self._num_players or 0, dataset.num_players)
        if self._num_players is None or required > self._num_players:
            self._num_players = required
            self._ratings = self._initialize_ratings(required)

        p1_sw = serve_stats["p1_serve_won"][:n_games]
        p1_st = serve_stats["p1_serve_total"][:n_games]
        p2_sw = serve_stats["p2_serve_won"][:n_games]
        p2_st = serve_stats["p2_serve_total"][:n_games]

        valid = (p1_st > 0) & (p2_st > 0) & np.isfinite(p1_sw) & np.isfinite(p2_sw)

        serve_p1 = player1[valid]
        serve_p2 = player2[valid]
        serve_scores = (p1_sw[valid] / p1_st[valid]).astype(np.float64)

        return_p1 = player2[valid]
        return_p2 = player1[valid]
        return_scores = (p2_sw[valid] / p2_st[valid]).astype(np.float64)

        game_days = np.empty(n_games, dtype=np.int32)
        for d in range(len(day_indices)):
            game_days[day_offsets[d]:day_offsets[d + 1]] = day_indices[d]
        valid_days = game_days[valid]

        import polars as pl
        if len(serve_p1) > 0:
            serve_df = pl.DataFrame({
                "Player1": serve_p1, "Player2": serve_p2,
                "Score": serve_scores, "Day": valid_days,
            })
            self._serve_elo.fit(GameDataset.from_dataframe(serve_df))

            return_df = pl.DataFrame({
                "Player1": return_p1, "Player2": return_p2,
                "Score": return_scores, "Day": valid_days,
            })
            self._return_elo.fit(GameDataset.from_dataframe(return_df))

        self._num_games_fitted = int(valid.sum())
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        return self

    def update(self, batch: GameBatch,
               serve_stats: Optional[dict] = None) -> "ServeReturnElo":
        """Incrementally update with a new batch."""
        if serve_stats is None:
            return self

        n = len(batch)
        p1_sw = serve_stats["p1_serve_won"][:n]
        p1_st = serve_stats["p1_serve_total"][:n]
        p2_sw = serve_stats["p2_serve_won"][:n]
        p2_st = serve_stats["p2_serve_total"][:n]

        valid = (p1_st > 0) & (p2_st > 0) & np.isfinite(p1_sw) & np.isfinite(p2_sw)
        if not np.any(valid):
            return self

        serve_batch = GameBatch(
            player1=batch.player1[valid],
            player2=batch.player2[valid],
            scores=(p1_sw[valid] / p1_st[valid]).astype(np.float64),
            day=batch.day,
        )
        self._serve_elo.update(serve_batch)

        return_batch = GameBatch(
            player1=batch.player2[valid],
            player2=batch.player1[valid],
            scores=(p2_sw[valid] / p2_st[valid]).astype(np.float64),
            day=batch.day,
        )
        self._return_elo.update(return_batch)

        self._current_day = batch.day
        self._num_games_fitted += int(valid.sum())
        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        best_of: Union[int, np.ndarray] = 3,
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict match win probability."""
        if not self._fitted:
            raise ValueError("Model not fitted.")

        p1 = np.atleast_1d(np.asarray(player1, dtype=np.int64))
        p2 = np.atleast_1d(np.asarray(player2, dtype=np.int64))

        p_serve_1, p_serve_2 = _ratings_to_serve_pct_elo(
            self._serve_elo._ratings.ratings,
            self._return_elo._ratings.ratings,
            p1, p2,
            self.config.mu, self.config.scale,
        )

        if isinstance(best_of, (int, np.integer)):
            bo_arr = np.full(len(p1), int(best_of), dtype=np.int32)
        else:
            bo_arr = np.asarray(best_of, dtype=np.int32)

        result = prob_win_match_batch(p_serve_1, p_serve_2, bo_arr)

        if isinstance(player1, (int, np.integer)):
            return float(result[0])
        return result

    def save_state(self, path: str) -> None:
        """Save serve, return, and base ratings to a single .npz."""
        arrays = {
            "ratings": self._ratings.ratings if self._ratings else np.array([]),
            "serve_ratings": self._serve_elo._ratings.ratings,
            "return_ratings": self._return_elo._ratings.ratings,
        }
        meta = json.dumps({
            "num_players": self._num_players,
            "num_games_fitted": self._num_games_fitted,
            "current_day": self._current_day,
        }).encode()
        arrays["_metadata"] = np.frombuffer(meta, dtype=np.uint8)
        np.savez_compressed(path, **arrays)

    def load_state(self, path: str) -> None:
        """Restore serve, return, and base ratings from .npz."""
        data = np.load(path, allow_pickle=True)

        meta = json.loads(bytes(data["_metadata"]))
        self._num_players = meta["num_players"]
        self._num_games_fitted = meta.get("num_games_fitted", 0)
        self._current_day = meta.get("current_day")

        self._ratings = PlayerRatings(
            ratings=data["ratings"],
            metadata={"system": "serve_return_elo"},
        )

        self._serve_elo._num_players = self._num_players
        self._serve_elo._ratings = PlayerRatings(
            ratings=data["serve_ratings"],
        )
        self._serve_elo._fitted = True

        self._return_elo._num_players = self._num_players
        self._return_elo._ratings = PlayerRatings(
            ratings=data["return_ratings"],
        )
        self._return_elo._fitted = True
        self._fitted = True

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        pass

    def reset(self) -> "ServeReturnElo":
        self._serve_elo.reset()
        self._return_elo.reset()
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (f"ServeReturnElo(k={self.config.k_factor}, mu={self.config.mu:.2f}, "
                f"scale={self.config.scale:.0f}, {self._num_games_fitted:,} games, {status})")
