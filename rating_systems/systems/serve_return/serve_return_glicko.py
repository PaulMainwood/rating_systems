"""Serve/Return Glicko rating system.

Maintains two parallel Glicko systems:
  - ServeGlicko: tracks each player's serve strength
  - ReturnGlicko: tracks each player's return strength

Each match with serve stats produces two Glicko observations:
  1. P1_serve vs P2_return: score = P1 serve point win %
  2. P2_serve vs P1_return: score = P2 serve point win %

Predictions use the Ingram/Sackmann model:
  serve_point_prob = sigmoid(mu + (serve_rating - return_rating) / scale)

where mu is the global server advantage (~0.49 in logit space → ~0.62 in
probability space) and scale converts rating units to logit units.

The point probabilities are then propagated through the exact tennis
scoring model (point → game → set → match) to get match win probability.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from numba import njit

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameDataset, GameBatch
from ..glicko.glicko import Glicko
from ._scoring import prob_win_match, prob_win_match_batch


@dataclass
class ServeReturnConfig:
    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    min_rd: float = 30.0
    max_rd: float = 350.0
    c: float = 10.0
    # Logistic-with-intercept parameters (Ingram 2019):
    #   serve_pct = sigmoid(mu + g(rd) * (r_serve - r_return) / scale)
    mu: float = 0.49          # server advantage in logit space; sigmoid(0.49) ≈ 0.62
    scale: float = 400.0      # rating units per logit unit (like Elo's 400/ln(10))
    # Surface-conditional mu offsets (added to mu)
    mu_clay: float = 0.0       # clay offset (negative = less serve advantage)
    mu_grass: float = 0.0      # grass offset (positive = more serve advantage)
    mu_indoor: float = 0.0     # indoor hard offset


@njit(cache=True)
def _sigmoid(x):
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


@njit(cache=True)
def _grow_rd(rd, last_played, day, c, max_rd):
    """Grow RD forward from last_played to day."""
    dt = max(0, day - last_played)
    if dt == 0:
        return rd
    return min(math.sqrt(rd * rd + c * c * dt), max_rd)


@njit(cache=True)
def _g_serve(rd_serve, rd_return, scale):
    """Glicko-style g() adapted for serve/return scale.

    Shrinks the rating difference when uncertainty is high.
    g = 1 / sqrt(1 + 3 * (rd_s² + rd_r²) / (π² * scale²))

    When both RDs are small relative to scale, g → 1 (full diff counts).
    When RDs are large, g → 0 (prediction shrinks toward mu).
    """
    combined_var = rd_serve * rd_serve + rd_return * rd_return
    return 1.0 / math.sqrt(1.0 + 3.0 * combined_var / (math.pi * math.pi * scale * scale))


@njit(cache=True)
def _ratings_to_serve_pct(
    serve_ratings, serve_rd, return_ratings, return_rd,
    serve_last_played, return_last_played,
    p1_ids, p2_ids, surface_ids, day, c, max_rd,
    mu, scale, mu_clay, mu_grass, mu_indoor,
):
    """Convert serve/return ratings to serve point win probabilities.

    Uses: serve_pct = sigmoid(mu_eff + g(rd_s, rd_r) * (r_serve - r_return) / scale)

    where mu_eff = mu + surface offset.

    The g() factor shrinks the rating difference when either player's
    serve or return rating is uncertain (high RD). This pulls predictions
    toward the server-advantage baseline for unknown players.

    Surface IDs: 0=hard, 1=clay, 2=grass, 3=indoor hard, -1=unknown.
    """
    n = len(p1_ids)
    p_serve_1 = np.empty(n, dtype=np.float64)
    p_serve_2 = np.empty(n, dtype=np.float64)

    for i in range(n):
        p1 = p1_ids[i]
        p2 = p2_ids[i]

        # Surface-conditional mu
        surf = surface_ids[i]
        mu_eff = mu
        if surf == 1:
            mu_eff += mu_clay
        elif surf == 2:
            mu_eff += mu_grass
        elif surf == 3:
            mu_eff += mu_indoor

        # P1 serving: P1's serve rating vs P2's return rating
        r_s1 = serve_ratings[p1]
        r_r2 = return_ratings[p2]

        # P2 serving: P2's serve rating vs P1's return rating
        r_s2 = serve_ratings[p2]
        r_r1 = return_ratings[p1]

        # Grow RD forward to prediction day
        rd_s1 = _grow_rd(serve_rd[p1], serve_last_played[p1], day, c, max_rd)
        rd_r2 = _grow_rd(return_rd[p2], return_last_played[p2], day, c, max_rd)
        rd_s2 = _grow_rd(serve_rd[p2], serve_last_played[p2], day, c, max_rd)
        rd_r1 = _grow_rd(return_rd[p1], return_last_played[p1], day, c, max_rd)

        # g() shrinks rating difference when uncertainty is high
        g1 = _g_serve(rd_s1, rd_r2, scale)
        g2 = _g_serve(rd_s2, rd_r1, scale)

        logit_1 = mu_eff + g1 * (r_s1 - r_r2) / scale
        logit_2 = mu_eff + g2 * (r_s2 - r_r1) / scale

        # Wide clamps: allow the scoring model to use the full signal
        p_serve_1[i] = max(0.35, min(0.90, _sigmoid(logit_1)))
        p_serve_2[i] = max(0.35, min(0.90, _sigmoid(logit_2)))

    return p_serve_1, p_serve_2


# Surface string -> integer mapping for Numba kernel
_SURFACE_TO_ID = {
    "hard": 0, "clay": 1, "grass": 2,
    "i.hard": 3, "carpet": 0,  # carpet → hard (rare, similar speed)
}


def _encode_surfaces(surfaces) -> np.ndarray:
    """Convert surface strings to int32 IDs for the Numba kernel."""
    n = len(surfaces)
    out = np.zeros(n, dtype=np.int32)  # default = hard
    for i in range(n):
        s = surfaces[i]
        if s:
            out[i] = _SURFACE_TO_ID.get(s.lower().strip(), 0)
    return out


class ServeReturnGlicko(RatingSystem):
    """Serve/Return Glicko: two Glicko instances + scoring model.

    Each match with serve stats produces two observations:
      1. P1_serve vs P2_return: score = p1_serve_won / p1_serve_total
      2. P2_serve vs P1_return: score = p2_serve_won / p2_serve_total

    Prediction uses logistic-with-intercept (Ingram 2019):
      serve_pct = sigmoid(mu + (serve_rating - return_rating) / scale)
    then exact scoring model for match probability.
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        min_rd: float = 30.0,
        max_rd: float = 350.0,
        c: float = 10.0,
        mu: float = 0.49,
        scale: float = 400.0,
        mu_clay: float = 0.0,
        mu_grass: float = 0.0,
        mu_indoor: float = 0.0,
        num_players: Optional[int] = None,
    ):
        self.config = ServeReturnConfig(
            initial_rating=initial_rating,
            initial_rd=initial_rd,
            min_rd=min_rd,
            max_rd=max_rd,
            c=c,
            mu=mu,
            scale=scale,
            mu_clay=mu_clay,
            mu_grass=mu_grass,
            mu_indoor=mu_indoor,
        )

        self._serve_glicko = Glicko(
            initial_rating=initial_rating, initial_rd=initial_rd,
            min_rd=min_rd, max_rd=max_rd, c=c,
            num_players=num_players,
        )
        self._return_glicko = Glicko(
            initial_rating=initial_rating, initial_rd=initial_rd,
            min_rd=min_rd, max_rd=max_rd, c=c,
            num_players=num_players,
        )

        self._num_games_fitted = 0
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        self._serve_glicko._num_players = num_players
        self._serve_glicko._ratings = self._serve_glicko._initialize_ratings(num_players)
        self._serve_glicko._fitted = True
        self._return_glicko._num_players = num_players
        self._return_glicko._ratings = self._return_glicko._initialize_ratings(num_players)
        self._return_glicko._fitted = True
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            rd=np.full(num_players, self.config.initial_rd, dtype=np.float64),
            metadata={"system": "serve_return_glicko"},
        )

    def fit(
        self,
        dataset: GameDataset,
        serve_stats: Optional[dict] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "ServeReturnGlicko":
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
            self._serve_glicko.fit(GameDataset.from_dataframe(serve_df))

            return_df = pl.DataFrame({
                "Player1": return_p1, "Player2": return_p2,
                "Score": return_scores, "Day": valid_days,
            })
            self._return_glicko.fit(GameDataset.from_dataframe(return_df))

        self._num_games_fitted = int(valid.sum())
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        return self

    def update(self, batch: GameBatch,
               serve_stats: Optional[dict] = None) -> "ServeReturnGlicko":
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
        self._serve_glicko.update(serve_batch)

        return_batch = GameBatch(
            player1=batch.player2[valid],
            player2=batch.player1[valid],
            scores=(p2_sw[valid] / p2_st[valid]).astype(np.float64),
            day=batch.day,
        )
        self._return_glicko.update(return_batch)

        self._current_day = batch.day
        self._num_games_fitted += int(valid.sum())
        return self

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        best_of: Union[int, np.ndarray] = 3,
        day: Optional[int] = None,
        surfaces: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """Predict match win probability.

        Args:
            surfaces: Optional int32 array of surface IDs per match
                (0=hard, 1=clay, 2=grass, 3=indoor). If None, assumes hard.
        """
        if not self._fitted:
            raise ValueError("Model not fitted.")

        p1 = np.atleast_1d(np.asarray(player1, dtype=np.int64))
        p2 = np.atleast_1d(np.asarray(player2, dtype=np.int64))

        if surfaces is None:
            surf_ids = np.zeros(len(p1), dtype=np.int32)
        else:
            surf_ids = np.asarray(surfaces, dtype=np.int32)

        p_serve_1, p_serve_2 = _ratings_to_serve_pct(
            self._serve_glicko._ratings.ratings,
            self._serve_glicko._ratings.rd,
            self._return_glicko._ratings.ratings,
            self._return_glicko._ratings.rd,
            self._serve_glicko._ratings.last_played if self._serve_glicko._ratings.last_played is not None else np.zeros(len(self._serve_glicko._ratings.ratings), dtype=np.int32),
            self._return_glicko._ratings.last_played if self._return_glicko._ratings.last_played is not None else np.zeros(len(self._return_glicko._ratings.ratings), dtype=np.int32),
            p1, p2, surf_ids, day or 0,
            self.config.c, self.config.max_rd,
            self.config.mu, self.config.scale,
            self.config.mu_clay, self.config.mu_grass, self.config.mu_indoor,
        )

        if isinstance(best_of, (int, np.integer)):
            bo_arr = np.full(len(p1), int(best_of), dtype=np.int32)
        else:
            bo_arr = np.asarray(best_of, dtype=np.int32)

        result = prob_win_match_batch(p_serve_1, p_serve_2, bo_arr)

        if isinstance(player1, (int, np.integer)):
            return float(result[0])
        return result

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        pass

    def save_state(self, path: str) -> None:
        """Save serve, return, and base ratings to a single .npz."""
        arrays = {
            "ratings": self._ratings.ratings if self._ratings else np.array([]),
            "rd": self._ratings.rd if self._ratings and self._ratings.rd is not None else np.array([]),
            "serve_ratings": self._serve_glicko._ratings.ratings,
            "serve_rd": self._serve_glicko._ratings.rd,
            "return_ratings": self._return_glicko._ratings.ratings,
            "return_rd": self._return_glicko._ratings.rd,
        }
        if self._serve_glicko._ratings.last_played is not None:
            arrays["serve_last_played"] = self._serve_glicko._ratings.last_played
        if self._return_glicko._ratings.last_played is not None:
            arrays["return_last_played"] = self._return_glicko._ratings.last_played
        if self._ratings and self._ratings.last_played is not None:
            arrays["last_played"] = self._ratings.last_played

        import json
        meta = json.dumps({
            "num_players": self._num_players,
            "num_games_fitted": self._num_games_fitted,
            "current_day": self._current_day,
        }).encode()
        arrays["_metadata"] = np.frombuffer(meta, dtype=np.uint8)
        np.savez_compressed(path, **arrays)

    def load_state(self, path: str) -> None:
        """Restore serve, return, and base ratings from .npz."""
        import json
        data = np.load(path, allow_pickle=True)

        meta = json.loads(bytes(data["_metadata"]))
        self._num_players = meta["num_players"]
        self._num_games_fitted = meta.get("num_games_fitted", 0)
        self._current_day = meta.get("current_day")

        # Restore base ratings
        self._ratings = PlayerRatings(
            ratings=data["ratings"],
            rd=data["rd"] if "rd" in data and len(data["rd"]) > 0 else None,
            last_played=data["last_played"] if "last_played" in data else None,
            metadata={"system": "serve_return_glicko"},
        )

        # Restore serve Glicko
        self._serve_glicko._num_players = self._num_players
        self._serve_glicko._ratings = PlayerRatings(
            ratings=data["serve_ratings"],
            rd=data["serve_rd"],
            last_played=data["serve_last_played"] if "serve_last_played" in data else None,
        )
        self._serve_glicko._fitted = True

        # Restore return Glicko
        self._return_glicko._num_players = self._num_players
        self._return_glicko._ratings = PlayerRatings(
            ratings=data["return_ratings"],
            rd=data["return_rd"],
            last_played=data["return_last_played"] if "return_last_played" in data else None,
        )
        self._return_glicko._fitted = True
        self._fitted = True

    def reset(self) -> "ServeReturnGlicko":
        self._serve_glicko.reset()
        self._return_glicko.reset()
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (f"ServeReturnGlicko(c={self.config.c}, mu={self.config.mu:.2f}, "
                f"scale={self.config.scale:.0f}, {self._num_games_fitted:,} games, {status})")
