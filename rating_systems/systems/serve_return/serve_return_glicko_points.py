"""Serve/Return Glicko with point-level observations.

Each serve point is treated as a separate Glicko game:
  - P1 serves 65 points, wins 42 → 42 wins + 23 losses for (P1_serve vs P2_return)
  - P2 serves 70 points, wins 44 → 44 wins + 26 losses for (P2_serve vs P1_return)

All points on the same day form one Glicko period. This gives
mathematically principled RD dynamics — more points = more information
= lower RD, with no ad hoc weighting.

Predictions use standard Glicko expected score for serve point probability:
  E = 1 / (1 + 10^(-g(RD) * (r_serve - r_return) / 400))

with optional surface mu offsets. Point probabilities are then propagated
through the exact scoring model (point → game → set → match).
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from numba import njit

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameDataset, GameBatch
from ..glicko.glicko import Glicko
from ._scoring import prob_win_match_batch


@dataclass
class ServeReturnGlickoPointsConfig:
    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    min_rd: float = 30.0
    max_rd: float = 350.0
    c: float = 10.0
    # Prediction mapping: serve_pct = sigmoid(mu + g(rd) * (r_serve - r_return) / scale)
    # The scale here is different from Glicko's internal 400 — it maps the
    # point-level ratings (which span a wider range) to serve probabilities.
    mu: float = 0.0             # server advantage in logit space
    scale: float = 400.0        # rating units per logit unit for prediction
    # Surface offsets
    mu_clay: float = 0.0
    mu_grass: float = 0.0
    mu_indoor: float = 0.0


def _expand_points(player1, player2, serve_won, serve_total, days,
                   surface_ids=None):
    """Expand match-level serve stats into point-level binary observations.

    For each match, creates serve_won wins (score=1) and
    (serve_total - serve_won) losses (score=0) for the same pair.

    Returns (p1, p2, scores, game_days, surf_ids) arrays where each row
    is a single point.
    """
    n = len(player1)
    # Pre-compute total points to allocate arrays
    total_points = 0
    for i in range(n):
        sw = int(serve_won[i])
        st = int(serve_total[i])
        if st > 0 and sw >= 0 and sw <= st:
            total_points += st

    out_p1 = np.empty(total_points, dtype=np.int64)
    out_p2 = np.empty(total_points, dtype=np.int64)
    out_scores = np.empty(total_points, dtype=np.float64)
    out_days = np.empty(total_points, dtype=np.int32)
    out_surfs = np.empty(total_points, dtype=np.int32) if surface_ids is not None else None

    idx = 0
    for i in range(n):
        sw = int(serve_won[i])
        st = int(serve_total[i])
        if st <= 0 or sw < 0 or sw > st:
            continue

        p1_val = player1[i]
        p2_val = player2[i]
        day_val = days[i]
        surf_val = surface_ids[i] if surface_ids is not None else 0

        # Wins
        for _ in range(sw):
            out_p1[idx] = p1_val
            out_p2[idx] = p2_val
            out_scores[idx] = 1.0
            out_days[idx] = day_val
            if out_surfs is not None:
                out_surfs[idx] = surf_val
            idx += 1

        # Losses
        for _ in range(st - sw):
            out_p1[idx] = p1_val
            out_p2[idx] = p2_val
            out_scores[idx] = 0.0
            out_days[idx] = day_val
            if out_surfs is not None:
                out_surfs[idx] = surf_val
            idx += 1

    return out_p1[:idx], out_p2[:idx], out_scores[:idx], out_days[:idx], \
        out_surfs[:idx] if out_surfs is not None else None


@njit(cache=True)
def _surface_mu_offset(surf, mu_clay, mu_grass, mu_indoor):
    """Return surface offset for Glicko prediction."""
    if surf == 1:
        return mu_clay
    elif surf == 2:
        return mu_grass
    elif surf == 3:
        return mu_indoor
    return 0.0


@njit(cache=True)
def _sigmoid_pts(x):
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


@njit(cache=True)
def _glicko_serve_pct(
    serve_ratings, serve_rd, return_ratings, return_rd,
    serve_last_played, return_last_played,
    p1_ids, p2_ids, surface_ids, day, c, max_rd,
    mu, scale, mu_clay, mu_grass, mu_indoor,
):
    """Predict serve point win probability.

    Uses: serve_pct = sigmoid(mu_eff + g(rd) * (r_serve - r_return) / scale)

    Same formula as srglicko, but the ratings come from point-level Glicko
    training (so they span a wider range and scale is correspondingly larger).
    """
    n = len(p1_ids)
    p_serve_1 = np.empty(n, dtype=np.float64)
    p_serve_2 = np.empty(n, dtype=np.float64)

    for i in range(n):
        p1 = p1_ids[i]
        p2 = p2_ids[i]

        # Surface mu
        mu_eff = mu
        surf = surface_ids[i]
        if surf == 1:
            mu_eff += mu_clay
        elif surf == 2:
            mu_eff += mu_grass
        elif surf == 3:
            mu_eff += mu_indoor

        # Grow RD forward
        rd_s1 = serve_rd[p1]
        lp_s1 = serve_last_played[p1]
        dt = max(0, day - lp_s1)
        if dt > 0:
            rd_s1 = min(math.sqrt(rd_s1*rd_s1 + c*c*dt), max_rd)

        rd_r2 = return_rd[p2]
        lp_r2 = return_last_played[p2]
        dt = max(0, day - lp_r2)
        if dt > 0:
            rd_r2 = min(math.sqrt(rd_r2*rd_r2 + c*c*dt), max_rd)

        rd_s2 = serve_rd[p2]
        lp_s2 = serve_last_played[p2]
        dt = max(0, day - lp_s2)
        if dt > 0:
            rd_s2 = min(math.sqrt(rd_s2*rd_s2 + c*c*dt), max_rd)

        rd_r1 = return_rd[p1]
        lp_r1 = return_last_played[p1]
        dt = max(0, day - lp_r1)
        if dt > 0:
            rd_r1 = min(math.sqrt(rd_r1*rd_r1 + c*c*dt), max_rd)

        # g() shrinks diff when uncertainty is high
        cvar1 = rd_s1*rd_s1 + rd_r2*rd_r2
        g1 = 1.0 / math.sqrt(1.0 + 3.0 * cvar1 / (math.pi * math.pi * scale * scale))
        cvar2 = rd_s2*rd_s2 + rd_r1*rd_r1
        g2 = 1.0 / math.sqrt(1.0 + 3.0 * cvar2 / (math.pi * math.pi * scale * scale))

        logit_1 = mu_eff + g1 * (serve_ratings[p1] - return_ratings[p2]) / scale
        logit_2 = mu_eff + g2 * (serve_ratings[p2] - return_ratings[p1]) / scale

        p_serve_1[i] = max(0.35, min(0.90, _sigmoid_pts(logit_1)))
        p_serve_2[i] = max(0.35, min(0.90, _sigmoid_pts(logit_2)))

    return p_serve_1, p_serve_2


class ServeReturnGlickoPoints(RatingSystem):
    """Serve/Return Glicko with point-level observations.

    Each serve point is a separate Glicko game. All points on the same day
    form one Glicko rating period. Predictions use standard Glicko expected
    score for serve point probability, then the exact scoring model for
    match probability.
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        min_rd: float = 30.0,
        max_rd: float = 350.0,
        c: float = 10.0,
        mu: float = 0.0,
        scale: float = 400.0,
        mu_clay: float = 0.0,
        mu_grass: float = 0.0,
        mu_indoor: float = 0.0,
        num_players: Optional[int] = None,
    ):
        self.config = ServeReturnGlickoPointsConfig(
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
            metadata={"system": "serve_return_glicko_points"},
        )

    def fit(
        self,
        dataset: GameDataset,
        serve_stats: Optional[dict] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "ServeReturnGlickoPoints":
        """Fit by expanding serve stats into point-level Glicko observations."""
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
        surface_ids = serve_stats.get("surface_ids")

        valid = (p1_st > 0) & (p2_st > 0) & np.isfinite(p1_sw) & np.isfinite(p2_sw)

        # Build day array for valid matches
        game_days = np.empty(n_games, dtype=np.int32)
        for d in range(len(day_indices)):
            game_days[day_offsets[d]:day_offsets[d + 1]] = day_indices[d]
        valid_days = game_days[valid]
        valid_surfs = surface_ids[valid] if surface_ids is not None else None

        # Expand P1 serve points into binary observations
        serve_p1, serve_p2, serve_scores, serve_days, _ = _expand_points(
            player1[valid], player2[valid],
            p1_sw[valid], p1_st[valid], valid_days,
        )
        # Expand P2 serve points (P2_serve vs P1_return)
        ret_p1, ret_p2, ret_scores, ret_days, _ = _expand_points(
            player2[valid], player1[valid],
            p2_sw[valid], p2_st[valid], valid_days,
        )

        import polars as pl
        if len(serve_p1) > 0:
            serve_df = pl.DataFrame({
                "Player1": serve_p1, "Player2": serve_p2,
                "Score": serve_scores, "Day": serve_days,
            })
            self._serve_glicko.fit(GameDataset.from_dataframe(serve_df))

        if len(ret_p1) > 0:
            return_df = pl.DataFrame({
                "Player1": ret_p1, "Player2": ret_p2,
                "Score": ret_scores, "Day": ret_days,
            })
            self._return_glicko.fit(GameDataset.from_dataframe(return_df))

        self._num_games_fitted = int(valid.sum())
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None
        return self

    def update(self, batch: GameBatch,
               serve_stats: Optional[dict] = None) -> "ServeReturnGlickoPoints":
        """Incrementally update with point-level observations."""
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

        day = batch.day
        day_arr = np.full(int(valid.sum()), day, dtype=np.int32)

        # Expand serve points
        serve_p1, serve_p2, serve_sc, _, _ = _expand_points(
            batch.player1[valid], batch.player2[valid],
            p1_sw[valid], p1_st[valid], day_arr,
        )
        if len(serve_p1) > 0:
            self._serve_glicko.update(GameBatch(
                player1=serve_p1, player2=serve_p2,
                scores=serve_sc, day=day,
            ))

        # Expand return points
        ret_p1, ret_p2, ret_sc, _, _ = _expand_points(
            batch.player2[valid], batch.player1[valid],
            p2_sw[valid], p2_st[valid], day_arr,
        )
        if len(ret_p1) > 0:
            self._return_glicko.update(GameBatch(
                player1=ret_p1, player2=ret_p2,
                scores=ret_sc, day=day,
            ))

        self._current_day = day
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

        Uses standard Glicko expected score for serve point probability,
        then exact scoring model for match probability.
        """
        if not self._fitted:
            raise ValueError("Model not fitted.")

        p1 = np.atleast_1d(np.asarray(player1, dtype=np.int64))
        p2 = np.atleast_1d(np.asarray(player2, dtype=np.int64))

        if surfaces is None:
            surf_ids = np.zeros(len(p1), dtype=np.int32)
        else:
            surf_ids = np.asarray(surfaces, dtype=np.int32)

        serve_lp = self._serve_glicko._ratings.last_played
        return_lp = self._return_glicko._ratings.last_played
        n_ratings = len(self._serve_glicko._ratings.ratings)

        p_serve_1, p_serve_2 = _glicko_serve_pct(
            self._serve_glicko._ratings.ratings,
            self._serve_glicko._ratings.rd,
            self._return_glicko._ratings.ratings,
            self._return_glicko._ratings.rd,
            serve_lp if serve_lp is not None else np.zeros(n_ratings, dtype=np.int32),
            return_lp if return_lp is not None else np.zeros(n_ratings, dtype=np.int32),
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

    def save_state(self, path: str) -> None:
        """Save serve, return, and base ratings."""
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

        meta = json.dumps({
            "num_players": self._num_players,
            "num_games_fitted": self._num_games_fitted,
            "current_day": self._current_day,
        }).encode()
        arrays["_metadata"] = np.frombuffer(meta, dtype=np.uint8)
        np.savez_compressed(path, **arrays)

    def load_state(self, path: str) -> None:
        """Restore serve, return, and base ratings."""
        data = np.load(path, allow_pickle=True)
        meta = json.loads(bytes(data["_metadata"]))
        self._num_players = meta["num_players"]
        self._num_games_fitted = meta.get("num_games_fitted", 0)
        self._current_day = meta.get("current_day")

        self._ratings = PlayerRatings(
            ratings=data["ratings"],
            rd=data["rd"] if "rd" in data and len(data["rd"]) > 0 else None,
            last_played=data["last_played"] if "last_played" in data else None,
            metadata={"system": "serve_return_glicko_points"},
        )
        self._serve_glicko._num_players = self._num_players
        self._serve_glicko._ratings = PlayerRatings(
            ratings=data["serve_ratings"],
            rd=data["serve_rd"],
            last_played=data["serve_last_played"] if "serve_last_played" in data else None,
        )
        self._serve_glicko._fitted = True
        self._return_glicko._num_players = self._num_players
        self._return_glicko._ratings = PlayerRatings(
            ratings=data["return_ratings"],
            rd=data["return_rd"],
            last_played=data["return_last_played"] if "return_last_played" in data else None,
        )
        self._return_glicko._fitted = True
        self._fitted = True

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        pass

    def reset(self) -> "ServeReturnGlickoPoints":
        self._serve_glicko.reset()
        self._return_glicko.reset()
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (f"ServeReturnGlickoPoints(c={self.config.c}, "
                f"{self._num_games_fitted:,} matches, {status})")
