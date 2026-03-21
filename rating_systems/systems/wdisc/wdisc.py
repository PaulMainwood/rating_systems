"""
Weighted Disc decomposition rating system.

Extends Disc with per-game weights (scaling update magnitude) and
handicaps (shifting the skill difference).  Used by the covariate
pipeline in tennis_ratings.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ._numba_core import (
    update_ratings_weighted_sequential,
    predict_proba_batch,
    predict_single,
    predict_single_with_handicap,
    fit_all_days_weighted,
    walk_forward_predict_update,
)


@dataclass
class WDiscConfig:
    """Configuration for weighted Disc decomposition."""

    initial_rating: float = 1500.0
    initial_consistency: float = 1.0
    k_factor: float = 32.0
    consistency_lr: float = 0.1
    scale: float = 400.0
    v_floor: float = 0.01


class WDisc(RatingSystem):
    """Weighted Disc decomposition with per-game weights and handicaps."""

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_consistency: float = 1.0,
        k_factor: float = 32.0,
        consistency_lr: float = 0.1,
        scale: float = 400.0,
        v_floor: float = 0.01,
        num_players: Optional[int] = None,
    ):
        self.config = WDiscConfig(
            initial_rating=initial_rating,
            initial_consistency=initial_consistency,
            k_factor=k_factor,
            consistency_lr=consistency_lr,
            scale=scale,
            v_floor=v_floor,
        )
        self._consistency: Optional[np.ndarray] = None
        self._num_games_fitted = 0
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        skills = np.full(num_players, self.config.initial_rating, dtype=np.float64)
        self._consistency = np.full(
            num_players, self.config.initial_consistency, dtype=np.float64
        )
        return PlayerRatings(ratings=skills)

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        n = len(batch.player1)
        weights = np.ones(n, dtype=np.float64)
        handicaps = np.zeros(n, dtype=np.float64)
        update_ratings_weighted_sequential(
            batch.player1, batch.player2, batch.scores,
            ratings.ratings, self._consistency,
            self.config.k_factor, self.config.consistency_lr,
            self.config.scale, self.config.v_floor,
            weights, handicaps,
        )
        self._num_games_fitted += n

    def update_weighted(
        self,
        batch: GameBatch,
        weights: np.ndarray,
        handicaps: Optional[np.ndarray] = None,
    ) -> None:
        """Incremental update with per-game weights and handicaps."""
        if self._ratings is None:
            raise RuntimeError("System not initialised — call fit() first")
        n = len(batch.player1)

        # Extend for new players
        max_id = max(int(batch.player1.max()), int(batch.player2.max()))
        if max_id >= len(self._ratings.ratings):
            self._extend_players(max_id + 1)

        if handicaps is None:
            handicaps = np.zeros(n, dtype=np.float64)
        update_ratings_weighted_sequential(
            batch.player1, batch.player2, batch.scores,
            self._ratings.ratings, self._consistency,
            self.config.k_factor, self.config.consistency_lr,
            self.config.scale, self.config.v_floor,
            weights, handicaps,
        )
        self._num_games_fitted += n
        self._current_day = int(batch.day) if hasattr(batch, "day") else self._current_day

    def fit(
        self,
        dataset: GameDataset,
        weights: Optional[np.ndarray] = None,
        handicaps: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names=None,
    ):
        """Train on dataset with optional per-game weights and handicaps."""
        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)
        elif self._consistency is not None and len(self._consistency) < dataset.num_players:
            self._extend_players(dataset.num_players)

        p1, p2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if end_day is not None and weights is not None:
            raise ValueError("Cannot use both end_day and weights")

        if weights is None:
            weights = np.ones(len(p1), dtype=np.float64)
        if handicaps is None:
            handicaps = np.zeros(len(p1), dtype=np.float64)

        if end_day is not None:
            mask = day_indices <= end_day
            n_games = int(mask.sum())
            p1, p2, scores = p1[:n_games], p2[:n_games], scores[:n_games]
            weights, handicaps = weights[:n_games], handicaps[:n_games]
            unique_days = sorted(set(day_indices[mask].tolist()))
            new_offsets = [0]
            idx = 0
            for d in unique_days:
                while idx < n_games and day_indices[idx] == d:
                    idx += 1
                new_offsets.append(idx)
            day_offsets = np.array(new_offsets, dtype=np.int64)

        fit_all_days_weighted(
            p1, p2, scores, day_offsets,
            self._ratings.ratings, self._consistency,
            self.config.k_factor, self.config.consistency_lr,
            self.config.scale, self.config.v_floor,
            weights, handicaps,
        )
        self._num_games_fitted += len(p1)
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else -1

    def predict_proba(
        self,
        player1: Union[int, np.ndarray],
        player2: Union[int, np.ndarray],
        handicaps: Optional[np.ndarray] = None,
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict P(player1 wins), optionally with handicaps."""
        if isinstance(player1, (int, np.integer)):
            h = 0.0 if handicaps is None else float(handicaps)
            if h == 0.0:
                return predict_single(
                    self._ratings.ratings[player1],
                    self._ratings.ratings[player2],
                    self._consistency[player1],
                    self._consistency[player2],
                    self.config.scale,
                )
            return predict_single_with_handicap(
                self._ratings.ratings[player1],
                self._ratings.ratings[player2],
                self._consistency[player1],
                self._consistency[player2],
                self.config.scale, h,
            )
        player1 = np.asarray(player1, dtype=np.int64)
        player2 = np.asarray(player2, dtype=np.int64)
        if handicaps is None:
            handicaps = np.zeros(len(player1), dtype=np.float64)
        return predict_proba_batch(
            player1, player2,
            self._ratings.ratings, self._consistency,
            self.config.scale, handicaps,
        )

    def fused_walk_forward(
        self,
        player1: np.ndarray,
        player2: np.ndarray,
        scores: np.ndarray,
        day_indices: np.ndarray,
        day_offsets: np.ndarray,
        weights: np.ndarray,
        handicaps: np.ndarray,
        n_train_days: int,
    ) -> np.ndarray:
        """Fused predict-then-update walk-forward. Returns predictions."""
        return walk_forward_predict_update(
            player1, player2, scores, day_indices, day_offsets,
            self._ratings.ratings, self._consistency,
            self.config.k_factor, self.config.consistency_lr,
            self.config.scale, self.config.v_floor,
            weights, handicaps, n_train_days,
        )

    def get_consistency(self) -> np.ndarray:
        return self._consistency.copy()

    def _extend_players(self, new_num_players: int):
        old_n = len(self._ratings.ratings)
        if new_num_players <= old_n:
            return
        new_skills = np.full(new_num_players, self.config.initial_rating, dtype=np.float64)
        new_skills[:old_n] = self._ratings.ratings
        self._ratings = PlayerRatings(ratings=new_skills)

        new_v = np.full(new_num_players, self.config.initial_consistency, dtype=np.float64)
        new_v[:old_n] = self._consistency
        self._consistency = new_v

        self._num_players = new_num_players

    def save_state(self, path: str) -> None:
        np.savez_compressed(
            path,
            ratings=self._ratings.ratings,
            consistency=self._consistency,
            current_day=np.array([self._current_day]),
            num_games=np.array([self._num_games_fitted]),
        )

    def load_state(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        skills = data["ratings"]
        self._num_players = len(skills)
        self._ratings = PlayerRatings(ratings=skills)
        self._consistency = data["consistency"]
        if "current_day" in data:
            self._current_day = int(data["current_day"][0])
        if "num_games" in data:
            self._num_games_fitted = int(data["num_games"][0])
        self._fitted = True

    def __repr__(self):
        return (
            f"WDisc(k_factor={self.config.k_factor}, "
            f"consistency_lr={self.config.consistency_lr}, "
            f"scale={self.config.scale}, "
            f"players={self._num_players})"
        )
