"""
Disc decomposition rating system (Bertrand, Czarnecki & Gidel, AISTATS 2023).

Each player has a skill u_i and consistency v_i:

    P(i beats j) = sigmoid(v_i * v_j * (u_i - u_j) * log10 / scale)

When v_i = v_j = 1 the model reduces to standard Elo.  The consistency
parameter captures how predictable a player's results are — high-consistency
players amplify skill differences, low-consistency players create upsets.

Upsets decrease both players' consistency; expected results increase it.

Reference: Bertrand, Czarnecki & Gidel, "On the Limitations of the Elo
Rating System", AISTATS 2023.  arXiv:2206.12301
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ._numba_core import (
    update_ratings_sequential,
    predict_proba_batch,
    predict_single,
    fit_all_days,
)


@dataclass
class DiscConfig:
    """Configuration for Disc decomposition rating system."""

    initial_rating: float = 1500.0
    initial_consistency: float = 1.0
    k_factor: float = 32.0
    consistency_lr: float = 0.1
    scale: float = 400.0
    v_floor: float = 0.01


class Disc(RatingSystem):
    """Disc decomposition rating system with Numba acceleration.

    Extends standard Elo with a per-player consistency parameter that
    modulates how decisive skill differences are.
    """

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
        self.config = DiscConfig(
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
        update_ratings_sequential(
            batch.player1, batch.player2, batch.scores,
            ratings.ratings, self._consistency,
            self.config.k_factor, self.config.consistency_lr,
            self.config.scale, self.config.v_floor,
        )
        self._num_games_fitted += len(batch.player1)

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names=None,
    ):
        """Train on dataset, optionally up to end_day."""
        # Initialize if needed
        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)
        elif self._consistency is not None and len(self._consistency) < dataset.num_players:
            self._extend_players(dataset.num_players)

        p1, p2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if end_day is not None:
            mask = day_indices <= end_day
            n_games = int(mask.sum())
            p1, p2, scores = p1[:n_games], p2[:n_games], scores[:n_games]
            unique_days = sorted(set(day_indices[mask].tolist()))
            new_offsets = [0]
            idx = 0
            for d in unique_days:
                while idx < n_games and day_indices[idx] == d:
                    idx += 1
                new_offsets.append(idx)
            day_offsets = np.array(new_offsets, dtype=np.int64)

        fit_all_days(
            p1, p2, scores, day_offsets,
            self._ratings.ratings, self._consistency,
            self.config.k_factor, self.config.consistency_lr,
            self.config.scale, self.config.v_floor,
        )
        self._num_games_fitted += len(p1)
        self._fitted = True
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else -1

    def predict_proba(
        self,
        player1: Union[int, np.ndarray],
        player2: Union[int, np.ndarray],
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict P(player1 wins)."""
        if isinstance(player1, (int, np.integer)):
            return predict_single(
                self._ratings.ratings[player1],
                self._ratings.ratings[player2],
                self._consistency[player1],
                self._consistency[player2],
                self.config.scale,
            )
        player1 = np.asarray(player1, dtype=np.int64)
        player2 = np.asarray(player2, dtype=np.int64)
        return predict_proba_batch(
            player1, player2,
            self._ratings.ratings, self._consistency,
            self.config.scale,
        )

    def get_consistency(self) -> np.ndarray:
        """Return a copy of the consistency array."""
        return self._consistency.copy()

    def _extend_players(self, new_num_players: int):
        """Extend arrays for new players."""
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
        """Save skills and consistency to .npz."""
        np.savez_compressed(
            path,
            ratings=self._ratings.ratings,
            consistency=self._consistency,
            current_day=np.array([self._current_day]),
            num_games=np.array([self._num_games_fitted]),
        )

    def load_state(self, path: str) -> None:
        """Load skills and consistency from .npz."""
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
            f"Disc(k_factor={self.config.k_factor}, "
            f"consistency_lr={self.config.consistency_lr}, "
            f"scale={self.config.scale}, "
            f"players={self._num_players})"
        )
