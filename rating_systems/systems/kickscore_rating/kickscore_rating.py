"""
KickScore rating system — native implementation.

Gaussian process paired comparisons with Constant + Matérn 3/2 kernel.
Inference via Expectation Propagation on the state-space representation
with Kalman filter/RTS smoother per player.

Each player's skill s(t) = c + f(t) where:
- c ~ N(0, var_constant)  is a time-invariant baseline
- f(t) ~ GP(0, k_matern)  captures smooth form variation

Match outcome: P(i beats j at time t) = sigmoid(s_i(t) - s_j(t))

Reference: Maystre, Kristof & Grossglauser, "Pairwise Comparisons with
Flexible Time-Dynamics", KDD 2019.  arXiv:1903.07746
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ._numba_core import (
    kalman_forward_backward,
    predict_at_time,
    ep_update_match,
    mm_logit_win,
)


@dataclass
class KickScoreConfig:
    """Configuration for KickScore rating system."""
    var_constant: float = 0.05   # baseline skill variance
    var_matern: float = 0.15     # form-variation variance
    lscale: float = 365.0        # lengthscale in days
    max_iter: int = 30           # EP iterations
    tol: float = 1e-3            # EP convergence tolerance
    ep_lr: float = 1.0           # EP damping (1.0 = no damping)


class KickScoreRating(RatingSystem):
    """KickScore with Constant + Matérn 3/2 kernel, native implementation."""

    system_type = RatingSystemType.BATCH

    def __init__(
        self,
        var_constant: float = 0.05,
        var_matern: float = 0.15,
        lscale: float = 365.0,
        obs_type: str = "logit",
        max_iter: int = 30,
        tol: float = 1e-3,
        ep_lr: float = 1.0,
        num_players: Optional[int] = None,
    ):
        self.config = KickScoreConfig(
            var_constant=var_constant,
            var_matern=var_matern,
            lscale=lscale,
            max_iter=max_iter,
            tol=tol,
            ep_lr=ep_lr,
        )
        self._lambda = math.sqrt(3.0) / lscale

        # Per-player state: dict of player_id -> PlayerState
        self._players: Dict[int, "_PlayerState"] = {}

        # Match list: [(t, p1_id, p2_id, winner_is_p1, p1_obs_idx, p2_obs_idx)]
        self._matches: List[Tuple[float, int, int, bool, int, int]] = []
        self._match_logparts: List[float] = []

        self._last_day = -1
        self._num_games_fitted = 0
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        return PlayerRatings(ratings=np.zeros(num_players, dtype=np.float64))

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Not used directly — update() is overridden for batch systems."""
        pass

    def _ensure_player(self, pid: int) -> "_PlayerState":
        if pid not in self._players:
            self._players[pid] = _PlayerState()
        return self._players[pid]

    def _add_match(self, p1: int, p2: int, score: float, t: float) -> None:
        """Add a single match observation."""
        ps1 = self._ensure_player(p1)
        ps2 = self._ensure_player(p2)
        idx1 = ps1.add_observation(t)
        idx2 = ps2.add_observation(t)
        winner_is_p1 = score >= 0.5
        self._matches.append((t, p1, p2, winner_is_p1, idx1, idx2))
        self._match_logparts.append(0.0)

    def _run_ep(self, verbose: bool = False) -> bool:
        """Run EP iterations over all matches."""
        # Allocate/extend per-player arrays
        for ps in self._players.values():
            ps.allocate(self.config.var_constant, self.config.var_matern, self._lambda)

        for iteration in range(self.config.max_iter):
            # Step 1: EP updates for all matches
            max_diff = 0.0
            for m_idx, (t, p1, p2, winner_is_p1, idx1, idx2) in enumerate(self._matches):
                ps1 = self._players[p1]
                ps2 = self._players[p2]

                if winner_is_p1:
                    diff = ep_update_match(
                        ps1.ms, ps1.vs, ps1.ns, ps1.xs, idx1,
                        ps2.ms, ps2.vs, ps2.ns, ps2.xs, idx2,
                        self.config.ep_lr,
                    )
                else:
                    # Player 2 won: swap roles
                    diff = ep_update_match(
                        ps2.ms, ps2.vs, ps2.ns, ps2.xs, idx2,
                        ps1.ms, ps1.vs, ps1.ns, ps1.xs, idx1,
                        self.config.ep_lr,
                    )

                old_lp = self._match_logparts[m_idx]
                self._match_logparts[m_idx] = diff
                max_diff = max(max_diff, abs(diff - old_lp))

            # Step 2: Recompute posteriors via Kalman forward-backward
            for ps in self._players.values():
                ps.run_kalman(
                    self.config.var_constant, self.config.var_matern, self._lambda
                )

            if verbose:
                print(f"  EP iter {iteration + 1}: max_diff={max_diff:.6f}")

            if max_diff < self.config.tol:
                return True

        return False

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
        player_names=None,
    ):
        """Fit on dataset via EP."""
        # Reset
        self._players = {}
        self._matches = []
        self._match_logparts = []

        p1, p2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        # Expand day_indices to per-game
        n_games = len(p1)
        game_days = np.empty(n_games, dtype=np.float64)
        for d_idx in range(len(day_indices)):
            start = day_offsets[d_idx]
            end_off = day_offsets[d_idx + 1] if d_idx + 1 < len(day_offsets) else n_games
            game_days[start:end_off] = float(day_indices[d_idx])

        if end_day is not None:
            mask = game_days <= float(end_day)
            n = int(mask.sum())
            p1, p2, scores, game_days = p1[:n], p2[:n], scores[:n], game_days[:n]

        if len(p1) == 0:
            self._fitted = True
            return

        # Add all matches
        for i in range(len(p1)):
            self._add_match(int(p1[i]), int(p2[i]), float(scores[i]), game_days[i])

        self._last_day = int(game_days[-1])
        self._num_games_fitted = len(p1)

        # Run EP
        self._run_ep()

        # Sync ratings
        num_players = max(self._players.keys()) + 1
        if self._num_players is None or self._num_players < num_players:
            self._num_players = num_players
            self._ratings = self._initialize_ratings(self._num_players)
        self._sync_ratings()
        self._fitted = True
        self._current_day = self._last_day

    def update(self, batch: GameBatch):
        """Incremental update: add new matches and re-run EP (warm-started)."""
        day = float(batch.day) if hasattr(batch, 'day') and batch.day is not None else float(self._last_day + 1)
        for i in range(len(batch.player1)):
            self._add_match(
                int(batch.player1[i]), int(batch.player2[i]),
                float(batch.scores[i]), day,
            )
        self._last_day = int(day)
        self._num_games_fitted += len(batch.player1)

        self._run_ep()

        # Extend ratings if needed
        if self._players:
            max_pid = max(self._players.keys()) + 1
            if self._num_players is None or self._num_players < max_pid:
                old = self._ratings.ratings if self._ratings is not None else np.array([])
                self._num_players = max_pid
                self._ratings = self._initialize_ratings(max_pid)
                self._ratings.ratings[:len(old)] = old
        self._sync_ratings()
        self._current_day = self._last_day
        return self

    def _sync_ratings(self) -> None:
        """Copy posterior means into ratings array."""
        if self._ratings is None:
            return
        t = float(self._last_day)
        for pid, ps in self._players.items():
            if pid < len(self._ratings.ratings) and len(ps.ts) > 0:
                m, _ = ps.predict(
                    t, self.config.var_constant, self.config.var_matern, self._lambda
                )
                self._ratings.ratings[pid] = m

    def predict_proba(
        self,
        player1: Union[int, np.ndarray],
        player2: Union[int, np.ndarray],
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict P(player1 wins)."""
        t = float(day) if day is not None else float(self._last_day)

        if isinstance(player1, (int, np.integer)):
            m1, v1 = self._get_score(int(player1), t)
            m2, v2 = self._get_score(int(player2), t)
            # P(1 beats 2) via logistic moment matching
            f_mean = m1 - m2
            f_var = v1 + v2
            logp, _, _ = mm_logit_win(f_mean, f_var)
            return float(np.clip(math.exp(logp), 1e-9, 1 - 1e-9))

        player1 = np.asarray(player1, dtype=np.int64)
        player2 = np.asarray(player2, dtype=np.int64)
        n = len(player1)
        preds = np.full(n, 0.5, dtype=np.float64)
        for i in range(n):
            m1, v1 = self._get_score(int(player1[i]), t)
            m2, v2 = self._get_score(int(player2[i]), t)
            f_mean = m1 - m2
            f_var = v1 + v2
            logp, _, _ = mm_logit_win(f_mean, f_var)
            preds[i] = np.clip(math.exp(logp), 1e-9, 1 - 1e-9)
        return preds

    def _get_score(self, pid: int, t: float) -> Tuple[float, float]:
        """Get (mean, variance) for a player at time t."""
        ps = self._players.get(pid)
        if ps is None or len(ps.ts) == 0:
            # Unknown player — return prior
            return 0.0, self.config.var_constant + self.config.var_matern
        return ps.predict(
            t, self.config.var_constant, self.config.var_matern, self._lambda
        )

    def save_state(self, path: str) -> None:
        """Save full EP state for later prediction."""
        import pickle
        state = {
            "config": self.config.__dict__,
            "players": {
                pid: ps.to_dict() for pid, ps in self._players.items()
            },
            "last_day": self._last_day,
            "num_games": self._num_games_fitted,
        }
        np.savez_compressed(path, meta=np.array([self._last_day, self._num_games_fitted]))
        pkl_path = path.replace(".npz", "_ks.pkl")
        if pkl_path == path:
            pkl_path = path + ".pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: str) -> None:
        """Load saved state for prediction."""
        import pickle
        pkl_path = path.replace(".npz", "_ks.pkl")
        if pkl_path == path:
            pkl_path = path + ".pkl"
        try:
            with open(pkl_path, "rb") as f:
                state = pickle.load(f)
        except FileNotFoundError:
            data = np.load(path, allow_pickle=False)
            self._last_day = int(data["meta"][0])
            self._num_games_fitted = int(data["meta"][1])
            self._fitted = True
            return

        cfg = state["config"]
        self.config = KickScoreConfig(**cfg)
        self._lambda = math.sqrt(3.0) / self.config.lscale
        self._last_day = state["last_day"]
        self._num_games_fitted = state["num_games"]

        self._players = {
            int(pid): _PlayerState.from_dict(d)
            for pid, d in state["players"].items()
        }

        if self._players:
            self._num_players = max(self._players.keys()) + 1
        else:
            self._num_players = 0
        self._ratings = self._initialize_ratings(self._num_players)
        self._sync_ratings()
        self._fitted = True
        self._current_day = self._last_day

    def __repr__(self):
        return (
            f"KickScoreRating(var_c={self.config.var_constant}, "
            f"var_m={self.config.var_matern}, "
            f"lscale={self.config.lscale}, "
            f"players={len(self._players)})"
        )


class _PlayerState:
    """Per-player EP/Kalman state."""

    def __init__(self):
        self.ts_pending: List[float] = []
        # Allocated arrays
        self.ts = np.zeros(0, dtype=np.float64)
        self.ms = np.zeros(0, dtype=np.float64)   # posterior means
        self.vs = np.zeros(0, dtype=np.float64)   # posterior variances
        self.ns = np.zeros(0, dtype=np.float64)   # EP natural means
        self.xs = np.zeros(0, dtype=np.float64)   # EP precisions
        # Kalman arrays (3D state)
        self.m_p = np.zeros((0, 3), dtype=np.float64)
        self.P_p = np.zeros((0, 3, 3), dtype=np.float64)
        self.m_f = np.zeros((0, 3), dtype=np.float64)
        self.P_f = np.zeros((0, 3, 3), dtype=np.float64)
        self.m_s = np.zeros((0, 3), dtype=np.float64)
        self.P_s = np.zeros((0, 3, 3), dtype=np.float64)

    def add_observation(self, t: float) -> int:
        """Register an observation time, return its index."""
        idx = len(self.ts) + len(self.ts_pending)
        self.ts_pending.append(t)
        return idx

    def allocate(self, var_c: float, var_m: float, lambda_: float) -> None:
        """Extend arrays for pending observations."""
        n_new = len(self.ts_pending)
        if n_new == 0:
            return

        n_old = len(self.ts)
        n_total = n_old + n_new

        self.ts = np.concatenate([self.ts, np.array(self.ts_pending)])
        self.ms = np.concatenate([self.ms, np.zeros(n_new)])
        # Initial variance from prior
        init_var = var_c + var_m
        self.vs = np.concatenate([self.vs, np.full(n_new, init_var)])
        self.ns = np.concatenate([self.ns, np.zeros(n_new)])
        self.xs = np.concatenate([self.xs, np.zeros(n_new)])

        # Kalman arrays
        m0 = np.zeros(3)
        from ._numba_core import compute_initial_cov
        P0 = compute_initial_cov(var_c, var_m, lambda_)

        new_m = np.tile(m0, (n_new, 1))
        new_P = np.tile(P0, (n_new, 1, 1))

        self.m_p = np.concatenate([self.m_p, new_m.copy()])
        self.P_p = np.concatenate([self.P_p, new_P.copy()])
        self.m_f = np.concatenate([self.m_f, new_m.copy()])
        self.P_f = np.concatenate([self.P_f, new_P.copy()])
        self.m_s = np.concatenate([self.m_s, new_m.copy()])
        self.P_s = np.concatenate([self.P_s, new_P.copy()])

        self.ts_pending = []

    def run_kalman(self, var_c: float, var_m: float, lambda_: float) -> None:
        """Run Kalman forward-backward, updating ms and vs."""
        if len(self.ts) == 0:
            return
        kalman_forward_backward(
            self.ts, self.ns, self.xs,
            var_c, var_m, lambda_,
            self.m_p, self.P_p, self.m_f, self.P_f, self.m_s, self.P_s,
            self.ms, self.vs,
        )

    def predict(self, t: float, var_c: float, var_m: float, lambda_: float) -> Tuple[float, float]:
        """Predict (mean, variance) at arbitrary time t."""
        return predict_at_time(
            t, self.ts, self.m_f, self.P_f, self.m_s, self.P_s,
            self.m_p, self.P_p,
            var_c, var_m, lambda_,
        )

    def to_dict(self) -> dict:
        return {
            "ts": self.ts, "ms": self.ms, "vs": self.vs,
            "ns": self.ns, "xs": self.xs,
            "m_p": self.m_p, "P_p": self.P_p,
            "m_f": self.m_f, "P_f": self.P_f,
            "m_s": self.m_s, "P_s": self.P_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_PlayerState":
        ps = cls()
        ps.ts = d["ts"]
        ps.ms = d["ms"]
        ps.vs = d["vs"]
        ps.ns = d["ns"]
        ps.xs = d["xs"]
        ps.m_p = d["m_p"]
        ps.P_p = d["P_p"]
        ps.m_f = d["m_f"]
        ps.P_f = d["P_f"]
        ps.m_s = d["m_s"]
        ps.P_s = d["P_s"]
        return ps
