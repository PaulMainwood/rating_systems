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
    mm_win,
)


OBS_TYPES = {"logit": 0, "probit": 1}


@dataclass
class KickScoreConfig:
    """Configuration for KickScore rating system."""
    var_constant: float = 0.05   # baseline skill variance
    var_matern: float = 0.15     # form-variation variance
    lscale: float = 365.0        # lengthscale in days
    obs_type: str = "logit"      # likelihood: "logit" (BT) or "probit"
    max_iter: int = 30           # EP iterations for initial fit
    refit_max_iterations: int = 1  # EP iterations for warm-started updates
    refit_interval: int = 1      # days between EP refits during update()
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
        refit_max_iterations: int = 1,
        refit_interval: int = 1,
        tol: float = 1e-3,
        ep_lr: float = 1.0,
        num_players: Optional[int] = None,
    ):
        if obs_type not in OBS_TYPES:
            raise ValueError(
                f"obs_type must be one of {sorted(OBS_TYPES)}, got {obs_type!r}"
            )
        self.config = KickScoreConfig(
            var_constant=var_constant,
            var_matern=var_matern,
            lscale=lscale,
            obs_type=obs_type,
            max_iter=max_iter,
            refit_max_iterations=refit_max_iterations,
            refit_interval=refit_interval,
            tol=tol,
            ep_lr=ep_lr,
        )
        self._obs_type = OBS_TYPES[obs_type]
        self._lambda = math.sqrt(3.0) / lscale
        self._last_refit_day: Optional[int] = None

        # Per-player state: dict of player_id -> PlayerState
        self._players: Dict[int, "_PlayerState"] = {}

        # Match list: [(t, p1_id, p2_id, winner_is_p1, p1_obs_idx, p2_obs_idx)]
        self._matches: List[Tuple[float, int, int, bool, int, int]] = []
        self._match_logparts: List[float] = []

        self._last_day = -1
        self._num_games_fitted = 0
        self._ratings_stale = False
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

    def _run_ep(self, max_iter: Optional[int] = None, verbose: bool = False) -> bool:
        """Run EP iterations over all matches."""
        if max_iter is None:
            max_iter = self.config.max_iter

        # Allocate/extend per-player arrays
        for ps in self._players.values():
            ps.allocate(self.config.var_constant, self.config.var_matern, self._lambda)

        for iteration in range(max_iter):
            # Step 1: EP updates for all matches
            max_diff = 0.0
            for m_idx, (t, p1, p2, winner_is_p1, idx1, idx2) in enumerate(self._matches):
                ps1 = self._players[p1]
                ps2 = self._players[p2]

                if winner_is_p1:
                    diff = ep_update_match(
                        ps1.ms, ps1.vs, ps1.ns, ps1.xs, idx1,
                        ps2.ms, ps2.vs, ps2.ns, ps2.xs, idx2,
                        self.config.ep_lr, self._obs_type,
                    )
                else:
                    # Player 2 won: swap roles
                    diff = ep_update_match(
                        ps2.ms, ps2.vs, ps2.ns, ps2.xs, idx2,
                        ps1.ms, ps1.vs, ps1.ns, ps1.xs, idx1,
                        self.config.ep_lr, self._obs_type,
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
        self._ratings_stale = False
        self._last_refit_day = self._last_day
        self._fitted = True
        self._current_day = self._last_day

    def update(self, batch: GameBatch):
        """Incremental update: add new matches and re-run EP (warm-started).

        EP replays the whole match history, so running it every day makes a
        walk-forward quadratic in the dataset. `refit_interval` amortises it:
        matches accumulate cheaply and EP runs every N days, warm-started from
        the existing sites (hence refit_max_iterations can be small).

        Only the EVIDENCE goes stale between refits, never the clock: _last_day
        advances every day, and predict_proba propagates the GP forward from the
        last refit through the Matérn transition. So predictions stay
        time-correct; they just haven't seen the last <N days of results.
        """
        day = float(batch.day) if hasattr(batch, 'day') and batch.day is not None else float(self._last_day + 1)
        for i in range(len(batch.player1)):
            self._add_match(
                int(batch.player1[i]), int(batch.player2[i]),
                float(batch.scores[i]), day,
            )
        self._last_day = int(day)
        self._num_games_fitted += len(batch.player1)

        if self._last_refit_day is None:
            self._last_refit_day = int(day)
        interval = max(1, int(self.config.refit_interval))
        if int(day) - self._last_refit_day >= interval:
            # Warm-started, so few iterations suffice.
            self._run_ep(max_iter=self.config.refit_max_iterations)
            self._last_refit_day = int(day)
            self._ratings_stale = True

        # Extend ratings if needed
        if self._players:
            max_pid = max(self._players.keys()) + 1
            if self._num_players is None or self._num_players < max_pid:
                old = self._ratings.ratings if self._ratings is not None else np.array([])
                self._num_players = max_pid
                self._ratings = self._initialize_ratings(max_pid)
                self._ratings.ratings[:len(old)] = old
                self._ratings_stale = True
        # Syncing walks every player with a 3x3 solve each, so only do it when
        # the posterior has actually moved. predict_proba reads _get_score
        # directly and never touches _ratings, so nothing else observes this.
        if self._ratings_stale:
            self._sync_ratings()
            self._ratings_stale = False
        self._current_day = self._last_day
        return self

    def get_ratings(self) -> PlayerRatings:
        """Current ratings, syncing lazily if a refit has moved the posterior."""
        if self._ratings_stale:
            self._sync_ratings()
            self._ratings_stale = False
        return super().get_ratings()

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
            # P(1 beats 2) — marginalise the skill uncertainty into the
            # likelihood via the same moment matching EP uses.
            f_mean = m1 - m2
            f_var = v1 + v2
            logp, _, _ = mm_win(f_mean, f_var, self._obs_type)
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
            logp, _, _ = mm_win(f_mean, f_var, self._obs_type)
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
        self._obs_type = OBS_TYPES[self.config.obs_type]
        self._lambda = math.sqrt(3.0) / self.config.lscale
        self._last_day = state["last_day"]
        self._num_games_fitted = state["num_games"]
        self._last_refit_day = self._last_day

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
        self._ratings_stale = False
        self._fitted = True
        self._current_day = self._last_day

    def __repr__(self):
        return (
            f"KickScoreRating(var_c={self.config.var_constant}, "
            f"var_m={self.config.var_matern}, "
            f"lscale={self.config.lscale}, "
            f"players={len(self._players)})"
        )


# Per-appearance arrays: name -> trailing shape. `ts/ms/vs/ns/xs` are the EP
# quantities; the rest are Kalman work arrays, fully overwritten on every
# kalman_forward_backward call (so they need capacity, not initialisation).
_FIELDS = (("ts", ()), ("ms", ()), ("vs", ()), ("ns", ()), ("xs", ()),
           ("m_p", (3,)), ("P_p", (3, 3)), ("m_f", (3,)), ("P_f", (3, 3)),
           ("m_s", (3,)), ("P_s", (3, 3)))


class _PlayerState:
    """Per-player EP/Kalman state.

    Appearances are appended one match at a time over a player's whole career,
    so the arrays are held in power-of-two buffers and exposed as slice views.
    Re-concatenating on every append instead would copy the full history each
    time — O(n^2) per player, which across a tour's history is hundreds of GB
    of memcpy. Views are cached and rebuilt only on growth: the EP inner loop
    reads them millions of times, so recomputing a view per access would cost
    more than the copying did.
    """

    def __init__(self):
        self.ts_pending: List[float] = []
        self._n = 0     # live appearances
        self._cap = 0   # buffer capacity
        self._bufs = {
            name: np.zeros((0,) + shape, dtype=np.float64)
            for name, shape in _FIELDS
        }
        self._refresh_views()

    def _refresh_views(self) -> None:
        """Point the public attributes at the live prefix of each buffer.

        Slices of a C-contiguous buffer are themselves contiguous, so numba
        accepts these directly and in-place writes land in the buffer.
        """
        for name, _ in _FIELDS:
            setattr(self, name, self._bufs[name][:self._n])

    def _grow(self, need: int) -> None:
        if need <= self._cap:
            return
        new_cap = max(8, self._cap)
        while new_cap < need:
            new_cap *= 2
        for name, _ in _FIELDS:
            buf = self._bufs[name]
            grown = np.zeros((new_cap,) + buf.shape[1:], dtype=np.float64)
            grown[:self._n] = buf[:self._n]
            self._bufs[name] = grown
        self._cap = new_cap

    def add_observation(self, t: float) -> int:
        """Register an observation time, return its index."""
        idx = self._n + len(self.ts_pending)
        self.ts_pending.append(t)
        return idx

    def allocate(self, var_c: float, var_m: float, lambda_: float) -> None:
        """Extend arrays for pending observations."""
        n_new = len(self.ts_pending)
        if n_new == 0:
            return

        n_old = self._n
        n_total = n_old + n_new
        self._grow(n_total)

        self._bufs["ts"][n_old:n_total] = self.ts_pending
        self._bufs["ms"][n_old:n_total] = 0.0
        self._bufs["vs"][n_old:n_total] = var_c + var_m   # prior marginal
        self._bufs["ns"][n_old:n_total] = 0.0
        self._bufs["xs"][n_old:n_total] = 0.0
        # Kalman work arrays are left zeroed — run_kalman rewrites every entry.

        self._n = n_total
        self._refresh_views()
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
        # Copy, so the serialised state carries only live appearances and not
        # whatever spare capacity the buffers happen to hold.
        return {name: np.asarray(getattr(self, name)).copy()
                for name, _ in _FIELDS}

    @classmethod
    def from_dict(cls, d: dict) -> "_PlayerState":
        ps = cls()
        n = len(d["ts"])
        ps._grow(n)
        ps._n = n
        for name, _ in _FIELDS:
            ps._bufs[name][:n] = d[name]
        ps._refresh_views()
        return ps
