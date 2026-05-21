"""Numba-accelerated core for Surface-Factor Whole History Rating.

Each player has FOUR trajectories instead of one:

    base_i(t)            ~ Wiener(w2_base)
    offset_i_s(t)        ~ Wiener(w2_off[s])     for s in {Hard, Clay, Grass}

On surface ``s`` at time ``t`` the effective rating is

    r_eff_i_s(t) = w_s * base_i(t) + offset_i_s(t).

The joint MAP over a player's four trajectories is found by **coordinate
descent**: each outer iteration updates each trajectory in turn using the
existing tridiagonal Newton step from plain WHR, with surface-aware
gradient and Hessian contributions.

Why coordinate descent rather than a full block-tridiagonal solver?
- The negative log posterior is jointly convex, so coordinate descent is
  guaranteed to converge to the global optimum.
- Each pass reuses the proven tridiagonal solver from ``whr/_numba_core``.
- 4× more work per outer iteration than plain WHR, but the inner steps are
  identical in form and the existing JIT-compiled solver is reused.

Storage convention:
- pd_r_base[total_pd]        — base trajectory, one value per player-day.
- pd_r_off[total_pd, 3]      — offsets per surface (Hard=0, Clay=1, Grass=2).
- pd_game_surface[2*n_games] — surface for each (pd, game) tuple, parallel
                                to pd_game_opp_pd / pd_game_score.
"""

import math

import numpy as np
from numba import njit, prange

from ..whr._numba_core import (
    LN10_400,
    sigmoid,
    solve_tridiagonal,
)


N_SURFACES = 3


@njit(cache=True, fastmath=True)
def update_base_trajectory(
    player_id: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r_base: np.ndarray,
    pd_r_off: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_surface: np.ndarray,
    pd_game_weight: np.ndarray,
    w_surfaces: np.ndarray,
    w2_base: float,
) -> float:
    """Newton step on this player's base trajectory.

    Holds all three offset trajectories (own + opponents) fixed; the
    gradient of the per-game log-likelihood with respect to base picks up
    a ``w_s`` factor from the chain rule and the Hessian picks up ``w_s**2``.
    The per-game ``pd_game_weight`` scales the Fisher information contribution
    exactly as in WWHR; unweighted runs pass all ones.
    """
    pd_start = player_offsets[player_id]
    pd_end = player_offsets[player_id + 1]
    n = pd_end - pd_start
    if n == 0:
        return 0.0

    gradient = np.zeros(n, dtype=np.float64)
    hess_diag = np.zeros(n, dtype=np.float64)
    hess_off = np.zeros(max(1, n - 1), dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        day_diff = max(1, pd_days[pd_start + i + 1] - pd_days[pd_start + i])
        hess_off[i] = 1.0 / (w2_base * day_diff)

    for i in range(n):
        pd_idx = pd_start + i
        base_i = pd_r_base[pd_idx]

        game_start = pd_game_offsets[pd_idx]
        game_end = pd_game_offsets[pd_idx + 1]
        for g in range(game_start, game_end):
            s = pd_game_surface[g]
            w_s = w_surfaces[s]
            opp_pd = pd_game_opp_pd[g]
            score = pd_game_score[g]
            gw = pd_game_weight[g]

            r_eff_i = w_s * base_i + pd_r_off[pd_idx, s]
            r_eff_o = w_s * pd_r_base[opp_pd] + pd_r_off[opp_pd, s]
            p_win = sigmoid(r_eff_i - r_eff_o)

            gradient[i] += gw * w_s * (score - p_win)
            hess_diag[i] -= gw * w_s * w_s * p_win * (1.0 - p_win)

        # Anchor: virtual win+loss vs an opponent at r=0 on this player's
        # first pd. Without this the joint base/offset optimum is identifiable
        # only up to a constant.
        if i == 0:
            p_v = sigmoid(base_i)
            gradient[i] += (1.0 - p_v)
            gradient[i] += -p_v
            hess_diag[i] -= 2.0 * p_v * (1.0 - p_v)

        if i > 0:
            inv_sigma2 = hess_off[i - 1]
            r_prev = pd_r_base[pd_start + i - 1]
            gradient[i] -= (base_i - r_prev) * inv_sigma2
            hess_diag[i] -= inv_sigma2
        if i < n - 1:
            inv_sigma2 = hess_off[i]
            r_next = pd_r_base[pd_start + i + 1]
            gradient[i] -= (base_i - r_next) * inv_sigma2
            hess_diag[i] -= inv_sigma2

    for i in range(n):
        hess_diag[i] -= 0.001
        if hess_diag[i] > -1e-10:
            hess_diag[i] = -1e-10

    solve_tridiagonal(hess_diag, hess_off, gradient, n, delta)

    max_change = 0.0
    for i in range(n):
        pd_r_base[pd_start + i] += delta[i]
        if abs(delta[i]) > max_change:
            max_change = abs(delta[i])
    return max_change


@njit(cache=True, fastmath=True)
def update_offset_trajectory(
    player_id: int,
    surface: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r_base: np.ndarray,
    pd_r_off: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_surface: np.ndarray,
    pd_game_weight: np.ndarray,
    w_surfaces: np.ndarray,
    w2_off_s: float,
    anchor: float,
) -> float:
    """Newton step on this player's offset trajectory for one surface.

    Likelihood contributes only at player-days that actually had a game on
    the target surface; at other player-days only the Wiener prior couples
    consecutive values. ``anchor`` is a soft Gaussian prior toward zero
    (precision = ``anchor``) to prevent unidentifiable drift in the offset.
    """
    pd_start = player_offsets[player_id]
    pd_end = player_offsets[player_id + 1]
    n = pd_end - pd_start
    if n == 0:
        return 0.0

    w_s = w_surfaces[surface]
    gradient = np.zeros(n, dtype=np.float64)
    hess_diag = np.zeros(n, dtype=np.float64)
    hess_off = np.zeros(max(1, n - 1), dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        day_diff = max(1, pd_days[pd_start + i + 1] - pd_days[pd_start + i])
        hess_off[i] = 1.0 / (w2_off_s * day_diff)

    for i in range(n):
        pd_idx = pd_start + i
        off_i = pd_r_off[pd_idx, surface]
        base_i = pd_r_base[pd_idx]

        game_start = pd_game_offsets[pd_idx]
        game_end = pd_game_offsets[pd_idx + 1]
        for g in range(game_start, game_end):
            if pd_game_surface[g] != surface:
                continue
            opp_pd = pd_game_opp_pd[g]
            score = pd_game_score[g]
            gw = pd_game_weight[g]

            r_eff_i = w_s * base_i + off_i
            r_eff_o = w_s * pd_r_base[opp_pd] + pd_r_off[opp_pd, surface]
            p_win = sigmoid(r_eff_i - r_eff_o)

            gradient[i] += gw * (score - p_win)
            hess_diag[i] -= gw * p_win * (1.0 - p_win)

        if i > 0:
            inv_sigma2 = hess_off[i - 1]
            r_prev = pd_r_off[pd_start + i - 1, surface]
            gradient[i] -= (off_i - r_prev) * inv_sigma2
            hess_diag[i] -= inv_sigma2
        if i < n - 1:
            inv_sigma2 = hess_off[i]
            r_next = pd_r_off[pd_start + i + 1, surface]
            gradient[i] -= (off_i - r_next) * inv_sigma2
            hess_diag[i] -= inv_sigma2

        # Soft anchor toward zero: prevents the offset from drifting
        # unboundedly when w_surfaces is small or data is sparse.
        gradient[i] -= anchor * off_i
        hess_diag[i] -= anchor

    for i in range(n):
        hess_diag[i] -= 1e-9
        if hess_diag[i] > -1e-10:
            hess_diag[i] = -1e-10

    solve_tridiagonal(hess_diag, hess_off, gradient, n, delta)

    max_change = 0.0
    for i in range(n):
        pd_r_off[pd_start + i, surface] += delta[i]
        if abs(delta[i]) > max_change:
            max_change = abs(delta[i])
    return max_change


@njit(cache=True, fastmath=True)
def run_outer_iteration(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r_base: np.ndarray,
    pd_r_off: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_surface: np.ndarray,
    pd_game_weight: np.ndarray,
    w_surfaces: np.ndarray,
    w2_base: float,
    w2_off: np.ndarray,
    offset_anchor: float,
) -> float:
    """One outer pass: base trajectories, then each surface offset in turn."""
    max_change = 0.0

    for player_id in range(num_players):
        ch = update_base_trajectory(
            player_id, player_offsets, pd_days,
            pd_r_base, pd_r_off,
            pd_game_offsets, pd_game_opp_pd, pd_game_score, pd_game_surface,
            pd_game_weight, w_surfaces, w2_base,
        )
        if ch > max_change:
            max_change = ch

    for s in range(N_SURFACES):
        w2_off_s = w2_off[s]
        for player_id in range(num_players):
            ch = update_offset_trajectory(
                player_id, s, player_offsets, pd_days,
                pd_r_base, pd_r_off,
                pd_game_offsets, pd_game_opp_pd, pd_game_score, pd_game_surface,
                pd_game_weight, w_surfaces, w2_off_s, offset_anchor,
            )
            if ch > max_change:
                max_change = ch

    return max_change


@njit(cache=True, fastmath=True)
def run_all_iterations(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r_base: np.ndarray,
    pd_r_off: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_surface: np.ndarray,
    pd_game_weight: np.ndarray,
    w_surfaces: np.ndarray,
    w2_base: float,
    w2_off: np.ndarray,
    max_iterations: int,
    convergence_threshold: float,
    offset_anchor: float,
) -> int:
    """Run outer iterations until max change drops below the threshold."""
    for it in range(max_iterations):
        ch = run_outer_iteration(
            num_players, player_offsets, pd_days,
            pd_r_base, pd_r_off,
            pd_game_offsets, pd_game_opp_pd, pd_game_score, pd_game_surface,
            pd_game_weight, w_surfaces, w2_base, w2_off, offset_anchor,
        )
        if ch < convergence_threshold:
            return it + 1
    return max_iterations


@njit(cache=True, fastmath=True)
def extract_current_ratings(
    num_players: int,
    player_offsets: np.ndarray,
    pd_r_base: np.ndarray,
    pd_r_off: np.ndarray,
    out_base: np.ndarray,
    out_off: np.ndarray,
    initial_rating_logged: float,
) -> None:
    """Pull the most-recent base + offset values into per-player arrays."""
    for p in range(num_players):
        start = player_offsets[p]
        end = player_offsets[p + 1]
        if end > start:
            last = end - 1
            out_base[p] = pd_r_base[last]
            for s in range(N_SURFACES):
                out_off[p, s] = pd_r_off[last, s]
        else:
            out_base[p] = initial_rating_logged
            for s in range(N_SURFACES):
                out_off[p, s] = 0.0


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch_factor(
    p1: np.ndarray,
    p2: np.ndarray,
    surfaces: np.ndarray,
    base: np.ndarray,
    off: np.ndarray,
    w_surfaces: np.ndarray,
) -> np.ndarray:
    """Batched P(p1 beats p2) on each game's surface, given current ratings."""
    n = p1.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        s = surfaces[i]
        w_s = w_surfaces[s]
        a = p1[i]
        b = p2[i]
        r_a = w_s * base[a] + off[a, s]
        r_b = w_s * base[b] + off[b, s]
        out[i] = sigmoid(r_a - r_b)
    return out


@njit(cache=True, fastmath=True)
def predict_single_factor(
    base_i: float, off_i_s: float,
    base_j: float, off_j_s: float,
    w_s: float,
) -> float:
    """Single-match win probability on a given surface."""
    return sigmoid((w_s * base_i + off_i_s) - (w_s * base_j + off_j_s))
