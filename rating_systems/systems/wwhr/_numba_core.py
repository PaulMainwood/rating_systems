"""
Numba-accelerated core functions for WWHR (Weighted Whole History Rating).

Imports shared functions from WHR and adds weighted variants where per-game
weights scale the gradient and Hessian contributions. Virtual game priors
and Wiener process priors are NOT weighted (they're structural, not data).

Native handicap support: per-game handicaps h_g shift the Bradley-Terry
probability in the gradient/Hessian:
    p_win = sigmoid(r_i - r_j + h_g)
where h_g > 0 favours the player whose perspective we're computing from.
"""

import math
import numpy as np
from numba import njit, prange

# Reuse shared functions from WHR
from ..whr._numba_core import (
    LN10_400,
    build_player_day_indices,
    sigmoid,
    solve_tridiagonal,
    extract_current_ratings,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
    get_rating_at_day,
    predict_proba_at_day,
    warm_start_ratings,
    anderson_mix,
    solve_small_system,
)


@njit(cache=True)
def fill_game_arrays_weighted(
    n_games: int,
    pd1_indices: np.ndarray,
    pd2_indices: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
) -> None:
    """
    Fill game arrays with opponent references, scores, and weights.

    Each game appears twice (once per player). Both perspectives get the
    same weight.
    """
    pd_game_pos = pd_game_offsets[:-1].copy()

    for i in range(n_games):
        pd1 = pd1_indices[i]
        pd2 = pd2_indices[i]
        score = scores[i]
        w = weights[i]

        # Player 1's perspective
        pos1 = pd_game_pos[pd1]
        pd_game_opp_pd[pos1] = pd2
        pd_game_score[pos1] = score
        pd_game_weights[pos1] = w
        pd_game_pos[pd1] += 1

        # Player 2's perspective
        pos2 = pd_game_pos[pd2]
        pd_game_opp_pd[pos2] = pd1
        pd_game_score[pos2] = 1.0 - score
        pd_game_weights[pos2] = w
        pd_game_pos[pd2] += 1


@njit(cache=True)
def fill_game_arrays_weighted_h(
    n_games: int,
    pd1_indices: np.ndarray,
    pd2_indices: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    handicaps: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    pd_game_handicaps: np.ndarray,
) -> None:
    """
    Fill game arrays with opponent references, scores, weights, and handicaps.

    Handicaps are from player1's perspective.  When stored from player2's
    perspective the handicap is negated.
    """
    pd_game_pos = pd_game_offsets[:-1].copy()

    for i in range(n_games):
        pd1 = pd1_indices[i]
        pd2 = pd2_indices[i]
        score = scores[i]
        w = weights[i]
        h = handicaps[i]

        # Player 1's perspective
        pos1 = pd_game_pos[pd1]
        pd_game_opp_pd[pos1] = pd2
        pd_game_score[pos1] = score
        pd_game_weights[pos1] = w
        pd_game_handicaps[pos1] = h
        pd_game_pos[pd1] += 1

        # Player 2's perspective (negate handicap)
        pos2 = pd_game_pos[pd2]
        pd_game_opp_pd[pos2] = pd1
        pd_game_score[pos2] = 1.0 - score
        pd_game_weights[pos2] = w
        pd_game_handicaps[pos2] = -h
        pd_game_pos[pd2] += 1


@njit(cache=True, fastmath=True)
def update_single_player_weighted_h(
    player_id: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    pd_game_handicaps: np.ndarray,
    w2_r: float,
) -> float:
    """
    Update a single player's ratings using weighted Newton-Raphson with handicaps.

    Game contributions include per-game handicap:
        p_win = sigmoid(r_i - opp_r + h_g)
        gradient[i] += w * (score - p_win)
        hess_diag[i] -= w * p_win * (1 - p_win)

    Returns the maximum absolute change in rating.
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
        sigma2 = w2_r * day_diff
        inv_sigma2 = 1.0 / sigma2
        hess_off[i] = inv_sigma2

    for i in range(n):
        pd_idx = pd_start + i
        r_i = pd_r[pd_idx]

        game_start = pd_game_offsets[pd_idx]
        game_end = pd_game_offsets[pd_idx + 1]

        for g in range(game_start, game_end):
            opp_pd = pd_game_opp_pd[g]
            score = pd_game_score[g]
            opp_r = pd_r[opp_pd]
            w = pd_game_weights[g]
            h = pd_game_handicaps[g]

            p_win = sigmoid(r_i - opp_r + h)

            gradient[i] += w * (score - p_win)
            hess_diag[i] -= w * p_win * (1.0 - p_win)

        if i == 0:
            p_virtual = sigmoid(r_i - 0.0)
            gradient[i] += (1.0 - p_virtual)
            gradient[i] += (0.0 - p_virtual)
            hess_diag[i] -= 2.0 * p_virtual * (1.0 - p_virtual)

        if i > 0:
            inv_sigma2 = hess_off[i - 1]
            r_prev = pd_r[pd_start + i - 1]
            gradient[i] -= (r_i - r_prev) * inv_sigma2
            hess_diag[i] -= inv_sigma2

        if i < n - 1:
            inv_sigma2 = hess_off[i]
            r_next = pd_r[pd_start + i + 1]
            gradient[i] -= (r_i - r_next) * inv_sigma2
            hess_diag[i] -= inv_sigma2

    for i in range(n):
        hess_diag[i] -= 0.001
        if hess_diag[i] > -1e-10:
            hess_diag[i] = -1e-10

    solve_tridiagonal(hess_diag, hess_off, gradient, n, delta)

    max_change = 0.0
    for i in range(n):
        pd_idx = pd_start + i
        change = delta[i]
        pd_r[pd_idx] += change
        if abs(change) > max_change:
            max_change = abs(change)

    return max_change


@njit(cache=True, fastmath=True)
def update_single_player_weighted_jacobi_h(
    player_id: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_r_read: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    pd_game_handicaps: np.ndarray,
    w2_r: float,
) -> float:
    """Jacobi variant of weighted Newton-Raphson with handicaps."""
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
        sigma2 = w2_r * day_diff
        inv_sigma2 = 1.0 / sigma2
        hess_off[i] = inv_sigma2

    for i in range(n):
        pd_idx = pd_start + i
        r_i = pd_r_read[pd_idx]

        game_start = pd_game_offsets[pd_idx]
        game_end = pd_game_offsets[pd_idx + 1]

        for g in range(game_start, game_end):
            opp_pd = pd_game_opp_pd[g]
            score = pd_game_score[g]
            opp_r = pd_r_read[opp_pd]
            w = pd_game_weights[g]
            h = pd_game_handicaps[g]

            p_win = sigmoid(r_i - opp_r + h)

            gradient[i] += w * (score - p_win)
            hess_diag[i] -= w * p_win * (1.0 - p_win)

        if i == 0:
            p_virtual = sigmoid(r_i - 0.0)
            gradient[i] += (1.0 - p_virtual)
            gradient[i] += (0.0 - p_virtual)
            hess_diag[i] -= 2.0 * p_virtual * (1.0 - p_virtual)

        if i > 0:
            inv_sigma2 = hess_off[i - 1]
            r_prev = pd_r_read[pd_start + i - 1]
            gradient[i] -= (r_i - r_prev) * inv_sigma2
            hess_diag[i] -= inv_sigma2

        if i < n - 1:
            inv_sigma2 = hess_off[i]
            r_next = pd_r_read[pd_start + i + 1]
            gradient[i] -= (r_i - r_next) * inv_sigma2
            hess_diag[i] -= inv_sigma2

    for i in range(n):
        hess_diag[i] -= 0.001
        if hess_diag[i] > -1e-10:
            hess_diag[i] = -1e-10

    solve_tridiagonal(hess_diag, hess_off, gradient, n, delta)

    max_change = 0.0
    for i in range(n):
        pd_idx = pd_start + i
        change = delta[i]
        pd_r[pd_idx] += change
        if abs(change) > max_change:
            max_change = abs(change)

    return max_change


@njit(cache=True, fastmath=True)
def compute_uncertainties_weighted_h(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_uncertainty: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    pd_game_handicaps: np.ndarray,
    w2_r: float,
) -> None:
    """Compute rating uncertainties from weighted Hessian with handicaps."""
    for player_id in range(num_players):
        pd_start = player_offsets[player_id]
        pd_end = player_offsets[player_id + 1]
        n = pd_end - pd_start

        if n == 0:
            continue

        sigma2 = np.empty(max(1, n - 1), dtype=np.float64)
        for i in range(n - 1):
            day_diff = max(1, pd_days[pd_start + i + 1] - pd_days[pd_start + i])
            sigma2[i] = w2_r * day_diff

        for i in range(n):
            pd_idx = pd_start + i
            r_i = pd_r[pd_idx]
            hess = 0.0

            game_start = pd_game_offsets[pd_idx]
            game_end = pd_game_offsets[pd_idx + 1]

            for g in range(game_start, game_end):
                opp_pd = pd_game_opp_pd[g]
                opp_r = pd_r[opp_pd]
                w = pd_game_weights[g]
                h = pd_game_handicaps[g]
                p_win = sigmoid(r_i - opp_r + h)
                hess -= w * p_win * (1.0 - p_win)

            if i == 0:
                p_virtual = sigmoid(r_i - 0.0)
                hess -= 2.0 * p_virtual * (1.0 - p_virtual)

            if i > 0:
                hess -= 1.0 / sigma2[i - 1]
            if i < n - 1:
                hess -= 1.0 / sigma2[i]

            hess -= 0.001

            if hess < -1e-10:
                var_r = -1.0 / hess
                pd_uncertainty[pd_idx] = math.sqrt(var_r) / LN10_400
            else:
                pd_uncertainty[pd_idx] = 350.0


@njit(cache=True, fastmath=True)
def run_iteration_weighted_h(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    pd_game_handicaps: np.ndarray,
    w2_r: float,
) -> float:
    """Run one weighted Newton-Raphson iteration with handicaps."""
    max_change = 0.0
    for player_id in range(num_players):
        change = update_single_player_weighted_h(
            player_id, player_offsets, pd_days, pd_r,
            pd_game_offsets, pd_game_opp_pd, pd_game_score,
            pd_game_weights, pd_game_handicaps, w2_r,
        )
        if change > max_change:
            max_change = change
    return max_change


@njit(cache=True, fastmath=True, parallel=True)
def run_iteration_jacobi_weighted_h(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_r_read: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    pd_game_handicaps: np.ndarray,
    w2_r: float,
) -> float:
    """Jacobi parallel Newton-Raphson iteration with handicaps."""
    pd_r_read[:] = pd_r[:]
    max_change = 0.0
    for player_id in prange(num_players):
        change = update_single_player_weighted_jacobi_h(
            player_id, player_offsets, pd_days, pd_r, pd_r_read,
            pd_game_offsets, pd_game_opp_pd, pd_game_score,
            pd_game_weights, pd_game_handicaps, w2_r,
        )
        max_change = max(max_change, change)
    return max_change


@njit(cache=True, fastmath=True)
def run_all_iterations_accelerated_weighted_h(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    pd_game_handicaps: np.ndarray,
    w2_r: float,
    max_iterations: int,
    convergence_threshold: float,
    anderson_window: int,
    use_active_set: bool,
    pd_to_player: np.ndarray,
    use_jacobi: bool = False,
) -> int:
    """
    Run weighted Newton-Raphson with handicaps, optional active-set, Anderson
    acceleration, and Jacobi parallelism.

    Returns the number of iterations performed.
    """
    total_pd = len(pd_r)
    active = np.ones(num_players, dtype=np.bool_)

    anderson_enabled = anderson_window > 0
    m = max(anderson_window, 1)
    if anderson_enabled:
        F_buf = np.zeros((m, total_pd), dtype=np.float64)
        G_buf = np.zeros((m, total_pd), dtype=np.float64)
    else:
        F_buf = np.zeros((1, 1), dtype=np.float64)
        G_buf = np.zeros((1, 1), dtype=np.float64)
    buf_count = 0
    regularization = 1e-10

    if use_jacobi:
        pd_r_read = np.empty(total_pd, dtype=np.float64)
    else:
        pd_r_read = np.empty(0, dtype=np.float64)

    for iteration in range(max_iterations):
        if anderson_enabled:
            x_before = pd_r.copy()

        if use_jacobi:
            max_change = run_iteration_jacobi_weighted_h(
                num_players, player_offsets, pd_days, pd_r, pd_r_read,
                pd_game_offsets, pd_game_opp_pd, pd_game_score,
                pd_game_weights, pd_game_handicaps, w2_r,
            )
        else:
            max_change = run_iteration_weighted_h(
                num_players, player_offsets, pd_days, pd_r,
                pd_game_offsets, pd_game_opp_pd, pd_game_score,
                pd_game_weights, pd_game_handicaps, w2_r,
            )

        if max_change < convergence_threshold:
            return iteration + 1

        if anderson_enabled:
            slot = buf_count % m
            for t in range(total_pd):
                F_buf[slot, t] = pd_r[t] - x_before[t]
                G_buf[slot, t] = pd_r[t]
            buf_count += 1

            if buf_count >= 2:
                anderson_mix(
                    pd_r, F_buf, G_buf, buf_count, m,
                    total_pd, regularization,
                )
                if use_active_set:
                    for pid in range(num_players):
                        active[pid] = True

    return max_iterations


@njit(cache=True, fastmath=True)
def update_single_player_weighted(
    player_id: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    w2_r: float,
) -> float:
    """
    Update a single player's ratings using weighted Newton-Raphson.

    Game contributions are scaled by per-game weight w:
        gradient[i] += w * (score - p_win)
        hess_diag[i] -= w * p_win * (1 - p_win)

    Virtual game prior and Wiener process prior are NOT weighted.

    Returns the maximum absolute change in rating.
    """
    pd_start = player_offsets[player_id]
    pd_end = player_offsets[player_id + 1]
    n = pd_end - pd_start

    if n == 0:
        return 0.0

    # Allocate working arrays
    gradient = np.zeros(n, dtype=np.float64)
    hess_diag = np.zeros(n, dtype=np.float64)
    hess_off = np.zeros(max(1, n - 1), dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)

    # Compute sigma² between consecutive days (Wiener process variance)
    for i in range(n - 1):
        day_diff = max(1, pd_days[pd_start + i + 1] - pd_days[pd_start + i])
        sigma2 = w2_r * day_diff
        inv_sigma2 = 1.0 / sigma2
        hess_off[i] = inv_sigma2

    # Build gradient and Hessian for each player-day
    for i in range(n):
        pd_idx = pd_start + i
        r_i = pd_r[pd_idx]

        # Game contributions (WEIGHTED)
        game_start = pd_game_offsets[pd_idx]
        game_end = pd_game_offsets[pd_idx + 1]

        for g in range(game_start, game_end):
            opp_pd = pd_game_opp_pd[g]
            score = pd_game_score[g]
            opp_r = pd_r[opp_pd]
            w = pd_game_weights[g]

            p_win = sigmoid(r_i - opp_r)

            gradient[i] += w * (score - p_win)
            hess_diag[i] -= w * p_win * (1.0 - p_win)

        # Virtual game prior on first day (NOT weighted)
        if i == 0:
            p_virtual = sigmoid(r_i - 0.0)
            gradient[i] += (1.0 - p_virtual)  # virtual win
            gradient[i] += (0.0 - p_virtual)  # virtual loss
            hess_diag[i] -= 2.0 * p_virtual * (1.0 - p_virtual)

        # Wiener process prior contributions (NOT weighted)
        if i > 0:
            inv_sigma2 = hess_off[i - 1]
            r_prev = pd_r[pd_start + i - 1]
            gradient[i] -= (r_i - r_prev) * inv_sigma2
            hess_diag[i] -= inv_sigma2

        if i < n - 1:
            inv_sigma2 = hess_off[i]
            r_next = pd_r[pd_start + i + 1]
            gradient[i] -= (r_i - r_next) * inv_sigma2
            hess_diag[i] -= inv_sigma2

    # Regularization
    for i in range(n):
        hess_diag[i] -= 0.001
        if hess_diag[i] > -1e-10:
            hess_diag[i] = -1e-10

    # Solve tridiagonal system: H * delta = -gradient
    solve_tridiagonal(hess_diag, hess_off, gradient, n, delta)

    # Apply updates and track max change
    max_change = 0.0
    for i in range(n):
        pd_idx = pd_start + i
        change = delta[i]
        pd_r[pd_idx] += change
        if abs(change) > max_change:
            max_change = abs(change)

    return max_change


@njit(cache=True, fastmath=True)
def update_single_player_weighted_jacobi(
    player_id: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_r_read: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    w2_r: float,
) -> float:
    """
    Update a single player's ratings using weighted Newton-Raphson (Jacobi variant).

    Reads from pd_r_read (snapshot), writes to pd_r (live).
    Safe for parallel execution since each player writes to disjoint indices.
    """
    pd_start = player_offsets[player_id]
    pd_end = player_offsets[player_id + 1]
    n = pd_end - pd_start

    if n == 0:
        return 0.0

    # Allocate working arrays
    gradient = np.zeros(n, dtype=np.float64)
    hess_diag = np.zeros(n, dtype=np.float64)
    hess_off = np.zeros(max(1, n - 1), dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)

    # Compute sigma² between consecutive days (Wiener process variance)
    for i in range(n - 1):
        day_diff = max(1, pd_days[pd_start + i + 1] - pd_days[pd_start + i])
        sigma2 = w2_r * day_diff
        inv_sigma2 = 1.0 / sigma2
        hess_off[i] = inv_sigma2

    # Build gradient and Hessian for each player-day
    for i in range(n):
        pd_idx = pd_start + i
        r_i = pd_r_read[pd_idx]  # Read from snapshot

        # Game contributions (WEIGHTED)
        game_start = pd_game_offsets[pd_idx]
        game_end = pd_game_offsets[pd_idx + 1]

        for g in range(game_start, game_end):
            opp_pd = pd_game_opp_pd[g]
            score = pd_game_score[g]
            opp_r = pd_r_read[opp_pd]  # Read from snapshot
            w = pd_game_weights[g]

            p_win = sigmoid(r_i - opp_r)

            gradient[i] += w * (score - p_win)
            hess_diag[i] -= w * p_win * (1.0 - p_win)

        # Virtual game prior on first day (NOT weighted)
        if i == 0:
            p_virtual = sigmoid(r_i - 0.0)
            gradient[i] += (1.0 - p_virtual)  # virtual win
            gradient[i] += (0.0 - p_virtual)  # virtual loss
            hess_diag[i] -= 2.0 * p_virtual * (1.0 - p_virtual)

        # Wiener process prior contributions (NOT weighted)
        if i > 0:
            inv_sigma2 = hess_off[i - 1]
            r_prev = pd_r_read[pd_start + i - 1]  # Read from snapshot
            gradient[i] -= (r_i - r_prev) * inv_sigma2
            hess_diag[i] -= inv_sigma2

        if i < n - 1:
            inv_sigma2 = hess_off[i]
            r_next = pd_r_read[pd_start + i + 1]  # Read from snapshot
            gradient[i] -= (r_i - r_next) * inv_sigma2
            hess_diag[i] -= inv_sigma2

    # Regularization
    for i in range(n):
        hess_diag[i] -= 0.001
        if hess_diag[i] > -1e-10:
            hess_diag[i] = -1e-10

    # Solve tridiagonal system: H * delta = -gradient
    solve_tridiagonal(hess_diag, hess_off, gradient, n, delta)

    # Apply updates and track max change
    max_change = 0.0
    for i in range(n):
        pd_idx = pd_start + i
        change = delta[i]
        pd_r[pd_idx] += change  # Write to live array
        if abs(change) > max_change:
            max_change = abs(change)

    return max_change


@njit(cache=True, fastmath=True)
def compute_uncertainties_weighted(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_uncertainty: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    w2_r: float,
) -> None:
    """
    Compute rating uncertainties from weighted Hessian diagonal.

    Uncertainty = sqrt(1/|H_ii|) converted to Elo scale.
    """
    for player_id in range(num_players):
        pd_start = player_offsets[player_id]
        pd_end = player_offsets[player_id + 1]
        n = pd_end - pd_start

        if n == 0:
            continue

        sigma2 = np.empty(max(1, n - 1), dtype=np.float64)
        for i in range(n - 1):
            day_diff = max(1, pd_days[pd_start + i + 1] - pd_days[pd_start + i])
            sigma2[i] = w2_r * day_diff

        for i in range(n):
            pd_idx = pd_start + i
            r_i = pd_r[pd_idx]
            hess = 0.0

            # Weighted game contributions
            game_start = pd_game_offsets[pd_idx]
            game_end = pd_game_offsets[pd_idx + 1]

            for g in range(game_start, game_end):
                opp_pd = pd_game_opp_pd[g]
                opp_r = pd_r[opp_pd]
                w = pd_game_weights[g]
                p_win = sigmoid(r_i - opp_r)
                hess -= w * p_win * (1.0 - p_win)

            # Virtual game prior (NOT weighted)
            if i == 0:
                p_virtual = sigmoid(r_i - 0.0)
                hess -= 2.0 * p_virtual * (1.0 - p_virtual)

            # Prior contributions (NOT weighted)
            if i > 0:
                hess -= 1.0 / sigma2[i - 1]
            if i < n - 1:
                hess -= 1.0 / sigma2[i]

            hess -= 0.001

            if hess < -1e-10:
                var_r = -1.0 / hess
                pd_uncertainty[pd_idx] = math.sqrt(var_r) / LN10_400
            else:
                pd_uncertainty[pd_idx] = 350.0


@njit(cache=True, fastmath=True)
def run_iteration_weighted(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    w2_r: float,
) -> float:
    """
    Run one weighted Newton-Raphson iteration for all players.

    Returns maximum rating change across all players.
    """
    max_change = 0.0

    for player_id in range(num_players):
        change = update_single_player_weighted(
            player_id,
            player_offsets,
            pd_days,
            pd_r,
            pd_game_offsets,
            pd_game_opp_pd,
            pd_game_score,
            pd_game_weights,
            w2_r,
        )
        if change > max_change:
            max_change = change

    return max_change


@njit(cache=True, fastmath=True, parallel=True)
def run_iteration_jacobi_weighted(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_r_read: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    w2_r: float,
) -> float:
    """
    Run one Jacobi weighted Newton-Raphson iteration for all players in parallel.

    Snapshots pd_r into pd_r_read, then updates all players reading from
    the snapshot. Safe for prange since each player writes to disjoint indices.
    """
    pd_r_read[:] = pd_r[:]  # Snapshot
    max_change = 0.0

    for player_id in prange(num_players):
        change = update_single_player_weighted_jacobi(
            player_id,
            player_offsets,
            pd_days,
            pd_r,
            pd_r_read,
            pd_game_offsets,
            pd_game_opp_pd,
            pd_game_score,
            pd_game_weights,
            w2_r,
        )
        max_change = max(max_change, change)

    return max_change


@njit(cache=True, fastmath=True)
def run_iteration_active_weighted(
    num_players: int,
    active: np.ndarray,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    w2_r: float,
    threshold: float,
    pd_to_player: np.ndarray,
) -> float:
    """
    Run one weighted Newton-Raphson iteration, only updating active players.

    Players whose update is below threshold are deactivated. When a player's
    update exceeds threshold, all its opponents are reactivated.

    Returns maximum rating change across all active players.
    """
    max_change = 0.0

    for player_id in range(num_players):
        if not active[player_id]:
            continue

        change = update_single_player_weighted(
            player_id,
            player_offsets,
            pd_days,
            pd_r,
            pd_game_offsets,
            pd_game_opp_pd,
            pd_game_score,
            pd_game_weights,
            w2_r,
        )

        if change < threshold:
            active[player_id] = False
        else:
            # Reactivate all opponents
            pd_start = player_offsets[player_id]
            pd_end = player_offsets[player_id + 1]
            for pd_idx in range(pd_start, pd_end):
                game_start = pd_game_offsets[pd_idx]
                game_end = pd_game_offsets[pd_idx + 1]
                for g in range(game_start, game_end):
                    opp_pd = pd_game_opp_pd[g]
                    opp_id = pd_to_player[opp_pd]
                    active[opp_id] = True

        if change > max_change:
            max_change = change

    return max_change


@njit(cache=True, fastmath=True, parallel=True)
def run_iteration_jacobi_active_weighted(
    num_players: int,
    active: np.ndarray,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_r_read: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    w2_r: float,
    threshold: float,
    pd_to_player: np.ndarray,
) -> float:
    """
    Run one Jacobi weighted Newton-Raphson iteration, only updating active players.

    Parallel version: snapshots pd_r, then updates active players via prange.
    Active array races are benign (only sets True for reactivation).
    """
    pd_r_read[:] = pd_r[:]  # Snapshot
    max_change = 0.0

    for player_id in prange(num_players):
        if not active[player_id]:
            continue

        change = update_single_player_weighted_jacobi(
            player_id,
            player_offsets,
            pd_days,
            pd_r,
            pd_r_read,
            pd_game_offsets,
            pd_game_opp_pd,
            pd_game_score,
            pd_game_weights,
            w2_r,
        )

        if change < threshold:
            active[player_id] = False
        else:
            # Reactivate all opponents (benign race: only sets True)
            pd_start = player_offsets[player_id]
            pd_end = player_offsets[player_id + 1]
            for pd_idx in range(pd_start, pd_end):
                game_start = pd_game_offsets[pd_idx]
                game_end = pd_game_offsets[pd_idx + 1]
                for g in range(game_start, game_end):
                    opp_pd = pd_game_opp_pd[g]
                    opp_id = pd_to_player[opp_pd]
                    active[opp_id] = True

        max_change = max(max_change, change)

    return max_change


@njit(cache=True, fastmath=True)
def run_all_iterations_weighted(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    w2_r: float,
    max_iterations: int,
    convergence_threshold: float,
) -> int:
    """
    Run weighted Newton-Raphson iterations until convergence or max iterations.

    Returns the number of iterations performed.
    """
    for iteration in range(max_iterations):
        max_change = run_iteration_weighted(
            num_players,
            player_offsets,
            pd_days,
            pd_r,
            pd_game_offsets,
            pd_game_opp_pd,
            pd_game_score,
            pd_game_weights,
            w2_r,
        )

        if max_change < convergence_threshold:
            return iteration + 1

    return max_iterations


@njit(cache=True, fastmath=True)
def run_all_iterations_accelerated_weighted(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    pd_game_weights: np.ndarray,
    w2_r: float,
    max_iterations: int,
    convergence_threshold: float,
    anderson_window: int,
    use_active_set: bool,
    pd_to_player: np.ndarray,
    use_jacobi: bool = False,
) -> int:
    """
    Run weighted Newton-Raphson with optional active-set, Anderson acceleration,
    and Jacobi parallelism.

    Parameters:
        anderson_window: Number of recent iterates for Anderson mixing (0 = disabled).
        use_active_set: If True, skip converged players each iteration.
        pd_to_player: Reverse mapping from player-day index to player ID.
        use_jacobi: If True, use Jacobi (parallel) iteration instead of
                     Gauss-Seidel (sequential).

    Returns the number of iterations performed.
    """
    total_pd = len(pd_r)

    # Active set initialisation
    active = np.ones(num_players, dtype=np.bool_)

    # Anderson buffer initialisation
    anderson_enabled = anderson_window > 0
    m = max(anderson_window, 1)
    if anderson_enabled:
        F_buf = np.zeros((m, total_pd), dtype=np.float64)
        G_buf = np.zeros((m, total_pd), dtype=np.float64)
    else:
        F_buf = np.zeros((1, 1), dtype=np.float64)
        G_buf = np.zeros((1, 1), dtype=np.float64)
    buf_count = 0
    regularization = 1e-10

    # Jacobi snapshot buffer (allocated once, reused each iteration)
    if use_jacobi:
        pd_r_read = np.empty(total_pd, dtype=np.float64)
    else:
        pd_r_read = np.empty(0, dtype=np.float64)

    for iteration in range(max_iterations):
        if anderson_enabled:
            x_before = pd_r.copy()

        # Run one sweep
        if use_jacobi:
            if use_active_set:
                max_change = run_iteration_jacobi_active_weighted(
                    num_players, active, player_offsets, pd_days, pd_r,
                    pd_r_read, pd_game_offsets, pd_game_opp_pd,
                    pd_game_score, pd_game_weights, w2_r,
                    convergence_threshold, pd_to_player,
                )
            else:
                max_change = run_iteration_jacobi_weighted(
                    num_players, player_offsets, pd_days, pd_r,
                    pd_r_read, pd_game_offsets, pd_game_opp_pd,
                    pd_game_score, pd_game_weights, w2_r,
                )
        else:
            if use_active_set:
                max_change = run_iteration_active_weighted(
                    num_players, active, player_offsets, pd_days, pd_r,
                    pd_game_offsets, pd_game_opp_pd, pd_game_score,
                    pd_game_weights, w2_r, convergence_threshold,
                    pd_to_player,
                )
            else:
                max_change = run_iteration_weighted(
                    num_players, player_offsets, pd_days, pd_r,
                    pd_game_offsets, pd_game_opp_pd, pd_game_score,
                    pd_game_weights, w2_r,
                )

        if max_change < convergence_threshold:
            return iteration + 1

        if anderson_enabled:
            slot = buf_count % m
            for t in range(total_pd):
                F_buf[slot, t] = pd_r[t] - x_before[t]
                G_buf[slot, t] = pd_r[t]
            buf_count += 1

            if buf_count >= 2:
                anderson_mix(
                    pd_r, F_buf, G_buf, buf_count, m,
                    total_pd, regularization,
                )
                if use_active_set:
                    for pid in range(num_players):
                        active[pid] = True

    return max_iterations
