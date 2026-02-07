"""
Numba-accelerated core functions for Weighted Yuksel rating system.

Weighted Yuksel extends Yuksel by allowing per-game weights w that scale
the forces and curvature contribution. Weight scales the outcome delta
and second derivative, making decisive wins more informative.

With w = 1.0 for all games, this is identical to standard Yuksel.
"""

import numpy as np
from numba import njit, prange

# Reuse constants and utility functions from Yuksel
from ..yuksel._numba_core import (
    Q,
    Q2,
    Q2_3_OVER_PI2,
    _sigmoid,
    _g,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
    compute_phi,
)


@njit(cache=True, fastmath=True)
def update_single_game_weighted(
    p1: int,
    p2: int,
    score: float,
    w: float,
    ratings: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    D: np.ndarray,
    delta_r_max: float,
    alpha: float,
    scaling_factor: float,
) -> None:
    """
    Update ratings for a single game using the Weighted Yuksel method.

    Like standard Yuksel, but weight w scales the forces and curvature:
    - F_1 = w * g_2 * (score - prob)
    - F_2 = -w * g_1 * (score - prob)
    - second_deriv = w * Q * prob * (1-prob)

    Direction D update, adaptive step, and Welford variance tracking
    are unchanged (structural, not data).
    """
    r1 = ratings[p1]
    r2 = ratings[p2]

    # Win probability
    prob = _sigmoid(Q * (r1 - r2))

    # Uncertainty estimates
    phi_1 = np.sqrt(V[p1] / max(W[p1], 1e-10))
    phi_2 = np.sqrt(V[p2] / max(W[p2], 1e-10))

    # g functions
    g_1 = _g(phi_1)
    g_2 = _g(phi_2)
    g_alpha_1 = _g(alpha * phi_1)
    g_alpha_2 = _g(alpha * phi_2)

    # Weighted forces
    outcome_delta = score - prob
    F_1 = w * g_2 * outcome_delta
    F_2 = -w * g_1 * outcome_delta

    # Weighted curvature
    second_deriv = w * Q * prob * (1.0 - prob)

    # Direction update with momentum
    D_1 = D[p1]
    D_1 = (g_2 * second_deriv) + (g_alpha_1 * D_1)

    D_2 = D[p2]
    D_2 = (g_1 * second_deriv) + (g_alpha_2 * D_2)

    # Adaptive step size
    denom = D_1 * D_1 + D_2 * D_2
    if denom > 1e-10:
        delta_r = (D_1 * F_1 - D_2 * F_2) / denom
    else:
        delta_r = 0.0

    # Clamp
    if delta_r > delta_r_max:
        delta_r = delta_r_max
    elif delta_r < -delta_r_max:
        delta_r = -delta_r_max

    # Adjust D
    if np.abs(delta_r) > 1e-10:
        D_1 = F_1 / delta_r
        D_2 = -F_2 / delta_r

    D[p1] = D_1
    D[p2] = D_2

    # Apply rating updates (zero-sum)
    scaled_update = scaling_factor * delta_r
    new_r_1 = r1 + scaled_update
    new_r_2 = r2 - scaled_update

    ratings[p1] = new_r_1
    ratings[p2] = new_r_2

    # Welford variance tracking (unchanged by weight)
    omega_1 = g_alpha_1
    W[p1] = omega_1 * W[p1] + 1.0
    delta_R_1 = new_r_1 - R[p1]
    R[p1] = R[p1] + delta_R_1 / W[p1]
    V[p1] = omega_1 * V[p1] + delta_R_1 * (new_r_1 - R[p1])

    omega_2 = g_alpha_2
    W[p2] = omega_2 * W[p2] + 1.0
    delta_R_2 = new_r_2 - R[p2]
    R[p2] = R[p2] + delta_R_2 / W[p2]
    V[p2] = omega_2 * V[p2] + delta_R_2 * (new_r_2 - R[p2])


@njit(cache=True, fastmath=True)
def update_ratings_sequential_weighted(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    ratings: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    D: np.ndarray,
    delta_r_max: float,
    alpha: float,
    scaling_factor: float,
) -> None:
    """
    Update ratings for a batch of games with per-game weights (sequential).

    Games MUST be processed sequentially to maintain correctness when
    players appear in multiple games within the same batch.
    """
    n_games = len(player1)

    for i in range(n_games):
        update_single_game_weighted(
            player1[i],
            player2[i],
            scores[i],
            weights[i],
            ratings,
            R,
            W,
            V,
            D,
            delta_r_max,
            alpha,
            scaling_factor,
        )


@njit(cache=True, fastmath=True)
def fit_all_days_weighted(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    day_offsets: np.ndarray,
    ratings: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    D: np.ndarray,
    delta_r_max: float,
    alpha: float,
    scaling_factor: float,
) -> None:
    """
    Fit Weighted Yuksel ratings for ALL days in a single Numba call.

    Per-game weights scale the forces and curvature contribution.
    Games within each day are processed sequentially.
    """
    n_days = len(day_offsets) - 1

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        for i in range(start, end):
            update_single_game_weighted(
                player1[i],
                player2[i],
                scores[i],
                weights[i],
                ratings,
                R,
                W,
                V,
                D,
                delta_r_max,
                alpha,
                scaling_factor,
            )
