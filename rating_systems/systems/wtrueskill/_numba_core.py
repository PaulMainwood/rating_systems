"""
Numba-accelerated core functions for Weighted TrueSkill rating system.

Weighted TrueSkill extends TrueSkill by allowing per-game weights w that
scale the mean shift and uncertainty reduction. A game with higher weight
has more influence on the rating update.

With w = 1.0 for all games, this is identical to standard TrueSkill.
"""

import math
import numpy as np
from numba import njit

# Reuse utility functions from base TrueSkill
from ..trueskill._numba_core import (
    _pdf,
    _cdf,
    _v_win,
    _w_win,
    _v_lose,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
    compute_conservative_rating,
)


@njit(cache=True, fastmath=True)
def update_single_game_weighted(
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    score: float,
    weight: float,
    beta: float,
    min_sigma: float,
) -> tuple:
    """
    Update ratings for a single 1v1 game with per-game weight.

    Weight scales both the mean shift and uncertainty reduction:
    - new_mu    = mu + weight * (sigma^2 / c) * v
    - new_sigma = sigma * sqrt(max(1 - weight * (sigma^2/c^2) * w, min_bound))

    With weight=1.0, this is identical to standard TrueSkill.

    Args:
        mu1, sigma1: Player 1's skill mean and std dev
        mu2, sigma2: Player 2's skill mean and std dev
        score: 1.0 if player 1 won, 0.0 if player 2 won, 0.5 for draw
        weight: Per-game weight (1.0 = standard)
        beta: Performance variability parameter
        min_sigma: Minimum sigma to prevent overconfidence

    Returns:
        (new_mu1, new_sigma1, new_mu2, new_sigma2)
    """
    # Total uncertainty in the game outcome (unchanged by weight)
    c_squared = 2.0 * beta * beta + sigma1 * sigma1 + sigma2 * sigma2
    c = math.sqrt(c_squared)

    # Normalized skill difference
    t = (mu1 - mu2) / c

    # Pre-compute sigma^2 / c terms
    sigma1_sq_over_c = sigma1 * sigma1 / c
    sigma2_sq_over_c = sigma2 * sigma2 / c
    sigma1_sq_over_c_sq = sigma1 * sigma1 / c_squared
    sigma2_sq_over_c_sq = sigma2 * sigma2 / c_squared

    if score > 0.5:
        # Player 1 won
        v = _v_win(t)
        w = _w_win(t, v)

        new_mu1 = mu1 + weight * sigma1_sq_over_c * v
        new_mu2 = mu2 - weight * sigma2_sq_over_c * v

        new_sigma1_sq = sigma1 * sigma1 * max(1.0 - weight * sigma1_sq_over_c_sq * w,
                                                min_sigma * min_sigma / (sigma1 * sigma1))
        new_sigma2_sq = sigma2 * sigma2 * max(1.0 - weight * sigma2_sq_over_c_sq * w,
                                                min_sigma * min_sigma / (sigma2 * sigma2))

    elif score < 0.5:
        # Player 2 won
        v = _v_win(-t)
        w = _w_win(-t, v)

        new_mu1 = mu1 - weight * sigma1_sq_over_c * v
        new_mu2 = mu2 + weight * sigma2_sq_over_c * v

        new_sigma1_sq = sigma1 * sigma1 * max(1.0 - weight * sigma1_sq_over_c_sq * w,
                                                min_sigma * min_sigma / (sigma1 * sigma1))
        new_sigma2_sq = sigma2 * sigma2 * max(1.0 - weight * sigma2_sq_over_c_sq * w,
                                                min_sigma * min_sigma / (sigma2 * sigma2))

    else:
        # Draw
        v = _pdf(t) / (_cdf(t) * _cdf(-t) + 1e-10)
        w = v * v

        new_mu1 = mu1 - weight * sigma1_sq_over_c * v * t / (abs(t) + 0.1)
        new_mu2 = mu2 + weight * sigma2_sq_over_c * v * t / (abs(t) + 0.1)

        new_sigma1_sq = sigma1 * sigma1 * max(1.0 - weight * sigma1_sq_over_c_sq * w * 0.5,
                                                min_sigma * min_sigma / (sigma1 * sigma1))
        new_sigma2_sq = sigma2 * sigma2 * max(1.0 - weight * sigma2_sq_over_c_sq * w * 0.5,
                                                min_sigma * min_sigma / (sigma2 * sigma2))

    new_sigma1 = math.sqrt(max(new_sigma1_sq, min_sigma * min_sigma))
    new_sigma2 = math.sqrt(max(new_sigma2_sq, min_sigma * min_sigma))

    return new_mu1, new_sigma1, new_mu2, new_sigma2


@njit(cache=True, fastmath=True)
def update_ratings_sequential_weighted(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    beta: float,
    min_sigma: float,
) -> None:
    """
    Update TrueSkill ratings for a batch of games with per-game weights.

    Games MUST be processed sequentially to maintain correctness when
    players appear in multiple games.

    Modifies mu and sigma arrays in-place.
    """
    n_games = len(player1)

    for i in range(n_games):
        p1 = player1[i]
        p2 = player2[i]

        new_mu1, new_sigma1, new_mu2, new_sigma2 = update_single_game_weighted(
            mu[p1], sigma[p1],
            mu[p2], sigma[p2],
            scores[i],
            weights[i],
            beta,
            min_sigma,
        )

        mu[p1] = new_mu1
        sigma[p1] = new_sigma1
        mu[p2] = new_mu2
        sigma[p2] = new_sigma2


@njit(cache=True, fastmath=True)
def fit_all_days_weighted(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    day_offsets: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    beta: float,
    min_sigma: float,
) -> None:
    """
    Fit Weighted TrueSkill ratings for ALL days in a single Numba call.

    Per-game weights scale the mean shift and uncertainty reduction.
    Games within each day are processed sequentially.

    Args:
        player1: All player1 IDs (sorted by day)
        player2: All player2 IDs (sorted by day)
        scores: All scores (sorted by day)
        weights: Per-game weights (same length as player1)
        day_offsets: Start index for each day (length = num_days + 1)
        mu: Mean ratings array to update in-place
        sigma: Uncertainty array to update in-place
        beta: Performance variability parameter
        min_sigma: Minimum sigma to prevent overconfidence
    """
    n_days = len(day_offsets) - 1

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]

            new_mu1, new_sigma1, new_mu2, new_sigma2 = update_single_game_weighted(
                mu[p1], sigma[p1],
                mu[p2], sigma[p2],
                scores[i],
                weights[i],
                beta,
                min_sigma,
            )

            mu[p1] = new_mu1
            sigma[p1] = new_sigma1
            mu[p2] = new_mu2
            sigma[p2] = new_sigma2
