"""
Numba-accelerated core functions for Weighted Elo (WElo) rating system.

Identical to Elo except the update rule includes a per-game weight:
    delta = k_factor * weight * (score - expected)

Design principles (same as Elo):
1. All hot-path functions compiled with @njit(cache=True)
2. Process ALL days in a single Numba call (no Python iteration)
3. Use fastmath=True where precision allows
4. Contiguous arrays throughout
5. Parallel execution for predictions
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, inline="always")
def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


@njit(cache=True, fastmath=True)
def update_ratings_weighted_sequential(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    ratings: np.ndarray,
    k_factor: float,
    scale: float,
) -> None:
    """
    Update WElo ratings for a batch of games with per-game weights.

    Games MUST be processed sequentially within a batch to maintain
    correctness when players appear in multiple games.

    Update rule: delta = k_factor * weight * (score - expected)

    Modifies ratings array in-place for zero allocation overhead.
    """
    n_games = len(player1)
    log10_scale = np.log(10.0) / scale

    for i in range(n_games):
        p1 = player1[i]
        p2 = player2[i]
        score = scores[i]
        w = weights[i]

        r1 = ratings[p1]
        r2 = ratings[p2]

        e1 = _sigmoid((r1 - r2) * log10_scale)
        delta = k_factor * w * (score - e1)

        ratings[p1] = r1 + delta
        ratings[p2] = r2 - delta


@njit(cache=True, fastmath=True)
def fit_all_days_weighted(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    day_offsets: np.ndarray,
    ratings: np.ndarray,
    k_factor: float,
    scale: float,
) -> None:
    """
    Fit WElo ratings for ALL days in a single Numba call.

    Identical to Elo's fit_all_days but with per-game weights.
    Games within each day are processed sequentially.

    Args:
        player1: All player1 IDs (sorted by day)
        player2: All player2 IDs (sorted by day)
        scores: All scores (sorted by day)
        weights: Per-game weights (sorted by day, same length as player1)
        day_offsets: Start index for each day (length = num_days + 1)
        ratings: Ratings array to update in-place
        k_factor: K-factor
        scale: Elo scale
    """
    n_days = len(day_offsets) - 1
    log10_scale = np.log(10.0) / scale

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            score = scores[i]
            w = weights[i]

            r1 = ratings[p1]
            r2 = ratings[p2]

            e1 = _sigmoid((r1 - r2) * log10_scale)
            delta = k_factor * w * (score - e1)

            ratings[p1] = r1 + delta
            ratings[p2] = r2 - delta


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Predict win probabilities for a batch of matchups (fully parallel).

    Prediction is identical to standard Elo - weights only affect updates.
    """
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)
    log10_scale = np.log(10.0) / scale

    for i in prange(n):
        r1 = ratings[player1[i]]
        r2 = ratings[player2[i]]
        proba[i] = _sigmoid((r1 - r2) * log10_scale)

    return proba


@njit(cache=True, fastmath=True)
def predict_single(
    p1_rating: float,
    p2_rating: float,
    scale: float,
) -> float:
    """Predict win probability for a single matchup."""
    return _sigmoid((p1_rating - p2_rating) * np.log(10.0) / scale)
