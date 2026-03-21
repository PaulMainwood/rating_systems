"""Numba-accelerated kernels for weighted Disc decomposition.

Extends the Disc kernels with per-game weights and handicaps.
Weight scales the update magnitude; handicap shifts the skill difference.
"""

import math

import numpy as np
from numba import njit, prange

_LOG10 = math.log(10.0)


@njit(cache=True, fastmath=True, inline="always")
def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


@njit(cache=True, fastmath=True)
def predict_single(u1, u2, v1, v2, scale):
    c = _LOG10 / scale
    z = v1 * v2 * (u1 - u2) * c
    return _sigmoid(z)


@njit(cache=True, fastmath=True)
def predict_single_with_handicap(u1, u2, v1, v2, scale, handicap):
    c = _LOG10 / scale
    z = v1 * v2 * (u1 - u2 + handicap) * c
    return _sigmoid(z)


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(player1, player2, skills, consistency, scale,
                        handicaps):
    """Batch prediction with per-game handicaps, parallel."""
    n = len(player1)
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        p1, p2 = player1[i], player2[i]
        out[i] = predict_single_with_handicap(
            skills[p1], skills[p2],
            consistency[p1], consistency[p2],
            scale, handicaps[i],
        )
    return out


@njit(cache=True, fastmath=True)
def update_ratings_weighted_sequential(player1, player2, scores,
                                       skills, consistency,
                                       k_factor, consistency_lr, scale,
                                       v_floor, weights, handicaps):
    """Update skills and consistency with per-game weights and handicaps."""
    c = _LOG10 / scale
    n = len(player1)
    for i in range(n):
        p1, p2 = player1[i], player2[i]
        u1, u2 = skills[p1], skills[p2]
        v1, v2 = consistency[p1], consistency[p2]
        score = scores[i]
        w = weights[i]
        h = handicaps[i]

        delta = u1 - u2 + h
        z = v1 * v2 * delta * c
        p = _sigmoid(z)
        error = score - p

        # Skill updates (weighted, Elo convention)
        skills[p1] += k_factor * w * error * v1 * v2
        skills[p2] -= k_factor * w * error * v1 * v2

        # Consistency updates (weighted, full gradient)
        consistency[p1] += consistency_lr * w * error * v2 * delta * c
        consistency[p2] += consistency_lr * w * error * v1 * delta * c

        if consistency[p1] < v_floor:
            consistency[p1] = v_floor
        if consistency[p2] < v_floor:
            consistency[p2] = v_floor


@njit(cache=True, fastmath=True)
def fit_all_days_weighted(player1, player2, scores, day_offsets,
                          skills, consistency,
                          k_factor, consistency_lr, scale, v_floor,
                          weights, handicaps):
    """Process all days with weights and handicaps in a single numba call."""
    n_days = len(day_offsets) - 1
    for d in range(n_days):
        start = day_offsets[d]
        end = day_offsets[d + 1]
        if start >= end:
            continue
        update_ratings_weighted_sequential(
            player1[start:end], player2[start:end], scores[start:end],
            skills, consistency,
            k_factor, consistency_lr, scale, v_floor,
            weights[start:end], handicaps[start:end],
        )


@njit(cache=True, fastmath=True)
def walk_forward_predict_update(player1, player2, scores, day_indices,
                                day_offsets, skills, consistency,
                                k_factor, consistency_lr, scale, v_floor,
                                weights, handicaps, n_train_days):
    """Fused walk-forward: predict then update, day by day.

    Returns prediction array (NaN during training period).
    """
    n_games = len(player1)
    n_days = len(day_offsets) - 1
    preds = np.full(n_games, np.nan, dtype=np.float64)

    for d in range(n_days):
        start = day_offsets[d]
        end = day_offsets[d + 1]
        if start >= end:
            continue

        # Predict (only after training period)
        if d >= n_train_days:
            for i in range(start, end):
                p1, p2 = player1[i], player2[i]
                preds[i] = predict_single_with_handicap(
                    skills[p1], skills[p2],
                    consistency[p1], consistency[p2],
                    scale, handicaps[i],
                )

        # Update
        update_ratings_weighted_sequential(
            player1[start:end], player2[start:end], scores[start:end],
            skills, consistency,
            k_factor, consistency_lr, scale, v_floor,
            weights[start:end], handicaps[start:end],
        )

    return preds
