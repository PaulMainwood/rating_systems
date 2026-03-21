"""Numba-accelerated kernels for Disc decomposition rating system.

Disc model: P(i beats j) = sigmoid(v_i * v_j * (u_i - u_j) * log10 / scale)

Each player has two parameters:
  u_i  — skill (analogous to Elo rating)
  v_i  — consistency (modulates how decisive skill gaps are)

When v_i = v_j = 1, the model reduces to standard Elo.
"""

import math

import numpy as np
from numba import njit, prange

_LOG10 = math.log(10.0)


@njit(cache=True, fastmath=True, inline="always")
def _sigmoid(x):
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


@njit(cache=True, fastmath=True, inline="always")
def _predict(u1, u2, v1, v2, scale):
    """P(player 1 beats player 2)."""
    c = _LOG10 / scale
    z = v1 * v2 * (u1 - u2) * c
    return _sigmoid(z)


@njit(cache=True, fastmath=True)
def predict_single(u1, u2, v1, v2, scale):
    """Predict P(player 1 wins) for a single matchup."""
    return _predict(u1, u2, v1, v2, scale)


@njit(cache=True, fastmath=True)
def predict_single_with_handicap(u1, u2, v1, v2, scale, handicap):
    """Predict P(player 1 wins) with additive handicap in skill space."""
    c = _LOG10 / scale
    z = v1 * v2 * (u1 - u2 + handicap) * c
    return _sigmoid(z)


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(player1, player2, skills, consistency, scale):
    """Batch prediction, parallel."""
    n = len(player1)
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        p1, p2 = player1[i], player2[i]
        out[i] = _predict(skills[p1], skills[p2],
                          consistency[p1], consistency[p2], scale)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch_with_handicap(player1, player2, skills, consistency,
                                      scale, handicaps):
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
def update_ratings_sequential(player1, player2, scores,
                              skills, consistency,
                              k_factor, consistency_lr, scale, v_floor):
    """Update skills and consistency for one day's games, in place."""
    c = _LOG10 / scale
    n = len(player1)
    for i in range(n):
        p1, p2 = player1[i], player2[i]
        u1, u2 = skills[p1], skills[p2]
        v1, v2 = consistency[p1], consistency[p2]
        score = scores[i]

        delta = u1 - u2
        z = v1 * v2 * delta * c
        p = _sigmoid(z)
        error = score - p

        # Skill updates (Elo convention: divide out c)
        skills[p1] += k_factor * error * v1 * v2
        skills[p2] -= k_factor * error * v1 * v2

        # Consistency updates (keep full gradient)
        consistency[p1] += consistency_lr * error * v2 * delta * c
        consistency[p2] += consistency_lr * error * v1 * delta * c

        # Floor consistency
        if consistency[p1] < v_floor:
            consistency[p1] = v_floor
        if consistency[p2] < v_floor:
            consistency[p2] = v_floor


@njit(cache=True, fastmath=True)
def fit_all_days(player1, player2, scores, day_offsets,
                 skills, consistency,
                 k_factor, consistency_lr, scale, v_floor):
    """Process all days sequentially in a single numba call."""
    n_days = len(day_offsets) - 1
    for d in range(n_days):
        start = day_offsets[d]
        end = day_offsets[d + 1]
        if start >= end:
            continue
        update_ratings_sequential(
            player1[start:end], player2[start:end], scores[start:end],
            skills, consistency,
            k_factor, consistency_lr, scale, v_floor,
        )
