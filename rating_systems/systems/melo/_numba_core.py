"""
Numba-accelerated core functions for mElo (multidimensional Elo) rating system.

mElo extends Elo with per-player style vectors that capture intransitive
(rock-paper-scissors) matchup effects. Each player has a scalar rating r_i
and a 2k-dimensional style vector c_i. The intransitive component is
c_i^T @ Omega @ c_j where Omega is a fixed block-diagonal antisymmetric matrix.

Style vectors are clipped to max_c_norm after each update to prevent the
gradient feedback loop (update ∝ ||c_opponent||) from causing divergence.

Reference: Balduzzi et al., "Re-evaluating Evaluation", NeurIPS 2018.

Design principles (same as Elo core):
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


@njit(cache=True, fastmath=True, inline="always")
def _intransitive_term(c_i: np.ndarray, c_j: np.ndarray, k_dim: int) -> float:
    """Compute c_i^T @ Omega @ c_j for k_dim-dimensional intransitivity.

    Omega is block-diagonal with k blocks of [[0, 1], [-1, 0]].
    So the term is sum over m: c_i[2m]*c_j[2m+1] - c_i[2m+1]*c_j[2m].
    This is the 2D cross-product generalised to k dimensions.
    """
    result = 0.0
    for m in range(k_dim):
        result += c_i[2 * m] * c_j[2 * m + 1] - c_i[2 * m + 1] * c_j[2 * m]
    return result


@njit(cache=True, fastmath=True, inline="always")
def _clip_c_norm(c_vectors: np.ndarray, player: int, c_dim: int, max_c_norm: float) -> None:
    """Clip a player's style vector to max_c_norm in-place."""
    norm_sq = 0.0
    for d in range(c_dim):
        norm_sq += c_vectors[player, d] * c_vectors[player, d]
    if norm_sq > max_c_norm * max_c_norm:
        inv_scale = max_c_norm / np.sqrt(norm_sq)
        for d in range(c_dim):
            c_vectors[player, d] *= inv_scale


@njit(cache=True, fastmath=True)
def update_ratings_sequential(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    ratings: np.ndarray,
    c_vectors: np.ndarray,
    k_factor: float,
    c_factor: float,
    scale: float,
    k_dim: int,
    max_c_norm: float,
) -> None:
    """
    Update mElo ratings + style vectors for a batch of games (sequential).

    Games MUST be processed sequentially to maintain correctness when
    players appear in multiple games within the same batch.

    Modifies ratings and c_vectors arrays in-place.

    The update rules (SGD on log-loss):
        delta = score - expected
        r_i += k_factor * delta
        r_j -= k_factor * delta
        c_i += c_factor * delta * (Omega @ c_j)
        c_j -= c_factor * delta * (Omega @ c_i)
        clip ||c_i||, ||c_j|| to max_c_norm

    where Omega @ c = [c[1], -c[0], c[3], -c[2], ...] for each 2x2 block.
    """
    n_games = len(player1)
    log10_scale = np.log(10.0) / scale
    c_dim = 2 * k_dim

    for i in range(n_games):
        p1 = player1[i]
        p2 = player2[i]
        score = scores[i]

        r1 = ratings[p1]
        r2 = ratings[p2]

        # Compute intransitive term using pre-update style vectors
        intrans = _intransitive_term(c_vectors[p1], c_vectors[p2], k_dim)

        # Expected score including intransitive component
        logit = (r1 - r2 + intrans) * log10_scale
        expected = _sigmoid(logit)

        # Rating delta
        delta = score - expected

        # Update scalar ratings (same as Elo)
        ratings[p1] = r1 + k_factor * delta
        ratings[p2] = r2 - k_factor * delta

        # Compute Omega @ c for both players BEFORE updating either
        # Omega @ c = [c[1], -c[0], c[3], -c[2], ...]
        omega_c2 = np.empty(c_dim, dtype=np.float64)
        omega_c1 = np.empty(c_dim, dtype=np.float64)
        for m in range(k_dim):
            omega_c2[2 * m] = c_vectors[p2, 2 * m + 1]
            omega_c2[2 * m + 1] = -c_vectors[p2, 2 * m]
            omega_c1[2 * m] = c_vectors[p1, 2 * m + 1]
            omega_c1[2 * m + 1] = -c_vectors[p1, 2 * m]

        # Update style vectors
        for d in range(c_dim):
            c_vectors[p1, d] += c_factor * delta * omega_c2[d]
            c_vectors[p2, d] -= c_factor * delta * omega_c1[d]

        # Clip to max norm to prevent divergence
        _clip_c_norm(c_vectors, p1, c_dim, max_c_norm)
        _clip_c_norm(c_vectors, p2, c_dim, max_c_norm)


@njit(cache=True, fastmath=True)
def fit_all_days(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    day_offsets: np.ndarray,
    ratings: np.ndarray,
    c_vectors: np.ndarray,
    k_factor: float,
    c_factor: float,
    scale: float,
    k_dim: int,
    max_c_norm: float,
) -> None:
    """
    Fit mElo ratings for ALL days in a single Numba call.

    Avoids Python iteration overhead entirely.
    Games within each day are processed sequentially.
    """
    n_days = len(day_offsets) - 1
    log10_scale = np.log(10.0) / scale
    c_dim = 2 * k_dim

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            score = scores[i]

            r1 = ratings[p1]
            r2 = ratings[p2]

            intrans = _intransitive_term(c_vectors[p1], c_vectors[p2], k_dim)
            logit = (r1 - r2 + intrans) * log10_scale
            expected = _sigmoid(logit)
            delta = score - expected

            # Update ratings
            ratings[p1] = r1 + k_factor * delta
            ratings[p2] = r2 - k_factor * delta

            # Compute Omega @ c before updating
            omega_c2 = np.empty(c_dim, dtype=np.float64)
            omega_c1 = np.empty(c_dim, dtype=np.float64)
            for m in range(k_dim):
                omega_c2[2 * m] = c_vectors[p2, 2 * m + 1]
                omega_c2[2 * m + 1] = -c_vectors[p2, 2 * m]
                omega_c1[2 * m] = c_vectors[p1, 2 * m + 1]
                omega_c1[2 * m + 1] = -c_vectors[p1, 2 * m]

            for d in range(c_dim):
                c_vectors[p1, d] += c_factor * delta * omega_c2[d]
                c_vectors[p2, d] -= c_factor * delta * omega_c1[d]

            # Clip to max norm
            _clip_c_norm(c_vectors, p1, c_dim, max_c_norm)
            _clip_c_norm(c_vectors, p2, c_dim, max_c_norm)


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
    c_vectors: np.ndarray,
    scale: float,
    k_dim: int,
) -> np.ndarray:
    """
    Predict win probabilities for a batch of matchups (fully parallel).

    Each prediction is independent — embarrassingly parallel.
    """
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)
    log10_scale = np.log(10.0) / scale

    for i in prange(n):
        p1 = player1[i]
        p2 = player2[i]
        diff = ratings[p1] - ratings[p2]
        diff += _intransitive_term(c_vectors[p1], c_vectors[p2], k_dim)
        proba[i] = _sigmoid(diff * log10_scale)

    return proba


@njit(cache=True, fastmath=True)
def predict_single(
    r1: float,
    r2: float,
    c1: np.ndarray,
    c2: np.ndarray,
    scale: float,
    k_dim: int,
) -> float:
    """Predict win probability for a single matchup."""
    diff = r1 - r2 + _intransitive_term(c1, c2, k_dim)
    return _sigmoid(diff * np.log(10.0) / scale)


@njit(cache=True)
def get_top_n_indices(ratings: np.ndarray, n: int) -> np.ndarray:
    """Get indices of top N rated players by scalar rating."""
    if n >= len(ratings):
        return np.argsort(ratings)[::-1]

    top_indices = np.argpartition(ratings, -n)[-n:]
    sorted_order = np.argsort(ratings[top_indices])[::-1]
    return top_indices[sorted_order]
