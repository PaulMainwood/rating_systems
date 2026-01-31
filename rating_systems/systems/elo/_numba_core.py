"""
Numba-accelerated core functions for Elo rating system.

Design principles for maximum efficiency:
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
def _expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    """Expected score for player A against player B."""
    return _sigmoid((rating_a - rating_b) * np.log(10.0) / scale)


@njit(cache=True, fastmath=True)
def update_ratings_sequential(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    ratings: np.ndarray,
    k_factor: float,
    scale: float,
) -> None:
    """
    Update Elo ratings for a batch of games (sequential within batch).

    This is the core update loop. Games MUST be processed sequentially
    to maintain correctness when players appear in multiple games.

    Modifies ratings array in-place for zero allocation overhead.
    """
    n_games = len(player1)
    log10_scale = np.log(10.0) / scale

    for i in range(n_games):
        p1 = player1[i]
        p2 = player2[i]
        score = scores[i]

        r1 = ratings[p1]
        r2 = ratings[p2]

        # Expected score using stable sigmoid
        e1 = _sigmoid((r1 - r2) * log10_scale)

        # Update both players
        delta = k_factor * (score - e1)
        ratings[p1] = r1 + delta
        ratings[p2] = r2 - delta


@njit(cache=True, fastmath=True)
def fit_all_days(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    day_offsets: np.ndarray,
    ratings: np.ndarray,
    k_factor: float,
    scale: float,
) -> None:
    """
    Fit Elo ratings for ALL days in a single Numba call.

    This avoids Python iteration overhead entirely.
    Games within each day are processed sequentially.

    Args:
        player1: All player1 IDs (sorted by day)
        player2: All player2 IDs (sorted by day)
        scores: All scores (sorted by day)
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

        # Process all games in this day sequentially
        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            score = scores[i]

            r1 = ratings[p1]
            r2 = ratings[p2]

            e1 = _sigmoid((r1 - r2) * log10_scale)
            delta = k_factor * (score - e1)

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

    This is embarrassingly parallel - each prediction is independent.
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


@njit(cache=True, fastmath=True, parallel=True)
def compute_all_vs_all_matrix(
    ratings: np.ndarray,
    player_indices: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Compute win probability matrix for selected players.

    Returns matrix where result[i,j] = P(player_indices[i] beats player_indices[j])
    """
    n = len(player_indices)
    matrix = np.empty((n, n), dtype=np.float64)
    log10_scale = np.log(10.0) / scale

    for i in prange(n):
        ri = ratings[player_indices[i]]
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.5
            else:
                rj = ratings[player_indices[j]]
                matrix[i, j] = _sigmoid((ri - rj) * log10_scale)

    return matrix


@njit(cache=True)
def get_top_n_indices(ratings: np.ndarray, n: int) -> np.ndarray:
    """
    Get indices of top N rated players.

    Uses partial sort for efficiency when n << len(ratings).
    """
    if n >= len(ratings):
        # Return all, sorted descending
        return np.argsort(ratings)[::-1]

    # Argpartition is O(n) vs O(n log n) for full sort
    # Get indices of top n (unordered)
    top_indices = np.argpartition(ratings, -n)[-n:]

    # Sort just these n indices by their ratings (descending)
    sorted_order = np.argsort(ratings[top_indices])[::-1]
    return top_indices[sorted_order]


@njit(cache=True)
def get_bottom_n_indices(ratings: np.ndarray, n: int) -> np.ndarray:
    """Get indices of bottom N rated players."""
    if n >= len(ratings):
        return np.argsort(ratings)

    bottom_indices = np.argpartition(ratings, n)[:n]
    sorted_order = np.argsort(ratings[bottom_indices])
    return bottom_indices[sorted_order]


@njit(cache=True, fastmath=True, parallel=True)
def compute_expected_scores_against_field(
    ratings: np.ndarray,
    field_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute each player's expected score against a field of opponents.

    Useful for "strength of schedule" or "rating vs field" analysis.
    """
    n_players = len(ratings)
    n_field = len(field_indices)
    expected = np.zeros(n_players, dtype=np.float64)

    if n_field == 0:
        return expected

    # Pre-compute field ratings
    field_ratings = ratings[field_indices]
    log10_400 = np.log(10.0) / 400.0

    for i in prange(n_players):
        total = 0.0
        ri = ratings[i]
        for j in range(n_field):
            if field_indices[j] != i:
                total += _sigmoid((ri - field_ratings[j]) * log10_400)
        expected[i] = total / n_field

    return expected
