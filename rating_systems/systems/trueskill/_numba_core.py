"""
Numba-accelerated core functions for TrueSkill rating system.

TrueSkill models player skill as a Gaussian distribution N(mu, sigma^2).
Updates use Gaussian message passing with truncated Gaussians for win/loss.

Key formulas:
- c = sqrt(2 * beta^2 + sigma1^2 + sigma2^2)  # total uncertainty
- t = (mu1 - mu2) / c  # normalized skill difference
- v(t) = pdf(t) / Phi(t)  # update factor for mean (truncated Gaussian)
- w(t) = v(t) * (v(t) + t)  # update factor for variance
"""

import math
import numpy as np
from numba import njit, prange


# Constants
SQRT_2 = math.sqrt(2.0)
SQRT_PI = math.sqrt(math.pi)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


@njit(cache=True, fastmath=True, inline="always")
def _pdf(x: float) -> float:
    """Standard normal PDF."""
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


@njit(cache=True, fastmath=True, inline="always")
def _cdf(x: float) -> float:
    """
    Standard normal CDF using error function approximation.
    Accurate to ~1e-7.
    """
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


@njit(cache=True, fastmath=True, inline="always")
def _v_win(t: float) -> float:
    """
    V function for winner update (no draw margin, epsilon=0).
    v(t) = pdf(t) / Phi(t)

    This is the amount the mean should shift toward the observed outcome.
    """
    cdf_t = _cdf(t)
    if cdf_t < 1e-10:
        # Avoid division by zero for extreme negative t
        return -t
    return _pdf(t) / cdf_t


@njit(cache=True, fastmath=True, inline="always")
def _w_win(t: float, v: float) -> float:
    """
    W function for winner update.
    w(t) = v * (v + t)

    This determines how much the variance should shrink.
    Must be bounded to (0, 1) to ensure variance stays positive.
    """
    w = v * (v + t)
    # Clamp to valid range
    if w < 0.0:
        return 0.0
    if w > 1.0:
        return 1.0
    return w


@njit(cache=True, fastmath=True, inline="always")
def _v_lose(t: float) -> float:
    """
    V function for loser update.
    For the loser, we use v(-t) = -pdf(t) / Phi(-t)
    """
    cdf_neg_t = _cdf(-t)
    if cdf_neg_t < 1e-10:
        return t
    return -_pdf(t) / cdf_neg_t


@njit(cache=True, fastmath=True)
def update_single_game(
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    score: float,
    beta: float,
) -> tuple:
    """
    Update ratings for a single 1v1 game.

    Args:
        mu1, sigma1: Player 1's skill mean and std dev
        mu2, sigma2: Player 2's skill mean and std dev
        score: 1.0 if player 1 won, 0.0 if player 2 won, 0.5 for draw
        beta: Performance variability parameter

    Returns:
        (new_mu1, new_sigma1, new_mu2, new_sigma2)
    """
    # Total uncertainty in the game outcome
    c_squared = 2.0 * beta * beta + sigma1 * sigma1 + sigma2 * sigma2
    c = math.sqrt(c_squared)

    # Normalized skill difference (positive means player 1 stronger)
    t = (mu1 - mu2) / c

    # Pre-compute sigma^2 / c for efficiency
    sigma1_sq_over_c = sigma1 * sigma1 / c
    sigma2_sq_over_c = sigma2 * sigma2 / c
    sigma1_sq_over_c_sq = sigma1 * sigma1 / c_squared
    sigma2_sq_over_c_sq = sigma2 * sigma2 / c_squared

    if score > 0.5:
        # Player 1 won
        v = _v_win(t)
        w = _w_win(t, v)

        # Winner gets boosted, loser gets reduced
        new_mu1 = mu1 + sigma1_sq_over_c * v
        new_mu2 = mu2 - sigma2_sq_over_c * v

        # Both players' uncertainties decrease
        new_sigma1_sq = sigma1 * sigma1 * (1.0 - sigma1_sq_over_c_sq * w)
        new_sigma2_sq = sigma2 * sigma2 * (1.0 - sigma2_sq_over_c_sq * w)

    elif score < 0.5:
        # Player 2 won (same as player 1 losing)
        v = _v_win(-t)  # Flip perspective
        w = _w_win(-t, v)

        new_mu1 = mu1 - sigma1_sq_over_c * v
        new_mu2 = mu2 + sigma2_sq_over_c * v

        new_sigma1_sq = sigma1 * sigma1 * (1.0 - sigma1_sq_over_c_sq * w)
        new_sigma2_sq = sigma2 * sigma2 * (1.0 - sigma2_sq_over_c_sq * w)

    else:
        # Draw - both move toward each other slightly
        # For draws, use a different v/w calculation
        # Simplified: small update toward the mean
        v = _pdf(t) / (_cdf(t) * _cdf(-t) + 1e-10)
        w = v * v

        new_mu1 = mu1 - sigma1_sq_over_c * v * t / (abs(t) + 0.1)
        new_mu2 = mu2 + sigma2_sq_over_c * v * t / (abs(t) + 0.1)

        new_sigma1_sq = sigma1 * sigma1 * (1.0 - sigma1_sq_over_c_sq * w * 0.5)
        new_sigma2_sq = sigma2 * sigma2 * (1.0 - sigma2_sq_over_c_sq * w * 0.5)

    # Ensure sigma stays positive
    new_sigma1 = math.sqrt(max(new_sigma1_sq, 1e-6))
    new_sigma2 = math.sqrt(max(new_sigma2_sq, 1e-6))

    return new_mu1, new_sigma1, new_mu2, new_sigma2


@njit(cache=True, fastmath=True)
def update_ratings_sequential(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    beta: float,
) -> None:
    """
    Update TrueSkill ratings for a batch of games (sequential within batch).

    Games MUST be processed sequentially to maintain correctness when
    players appear in multiple games.

    Modifies mu and sigma arrays in-place.
    """
    n_games = len(player1)

    for i in range(n_games):
        p1 = player1[i]
        p2 = player2[i]
        score = scores[i]

        new_mu1, new_sigma1, new_mu2, new_sigma2 = update_single_game(
            mu[p1], sigma[p1],
            mu[p2], sigma[p2],
            score,
            beta,
        )

        mu[p1] = new_mu1
        sigma[p1] = new_sigma1
        mu[p2] = new_mu2
        sigma[p2] = new_sigma2


@njit(cache=True, fastmath=True)
def fit_all_days(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    day_offsets: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    beta: float,
) -> None:
    """
    Fit TrueSkill ratings for ALL days in a single Numba call.

    This avoids Python iteration overhead entirely.
    Games within each day are processed sequentially.

    Args:
        player1: All player1 IDs (sorted by day)
        player2: All player2 IDs (sorted by day)
        scores: All scores (sorted by day)
        day_offsets: Start index for each day (length = num_days + 1)
        mu: Mean ratings array to update in-place
        sigma: Uncertainty array to update in-place
        beta: Performance variability parameter
    """
    n_days = len(day_offsets) - 1

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        # Process all games in this day sequentially
        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            score = scores[i]

            new_mu1, new_sigma1, new_mu2, new_sigma2 = update_single_game(
                mu[p1], sigma[p1],
                mu[p2], sigma[p2],
                score,
                beta,
            )

            mu[p1] = new_mu1
            sigma[p1] = new_sigma1
            mu[p2] = new_mu2
            sigma[p2] = new_sigma2


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Predict win probabilities for a batch of matchups (fully parallel).

    P(player1 wins) = Phi((mu1 - mu2) / c)
    where c = sqrt(2*beta^2 + sigma1^2 + sigma2^2)
    """
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)

    for i in prange(n):
        p1 = player1[i]
        p2 = player2[i]

        mu1 = mu[p1]
        mu2 = mu[p2]
        sigma1 = sigma[p1]
        sigma2 = sigma[p2]

        c = math.sqrt(2.0 * beta * beta + sigma1 * sigma1 + sigma2 * sigma2)
        t = (mu1 - mu2) / c

        proba[i] = _cdf(t)

    return proba


@njit(cache=True, fastmath=True)
def predict_single(
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    beta: float,
) -> float:
    """Predict win probability for a single matchup."""
    c = math.sqrt(2.0 * beta * beta + sigma1 * sigma1 + sigma2 * sigma2)
    t = (mu1 - mu2) / c
    return _cdf(t)


@njit(cache=True)
def get_top_n_indices(ratings: np.ndarray, n: int) -> np.ndarray:
    """
    Get indices of top N rated players.

    Uses partial sort for efficiency when n << len(ratings).
    """
    if n >= len(ratings):
        return np.argsort(ratings)[::-1]

    top_indices = np.argpartition(ratings, -n)[-n:]
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


@njit(cache=True, fastmath=True)
def compute_conservative_rating(
    mu: np.ndarray,
    sigma: np.ndarray,
    k: float = 3.0,
) -> np.ndarray:
    """
    Compute conservative rating: mu - k * sigma.

    This is the "minimum plausible rating" at a given confidence level.
    Default k=3 gives ~99.7% confidence.
    """
    return mu - k * sigma


@njit(cache=True, fastmath=True, parallel=True)
def compute_all_vs_all_matrix(
    mu: np.ndarray,
    sigma: np.ndarray,
    player_indices: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Compute win probability matrix for selected players.

    Returns matrix where result[i,j] = P(player_indices[i] beats player_indices[j])
    """
    n = len(player_indices)
    matrix = np.empty((n, n), dtype=np.float64)

    for i in prange(n):
        p1 = player_indices[i]
        mu1 = mu[p1]
        sigma1 = sigma[p1]

        for j in range(n):
            if i == j:
                matrix[i, j] = 0.5
            else:
                p2 = player_indices[j]
                mu2 = mu[p2]
                sigma2 = sigma[p2]

                c = math.sqrt(2.0 * beta * beta + sigma1 * sigma1 + sigma2 * sigma2)
                matrix[i, j] = _cdf((mu1 - mu2) / c)

    return matrix
