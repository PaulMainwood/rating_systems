"""
Numba-accelerated core functions for Yuksel rating system.

The Yuksel method (Yuksel 2024, "Skill-Based Matchmaking for Competitive Two-Player Games")
is an online rating system that combines ideas from Elo and Glicko with adaptive
step-size control similar to modern optimizers like Adam.

Key innovations:
1. Tracks rating variance via Welford's online algorithm to estimate uncertainty
2. Uses Glicko's g function to downweight updates when uncertainty is high
3. Maintains a "direction" term D that accumulates curvature for adaptive updates
4. Zero-sum updates that respect both players' uncertainties

Design principles for maximum efficiency:
1. All hot-path functions compiled with @njit(cache=True)
2. Process ALL days in a single Numba call (no Python iteration)
3. Use fastmath=True where precision allows
4. Contiguous arrays throughout
5. Parallel execution for predictions
"""

import numpy as np
from numba import njit, prange

# Constants (same as Glicko)
# Q = ln(10) / 400 ≈ 0.00575646273
Q = np.log(10.0) / 400.0
Q2 = Q * Q
Q2_3_OVER_PI2 = 3.0 * Q2 / (np.pi * np.pi)


@njit(cache=True, fastmath=True, inline="always")
def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


@njit(cache=True, fastmath=True, inline="always")
def _g(phi: float) -> float:
    """
    Glicko g function - reduces weight when uncertainty (phi) is high.

    g(phi) = 1 / sqrt(1 + 3*Q²*phi² / π²)

    When phi is large (high uncertainty), g approaches 0.
    When phi is small (low uncertainty), g approaches 1.
    """
    return 1.0 / np.sqrt(1.0 + Q2_3_OVER_PI2 * phi * phi)


@njit(cache=True, fastmath=True)
def update_single_game(
    p1: int,
    p2: int,
    score: float,
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
    Update ratings for a single game using the Yuksel method.

    The algorithm proceeds as follows:

    1. Compute win probability using sigmoid (like Elo)
    2. Estimate uncertainty (phi) for each player from their variance history
    3. Apply g function to downweight updates when uncertainty is high
    4. Compute forces F1, F2 based on prediction error
    5. Update direction terms D1, D2 (accumulated curvature with momentum)
    6. Compute adaptive step size delta_r using D1, D2, F1, F2
    7. Apply zero-sum update: r1 += delta_r, r2 -= delta_r
    8. Update variance tracking (W, R, V) for each player

    Args:
        p1, p2: Player indices
        score: Outcome (1.0 = p1 wins, 0.0 = p2 wins)
        ratings: Current ratings array
        R: Weighted mean of ratings (for variance calculation)
        W: Accumulated weights (for Welford's algorithm)
        V: Weighted variance of ratings
        D: Accumulated direction/curvature term
        delta_r_max: Maximum rating change per game
        alpha: Decay factor for uncertainty (typically 2.0)
        scaling_factor: Scale factor for final update (typically 0.9)
    """
    r1 = ratings[p1]
    r2 = ratings[p2]

    # Step 1: Compute win probability using Elo-style sigmoid
    # P(p1 wins) = 1 / (1 + 10^((r2-r1)/400))
    #            = sigmoid(Q * (r1 - r2))
    prob = _sigmoid(Q * (r1 - r2))

    # Step 2: Estimate uncertainty (phi) for each player
    # phi = sqrt(V / W) is the standard deviation of the rating history
    # This gives us a measure of how stable the player's rating is
    # Use small epsilon to avoid division by zero
    phi_1 = np.sqrt(V[p1] / max(W[p1], 1e-10))
    phi_2 = np.sqrt(V[p2] / max(W[p2], 1e-10))

    # Step 3: Apply g function to reduce influence of uncertain players
    # g approaches 1 for low uncertainty, 0 for high uncertainty
    g_1 = _g(phi_1)
    g_2 = _g(phi_2)

    # Also compute g with alpha*phi for momentum decay
    g_alpha_1 = _g(alpha * phi_1)
    g_alpha_2 = _g(alpha * phi_2)

    # Step 4: Compute forces (prediction error scaled by opponent's certainty)
    outcome_delta = score - prob
    F_1 = g_2 * outcome_delta   # Force on p1, scaled by p2's certainty
    F_2 = -g_1 * outcome_delta  # Force on p2, scaled by p1's certainty

    # Step 5: Update direction terms with momentum
    # D accumulates curvature information (like second moments in Adam)
    # The second derivative of log-likelihood is Q * prob * (1-prob)
    second_deriv = Q * prob * (1.0 - prob)

    D_1 = D[p1]
    D_1 = (g_2 * second_deriv) + (g_alpha_1 * D_1)  # Accumulate with decay

    D_2 = D[p2]
    D_2 = (g_1 * second_deriv) + (g_alpha_2 * D_2)

    # Step 6: Compute adaptive step size
    # This is a form of natural gradient descent:
    # delta_r = (D1*F1 - D2*F2) / (D1² + D2²)
    # The numerator combines both players' contributions
    # The denominator normalizes by accumulated curvature
    denom = D_1 * D_1 + D_2 * D_2
    if denom > 1e-10:
        delta_r = (D_1 * F_1 - D_2 * F_2) / denom
    else:
        delta_r = 0.0

    # Clamp to maximum rating change
    if delta_r > delta_r_max:
        delta_r = delta_r_max
    elif delta_r < -delta_r_max:
        delta_r = -delta_r_max

    # Adjust D based on actual delta_r (if non-zero)
    # This normalizes D for future updates
    if np.abs(delta_r) > 1e-10:
        D_1 = F_1 / delta_r
        D_2 = -F_2 / delta_r

    D[p1] = D_1
    D[p2] = D_2

    # Step 7: Apply rating updates (zero-sum)
    scaled_update = scaling_factor * delta_r
    new_r_1 = r1 + scaled_update
    new_r_2 = r2 - scaled_update

    ratings[p1] = new_r_1
    ratings[p2] = new_r_2

    # Step 8: Update variance tracking using Welford's online algorithm
    # This allows us to estimate rating variance without storing history
    # W is the accumulated weight, R is the weighted mean, V is the weighted variance

    # Player 1 update
    omega_1 = g_alpha_1  # Decay factor based on uncertainty
    W[p1] = omega_1 * W[p1] + 1.0
    delta_R_1 = new_r_1 - R[p1]
    R[p1] = R[p1] + delta_R_1 / W[p1]
    V[p1] = omega_1 * V[p1] + delta_R_1 * (new_r_1 - R[p1])

    # Player 2 update
    omega_2 = g_alpha_2
    W[p2] = omega_2 * W[p2] + 1.0
    delta_R_2 = new_r_2 - R[p2]
    R[p2] = R[p2] + delta_R_2 / W[p2]
    V[p2] = omega_2 * V[p2] + delta_R_2 * (new_r_2 - R[p2])


@njit(cache=True, fastmath=True)
def update_ratings_sequential(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
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
    Update ratings for a batch of games (sequential within batch).

    Games MUST be processed sequentially to maintain correctness when
    players appear in multiple games within the same batch.
    """
    n_games = len(player1)

    for i in range(n_games):
        update_single_game(
            player1[i],
            player2[i],
            scores[i],
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
def fit_all_days(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
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
    Fit ratings for ALL days in a single Numba call.

    This avoids Python iteration overhead entirely.
    Games within each day are processed sequentially.
    """
    n_days = len(day_offsets) - 1

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        # Process all games in this day sequentially
        for i in range(start, end):
            update_single_game(
                player1[i],
                player2[i],
                scores[i],
                ratings,
                R,
                W,
                V,
                D,
                delta_r_max,
                alpha,
                scaling_factor,
            )


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
) -> np.ndarray:
    """
    Predict win probabilities for a batch of matchups (fully parallel).

    Uses simple Elo-style prediction (no uncertainty adjustment).
    This is appropriate because the ratings already incorporate uncertainty
    through the adaptive update mechanism.
    """
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)

    for i in prange(n):
        r1 = ratings[player1[i]]
        r2 = ratings[player2[i]]
        proba[i] = _sigmoid(Q * (r1 - r2))

    return proba


@njit(cache=True, fastmath=True)
def predict_single(r1: float, r2: float) -> float:
    """Predict win probability for a single matchup."""
    return _sigmoid(Q * (r1 - r2))


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
def compute_phi(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute uncertainty (phi) for all players.

    phi = sqrt(V / W) is the standard deviation of the rating history.
    Returns array of phi values.
    """
    n = len(V)
    phi = np.empty(n, dtype=np.float64)

    for i in range(n):
        if W[i] > 1e-10:
            phi[i] = np.sqrt(V[i] / W[i])
        else:
            phi[i] = 350.0  # Default high uncertainty

    return phi
