"""
Numba-accelerated core functions for Glicko-2 rating system.

Design principles for maximum efficiency:
1. All hot-path functions compiled with @njit(cache=True)
2. Process ALL days in a single Numba call (no Python iteration)
3. Use fastmath=True where precision allows
4. Contiguous arrays throughout
5. Parallel execution for predictions
"""

import math
import numpy as np
from numba import njit, prange


# =============================================================================
# Core Glicko-2 functions
# =============================================================================

@njit(cache=True, fastmath=True, inline="always")
def _g(phi: float) -> float:
    """Calculate g(phi) function."""
    return 1.0 / math.sqrt(1.0 + 3.0 * (phi * phi) / (math.pi * math.pi))


@njit(cache=True, fastmath=True, inline="always")
def _expected_score(mu: float, opp_mu: float, opp_phi: float) -> float:
    """Calculate expected score in Glicko-2 scale."""
    g_phi = _g(opp_phi)
    return 1.0 / (1.0 + math.exp(-g_phi * (mu - opp_mu)))


@njit(cache=True, fastmath=True)
def _update_volatility(
    sigma: float,
    phi: float,
    v: float,
    delta: float,
    tau: float,
    epsilon: float,
) -> float:
    """Update volatility using iterative algorithm (Step 5)."""
    a = math.log(sigma * sigma)
    phi_sq = phi * phi
    delta_sq = delta * delta

    # Set initial bounds
    A = a
    if delta_sq > phi_sq + v:
        B = math.log(delta_sq - phi_sq - v)
    else:
        k = 1
        while True:
            x = a - k * tau
            ex = math.exp(x)
            num1 = ex * (delta_sq - phi_sq - v - ex)
            den1 = 2.0 * ((phi_sq + v + ex) ** 2)
            f_val = num1 / den1 - (x - a) / (tau * tau)
            if f_val >= 0:
                break
            k += 1
            if k > 100:  # Safety limit
                break
        B = a - k * tau

    # Compute f(A) and f(B)
    ex_A = math.exp(A)
    num1_A = ex_A * (delta_sq - phi_sq - v - ex_A)
    den1_A = 2.0 * ((phi_sq + v + ex_A) ** 2)
    f_A = num1_A / den1_A - (A - a) / (tau * tau)

    ex_B = math.exp(B)
    num1_B = ex_B * (delta_sq - phi_sq - v - ex_B)
    den1_B = 2.0 * ((phi_sq + v + ex_B) ** 2)
    f_B = num1_B / den1_B - (B - a) / (tau * tau)

    # Iterative algorithm
    iterations = 0
    max_iterations = 100

    while abs(B - A) > epsilon and iterations < max_iterations:
        C = A + (A - B) * f_A / (f_B - f_A)

        ex_C = math.exp(C)
        num1_C = ex_C * (delta_sq - phi_sq - v - ex_C)
        den1_C = 2.0 * ((phi_sq + v + ex_C) ** 2)
        f_C = num1_C / den1_C - (C - a) / (tau * tau)

        if f_C * f_B <= 0:
            A = B
            f_A = f_B
        else:
            f_A = f_A / 2.0

        B = C
        f_B = f_C
        iterations += 1

    return math.exp(A / 2.0)


@njit(cache=True, fastmath=True)
def _update_player_glicko2(
    player_mu: float,
    player_phi: float,
    player_sigma: float,
    opp_mus: np.ndarray,
    opp_phis: np.ndarray,
    player_scores: np.ndarray,
    tau: float,
    epsilon: float,
) -> tuple:
    """
    Update a single player's rating, RD, and volatility.

    Returns (new_mu, new_phi, new_sigma).
    """
    n_games = len(opp_mus)
    if n_games == 0:
        return player_mu, player_phi, player_sigma

    # Step 3: Compute variance v and delta
    v_inv = 0.0
    delta_sum = 0.0

    for i in range(n_games):
        g_val = _g(opp_phis[i])
        e_val = _expected_score(player_mu, opp_mus[i], opp_phis[i])

        g_sq_e = g_val * g_val * e_val * (1.0 - e_val)
        v_inv += g_sq_e
        delta_sum += g_val * (player_scores[i] - e_val)

    if v_inv > 0:
        v = 1.0 / v_inv
    else:
        v = 1e10

    # Step 4: Compute delta
    delta = v * delta_sum

    # Step 5: Update volatility
    new_sigma = _update_volatility(player_sigma, player_phi, v, delta, tau, epsilon)

    # Step 6: Update phi*
    phi_star = math.sqrt(player_phi * player_phi + new_sigma * new_sigma)

    # Step 7: Update rating and RD
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    new_mu = player_mu + new_phi * new_phi * delta_sum

    return new_mu, new_phi, new_sigma


@njit(cache=True)
def update_ratings_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    sigma: np.ndarray,
    last_played: np.ndarray,
    current_day: int,
    tau: float,
    epsilon: float,
    max_phi: float,
) -> int:
    """
    Update Glicko-2 ratings for all players in a rating period.

    Returns number of players updated.
    """
    n_games = len(player1)
    if n_games == 0:
        return 0

    # Get unique players
    all_players_set = set()
    for i in range(n_games):
        all_players_set.add(player1[i])
        all_players_set.add(player2[i])

    unique_players = np.array(list(all_players_set), dtype=np.int64)
    n_players = len(unique_players)

    # Step 1: Update phi for inactivity and store pre-period values
    pre_mu = mu.copy()
    pre_phi = phi.copy()
    pre_sigma = sigma.copy()

    for i in range(n_players):
        p = unique_players[i]
        days_inactive = current_day - last_played[p]
        if days_inactive > 0:
            # phi* = sqrt(phi^2 + sigma^2 * days)
            new_phi = math.sqrt(phi[p] * phi[p] + sigma[p] * sigma[p] * days_inactive)
            phi[p] = min(new_phi, max_phi)
        pre_phi[p] = phi[p]

    # Process each player's games
    for i in range(n_players):
        player = unique_players[i]

        # Collect games for this player
        opponents = []
        player_scores_list = []

        for j in range(n_games):
            if player1[j] == player:
                opponents.append(player2[j])
                player_scores_list.append(scores[j])
            elif player2[j] == player:
                opponents.append(player1[j])
                player_scores_list.append(1.0 - scores[j])

        count = len(opponents)
        if count == 0:
            continue

        # Get opponent values (pre-period)
        opp_mus = np.empty(count, dtype=np.float64)
        opp_phis = np.empty(count, dtype=np.float64)
        player_scores_arr = np.empty(count, dtype=np.float64)

        for j in range(count):
            opp_mus[j] = pre_mu[opponents[j]]
            opp_phis[j] = pre_phi[opponents[j]]
            player_scores_arr[j] = player_scores_list[j]

        # Update rating, RD, and volatility
        new_mu, new_phi, new_sigma = _update_player_glicko2(
            pre_mu[player],
            pre_phi[player],
            pre_sigma[player],
            opp_mus,
            opp_phis,
            player_scores_arr,
            tau,
            epsilon,
        )

        mu[player] = new_mu
        phi[player] = new_phi
        sigma[player] = new_sigma
        last_played[player] = current_day

    return n_players


@njit(cache=True, fastmath=True)
def fit_all_days(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    day_indices: np.ndarray,
    day_offsets: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    sigma: np.ndarray,
    last_played: np.ndarray,
    tau: float,
    epsilon: float,
    max_phi: float,
) -> int:
    """
    Fit Glicko-2 ratings for ALL days in a single Numba call.

    This avoids Python iteration overhead entirely.
    Each day is processed as a rating period (games simultaneous within day).

    Args:
        player1: All player1 IDs (sorted by day)
        player2: All player2 IDs (sorted by day)
        scores: All scores (sorted by day)
        day_indices: Day values for each rating period
        day_offsets: Start index for each day (length = num_days + 1)
        mu: Rating array (Glicko-2 scale) to update in-place
        phi: Rating deviation array (Glicko-2 scale) to update in-place
        sigma: Volatility array to update in-place
        last_played: Last played day array to update in-place
        tau: System constant
        epsilon: Convergence tolerance
        max_phi: Maximum phi value

    Returns:
        Total number of player-updates
    """
    n_days = len(day_offsets) - 1
    total_updates = 0

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]
        current_day = day_indices[day_idx]

        if start >= end:
            continue

        # Create views for this day's games
        p1_day = player1[start:end]
        p2_day = player2[start:end]
        scores_day = scores[start:end]
        n_games = end - start

        # Find unique active players for this day
        players_set = set()
        for i in range(n_games):
            players_set.add(p1_day[i])
            players_set.add(p2_day[i])

        active_players = np.array(list(players_set), dtype=np.int64)
        n_players = len(active_players)

        # Update phi for inactivity and store pre-period values
        pre_mu = mu.copy()
        pre_phi = phi.copy()
        pre_sigma = sigma.copy()

        for i in range(n_players):
            p = active_players[i]
            days_inactive = current_day - last_played[p]
            if days_inactive > 0:
                new_phi = math.sqrt(phi[p] * phi[p] + sigma[p] * sigma[p] * days_inactive)
                phi[p] = min(new_phi, max_phi)
            pre_phi[p] = phi[p]

        # Process each active player
        for i in range(n_players):
            player = active_players[i]
            player_mu = pre_mu[player]

            # Collect games for this player
            opponents = []
            player_scores_list = []

            for j in range(n_games):
                if p1_day[j] == player:
                    opponents.append(p2_day[j])
                    player_scores_list.append(scores_day[j])
                elif p2_day[j] == player:
                    opponents.append(p1_day[j])
                    player_scores_list.append(1.0 - scores_day[j])

            count = len(opponents)
            if count == 0:
                continue

            # Get opponent values (pre-period)
            opp_mus = np.empty(count, dtype=np.float64)
            opp_phis = np.empty(count, dtype=np.float64)
            player_scores_arr = np.empty(count, dtype=np.float64)

            for j in range(count):
                opp_mus[j] = pre_mu[opponents[j]]
                opp_phis[j] = pre_phi[opponents[j]]
                player_scores_arr[j] = player_scores_list[j]

            # Update rating, RD, and volatility
            new_mu, new_phi, new_sigma = _update_player_glicko2(
                player_mu,
                pre_phi[player],
                pre_sigma[player],
                opp_mus,
                opp_phis,
                player_scores_arr,
                tau,
                epsilon,
            )

            mu[player] = new_mu
            phi[player] = new_phi
            sigma[player] = new_sigma
            last_played[player] = current_day

        total_updates += n_players

    return total_updates


# =============================================================================
# Prediction functions
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Predict win probabilities for a batch of games (parallelized).
    """
    n_games = len(player1)
    proba = np.empty(n_games, dtype=np.float64)

    for i in prange(n_games):
        p1 = player1[i]
        p2 = player2[i]

        mu1 = mu[p1]
        mu2 = mu[p2]
        phi1 = phi[p1]
        phi2 = phi[p2]

        # Combined phi for prediction
        combined_phi = math.sqrt(phi1 * phi1 + phi2 * phi2)
        g_combined = _g(combined_phi)

        proba[i] = 1.0 / (1.0 + math.exp(-g_combined * (mu1 - mu2)))

    return proba


@njit(cache=True, fastmath=True)
def predict_single(
    mu1: float,
    phi1: float,
    mu2: float,
    phi2: float,
) -> float:
    """Predict win probability for a single matchup."""
    combined_phi = math.sqrt(phi1 * phi1 + phi2 * phi2)
    g_combined = _g(combined_phi)
    return 1.0 / (1.0 + math.exp(-g_combined * (mu1 - mu2)))


# =============================================================================
# Utility functions
# =============================================================================

@njit(cache=True)
def get_top_n_indices(ratings: np.ndarray, n: int) -> np.ndarray:
    """Get indices of top N rated players."""
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


@njit(cache=True, fastmath=True, parallel=True)
def compute_all_vs_all_matrix(
    mu: np.ndarray,
    phi: np.ndarray,
    player_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute win probability matrix for selected players.

    Returns matrix where result[i,j] = P(player_indices[i] beats player_indices[j])
    """
    n = len(player_indices)
    matrix = np.empty((n, n), dtype=np.float64)

    for i in prange(n):
        pi = player_indices[i]
        mui = mu[pi]
        phii = phi[pi]

        for j in range(n):
            if i == j:
                matrix[i, j] = 0.5
            else:
                pj = player_indices[j]
                muj = mu[pj]
                phij = phi[pj]

                combined_phi = math.sqrt(phii * phii + phij * phij)
                g_combined = _g(combined_phi)
                matrix[i, j] = 1.0 / (1.0 + math.exp(-g_combined * (mui - muj)))

    return matrix
