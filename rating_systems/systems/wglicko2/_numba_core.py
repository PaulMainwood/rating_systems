"""
Numba-accelerated core functions for Weighted Glicko-2 rating system.

Weighted Glicko-2 extends Glicko-2 by allowing per-game weights w_j that scale
the Fisher information contribution of each game. The weighted update equations
(Proposition 3 from the paper) are:

    v^{-1} = sum_j( w_j * g(phi_j)^2 * E_j * (1 - E_j) )        [eq. 18]
    delta  = v * sum_j( w_j * g(phi_j) * (s_j - E_j) )            [eq. 19]
    mu'    = mu + phi'^2 * sum_j( w_j * g(phi_j) * (s_j - E_j) )  [Remark 4]

The volatility update (eqs 14-17) is unchanged in form but receives the
weighted v and delta. With w_j = 1 for all games, this is identical to
standard Glicko-2.

Native handicap support adds a per-game handicap h_j (on the Glicko-2 mu
scale) that shifts the expected score:

    E_j = 1 / (1 + exp(-g(phi_j) * (mu_i - mu_j + h_j)))

where h_j > 0 favours player 1. When viewed from player 2's perspective
the handicap is negated.
"""

import math
import numpy as np
from numba import njit, prange

# Reuse utility functions from Glicko-2
from ..glicko2._numba_core import (
    _g,
    _expected_score,
    _update_volatility,
    predict_proba_batch,
    predict_single,
    predict_proba_batch_at_day,
    predict_single_at_day,
    get_top_n_indices,
)


@njit(cache=True, fastmath=True)
def _update_player_wglicko2(
    player_mu: float,
    player_phi: float,
    player_sigma: float,
    opp_mus: np.ndarray,
    opp_phis: np.ndarray,
    player_scores: np.ndarray,
    game_weights: np.ndarray,
    tau: float,
    epsilon: float,
) -> tuple:
    """
    Update a single player's rating, RD, and volatility with per-game weights.

    Returns (new_mu, new_phi, new_sigma).
    """
    n_games = len(opp_mus)
    if n_games == 0:
        return player_mu, player_phi, player_sigma

    # Step 3: Compute weighted variance v and weighted delta_sum
    v_inv = 0.0
    delta_sum = 0.0

    for i in range(n_games):
        g_val = _g(opp_phis[i])
        e_val = _expected_score(player_mu, opp_mus[i], opp_phis[i])
        w_j = game_weights[i]

        g_sq_e = g_val * g_val * e_val * (1.0 - e_val)
        v_inv += w_j * g_sq_e
        delta_sum += w_j * g_val * (player_scores[i] - e_val)

    if v_inv > 0:
        v = 1.0 / v_inv
    else:
        v = 1e10

    # Step 4: Compute delta (for volatility update)
    delta = v * delta_sum

    # Step 5: Update volatility (unchanged in form, receives weighted v and delta)
    new_sigma = _update_volatility(player_sigma, player_phi, v, delta, tau, epsilon)

    # Step 6: Update phi*
    phi_star = math.sqrt(player_phi * player_phi + new_sigma * new_sigma)

    # Step 7: Update rating and RD
    # Remark 4: mu' = mu + phi'^2 * raw_weighted_sum (not delta/v)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    new_mu = player_mu + new_phi * new_phi * delta_sum

    return new_mu, new_phi, new_sigma


@njit(cache=True, fastmath=True, inline="always")
def _expected_score_h(mu: float, opp_mu: float, opp_phi: float, h: float) -> float:
    """Expected score with handicap: E = 1/(1 + exp(-g(phi) * (mu - opp_mu + h)))."""
    g_phi = _g(opp_phi)
    return 1.0 / (1.0 + math.exp(-g_phi * (mu - opp_mu + h)))


@njit(cache=True, fastmath=True)
def _update_player_wglicko2_h(
    player_mu: float,
    player_phi: float,
    player_sigma: float,
    opp_mus: np.ndarray,
    opp_phis: np.ndarray,
    player_scores: np.ndarray,
    game_weights: np.ndarray,
    game_handicaps: np.ndarray,
    tau: float,
    epsilon: float,
) -> tuple:
    """
    Update a single player's rating with per-game weights and handicaps.

    Handicaps are already from this player's perspective (positive = this
    player is favoured).  Returns (new_mu, new_phi, new_sigma).
    """
    n_games = len(opp_mus)
    if n_games == 0:
        return player_mu, player_phi, player_sigma

    v_inv = 0.0
    delta_sum = 0.0

    for i in range(n_games):
        g_val = _g(opp_phis[i])
        e_val = _expected_score_h(player_mu, opp_mus[i], opp_phis[i], game_handicaps[i])
        w_j = game_weights[i]

        g_sq_e = g_val * g_val * e_val * (1.0 - e_val)
        v_inv += w_j * g_sq_e
        delta_sum += w_j * g_val * (player_scores[i] - e_val)

    if v_inv > 0:
        v = 1.0 / v_inv
    else:
        v = 1e10

    delta = v * delta_sum
    new_sigma = _update_volatility(player_sigma, player_phi, v, delta, tau, epsilon)
    phi_star = math.sqrt(player_phi * player_phi + new_sigma * new_sigma)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    new_mu = player_mu + new_phi * new_phi * delta_sum

    return new_mu, new_phi, new_sigma


@njit(cache=True)
def update_ratings_batch_weighted_h(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    handicaps: np.ndarray,
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
    Update Weighted Glicko-2 ratings with per-game handicaps.

    Handicaps are from player1's perspective (positive = player1 advantage).
    Returns number of players updated.
    """
    n_games = len(player1)
    if n_games == 0:
        return 0

    all_players_set = set()
    for i in range(n_games):
        all_players_set.add(player1[i])
        all_players_set.add(player2[i])

    unique_players = np.array(list(all_players_set), dtype=np.int64)
    n_players = len(unique_players)

    pre_mu = mu.copy()
    pre_phi = phi.copy()
    pre_sigma = sigma.copy()

    for i in range(n_players):
        p = unique_players[i]
        days_inactive = current_day - last_played[p]
        if days_inactive > 0:
            new_phi = math.sqrt(phi[p] * phi[p] + sigma[p] * sigma[p] * days_inactive)
            phi[p] = min(new_phi, max_phi)
        pre_phi[p] = phi[p]

    for i in range(n_players):
        player = unique_players[i]

        opponents = []
        player_scores_list = []
        player_weights_list = []
        player_handicaps_list = []

        for j in range(n_games):
            if player1[j] == player:
                opponents.append(player2[j])
                player_scores_list.append(scores[j])
                player_weights_list.append(weights[j])
                player_handicaps_list.append(handicaps[j])
            elif player2[j] == player:
                opponents.append(player1[j])
                player_scores_list.append(1.0 - scores[j])
                player_weights_list.append(weights[j])
                player_handicaps_list.append(-handicaps[j])  # negate for p2

        count = len(opponents)
        if count == 0:
            continue

        opp_mus = np.empty(count, dtype=np.float64)
        opp_phis = np.empty(count, dtype=np.float64)
        player_scores_arr = np.empty(count, dtype=np.float64)
        game_weights = np.empty(count, dtype=np.float64)
        game_handicaps = np.empty(count, dtype=np.float64)

        for j in range(count):
            opp_mus[j] = pre_mu[opponents[j]]
            opp_phis[j] = pre_phi[opponents[j]]
            player_scores_arr[j] = player_scores_list[j]
            game_weights[j] = player_weights_list[j]
            game_handicaps[j] = player_handicaps_list[j]

        new_mu, new_phi, new_sigma = _update_player_wglicko2_h(
            pre_mu[player],
            pre_phi[player],
            pre_sigma[player],
            opp_mus,
            opp_phis,
            player_scores_arr,
            game_weights,
            game_handicaps,
            tau,
            epsilon,
        )

        mu[player] = new_mu
        phi[player] = new_phi
        sigma[player] = new_sigma
        last_played[player] = current_day

    return n_players


@njit(cache=True, fastmath=True)
def fit_all_days_weighted_h(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    handicaps: np.ndarray,
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
    Fit Weighted Glicko-2 ratings for ALL days with per-game handicaps.

    Returns total number of player-updates.
    """
    n_days = len(day_offsets) - 1
    total_updates = 0

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]
        current_day = day_indices[day_idx]

        if start >= end:
            continue

        p1_day = player1[start:end]
        p2_day = player2[start:end]
        scores_day = scores[start:end]
        weights_day = weights[start:end]
        handicaps_day = handicaps[start:end]
        n_games = end - start

        players_set = set()
        for i in range(n_games):
            players_set.add(p1_day[i])
            players_set.add(p2_day[i])

        active_players = np.array(list(players_set), dtype=np.int64)
        n_players = len(active_players)

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

        for i in range(n_players):
            player = active_players[i]
            player_mu = pre_mu[player]

            opponents = []
            player_scores_list = []
            player_weights_list = []
            player_handicaps_list = []

            for j in range(n_games):
                if p1_day[j] == player:
                    opponents.append(p2_day[j])
                    player_scores_list.append(scores_day[j])
                    player_weights_list.append(weights_day[j])
                    player_handicaps_list.append(handicaps_day[j])
                elif p2_day[j] == player:
                    opponents.append(p1_day[j])
                    player_scores_list.append(1.0 - scores_day[j])
                    player_weights_list.append(weights_day[j])
                    player_handicaps_list.append(-handicaps_day[j])

            count = len(opponents)
            if count == 0:
                continue

            opp_mus = np.empty(count, dtype=np.float64)
            opp_phis = np.empty(count, dtype=np.float64)
            player_scores_arr = np.empty(count, dtype=np.float64)
            game_weights = np.empty(count, dtype=np.float64)
            game_handicaps = np.empty(count, dtype=np.float64)

            for j in range(count):
                opp_mus[j] = pre_mu[opponents[j]]
                opp_phis[j] = pre_phi[opponents[j]]
                player_scores_arr[j] = player_scores_list[j]
                game_weights[j] = player_weights_list[j]
                game_handicaps[j] = player_handicaps_list[j]

            new_mu, new_phi, new_sigma = _update_player_wglicko2_h(
                player_mu,
                pre_phi[player],
                pre_sigma[player],
                opp_mus,
                opp_phis,
                player_scores_arr,
                game_weights,
                game_handicaps,
                tau,
                epsilon,
            )

            mu[player] = new_mu
            phi[player] = new_phi
            sigma[player] = new_sigma
            last_played[player] = current_day

        total_updates += n_players

    return total_updates


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch_h(
    player1: np.ndarray,
    player2: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    handicaps: np.ndarray,
) -> np.ndarray:
    """Predict win probabilities with per-game handicaps (parallelised)."""
    n_games = len(player1)
    proba = np.empty(n_games, dtype=np.float64)

    for i in prange(n_games):
        p1 = player1[i]
        p2 = player2[i]
        mu1 = mu[p1]
        mu2 = mu[p2]
        phi1 = phi[p1]
        phi2 = phi[p2]
        combined_phi = math.sqrt(phi1 * phi1 + phi2 * phi2)
        g_combined = _g(combined_phi)
        proba[i] = 1.0 / (1.0 + math.exp(-g_combined * (mu1 - mu2 + handicaps[i])))

    return proba


@njit(cache=True, fastmath=True)
def predict_single_h(
    mu1: float,
    phi1: float,
    mu2: float,
    phi2: float,
    handicap: float,
) -> float:
    """Predict single win probability with handicap."""
    combined_phi = math.sqrt(phi1 * phi1 + phi2 * phi2)
    g_combined = _g(combined_phi)
    return 1.0 / (1.0 + math.exp(-g_combined * (mu1 - mu2 + handicap)))


@njit(cache=True)
def update_ratings_batch_weighted(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
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
    Update Weighted Glicko-2 ratings for all players in a rating period.

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
            new_phi = math.sqrt(phi[p] * phi[p] + sigma[p] * sigma[p] * days_inactive)
            phi[p] = min(new_phi, max_phi)
        pre_phi[p] = phi[p]

    # Process each player's games
    for i in range(n_players):
        player = unique_players[i]

        # Collect games for this player
        opponents = []
        player_scores_list = []
        player_weights_list = []

        for j in range(n_games):
            if player1[j] == player:
                opponents.append(player2[j])
                player_scores_list.append(scores[j])
                player_weights_list.append(weights[j])
            elif player2[j] == player:
                opponents.append(player1[j])
                player_scores_list.append(1.0 - scores[j])
                player_weights_list.append(weights[j])

        count = len(opponents)
        if count == 0:
            continue

        # Get opponent values (pre-period)
        opp_mus = np.empty(count, dtype=np.float64)
        opp_phis = np.empty(count, dtype=np.float64)
        player_scores_arr = np.empty(count, dtype=np.float64)
        game_weights = np.empty(count, dtype=np.float64)

        for j in range(count):
            opp_mus[j] = pre_mu[opponents[j]]
            opp_phis[j] = pre_phi[opponents[j]]
            player_scores_arr[j] = player_scores_list[j]
            game_weights[j] = player_weights_list[j]

        # Update rating, RD, and volatility
        new_mu, new_phi, new_sigma = _update_player_wglicko2(
            pre_mu[player],
            pre_phi[player],
            pre_sigma[player],
            opp_mus,
            opp_phis,
            player_scores_arr,
            game_weights,
            tau,
            epsilon,
        )

        mu[player] = new_mu
        phi[player] = new_phi
        sigma[player] = new_sigma
        last_played[player] = current_day

    return n_players


@njit(cache=True, fastmath=True)
def fit_all_days_weighted(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
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
    Fit Weighted Glicko-2 ratings for ALL days in a single Numba call.

    Each day is a rating period with simultaneous games.
    Per-game weights w_j scale the information contribution.

    Args:
        player1: All player1 IDs (sorted by day)
        player2: All player2 IDs (sorted by day)
        scores: All scores (sorted by day)
        weights: Per-game weights (same order as games)
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

        # Views for this day's games
        p1_day = player1[start:end]
        p2_day = player2[start:end]
        scores_day = scores[start:end]
        weights_day = weights[start:end]
        n_games = end - start

        # Find unique active players
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
            player_weights_list = []

            for j in range(n_games):
                if p1_day[j] == player:
                    opponents.append(p2_day[j])
                    player_scores_list.append(scores_day[j])
                    player_weights_list.append(weights_day[j])
                elif p2_day[j] == player:
                    opponents.append(p1_day[j])
                    player_scores_list.append(1.0 - scores_day[j])
                    player_weights_list.append(weights_day[j])

            count = len(opponents)
            if count == 0:
                continue

            # Get opponent values (pre-period)
            opp_mus = np.empty(count, dtype=np.float64)
            opp_phis = np.empty(count, dtype=np.float64)
            player_scores_arr = np.empty(count, dtype=np.float64)
            game_weights = np.empty(count, dtype=np.float64)

            for j in range(count):
                opp_mus[j] = pre_mu[opponents[j]]
                opp_phis[j] = pre_phi[opponents[j]]
                player_scores_arr[j] = player_scores_list[j]
                game_weights[j] = player_weights_list[j]

            # Update rating, RD, and volatility
            new_mu, new_phi, new_sigma = _update_player_wglicko2(
                player_mu,
                pre_phi[player],
                pre_sigma[player],
                opp_mus,
                opp_phis,
                player_scores_arr,
                game_weights,
                tau,
                epsilon,
            )

            mu[player] = new_mu
            phi[player] = new_phi
            sigma[player] = new_sigma
            last_played[player] = current_day

        total_updates += n_players

    return total_updates
