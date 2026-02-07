"""
Numba-accelerated core functions for Weighted Glicko rating system.

Weighted Glicko extends Glicko by allowing per-game weights w_j that scale
the Fisher information contribution of each game. The weighted update equations
(Proposition 2 from the paper) are:

    d_i^{-2} = q^2 * sum_j( w_j * g(RD_j)^2 * E_ij * (1 - E_ij) )
    RD'_i = (RD_i^{-2} + d_i^{-2})^{-1/2}
    r'_i = r_i + q * RD'^2_i * sum_j( w_j * g(RD_j) * (s_j - E_ij) )

With w_j = 1 for all games, this is identical to standard Glicko.
"""

import math
import numpy as np
from numba import njit, prange

# Reuse constants and utility functions from Glicko
from ..glicko._numba_core import (
    Q,
    Q_SQUARED,
    THREE_Q_SQUARED_OVER_PI_SQUARED,
    _g,
    _expected_score,
    update_rd_for_inactivity,
    predict_proba_batch,
    predict_single,
    get_top_n_indices,
)


@njit(cache=True, fastmath=True)
def update_ratings_batch_weighted(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    ratings: np.ndarray,
    rd: np.ndarray,
    last_played: np.ndarray,
    current_day: int,
    c: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """
    Update Weighted Glicko ratings for a rating period.

    All games are treated as simultaneous (use pre-period ratings).
    Each game has a weight w_j that scales its contribution.

    Returns number of players updated.
    """
    n_games = len(player1)
    if n_games == 0:
        return 0

    # Find unique active players
    players_set = set()
    for i in range(n_games):
        players_set.add(player1[i])
        players_set.add(player2[i])

    active_players = np.array(list(players_set), dtype=np.int64)
    n_players = len(active_players)

    # Update RD for inactivity
    update_rd_for_inactivity(
        rd, last_played, active_players, current_day, c, min_rd, max_rd
    )

    # Store pre-period values
    pre_ratings = ratings.copy()
    pre_rd = rd.copy()

    # Process each active player
    for i in range(n_players):
        player = active_players[i]
        player_rating = pre_ratings[player]

        sum_g_sq_e = 0.0
        sum_g_diff = 0.0
        games_found = 0

        for j in range(n_games):
            opp = -1
            score = 0.0
            w_j = 1.0

            if player1[j] == player:
                opp = player2[j]
                score = scores[j]
                w_j = weights[j]
            elif player2[j] == player:
                opp = player1[j]
                score = 1.0 - scores[j]
                w_j = weights[j]

            if opp >= 0:
                opp_rating = pre_ratings[opp]
                opp_rd = pre_rd[opp]
                g = _g(opp_rd)
                e = _expected_score(player_rating, opp_rating, g)

                sum_g_sq_e += w_j * g * g * e * (1.0 - e)
                sum_g_diff += w_j * g * (score - e)
                games_found += 1

        if games_found > 0:
            # d^2 = 1 / (q^2 * sum(w_j * g^2 * E * (1-E)))
            d_squared_inv = Q_SQUARED * sum_g_sq_e
            if d_squared_inv > 1e-10:
                d_squared = 1.0 / d_squared_inv
            else:
                d_squared = 1e10

            rd_squared = pre_rd[player] ** 2
            new_rd_squared = 1.0 / (1.0 / rd_squared + 1.0 / d_squared)
            new_rd = math.sqrt(new_rd_squared)
            new_rd = min(max(new_rd, min_rd), max_rd)

            rating_change = Q * new_rd_squared * sum_g_diff
            new_rating = player_rating + rating_change

            ratings[player] = new_rating
            rd[player] = new_rd
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
    ratings: np.ndarray,
    rd: np.ndarray,
    last_played: np.ndarray,
    c: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """
    Fit Weighted Glicko ratings for ALL days in a single Numba call.

    Each day is a rating period with simultaneous games.
    Per-game weights w_j scale the information contribution.

    Args:
        player1: All player1 IDs (sorted by day)
        player2: All player2 IDs (sorted by day)
        scores: All scores (sorted by day)
        weights: Per-game weights (same order as games)
        day_indices: Day values for each rating period
        day_offsets: Start index for each day (length = num_days + 1)
        ratings: Ratings array to update in-place
        rd: Rating deviation array to update in-place
        last_played: Last played day array to update in-place
        c: RD increase per period of inactivity
        min_rd: Minimum RD
        max_rd: Maximum RD

    Returns:
        Total number of player-updates
    """
    n_days = len(day_offsets) - 1
    total_updates = 0
    c_squared = c * c

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

        # Update RD for inactivity
        for i in range(n_players):
            p = active_players[i]
            days_inactive = current_day - last_played[p]
            if days_inactive > 0:
                new_rd = math.sqrt(rd[p] * rd[p] + c_squared * days_inactive)
                rd[p] = min(max(new_rd, min_rd), max_rd)

        # Store pre-period values
        pre_ratings = ratings.copy()
        pre_rd = rd.copy()

        # Process each active player
        for i in range(n_players):
            player = active_players[i]
            player_rating = pre_ratings[player]

            sum_g_sq_e = 0.0
            sum_g_diff = 0.0
            games_found = 0

            for j in range(n_games):
                opp = -1
                score = 0.0
                w_j = 1.0

                if p1_day[j] == player:
                    opp = p2_day[j]
                    score = scores_day[j]
                    w_j = weights_day[j]
                elif p2_day[j] == player:
                    opp = p1_day[j]
                    score = 1.0 - scores_day[j]
                    w_j = weights_day[j]

                if opp >= 0:
                    opp_rating = pre_ratings[opp]
                    opp_rd = pre_rd[opp]
                    g = _g(opp_rd)
                    e = _expected_score(player_rating, opp_rating, g)

                    sum_g_sq_e += w_j * g * g * e * (1.0 - e)
                    sum_g_diff += w_j * g * (score - e)
                    games_found += 1

            if games_found > 0:
                d_squared_inv = Q_SQUARED * sum_g_sq_e
                if d_squared_inv > 1e-10:
                    d_squared = 1.0 / d_squared_inv
                else:
                    d_squared = 1e10

                rd_squared = pre_rd[player] ** 2
                new_rd_squared = 1.0 / (1.0 / rd_squared + 1.0 / d_squared)
                new_rd = math.sqrt(new_rd_squared)
                new_rd = min(max(new_rd, min_rd), max_rd)

                rating_change = Q * new_rd_squared * sum_g_diff
                new_rating = player_rating + rating_change

                ratings[player] = new_rating
                rd[player] = new_rd
                last_played[player] = current_day

        total_updates += n_players

    return total_updates
