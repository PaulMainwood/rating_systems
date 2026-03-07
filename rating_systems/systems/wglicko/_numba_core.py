"""
Numba-accelerated core functions for Weighted Glicko rating system.

Weighted Glicko extends Glicko by allowing per-game weights w_j that scale
the Fisher information contribution of each game. The weighted update equations
(Proposition 2 from the paper) are:

    d_i^{-2} = q^2 * sum_j( w_j * g(RD_j)^2 * E_ij * (1 - E_ij) )
    RD'_i = (RD_i^{-2} + d_i^{-2})^{-1/2}
    r'_i = r_i + q * RD'^2_i * sum_j( w_j * g(RD_j) * (s_j - E_ij) )

With w_j = 1 for all games, this is identical to standard Glicko.

Native handicap support adds a per-game handicap h_j (in Elo rating points)
that shifts the expected score:

    E_ij = 1 / (1 + 10^(-g(RD_j) * (r_i - r_j + h_j) / 400))

where h_j > 0 favours player 1. When viewed from player 2's perspective
the handicap is negated.
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
    predict_proba_batch_at_day,
    predict_single_at_day,
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


# ---------------------------------------------------------------------------
# Handicap-aware variants
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, inline="always")
def _expected_score_h(rating: float, opp_rating: float, g_rd: float, h: float) -> float:
    """Expected score with handicap: E = 1/(1 + 10^(-g(RD) * (r - opp_r + h) / 400))."""
    exponent = -g_rd * (rating - opp_rating + h) / 400.0
    return 1.0 / (1.0 + math.pow(10.0, exponent))


@njit(cache=True, fastmath=True)
def update_ratings_batch_weighted_h(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    handicaps: np.ndarray,
    ratings: np.ndarray,
    rd: np.ndarray,
    last_played: np.ndarray,
    current_day: int,
    c: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """
    Update Weighted Glicko ratings with per-game handicaps.

    Handicaps are from player1's perspective (positive = player1 advantage,
    in Elo rating points). Returns number of players updated.
    """
    n_games = len(player1)
    if n_games == 0:
        return 0

    players_set = set()
    for i in range(n_games):
        players_set.add(player1[i])
        players_set.add(player2[i])

    active_players = np.array(list(players_set), dtype=np.int64)
    n_players = len(active_players)

    update_rd_for_inactivity(
        rd, last_played, active_players, current_day, c, min_rd, max_rd
    )

    pre_ratings = ratings.copy()
    pre_rd = rd.copy()

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
            h_j = 0.0

            if player1[j] == player:
                opp = player2[j]
                score = scores[j]
                w_j = weights[j]
                h_j = handicaps[j]
            elif player2[j] == player:
                opp = player1[j]
                score = 1.0 - scores[j]
                w_j = weights[j]
                h_j = -handicaps[j]  # negate for player 2

            if opp >= 0:
                opp_rating = pre_ratings[opp]
                opp_rd = pre_rd[opp]
                g = _g(opp_rd)
                e = _expected_score_h(player_rating, opp_rating, g, h_j)

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
    ratings: np.ndarray,
    rd: np.ndarray,
    last_played: np.ndarray,
    c: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """
    Fit Weighted Glicko ratings for ALL days with per-game handicaps.

    Returns total number of player-updates.
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

        for i in range(n_players):
            p = active_players[i]
            days_inactive = current_day - last_played[p]
            if days_inactive > 0:
                new_rd = math.sqrt(rd[p] * rd[p] + c_squared * days_inactive)
                rd[p] = min(max(new_rd, min_rd), max_rd)

        pre_ratings = ratings.copy()
        pre_rd = rd.copy()

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
                h_j = 0.0

                if p1_day[j] == player:
                    opp = p2_day[j]
                    score = scores_day[j]
                    w_j = weights_day[j]
                    h_j = handicaps_day[j]
                elif p2_day[j] == player:
                    opp = p1_day[j]
                    score = 1.0 - scores_day[j]
                    w_j = weights_day[j]
                    h_j = -handicaps_day[j]

                if opp >= 0:
                    opp_rating = pre_ratings[opp]
                    opp_rd = pre_rd[opp]
                    g = _g(opp_rd)
                    e = _expected_score_h(player_rating, opp_rating, g, h_j)

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


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch_h(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
    rd: np.ndarray,
    handicaps: np.ndarray,
) -> np.ndarray:
    """Predict win probabilities with per-game handicaps (parallelised)."""
    n_games = len(player1)
    proba = np.empty(n_games, dtype=np.float64)

    for i in prange(n_games):
        p1 = player1[i]
        p2 = player2[i]
        r1 = ratings[p1]
        r2 = ratings[p2]
        rd1 = rd[p1]
        rd2 = rd[p2]
        combined_rd = math.sqrt(rd1 * rd1 + rd2 * rd2)
        g = _g(combined_rd)
        exponent = -g * (r1 - r2 + handicaps[i]) / 400.0
        proba[i] = 1.0 / (1.0 + math.pow(10.0, exponent))

    return proba


@njit(cache=True, fastmath=True)
def predict_single_h(
    r1: float,
    rd1: float,
    r2: float,
    rd2: float,
    handicap: float,
) -> float:
    """Predict single win probability with handicap."""
    combined_rd = math.sqrt(rd1 * rd1 + rd2 * rd2)
    g = _g(combined_rd)
    exponent = -g * (r1 - r2 + handicap) / 400.0
    return 1.0 / (1.0 + math.pow(10.0, exponent))


@njit(cache=True, fastmath=True)
def walk_forward_predict_update(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    day_indices: np.ndarray,
    day_offsets: np.ndarray,
    weights: np.ndarray,
    handicaps: np.ndarray,
    ratings: np.ndarray,
    rd: np.ndarray,
    last_played: np.ndarray,
    c: float,
    min_rd: float,
    max_rd: float,
    n_train_days: int,
) -> np.ndarray:
    """Fused predict+update walk-forward in a single Numba call.

    For each post-training day:
    1. Predict using current ratings/RDs (before inactivity adjustment)
    2. Adjust RD for inactivity
    3. Batch-update ratings/RDs (simultaneous games within day)

    Returns predictions array (NaN for training period).
    """
    n_games = len(player1)
    predictions = np.full(n_games, np.nan, dtype=np.float64)
    c_squared = c * c
    n_days = len(day_offsets) - 1

    for day_idx in range(n_train_days, n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]
        current_day = day_indices[day_idx]
        n_day_games = end - start

        if n_day_games == 0:
            continue

        # Predict using current RD (before inactivity adjustment)
        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            r1 = ratings[p1]
            r2 = ratings[p2]
            rd1 = rd[p1]
            rd2 = rd[p2]
            combined_rd = math.sqrt(rd1 * rd1 + rd2 * rd2)
            g_val = _g(combined_rd)
            exponent = -g_val * (r1 - r2 + handicaps[i]) / 400.0
            predictions[i] = 1.0 / (1.0 + math.pow(10.0, exponent))

        # Find active players
        players_set = set()
        for i in range(start, end):
            players_set.add(player1[i])
            players_set.add(player2[i])
        active_players = np.array(list(players_set), dtype=np.int64)
        n_players = len(active_players)

        # RD inactivity growth
        for i in range(n_players):
            p = active_players[i]
            days_inactive = current_day - last_played[p]
            if days_inactive > 0:
                new_rd = math.sqrt(rd[p] * rd[p] + c_squared * days_inactive)
                rd[p] = min(max(new_rd, min_rd), max_rd)

        # Store pre-period values
        pre_ratings = ratings.copy()
        pre_rd = rd.copy()

        # Batch update: each active player
        for i in range(n_players):
            player = active_players[i]
            player_rating = pre_ratings[player]

            sum_g_sq_e = 0.0
            sum_g_diff = 0.0
            games_found = 0

            for j in range(start, end):
                opp = -1
                score = 0.0
                w_j = 1.0
                h_j = 0.0

                if player1[j] == player:
                    opp = player2[j]
                    score = scores[j]
                    w_j = weights[j]
                    h_j = handicaps[j]
                elif player2[j] == player:
                    opp = player1[j]
                    score = 1.0 - scores[j]
                    w_j = weights[j]
                    h_j = -handicaps[j]

                if opp >= 0:
                    opp_rating = pre_ratings[opp]
                    opp_rd = pre_rd[opp]
                    g_val = _g(opp_rd)
                    exponent = -g_val * (player_rating - opp_rating + h_j) / 400.0
                    e = 1.0 / (1.0 + math.pow(10.0, exponent))

                    sum_g_sq_e += w_j * g_val * g_val * e * (1.0 - e)
                    sum_g_diff += w_j * g_val * (score - e)
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
                ratings[player] = player_rating + rating_change
                rd[player] = new_rd
                last_played[player] = current_day

    return predictions
