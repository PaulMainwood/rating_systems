"""
Numba-accelerated core functions for Weighted Stephenson rating system.

Weighted Stephenson extends Stephenson by allowing per-game weights w_j
that scale the Fisher information contribution of each game, following
the same pattern as WGlicko (Proposition 2).

The weighted update equations are:
    dscore_i = sum_j( w_j * g(RD_j) * (s_j + bval - E_ij) )
    dval_i = q^2 * sum_j( w_j * g(RD_j)^2 * E_ij * (1 - E_ij) )

With w_j = 1 for all games, this is identical to standard Stephenson.
"""

import math
import numpy as np
from numba import njit, prange

# Reuse constants and utility functions from Stephenson
from ..stephenson._numba_core import (
    Q,
    Q_SQUARED,
    _g_rd,
    _expected_score_steph,
    update_rd_for_inactivity_steph,
    predict_proba_batch,
    predict_single,
    predict_proba_batch_at_day,
    predict_single_at_day,
    get_top_n_indices,
    compute_player_update_steph,
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
    cval: float,
    hval: float,
    bval: float,
    lambda_param: float,
    gamma: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """
    Update Weighted Stephenson ratings for a rating period.

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
    update_rd_for_inactivity_steph(
        rd, last_played, active_players, current_day, cval, min_rd, max_rd
    )

    # Store pre-period values
    pre_ratings = ratings.copy()
    pre_rd = rd.copy()

    # Process each active player
    for i in range(n_players):
        player = active_players[i]
        player_rating = pre_ratings[player]

        dscore = 0.0
        dval = 0.0
        l1t = 0.0
        games_found = 0

        for j in range(n_games):
            opp = -1
            score = 0.0
            player_gamma = 0.0
            w_j = 1.0

            if player1[j] == player:
                opp = player2[j]
                score = scores[j]
                player_gamma = gamma
                w_j = weights[j]
            elif player2[j] == player:
                opp = player1[j]
                score = 1.0 - scores[j]
                player_gamma = -gamma
                w_j = weights[j]

            if opp >= 0:
                opp_rating = pre_ratings[opp]
                opp_rd = pre_rd[opp]
                g_opp = _g_rd(opp_rd)

                e = _expected_score_steph(player_rating, opp_rating, opp_rd, player_gamma)
                actual = score + bval

                dscore += w_j * g_opp * (actual - e)
                dval += w_j * Q_SQUARED * g_opp * g_opp * e * (1.0 - e)
                l1t += opp_rating - player_rating
                games_found += 1

        if games_found > 0:
            new_rating, new_rd = compute_player_update_steph(
                pre_ratings[player],
                pre_rd[player],
                dscore,
                dval,
                l1t,
                games_found,
                hval,
                lambda_param,
                min_rd,
                max_rd,
            )

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
    cval: float,
    hval: float,
    bval: float,
    lambda_param: float,
    gamma: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """
    Fit Weighted Stephenson ratings for ALL days in a single Numba call.

    Each day is a rating period with simultaneous games.
    Per-game weights w_j scale the information contribution.

    Returns total number of player-updates.
    """
    n_days = len(day_offsets) - 1
    total_updates = 0
    c_squared = cval * cval

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

            dscore = 0.0
            dval = 0.0
            l1t = 0.0
            games_found = 0

            for j in range(n_games):
                opp = -1
                score = 0.0
                player_gamma = 0.0
                w_j = 1.0

                if p1_day[j] == player:
                    opp = p2_day[j]
                    score = scores_day[j]
                    player_gamma = gamma
                    w_j = weights_day[j]
                elif p2_day[j] == player:
                    opp = p1_day[j]
                    score = 1.0 - scores_day[j]
                    player_gamma = -gamma
                    w_j = weights_day[j]

                if opp >= 0:
                    opp_rating = pre_ratings[opp]
                    opp_rd = pre_rd[opp]
                    g_opp = _g_rd(opp_rd)

                    e = _expected_score_steph(player_rating, opp_rating, opp_rd, player_gamma)
                    actual = score + bval

                    dscore += w_j * g_opp * (actual - e)
                    dval += w_j * Q_SQUARED * g_opp * g_opp * e * (1.0 - e)
                    l1t += opp_rating - player_rating
                    games_found += 1

            if games_found > 0:
                # Update RD with hval term
                player_rd = pre_rd[player]
                rd_with_h = math.sqrt(player_rd * player_rd + games_found * hval * hval)

                # New RD using Glicko formula
                rd_squared = rd_with_h * rd_with_h
                if dval > 1e-10:
                    new_rd_squared = 1.0 / (1.0 / rd_squared + dval)
                else:
                    new_rd_squared = rd_squared
                new_rd = math.sqrt(new_rd_squared)
                new_rd = min(max(new_rd, min_rd), max_rd)

                # New rating
                rating_change = Q * new_rd_squared * dscore
                neighbourhood_term = (lambda_param / 100.0) * l1t / games_found
                new_rating = player_rating + rating_change + neighbourhood_term

                ratings[player] = new_rating
                rd[player] = new_rd
                last_played[player] = current_day

        total_updates += n_players

    return total_updates


# ---------------------------------------------------------------------------
# Handicap-aware variants
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, inline="always")
def _expected_score_steph_h(
    player_rating: float,
    opp_rating: float,
    opp_rd: float,
    gamma: float,
    h: float,
) -> float:
    """Stephenson expected score with handicap.

    E = 1 / (1 + 10^(g(RD_opp) * (R_opp - R_player - gamma - h) / 400))

    where h > 0 favours the player (shifts effective rating up).
    """
    g = _g_rd(opp_rd)
    exponent = g * (opp_rating - player_rating - gamma - h) / 400.0
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
    cval: float,
    hval: float,
    bval: float,
    lambda_param: float,
    gamma: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """Update Weighted Stephenson ratings with per-game handicaps.

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

    update_rd_for_inactivity_steph(
        rd, last_played, active_players, current_day, cval, min_rd, max_rd
    )

    pre_ratings = ratings.copy()
    pre_rd = rd.copy()

    for i in range(n_players):
        player = active_players[i]
        player_rating = pre_ratings[player]

        dscore = 0.0
        dval = 0.0
        l1t = 0.0
        games_found = 0

        for j in range(n_games):
            opp = -1
            score = 0.0
            player_gamma = 0.0
            w_j = 1.0
            h_j = 0.0

            if player1[j] == player:
                opp = player2[j]
                score = scores[j]
                player_gamma = gamma
                w_j = weights[j]
                h_j = handicaps[j]
            elif player2[j] == player:
                opp = player1[j]
                score = 1.0 - scores[j]
                player_gamma = -gamma
                w_j = weights[j]
                h_j = -handicaps[j]

            if opp >= 0:
                opp_rating = pre_ratings[opp]
                opp_rd = pre_rd[opp]
                g_opp = _g_rd(opp_rd)

                e = _expected_score_steph_h(player_rating, opp_rating, opp_rd, player_gamma, h_j)
                actual = score + bval

                dscore += w_j * g_opp * (actual - e)
                dval += w_j * Q_SQUARED * g_opp * g_opp * e * (1.0 - e)
                l1t += opp_rating - player_rating
                games_found += 1

        if games_found > 0:
            new_rating, new_rd = compute_player_update_steph(
                pre_ratings[player],
                pre_rd[player],
                dscore,
                dval,
                l1t,
                games_found,
                hval,
                lambda_param,
                min_rd,
                max_rd,
            )

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
    cval: float,
    hval: float,
    bval: float,
    lambda_param: float,
    gamma: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """Fit Weighted Stephenson ratings for ALL days with per-game handicaps.

    Returns total number of player-updates.
    """
    n_days = len(day_offsets) - 1
    total_updates = 0
    c_squared = cval * cval

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

            dscore = 0.0
            dval = 0.0
            l1t = 0.0
            games_found = 0

            for j in range(n_games):
                opp = -1
                score = 0.0
                player_gamma = 0.0
                w_j = 1.0
                h_j = 0.0

                if p1_day[j] == player:
                    opp = p2_day[j]
                    score = scores_day[j]
                    player_gamma = gamma
                    w_j = weights_day[j]
                    h_j = handicaps_day[j]
                elif p2_day[j] == player:
                    opp = p1_day[j]
                    score = 1.0 - scores_day[j]
                    player_gamma = -gamma
                    w_j = weights_day[j]
                    h_j = -handicaps_day[j]

                if opp >= 0:
                    opp_rating = pre_ratings[opp]
                    opp_rd = pre_rd[opp]
                    g_opp = _g_rd(opp_rd)

                    e = _expected_score_steph_h(player_rating, opp_rating, opp_rd, player_gamma, h_j)
                    actual = score + bval

                    dscore += w_j * g_opp * (actual - e)
                    dval += w_j * Q_SQUARED * g_opp * g_opp * e * (1.0 - e)
                    l1t += opp_rating - player_rating
                    games_found += 1

            if games_found > 0:
                player_rd = pre_rd[player]
                rd_with_h = math.sqrt(player_rd * player_rd + games_found * hval * hval)

                rd_squared = rd_with_h * rd_with_h
                if dval > 1e-10:
                    new_rd_squared = 1.0 / (1.0 / rd_squared + dval)
                else:
                    new_rd_squared = rd_squared
                new_rd = math.sqrt(new_rd_squared)
                new_rd = min(max(new_rd, min_rd), max_rd)

                rating_change = Q * new_rd_squared * dscore
                neighbourhood_term = (lambda_param / 100.0) * l1t / games_found
                new_rating = player_rating + rating_change + neighbourhood_term

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
    gamma: float,
    handicaps: np.ndarray,
) -> np.ndarray:
    """Predict win probabilities with per-game handicaps (parallelised)."""
    n_games = len(player1)
    proba = np.empty(n_games, dtype=np.float64)

    for i in prange(n_games):
        p1 = player1[i]
        p2 = player2[i]
        proba[i] = _expected_score_steph_h(
            ratings[p1], ratings[p2], rd[p2], gamma, handicaps[i],
        )

    return proba


@njit(cache=True, fastmath=True)
def predict_single_h(
    r1: float,
    rd1: float,
    r2: float,
    rd2: float,
    gamma: float,
    handicap: float,
) -> float:
    """Predict single win probability with handicap."""
    return _expected_score_steph_h(r1, r2, rd2, gamma, handicap)
