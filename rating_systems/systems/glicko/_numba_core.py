"""
Numba-accelerated core functions for Glicko rating system.

Glicko processes all games in a rating period simultaneously,
using pre-period ratings. This allows more vectorization than Elo.
"""

import math
import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList


# Constants
PI_SQUARED = math.pi ** 2
Q = math.log(10) / 400.0  # ~0.00575646273
Q_SQUARED = Q ** 2
THREE_Q_SQUARED_OVER_PI_SQUARED = 3.0 * Q_SQUARED / PI_SQUARED


@njit(cache=True, fastmath=True, inline="always")
def _g(rd: float) -> float:
    """Calculate g(RD) function."""
    return 1.0 / math.sqrt(1.0 + THREE_Q_SQUARED_OVER_PI_SQUARED * rd * rd)


@njit(cache=True, fastmath=True, inline="always")
def _expected_score(rating: float, opp_rating: float, g_rd: float) -> float:
    """Calculate expected score with precomputed g(RD)."""
    exponent = -g_rd * (rating - opp_rating) / 400.0
    return 1.0 / (1.0 + math.pow(10.0, exponent))


@njit(cache=True, fastmath=True)
def update_rd_for_inactivity(
    rd: np.ndarray,
    last_played: np.ndarray,
    active_players: np.ndarray,
    current_day: int,
    c: float,
    min_rd: float,
    max_rd: float,
) -> None:
    """Update RD for inactivity for selected players (in-place)."""
    c_squared = c * c
    for i in range(len(active_players)):
        p = active_players[i]
        days_inactive = current_day - last_played[p]
        if days_inactive > 0:
            new_rd = math.sqrt(rd[p] * rd[p] + c_squared * days_inactive)
            rd[p] = min(max(new_rd, min_rd), max_rd)


@njit(cache=True, fastmath=True)
def process_player_games(
    player: int,
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    pre_ratings: np.ndarray,
    pre_rd: np.ndarray,
) -> tuple:
    """
    Extract and process games for a single player.

    Returns: (sum_g_sq_e, sum_g_diff, n_games)
    where these are the aggregates needed for Glicko update.
    """
    n_games = len(player1)
    player_rating = pre_ratings[player]

    sum_g_sq_e = 0.0  # Sum of g^2 * E * (1-E) for d^2 calculation
    sum_g_diff = 0.0  # Sum of g * (score - E) for rating update
    games_found = 0

    for i in range(n_games):
        opp = -1
        score = 0.0

        if player1[i] == player:
            opp = player2[i]
            score = scores[i]
        elif player2[i] == player:
            opp = player1[i]
            score = 1.0 - scores[i]

        if opp >= 0:
            opp_rating = pre_ratings[opp]
            opp_rd = pre_rd[opp]
            g = _g(opp_rd)
            e = _expected_score(player_rating, opp_rating, g)

            sum_g_sq_e += g * g * e * (1.0 - e)
            sum_g_diff += g * (score - e)
            games_found += 1

    return sum_g_sq_e, sum_g_diff, games_found


@njit(cache=True, fastmath=True)
def compute_player_update(
    player_rating: float,
    player_rd: float,
    sum_g_sq_e: float,
    sum_g_diff: float,
    min_rd: float,
    max_rd: float,
) -> tuple:
    """
    Compute new rating and RD for a player.

    Returns: (new_rating, new_rd)
    """
    # d^2 = 1 / (q^2 * sum(g^2 * E * (1-E)))
    d_squared_inv = Q_SQUARED * sum_g_sq_e

    if d_squared_inv > 1e-10:
        d_squared = 1.0 / d_squared_inv
    else:
        d_squared = 1e10

    # New RD
    rd_squared = player_rd * player_rd
    new_rd_squared = 1.0 / (1.0 / rd_squared + 1.0 / d_squared)
    new_rd = math.sqrt(new_rd_squared)
    new_rd = min(max(new_rd, min_rd), max_rd)

    # New rating
    rating_change = Q * new_rd_squared * sum_g_diff
    new_rating = player_rating + rating_change

    return new_rating, new_rd


@njit(cache=True, fastmath=True)
def update_ratings_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    ratings: np.ndarray,
    rd: np.ndarray,
    last_played: np.ndarray,
    current_day: int,
    c: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """
    Update Glicko ratings for a rating period.

    All games are treated as simultaneous (use pre-period ratings).

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

        sum_g_sq_e, sum_g_diff, games_found = process_player_games(
            player, player1, player2, scores, pre_ratings, pre_rd
        )

        if games_found > 0:
            new_rating, new_rd = compute_player_update(
                pre_ratings[player],
                pre_rd[player],
                sum_g_sq_e,
                sum_g_diff,
                min_rd,
                max_rd,
            )

            ratings[player] = new_rating
            rd[player] = new_rd
            last_played[player] = current_day

    return n_players


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
    rd: np.ndarray,
) -> np.ndarray:
    """
    Predict win probabilities for a batch of games (parallel).

    Uses combined RD of both players.
    """
    n_games = len(player1)
    proba = np.empty(n_games, dtype=np.float64)

    for i in prange(n_games):
        p1 = player1[i]
        p2 = player2[i]

        r1 = ratings[p1]
        r2 = ratings[p2]
        rd1 = rd[p1]
        rd2 = rd[p2]

        # Combined RD
        combined_rd = math.sqrt(rd1 * rd1 + rd2 * rd2)
        g = _g(combined_rd)

        exponent = -g * (r1 - r2) / 400.0
        proba[i] = 1.0 / (1.0 + math.pow(10.0, exponent))

    return proba


@njit(cache=True, fastmath=True)
def predict_single(
    r1: float,
    rd1: float,
    r2: float,
    rd2: float,
) -> float:
    """Predict win probability for a single matchup."""
    combined_rd = math.sqrt(rd1 * rd1 + rd2 * rd2)
    g = _g(combined_rd)
    exponent = -g * (r1 - r2) / 400.0
    return 1.0 / (1.0 + math.pow(10.0, exponent))


@njit(cache=True)
def get_top_n_indices(ratings: np.ndarray, n: int) -> np.ndarray:
    """Get indices of top N rated players."""
    if n >= len(ratings):
        return np.argsort(ratings)[::-1]
    top_indices = np.argpartition(ratings, -n)[-n:]
    sorted_order = np.argsort(ratings[top_indices])[::-1]
    return top_indices[sorted_order]


@njit(cache=True, fastmath=True)
def fit_all_days(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
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
    Fit Glicko ratings for ALL days in a single Numba call.

    This avoids Python iteration overhead entirely.
    Each day is processed as a rating period (games simultaneous within day).

    Args:
        player1: All player1 IDs (sorted by day)
        player2: All player2 IDs (sorted by day)
        scores: All scores (sorted by day)
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

        # Update RD for inactivity
        for i in range(n_players):
            p = active_players[i]
            days_inactive = current_day - last_played[p]
            if days_inactive > 0:
                new_rd = math.sqrt(rd[p] * rd[p] + c_squared * days_inactive)
                rd[p] = min(max(new_rd, min_rd), max_rd)

        # Store pre-period values for simultaneous processing
        pre_ratings = ratings.copy()
        pre_rd = rd.copy()

        # Process each active player
        for i in range(n_players):
            player = active_players[i]
            player_rating = pre_ratings[player]

            # Accumulate stats from all games
            sum_g_sq_e = 0.0
            sum_g_diff = 0.0
            games_found = 0

            for j in range(n_games):
                opp = -1
                score = 0.0

                if p1_day[j] == player:
                    opp = p2_day[j]
                    score = scores_day[j]
                elif p2_day[j] == player:
                    opp = p1_day[j]
                    score = 1.0 - scores_day[j]

                if opp >= 0:
                    opp_rating = pre_ratings[opp]
                    opp_rd = pre_rd[opp]
                    g = _g(opp_rd)
                    e = _expected_score(player_rating, opp_rating, g)

                    sum_g_sq_e += g * g * e * (1.0 - e)
                    sum_g_diff += g * (score - e)
                    games_found += 1

            if games_found > 0:
                # Compute update
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
