"""
Numba-accelerated core functions for Stephenson rating system.

The Stephenson system extends Glicko with:
1. Opponent's RD directly weights the expected score calculation
2. hval parameter adds uncertainty proportional to games played
3. bval bonus rewards frequent play (creates inflation)
4. lambda neighbourhood parameter shrinks ratings toward opponents
5. gamma first-player advantage parameter

Reference:
- Stephenson & Sonas (2012), PlayerRatings R package
- https://cran.r-project.org/package=PlayerRatings
"""

import math
import numpy as np
from numba import njit, prange


# Constants
Q = math.log(10) / 400.0  # ~0.00575646273
Q_SQUARED = Q ** 2


@njit(cache=True, fastmath=True, inline="always")
def _expected_score_steph(
    player_rating: float,
    opp_rating: float,
    opp_rd: float,
    gamma: float,
) -> float:
    """
    Stephenson expected score calculation.

    Unlike Glicko where g(RD) scales the rating difference,
    Stephenson uses the opponent's RD directly:

    E = 1 / (1 + 10^(RD_opp * (R_opp - R_player - gamma) / 400))

    Args:
        player_rating: Player's rating
        opp_rating: Opponent's rating
        opp_rd: Opponent's rating deviation
        gamma: First-player advantage (added to player's effective rating)

    Returns:
        Expected score (probability of winning)
    """
    exponent = opp_rd * (opp_rating - player_rating - gamma) / 400.0
    return 1.0 / (1.0 + math.pow(10.0, exponent))


@njit(cache=True, fastmath=True)
def update_rd_for_inactivity_steph(
    rd: np.ndarray,
    last_played: np.ndarray,
    active_players: np.ndarray,
    current_day: int,
    cval: float,
    min_rd: float,
    max_rd: float,
) -> None:
    """
    Update RD for inactivity for selected players (in-place).

    new_RD = sqrt(old_RD² + c² * days_inactive)
    """
    c_squared = cval * cval
    for i in range(len(active_players)):
        p = active_players[i]
        days_inactive = current_day - last_played[p]
        if days_inactive > 0:
            new_rd = math.sqrt(rd[p] * rd[p] + c_squared * days_inactive)
            rd[p] = min(max(new_rd, min_rd), max_rd)


@njit(cache=True, fastmath=True)
def process_player_games_steph(
    player: int,
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    pre_ratings: np.ndarray,
    pre_rd: np.ndarray,
    bval: float,
    gamma: float,
) -> tuple:
    """
    Process all games for a single player and compute Stephenson aggregates.

    Returns: (dscore, dval, l1t, n_games)
    where:
        dscore = sum of RD_opp * (actual_score + bval - expected_score)
        dval = sum of q² * RD_opp² * E * (1-E)
        l1t = sum of (R_opp - R_player) rating differences
        n_games = number of games played
    """
    n_total_games = len(player1)
    player_rating = pre_ratings[player]

    dscore = 0.0
    dval = 0.0
    l1t = 0.0
    n_games = 0

    for i in range(n_total_games):
        opp = -1
        score = 0.0
        player_gamma = 0.0  # Effective gamma for this game

        if player1[i] == player:
            opp = player2[i]
            score = scores[i]
            player_gamma = gamma  # Player is "white" (player 1)
        elif player2[i] == player:
            opp = player1[i]
            score = 1.0 - scores[i]
            player_gamma = -gamma  # Player is "black" (player 2)

        if opp >= 0:
            opp_rating = pre_ratings[opp]
            opp_rd = pre_rd[opp]

            # Expected score (Stephenson formula)
            e = _expected_score_steph(player_rating, opp_rating, opp_rd, player_gamma)

            # Actual score with bonus
            actual = score + bval

            # Accumulate sums
            dscore += opp_rd * (actual - e)
            dval += Q_SQUARED * opp_rd * opp_rd * e * (1.0 - e)
            l1t += opp_rating - player_rating
            n_games += 1

    return dscore, dval, l1t, n_games


@njit(cache=True, fastmath=True)
def compute_player_update_steph(
    player_rating: float,
    player_rd: float,
    dscore: float,
    dval: float,
    l1t: float,
    n_games: int,
    hval: float,
    lambda_param: float,
    min_rd: float,
    max_rd: float,
) -> tuple:
    """
    Compute new rating and RD for a player using Stephenson formulas.

    New RD: 1 / (1/(old_RD + n_games * h²) + dval)
    New rating: old_R + new_RD * q * dscore + (lambda/100) * l1t / n_games

    Returns: (new_rating, new_rd)
    """
    if n_games == 0:
        return player_rating, player_rd

    # Update RD with hval term (additional uncertainty per game)
    rd_with_h = player_rd + n_games * hval * hval

    # New RD (inverse sum formula)
    if dval > 1e-10:
        new_rd = 1.0 / (1.0 / rd_with_h + dval)
    else:
        new_rd = rd_with_h

    new_rd = min(max(new_rd, min_rd), max_rd)

    # New rating with neighbourhood term
    rating_change = new_rd * Q * dscore
    neighbourhood_term = (lambda_param / 100.0) * l1t / n_games
    new_rating = player_rating + rating_change + neighbourhood_term

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
    cval: float,
    hval: float,
    bval: float,
    lambda_param: float,
    gamma: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """
    Update Stephenson ratings for a rating period.

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
    update_rd_for_inactivity_steph(
        rd, last_played, active_players, current_day, cval, min_rd, max_rd
    )

    # Store pre-period values
    pre_ratings = ratings.copy()
    pre_rd = rd.copy()

    # Process each active player
    for i in range(n_players):
        player = active_players[i]

        dscore, dval, l1t, games_found = process_player_games_steph(
            player, player1, player2, scores, pre_ratings, pre_rd, bval, gamma
        )

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


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
    rd: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Predict win probabilities for a batch of games (parallel).

    Uses Stephenson expected score formula.
    """
    n_games = len(player1)
    proba = np.empty(n_games, dtype=np.float64)

    for i in prange(n_games):
        p1 = player1[i]
        p2 = player2[i]

        r1 = ratings[p1]
        r2 = ratings[p2]
        rd2 = rd[p2]  # Use opponent's RD for player 1's expected score

        proba[i] = _expected_score_steph(r1, r2, rd2, gamma)

    return proba


@njit(cache=True, fastmath=True)
def predict_single(
    r1: float,
    rd1: float,
    r2: float,
    rd2: float,
    gamma: float,
) -> float:
    """
    Predict win probability for a single matchup.

    Uses player 2's RD in the expected score calculation.
    """
    return _expected_score_steph(r1, r2, rd2, gamma)


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
    cval: float,
    hval: float,
    bval: float,
    lambda_param: float,
    gamma: float,
    min_rd: float,
    max_rd: float,
) -> int:
    """
    Fit Stephenson ratings for ALL days in a single Numba call.

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
        cval: RD increase per period of inactivity
        hval: Additional RD increase per game
        bval: Bonus for playing (added to actual score)
        lambda_param: Neighbourhood parameter (shrinks toward opponents)
        gamma: First-player advantage
        min_rd: Minimum RD
        max_rd: Maximum RD

    Returns:
        Total number of player-updates
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

            # Accumulate Stephenson stats from all games
            dscore = 0.0
            dval = 0.0
            l1t = 0.0
            games_found = 0

            for j in range(n_games):
                opp = -1
                score = 0.0
                player_gamma = 0.0

                if p1_day[j] == player:
                    opp = p2_day[j]
                    score = scores_day[j]
                    player_gamma = gamma
                elif p2_day[j] == player:
                    opp = p1_day[j]
                    score = 1.0 - scores_day[j]
                    player_gamma = -gamma

                if opp >= 0:
                    opp_rating = pre_ratings[opp]
                    opp_rd = pre_rd[opp]

                    # Expected score (Stephenson formula)
                    e = _expected_score_steph(player_rating, opp_rating, opp_rd, player_gamma)

                    # Actual score with bonus
                    actual = score + bval

                    # Accumulate sums
                    dscore += opp_rd * (actual - e)
                    dval += Q_SQUARED * opp_rd * opp_rd * e * (1.0 - e)
                    l1t += opp_rating - player_rating
                    games_found += 1

            if games_found > 0:
                # Update RD with hval term
                rd_with_h = pre_rd[player] + games_found * hval * hval

                # New RD
                if dval > 1e-10:
                    new_rd = 1.0 / (1.0 / rd_with_h + dval)
                else:
                    new_rd = rd_with_h
                new_rd = min(max(new_rd, min_rd), max_rd)

                # New rating with neighbourhood term
                rating_change = new_rd * Q * dscore
                neighbourhood_term = (lambda_param / 100.0) * l1t / games_found
                new_rating = player_rating + rating_change + neighbourhood_term

                ratings[player] = new_rating
                rd[player] = new_rd
                last_played[player] = current_day

        total_updates += n_players

    return total_updates
