"""Exact tennis scoring model: point → game → set → match probability.

Given p (probability server wins a point on their serve), computes the
exact probability of holding serve, winning a set, and winning a match
via the combinatorial scoring structure.

All functions are Numba-compiled for speed.

References:
  Newton & Keller (2005) — "Probability of winning at tennis"
  Barnett & Clarke (2005) — "Combining player statistics to predict outcomes"
"""

import math
import numpy as np
from numba import njit


@njit(cache=True)
def prob_hold_serve(p: float) -> float:
    """Probability that the server holds serve (wins the game).

    A game is first to 4 points with deuce at 3-3. At deuce, the server
    must win 2 consecutive points.

    Args:
        p: Probability server wins any single point on their serve.

    Returns:
        Probability of holding serve.
    """
    q = 1.0 - p

    # Ways to win without reaching deuce: win 4, lose 0/1/2
    # P(4-0) = p^4
    # P(4-1) = C(4,1) * p^4 * q^1
    # P(4-2) = C(5,2) * p^4 * q^2
    p_no_deuce = (p ** 4) * (1.0 + 4.0 * q + 10.0 * q * q)

    # Probability of reaching deuce (3-3): C(6,3) * p^3 * q^3 = 20 * p^3 * q^3
    p_deuce = 20.0 * (p ** 3) * (q ** 3)

    # At deuce, probability server wins 2 in a row before receiver does
    p_win_deuce = (p * p) / (p * p + q * q)

    return p_no_deuce + p_deuce * p_win_deuce


@njit(cache=True)
def prob_win_tiebreak(p_server: float, p_returner: float) -> float:
    """Probability P1 wins the tiebreak.

    Tiebreak is first to 7 points with serve alternating every 2 points
    (after the first point). P1 serves first.

    Simplified model: since serve alternates frequently, approximate as
    P1 wins each point with probability (p_server + (1 - p_returner)) / 2
    on average, then use the standard tiebreak recursion.

    More precisely: P1 serves on points 1, 4, 5, 8, 9, 12, 13, ...
    P2 serves on points 2, 3, 6, 7, 10, 11, ...

    For tractability, use the average point win probability.
    """
    # Average point win probability for P1 across serve/return points
    # In a 12-point tiebreak: P1 serves 6, returns 6 (approximately)
    p_avg = (p_server + (1.0 - p_returner)) / 2.0
    q_avg = 1.0 - p_avg

    # First to 7, deuce at 6-6 (same structure as a game but to 7)
    # P(win without deuce) = sum over i=0..5 of C(6+i, i) * p^7 * q^i
    p_no_deuce = 0.0
    for i in range(7):
        # C(6+i, i) * p^7 * q^i
        coeff = 1.0
        for j in range(1, i + 1):
            coeff = coeff * (6 + j) / j
        p_no_deuce += coeff * (p_avg ** 7) * (q_avg ** i)

    # P(reaching 6-6)
    coeff_66 = 1.0
    for j in range(1, 7):
        coeff_66 = coeff_66 * (6 + j) / j
    p_66 = coeff_66 * (p_avg ** 6) * (q_avg ** 6)

    # At 6-6, must win 2 in a row
    p_win_66 = (p_avg * p_avg) / (p_avg * p_avg + q_avg * q_avg)

    return p_no_deuce + p_66 * p_win_66


@njit(cache=True)
def prob_win_set(p_hold_1: float, p_hold_2: float,
                 p_tb_1: float) -> float:
    """Probability P1 wins a set.

    A set is first to 6 games, must win by 2. Tiebreak at 6-6.

    Args:
        p_hold_1: Probability P1 holds serve.
        p_hold_2: Probability P2 holds serve.
        p_tb_1: Probability P1 wins tiebreak.

    Returns:
        Probability P1 wins the set.
    """
    p_break_2 = 1.0 - p_hold_2  # P1 breaks P2's serve
    p_break_1 = 1.0 - p_hold_1  # P2 breaks P1's serve

    # Each "round" of 2 games: P1 serves, then P2 serves
    # P1 wins both: p_hold_1 * p_break_2
    # P1 wins 1: p_hold_1 * p_hold_2 + p_break_1 * p_break_2
    # P1 wins 0: p_break_1 * p_hold_2

    # Use dynamic programming on game scores (i, j) where i = P1 games, j = P2 games
    # State: prob[i][j] = probability of reaching score (i, j)
    # We need to track who serves at each game score

    # Serve alternates: P1 serves at games 0, 2, 4, ... (when total games played is even
    # and P1 served first — but actually serve depends on who started)
    # Simplification: P1 serves first. At total games (i+j):
    #   P1 serves if (i+j) is even, P2 serves if (i+j) is odd

    # prob[i][j] = probability of reaching exactly this score
    prob = np.zeros((8, 8), dtype=np.float64)
    prob[0][0] = 1.0

    for total_games in range(12):  # max 12 games before tiebreak (6-6)
        for i in range(min(total_games + 1, 7)):
            j = total_games - i
            if j < 0 or j > 6 or i > 6:
                continue
            if prob[i][j] == 0.0:
                continue
            # Skip if already won (6-x where x <= 4, or x-6 where x <= 4)
            if (i == 6 and j <= 4) or (j == 6 and i <= 4):
                continue
            if (i == 7) or (j == 7):
                continue

            p_current = prob[i][j]
            # Who serves this game?
            if (i + j) % 2 == 0:
                # P1 serves
                p_win = p_hold_1
            else:
                # P2 serves — P1 wins by breaking
                p_win = p_break_2

            # P1 wins this game
            if i + 1 <= 7:
                prob[i + 1][j] += p_current * p_win
            # P2 wins this game
            if j + 1 <= 7:
                prob[i][j + 1] += p_current * (1.0 - p_win)

    # Sum probabilities of P1 winning the set
    p1_wins = 0.0

    # P1 wins 6-0 through 6-4
    for j in range(5):
        p1_wins += prob[6][j]

    # P1 wins 7-5 (reached 5-5, then 6-5, then 7-5)
    p1_wins += prob[7][5]

    # P1 wins tiebreak (reached 6-6)
    p1_wins += prob[6][6] * p_tb_1

    # Also handle 7-5 for P2 → not needed, we're computing P1 winning

    return p1_wins


@njit(cache=True)
def prob_win_match(p_serve_1: float, p_serve_2: float,
                   best_of: int = 3) -> float:
    """Probability P1 wins the match.

    Args:
        p_serve_1: Probability P1 wins a point on P1's serve.
        p_serve_2: Probability P2 wins a point on P2's serve.
        best_of: 3 or 5 sets.

    Returns:
        Probability P1 wins the match.
    """
    # Compute game-level probabilities
    p_hold_1 = prob_hold_serve(p_serve_1)
    p_hold_2 = prob_hold_serve(p_serve_2)

    # Tiebreak probability (P1 perspective)
    p_tb_1 = prob_win_tiebreak(p_serve_1, p_serve_2)

    # Set probability
    p_set = prob_win_set(p_hold_1, p_hold_2, p_tb_1)

    if best_of == 3:
        # Win 2 out of 3: win in 2 or win in 3
        return (p_set * p_set +
                2.0 * p_set * p_set * (1.0 - p_set))
    else:
        # Win 3 out of 5
        q_set = 1.0 - p_set
        # Win in 3, 4, or 5 sets
        return (p_set ** 3 +                            # 3-0
                3.0 * (p_set ** 3) * q_set +            # 3-1
                6.0 * (p_set ** 3) * (q_set ** 2))      # 3-2


@njit(cache=True)
def prob_win_match_batch(p_serve_1: np.ndarray, p_serve_2: np.ndarray,
                         best_of: np.ndarray) -> np.ndarray:
    """Batch version of prob_win_match."""
    n = len(p_serve_1)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = prob_win_match(p_serve_1[i], p_serve_2[i], best_of[i])
    return result
