"""
Numba-accelerated core functions for G-Elo rating system.

G-Elo (Generalized Elo) extends standard Elo to use ordinal margin-of-victory
via a cumulative logistic model. The update has the SAME FORM as standard Elo
but with a redefined generalized score and expected score.

For tennis with J ordinal categories (e.g. J=4: {0-2, 1-2, 2-1, 2-0}):
    P(MOV ≤ j | Δ) = σ(θ_j - Δ)   for j = 1, ..., J-1

The generalized score for outcome in category m is:
    S_gen = Σ_{j=1}^{J-1} I(m > j)  where I is the indicator function
          = m - 1  (i.e., the number of thresholds below the outcome)

The generalized expected score is:
    E_gen = Σ_{j=1}^{J-1} P(MOV > j | Δ) = Σ_{j=1}^{J-1} (1 - σ(θ_j - Δ))

Update: Δr = K * (S_gen - E_gen)

Win probability for prediction:
    P(player 1 wins) = P(MOV > J/2 | Δ) = 1 - σ(θ_{J/2} - Δ)
    which for symmetric thresholds = σ(Δ) as in standard Elo.

Reference: Szczecinski, "G-Elo: Generalization of the Elo Algorithm by
Modeling the Discretized Margin of Victory", JQAS 2022.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, inline="always")
def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


@njit(cache=True, fastmath=True)
def _expected_gen_score(delta: float, thresholds: np.ndarray,
                        log10_scale: float) -> float:
    """Compute the generalized expected score.

    E_gen = Σ_{j=1}^{J-1} (1 - σ(θ_j - Δ))
          = Σ_{j=1}^{J-1} σ(Δ - θ_j)

    where Δ is the (scaled) rating difference.
    """
    e_gen = 0.0
    for j in range(len(thresholds)):
        e_gen += _sigmoid((delta - thresholds[j]) * log10_scale)
    return e_gen


@njit(cache=True, fastmath=True)
def update_ratings_sequential(
    player1: np.ndarray,
    player2: np.ndarray,
    margins: np.ndarray,
    ratings: np.ndarray,
    k_factor: float,
    scale: float,
    thresholds: np.ndarray,
) -> None:
    """
    Update G-Elo ratings for a batch of games (sequential within batch).

    Args:
        player1: (N,) Player 1 IDs
        player2: (N,) Player 2 IDs
        margins: (N,) Ordinal margin categories for P1 (0-indexed).
                 For Bo3 tennis with J=4: 0=lost 0-2, 1=lost 1-2, 2=won 2-1, 3=won 2-0
                 For Bo5 tennis with J=6: 0=lost 0-3, ..., 5=won 3-0
        ratings: (n_players,) Ratings array (modified in-place)
        k_factor: K-factor
        scale: Logistic scale
        thresholds: (J-1,) Ordinal thresholds
    """
    n_games = len(player1)
    log10_scale = np.log(10.0) / scale

    for i in range(n_games):
        p1 = player1[i]
        p2 = player2[i]
        m = margins[i]

        r1 = ratings[p1]
        r2 = ratings[p2]
        delta = r1 - r2

        # Generalized actual score: number of thresholds below the outcome
        s_gen = float(m)

        # Generalized expected score
        e_gen = _expected_gen_score(delta, thresholds, log10_scale)

        # Update
        update = k_factor * (s_gen - e_gen)
        ratings[p1] = r1 + update
        ratings[p2] = r2 - update


@njit(cache=True, fastmath=True)
def fit_all_days(
    player1: np.ndarray,
    player2: np.ndarray,
    margins: np.ndarray,
    day_offsets: np.ndarray,
    ratings: np.ndarray,
    k_factor: float,
    scale: float,
    thresholds: np.ndarray,
) -> None:
    """Fit G-Elo ratings for ALL days in a single Numba call."""
    n_days = len(day_offsets) - 1
    log10_scale = np.log(10.0) / scale

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            m = margins[i]

            r1 = ratings[p1]
            r2 = ratings[p2]
            delta = r1 - r2

            s_gen = float(m)
            e_gen = _expected_gen_score(delta, thresholds, log10_scale)

            update = k_factor * (s_gen - e_gen)
            ratings[p1] = r1 + update
            ratings[p2] = r2 - update


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Predict P(player1 wins) for a batch.

    For symmetric thresholds (θ_j = -θ_{J-j}), the win probability
    reduces to the standard Elo sigmoid. We use this directly.
    """
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)
    log10_scale = np.log(10.0) / scale

    for i in prange(n):
        r1 = ratings[player1[i]]
        r2 = ratings[player2[i]]
        proba[i] = _sigmoid((r1 - r2) * log10_scale)

    return proba


@njit(cache=True, fastmath=True)
def predict_single(
    p1_rating: float,
    p2_rating: float,
    scale: float,
) -> float:
    """Predict P(player1 wins) for a single matchup."""
    return _sigmoid((p1_rating - p2_rating) * np.log(10.0) / scale)


@njit(cache=True, fastmath=True)
def fit_all_days_weighted(
    player1: np.ndarray,
    player2: np.ndarray,
    margins: np.ndarray,
    weights: np.ndarray,
    handicaps: np.ndarray,
    day_offsets: np.ndarray,
    ratings: np.ndarray,
    k_factor: float,
    scale: float,
    thresholds: np.ndarray,
) -> None:
    """Fit weighted G-Elo ratings for ALL days in a single Numba call."""
    n_days = len(day_offsets) - 1
    log10_scale = np.log(10.0) / scale

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            m = margins[i]
            w = weights[i]
            h = handicaps[i]

            r1 = ratings[p1]
            r2 = ratings[p2]
            delta = r1 - r2 + h

            s_gen = float(m)
            e_gen = _expected_gen_score(delta, thresholds, log10_scale)

            update = k_factor * w * (s_gen - e_gen)
            ratings[p1] = r1 + update
            ratings[p2] = r2 - update


@njit(cache=True, fastmath=True)
def update_ratings_weighted_sequential(
    player1: np.ndarray,
    player2: np.ndarray,
    margins: np.ndarray,
    weights: np.ndarray,
    handicaps: np.ndarray,
    ratings: np.ndarray,
    k_factor: float,
    scale: float,
    thresholds: np.ndarray,
) -> None:
    """Update G-Elo ratings with per-game weights and handicaps."""
    n_games = len(player1)
    log10_scale = np.log(10.0) / scale

    for i in range(n_games):
        p1 = player1[i]
        p2 = player2[i]
        m = margins[i]
        w = weights[i]
        h = handicaps[i]

        r1 = ratings[p1]
        r2 = ratings[p2]
        delta = r1 - r2 + h

        s_gen = float(m)
        e_gen = _expected_gen_score(delta, thresholds, log10_scale)

        update = k_factor * w * (s_gen - e_gen)
        ratings[p1] = r1 + update
        ratings[p2] = r2 - update


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_with_handicap_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
    scale: float,
    handicaps: np.ndarray,
) -> np.ndarray:
    """Predict P(player1 wins) with per-game handicaps."""
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)
    log10_scale = np.log(10.0) / scale

    for i in prange(n):
        r1 = ratings[player1[i]]
        r2 = ratings[player2[i]]
        proba[i] = _sigmoid((r1 - r2 + handicaps[i]) * log10_scale)

    return proba


@njit(cache=True, fastmath=True)
def predict_single_with_handicap(
    p1_rating: float,
    p2_rating: float,
    scale: float,
    handicap: float,
) -> float:
    """Predict P(player1 wins) for a single matchup with handicap."""
    return _sigmoid((p1_rating - p2_rating + handicap) * np.log(10.0) / scale)


@njit(cache=True, fastmath=True)
def walk_forward_predict_update(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    margins: np.ndarray,
    day_offsets: np.ndarray,
    weights: np.ndarray,
    handicaps: np.ndarray,
    ratings: np.ndarray,
    k_factor: float,
    scale: float,
    thresholds: np.ndarray,
    n_train_days: int,
) -> np.ndarray:
    """Fused predict+update walk-forward in a single Numba call.

    For each post-training day: predict all games using current ratings
    (standard Elo sigmoid for symmetric thresholds), then update with
    G-Elo generalized score.

    Returns predictions array (NaN for training period).
    """
    n_games = len(player1)
    predictions = np.full(n_games, np.nan, dtype=np.float64)
    log10_scale = np.log(10.0) / scale
    n_days = len(day_offsets) - 1

    for day_idx in range(n_train_days, n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        # Predict before update (standard Elo sigmoid for win prob)
        for i in range(start, end):
            r1 = ratings[player1[i]]
            r2 = ratings[player2[i]]
            predictions[i] = _sigmoid((r1 - r2 + handicaps[i]) * log10_scale)

        # Update with G-Elo generalized score
        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            m = margins[i]
            w = weights[i]
            h = handicaps[i]

            r1 = ratings[p1]
            r2 = ratings[p2]
            delta = r1 - r2 + h

            s_gen = float(m)
            e_gen = _expected_gen_score(delta, thresholds, log10_scale)

            update = k_factor * w * (s_gen - e_gen)
            ratings[p1] = r1 + update
            ratings[p2] = r2 - update

    return predictions


@njit(cache=True, fastmath=True)
def scores_to_margins_bo3(scores: np.ndarray, p1sets: np.ndarray,
                           p2sets: np.ndarray) -> np.ndarray:
    """Convert binary scores + set counts to ordinal margin categories.

    For Bo3: J=4 categories (0-indexed):
        0: P1 lost 0-2  (p1sets=0, p2sets=2)
        1: P1 lost 1-2  (p1sets=1, p2sets=2)
        2: P1 won 2-1   (p1sets=2, p2sets=1)
        3: P1 won 2-0   (p1sets=2, p2sets=0)

    For Bo5: J=6 categories (0-indexed):
        0: P1 lost 0-3
        1: P1 lost 1-3
        2: P1 lost 2-3
        3: P1 won 3-2
        4: P1 won 3-1
        5: P1 won 3-0

    When set counts are unavailable (both 0), falls back to:
        score=1.0 → middle win category (2 for Bo3, 4 for Bo5)
        score=0.0 → middle loss category (1 for Bo3, 1 for Bo5)

    Args:
        scores: (N,) Binary scores (1=P1 wins, 0=P2 wins)
        p1sets: (N,) Sets won by Player 1
        p2sets: (N,) Sets won by Player 2

    Returns:
        (N,) Ordinal margin categories (0-indexed, float64 for Numba compat)
    """
    n = len(scores)
    margins = np.empty(n, dtype=np.float64)

    for i in range(n):
        s1 = p1sets[i]
        s2 = p2sets[i]
        sc = scores[i]

        if s1 + s2 == 0:
            # No set data — use default middle category
            if sc > 0.5:
                margins[i] = 2.0  # middle win
            else:
                margins[i] = 1.0  # middle loss
        elif s1 > s2:
            # P1 won
            # winner_sets is always max(s1,s2), loser_sets is min
            # For Bo3: won 2-0 → cat 3, won 2-1 → cat 2
            # For Bo5: won 3-0 → cat 5, won 3-1 → cat 4, won 3-2 → cat 3
            loser_sets = s2
            if s1 == 3:
                # Bo5
                margins[i] = float(5 - loser_sets)
            else:
                # Bo3
                margins[i] = float(3 - loser_sets)
        else:
            # P1 lost
            # For Bo3: lost 0-2 → cat 0, lost 1-2 → cat 1
            # For Bo5: lost 0-3 → cat 0, lost 1-3 → cat 1, lost 2-3 → cat 2
            loser_sets = s1
            margins[i] = float(loser_sets)

    return margins
