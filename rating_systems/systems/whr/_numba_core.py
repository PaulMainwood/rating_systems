"""
Numba-accelerated core functions for WHR (Whole History Rating).

Uses CSR-like data structures for efficient Numba compilation:
- Player timelines stored as contiguous arrays with offset indices
- Games per player-day stored similarly
- All hot paths JIT-compiled with caching

Data Structure Overview:
- player_offsets[num_players + 1]: Boundaries for each player's timeline
- pd_days[total_pd]: Day number for each player-day
- pd_r[total_pd]: Log-gamma rating (optimized variable)
- pd_game_offsets[total_pd + 1]: Boundaries for games per player-day
- pd_game_opp_pd[total_games_x2]: Opponent's player-day index
- pd_game_score[total_games_x2]: Score from this player's perspective
"""

import math
import numpy as np
from numba import njit, prange

# Conversion constant
LN10_400 = math.log(10) / 400.0


@njit(cache=True, fastmath=True)
def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x > 20.0:
        return 1.0 - 1e-9
    elif x < -20.0:
        return 1e-9
    else:
        return 1.0 / (1.0 + math.exp(-x))


@njit(cache=True, fastmath=True)
def solve_tridiagonal(
    diag: np.ndarray,
    off_diag: np.ndarray,
    rhs: np.ndarray,
    n: int,
    result: np.ndarray,
) -> None:
    """
    Solve tridiagonal system Ax = b using Thomas algorithm (in-place).

    A has:
    - diag[i] on main diagonal (negative values expected)
    - off_diag[i] on both sub and super diagonals (symmetric, positive)

    Solves: A * x = -rhs (Newton step direction)
    Result is stored in 'result' array.
    """
    if n == 0:
        return

    if n == 1:
        if abs(diag[0]) > 1e-15:
            result[0] = -rhs[0] / diag[0]
        else:
            result[0] = 0.0
        return

    # Work arrays (allocated on stack for small n, heap for large)
    c = np.empty(n - 1, dtype=np.float64)
    d = np.empty(n, dtype=np.float64)

    # Forward elimination
    c[0] = off_diag[0] / diag[0]
    d[0] = -rhs[0] / diag[0]

    for i in range(1, n - 1):
        denom = diag[i] - off_diag[i - 1] * c[i - 1]
        if abs(denom) < 1e-15:
            denom = -1e-15 if denom <= 0 else 1e-15
        c[i] = off_diag[i] / denom
        d[i] = (-rhs[i] - off_diag[i - 1] * d[i - 1]) / denom

    # Last row
    denom = diag[n - 1] - off_diag[n - 2] * c[n - 2]
    if abs(denom) < 1e-15:
        denom = -1e-15 if denom <= 0 else 1e-15
    d[n - 1] = (-rhs[n - 1] - off_diag[n - 2] * d[n - 2]) / denom

    # Back substitution
    result[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        result[i] = d[i] - c[i] * result[i + 1]


@njit(cache=True, fastmath=True)
def update_single_player(
    player_id: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    w2_r: float,
) -> float:
    """
    Update a single player's ratings using Newton-Raphson.

    Returns the maximum absolute change in rating.
    """
    pd_start = player_offsets[player_id]
    pd_end = player_offsets[player_id + 1]
    n = pd_end - pd_start

    if n == 0:
        return 0.0

    # Allocate working arrays
    gradient = np.zeros(n, dtype=np.float64)
    hess_diag = np.zeros(n, dtype=np.float64)
    hess_off = np.zeros(max(1, n - 1), dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)

    # Compute sigma² between consecutive days (Wiener process variance)
    for i in range(n - 1):
        day_diff = max(1, pd_days[pd_start + i + 1] - pd_days[pd_start + i])
        sigma2 = w2_r * day_diff
        inv_sigma2 = 1.0 / sigma2
        hess_off[i] = inv_sigma2

    # Build gradient and Hessian for each player-day
    for i in range(n):
        pd_idx = pd_start + i
        r_i = pd_r[pd_idx]

        # Game contributions
        game_start = pd_game_offsets[pd_idx]
        game_end = pd_game_offsets[pd_idx + 1]

        for g in range(game_start, game_end):
            opp_pd = pd_game_opp_pd[g]
            score = pd_game_score[g]
            opp_r = pd_r[opp_pd]

            # Probability of winning
            p_win = sigmoid(r_i - opp_r)

            # Gradient contribution: score - p_win
            gradient[i] += score - p_win

            # Hessian diagonal contribution: -p_win * (1 - p_win)
            hess_diag[i] -= p_win * (1.0 - p_win)

        # Virtual game prior on first day (anchors ratings toward 0)
        # Reference implementation adds virtual win+loss vs gamma=1 player
        # This is equivalent to a game with score=0.5 vs a r=0 opponent
        if i == 0:
            p_virtual = sigmoid(r_i - 0.0)  # vs r=0 opponent
            # Add both a win (score=1) and loss (score=0) -> average to 0.5
            gradient[i] += (1.0 - p_virtual)  # virtual win
            gradient[i] += (0.0 - p_virtual)  # virtual loss (p_virtual is our win prob)
            hess_diag[i] -= 2.0 * p_virtual * (1.0 - p_virtual)

        # Wiener process prior contributions
        if i > 0:
            # Connection to previous day
            inv_sigma2 = hess_off[i - 1]
            r_prev = pd_r[pd_start + i - 1]
            gradient[i] -= (r_i - r_prev) * inv_sigma2
            hess_diag[i] -= inv_sigma2

        if i < n - 1:
            # Connection to next day
            inv_sigma2 = hess_off[i]
            r_next = pd_r[pd_start + i + 1]
            gradient[i] -= (r_i - r_next) * inv_sigma2
            hess_diag[i] -= inv_sigma2

    # Regularization: add small prior toward zero to prevent divergence
    # This matches the reference implementation's -0.001 term
    # Acts as a weak Gaussian prior centered at 0
    for i in range(n):
        hess_diag[i] -= 0.001
        # Additional safety check
        if hess_diag[i] > -1e-10:
            hess_diag[i] = -1e-10

    # Solve tridiagonal system: H * delta = -gradient
    solve_tridiagonal(hess_diag, hess_off, gradient, n, delta)

    # Apply updates and track max change
    max_change = 0.0
    for i in range(n):
        pd_idx = pd_start + i
        change = delta[i]
        pd_r[pd_idx] += change
        if abs(change) > max_change:
            max_change = abs(change)

    return max_change


@njit(cache=True, fastmath=True)
def run_iteration(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    w2_r: float,
) -> float:
    """
    Run one Newton-Raphson iteration for all players.

    Returns maximum rating change across all players.

    Note: Player updates are NOT parallelized because each update
    reads opponent ratings that may be updated by other players.
    Sequential update provides Gauss-Seidel-like behavior which
    often converges faster than Jacobi-like parallel updates.
    """
    max_change = 0.0

    for player_id in range(num_players):
        change = update_single_player(
            player_id,
            player_offsets,
            pd_days,
            pd_r,
            pd_game_offsets,
            pd_game_opp_pd,
            pd_game_score,
            w2_r,
        )
        if change > max_change:
            max_change = change

    return max_change


@njit(cache=True, fastmath=True)
def run_all_iterations(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    w2_r: float,
    max_iterations: int,
    convergence_threshold: float,
) -> int:
    """
    Run Newton-Raphson iterations until convergence or max iterations.

    Returns the number of iterations performed.
    """
    for iteration in range(max_iterations):
        max_change = run_iteration(
            num_players,
            player_offsets,
            pd_days,
            pd_r,
            pd_game_offsets,
            pd_game_opp_pd,
            pd_game_score,
            w2_r,
        )

        if max_change < convergence_threshold:
            return iteration + 1

    return max_iterations


@njit(cache=True, fastmath=True)
def compute_uncertainties(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_uncertainty: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    w2_r: float,
) -> None:
    """
    Compute rating uncertainties from Hessian diagonal.

    Uncertainty = sqrt(1/|H_ii|) converted to Elo scale.
    """
    for player_id in range(num_players):
        pd_start = player_offsets[player_id]
        pd_end = player_offsets[player_id + 1]
        n = pd_end - pd_start

        if n == 0:
            continue

        # Compute sigma² between consecutive days
        sigma2 = np.empty(max(1, n - 1), dtype=np.float64)
        for i in range(n - 1):
            day_diff = max(1, pd_days[pd_start + i + 1] - pd_days[pd_start + i])
            sigma2[i] = w2_r * day_diff

        # Compute Hessian diagonal for each player-day
        for i in range(n):
            pd_idx = pd_start + i
            r_i = pd_r[pd_idx]
            hess = 0.0

            # Game contributions
            game_start = pd_game_offsets[pd_idx]
            game_end = pd_game_offsets[pd_idx + 1]

            for g in range(game_start, game_end):
                opp_pd = pd_game_opp_pd[g]
                opp_r = pd_r[opp_pd]
                p_win = sigmoid(r_i - opp_r)
                hess -= p_win * (1.0 - p_win)

            # Virtual game prior on first day
            if i == 0:
                p_virtual = sigmoid(r_i - 0.0)
                hess -= 2.0 * p_virtual * (1.0 - p_virtual)

            # Prior contributions
            if i > 0:
                hess -= 1.0 / sigma2[i - 1]
            if i < n - 1:
                hess -= 1.0 / sigma2[i]

            # Regularization term
            hess -= 0.001

            # Compute uncertainty from inverse Hessian
            if hess < -1e-10:
                var_r = -1.0 / hess
                pd_uncertainty[pd_idx] = math.sqrt(var_r) / LN10_400
            else:
                pd_uncertainty[pd_idx] = 350.0


@njit(cache=True, fastmath=True)
def extract_current_ratings(
    num_players: int,
    player_offsets: np.ndarray,
    pd_r: np.ndarray,
    pd_uncertainty: np.ndarray,
    ratings_out: np.ndarray,
    rd_out: np.ndarray,
    initial_rating: float,
) -> None:
    """
    Extract the most recent rating for each player.

    Converts from log-gamma scale to Elo scale.
    """
    for player_id in range(num_players):
        pd_start = player_offsets[player_id]
        pd_end = player_offsets[player_id + 1]

        if pd_end > pd_start:
            # Get most recent player-day
            last_pd = pd_end - 1
            ratings_out[player_id] = pd_r[last_pd] / LN10_400 + initial_rating
            rd_out[player_id] = pd_uncertainty[last_pd]
        else:
            # Player has no games
            ratings_out[player_id] = initial_rating
            rd_out[player_id] = 350.0


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
) -> np.ndarray:
    """
    Predict probability that player1 beats player2 (batch).

    Uses Bradley-Terry model with log-gamma scale ratings.
    Parallelized for efficiency on large batches.
    """
    n = len(player1)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        r1 = (ratings[player1[i]] - 1500.0) * LN10_400
        r2 = (ratings[player2[i]] - 1500.0) * LN10_400
        result[i] = sigmoid(r1 - r2)

    return result


@njit(cache=True, fastmath=True)
def predict_single(rating1: float, rating2: float) -> float:
    """Predict probability that player 1 beats player 2."""
    r1 = (rating1 - 1500.0) * LN10_400
    r2 = (rating2 - 1500.0) * LN10_400
    return sigmoid(r1 - r2)


@njit(cache=True)
def get_top_n_indices(ratings: np.ndarray, n: int) -> np.ndarray:
    """Get indices of top N rated players."""
    n = min(n, len(ratings))
    indices = np.argsort(-ratings)[:n]
    return indices


@njit(cache=True)
def get_bottom_n_indices(ratings: np.ndarray, n: int) -> np.ndarray:
    """Get indices of bottom N rated players."""
    n = min(n, len(ratings))
    indices = np.argsort(ratings)[:n]
    return indices


@njit(cache=True, fastmath=True)
def get_rating_at_day(
    player_id: int,
    target_day: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    initial_rating: float,
) -> float:
    """
    Get a player's rating at a specific day.

    If the player hasn't played on that exact day, returns the rating
    from their most recent game before that day, or initial_rating if
    no games before that day.
    """
    pd_start = player_offsets[player_id]
    pd_end = player_offsets[player_id + 1]

    if pd_end <= pd_start:
        return initial_rating

    # Binary search for the day (or closest day before)
    best_pd = -1
    for i in range(pd_start, pd_end):
        if pd_days[i] <= target_day:
            best_pd = i
        else:
            break

    if best_pd < 0:
        return initial_rating

    return pd_r[best_pd] / LN10_400 + initial_rating


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_at_day(
    player1: np.ndarray,
    player2: np.ndarray,
    target_day: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    initial_rating: float,
) -> np.ndarray:
    """
    Predict match outcomes using ratings as of a specific day.

    Useful for backtesting where we want predictions using only
    information available at the time of the match.
    """
    n = len(player1)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        r1 = get_rating_at_day(
            player1[i], target_day, player_offsets, pd_days, pd_r, initial_rating
        )
        r2 = get_rating_at_day(
            player2[i], target_day, player_offsets, pd_days, pd_r, initial_rating
        )
        r1_lg = (r1 - initial_rating) * LN10_400
        r2_lg = (r2 - initial_rating) * LN10_400
        result[i] = sigmoid(r1_lg - r2_lg)

    return result


@njit(cache=True)
def warm_start_ratings(
    num_players: int,
    old_player_offsets: np.ndarray,
    old_pd_days: np.ndarray,
    old_pd_r: np.ndarray,
    new_player_offsets: np.ndarray,
    new_pd_days: np.ndarray,
    new_pd_r: np.ndarray,
) -> None:
    """
    Transfer converged ratings from old data structures to new ones.

    For each player, uses a two-pointer merge on sorted day arrays to:
    - Copy exact (player, day) matches from old to new
    - Extrapolate new player-days from the nearest prior rating

    This provides an excellent starting point for Newton-Raphson,
    dramatically reducing the iterations needed for convergence
    when refitting with slightly more data (e.g. walk-forward backtesting).
    """
    for player_id in range(num_players):
        old_start = old_player_offsets[player_id]
        old_end = old_player_offsets[player_id + 1]
        new_start = new_player_offsets[player_id]
        new_end = new_player_offsets[player_id + 1]

        old_n = old_end - old_start
        new_n = new_end - new_start

        if old_n == 0 or new_n == 0:
            continue

        # Two-pointer merge on sorted day arrays
        old_i = 0
        new_i = 0
        last_known_r = old_pd_r[old_start]  # fallback for days before any old day

        while new_i < new_n:
            new_day = new_pd_days[new_start + new_i]

            # Advance old pointer past days before current new day
            while old_i < old_n and old_pd_days[old_start + old_i] < new_day:
                last_known_r = old_pd_r[old_start + old_i]
                old_i += 1

            if old_i < old_n and old_pd_days[old_start + old_i] == new_day:
                # Exact match: copy old converged rating
                new_pd_r[new_start + new_i] = old_pd_r[old_start + old_i]
                last_known_r = old_pd_r[old_start + old_i]
                old_i += 1
            else:
                # New day: extrapolate from most recent prior rating
                new_pd_r[new_start + new_i] = last_known_r

            new_i += 1


@njit(cache=True)
def fill_game_arrays(
    n_games: int,
    pd1_indices: np.ndarray,
    pd2_indices: np.ndarray,
    scores: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
) -> None:
    """
    Fill game arrays with opponent references and scores.

    Used by the NumPy-based _build_data_structures for the final step
    that requires tracking per-player-day positions.
    """
    pd_game_pos = pd_game_offsets[:-1].copy()

    for i in range(n_games):
        pd1 = pd1_indices[i]
        pd2 = pd2_indices[i]
        score = scores[i]

        # Add game from player1's perspective
        pos1 = pd_game_pos[pd1]
        pd_game_opp_pd[pos1] = pd2
        pd_game_score[pos1] = score
        pd_game_pos[pd1] += 1

        # Add game from player2's perspective
        pos2 = pd_game_pos[pd2]
        pd_game_opp_pd[pos2] = pd1
        pd_game_score[pos2] = 1.0 - score
        pd_game_pos[pd2] += 1


# ---------------------------------------------------------------------------
# Optimization 2: Active-set player updates
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def run_iteration_active(
    num_players: int,
    active: np.ndarray,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    w2_r: float,
    threshold: float,
    pd_to_player: np.ndarray,
) -> float:
    """
    Run one Newton-Raphson iteration, only updating active players.

    Players whose update is below threshold are deactivated. When a player's
    update exceeds threshold, all its opponents are reactivated.

    Returns maximum rating change across all active players.
    """
    max_change = 0.0

    for player_id in range(num_players):
        if not active[player_id]:
            continue

        change = update_single_player(
            player_id,
            player_offsets,
            pd_days,
            pd_r,
            pd_game_offsets,
            pd_game_opp_pd,
            pd_game_score,
            w2_r,
        )

        if change < threshold:
            active[player_id] = False
        else:
            # Reactivate all opponents of this player
            pd_start = player_offsets[player_id]
            pd_end = player_offsets[player_id + 1]
            for pd_idx in range(pd_start, pd_end):
                game_start = pd_game_offsets[pd_idx]
                game_end = pd_game_offsets[pd_idx + 1]
                for g in range(game_start, game_end):
                    opp_pd = pd_game_opp_pd[g]
                    opp_id = pd_to_player[opp_pd]
                    active[opp_id] = True

        if change > max_change:
            max_change = change

    return max_change


# ---------------------------------------------------------------------------
# Optimization 3: Anderson acceleration helpers
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def solve_small_system(A: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Solve dense n×n linear system Ax = b via Gaussian elimination with
    partial pivoting. n is at most anderson_window (typically 3-5).
    """
    # Work on copies to avoid modifying originals
    M = A.copy()
    rhs = b.copy()

    for col in range(n):
        # Partial pivoting
        best_row = col
        best_val = abs(M[col, col])
        for row in range(col + 1, n):
            val = abs(M[row, col])
            if val > best_val:
                best_val = val
                best_row = row
        if best_row != col:
            for j in range(n):
                M[col, j], M[best_row, j] = M[best_row, j], M[col, j]
            rhs[col], rhs[best_row] = rhs[best_row], rhs[col]

        pivot = M[col, col]
        if abs(pivot) < 1e-30:
            continue

        # Eliminate below
        for row in range(col + 1, n):
            factor = M[row, col] / pivot
            for j in range(col + 1, n):
                M[row, j] -= factor * M[col, j]
            M[row, col] = 0.0
            rhs[row] -= factor * rhs[col]

    # Back substitution
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        if abs(M[i, i]) < 1e-30:
            x[i] = 0.0
        else:
            x[i] = s / M[i, i]

    return x


@njit(cache=True, fastmath=True)
def anderson_mix(
    pd_r: np.ndarray,
    F_buf: np.ndarray,
    G_buf: np.ndarray,
    buf_count: int,
    m: int,
    total_pd: int,
    regularization: float,
) -> None:
    """
    Anderson mixing step: combine recent iterates to accelerate convergence.

    F_buf: circular buffer of residuals (f_k = x_after - x_before)
    G_buf: circular buffer of post-iterate states (x_after)

    Modifies pd_r in-place with the mixed result.
    """
    k = min(buf_count, m)
    if k < 2:
        return

    # Build k×k Gram matrix FTF[i,j] = dot(F_buf[i], F_buf[j])
    FTF = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        fi_idx = (buf_count - k + i) % m
        for j in range(i, k):
            fj_idx = (buf_count - k + j) % m
            dot_val = 0.0
            for t in range(total_pd):
                dot_val += F_buf[fi_idx, t] * F_buf[fj_idx, t]
            FTF[i, j] = dot_val
            FTF[j, i] = dot_val

    # Regularise diagonal
    for i in range(k):
        FTF[i, i] += regularization

    # Solve FTF @ beta = ones(k)
    ones_k = np.ones(k, dtype=np.float64)
    beta = solve_small_system(FTF, ones_k, k)

    # Normalise: alpha = beta / sum(beta)
    beta_sum = 0.0
    for i in range(k):
        beta_sum += beta[i]
    if abs(beta_sum) < 1e-30:
        return
    for i in range(k):
        beta[i] /= beta_sum

    # Mix: pd_r[t] = sum(alpha_i * G_buf[i, t])
    for t in range(total_pd):
        mixed = 0.0
        for i in range(k):
            g_idx = (buf_count - k + i) % m
            mixed += beta[i] * G_buf[g_idx, t]
        pd_r[t] = mixed


# ---------------------------------------------------------------------------
# Combined accelerated iteration loop
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def run_all_iterations_accelerated(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_r: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    w2_r: float,
    max_iterations: int,
    convergence_threshold: float,
    anderson_window: int,
    use_active_set: bool,
    pd_to_player: np.ndarray,
) -> int:
    """
    Run Newton-Raphson with optional active-set and Anderson acceleration.

    Parameters:
        anderson_window: Number of recent iterates to use for Anderson mixing.
                         0 = disabled.
        use_active_set: If True, skip converged players each iteration.
        pd_to_player: Reverse mapping from player-day index to player ID.

    Returns the number of iterations performed.
    """
    total_pd = len(pd_r)

    # Active set initialisation: all players start active
    active = np.ones(num_players, dtype=np.bool_)

    # Anderson buffer initialisation
    anderson_enabled = anderson_window > 0
    m = max(anderson_window, 1)  # buffer size (at least 1 to avoid zero-size alloc)
    if anderson_enabled:
        F_buf = np.zeros((m, total_pd), dtype=np.float64)
        G_buf = np.zeros((m, total_pd), dtype=np.float64)
    else:
        # Dummy arrays (never accessed)
        F_buf = np.zeros((1, 1), dtype=np.float64)
        G_buf = np.zeros((1, 1), dtype=np.float64)
    buf_count = 0
    regularization = 1e-10

    for iteration in range(max_iterations):
        # Save pre-iteration state if Anderson is enabled
        if anderson_enabled:
            x_before = pd_r.copy()

        # Run one sweep
        if use_active_set:
            max_change = run_iteration_active(
                num_players, active, player_offsets, pd_days, pd_r,
                pd_game_offsets, pd_game_opp_pd, pd_game_score,
                w2_r, convergence_threshold, pd_to_player,
            )
        else:
            max_change = run_iteration(
                num_players, player_offsets, pd_days, pd_r,
                pd_game_offsets, pd_game_opp_pd, pd_game_score,
                w2_r,
            )

        if max_change < convergence_threshold:
            return iteration + 1

        # Anderson acceleration step
        if anderson_enabled:
            slot = buf_count % m
            for t in range(total_pd):
                F_buf[slot, t] = pd_r[t] - x_before[t]  # residual
                G_buf[slot, t] = pd_r[t]                 # post-state
            buf_count += 1

            if buf_count >= 2:
                anderson_mix(
                    pd_r, F_buf, G_buf, buf_count, m,
                    total_pd, regularization,
                )
                # After mixing, reactivate all players for the next sweep
                if use_active_set:
                    for pid in range(num_players):
                        active[pid] = True

    return max_iterations
