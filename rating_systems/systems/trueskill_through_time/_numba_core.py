"""
Numba-accelerated core functions for TrueSkill Through Time (TTT).

This implementation uses SPARSE state storage: instead of dense arrays of
size (num_batches * num_players), state is stored per-appearance, where an
appearance is a unique (player, batch) pair. With ~500K appearances vs ~119M
dense entries, this gives ~240x memory reduction and proportional speedups.

Key data structures:
- Batches: time steps, each containing games played at that time
- Appearances: sparse (player, batch) pairs, indexed by a flat appearance ID
- For each appearance: forward, backward, likelihood messages (Gaussians)
- Linked list: each appearance stores prev/next appearance for the same player

Algorithm:
1. initial_forward_pass(): Forward pass creating batches sequentially
2. run_convergence(): Backward sweep then forward sweep, updating likelihoods
"""

import math
import numpy as np
from numba import njit, prange

# Constants
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
INV_SQRT2 = 1.0 / SQRT2
INF_SIGMA = 1e10  # Represents infinite sigma (uninformative)


# =============================================================================
# Gaussian utilities
# =============================================================================

@njit(cache=True, fastmath=True)
def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / SQRT2PI


@njit(cache=True, fastmath=True)
def norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x * INV_SQRT2))


@njit(cache=True, fastmath=True)
def gaussian_mul(mu1: float, sigma1: float, mu2: float, sigma2: float) -> tuple:
    """
    Multiply two Gaussians: N(mu1,σ1²) * N(mu2,σ2²).
    Returns (mu, sigma) of the product.
    """
    if sigma1 > INF_SIGMA * 0.1:
        return mu2, sigma2
    if sigma2 > INF_SIGMA * 0.1:
        return mu1, sigma1

    pi1 = 1.0 / (sigma1 * sigma1)
    pi2 = 1.0 / (sigma2 * sigma2)
    pi = pi1 + pi2
    tau = mu1 * pi1 + mu2 * pi2

    sigma = 1.0 / math.sqrt(pi)
    mu = tau / pi
    return mu, sigma


@njit(cache=True, fastmath=True)
def gaussian_div(mu1: float, sigma1: float, mu2: float, sigma2: float,
                 prior_mu: float, prior_sigma: float) -> tuple:
    """
    Divide two Gaussians: N(mu1,σ1²) / N(mu2,σ2²).
    Returns (mu, sigma) of the quotient.
    """
    if sigma2 > INF_SIGMA * 0.1:
        return mu1, sigma1
    if sigma1 > INF_SIGMA * 0.1:
        return prior_mu, prior_sigma

    pi1 = 1.0 / (sigma1 * sigma1)
    pi2 = 1.0 / (sigma2 * sigma2)
    pi = pi1 - pi2

    if pi < 1e-10:
        return prior_mu, prior_sigma

    tau = mu1 * pi1 - mu2 * pi2
    sigma = 1.0 / math.sqrt(pi)
    mu = tau / pi
    return mu, sigma


@njit(cache=True, fastmath=True)
def gaussian_forget(mu: float, sigma: float, gamma: float, elapsed: float) -> tuple:
    """
    Apply skill drift: increase sigma based on time elapsed.
    sigma_new = sqrt(sigma² + elapsed * gamma²)
    """
    if elapsed <= 0 or sigma > INF_SIGMA * 0.1:
        return mu, sigma
    new_var = sigma * sigma + elapsed * gamma * gamma
    return mu, math.sqrt(new_var)


# =============================================================================
# Truncated Gaussian (v and w functions for game outcomes)
# =============================================================================

@njit(cache=True, fastmath=True)
def v_w_win(t: float) -> tuple:
    """
    Compute v and w for a WIN outcome (no draw).

    For observing diff > margin (margin=0 for win):
    - v = pdf(t) / cdf(t)
    - w = v * (v + t)
    """
    cdf_t = norm_cdf(t)
    if cdf_t < 1e-10:
        v = -t
        w = 1.0 - 1e-6
    else:
        v = norm_pdf(t) / cdf_t
        w = v * (v + t)

    if w < 1e-10:
        w = 1e-10
    elif w > 1.0 - 1e-10:
        w = 1.0 - 1e-10

    return v, w


@njit(cache=True, fastmath=True)
def trunc(mu: float, sigma: float, margin: float, is_win: bool) -> tuple:
    """
    Compute truncated Gaussian mean and sigma after observing outcome.
    """
    t = (mu - margin) / sigma if sigma > 1e-10 else 0.0

    if is_win:
        v, w = v_w_win(t)
    else:
        v, w = v_w_win(-t)
        v = -v

    mu_trunc = mu + sigma * v
    sigma_trunc = sigma * math.sqrt(max(1.0 - w, 1e-10))

    return mu_trunc, sigma_trunc


# =============================================================================
# Game likelihood computation (for 2-player games)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_game_likelihoods(
    p1_mu: float, p1_sigma: float,
    p2_mu: float, p2_sigma: float,
    p1_wins: bool,
    beta: float,
) -> tuple:
    """
    Compute likelihood messages for both players from a single game.
    Returns: (lik1_mu, lik1_sigma, lik2_mu, lik2_sigma)
    """
    perf1_sigma = math.sqrt(p1_sigma * p1_sigma + beta * beta)
    perf2_sigma = math.sqrt(p2_sigma * p2_sigma + beta * beta)

    if p1_wins:
        diff_mu = p1_mu - p2_mu
    else:
        diff_mu = p2_mu - p1_mu
    diff_sigma = math.sqrt(perf1_sigma * perf1_sigma + perf2_sigma * perf2_sigma)

    mu_trunc, sigma_trunc = trunc(diff_mu, diff_sigma, 0.0, True)

    if abs(diff_sigma - sigma_trunc) < 1e-10:
        return 0.0, INF_SIGMA, 0.0, INF_SIGMA

    diff_var = diff_sigma * diff_sigma
    trunc_var = sigma_trunc * sigma_trunc

    delta_div = (diff_var * mu_trunc - trunc_var * diff_mu) / (diff_var - trunc_var)
    theta_div_sq = (trunc_var * diff_var) / (diff_var - trunc_var)

    lik1_sigma = math.sqrt(theta_div_sq + diff_var - p1_sigma * p1_sigma)
    lik2_sigma = math.sqrt(theta_div_sq + diff_var - p2_sigma * p2_sigma)

    if p1_wins:
        lik1_mu = p1_mu + (delta_div - diff_mu)
        lik2_mu = p2_mu - (delta_div - diff_mu)
    else:
        lik1_mu = p1_mu - (delta_div - diff_mu)
        lik2_mu = p2_mu + (delta_div - diff_mu)

    return lik1_mu, lik1_sigma, lik2_mu, lik2_sigma


# =============================================================================
# Batch game processing (unchanged - works with temp arrays by player_id)
# =============================================================================

@njit(cache=True)
def process_batch_games(
    batch_idx: int,
    batch_offsets: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    player_forward_mu: np.ndarray,
    player_forward_sigma: np.ndarray,
    player_backward_mu: np.ndarray,
    player_backward_sigma: np.ndarray,
    player_likelihood_mu: np.ndarray,
    player_likelihood_sigma: np.ndarray,
    beta: float,
    prior_mu: float,
    prior_sigma: float,
) -> None:
    """
    Process all games in a batch and update likelihood messages.
    Uses temp arrays indexed by player_id.
    """
    game_start = batch_offsets[batch_idx]
    game_end = batch_offsets[batch_idx + 1]

    # Reset likelihoods for players in this batch
    for g in range(game_start, game_end):
        p1, p2 = game_p1[g], game_p2[g]
        player_likelihood_mu[p1] = 0.0
        player_likelihood_sigma[p1] = INF_SIGMA
        player_likelihood_mu[p2] = 0.0
        player_likelihood_sigma[p2] = INF_SIGMA

    # Process each game
    for g in range(game_start, game_end):
        p1, p2 = game_p1[g], game_p2[g]
        p1_wins = game_scores[g] > 0.5

        p1_prior_mu, p1_prior_sigma = gaussian_mul(
            player_forward_mu[p1], player_forward_sigma[p1],
            player_backward_mu[p1], player_backward_sigma[p1]
        )
        p2_prior_mu, p2_prior_sigma = gaussian_mul(
            player_forward_mu[p2], player_forward_sigma[p2],
            player_backward_mu[p2], player_backward_sigma[p2]
        )

        lik1_mu, lik1_sigma, lik2_mu, lik2_sigma = compute_game_likelihoods(
            p1_prior_mu, p1_prior_sigma,
            p2_prior_mu, p2_prior_sigma,
            p1_wins, beta
        )

        player_likelihood_mu[p1], player_likelihood_sigma[p1] = gaussian_mul(
            player_likelihood_mu[p1], player_likelihood_sigma[p1],
            lik1_mu, lik1_sigma
        )
        player_likelihood_mu[p2], player_likelihood_sigma[p2] = gaussian_mul(
            player_likelihood_mu[p2], player_likelihood_sigma[p2],
            lik2_mu, lik2_sigma
        )


# =============================================================================
# Sparse structure building
# =============================================================================

@njit(cache=True)
def build_appearance_structure(
    num_batches: int,
    batch_offsets: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    num_players: int,
) -> tuple:
    """
    Build sparse appearance structures from game data.

    Returns:
        app_offsets: int64[num_batches+1] - CSR boundaries per batch
        app_player: int64[num_appearances] - player ID per appearance
        app_prev: int64[num_appearances] - prev appearance for same player (-1 if first)
        app_next: int64[num_appearances] - next appearance for same player (-1 if last)
        app_batch: int64[num_appearances] - batch index for each appearance
        player_last_app: int64[num_players] - last appearance index per player (-1 if never)
    """
    # Track which batch each player was last seen in (-1 = never)
    player_last_batch = np.full(num_players, -1, dtype=np.int64)
    # Temp array to collect players per batch (reused, avoids per-batch allocation)
    batch_players = np.empty(num_players, dtype=np.int64)

    # First pass: count unique appearances per batch
    batch_app_counts = np.zeros(num_batches, dtype=np.int64)
    for b in range(num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        for g in range(game_start, game_end):
            p1, p2 = game_p1[g], game_p2[g]
            if player_last_batch[p1] != b:
                player_last_batch[p1] = b
                batch_app_counts[b] += 1
            if player_last_batch[p2] != b:
                player_last_batch[p2] = b
                batch_app_counts[b] += 1

    # Build app_offsets (CSR)
    app_offsets = np.zeros(num_batches + 1, dtype=np.int64)
    for b in range(num_batches):
        app_offsets[b + 1] = app_offsets[b] + batch_app_counts[b]
    total_appearances = app_offsets[num_batches]

    # Second pass: fill app_player and app_batch
    app_player = np.empty(total_appearances, dtype=np.int64)
    app_batch = np.empty(total_appearances, dtype=np.int64)
    player_last_batch[:] = -1

    for b in range(num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        n_in_batch = 0
        for g in range(game_start, game_end):
            p1, p2 = game_p1[g], game_p2[g]
            if player_last_batch[p1] != b:
                player_last_batch[p1] = b
                batch_players[n_in_batch] = p1
                n_in_batch += 1
            if player_last_batch[p2] != b:
                player_last_batch[p2] = b
                batch_players[n_in_batch] = p2
                n_in_batch += 1

        # Sort for deterministic ordering
        batch_players[:n_in_batch].sort()

        # Write to arrays
        pos = app_offsets[b]
        for i in range(n_in_batch):
            app_player[pos + i] = batch_players[i]
            app_batch[pos + i] = b

    # Build prev/next linked lists
    app_prev = np.full(total_appearances, -1, dtype=np.int64)
    app_next = np.full(total_appearances, -1, dtype=np.int64)
    player_last_app = np.full(num_players, -1, dtype=np.int64)

    for a in range(total_appearances):
        p = app_player[a]
        prev_a = player_last_app[p]
        if prev_a >= 0:
            app_prev[a] = prev_a
            app_next[prev_a] = a
        player_last_app[p] = a

    return app_offsets, app_player, app_prev, app_next, app_batch, player_last_app


# =============================================================================
# Sparse forward/backward sweeps
# =============================================================================

@njit(cache=True)
def initial_forward_pass_sparse(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    # Sparse structures
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_prev: np.ndarray,
    app_batch: np.ndarray,
    # Sparse state arrays (indexed by appearance)
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    # Pre-allocated temp arrays (size num_players)
    temp_fwd_mu: np.ndarray,
    temp_fwd_sigma: np.ndarray,
    temp_bwd_mu: np.ndarray,
    temp_bwd_sigma: np.ndarray,
    temp_lik_mu: np.ndarray,
    temp_lik_sigma: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    beta: float,
    gamma: float,
    start_batch: int,
) -> None:
    """
    Forward pass with sparse state storage.

    Processes batches from start_batch to num_batches-1. For start_batch=0
    this is the full initial forward pass. For start_batch>0 (warm-start
    refits), it only processes new batches, reading forward/likelihood state
    from previous appearances that were already computed.
    """
    for b in range(start_batch, num_batches):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        # Set forward messages for appearances in this batch
        for a in range(a_start, a_end):
            p = app_player[a]
            prev_a = app_prev[a]

            if prev_a < 0:
                # First appearance: use prior
                state_forward_mu[a] = prior_mu
                state_forward_sigma[a] = prior_sigma
            else:
                # Propagate from previous appearance (forward * likelihood, then drift)
                prev_time = batch_times[app_batch[prev_a]]
                fwd_lik_mu, fwd_lik_sigma = gaussian_mul(
                    state_forward_mu[prev_a], state_forward_sigma[prev_a],
                    state_likelihood_mu[prev_a], state_likelihood_sigma[prev_a]
                )
                elapsed = batch_time - prev_time
                fwd_mu, fwd_sigma = gaussian_forget(fwd_lik_mu, fwd_lik_sigma, gamma, elapsed)
                state_forward_mu[a] = fwd_mu
                state_forward_sigma[a] = fwd_sigma

            # Backward is uninformative initially
            state_backward_mu[a] = 0.0
            state_backward_sigma[a] = INF_SIGMA

            # Copy to temp for game processing
            temp_fwd_mu[p] = state_forward_mu[a]
            temp_fwd_sigma[p] = state_forward_sigma[a]
            temp_bwd_mu[p] = state_backward_mu[a]
            temp_bwd_sigma[p] = state_backward_sigma[a]

        # Process games
        process_batch_games(
            b, batch_offsets, game_p1, game_p2, game_scores,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            beta, prior_mu, prior_sigma
        )

        # Copy likelihoods back to sparse state
        for a in range(a_start, a_end):
            p = app_player[a]
            state_likelihood_mu[a] = temp_lik_mu[p]
            state_likelihood_sigma[a] = temp_lik_sigma[p]


@njit(cache=True)
def backward_sweep_sparse(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    # Sparse structures
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_next: np.ndarray,
    app_batch: np.ndarray,
    # Sparse state
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    # Pre-allocated temps
    temp_fwd_mu: np.ndarray,
    temp_fwd_sigma: np.ndarray,
    temp_bwd_mu: np.ndarray,
    temp_bwd_sigma: np.ndarray,
    temp_lik_mu: np.ndarray,
    temp_lik_sigma: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    beta: float,
    gamma: float,
) -> float:
    """
    Backward sweep with sparse state. Returns max change for convergence.

    Uses actual elapsed time between a player's consecutive appearances for
    drift, which is the correct TTT algorithm per Dangauthier et al.
    """
    max_change = 0.0

    for b in range(num_batches - 1, -1, -1):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        # Update backward messages from next appearance
        for a in range(a_start, a_end):
            p = app_player[a]
            next_a = app_next[a]
            old_mu = state_backward_mu[a]

            if next_a < 0:
                # Last appearance: backward is uninformative
                state_backward_mu[a] = 0.0
                state_backward_sigma[a] = INF_SIGMA
            else:
                # Propagate from next appearance (likelihood * backward, then drift)
                lik_bwd_mu, lik_bwd_sigma = gaussian_mul(
                    state_likelihood_mu[next_a], state_likelihood_sigma[next_a],
                    state_backward_mu[next_a], state_backward_sigma[next_a]
                )
                # Actual elapsed between this appearance and the next
                next_time = batch_times[app_batch[next_a]]
                elapsed = next_time - batch_time
                bwd_mu, bwd_sigma = gaussian_forget(lik_bwd_mu, lik_bwd_sigma, gamma, elapsed)
                state_backward_mu[a] = bwd_mu
                state_backward_sigma[a] = bwd_sigma

            change = abs(state_backward_mu[a] - old_mu)
            if change > max_change:
                max_change = change

            # Copy to temp for game processing
            temp_fwd_mu[p] = state_forward_mu[a]
            temp_fwd_sigma[p] = state_forward_sigma[a]
            temp_bwd_mu[p] = state_backward_mu[a]
            temp_bwd_sigma[p] = state_backward_sigma[a]

        # Recompute likelihoods
        process_batch_games(
            b, batch_offsets, game_p1, game_p2, game_scores,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            beta, prior_mu, prior_sigma
        )

        # Copy back
        for a in range(a_start, a_end):
            p = app_player[a]
            state_likelihood_mu[a] = temp_lik_mu[p]
            state_likelihood_sigma[a] = temp_lik_sigma[p]

    return max_change


@njit(cache=True)
def forward_sweep_sparse(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    # Sparse structures
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_prev: np.ndarray,
    app_batch: np.ndarray,
    # Sparse state
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    # Pre-allocated temps
    temp_fwd_mu: np.ndarray,
    temp_fwd_sigma: np.ndarray,
    temp_bwd_mu: np.ndarray,
    temp_bwd_sigma: np.ndarray,
    temp_lik_mu: np.ndarray,
    temp_lik_sigma: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    beta: float,
    gamma: float,
) -> float:
    """
    Forward sweep with sparse state. Returns max change for convergence.

    Uses actual elapsed time between a player's consecutive appearances for
    drift, which is the correct TTT algorithm per Dangauthier et al.
    """
    max_change = 0.0

    for b in range(num_batches):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        # Update forward messages from previous appearance
        for a in range(a_start, a_end):
            p = app_player[a]
            prev_a = app_prev[a]
            old_mu = state_forward_mu[a]

            if prev_a < 0:
                # First appearance: use prior
                state_forward_mu[a] = prior_mu
                state_forward_sigma[a] = prior_sigma
            else:
                fwd_lik_mu, fwd_lik_sigma = gaussian_mul(
                    state_forward_mu[prev_a], state_forward_sigma[prev_a],
                    state_likelihood_mu[prev_a], state_likelihood_sigma[prev_a]
                )
                # Actual elapsed between previous appearance and this one
                prev_time = batch_times[app_batch[prev_a]]
                elapsed = batch_time - prev_time
                fwd_mu, fwd_sigma = gaussian_forget(fwd_lik_mu, fwd_lik_sigma, gamma, elapsed)
                state_forward_mu[a] = fwd_mu
                state_forward_sigma[a] = fwd_sigma

            change = abs(state_forward_mu[a] - old_mu)
            if change > max_change:
                max_change = change

            # Copy to temp for game processing
            temp_fwd_mu[p] = state_forward_mu[a]
            temp_fwd_sigma[p] = state_forward_sigma[a]
            temp_bwd_mu[p] = state_backward_mu[a]
            temp_bwd_sigma[p] = state_backward_sigma[a]

        # Recompute likelihoods
        process_batch_games(
            b, batch_offsets, game_p1, game_p2, game_scores,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            beta, prior_mu, prior_sigma
        )

        # Copy back
        for a in range(a_start, a_end):
            p = app_player[a]
            state_likelihood_mu[a] = temp_lik_mu[p]
            state_likelihood_sigma[a] = temp_lik_sigma[p]

    return max_change


@njit(cache=True)
def run_convergence_sparse(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    # Sparse structures
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_prev: np.ndarray,
    app_next: np.ndarray,
    app_batch: np.ndarray,
    # Sparse state
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    # Pre-allocated temps
    temp_fwd_mu: np.ndarray,
    temp_fwd_sigma: np.ndarray,
    temp_bwd_mu: np.ndarray,
    temp_bwd_sigma: np.ndarray,
    temp_lik_mu: np.ndarray,
    temp_lik_sigma: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    beta: float,
    gamma: float,
    max_iterations: int,
    epsilon: float,
) -> int:
    """
    Run forward-backward iterations until convergence.
    Returns number of iterations performed.
    """
    for iteration in range(max_iterations):
        bwd_change = backward_sweep_sparse(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            app_offsets, app_player, app_next, app_batch,
            state_forward_mu, state_forward_sigma,
            state_backward_mu, state_backward_sigma,
            state_likelihood_mu, state_likelihood_sigma,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            prior_mu, prior_sigma, beta, gamma
        )

        fwd_change = forward_sweep_sparse(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            app_offsets, app_player, app_prev, app_batch,
            state_forward_mu, state_forward_sigma,
            state_backward_mu, state_backward_sigma,
            state_likelihood_mu, state_likelihood_sigma,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            prior_mu, prior_sigma, beta, gamma
        )

        max_change = max(bwd_change, fwd_change)
        if max_change < epsilon:
            return iteration + 1

    return max_iterations


@njit(cache=True)
def extract_final_ratings_sparse(
    num_players: int,
    player_last_app: np.ndarray,
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    ratings_out: np.ndarray,
    rd_out: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    display_scale: float,
    display_offset: float,
) -> None:
    """
    Extract the most recent rating (posterior) for each player.
    Posterior = forward * backward * likelihood at last appearance.
    """
    for p in range(num_players):
        a = player_last_app[p]
        if a < 0:
            ratings_out[p] = display_offset
            rd_out[p] = prior_sigma * display_scale
        else:
            fwd_bwd_mu, fwd_bwd_sigma = gaussian_mul(
                state_forward_mu[a], state_forward_sigma[a],
                state_backward_mu[a], state_backward_sigma[a]
            )
            post_mu, post_sigma = gaussian_mul(
                fwd_bwd_mu, fwd_bwd_sigma,
                state_likelihood_mu[a], state_likelihood_sigma[a]
            )

            ratings_out[p] = post_mu * display_scale + display_offset
            rd_out[p] = post_sigma * display_scale


# =============================================================================
# Prediction functions (unchanged)
# =============================================================================

@njit(cache=True, parallel=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
    rd: np.ndarray,
    beta: float,
    display_scale: float,
    display_offset: float,
) -> np.ndarray:
    """Predict probability that player1 beats player2 (batch, parallel)."""
    n = len(player1)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        p1, p2 = player1[i], player2[i]
        mu1 = (ratings[p1] - display_offset) / display_scale
        mu2 = (ratings[p2] - display_offset) / display_scale
        sigma1 = rd[p1] / display_scale
        sigma2 = rd[p2] / display_scale
        diff_mu = mu1 - mu2
        diff_sigma = math.sqrt(sigma1*sigma1 + sigma2*sigma2 + 2.0*beta*beta)
        result[i] = norm_cdf(diff_mu / diff_sigma)

    return result


@njit(cache=True)
def predict_single(
    rating1: float,
    rating2: float,
    rd1: float,
    rd2: float,
    beta: float,
    display_scale: float,
    display_offset: float,
) -> float:
    """Predict probability that player 1 beats player 2."""
    mu1 = (rating1 - display_offset) / display_scale
    mu2 = (rating2 - display_offset) / display_scale
    sigma1 = rd1 / display_scale
    sigma2 = rd2 / display_scale
    diff_mu = mu1 - mu2
    diff_sigma = math.sqrt(sigma1*sigma1 + sigma2*sigma2 + 2.0*beta*beta)
    return norm_cdf(diff_mu / diff_sigma)


# =============================================================================
# Weighted team game likelihood computation (for surface-specific TTT)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_weighted_team_likelihoods(
    t1_base_mu: float, t1_base_sigma: float,
    t1_surf_mu: float, t1_surf_sigma: float,
    t2_base_mu: float, t2_base_sigma: float,
    t2_surf_mu: float, t2_surf_sigma: float,
    w_base: float, w_surf: float,
    t1_wins: bool,
    beta: float,
) -> tuple:
    """
    Compute likelihood messages for all 4 players from a weighted team game.
    """
    w_base_sq = w_base * w_base
    w_surf_sq = w_surf * w_surf
    beta_sq = beta * beta

    t1_mu = w_base * t1_base_mu + w_surf * t1_surf_mu
    t1_perf_var = w_base_sq * (t1_base_sigma * t1_base_sigma + beta_sq) + \
                  w_surf_sq * (t1_surf_sigma * t1_surf_sigma + beta_sq)

    t2_mu = w_base * t2_base_mu + w_surf * t2_surf_mu
    t2_perf_var = w_base_sq * (t2_base_sigma * t2_base_sigma + beta_sq) + \
                  w_surf_sq * (t2_surf_sigma * t2_surf_sigma + beta_sq)

    if t1_wins:
        diff_mu = t1_mu - t2_mu
    else:
        diff_mu = t2_mu - t1_mu
    diff_var = t1_perf_var + t2_perf_var
    diff_sigma = math.sqrt(diff_var)

    mu_trunc, sigma_trunc = trunc(diff_mu, diff_sigma, 0.0, True)
    trunc_var = sigma_trunc * sigma_trunc

    if abs(diff_var - trunc_var) < 1e-10:
        return (0.0, INF_SIGMA, 0.0, INF_SIGMA,
                0.0, INF_SIGMA, 0.0, INF_SIGMA)

    delta_div = (diff_var * mu_trunc - trunc_var * diff_mu) / (diff_var - trunc_var)
    theta_div_sq = (trunc_var * diff_var) / (diff_var - trunc_var)
    team_shift = delta_div - diff_mu

    lik_var_base = theta_div_sq + diff_var - t1_base_sigma * t1_base_sigma
    lik_var_surf = theta_div_sq + diff_var - t1_surf_sigma * t1_surf_sigma
    if lik_var_base < 1e-10:
        lik_var_base = 1e-10
    if lik_var_surf < 1e-10:
        lik_var_surf = 1e-10

    t1_base_lik_sigma = math.sqrt(lik_var_base) / w_base if w_base > 1e-10 else INF_SIGMA
    t1_surf_lik_sigma = math.sqrt(lik_var_surf) / w_surf if w_surf > 1e-10 else INF_SIGMA

    lik_var_base2 = theta_div_sq + diff_var - t2_base_sigma * t2_base_sigma
    lik_var_surf2 = theta_div_sq + diff_var - t2_surf_sigma * t2_surf_sigma
    if lik_var_base2 < 1e-10:
        lik_var_base2 = 1e-10
    if lik_var_surf2 < 1e-10:
        lik_var_surf2 = 1e-10

    t2_base_lik_sigma = math.sqrt(lik_var_base2) / w_base if w_base > 1e-10 else INF_SIGMA
    t2_surf_lik_sigma = math.sqrt(lik_var_surf2) / w_surf if w_surf > 1e-10 else INF_SIGMA

    if t1_wins:
        t1_base_lik_mu = t1_base_mu + team_shift / w_base if w_base > 1e-10 else t1_base_mu
        t1_surf_lik_mu = t1_surf_mu + team_shift / w_surf if w_surf > 1e-10 else t1_surf_mu
        t2_base_lik_mu = t2_base_mu - team_shift / w_base if w_base > 1e-10 else t2_base_mu
        t2_surf_lik_mu = t2_surf_mu - team_shift / w_surf if w_surf > 1e-10 else t2_surf_mu
    else:
        t1_base_lik_mu = t1_base_mu - team_shift / w_base if w_base > 1e-10 else t1_base_mu
        t1_surf_lik_mu = t1_surf_mu - team_shift / w_surf if w_surf > 1e-10 else t1_surf_mu
        t2_base_lik_mu = t2_base_mu + team_shift / w_base if w_base > 1e-10 else t2_base_mu
        t2_surf_lik_mu = t2_surf_mu + team_shift / w_surf if w_surf > 1e-10 else t2_surf_mu

    return (t1_base_lik_mu, t1_base_lik_sigma,
            t1_surf_lik_mu, t1_surf_lik_sigma,
            t2_base_lik_mu, t2_base_lik_sigma,
            t2_surf_lik_mu, t2_surf_lik_sigma)


@njit(cache=True, fastmath=True)
def predict_weighted_team(
    t1_base_mu: float, t1_base_sigma: float,
    t1_surf_mu: float, t1_surf_sigma: float,
    t2_base_mu: float, t2_base_sigma: float,
    t2_surf_mu: float, t2_surf_sigma: float,
    w_base: float, w_surf: float,
    beta: float,
) -> float:
    """Predict probability that team 1 beats team 2."""
    w_base_sq = w_base * w_base
    w_surf_sq = w_surf * w_surf
    beta_sq = beta * beta

    t1_mu = w_base * t1_base_mu + w_surf * t1_surf_mu
    t2_mu = w_base * t2_base_mu + w_surf * t2_surf_mu

    t1_var = w_base_sq * (t1_base_sigma * t1_base_sigma + beta_sq) + \
             w_surf_sq * (t1_surf_sigma * t1_surf_sigma + beta_sq)
    t2_var = w_base_sq * (t2_base_sigma * t2_base_sigma + beta_sq) + \
             w_surf_sq * (t2_surf_sigma * t2_surf_sigma + beta_sq)

    diff_mu = t1_mu - t2_mu
    diff_sigma = math.sqrt(t1_var + t2_var)

    return norm_cdf(diff_mu / diff_sigma)
