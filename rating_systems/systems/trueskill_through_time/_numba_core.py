"""
Numba-accelerated core functions for TrueSkill Through Time (TTT).

This implementation follows the exact structure of the reference implementation
(https://github.com/glandfried/TrueSkillThroughTime.py) but uses Numba for speed.

Key data structures:
- Batches: time steps, each containing games played at that time
- For each player at each batch: forward, backward, likelihood messages
- Messages are Gaussians in (mu, sigma) or precision (pi, tau) form

Algorithm:
1. trueskill(): Initial forward pass creating batches sequentially
2. iteration(): Backward sweep then forward sweep, updating likelihoods at each batch
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
    - alpha = (margin - mu) / sigma = -t
    - v = pdf(-alpha) / cdf(-alpha) = pdf(t) / cdf(t)
    - w = v * (v + (-alpha)) = v * (v + t)
    """
    cdf_t = norm_cdf(t)
    if cdf_t < 1e-10:
        # Asymptotic for large negative t (big upset)
        v = -t
        w = 1.0 - 1e-6
    else:
        v = norm_pdf(t) / cdf_t
        w = v * (v + t)

    # Clamp w to valid range
    if w < 1e-10:
        w = 1e-10
    elif w > 1.0 - 1e-10:
        w = 1.0 - 1e-10

    return v, w


@njit(cache=True, fastmath=True)
def trunc(mu: float, sigma: float, margin: float, is_win: bool) -> tuple:
    """
    Compute truncated Gaussian mean and sigma after observing outcome.

    For a win: observe diff > margin
    For a loss: observe diff < -margin (handled by negating)
    """
    t = (mu - margin) / sigma if sigma > 1e-10 else 0.0

    if is_win:
        v, w = v_w_win(t)
    else:
        # Loss: negate t and v
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

    This follows the reference's likelihood_analitico() for 2-team games.

    Returns: (lik1_mu, lik1_sigma, lik2_mu, lik2_sigma)
    """
    # Performance distributions include beta noise
    perf1_sigma = math.sqrt(p1_sigma * p1_sigma + beta * beta)
    perf2_sigma = math.sqrt(p2_sigma * p2_sigma + beta * beta)

    # Difference distribution: perf1 - perf2 for winner - loser perspective
    if p1_wins:
        diff_mu = p1_mu - p2_mu
    else:
        diff_mu = p2_mu - p1_mu
    diff_sigma = math.sqrt(perf1_sigma * perf1_sigma + perf2_sigma * perf2_sigma)

    # Truncate (observe diff > 0 for the "winner")
    mu_trunc, sigma_trunc = trunc(diff_mu, diff_sigma, 0.0, True)

    if abs(diff_sigma - sigma_trunc) < 1e-10:
        # No update (degenerate case)
        return 0.0, INF_SIGMA, 0.0, INF_SIGMA

    # Compute likelihood parameters (from reference likelihood_analitico)
    diff_var = diff_sigma * diff_sigma
    trunc_var = sigma_trunc * sigma_trunc

    delta_div = (diff_var * mu_trunc - trunc_var * diff_mu) / (diff_var - trunc_var)
    theta_div_sq = (trunc_var * diff_var) / (diff_var - trunc_var)

    # Player likelihoods
    # Note: Reference uses prior sigma, not performance sigma
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
# Weighted team game likelihood computation (for surface-specific TTT)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_weighted_team_likelihoods(
    # Team 1: base player + surface player
    t1_base_mu: float, t1_base_sigma: float,
    t1_surf_mu: float, t1_surf_sigma: float,
    # Team 2: base player + surface player
    t2_base_mu: float, t2_base_sigma: float,
    t2_surf_mu: float, t2_surf_sigma: float,
    # Weights
    w_base: float, w_surf: float,
    # Team 1 wins?
    t1_wins: bool,
    beta: float,
) -> tuple:
    """
    Compute likelihood messages for all 4 players from a weighted team game.

    Team performance = w_base * base_skill + w_surf * surface_skill
    Team perf variance = w_base² * (σ_base² + β²) + w_surf² * (σ_surf² + β²)

    Returns: (t1_base_lik_mu, t1_base_lik_sigma,
              t1_surf_lik_mu, t1_surf_lik_sigma,
              t2_base_lik_mu, t2_base_lik_sigma,
              t2_surf_lik_mu, t2_surf_lik_sigma)
    """
    w_base_sq = w_base * w_base
    w_surf_sq = w_surf * w_surf
    beta_sq = beta * beta

    # Team 1 combined skill
    t1_mu = w_base * t1_base_mu + w_surf * t1_surf_mu
    t1_perf_var = w_base_sq * (t1_base_sigma * t1_base_sigma + beta_sq) + \
                  w_surf_sq * (t1_surf_sigma * t1_surf_sigma + beta_sq)
    t1_perf_sigma = math.sqrt(t1_perf_var)

    # Team 2 combined skill
    t2_mu = w_base * t2_base_mu + w_surf * t2_surf_mu
    t2_perf_var = w_base_sq * (t2_base_sigma * t2_base_sigma + beta_sq) + \
                  w_surf_sq * (t2_surf_sigma * t2_surf_sigma + beta_sq)
    t2_perf_sigma = math.sqrt(t2_perf_var)

    # Difference distribution
    if t1_wins:
        diff_mu = t1_mu - t2_mu
    else:
        diff_mu = t2_mu - t1_mu
    diff_var = t1_perf_var + t2_perf_var
    diff_sigma = math.sqrt(diff_var)

    # Truncate (observe diff > 0 for winner)
    mu_trunc, sigma_trunc = trunc(diff_mu, diff_sigma, 0.0, True)

    trunc_var = sigma_trunc * sigma_trunc

    if abs(diff_var - trunc_var) < 1e-10:
        # No update (degenerate case)
        return (0.0, INF_SIGMA, 0.0, INF_SIGMA,
                0.0, INF_SIGMA, 0.0, INF_SIGMA)

    # Compute the update direction and magnitude
    # delta_div is the posterior mean shift for the difference
    delta_div = (diff_var * mu_trunc - trunc_var * diff_mu) / (diff_var - trunc_var)
    theta_div_sq = (trunc_var * diff_var) / (diff_var - trunc_var)

    # The shift in team means
    team_shift = delta_div - diff_mu

    # Distribute the shift to individual components proportional to weight²/variance contribution
    # Each component's contribution to team perf variance is w² * (σ² + β²)
    t1_base_contrib = w_base_sq * (t1_base_sigma * t1_base_sigma + beta_sq)
    t1_surf_contrib = w_surf_sq * (t1_surf_sigma * t1_surf_sigma + beta_sq)
    t2_base_contrib = w_base_sq * (t2_base_sigma * t2_base_sigma + beta_sq)
    t2_surf_contrib = w_surf_sq * (t2_surf_sigma * t2_surf_sigma + beta_sq)

    total_var = t1_base_contrib + t1_surf_contrib + t2_base_contrib + t2_surf_contrib

    # Likelihood sigma: derived from precision update
    # For weighted contribution, the effective sigma is scaled
    lik_var_base = theta_div_sq + diff_var - t1_base_sigma * t1_base_sigma
    lik_var_surf = theta_div_sq + diff_var - t1_surf_sigma * t1_surf_sigma

    if lik_var_base < 1e-10:
        lik_var_base = 1e-10
    if lik_var_surf < 1e-10:
        lik_var_surf = 1e-10

    # Scale likelihood sigma by weight (smaller weight = weaker update = larger sigma)
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

    # Likelihood means: shift proportional to contribution
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
    """
    Predict probability that team 1 beats team 2.

    Team skill = w_base * base + w_surf * surface
    """
    w_base_sq = w_base * w_base
    w_surf_sq = w_surf * w_surf
    beta_sq = beta * beta

    # Team means
    t1_mu = w_base * t1_base_mu + w_surf * t1_surf_mu
    t2_mu = w_base * t2_base_mu + w_surf * t2_surf_mu

    # Team performance variances
    t1_var = w_base_sq * (t1_base_sigma * t1_base_sigma + beta_sq) + \
             w_surf_sq * (t1_surf_sigma * t1_surf_sigma + beta_sq)
    t2_var = w_base_sq * (t2_base_sigma * t2_base_sigma + beta_sq) + \
             w_surf_sq * (t2_surf_sigma * t2_surf_sigma + beta_sq)

    # Difference
    diff_mu = t1_mu - t2_mu
    diff_sigma = math.sqrt(t1_var + t2_var)

    return norm_cdf(diff_mu / diff_sigma)


# =============================================================================
# Batch processing - core TTT algorithm
# =============================================================================

@njit(cache=True)
def process_batch_games(
    batch_idx: int,
    batch_offsets: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    # Player state at this batch (indexed by player_id)
    player_forward_mu: np.ndarray,
    player_forward_sigma: np.ndarray,
    player_backward_mu: np.ndarray,
    player_backward_sigma: np.ndarray,
    player_likelihood_mu: np.ndarray,
    player_likelihood_sigma: np.ndarray,
    # Parameters
    beta: float,
    prior_mu: float,
    prior_sigma: float,
) -> None:
    """
    Process all games in a batch and update likelihood messages.

    For each game, computes likelihood contributions for both players.
    If a player has multiple games in the batch, likelihoods are multiplied.
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

        # Get player priors (forward * backward, excluding old likelihood)
        p1_prior_mu, p1_prior_sigma = gaussian_mul(
            player_forward_mu[p1], player_forward_sigma[p1],
            player_backward_mu[p1], player_backward_sigma[p1]
        )
        p2_prior_mu, p2_prior_sigma = gaussian_mul(
            player_forward_mu[p2], player_forward_sigma[p2],
            player_backward_mu[p2], player_backward_sigma[p2]
        )

        # Compute game likelihoods
        lik1_mu, lik1_sigma, lik2_mu, lik2_sigma = compute_game_likelihoods(
            p1_prior_mu, p1_prior_sigma,
            p2_prior_mu, p2_prior_sigma,
            p1_wins, beta
        )

        # Multiply into player's total likelihood for this batch
        player_likelihood_mu[p1], player_likelihood_sigma[p1] = gaussian_mul(
            player_likelihood_mu[p1], player_likelihood_sigma[p1],
            lik1_mu, lik1_sigma
        )
        player_likelihood_mu[p2], player_likelihood_sigma[p2] = gaussian_mul(
            player_likelihood_mu[p2], player_likelihood_sigma[p2],
            lik2_mu, lik2_sigma
        )


@njit(cache=True)
def initial_forward_pass(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    # Per-batch, per-player state arrays (batch_idx * num_players + player_id)
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    # Agent state (most recent outgoing message per player)
    agent_message_mu: np.ndarray,
    agent_message_sigma: np.ndarray,
    agent_last_time: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    beta: float,
    gamma: float,
) -> None:
    """
    Initial forward pass: create batches and propagate forward messages.

    This matches the reference's trueskill() method.
    """
    for b in range(num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        batch_time = batch_times[b]

        # Get players in this batch
        for g in range(game_start, game_end):
            p1, p2 = game_p1[g], game_p2[g]

            for p in [p1, p2]:
                idx = b * num_players + p

                # Receive forward message from agent
                if agent_last_time[p] < -1e9:  # First appearance
                    fwd_mu = prior_mu
                    fwd_sigma = prior_sigma
                else:
                    elapsed = batch_time - agent_last_time[p]
                    fwd_mu, fwd_sigma = gaussian_forget(
                        agent_message_mu[p], agent_message_sigma[p],
                        gamma, elapsed
                    )

                state_forward_mu[idx] = fwd_mu
                state_forward_sigma[idx] = fwd_sigma
                state_backward_mu[idx] = 0.0
                state_backward_sigma[idx] = INF_SIGMA

        # Process games in this batch
        # Create temporary arrays for this batch's players
        temp_fwd_mu = np.zeros(num_players)
        temp_fwd_sigma = np.full(num_players, INF_SIGMA)
        temp_bwd_mu = np.zeros(num_players)
        temp_bwd_sigma = np.full(num_players, INF_SIGMA)
        temp_lik_mu = np.zeros(num_players)
        temp_lik_sigma = np.full(num_players, INF_SIGMA)

        # Copy state to temp
        for g in range(game_start, game_end):
            for p in [game_p1[g], game_p2[g]]:
                idx = b * num_players + p
                temp_fwd_mu[p] = state_forward_mu[idx]
                temp_fwd_sigma[p] = state_forward_sigma[idx]
                temp_bwd_mu[p] = state_backward_mu[idx]
                temp_bwd_sigma[p] = state_backward_sigma[idx]

        # Process games
        process_batch_games(
            b, batch_offsets, game_p1, game_p2, game_scores,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            beta, prior_mu, prior_sigma
        )

        # Copy likelihood back to state and update agent messages
        for g in range(game_start, game_end):
            for p in [game_p1[g], game_p2[g]]:
                idx = b * num_players + p
                state_likelihood_mu[idx] = temp_lik_mu[p]
                state_likelihood_sigma[idx] = temp_lik_sigma[p]

                # Update agent: forward_prior_out = forward * likelihood
                fwd_out_mu, fwd_out_sigma = gaussian_mul(
                    temp_fwd_mu[p], temp_fwd_sigma[p],
                    temp_lik_mu[p], temp_lik_sigma[p]
                )
                agent_message_mu[p] = fwd_out_mu
                agent_message_sigma[p] = fwd_out_sigma
                agent_last_time[p] = batch_time


@njit(cache=True)
def backward_sweep(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    agent_message_mu: np.ndarray,
    agent_message_sigma: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    beta: float,
    gamma: float,
) -> float:
    """
    Backward sweep: propagate backward messages from future to past.

    For each batch (in reverse order):
    1. Receive backward message from next batch
    2. Recompute likelihoods

    Returns max change for convergence check.
    """
    max_change = 0.0

    # Reset agent messages
    agent_message_mu[:] = 0.0
    agent_message_sigma[:] = INF_SIGMA

    # Process batches in reverse order (skip the last one)
    for b in range(num_batches - 2, -1, -1):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        batch_time = batch_times[b]
        next_batch_time = batch_times[b + 1]

        # Get backward messages from next batch for players in this batch
        next_game_start = batch_offsets[b + 1]
        next_game_end = batch_offsets[b + 2]

        # Compute backward_prior_out for players in next batch
        for g in range(next_game_start, next_game_end):
            for p in [game_p1[g], game_p2[g]]:
                next_idx = (b + 1) * num_players + p
                # backward_prior_out = (likelihood * backward).forget(gamma, elapsed)
                lik_bwd_mu, lik_bwd_sigma = gaussian_mul(
                    state_likelihood_mu[next_idx], state_likelihood_sigma[next_idx],
                    state_backward_mu[next_idx], state_backward_sigma[next_idx]
                )
                elapsed = next_batch_time - batch_time
                out_mu, out_sigma = gaussian_forget(lik_bwd_mu, lik_bwd_sigma, gamma, elapsed)
                agent_message_mu[p] = out_mu
                agent_message_sigma[p] = out_sigma

        # Update backward messages for this batch
        for g in range(game_start, game_end):
            for p in [game_p1[g], game_p2[g]]:
                idx = b * num_players + p
                old_mu = state_backward_mu[idx]
                state_backward_mu[idx] = agent_message_mu[p]
                state_backward_sigma[idx] = agent_message_sigma[p]
                change = abs(state_backward_mu[idx] - old_mu)
                if change > max_change:
                    max_change = change

        # Recompute likelihoods for this batch
        temp_fwd_mu = np.zeros(num_players)
        temp_fwd_sigma = np.full(num_players, INF_SIGMA)
        temp_bwd_mu = np.zeros(num_players)
        temp_bwd_sigma = np.full(num_players, INF_SIGMA)
        temp_lik_mu = np.zeros(num_players)
        temp_lik_sigma = np.full(num_players, INF_SIGMA)

        for g in range(game_start, game_end):
            for p in [game_p1[g], game_p2[g]]:
                idx = b * num_players + p
                temp_fwd_mu[p] = state_forward_mu[idx]
                temp_fwd_sigma[p] = state_forward_sigma[idx]
                temp_bwd_mu[p] = state_backward_mu[idx]
                temp_bwd_sigma[p] = state_backward_sigma[idx]

        process_batch_games(
            b, batch_offsets, game_p1, game_p2, game_scores,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            beta, prior_mu, prior_sigma
        )

        for g in range(game_start, game_end):
            for p in [game_p1[g], game_p2[g]]:
                idx = b * num_players + p
                state_likelihood_mu[idx] = temp_lik_mu[p]
                state_likelihood_sigma[idx] = temp_lik_sigma[p]

    return max_change


@njit(cache=True)
def forward_sweep(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    agent_message_mu: np.ndarray,
    agent_message_sigma: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    beta: float,
    gamma: float,
) -> float:
    """
    Forward sweep: propagate forward messages from past to future.

    For each batch (in forward order, starting from 1):
    1. Receive forward message from previous batch
    2. Recompute likelihoods

    Returns max change for convergence check.
    """
    max_change = 0.0

    # Reset agent messages
    agent_message_mu[:] = 0.0
    agent_message_sigma[:] = INF_SIGMA

    # Process batches in forward order (skip the first one)
    for b in range(1, num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        batch_time = batch_times[b]
        prev_batch_time = batch_times[b - 1]

        # Get forward messages from previous batch
        prev_game_start = batch_offsets[b - 1]
        prev_game_end = batch_offsets[b]

        for g in range(prev_game_start, prev_game_end):
            for p in [game_p1[g], game_p2[g]]:
                prev_idx = (b - 1) * num_players + p
                # forward_prior_out = forward * likelihood
                fwd_out_mu, fwd_out_sigma = gaussian_mul(
                    state_forward_mu[prev_idx], state_forward_sigma[prev_idx],
                    state_likelihood_mu[prev_idx], state_likelihood_sigma[prev_idx]
                )
                agent_message_mu[p] = fwd_out_mu
                agent_message_sigma[p] = fwd_out_sigma

        # Update forward messages for this batch (with drift)
        for g in range(game_start, game_end):
            for p in [game_p1[g], game_p2[g]]:
                idx = b * num_players + p
                old_mu = state_forward_mu[idx]

                if agent_message_sigma[p] < INF_SIGMA * 0.1:
                    elapsed = batch_time - prev_batch_time
                    fwd_mu, fwd_sigma = gaussian_forget(
                        agent_message_mu[p], agent_message_sigma[p],
                        gamma, elapsed
                    )
                    state_forward_mu[idx] = fwd_mu
                    state_forward_sigma[idx] = fwd_sigma

                change = abs(state_forward_mu[idx] - old_mu)
                if change > max_change:
                    max_change = change

        # Recompute likelihoods for this batch
        temp_fwd_mu = np.zeros(num_players)
        temp_fwd_sigma = np.full(num_players, INF_SIGMA)
        temp_bwd_mu = np.zeros(num_players)
        temp_bwd_sigma = np.full(num_players, INF_SIGMA)
        temp_lik_mu = np.zeros(num_players)
        temp_lik_sigma = np.full(num_players, INF_SIGMA)

        for g in range(game_start, game_end):
            for p in [game_p1[g], game_p2[g]]:
                idx = b * num_players + p
                temp_fwd_mu[p] = state_forward_mu[idx]
                temp_fwd_sigma[p] = state_forward_sigma[idx]
                temp_bwd_mu[p] = state_backward_mu[idx]
                temp_bwd_sigma[p] = state_backward_sigma[idx]

        process_batch_games(
            b, batch_offsets, game_p1, game_p2, game_scores,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            beta, prior_mu, prior_sigma
        )

        for g in range(game_start, game_end):
            for p in [game_p1[g], game_p2[g]]:
                idx = b * num_players + p
                state_likelihood_mu[idx] = temp_lik_mu[p]
                state_likelihood_sigma[idx] = temp_lik_sigma[p]

    return max_change


@njit(cache=True)
def run_convergence(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    agent_message_mu: np.ndarray,
    agent_message_sigma: np.ndarray,
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
        # Backward sweep
        bwd_change = backward_sweep(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            state_forward_mu, state_forward_sigma,
            state_backward_mu, state_backward_sigma,
            state_likelihood_mu, state_likelihood_sigma,
            agent_message_mu, agent_message_sigma,
            prior_mu, prior_sigma, beta, gamma
        )

        # Forward sweep
        fwd_change = forward_sweep(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            state_forward_mu, state_forward_sigma,
            state_backward_mu, state_backward_sigma,
            state_likelihood_mu, state_likelihood_sigma,
            agent_message_mu, agent_message_sigma,
            prior_mu, prior_sigma, beta, gamma
        )

        max_change = max(bwd_change, fwd_change)
        if max_change < epsilon:
            return iteration + 1

    return max_iterations


@njit(cache=True)
def extract_final_ratings(
    num_batches: int,
    batch_offsets: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    num_players: int,
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    player_last_batch: np.ndarray,
    ratings_out: np.ndarray,
    rd_out: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    display_scale: float,
    display_offset: float,
) -> None:
    """
    Extract the most recent rating (posterior) for each player.

    Posterior = forward * backward * likelihood
    """
    # Find last batch for each player
    player_last_batch[:] = -1
    for b in range(num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        for g in range(game_start, game_end):
            player_last_batch[game_p1[g]] = b
            player_last_batch[game_p2[g]] = b

    # Compute posterior for each player
    for p in range(num_players):
        b = player_last_batch[p]
        if b < 0:
            # Player never appeared
            ratings_out[p] = display_offset
            rd_out[p] = prior_sigma * display_scale
        else:
            idx = b * num_players + p

            # Posterior = forward * backward * likelihood
            fwd_bwd_mu, fwd_bwd_sigma = gaussian_mul(
                state_forward_mu[idx], state_forward_sigma[idx],
                state_backward_mu[idx], state_backward_sigma[idx]
            )
            post_mu, post_sigma = gaussian_mul(
                fwd_bwd_mu, fwd_bwd_sigma,
                state_likelihood_mu[idx], state_likelihood_sigma[idx]
            )

            ratings_out[p] = post_mu * display_scale + display_offset
            rd_out[p] = post_sigma * display_scale


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
    """
    Predict probability that player1 beats player2 (batch, parallel).
    """
    n = len(player1)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        p1, p2 = player1[i], player2[i]

        # Convert from display scale to internal scale
        mu1 = (ratings[p1] - display_offset) / display_scale
        mu2 = (ratings[p2] - display_offset) / display_scale
        sigma1 = rd[p1] / display_scale
        sigma2 = rd[p2] / display_scale

        # Performance difference distribution
        diff_mu = mu1 - mu2
        diff_sigma = math.sqrt(sigma1*sigma1 + sigma2*sigma2 + 2.0*beta*beta)

        # P(player1 wins) = P(diff > 0)
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
