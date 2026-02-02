"""
Numba-accelerated core functions for Team-based TrueSkill Through Time.

This module implements the forward-backward message passing algorithm for
surface-specific skill estimation, where each player is modeled as a team
of [base_skill, surface_skill].

Memory optimized with float32 arrays (~12GB vs ~24GB with float64).
"""

import math
import numpy as np
from numba import njit, prange

# Type aliases for clarity
F32 = np.float32

# Constants (float32 compatible)
SQRT2 = np.float32(math.sqrt(2.0))
SQRT2PI = np.float32(math.sqrt(2.0 * math.pi))
INV_SQRT2 = np.float32(1.0 / math.sqrt(2.0))
INF_SIGMA = np.float32(1e6)  # Reduced from 1e10 for float32 stability
MIN_SIGMA = np.float32(1e-6)


# =============================================================================
# Gaussian Utilities
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
def gaussian_multiply(mu1: float, sigma1: float, mu2: float, sigma2: float) -> tuple:
    """
    Multiply two Gaussians: N(mu1, sigma1²) × N(mu2, sigma2²).

    Returns (mu, sigma) of the product distribution.
    """
    if sigma1 > INF_SIGMA * 0.1:
        return mu2, sigma2
    if sigma2 > INF_SIGMA * 0.1:
        return mu1, sigma1

    prec1 = 1.0 / (sigma1 * sigma1)
    prec2 = 1.0 / (sigma2 * sigma2)
    prec = prec1 + prec2
    tau = mu1 * prec1 + mu2 * prec2

    sigma = 1.0 / math.sqrt(prec)
    mu = tau / prec
    return mu, sigma


@njit(cache=True, fastmath=True)
def gaussian_drift(mu: float, sigma: float, gamma: float, elapsed: float) -> tuple:
    """
    Apply skill drift over time: sigma_new = sqrt(sigma² + elapsed × gamma²).

    Models increasing uncertainty as time passes without observations.
    """
    if elapsed <= 0 or sigma > INF_SIGMA * 0.1:
        return mu, sigma
    new_var = sigma * sigma + elapsed * gamma * gamma
    return mu, math.sqrt(new_var)


# =============================================================================
# Truncated Gaussian (Game Outcome Update)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_v_w(t: float) -> tuple:
    """
    Compute v and w for truncated Gaussian update (win case).

    v = pdf(t) / cdf(t)  (mean shift factor)
    w = v × (v + t)      (variance reduction factor)
    """
    cdf_t = norm_cdf(t)
    if cdf_t < 1e-8:
        # Asymptotic approximation for extreme upsets
        return -t, 1.0 - 1e-6

    v = norm_pdf(t) / cdf_t
    w = v * (v + t)

    # Clamp w to valid range
    if w < 1e-8:
        w = 1e-8
    elif w > 1.0 - 1e-8:
        w = 1.0 - 1e-8

    return v, w


@njit(cache=True, fastmath=True)
def truncate_gaussian(mu: float, sigma: float, margin: float, is_win: bool) -> tuple:
    """
    Compute truncated Gaussian after observing game outcome.

    For win: observe (skill_diff > margin)
    For loss: observe (skill_diff < -margin)
    """
    if sigma < MIN_SIGMA:
        return mu, sigma

    t = (mu - margin) / sigma

    if is_win:
        v, w = compute_v_w(t)
    else:
        v, w = compute_v_w(-t)
        v = -v

    mu_trunc = mu + sigma * v
    sigma_trunc = sigma * math.sqrt(max(1.0 - w, MIN_SIGMA))

    return mu_trunc, sigma_trunc


# =============================================================================
# Team Game Likelihood
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_team_likelihoods(
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

    Team performance model:
        team_skill = w_base × base_skill + w_surf × surface_skill
        team_perf_var = w_base² × (σ_base² + β²) + w_surf² × (σ_surf² + β²)

    Returns:
        (t1_base_lik_mu, t1_base_lik_sigma,
         t1_surf_lik_mu, t1_surf_lik_sigma,
         t2_base_lik_mu, t2_base_lik_sigma,
         t2_surf_lik_mu, t2_surf_lik_sigma)
    """
    w_base_sq = w_base * w_base
    w_surf_sq = w_surf * w_surf
    beta_sq = beta * beta

    # Individual performance variances
    t1_base_perf_var = t1_base_sigma * t1_base_sigma + beta_sq
    t1_surf_perf_var = t1_surf_sigma * t1_surf_sigma + beta_sq
    t2_base_perf_var = t2_base_sigma * t2_base_sigma + beta_sq
    t2_surf_perf_var = t2_surf_sigma * t2_surf_sigma + beta_sq

    # Team skill means and variances
    t1_mu = w_base * t1_base_mu + w_surf * t1_surf_mu
    t2_mu = w_base * t2_base_mu + w_surf * t2_surf_mu
    t1_var = w_base_sq * t1_base_perf_var + w_surf_sq * t1_surf_perf_var
    t2_var = w_base_sq * t2_base_perf_var + w_surf_sq * t2_surf_perf_var

    # Difference distribution (winner - loser perspective)
    if t1_wins:
        diff_mu = t1_mu - t2_mu
    else:
        diff_mu = t2_mu - t1_mu

    diff_var = t1_var + t2_var
    diff_sigma = math.sqrt(diff_var)

    # Truncate: observe diff > 0
    mu_trunc, sigma_trunc = truncate_gaussian(diff_mu, diff_sigma, 0.0, True)

    trunc_var = sigma_trunc * sigma_trunc
    if abs(diff_var - trunc_var) < 1e-8:
        # No update (degenerate case)
        return (0.0, INF_SIGMA, 0.0, INF_SIGMA,
                0.0, INF_SIGMA, 0.0, INF_SIGMA)

    # Compute likelihood parameters (from TTT derivation)
    delta_div = (diff_var * mu_trunc - trunc_var * diff_mu) / (diff_var - trunc_var)
    theta_div_sq = (trunc_var * diff_var) / (diff_var - trunc_var)
    team_shift = delta_div - diff_mu

    # Signs for winner/loser
    t1_sign = 1.0 if t1_wins else -1.0
    t2_sign = -t1_sign

    # Distribute shift proportional to variance contribution
    t1_contrib = w_base_sq * t1_base_perf_var + w_surf_sq * t1_surf_perf_var
    t2_contrib = w_base_sq * t2_base_perf_var + w_surf_sq * t2_surf_perf_var

    eps = 1e-8
    if t1_contrib > eps:
        t1_base_share = w_base_sq * t1_base_perf_var / t1_contrib
        t1_surf_share = w_surf_sq * t1_surf_perf_var / t1_contrib
    else:
        t1_base_share = 0.5
        t1_surf_share = 0.5

    if t2_contrib > eps:
        t2_base_share = w_base_sq * t2_base_perf_var / t2_contrib
        t2_surf_share = w_surf_sq * t2_surf_perf_var / t2_contrib
    else:
        t2_base_share = 0.5
        t2_surf_share = 0.5

    # Mean updates (divide by weight to convert from team to player space)
    if w_base > eps:
        t1_base_lik_mu = t1_base_mu + t1_sign * team_shift * t1_base_share / w_base
        t2_base_lik_mu = t2_base_mu + t2_sign * team_shift * t2_base_share / w_base
    else:
        t1_base_lik_mu = t1_base_mu
        t2_base_lik_mu = t2_base_mu

    if w_surf > eps:
        t1_surf_lik_mu = t1_surf_mu + t1_sign * team_shift * t1_surf_share / w_surf
        t2_surf_lik_mu = t2_surf_mu + t2_sign * team_shift * t2_surf_share / w_surf
    else:
        t1_surf_lik_mu = t1_surf_mu
        t2_surf_lik_mu = t2_surf_mu

    # Sigma updates (scaled by weight)
    if w_base > eps:
        t1_base_lik_sigma_sq = theta_div_sq / w_base_sq + diff_var - t1_base_sigma * t1_base_sigma
        t2_base_lik_sigma_sq = theta_div_sq / w_base_sq + diff_var - t2_base_sigma * t2_base_sigma
        t1_base_lik_sigma = math.sqrt(max(t1_base_lik_sigma_sq, MIN_SIGMA))
        t2_base_lik_sigma = math.sqrt(max(t2_base_lik_sigma_sq, MIN_SIGMA))
    else:
        t1_base_lik_sigma = INF_SIGMA
        t2_base_lik_sigma = INF_SIGMA

    if w_surf > eps:
        t1_surf_lik_sigma_sq = theta_div_sq / w_surf_sq + diff_var - t1_surf_sigma * t1_surf_sigma
        t2_surf_lik_sigma_sq = theta_div_sq / w_surf_sq + diff_var - t2_surf_sigma * t2_surf_sigma
        t1_surf_lik_sigma = math.sqrt(max(t1_surf_lik_sigma_sq, MIN_SIGMA))
        t2_surf_lik_sigma = math.sqrt(max(t2_surf_lik_sigma_sq, MIN_SIGMA))
    else:
        t1_surf_lik_sigma = INF_SIGMA
        t2_surf_lik_sigma = INF_SIGMA

    return (t1_base_lik_mu, t1_base_lik_sigma,
            t1_surf_lik_mu, t1_surf_lik_sigma,
            t2_base_lik_mu, t2_base_lik_sigma,
            t2_surf_lik_mu, t2_surf_lik_sigma)


@njit(cache=True, fastmath=True)
def predict_team_match(
    t1_base_mu: float, t1_base_sigma: float,
    t1_surf_mu: float, t1_surf_sigma: float,
    t2_base_mu: float, t2_base_sigma: float,
    t2_surf_mu: float, t2_surf_sigma: float,
    w_base: float, w_surf: float,
    beta: float,
) -> float:
    """Predict P(team1 wins) for a weighted team match."""
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


# =============================================================================
# Batch Processing
# =============================================================================

@njit(cache=True)
def process_batch_games(
    game_start: int,
    game_end: int,
    game_t1_base: np.ndarray,
    game_t1_surf: np.ndarray,
    game_t2_base: np.ndarray,
    game_t2_surf: np.ndarray,
    game_scores: np.ndarray,
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    w_base: float,
    w_surf: float,
    beta: float,
) -> None:
    """
    Process all games in a batch and update likelihood messages in-place.

    State arrays are indexed directly by player ID for this batch's slice.
    """
    # Reset likelihoods for players in this batch
    for g in range(game_start, game_end):
        t1b, t1s = game_t1_base[g], game_t1_surf[g]
        t2b, t2s = game_t2_base[g], game_t2_surf[g]
        state_likelihood_mu[t1b] = 0.0
        state_likelihood_sigma[t1b] = INF_SIGMA
        state_likelihood_mu[t1s] = 0.0
        state_likelihood_sigma[t1s] = INF_SIGMA
        state_likelihood_mu[t2b] = 0.0
        state_likelihood_sigma[t2b] = INF_SIGMA
        state_likelihood_mu[t2s] = 0.0
        state_likelihood_sigma[t2s] = INF_SIGMA

    # Process each game
    for g in range(game_start, game_end):
        t1b, t1s = game_t1_base[g], game_t1_surf[g]
        t2b, t2s = game_t2_base[g], game_t2_surf[g]
        t1_wins = game_scores[g] > 0.5

        # Compute priors: forward × backward
        t1b_prior_mu, t1b_prior_sigma = gaussian_multiply(
            state_forward_mu[t1b], state_forward_sigma[t1b],
            state_backward_mu[t1b], state_backward_sigma[t1b]
        )
        t1s_prior_mu, t1s_prior_sigma = gaussian_multiply(
            state_forward_mu[t1s], state_forward_sigma[t1s],
            state_backward_mu[t1s], state_backward_sigma[t1s]
        )
        t2b_prior_mu, t2b_prior_sigma = gaussian_multiply(
            state_forward_mu[t2b], state_forward_sigma[t2b],
            state_backward_mu[t2b], state_backward_sigma[t2b]
        )
        t2s_prior_mu, t2s_prior_sigma = gaussian_multiply(
            state_forward_mu[t2s], state_forward_sigma[t2s],
            state_backward_mu[t2s], state_backward_sigma[t2s]
        )

        # Compute likelihoods from game outcome
        (lik_t1b_mu, lik_t1b_sigma,
         lik_t1s_mu, lik_t1s_sigma,
         lik_t2b_mu, lik_t2b_sigma,
         lik_t2s_mu, lik_t2s_sigma) = compute_team_likelihoods(
            t1b_prior_mu, t1b_prior_sigma,
            t1s_prior_mu, t1s_prior_sigma,
            t2b_prior_mu, t2b_prior_sigma,
            t2s_prior_mu, t2s_prior_sigma,
            w_base, w_surf, t1_wins, beta
        )

        # Multiply into cumulative likelihood for this batch
        state_likelihood_mu[t1b], state_likelihood_sigma[t1b] = gaussian_multiply(
            state_likelihood_mu[t1b], state_likelihood_sigma[t1b],
            lik_t1b_mu, lik_t1b_sigma
        )
        state_likelihood_mu[t1s], state_likelihood_sigma[t1s] = gaussian_multiply(
            state_likelihood_mu[t1s], state_likelihood_sigma[t1s],
            lik_t1s_mu, lik_t1s_sigma
        )
        state_likelihood_mu[t2b], state_likelihood_sigma[t2b] = gaussian_multiply(
            state_likelihood_mu[t2b], state_likelihood_sigma[t2b],
            lik_t2b_mu, lik_t2b_sigma
        )
        state_likelihood_mu[t2s], state_likelihood_sigma[t2s] = gaussian_multiply(
            state_likelihood_mu[t2s], state_likelihood_sigma[t2s],
            lik_t2s_mu, lik_t2s_sigma
        )


# =============================================================================
# Forward-Backward Algorithm
# =============================================================================

@njit(cache=True)
def initial_forward_pass(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_t1_base: np.ndarray,
    game_t1_surf: np.ndarray,
    game_t2_base: np.ndarray,
    game_t2_surf: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    num_base_players: int,
    # State arrays: [batch × num_players + player]
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    # Agent state (per player)
    agent_message_mu: np.ndarray,
    agent_message_sigma: np.ndarray,
    agent_last_time: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    surface_prior_sigma: float,
    w_base: float,
    w_surf: float,
    beta: float,
    gamma: float,
    surface_gamma: float,
) -> None:
    """
    Initial forward pass: create batches and propagate forward messages.

    Uses different sigma and gamma for base vs surface players:
    - Players [0, num_base_players): base players with prior_sigma, gamma
    - Players [num_base_players, num_players): surface players with surface_prior_sigma, surface_gamma
    """
    for b in range(num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        batch_time = batch_times[b]
        batch_offset = b * num_players

        # Initialize state for players in this batch
        for g in range(game_start, game_end):
            t1b, t1s = game_t1_base[g], game_t1_surf[g]
            t2b, t2s = game_t2_base[g], game_t2_surf[g]

            for p in (t1b, t1s, t2b, t2s):
                idx = batch_offset + p
                is_surface = p >= num_base_players
                p_sigma = surface_prior_sigma if is_surface else prior_sigma
                p_gamma = surface_gamma if is_surface else gamma

                # Forward message from agent (with drift)
                if agent_last_time[p] < -1e9:
                    fwd_mu = prior_mu
                    fwd_sigma = p_sigma
                else:
                    elapsed = batch_time - agent_last_time[p]
                    fwd_mu, fwd_sigma = gaussian_drift(
                        agent_message_mu[p], agent_message_sigma[p],
                        p_gamma, elapsed
                    )

                state_forward_mu[idx] = fwd_mu
                state_forward_sigma[idx] = fwd_sigma
                state_backward_mu[idx] = 0.0
                state_backward_sigma[idx] = INF_SIGMA

        # Process games using batch slice views
        fwd_mu = state_forward_mu[batch_offset:batch_offset + num_players]
        fwd_sigma = state_forward_sigma[batch_offset:batch_offset + num_players]
        bwd_mu = state_backward_mu[batch_offset:batch_offset + num_players]
        bwd_sigma = state_backward_sigma[batch_offset:batch_offset + num_players]
        lik_mu = state_likelihood_mu[batch_offset:batch_offset + num_players]
        lik_sigma = state_likelihood_sigma[batch_offset:batch_offset + num_players]

        process_batch_games(
            game_start, game_end,
            game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
            fwd_mu, fwd_sigma, bwd_mu, bwd_sigma, lik_mu, lik_sigma,
            w_base, w_surf, beta
        )

        # Update agent messages: forward_out = forward × likelihood
        for g in range(game_start, game_end):
            for p in (game_t1_base[g], game_t1_surf[g], game_t2_base[g], game_t2_surf[g]):
                fwd_out_mu, fwd_out_sigma = gaussian_multiply(
                    fwd_mu[p], fwd_sigma[p], lik_mu[p], lik_sigma[p]
                )
                agent_message_mu[p] = fwd_out_mu
                agent_message_sigma[p] = fwd_out_sigma
                agent_last_time[p] = batch_time


@njit(cache=True)
def backward_sweep(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_t1_base: np.ndarray,
    game_t1_surf: np.ndarray,
    game_t2_base: np.ndarray,
    game_t2_surf: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    num_base_players: int,
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    agent_message_mu: np.ndarray,
    agent_message_sigma: np.ndarray,
    w_base: float,
    w_surf: float,
    beta: float,
    gamma: float,
    surface_gamma: float,
) -> float:
    """
    Backward sweep: propagate messages from future to past.

    Returns max change for convergence check.
    """
    max_change = np.float32(0.0)

    # Reset agent messages
    agent_message_mu[:] = 0.0
    agent_message_sigma[:] = INF_SIGMA

    for b in range(num_batches - 2, -1, -1):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        batch_time = batch_times[b]
        next_batch_time = batch_times[b + 1]
        batch_offset = b * num_players
        next_batch_offset = (b + 1) * num_players

        next_game_start = batch_offsets[b + 1]
        next_game_end = batch_offsets[b + 2]

        # Compute backward_out for players in next batch
        for g in range(next_game_start, next_game_end):
            for p in (game_t1_base[g], game_t1_surf[g], game_t2_base[g], game_t2_surf[g]):
                next_idx = next_batch_offset + p
                is_surface = p >= num_base_players
                p_gamma = surface_gamma if is_surface else gamma

                # backward_out = (likelihood × backward).drift()
                lik_bwd_mu, lik_bwd_sigma = gaussian_multiply(
                    state_likelihood_mu[next_idx], state_likelihood_sigma[next_idx],
                    state_backward_mu[next_idx], state_backward_sigma[next_idx]
                )
                elapsed = next_batch_time - batch_time
                out_mu, out_sigma = gaussian_drift(lik_bwd_mu, lik_bwd_sigma, p_gamma, elapsed)
                agent_message_mu[p] = out_mu
                agent_message_sigma[p] = out_sigma

        # Update backward messages for this batch
        for g in range(game_start, game_end):
            for p in (game_t1_base[g], game_t1_surf[g], game_t2_base[g], game_t2_surf[g]):
                idx = batch_offset + p
                old_mu = state_backward_mu[idx]
                state_backward_mu[idx] = agent_message_mu[p]
                state_backward_sigma[idx] = agent_message_sigma[p]
                change = abs(state_backward_mu[idx] - old_mu)
                if change > max_change:
                    max_change = change

        # Recompute likelihoods with updated backward messages
        fwd_mu = state_forward_mu[batch_offset:batch_offset + num_players]
        fwd_sigma = state_forward_sigma[batch_offset:batch_offset + num_players]
        bwd_mu = state_backward_mu[batch_offset:batch_offset + num_players]
        bwd_sigma = state_backward_sigma[batch_offset:batch_offset + num_players]
        lik_mu = state_likelihood_mu[batch_offset:batch_offset + num_players]
        lik_sigma = state_likelihood_sigma[batch_offset:batch_offset + num_players]

        process_batch_games(
            game_start, game_end,
            game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
            fwd_mu, fwd_sigma, bwd_mu, bwd_sigma, lik_mu, lik_sigma,
            w_base, w_surf, beta
        )

    return max_change


@njit(cache=True)
def forward_sweep(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_t1_base: np.ndarray,
    game_t1_surf: np.ndarray,
    game_t2_base: np.ndarray,
    game_t2_surf: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    num_base_players: int,
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    agent_message_mu: np.ndarray,
    agent_message_sigma: np.ndarray,
    w_base: float,
    w_surf: float,
    beta: float,
    gamma: float,
    surface_gamma: float,
) -> float:
    """
    Forward sweep: propagate messages from past to future.

    Returns max change for convergence check.
    """
    max_change = np.float32(0.0)

    # Reset agent messages
    agent_message_mu[:] = 0.0
    agent_message_sigma[:] = INF_SIGMA

    for b in range(1, num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        batch_time = batch_times[b]
        prev_batch_time = batch_times[b - 1]
        batch_offset = b * num_players
        prev_batch_offset = (b - 1) * num_players

        prev_game_start = batch_offsets[b - 1]
        prev_game_end = batch_offsets[b]

        # Get forward messages from previous batch
        for g in range(prev_game_start, prev_game_end):
            for p in (game_t1_base[g], game_t1_surf[g], game_t2_base[g], game_t2_surf[g]):
                prev_idx = prev_batch_offset + p
                # forward_out = forward × likelihood
                fwd_out_mu, fwd_out_sigma = gaussian_multiply(
                    state_forward_mu[prev_idx], state_forward_sigma[prev_idx],
                    state_likelihood_mu[prev_idx], state_likelihood_sigma[prev_idx]
                )
                agent_message_mu[p] = fwd_out_mu
                agent_message_sigma[p] = fwd_out_sigma

        # Update forward messages for this batch
        for g in range(game_start, game_end):
            for p in (game_t1_base[g], game_t1_surf[g], game_t2_base[g], game_t2_surf[g]):
                idx = batch_offset + p
                old_mu = state_forward_mu[idx]
                is_surface = p >= num_base_players
                p_gamma = surface_gamma if is_surface else gamma

                if agent_message_sigma[p] < INF_SIGMA * 0.1:
                    elapsed = batch_time - prev_batch_time
                    fwd_mu, fwd_sigma = gaussian_drift(
                        agent_message_mu[p], agent_message_sigma[p],
                        p_gamma, elapsed
                    )
                    state_forward_mu[idx] = fwd_mu
                    state_forward_sigma[idx] = fwd_sigma

                change = abs(state_forward_mu[idx] - old_mu)
                if change > max_change:
                    max_change = change

        # Recompute likelihoods
        fwd_mu = state_forward_mu[batch_offset:batch_offset + num_players]
        fwd_sigma = state_forward_sigma[batch_offset:batch_offset + num_players]
        bwd_mu = state_backward_mu[batch_offset:batch_offset + num_players]
        bwd_sigma = state_backward_sigma[batch_offset:batch_offset + num_players]
        lik_mu = state_likelihood_mu[batch_offset:batch_offset + num_players]
        lik_sigma = state_likelihood_sigma[batch_offset:batch_offset + num_players]

        process_batch_games(
            game_start, game_end,
            game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
            fwd_mu, fwd_sigma, bwd_mu, bwd_sigma, lik_mu, lik_sigma,
            w_base, w_surf, beta
        )

    return max_change


@njit(cache=True)
def run_convergence(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_t1_base: np.ndarray,
    game_t1_surf: np.ndarray,
    game_t2_base: np.ndarray,
    game_t2_surf: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    num_base_players: int,
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    agent_message_mu: np.ndarray,
    agent_message_sigma: np.ndarray,
    w_base: float,
    w_surf: float,
    beta: float,
    gamma: float,
    surface_gamma: float,
    max_iterations: int,
    epsilon: float,
) -> int:
    """
    Run forward-backward iterations until convergence.

    Returns number of iterations performed.
    """
    for iteration in range(max_iterations):
        bwd_change = backward_sweep(
            num_batches, batch_offsets, batch_times,
            game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
            num_players, num_base_players,
            state_forward_mu, state_forward_sigma,
            state_backward_mu, state_backward_sigma,
            state_likelihood_mu, state_likelihood_sigma,
            agent_message_mu, agent_message_sigma,
            w_base, w_surf, beta, gamma, surface_gamma
        )

        fwd_change = forward_sweep(
            num_batches, batch_offsets, batch_times,
            game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
            num_players, num_base_players,
            state_forward_mu, state_forward_sigma,
            state_backward_mu, state_backward_sigma,
            state_likelihood_mu, state_likelihood_sigma,
            agent_message_mu, agent_message_sigma,
            w_base, w_surf, beta, gamma, surface_gamma
        )

        if max(bwd_change, fwd_change) < epsilon:
            return iteration + 1

    return max_iterations


# =============================================================================
# Rating Extraction and Prediction
# =============================================================================

@njit(cache=True)
def extract_final_ratings(
    num_batches: int,
    batch_offsets: np.ndarray,
    game_t1_base: np.ndarray,
    game_t1_surf: np.ndarray,
    game_t2_base: np.ndarray,
    game_t2_surf: np.ndarray,
    num_players: int,
    num_base_players: int,
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
    surface_prior_sigma: float,
    display_scale: float,
    display_offset: float,
) -> None:
    """Extract final ratings (posterior) for all players."""
    # Find last batch for each player
    for p in range(num_players):
        player_last_batch[p] = -1

    for b in range(num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        for g in range(game_start, game_end):
            player_last_batch[game_t1_base[g]] = b
            player_last_batch[game_t1_surf[g]] = b
            player_last_batch[game_t2_base[g]] = b
            player_last_batch[game_t2_surf[g]] = b

    # Compute posterior for each player
    for p in range(num_players):
        b = player_last_batch[p]
        is_surface = p >= num_base_players
        p_sigma = surface_prior_sigma if is_surface else prior_sigma

        if b < 0:
            ratings_out[p] = display_offset
            rd_out[p] = p_sigma * display_scale
        else:
            idx = b * num_players + p

            # Posterior = forward × backward × likelihood
            fwd_bwd_mu, fwd_bwd_sigma = gaussian_multiply(
                state_forward_mu[idx], state_forward_sigma[idx],
                state_backward_mu[idx], state_backward_sigma[idx]
            )
            post_mu, post_sigma = gaussian_multiply(
                fwd_bwd_mu, fwd_bwd_sigma,
                state_likelihood_mu[idx], state_likelihood_sigma[idx]
            )

            ratings_out[p] = post_mu * display_scale + display_offset
            rd_out[p] = post_sigma * display_scale


@njit(cache=True, parallel=True)
def predict_proba_batch(
    t1_base: np.ndarray,
    t1_surf: np.ndarray,
    t2_base: np.ndarray,
    t2_surf: np.ndarray,
    ratings: np.ndarray,
    rd: np.ndarray,
    w_base: float,
    w_surf: float,
    beta: float,
    display_scale: float,
    display_offset: float,
) -> np.ndarray:
    """Batch prediction for team games (parallelized)."""
    n = len(t1_base)
    result = np.empty(n, dtype=np.float32)

    for i in prange(n):
        t1_base_mu = (ratings[t1_base[i]] - display_offset) / display_scale
        t1_base_sigma = rd[t1_base[i]] / display_scale
        t1_surf_mu = (ratings[t1_surf[i]] - display_offset) / display_scale
        t1_surf_sigma = rd[t1_surf[i]] / display_scale
        t2_base_mu = (ratings[t2_base[i]] - display_offset) / display_scale
        t2_base_sigma = rd[t2_base[i]] / display_scale
        t2_surf_mu = (ratings[t2_surf[i]] - display_offset) / display_scale
        t2_surf_sigma = rd[t2_surf[i]] / display_scale

        result[i] = predict_team_match(
            t1_base_mu, t1_base_sigma,
            t1_surf_mu, t1_surf_sigma,
            t2_base_mu, t2_base_sigma,
            t2_surf_mu, t2_surf_sigma,
            w_base, w_surf, beta
        )

    return result
