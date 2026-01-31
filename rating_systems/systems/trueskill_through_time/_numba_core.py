"""
Numba-accelerated core functions for TrueSkill Through Time (TTT).

Uses CSR-like data structures for efficient Numba compilation:
- Player timelines stored as contiguous arrays with offset indices
- Games per player-day stored similarly
- All hot paths JIT-compiled with caching

Key optimizations:
- Custom norm_cdf/norm_pdf using math.erf (no scipy dependency)
- Single Numba call for entire iteration loop
- Parallel predictions via prange
"""

import math
import numpy as np
from numba import njit, prange

# Constants
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
INV_SQRT2 = 1.0 / SQRT2


@njit(cache=True, fastmath=True)
def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / SQRT2PI


@njit(cache=True, fastmath=True)
def norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x * INV_SQRT2))


@njit(cache=True, fastmath=True)
def v_win(t: float, eps: float = 1e-10) -> float:
    """
    Compute v for win/loss outcome (truncated Gaussian mean correction).

    v = pdf(t) / cdf(-t) = pdf(t) / (1 - cdf(t))
    """
    denom = 1.0 - norm_cdf(t)
    if denom < eps:
        # Asymptotic approximation for large t
        if t > 5.0:
            return t + 1.0 / t
        return 10.0
    return norm_pdf(t) / denom


@njit(cache=True, fastmath=True)
def w_win(t: float, v: float) -> float:
    """
    Compute w for win/loss outcome (variance correction).

    w = v * (v + t), clamped to [0, 1-eps]
    """
    w = v * (v + t)
    if w < 0.0:
        return 0.0
    if w > 1.0 - 1e-6:
        return 1.0 - 1e-6
    return w


@njit(cache=True, fastmath=True)
def compute_drift_sigma(elapsed: int, gamma: float, sigma: float) -> float:
    """Compute additional sigma due to skill drift over time."""
    if elapsed <= 0:
        return 0.0
    # Cap drift at 1.67 * sigma to prevent unbounded growth
    return min(math.sqrt(float(elapsed)) * gamma, 1.67 * sigma)


@njit(cache=True, fastmath=True)
def add_variances(sigma1: float, sigma2: float) -> float:
    """Add two independent Gaussian variances: sqrt(σ1² + σ2²)."""
    return math.sqrt(sigma1 * sigma1 + sigma2 * sigma2)


@njit(cache=True, fastmath=True)
def get_posterior_from_messages(
    forward_mu: float,
    forward_sigma: float,
    backward_mu: float,
    backward_sigma: float,
    likelihood_pi: float,
    likelihood_tau: float,
    prior_mu: float,
    prior_sigma: float,
) -> tuple:
    """
    Compute posterior mean and sigma from forward/backward messages and likelihood.

    Returns (mu, sigma) of the posterior distribution.
    """
    # Convert messages to precision form
    fwd_prec = 1.0 / (forward_sigma * forward_sigma) if forward_sigma < 1e5 else 0.0
    bwd_prec = 1.0 / (backward_sigma * backward_sigma) if backward_sigma < 1e5 else 0.0

    total_prec = fwd_prec + bwd_prec + likelihood_pi

    if total_prec < 1e-10:
        return prior_mu, prior_sigma

    total_tau = fwd_prec * forward_mu + bwd_prec * backward_mu + likelihood_tau

    mu = total_tau / total_prec
    sigma = 1.0 / math.sqrt(total_prec)

    return mu, sigma


@njit(cache=True, fastmath=True)
def get_prior_from_messages(
    forward_mu: float,
    forward_sigma: float,
    backward_mu: float,
    backward_sigma: float,
    prior_mu: float,
    prior_sigma: float,
) -> tuple:
    """
    Compute prior mean and sigma from forward/backward messages only (no likelihood).

    Used when computing likelihood updates to avoid circular dependency.
    """
    fwd_prec = 1.0 / (forward_sigma * forward_sigma) if forward_sigma < 1e5 else 0.0
    bwd_prec = 1.0 / (backward_sigma * backward_sigma) if backward_sigma < 1e5 else 0.0

    total_prec = fwd_prec + bwd_prec

    if total_prec < 1e-10:
        return prior_mu, prior_sigma

    total_tau = fwd_prec * forward_mu + bwd_prec * backward_mu

    mu = total_tau / total_prec
    sigma = 1.0 / math.sqrt(total_prec)

    return mu, sigma


@njit(cache=True, fastmath=True)
def forward_pass(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_forward_mu: np.ndarray,
    pd_forward_sigma: np.ndarray,
    pd_backward_mu: np.ndarray,
    pd_backward_sigma: np.ndarray,
    pd_likelihood_pi: np.ndarray,
    pd_likelihood_tau: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    gamma: float,
) -> float:
    """
    Run forward pass: propagate beliefs from past to future.

    Returns maximum change in any forward message (for convergence check).
    """
    max_change = 0.0

    for player_id in range(num_players):
        pd_start = player_offsets[player_id]
        pd_end = player_offsets[player_id + 1]
        n = pd_end - pd_start

        if n == 0:
            continue

        # Start with prior
        prev_mu = prior_mu
        prev_sigma = prior_sigma
        prev_day = pd_days[pd_start]

        for i in range(n):
            pd_idx = pd_start + i
            day = pd_days[pd_idx]

            # Apply drift since previous appearance
            elapsed = day - prev_day
            drift_sigma = compute_drift_sigma(elapsed, gamma, prior_sigma)
            msg_sigma = add_variances(prev_sigma, drift_sigma)

            # Update forward message
            old_mu = pd_forward_mu[pd_idx]
            pd_forward_mu[pd_idx] = prev_mu
            pd_forward_sigma[pd_idx] = msg_sigma

            change = abs(pd_forward_mu[pd_idx] - old_mu)
            if change > max_change:
                max_change = change

            # Compute current posterior for next iteration
            prev_mu, prev_sigma = get_posterior_from_messages(
                pd_forward_mu[pd_idx], pd_forward_sigma[pd_idx],
                pd_backward_mu[pd_idx], pd_backward_sigma[pd_idx],
                pd_likelihood_pi[pd_idx], pd_likelihood_tau[pd_idx],
                prior_mu, prior_sigma,
            )
            prev_day = day

    return max_change


@njit(cache=True, fastmath=True)
def backward_pass(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_forward_mu: np.ndarray,
    pd_forward_sigma: np.ndarray,
    pd_backward_mu: np.ndarray,
    pd_backward_sigma: np.ndarray,
    pd_likelihood_pi: np.ndarray,
    pd_likelihood_tau: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    gamma: float,
) -> float:
    """
    Run backward pass: propagate beliefs from future to past.

    Returns maximum change in any backward message.
    """
    max_change = 0.0

    for player_id in range(num_players):
        pd_start = player_offsets[player_id]
        pd_end = player_offsets[player_id + 1]
        n = pd_end - pd_start

        if n == 0:
            continue

        # Start uninformative (from the future)
        prev_mu = 0.0
        prev_sigma = 1e6
        prev_day = pd_days[pd_end - 1]

        # Iterate backwards through timeline
        for i in range(n - 1, -1, -1):
            pd_idx = pd_start + i
            day = pd_days[pd_idx]

            # Apply drift since next appearance
            elapsed = prev_day - day
            drift_sigma = compute_drift_sigma(elapsed, gamma, prior_sigma)
            msg_sigma = add_variances(prev_sigma, drift_sigma)

            # Update backward message
            old_mu = pd_backward_mu[pd_idx]
            pd_backward_mu[pd_idx] = prev_mu
            pd_backward_sigma[pd_idx] = msg_sigma

            change = abs(pd_backward_mu[pd_idx] - old_mu)
            if change > max_change:
                max_change = change

            # Compute current posterior for next iteration
            prev_mu, prev_sigma = get_posterior_from_messages(
                pd_forward_mu[pd_idx], pd_forward_sigma[pd_idx],
                pd_backward_mu[pd_idx], pd_backward_sigma[pd_idx],
                pd_likelihood_pi[pd_idx], pd_likelihood_tau[pd_idx],
                prior_mu, prior_sigma,
            )
            prev_day = day

    return max_change


@njit(cache=True, fastmath=True)
def update_likelihoods(
    num_players: int,
    player_offsets: np.ndarray,
    pd_forward_mu: np.ndarray,
    pd_forward_sigma: np.ndarray,
    pd_backward_mu: np.ndarray,
    pd_backward_sigma: np.ndarray,
    pd_likelihood_pi: np.ndarray,
    pd_likelihood_tau: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    beta: float,
) -> float:
    """
    Update likelihood messages from games.

    This is the computational hot path - processes all games to update
    the likelihood precision (pi) and precision-weighted mean (tau).

    Returns maximum change in any likelihood.
    """
    max_change = 0.0
    beta_sq = beta * beta

    for player_id in range(num_players):
        pd_start = player_offsets[player_id]
        pd_end = player_offsets[player_id + 1]

        for pd_idx in range(pd_start, pd_end):
            game_start = pd_game_offsets[pd_idx]
            game_end = pd_game_offsets[pd_idx + 1]

            if game_start == game_end:
                # No games at this player-day
                old_pi = pd_likelihood_pi[pd_idx]
                pd_likelihood_pi[pd_idx] = 0.0
                pd_likelihood_tau[pd_idx] = 0.0
                change = abs(pd_likelihood_pi[pd_idx] - old_pi)
                if change > max_change:
                    max_change = change
                continue

            # Get my prior from messages only (excluding current likelihood)
            my_mu, my_sigma = get_prior_from_messages(
                pd_forward_mu[pd_idx], pd_forward_sigma[pd_idx],
                pd_backward_mu[pd_idx], pd_backward_sigma[pd_idx],
                prior_mu, prior_sigma,
            )
            my_var = my_sigma * my_sigma

            # Aggregate likelihood updates from all games
            total_pi = 0.0
            total_tau = 0.0

            for g in range(game_start, game_end):
                opp_pd = pd_game_opp_pd[g]
                score = pd_game_score[g]

                # Get opponent's prior from messages
                opp_mu, opp_sigma = get_prior_from_messages(
                    pd_forward_mu[opp_pd], pd_forward_sigma[opp_pd],
                    pd_backward_mu[opp_pd], pd_backward_sigma[opp_pd],
                    prior_mu, prior_sigma,
                )
                opp_var = opp_sigma * opp_sigma

                # Performance difference distribution
                # d ~ N(my_mu - opp_mu, my_var + opp_var + 2*beta²)
                diff_mu = my_mu - opp_mu
                diff_var = my_var + opp_var + 2.0 * beta_sq
                diff_sigma = math.sqrt(diff_var)

                if diff_sigma < 1e-10:
                    continue

                t = diff_mu / diff_sigma

                # Compute v and w based on outcome
                if score > 0.5:  # Win
                    v = v_win(t)
                    w = w_win(t, v)
                else:  # Loss (t becomes -t)
                    v = -v_win(-t)
                    w = w_win(-t, v_win(-t))

                # Clamp w for numerical stability
                if w < 1e-6:
                    w = 1e-6
                elif w > 1.0 - 1e-6:
                    w = 1.0 - 1e-6

                # My contribution factor to the difference
                c = (my_var + beta_sq) / diff_var

                # Truncated Gaussian update
                new_mu = my_mu + c * diff_sigma * v
                new_var = my_var * (1.0 - c * w)
                if new_var < 1e-6:
                    new_var = 1e-6

                # Convert to precision form: this game's contribution
                game_pi = 1.0 / new_var - 1.0 / my_var
                game_tau = new_mu / new_var - my_mu / my_var

                # Clamp for stability
                if game_pi < 0.0:
                    game_pi = 0.0
                elif game_pi > 10.0:
                    game_pi = 10.0
                if game_tau < -10.0:
                    game_tau = -10.0
                elif game_tau > 10.0:
                    game_tau = 10.0

                total_pi += game_pi
                total_tau += game_tau

            old_pi = pd_likelihood_pi[pd_idx]
            pd_likelihood_pi[pd_idx] = total_pi
            pd_likelihood_tau[pd_idx] = total_tau

            change = abs(pd_likelihood_pi[pd_idx] - old_pi)
            if change > max_change:
                max_change = change

    return max_change


@njit(cache=True, fastmath=True)
def run_all_iterations(
    num_players: int,
    player_offsets: np.ndarray,
    pd_days: np.ndarray,
    pd_forward_mu: np.ndarray,
    pd_forward_sigma: np.ndarray,
    pd_backward_mu: np.ndarray,
    pd_backward_sigma: np.ndarray,
    pd_likelihood_pi: np.ndarray,
    pd_likelihood_tau: np.ndarray,
    pd_game_offsets: np.ndarray,
    pd_game_opp_pd: np.ndarray,
    pd_game_score: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    beta: float,
    gamma: float,
    max_iterations: int,
    convergence_threshold: float,
) -> int:
    """
    Run forward-backward iterations until convergence or max iterations.

    Each iteration: Forward → Likelihood → Backward → Likelihood

    Returns the number of iterations performed.
    """
    for iteration in range(max_iterations):
        # Forward pass
        fwd_change = forward_pass(
            num_players, player_offsets, pd_days,
            pd_forward_mu, pd_forward_sigma,
            pd_backward_mu, pd_backward_sigma,
            pd_likelihood_pi, pd_likelihood_tau,
            prior_mu, prior_sigma, gamma,
        )

        # Likelihood update
        lik_change1 = update_likelihoods(
            num_players, player_offsets,
            pd_forward_mu, pd_forward_sigma,
            pd_backward_mu, pd_backward_sigma,
            pd_likelihood_pi, pd_likelihood_tau,
            pd_game_offsets, pd_game_opp_pd, pd_game_score,
            prior_mu, prior_sigma, beta,
        )

        # Backward pass
        bwd_change = backward_pass(
            num_players, player_offsets, pd_days,
            pd_forward_mu, pd_forward_sigma,
            pd_backward_mu, pd_backward_sigma,
            pd_likelihood_pi, pd_likelihood_tau,
            prior_mu, prior_sigma, gamma,
        )

        # Likelihood update again
        lik_change2 = update_likelihoods(
            num_players, player_offsets,
            pd_forward_mu, pd_forward_sigma,
            pd_backward_mu, pd_backward_sigma,
            pd_likelihood_pi, pd_likelihood_tau,
            pd_game_offsets, pd_game_opp_pd, pd_game_score,
            prior_mu, prior_sigma, beta,
        )

        # Check convergence
        max_change = max(fwd_change, bwd_change, lik_change1, lik_change2)
        if max_change < convergence_threshold:
            return iteration + 1

    return max_iterations


@njit(cache=True, fastmath=True)
def extract_ratings(
    num_players: int,
    player_offsets: np.ndarray,
    pd_forward_mu: np.ndarray,
    pd_forward_sigma: np.ndarray,
    pd_backward_mu: np.ndarray,
    pd_backward_sigma: np.ndarray,
    pd_likelihood_pi: np.ndarray,
    pd_likelihood_tau: np.ndarray,
    ratings_out: np.ndarray,
    rd_out: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    display_scale: float,
    display_offset: float,
) -> None:
    """
    Extract the most recent rating for each player.

    Converts from internal scale to display scale (Elo-like).
    """
    for player_id in range(num_players):
        pd_start = player_offsets[player_id]
        pd_end = player_offsets[player_id + 1]

        if pd_end > pd_start:
            # Get most recent player-day
            last_pd = pd_end - 1

            # Compute posterior
            mu, sigma = get_posterior_from_messages(
                pd_forward_mu[last_pd], pd_forward_sigma[last_pd],
                pd_backward_mu[last_pd], pd_backward_sigma[last_pd],
                pd_likelihood_pi[last_pd], pd_likelihood_tau[last_pd],
                prior_mu, prior_sigma,
            )

            # Convert to display scale
            ratings_out[player_id] = mu * display_scale + display_offset
            rd_out[player_id] = sigma * display_scale
        else:
            # No games for this player
            ratings_out[player_id] = display_offset
            rd_out[player_id] = prior_sigma * display_scale


@njit(cache=True, fastmath=True, parallel=True)
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

    Uses the Gaussian CDF on the performance difference distribution.
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

        # P(player1 wins) = P(diff > 0) = Phi(diff_mu / diff_sigma)
        result[i] = norm_cdf(diff_mu / diff_sigma)

    return result


@njit(cache=True, fastmath=True)
def predict_single(
    rating1: float,
    rating2: float,
    rd1: float,
    rd2: float,
    beta: float,
    display_scale: float,
    display_offset: float,
) -> float:
    """Predict probability that player 1 beats player 2 (single matchup)."""
    mu1 = (rating1 - display_offset) / display_scale
    mu2 = (rating2 - display_offset) / display_scale
    sigma1 = rd1 / display_scale
    sigma2 = rd2 / display_scale

    diff_mu = mu1 - mu2
    diff_sigma = math.sqrt(sigma1*sigma1 + sigma2*sigma2 + 2.0*beta*beta)

    return norm_cdf(diff_mu / diff_sigma)


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
