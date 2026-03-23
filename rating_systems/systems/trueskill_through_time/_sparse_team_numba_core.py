"""Sparse-storage team-based TrueSkill Through Time Numba kernels.

Provides appearance-based sparse storage for SurfaceTTT, reducing memory
from O(num_batches × num_expanded_players) to O(num_appearances).

Reuses process_batch_games and compute_team_likelihoods from
_team_numba_core.py unchanged — game processing uses temp arrays indexed
by player ID, so it doesn't depend on the storage format.
"""

import math
import numpy as np
from numba import njit

from ._team_numba_core import (
    INF_SIGMA, MIN_SIGMA,
    gaussian_multiply, gaussian_drift,
    process_batch_games,
)


# =============================================================================
# Sparse Appearance Structure (team-aware: 4 players per game)
# =============================================================================

@njit(cache=True)
def build_team_appearance_structure(
    num_batches, batch_offsets,
    game_t1_base, game_t1_surf, game_t2_base, game_t2_surf,
    num_players,
):
    """Build sparse appearance structures for team games (4 players per game).

    Returns:
        app_offsets: int64[num_batches+1] — CSR boundaries per batch
        app_player: int64[num_appearances] — player ID per appearance
        app_prev: int64[num_appearances] — prev appearance for same player (-1 if first)
        app_next: int64[num_appearances] — next appearance for same player (-1 if last)
        app_batch: int64[num_appearances] — batch index per appearance
        player_last_app: int64[num_players] — last appearance index per player
    """
    player_last_batch = np.full(num_players, -1, dtype=np.int64)
    batch_players = np.empty(num_players, dtype=np.int64)

    # First pass: count unique appearances per batch
    batch_app_counts = np.zeros(num_batches, dtype=np.int64)
    for b in range(num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        for g in range(game_start, game_end):
            for p in (game_t1_base[g], game_t1_surf[g],
                      game_t2_base[g], game_t2_surf[g]):
                if player_last_batch[p] != b:
                    player_last_batch[p] = b
                    batch_app_counts[b] += 1

    # Build CSR offsets
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
            for p in (game_t1_base[g], game_t1_surf[g],
                      game_t2_base[g], game_t2_surf[g]):
                if player_last_batch[p] != b:
                    player_last_batch[p] = b
                    batch_players[n_in_batch] = p
                    n_in_batch += 1

        batch_players[:n_in_batch].sort()
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
# Sparse Forward-Backward Algorithm
# =============================================================================

@njit(cache=True)
def sparse_initial_forward_pass(
    num_batches, batch_offsets, batch_times,
    game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
    num_players, num_base_players,
    # Sparse state [num_appearances]
    state_fwd_mu, state_fwd_sigma,
    state_bwd_mu, state_bwd_sigma,
    state_lik_mu, state_lik_sigma,
    # Sparse indexing
    app_offsets, app_player, app_prev, app_batch,
    # Temp arrays [num_players] — reused per batch
    tmp_fwd_mu, tmp_fwd_sigma, tmp_bwd_mu, tmp_bwd_sigma,
    tmp_lik_mu, tmp_lik_sigma,
    # Parameters
    prior_mu, prior_sigma, surface_prior_sigma,
    w_base, w_surf, beta, gamma, surface_gamma,
    start_batch,
):
    """Sparse initial forward pass for team games."""
    for b in range(start_batch, num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        # Load forward messages into temp arrays from sparse state
        for a in range(a_start, a_end):
            p = app_player[a]
            is_surface = p >= num_base_players
            p_sigma = surface_prior_sigma if is_surface else prior_sigma
            p_gamma = surface_gamma if is_surface else gamma
            prev_a = app_prev[a]

            if prev_a < 0:
                # First appearance ever
                tmp_fwd_mu[p] = prior_mu
                tmp_fwd_sigma[p] = p_sigma
            else:
                # Forward message = (forward × likelihood at prev) drifted
                prev_fwd_mu, prev_fwd_sigma = gaussian_multiply(
                    state_fwd_mu[prev_a], state_fwd_sigma[prev_a],
                    state_lik_mu[prev_a], state_lik_sigma[prev_a],
                )
                elapsed = batch_time - batch_times[app_batch[prev_a]]
                tmp_fwd_mu[p], tmp_fwd_sigma[p] = gaussian_drift(
                    prev_fwd_mu, prev_fwd_sigma, p_gamma, elapsed,
                )

            # Init backward to uninformative
            tmp_bwd_mu[p] = 0.0
            tmp_bwd_sigma[p] = INF_SIGMA

        # Process games (writes into tmp_lik arrays)
        process_batch_games(
            game_start, game_end,
            game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
            tmp_fwd_mu, tmp_fwd_sigma, tmp_bwd_mu, tmp_bwd_sigma,
            tmp_lik_mu, tmp_lik_sigma,
            w_base, w_surf, beta,
        )

        # Write results back to sparse state
        for a in range(a_start, a_end):
            p = app_player[a]
            state_fwd_mu[a] = tmp_fwd_mu[p]
            state_fwd_sigma[a] = tmp_fwd_sigma[p]
            state_bwd_mu[a] = tmp_bwd_mu[p]
            state_bwd_sigma[a] = tmp_bwd_sigma[p]
            state_lik_mu[a] = tmp_lik_mu[p]
            state_lik_sigma[a] = tmp_lik_sigma[p]


@njit(cache=True)
def sparse_backward_sweep(
    num_batches, batch_offsets, batch_times,
    game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
    num_players, num_base_players,
    state_fwd_mu, state_fwd_sigma,
    state_bwd_mu, state_bwd_sigma,
    state_lik_mu, state_lik_sigma,
    app_offsets, app_player, app_prev, app_next, app_batch,
    tmp_fwd_mu, tmp_fwd_sigma, tmp_bwd_mu, tmp_bwd_sigma,
    tmp_lik_mu, tmp_lik_sigma,
    w_base, w_surf, beta, gamma, surface_gamma,
):
    """Sparse backward sweep. Returns max change for convergence."""
    max_change = np.float32(0.0)

    for b in range(num_batches - 1, -1, -1):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]

        # Compute backward messages from future appearances
        for a in range(a_start, a_end):
            p = app_player[a]
            is_surface = p >= num_base_players
            p_gamma = surface_gamma if is_surface else gamma
            next_a = app_next[a]

            if next_a < 0:
                # No future appearance
                new_bwd_mu = np.float32(0.0)
                new_bwd_sigma = INF_SIGMA
            else:
                # backward_out = (likelihood × backward at next) drifted back
                lik_bwd_mu, lik_bwd_sigma = gaussian_multiply(
                    state_lik_mu[next_a], state_lik_sigma[next_a],
                    state_bwd_mu[next_a], state_bwd_sigma[next_a],
                )
                elapsed = batch_times[app_batch[next_a]] - batch_times[b]
                new_bwd_mu, new_bwd_sigma = gaussian_drift(
                    lik_bwd_mu, lik_bwd_sigma, p_gamma, elapsed,
                )

            old_mu = state_bwd_mu[a]
            state_bwd_mu[a] = new_bwd_mu
            state_bwd_sigma[a] = new_bwd_sigma
            change = abs(new_bwd_mu - old_mu)
            if change > max_change:
                max_change = change

            tmp_fwd_mu[p] = state_fwd_mu[a]
            tmp_fwd_sigma[p] = state_fwd_sigma[a]
            tmp_bwd_mu[p] = new_bwd_mu
            tmp_bwd_sigma[p] = new_bwd_sigma

        # Recompute likelihoods with updated backward
        process_batch_games(
            game_start, game_end,
            game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
            tmp_fwd_mu, tmp_fwd_sigma, tmp_bwd_mu, tmp_bwd_sigma,
            tmp_lik_mu, tmp_lik_sigma,
            w_base, w_surf, beta,
        )

        # Write likelihoods back to sparse state
        for a in range(a_start, a_end):
            p = app_player[a]
            state_lik_mu[a] = tmp_lik_mu[p]
            state_lik_sigma[a] = tmp_lik_sigma[p]

    return max_change


@njit(cache=True)
def sparse_forward_sweep(
    num_batches, batch_offsets, batch_times,
    game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
    num_players, num_base_players,
    state_fwd_mu, state_fwd_sigma,
    state_bwd_mu, state_bwd_sigma,
    state_lik_mu, state_lik_sigma,
    app_offsets, app_player, app_prev, app_batch,
    tmp_fwd_mu, tmp_fwd_sigma, tmp_bwd_mu, tmp_bwd_sigma,
    tmp_lik_mu, tmp_lik_sigma,
    prior_mu, prior_sigma, surface_prior_sigma,
    w_base, w_surf, beta, gamma, surface_gamma,
):
    """Sparse forward sweep. Returns max change for convergence."""
    max_change = np.float32(0.0)

    for b in range(num_batches):
        game_start = batch_offsets[b]
        game_end = batch_offsets[b + 1]
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        # Compute forward messages from past appearances
        for a in range(a_start, a_end):
            p = app_player[a]
            is_surface = p >= num_base_players
            p_sigma = surface_prior_sigma if is_surface else prior_sigma
            p_gamma = surface_gamma if is_surface else gamma
            prev_a = app_prev[a]

            if prev_a < 0:
                new_fwd_mu = prior_mu
                new_fwd_sigma = p_sigma
            else:
                prev_fwd_mu, prev_fwd_sigma = gaussian_multiply(
                    state_fwd_mu[prev_a], state_fwd_sigma[prev_a],
                    state_lik_mu[prev_a], state_lik_sigma[prev_a],
                )
                elapsed = batch_time - batch_times[app_batch[prev_a]]
                new_fwd_mu, new_fwd_sigma = gaussian_drift(
                    prev_fwd_mu, prev_fwd_sigma, p_gamma, elapsed,
                )

            old_mu = state_fwd_mu[a]
            state_fwd_mu[a] = new_fwd_mu
            state_fwd_sigma[a] = new_fwd_sigma
            change = abs(new_fwd_mu - old_mu)
            if change > max_change:
                max_change = change

            tmp_fwd_mu[p] = new_fwd_mu
            tmp_fwd_sigma[p] = new_fwd_sigma
            tmp_bwd_mu[p] = state_bwd_mu[a]
            tmp_bwd_sigma[p] = state_bwd_sigma[a]

        # Recompute likelihoods with updated forward
        process_batch_games(
            game_start, game_end,
            game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
            tmp_fwd_mu, tmp_fwd_sigma, tmp_bwd_mu, tmp_bwd_sigma,
            tmp_lik_mu, tmp_lik_sigma,
            w_base, w_surf, beta,
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            state_lik_mu[a] = tmp_lik_mu[p]
            state_lik_sigma[a] = tmp_lik_sigma[p]

    return max_change


@njit(cache=True)
def sparse_run_convergence(
    num_batches, batch_offsets, batch_times,
    game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
    num_players, num_base_players,
    state_fwd_mu, state_fwd_sigma,
    state_bwd_mu, state_bwd_sigma,
    state_lik_mu, state_lik_sigma,
    app_offsets, app_player, app_prev, app_next, app_batch,
    tmp_fwd_mu, tmp_fwd_sigma, tmp_bwd_mu, tmp_bwd_sigma,
    tmp_lik_mu, tmp_lik_sigma,
    prior_mu, prior_sigma, surface_prior_sigma,
    w_base, w_surf, beta, gamma, surface_gamma,
    max_iterations, epsilon,
):
    """Run forward-backward iterations until convergence."""
    for iteration in range(max_iterations):
        bwd_change = sparse_backward_sweep(
            num_batches, batch_offsets, batch_times,
            game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
            num_players, num_base_players,
            state_fwd_mu, state_fwd_sigma,
            state_bwd_mu, state_bwd_sigma,
            state_lik_mu, state_lik_sigma,
            app_offsets, app_player, app_prev, app_next, app_batch,
            tmp_fwd_mu, tmp_fwd_sigma, tmp_bwd_mu, tmp_bwd_sigma,
            tmp_lik_mu, tmp_lik_sigma,
            w_base, w_surf, beta, gamma, surface_gamma,
        )

        fwd_change = sparse_forward_sweep(
            num_batches, batch_offsets, batch_times,
            game_t1_base, game_t1_surf, game_t2_base, game_t2_surf, game_scores,
            num_players, num_base_players,
            state_fwd_mu, state_fwd_sigma,
            state_bwd_mu, state_bwd_sigma,
            state_lik_mu, state_lik_sigma,
            app_offsets, app_player, app_prev, app_batch,
            tmp_fwd_mu, tmp_fwd_sigma, tmp_bwd_mu, tmp_bwd_sigma,
            tmp_lik_mu, tmp_lik_sigma,
            prior_mu, prior_sigma, surface_prior_sigma,
            w_base, w_surf, beta, gamma, surface_gamma,
        )

        if max(bwd_change, fwd_change) < epsilon:
            return iteration + 1

    return max_iterations


@njit(cache=True)
def sparse_extract_final_ratings(
    num_players, num_base_players,
    state_fwd_mu, state_fwd_sigma,
    state_bwd_mu, state_bwd_sigma,
    state_lik_mu, state_lik_sigma,
    player_last_app,
    ratings_out, rd_out,
    prior_mu, prior_sigma, surface_prior_sigma,
    display_scale, display_offset,
):
    """Extract final ratings from sparse state."""
    for p in range(num_players):
        is_surface = p >= num_base_players
        p_sigma = surface_prior_sigma if is_surface else prior_sigma
        a = player_last_app[p]

        if a < 0:
            ratings_out[p] = display_offset
            rd_out[p] = p_sigma * display_scale
        else:
            fwd_bwd_mu, fwd_bwd_sigma = gaussian_multiply(
                state_fwd_mu[a], state_fwd_sigma[a],
                state_bwd_mu[a], state_bwd_sigma[a],
            )
            post_mu, post_sigma = gaussian_multiply(
                fwd_bwd_mu, fwd_bwd_sigma,
                state_lik_mu[a], state_lik_sigma[a],
            )
            ratings_out[p] = post_mu * display_scale + display_offset
            rd_out[p] = post_sigma * display_scale
