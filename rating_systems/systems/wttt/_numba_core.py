"""
Numba-accelerated core functions for Weighted TrueSkill Through Time (WTTT).

WTTT extends TTT by allowing per-game performance noise: β_eff = β / √w.
When w > 1, a game is more informative (tighter update); when w < 1, less.
With w = 1 for all games, this reduces to standard TTT.

Only the game-processing and sweep functions need weighted variants.
All other functions (Gaussian utilities, structure building, extraction,
prediction) are reused directly from TTT's _numba_core.
"""

import math
import numpy as np
from numba import njit

# Reuse all shared functions from TTT
from ..trueskill_through_time._numba_core import (
    # Constants
    INF_SIGMA,
    # Gaussian utilities
    gaussian_mul,
    gaussian_div,
    gaussian_forget,
    # Game likelihood (takes scalar beta — called per-game with beta_eff[g])
    compute_game_likelihoods,
    # Structure building
    build_appearance_structure,
    # Rating extraction
    extract_final_ratings_sparse,
    # Prediction
    predict_proba_batch,
    predict_single,
)


@njit(cache=True)
def process_batch_games_weighted(
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
    game_beta_eff: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
) -> None:
    """
    Process all games in a batch with per-game beta_eff.

    Identical to process_batch_games() except uses game_beta_eff[g]
    instead of a scalar beta for each game.
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

    # Process each game with its own beta_eff
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
            p1_wins, game_beta_eff[g]
        )

        player_likelihood_mu[p1], player_likelihood_sigma[p1] = gaussian_mul(
            player_likelihood_mu[p1], player_likelihood_sigma[p1],
            lik1_mu, lik1_sigma
        )
        player_likelihood_mu[p2], player_likelihood_sigma[p2] = gaussian_mul(
            player_likelihood_mu[p2], player_likelihood_sigma[p2],
            lik2_mu, lik2_sigma
        )


@njit(cache=True)
def initial_forward_pass_weighted(
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
    # Sparse state arrays
    state_forward_mu: np.ndarray,
    state_forward_sigma: np.ndarray,
    state_backward_mu: np.ndarray,
    state_backward_sigma: np.ndarray,
    state_likelihood_mu: np.ndarray,
    state_likelihood_sigma: np.ndarray,
    # Pre-allocated temp arrays
    temp_fwd_mu: np.ndarray,
    temp_fwd_sigma: np.ndarray,
    temp_bwd_mu: np.ndarray,
    temp_bwd_sigma: np.ndarray,
    temp_lik_mu: np.ndarray,
    temp_lik_sigma: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    game_beta_eff: np.ndarray,
    gamma: float,
    start_batch: int,
) -> None:
    """Forward pass with per-game beta_eff (weighted variant)."""
    for b in range(start_batch, num_batches):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        for a in range(a_start, a_end):
            p = app_player[a]
            prev_a = app_prev[a]

            if prev_a < 0:
                state_forward_mu[a] = prior_mu
                state_forward_sigma[a] = prior_sigma
            else:
                prev_time = batch_times[app_batch[prev_a]]
                fwd_lik_mu, fwd_lik_sigma = gaussian_mul(
                    state_forward_mu[prev_a], state_forward_sigma[prev_a],
                    state_likelihood_mu[prev_a], state_likelihood_sigma[prev_a]
                )
                elapsed = batch_time - prev_time
                fwd_mu, fwd_sigma = gaussian_forget(fwd_lik_mu, fwd_lik_sigma, gamma, elapsed)
                state_forward_mu[a] = fwd_mu
                state_forward_sigma[a] = fwd_sigma

            state_backward_mu[a] = 0.0
            state_backward_sigma[a] = INF_SIGMA

            temp_fwd_mu[p] = state_forward_mu[a]
            temp_fwd_sigma[p] = state_forward_sigma[a]
            temp_bwd_mu[p] = state_backward_mu[a]
            temp_bwd_sigma[p] = state_backward_sigma[a]

        process_batch_games_weighted(
            b, batch_offsets, game_p1, game_p2, game_scores,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            game_beta_eff, prior_mu, prior_sigma
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            state_likelihood_mu[a] = temp_lik_mu[p]
            state_likelihood_sigma[a] = temp_lik_sigma[p]


@njit(cache=True)
def backward_sweep_weighted(
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
    game_beta_eff: np.ndarray,
    gamma: float,
) -> float:
    """Backward sweep with per-game beta_eff. Returns max change."""
    max_change = 0.0

    for b in range(num_batches - 1, -1, -1):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        for a in range(a_start, a_end):
            p = app_player[a]
            next_a = app_next[a]
            old_mu = state_backward_mu[a]

            if next_a < 0:
                state_backward_mu[a] = 0.0
                state_backward_sigma[a] = INF_SIGMA
            else:
                lik_bwd_mu, lik_bwd_sigma = gaussian_mul(
                    state_likelihood_mu[next_a], state_likelihood_sigma[next_a],
                    state_backward_mu[next_a], state_backward_sigma[next_a]
                )
                next_time = batch_times[app_batch[next_a]]
                elapsed = next_time - batch_time
                bwd_mu, bwd_sigma = gaussian_forget(lik_bwd_mu, lik_bwd_sigma, gamma, elapsed)
                state_backward_mu[a] = bwd_mu
                state_backward_sigma[a] = bwd_sigma

            change = abs(state_backward_mu[a] - old_mu)
            if change > max_change:
                max_change = change

            temp_fwd_mu[p] = state_forward_mu[a]
            temp_fwd_sigma[p] = state_forward_sigma[a]
            temp_bwd_mu[p] = state_backward_mu[a]
            temp_bwd_sigma[p] = state_backward_sigma[a]

        process_batch_games_weighted(
            b, batch_offsets, game_p1, game_p2, game_scores,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            game_beta_eff, prior_mu, prior_sigma
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            state_likelihood_mu[a] = temp_lik_mu[p]
            state_likelihood_sigma[a] = temp_lik_sigma[p]

    return max_change


@njit(cache=True)
def forward_sweep_weighted(
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
    game_beta_eff: np.ndarray,
    gamma: float,
) -> float:
    """Forward sweep with per-game beta_eff. Returns max change."""
    max_change = 0.0

    for b in range(num_batches):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        for a in range(a_start, a_end):
            p = app_player[a]
            prev_a = app_prev[a]
            old_mu = state_forward_mu[a]

            if prev_a < 0:
                state_forward_mu[a] = prior_mu
                state_forward_sigma[a] = prior_sigma
            else:
                fwd_lik_mu, fwd_lik_sigma = gaussian_mul(
                    state_forward_mu[prev_a], state_forward_sigma[prev_a],
                    state_likelihood_mu[prev_a], state_likelihood_sigma[prev_a]
                )
                prev_time = batch_times[app_batch[prev_a]]
                elapsed = batch_time - prev_time
                fwd_mu, fwd_sigma = gaussian_forget(fwd_lik_mu, fwd_lik_sigma, gamma, elapsed)
                state_forward_mu[a] = fwd_mu
                state_forward_sigma[a] = fwd_sigma

            change = abs(state_forward_mu[a] - old_mu)
            if change > max_change:
                max_change = change

            temp_fwd_mu[p] = state_forward_mu[a]
            temp_fwd_sigma[p] = state_forward_sigma[a]
            temp_bwd_mu[p] = state_backward_mu[a]
            temp_bwd_sigma[p] = state_backward_sigma[a]

        process_batch_games_weighted(
            b, batch_offsets, game_p1, game_p2, game_scores,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            game_beta_eff, prior_mu, prior_sigma
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            state_likelihood_mu[a] = temp_lik_mu[p]
            state_likelihood_sigma[a] = temp_lik_sigma[p]

    return max_change


@njit(cache=True)
def run_convergence_weighted(
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
    game_beta_eff: np.ndarray,
    gamma: float,
    max_iterations: int,
    epsilon: float,
) -> int:
    """Run forward-backward iterations with per-game beta_eff until convergence."""
    for iteration in range(max_iterations):
        bwd_change = backward_sweep_weighted(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            app_offsets, app_player, app_next, app_batch,
            state_forward_mu, state_forward_sigma,
            state_backward_mu, state_backward_sigma,
            state_likelihood_mu, state_likelihood_sigma,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            prior_mu, prior_sigma, game_beta_eff, gamma
        )

        fwd_change = forward_sweep_weighted(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            app_offsets, app_player, app_prev, app_batch,
            state_forward_mu, state_forward_sigma,
            state_backward_mu, state_backward_sigma,
            state_likelihood_mu, state_likelihood_sigma,
            temp_fwd_mu, temp_fwd_sigma,
            temp_bwd_mu, temp_bwd_sigma,
            temp_lik_mu, temp_lik_sigma,
            prior_mu, prior_sigma, game_beta_eff, gamma
        )

        max_change = max(bwd_change, fwd_change)
        if max_change < epsilon:
            return iteration + 1

    return max_iterations
