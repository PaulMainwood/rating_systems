"""
Numba-accelerated core functions for Weighted Matérn 3/2 TTT.

Extends MaternTTT with per-game weights (β_eff = β / √w) and handicaps.
Only the game-processing functions need weighted variants; all 2D Gaussian
operations, transitions, and sweeps are reused from matern_ttt._numba_core.
"""

import math
import numpy as np
from numba import njit

# Reuse everything from matern_ttt
from ..matern_ttt._numba_core import (
    INF_COV,
    gaussian_mul_2d,
    matern32_transition,
    matern32_backward_transition,
    mul_2d_with_scalar_lik,
    extract_scalar_marginal,
    extract_final_ratings_matern,
    extract_player_posteriors_matern,
    predict_proba_batch_at_day_matern,
    predict_single_at_day_matern,
)

# Scalar Gaussian utilities and game likelihoods from TTT
from ..trueskill_through_time._numba_core import (
    INF_SIGMA,
    gaussian_mul,
    compute_game_likelihoods,
    compute_game_likelihoods_h,
    build_appearance_structure,
    predict_proba_batch,
    predict_single,
    predict_proba_batch_h,
    predict_single_h,
    extract_player_last_day_ttt,
)


# =============================================================================
# Weighted batch game processing (per-game beta_eff, no handicap)
# =============================================================================

@njit(cache=True)
def process_batch_games_matern_weighted(
    batch_idx: int,
    batch_offsets: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    # 2D forward state (temp, by player)
    player_fwd_mu0: np.ndarray, player_fwd_mu1: np.ndarray,
    player_fwd_c00: np.ndarray, player_fwd_c01: np.ndarray, player_fwd_c11: np.ndarray,
    # 2D backward state (temp, by player)
    player_bwd_mu0: np.ndarray, player_bwd_mu1: np.ndarray,
    player_bwd_c00: np.ndarray, player_bwd_c01: np.ndarray, player_bwd_c11: np.ndarray,
    # Scalar likelihood (temp, by player)
    player_lik_mu: np.ndarray,
    player_lik_sigma: np.ndarray,
    # Per-game
    game_beta_eff: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
) -> None:
    """Process games with per-game beta_eff (weighted variant)."""
    game_start = batch_offsets[batch_idx]
    game_end = batch_offsets[batch_idx + 1]

    for g in range(game_start, game_end):
        p1, p2 = game_p1[g], game_p2[g]
        player_lik_mu[p1] = 0.0
        player_lik_sigma[p1] = INF_SIGMA
        player_lik_mu[p2] = 0.0
        player_lik_sigma[p2] = INF_SIGMA

    for g in range(game_start, game_end):
        p1, p2 = game_p1[g], game_p2[g]
        p1_wins = game_scores[g] > 0.5

        post_mu0, post_mu1, post_c00, post_c01, post_c11 = gaussian_mul_2d(
            player_fwd_mu0[p1], player_fwd_mu1[p1],
            player_fwd_c00[p1], player_fwd_c01[p1], player_fwd_c11[p1],
            player_bwd_mu0[p1], player_bwd_mu1[p1],
            player_bwd_c00[p1], player_bwd_c01[p1], player_bwd_c11[p1],
        )
        p1_mu, p1_sigma = extract_scalar_marginal(
            post_mu0, post_mu1, post_c00, post_c01, post_c11
        )

        post_mu0, post_mu1, post_c00, post_c01, post_c11 = gaussian_mul_2d(
            player_fwd_mu0[p2], player_fwd_mu1[p2],
            player_fwd_c00[p2], player_fwd_c01[p2], player_fwd_c11[p2],
            player_bwd_mu0[p2], player_bwd_mu1[p2],
            player_bwd_c00[p2], player_bwd_c01[p2], player_bwd_c11[p2],
        )
        p2_mu, p2_sigma = extract_scalar_marginal(
            post_mu0, post_mu1, post_c00, post_c01, post_c11
        )

        lik1_mu, lik1_sigma, lik2_mu, lik2_sigma = compute_game_likelihoods(
            p1_mu, p1_sigma, p2_mu, p2_sigma, p1_wins, game_beta_eff[g]
        )

        player_lik_mu[p1], player_lik_sigma[p1] = gaussian_mul(
            player_lik_mu[p1], player_lik_sigma[p1], lik1_mu, lik1_sigma
        )
        player_lik_mu[p2], player_lik_sigma[p2] = gaussian_mul(
            player_lik_mu[p2], player_lik_sigma[p2], lik2_mu, lik2_sigma
        )


# =============================================================================
# Weighted + handicap batch game processing
# =============================================================================

@njit(cache=True)
def process_batch_games_matern_weighted_h(
    batch_idx: int,
    batch_offsets: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    player_fwd_mu0: np.ndarray, player_fwd_mu1: np.ndarray,
    player_fwd_c00: np.ndarray, player_fwd_c01: np.ndarray, player_fwd_c11: np.ndarray,
    player_bwd_mu0: np.ndarray, player_bwd_mu1: np.ndarray,
    player_bwd_c00: np.ndarray, player_bwd_c01: np.ndarray, player_bwd_c11: np.ndarray,
    player_lik_mu: np.ndarray,
    player_lik_sigma: np.ndarray,
    game_beta_eff: np.ndarray,
    game_handicaps: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
) -> None:
    """Process games with per-game beta_eff and handicaps."""
    game_start = batch_offsets[batch_idx]
    game_end = batch_offsets[batch_idx + 1]

    for g in range(game_start, game_end):
        p1, p2 = game_p1[g], game_p2[g]
        player_lik_mu[p1] = 0.0
        player_lik_sigma[p1] = INF_SIGMA
        player_lik_mu[p2] = 0.0
        player_lik_sigma[p2] = INF_SIGMA

    for g in range(game_start, game_end):
        p1, p2 = game_p1[g], game_p2[g]
        p1_wins = game_scores[g] > 0.5

        post_mu0, post_mu1, post_c00, post_c01, post_c11 = gaussian_mul_2d(
            player_fwd_mu0[p1], player_fwd_mu1[p1],
            player_fwd_c00[p1], player_fwd_c01[p1], player_fwd_c11[p1],
            player_bwd_mu0[p1], player_bwd_mu1[p1],
            player_bwd_c00[p1], player_bwd_c01[p1], player_bwd_c11[p1],
        )
        p1_mu, p1_sigma = extract_scalar_marginal(
            post_mu0, post_mu1, post_c00, post_c01, post_c11
        )

        post_mu0, post_mu1, post_c00, post_c01, post_c11 = gaussian_mul_2d(
            player_fwd_mu0[p2], player_fwd_mu1[p2],
            player_fwd_c00[p2], player_fwd_c01[p2], player_fwd_c11[p2],
            player_bwd_mu0[p2], player_bwd_mu1[p2],
            player_bwd_c00[p2], player_bwd_c01[p2], player_bwd_c11[p2],
        )
        p2_mu, p2_sigma = extract_scalar_marginal(
            post_mu0, post_mu1, post_c00, post_c01, post_c11
        )

        lik1_mu, lik1_sigma, lik2_mu, lik2_sigma = compute_game_likelihoods_h(
            p1_mu, p1_sigma, p2_mu, p2_sigma,
            p1_wins, game_beta_eff[g], game_handicaps[g]
        )

        player_lik_mu[p1], player_lik_sigma[p1] = gaussian_mul(
            player_lik_mu[p1], player_lik_sigma[p1], lik1_mu, lik1_sigma
        )
        player_lik_mu[p2], player_lik_sigma[p2] = gaussian_mul(
            player_lik_mu[p2], player_lik_sigma[p2], lik2_mu, lik2_sigma
        )


# =============================================================================
# Forward/backward sweeps with 2D Matérn state + weighted games
# =============================================================================

@njit(cache=True)
def initial_forward_pass_matern_weighted(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_prev: np.ndarray,
    app_batch: np.ndarray,
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
    game_beta_eff: np.ndarray,
    lambda_: float,
    sigma_sq: float,
    start_batch: int,
) -> None:
    """Forward pass with 2D Matérn state and per-game beta_eff."""
    for b in range(start_batch, num_batches):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        for a in range(a_start, a_end):
            p = app_player[a]
            prev_a = app_prev[a]

            if prev_a < 0:
                fwd_mu0[a] = prior_mu
                fwd_mu1[a] = 0.0
                fwd_c00[a] = prior_c00
                fwd_c01[a] = prior_c01
                fwd_c11[a] = prior_c11
            else:
                prev_time = batch_times[app_batch[prev_a]]
                fl_mu0, fl_mu1, fl_c00, fl_c01, fl_c11 = \
                    mul_2d_with_scalar_lik(
                        fwd_mu0[prev_a], fwd_mu1[prev_a],
                        fwd_c00[prev_a], fwd_c01[prev_a], fwd_c11[prev_a],
                        lik_mu[prev_a], lik_sigma[prev_a],
                    )
                elapsed = batch_time - prev_time
                nm0, nm1, nc00, nc01, nc11 = matern32_transition(
                    fl_mu0, fl_mu1, fl_c00, fl_c01, fl_c11,
                    elapsed, lambda_, sigma_sq,
                )
                fwd_mu0[a] = nm0
                fwd_mu1[a] = nm1
                fwd_c00[a] = nc00
                fwd_c01[a] = nc01
                fwd_c11[a] = nc11

            bwd_mu0[a] = 0.0
            bwd_mu1[a] = 0.0
            bwd_c00[a] = INF_COV
            bwd_c01[a] = 0.0
            bwd_c11[a] = INF_COV

            t_fwd_mu0[p] = fwd_mu0[a]
            t_fwd_mu1[p] = fwd_mu1[a]
            t_fwd_c00[p] = fwd_c00[a]
            t_fwd_c01[p] = fwd_c01[a]
            t_fwd_c11[p] = fwd_c11[a]
            t_bwd_mu0[p] = bwd_mu0[a]
            t_bwd_mu1[p] = bwd_mu1[a]
            t_bwd_c00[p] = bwd_c00[a]
            t_bwd_c01[p] = bwd_c01[a]
            t_bwd_c11[p] = bwd_c11[a]

        process_batch_games_matern_weighted(
            b, batch_offsets, game_p1, game_p2, game_scores,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            game_beta_eff, prior_mu, prior_sigma,
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            lik_mu[a] = t_lik_mu[p]
            lik_sigma[a] = t_lik_sigma[p]


@njit(cache=True)
def backward_sweep_matern_weighted(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_next: np.ndarray,
    app_batch: np.ndarray,
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    game_beta_eff: np.ndarray,
    lambda_: float,
    sigma_sq: float,
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
            old_mu0 = bwd_mu0[a]

            if next_a < 0:
                bwd_mu0[a] = 0.0
                bwd_mu1[a] = 0.0
                bwd_c00[a] = INF_COV
                bwd_c01[a] = 0.0
                bwd_c11[a] = INF_COV
            else:
                bl_mu0, bl_mu1, bl_c00, bl_c01, bl_c11 = \
                    mul_2d_with_scalar_lik(
                        bwd_mu0[next_a], bwd_mu1[next_a],
                        bwd_c00[next_a], bwd_c01[next_a], bwd_c11[next_a],
                        lik_mu[next_a], lik_sigma[next_a],
                    )
                next_time = batch_times[app_batch[next_a]]
                elapsed = next_time - batch_time
                nm0, nm1, nc00, nc01, nc11 = matern32_backward_transition(
                    bl_mu0, bl_mu1, bl_c00, bl_c01, bl_c11,
                    elapsed, lambda_, sigma_sq,
                )
                bwd_mu0[a] = nm0
                bwd_mu1[a] = nm1
                bwd_c00[a] = nc00
                bwd_c01[a] = nc01
                bwd_c11[a] = nc11

            change = abs(bwd_mu0[a] - old_mu0)
            if change > max_change:
                max_change = change

            t_fwd_mu0[p] = fwd_mu0[a]
            t_fwd_mu1[p] = fwd_mu1[a]
            t_fwd_c00[p] = fwd_c00[a]
            t_fwd_c01[p] = fwd_c01[a]
            t_fwd_c11[p] = fwd_c11[a]
            t_bwd_mu0[p] = bwd_mu0[a]
            t_bwd_mu1[p] = bwd_mu1[a]
            t_bwd_c00[p] = bwd_c00[a]
            t_bwd_c01[p] = bwd_c01[a]
            t_bwd_c11[p] = bwd_c11[a]

        process_batch_games_matern_weighted(
            b, batch_offsets, game_p1, game_p2, game_scores,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            game_beta_eff, prior_mu, prior_sigma,
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            lik_mu[a] = t_lik_mu[p]
            lik_sigma[a] = t_lik_sigma[p]

    return max_change


@njit(cache=True)
def forward_sweep_matern_weighted(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_prev: np.ndarray,
    app_batch: np.ndarray,
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
    game_beta_eff: np.ndarray,
    lambda_: float,
    sigma_sq: float,
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
            old_mu0 = fwd_mu0[a]

            if prev_a < 0:
                fwd_mu0[a] = prior_mu
                fwd_mu1[a] = 0.0
                fwd_c00[a] = prior_c00
                fwd_c01[a] = prior_c01
                fwd_c11[a] = prior_c11
            else:
                prev_time = batch_times[app_batch[prev_a]]
                fl_mu0, fl_mu1, fl_c00, fl_c01, fl_c11 = \
                    mul_2d_with_scalar_lik(
                        fwd_mu0[prev_a], fwd_mu1[prev_a],
                        fwd_c00[prev_a], fwd_c01[prev_a], fwd_c11[prev_a],
                        lik_mu[prev_a], lik_sigma[prev_a],
                    )
                elapsed = batch_time - prev_time
                nm0, nm1, nc00, nc01, nc11 = matern32_transition(
                    fl_mu0, fl_mu1, fl_c00, fl_c01, fl_c11,
                    elapsed, lambda_, sigma_sq,
                )
                fwd_mu0[a] = nm0
                fwd_mu1[a] = nm1
                fwd_c00[a] = nc00
                fwd_c01[a] = nc01
                fwd_c11[a] = nc11

            change = abs(fwd_mu0[a] - old_mu0)
            if change > max_change:
                max_change = change

            t_fwd_mu0[p] = fwd_mu0[a]
            t_fwd_mu1[p] = fwd_mu1[a]
            t_fwd_c00[p] = fwd_c00[a]
            t_fwd_c01[p] = fwd_c01[a]
            t_fwd_c11[p] = fwd_c11[a]
            t_bwd_mu0[p] = bwd_mu0[a]
            t_bwd_mu1[p] = bwd_mu1[a]
            t_bwd_c00[p] = bwd_c00[a]
            t_bwd_c01[p] = bwd_c01[a]
            t_bwd_c11[p] = bwd_c11[a]

        process_batch_games_matern_weighted(
            b, batch_offsets, game_p1, game_p2, game_scores,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            game_beta_eff, prior_mu, prior_sigma,
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            lik_mu[a] = t_lik_mu[p]
            lik_sigma[a] = t_lik_sigma[p]

    return max_change


@njit(cache=True)
def run_convergence_matern_weighted(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_prev: np.ndarray,
    app_next: np.ndarray,
    app_batch: np.ndarray,
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
    game_beta_eff: np.ndarray,
    lambda_: float,
    sigma_sq: float,
    max_iterations: int,
    epsilon: float,
) -> int:
    """Run convergence with per-game beta_eff."""
    for iteration in range(max_iterations):
        bwd_change = backward_sweep_matern_weighted(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            app_offsets, app_player, app_next, app_batch,
            fwd_mu0, fwd_mu1, fwd_c00, fwd_c01, fwd_c11,
            bwd_mu0, bwd_mu1, bwd_c00, bwd_c01, bwd_c11,
            lik_mu, lik_sigma,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            prior_mu, prior_sigma, game_beta_eff, lambda_, sigma_sq,
        )

        fwd_change = forward_sweep_matern_weighted(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            app_offsets, app_player, app_prev, app_batch,
            fwd_mu0, fwd_mu1, fwd_c00, fwd_c01, fwd_c11,
            bwd_mu0, bwd_mu1, bwd_c00, bwd_c01, bwd_c11,
            lik_mu, lik_sigma,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            prior_mu, prior_sigma,
            prior_c00, prior_c01, prior_c11,
            game_beta_eff, lambda_, sigma_sq,
        )

        max_change = max(bwd_change, fwd_change)
        if max_change < epsilon:
            return iteration + 1

    return max_iterations


# =============================================================================
# Handicap variants (weighted + handicap)
# =============================================================================

@njit(cache=True)
def initial_forward_pass_matern_weighted_h(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_prev: np.ndarray,
    app_batch: np.ndarray,
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
    game_beta_eff: np.ndarray,
    game_handicaps: np.ndarray,
    lambda_: float,
    sigma_sq: float,
    start_batch: int,
) -> None:
    """Forward pass with per-game beta_eff and handicaps."""
    for b in range(start_batch, num_batches):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        for a in range(a_start, a_end):
            p = app_player[a]
            prev_a = app_prev[a]

            if prev_a < 0:
                fwd_mu0[a] = prior_mu
                fwd_mu1[a] = 0.0
                fwd_c00[a] = prior_c00
                fwd_c01[a] = prior_c01
                fwd_c11[a] = prior_c11
            else:
                prev_time = batch_times[app_batch[prev_a]]
                fl_mu0, fl_mu1, fl_c00, fl_c01, fl_c11 = \
                    mul_2d_with_scalar_lik(
                        fwd_mu0[prev_a], fwd_mu1[prev_a],
                        fwd_c00[prev_a], fwd_c01[prev_a], fwd_c11[prev_a],
                        lik_mu[prev_a], lik_sigma[prev_a],
                    )
                elapsed = batch_time - prev_time
                nm0, nm1, nc00, nc01, nc11 = matern32_transition(
                    fl_mu0, fl_mu1, fl_c00, fl_c01, fl_c11,
                    elapsed, lambda_, sigma_sq,
                )
                fwd_mu0[a] = nm0
                fwd_mu1[a] = nm1
                fwd_c00[a] = nc00
                fwd_c01[a] = nc01
                fwd_c11[a] = nc11

            bwd_mu0[a] = 0.0
            bwd_mu1[a] = 0.0
            bwd_c00[a] = INF_COV
            bwd_c01[a] = 0.0
            bwd_c11[a] = INF_COV

            t_fwd_mu0[p] = fwd_mu0[a]
            t_fwd_mu1[p] = fwd_mu1[a]
            t_fwd_c00[p] = fwd_c00[a]
            t_fwd_c01[p] = fwd_c01[a]
            t_fwd_c11[p] = fwd_c11[a]
            t_bwd_mu0[p] = bwd_mu0[a]
            t_bwd_mu1[p] = bwd_mu1[a]
            t_bwd_c00[p] = bwd_c00[a]
            t_bwd_c01[p] = bwd_c01[a]
            t_bwd_c11[p] = bwd_c11[a]

        process_batch_games_matern_weighted_h(
            b, batch_offsets, game_p1, game_p2, game_scores,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            game_beta_eff, game_handicaps, prior_mu, prior_sigma,
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            lik_mu[a] = t_lik_mu[p]
            lik_sigma[a] = t_lik_sigma[p]


@njit(cache=True)
def backward_sweep_matern_weighted_h(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_next: np.ndarray,
    app_batch: np.ndarray,
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    game_beta_eff: np.ndarray,
    game_handicaps: np.ndarray,
    lambda_: float,
    sigma_sq: float,
) -> float:
    """Backward sweep with per-game beta_eff and handicaps."""
    max_change = 0.0

    for b in range(num_batches - 1, -1, -1):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        for a in range(a_start, a_end):
            p = app_player[a]
            next_a = app_next[a]
            old_mu0 = bwd_mu0[a]

            if next_a < 0:
                bwd_mu0[a] = 0.0
                bwd_mu1[a] = 0.0
                bwd_c00[a] = INF_COV
                bwd_c01[a] = 0.0
                bwd_c11[a] = INF_COV
            else:
                bl_mu0, bl_mu1, bl_c00, bl_c01, bl_c11 = \
                    mul_2d_with_scalar_lik(
                        bwd_mu0[next_a], bwd_mu1[next_a],
                        bwd_c00[next_a], bwd_c01[next_a], bwd_c11[next_a],
                        lik_mu[next_a], lik_sigma[next_a],
                    )
                next_time = batch_times[app_batch[next_a]]
                elapsed = next_time - batch_time
                nm0, nm1, nc00, nc01, nc11 = matern32_backward_transition(
                    bl_mu0, bl_mu1, bl_c00, bl_c01, bl_c11,
                    elapsed, lambda_, sigma_sq,
                )
                bwd_mu0[a] = nm0
                bwd_mu1[a] = nm1
                bwd_c00[a] = nc00
                bwd_c01[a] = nc01
                bwd_c11[a] = nc11

            change = abs(bwd_mu0[a] - old_mu0)
            if change > max_change:
                max_change = change

            t_fwd_mu0[p] = fwd_mu0[a]
            t_fwd_mu1[p] = fwd_mu1[a]
            t_fwd_c00[p] = fwd_c00[a]
            t_fwd_c01[p] = fwd_c01[a]
            t_fwd_c11[p] = fwd_c11[a]
            t_bwd_mu0[p] = bwd_mu0[a]
            t_bwd_mu1[p] = bwd_mu1[a]
            t_bwd_c00[p] = bwd_c00[a]
            t_bwd_c01[p] = bwd_c01[a]
            t_bwd_c11[p] = bwd_c11[a]

        process_batch_games_matern_weighted_h(
            b, batch_offsets, game_p1, game_p2, game_scores,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            game_beta_eff, game_handicaps, prior_mu, prior_sigma,
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            lik_mu[a] = t_lik_mu[p]
            lik_sigma[a] = t_lik_sigma[p]

    return max_change


@njit(cache=True)
def forward_sweep_matern_weighted_h(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_prev: np.ndarray,
    app_batch: np.ndarray,
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
    game_beta_eff: np.ndarray,
    game_handicaps: np.ndarray,
    lambda_: float,
    sigma_sq: float,
) -> float:
    """Forward sweep with per-game beta_eff and handicaps."""
    max_change = 0.0

    for b in range(num_batches):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        for a in range(a_start, a_end):
            p = app_player[a]
            prev_a = app_prev[a]
            old_mu0 = fwd_mu0[a]

            if prev_a < 0:
                fwd_mu0[a] = prior_mu
                fwd_mu1[a] = 0.0
                fwd_c00[a] = prior_c00
                fwd_c01[a] = prior_c01
                fwd_c11[a] = prior_c11
            else:
                prev_time = batch_times[app_batch[prev_a]]
                fl_mu0, fl_mu1, fl_c00, fl_c01, fl_c11 = \
                    mul_2d_with_scalar_lik(
                        fwd_mu0[prev_a], fwd_mu1[prev_a],
                        fwd_c00[prev_a], fwd_c01[prev_a], fwd_c11[prev_a],
                        lik_mu[prev_a], lik_sigma[prev_a],
                    )
                elapsed = batch_time - prev_time
                nm0, nm1, nc00, nc01, nc11 = matern32_transition(
                    fl_mu0, fl_mu1, fl_c00, fl_c01, fl_c11,
                    elapsed, lambda_, sigma_sq,
                )
                fwd_mu0[a] = nm0
                fwd_mu1[a] = nm1
                fwd_c00[a] = nc00
                fwd_c01[a] = nc01
                fwd_c11[a] = nc11

            change = abs(fwd_mu0[a] - old_mu0)
            if change > max_change:
                max_change = change

            t_fwd_mu0[p] = fwd_mu0[a]
            t_fwd_mu1[p] = fwd_mu1[a]
            t_fwd_c00[p] = fwd_c00[a]
            t_fwd_c01[p] = fwd_c01[a]
            t_fwd_c11[p] = fwd_c11[a]
            t_bwd_mu0[p] = bwd_mu0[a]
            t_bwd_mu1[p] = bwd_mu1[a]
            t_bwd_c00[p] = bwd_c00[a]
            t_bwd_c01[p] = bwd_c01[a]
            t_bwd_c11[p] = bwd_c11[a]

        process_batch_games_matern_weighted_h(
            b, batch_offsets, game_p1, game_p2, game_scores,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            game_beta_eff, game_handicaps, prior_mu, prior_sigma,
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            lik_mu[a] = t_lik_mu[p]
            lik_sigma[a] = t_lik_sigma[p]

    return max_change


@njit(cache=True)
def run_convergence_matern_weighted_h(
    num_batches: int,
    batch_offsets: np.ndarray,
    batch_times: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    num_players: int,
    app_offsets: np.ndarray,
    app_player: np.ndarray,
    app_prev: np.ndarray,
    app_next: np.ndarray,
    app_batch: np.ndarray,
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
    game_beta_eff: np.ndarray,
    game_handicaps: np.ndarray,
    lambda_: float,
    sigma_sq: float,
    max_iterations: int,
    epsilon: float,
) -> int:
    """Run convergence with per-game beta_eff and handicaps."""
    for iteration in range(max_iterations):
        bwd_change = backward_sweep_matern_weighted_h(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            app_offsets, app_player, app_next, app_batch,
            fwd_mu0, fwd_mu1, fwd_c00, fwd_c01, fwd_c11,
            bwd_mu0, bwd_mu1, bwd_c00, bwd_c01, bwd_c11,
            lik_mu, lik_sigma,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            prior_mu, prior_sigma,
            game_beta_eff, game_handicaps, lambda_, sigma_sq,
        )

        fwd_change = forward_sweep_matern_weighted_h(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            app_offsets, app_player, app_prev, app_batch,
            fwd_mu0, fwd_mu1, fwd_c00, fwd_c01, fwd_c11,
            bwd_mu0, bwd_mu1, bwd_c00, bwd_c01, bwd_c11,
            lik_mu, lik_sigma,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            prior_mu, prior_sigma,
            prior_c00, prior_c01, prior_c11,
            game_beta_eff, game_handicaps, lambda_, sigma_sq,
        )

        max_change = max(bwd_change, fwd_change)
        if max_change < epsilon:
            return iteration + 1

    return max_iterations
