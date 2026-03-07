"""
Numba-accelerated core functions for Matérn 3/2 TrueSkill Through Time.

Replaces the Wiener process (Matérn 1/2) dynamics of standard TTT with
Matérn 3/2 dynamics that model autocorrelated skill trajectories.

The key difference: state is 2D [skill, velocity/λ] instead of scalar.
The velocity component is rescaled by 1/λ to ensure the stationary
covariance is σ²·I (condition number = 1), avoiding catastrophic
cancellation in the covariance propagation.

The game observation model (win/loss) only depends on the skill component
(H = [1, 0]), so compute_game_likelihoods() from TTT is 100% reusable.

State storage per appearance:
  Forward:    mu0, mu1, cov00, cov01, cov11  (5 arrays)
  Backward:   mu0, mu1, cov00, cov01, cov11  (5 arrays)
  Likelihood: mu, sigma                       (2 arrays, scalar)
  Total: 12 arrays vs 6 in TTT

Matérn 3/2 transition (Hartikainen & Särkkä 2010, KickScore):
  λ = √3 / lengthscale, ldt = λ·dt
  Rescaled: F_z(dt) = exp(-ldt) × [[1+ldt,  ldt ],
                                    [-ldt,  1-ldt]]
  Q_z(dt) = rescaled process noise (see matern32_transition)
  P∞_z = σ² × I  (rescaled stationary covariance)
"""

import math
import numpy as np
from numba import njit, prange

# Reuse all shared functions from TTT
from ..trueskill_through_time._numba_core import (
    # Constants
    INF_SIGMA,
    SQRT2PI,
    # Gaussian utilities (scalar)
    norm_pdf,
    norm_cdf,
    gaussian_mul,
    gaussian_div,
    # Truncation
    v_w_win,
    trunc,
    # Game likelihood (scalar — observation is 1D)
    compute_game_likelihoods,
    compute_game_likelihoods_h,
    # Structure building
    build_appearance_structure,
    # Prediction (once ratings/rd are extracted, prediction is identical)
    predict_proba_batch,
    predict_single,
    predict_proba_batch_h,
    predict_single_h,
    # Day extraction
    extract_player_last_day_ttt,
)

# Threshold for "uninformative" 2D state
INF_COV = 1e20


# =============================================================================
# 2D Gaussian operations
# =============================================================================

@njit(cache=True, fastmath=True)
def matern32_transition(
    mu0: float, mu1: float,
    cov00: float, cov01: float, cov11: float,
    dt: float,
    lambda_: float,
    sigma_sq: float,
) -> tuple:
    """
    Apply Matérn 3/2 state transition to 2D Gaussian.

    Uses rescaled state [skill, velocity/λ] so the stationary covariance is
    σ²·I (condition number = 1), avoiding catastrophic cancellation.

    Rescaled transition matrix:
      F_z(dt) = exp(-λdt) × [[1+λdt,  λdt ],
                              [-λdt,  1-λdt]]

    Rescaled process noise Q_z(dt) entries use ldt = λdt, c = exp(-2λdt):
      qz00 = σ²·(1 - c·(2ldt² + 2ldt + 1))
      qz01 = σ²·c·2·ldt²
      qz11 = σ²·(1 - c·(2ldt² - 2ldt + 1))

    Args:
        mu0, mu1: 2D mean [skill, velocity/λ]
        cov00, cov01, cov11: 2D covariance (symmetric, cov10 = cov01)
        dt: time elapsed (days)
        lambda_: √3 / lengthscale
        sigma_sq: marginal variance σ²

    Returns:
        (new_mu0, new_mu1, new_cov00, new_cov01, new_cov11)
    """
    if dt <= 0:
        return mu0, mu1, cov00, cov01, cov11

    ldt = lambda_ * dt
    e = math.exp(-ldt)
    c = e * e  # exp(-2λdt)

    # Rescaled F_z(dt) matrix elements (all O(1), well-conditioned)
    f00 = e * (1.0 + ldt)
    f01 = e * ldt
    f10 = e * (-ldt)
    f11 = e * (1.0 - ldt)

    # Propagate mean: mu_new = F_z · mu
    new_mu0 = f00 * mu0 + f01 * mu1
    new_mu1 = f10 * mu0 + f11 * mu1

    # Propagate covariance: F_z · Σ · F_z^T
    # First: A = F_z · Σ
    fs00 = f00 * cov00 + f01 * cov01
    fs01 = f00 * cov01 + f01 * cov11
    fs10 = f10 * cov00 + f11 * cov01
    fs11 = f10 * cov01 + f11 * cov11

    # Then: result = A · F_z^T  (F_z^T = [[f00,f10],[f01,f11]])
    fcov00 = fs00 * f00 + fs01 * f01
    fcov01 = fs00 * f10 + fs01 * f11
    fcov11 = fs10 * f10 + fs11 * f11

    # Rescaled process noise Q_z(dt)
    ldt2 = ldt * ldt
    q00 = sigma_sq * (1.0 - c * (2.0 * ldt2 + 2.0 * ldt + 1.0))
    q01 = sigma_sq * c * 2.0 * ldt2
    q11 = sigma_sq * (1.0 - c * (2.0 * ldt2 - 2.0 * ldt + 1.0))

    new_cov00 = fcov00 + q00
    new_cov01 = fcov01 + q01
    new_cov11 = fcov11 + q11

    return new_mu0, new_mu1, new_cov00, new_cov01, new_cov11


@njit(cache=True, fastmath=True)
def matern32_backward_transition(
    mu0: float, mu1: float,
    cov00: float, cov01: float, cov11: float,
    dt: float,
    lambda_: float,
    sigma_sq: float,
) -> tuple:
    """
    Apply backward Matérn 3/2 transition to 2D Gaussian (precision form).

    Uses the same rescaled state [skill, velocity/λ] as matern32_transition.
    Computes via precision form (avoids F^{-1}):
        Λ_back = F_z^T · (S + Q_z)^{-1} · F_z
        η_back = F_z^T · (S + Q_z)^{-1} · m
        Σ_back = Λ_back^{-1}
        μ_back = Σ_back · η_back
    """
    if dt <= 0:
        return mu0, mu1, cov00, cov01, cov11

    # If incoming covariance is uninformative, backward message is uninformative
    if cov00 > INF_COV * 0.1:
        return 0.0, 0.0, INF_COV, 0.0, INF_COV

    ldt = lambda_ * dt
    e = math.exp(-ldt)

    # Rescaled F_z(dt) matrix elements (all O(1), well-conditioned)
    f00 = e * (1.0 + ldt)
    f01 = e * ldt
    f10 = e * (-ldt)
    f11 = e * (1.0 - ldt)

    # Rescaled process noise Q_z(dt)
    ldt2 = ldt * ldt
    c = e * e  # exp(-2λdt)
    q00 = sigma_sq * (1.0 - c * (2.0 * ldt2 + 2.0 * ldt + 1.0))
    q01 = sigma_sq * c * 2.0 * ldt2
    q11 = sigma_sq * (1.0 - c * (2.0 * ldt2 - 2.0 * ldt + 1.0))

    # M = S + Q_z
    m00 = cov00 + q00
    m01 = cov01 + q01
    m11 = cov11 + q11

    # M^{-1}
    det_m = m00 * m11 - m01 * m01
    if abs(det_m) < 1e-30:
        return 0.0, 0.0, INF_COV, 0.0, INF_COV
    inv_det_m = 1.0 / det_m
    mi00 = m11 * inv_det_m
    mi01 = -m01 * inv_det_m
    mi11 = m00 * inv_det_m

    # η_back = F_z^T · M^{-1} · m  (F_z^T is [[f00, f10], [f01, f11]])
    v0 = mi00 * mu0 + mi01 * mu1
    v1 = mi01 * mu0 + mi11 * mu1
    eta0 = f00 * v0 + f10 * v1
    eta1 = f01 * v0 + f11 * v1

    # Λ_back = F_z^T · M^{-1} · F_z
    w00 = mi00 * f00 + mi01 * f10
    w01 = mi00 * f01 + mi01 * f11
    w10 = mi01 * f00 + mi11 * f10
    w11 = mi01 * f01 + mi11 * f11
    lam00 = f00 * w00 + f10 * w10
    lam01 = f00 * w01 + f10 * w11
    lam11 = f01 * w01 + f11 * w11

    # Σ_back = Λ_back^{-1}
    det_lam = lam00 * lam11 - lam01 * lam01
    if det_lam < 1e-30:
        return 0.0, 0.0, INF_COV, 0.0, INF_COV
    inv_det_lam = 1.0 / det_lam
    new_cov00 = lam11 * inv_det_lam
    new_cov01 = -lam01 * inv_det_lam
    new_cov11 = lam00 * inv_det_lam

    # μ_back = Σ_back · η_back
    new_mu0 = new_cov00 * eta0 + new_cov01 * eta1
    new_mu1 = new_cov01 * eta0 + new_cov11 * eta1

    return new_mu0, new_mu1, new_cov00, new_cov01, new_cov11


@njit(cache=True, fastmath=True)
def gaussian_mul_2d(
    mu0_a: float, mu1_a: float,
    c00_a: float, c01_a: float, c11_a: float,
    mu0_b: float, mu1_b: float,
    c00_b: float, c01_b: float, c11_b: float,
) -> tuple:
    """
    Multiply two 2D Gaussians in precision form.

    Product N(μ_a, Σ_a) × N(μ_b, Σ_b):
      Π = Σ_a^{-1} + Σ_b^{-1}
      τ = Σ_a^{-1}·μ_a + Σ_b^{-1}·μ_b
      Σ = Π^{-1}
      μ = Σ · τ

    Returns (mu0, mu1, cov00, cov01, cov11) of the product.
    """
    # Check if either is uninformative
    if c00_a > INF_COV * 0.1:
        return mu0_b, mu1_b, c00_b, c01_b, c11_b
    if c00_b > INF_COV * 0.1:
        return mu0_a, mu1_a, c00_a, c01_a, c11_a

    # Invert Σ_a: precision_a = Σ_a^{-1}
    det_a = c00_a * c11_a - c01_a * c01_a
    if abs(det_a) < 1e-30:
        return mu0_b, mu1_b, c00_b, c01_b, c11_b
    inv_det_a = 1.0 / det_a
    pa00 = c11_a * inv_det_a
    pa01 = -c01_a * inv_det_a
    pa11 = c00_a * inv_det_a

    # Invert Σ_b
    det_b = c00_b * c11_b - c01_b * c01_b
    if abs(det_b) < 1e-30:
        return mu0_a, mu1_a, c00_a, c01_a, c11_a
    inv_det_b = 1.0 / det_b
    pb00 = c11_b * inv_det_b
    pb01 = -c01_b * inv_det_b
    pb11 = c00_b * inv_det_b

    # Π = Pa + Pb
    pi00 = pa00 + pb00
    pi01 = pa01 + pb01
    pi11 = pa11 + pb11

    # τ = Pa·μ_a + Pb·μ_b
    tau0 = (pa00 * mu0_a + pa01 * mu1_a) + (pb00 * mu0_b + pb01 * mu1_b)
    tau1 = (pa01 * mu0_a + pa11 * mu1_a) + (pb01 * mu0_b + pb11 * mu1_b)

    # Σ = Π^{-1}
    det_pi = pi00 * pi11 - pi01 * pi01
    if abs(det_pi) < 1e-30:
        return mu0_a, mu1_a, c00_a, c01_a, c11_a
    inv_det_pi = 1.0 / det_pi
    cov00 = pi11 * inv_det_pi
    cov01 = -pi01 * inv_det_pi
    cov11 = pi00 * inv_det_pi

    # μ = Σ · τ
    mu0 = cov00 * tau0 + cov01 * tau1
    mu1 = cov01 * tau0 + cov11 * tau1

    return mu0, mu1, cov00, cov01, cov11


@njit(cache=True, fastmath=True)
def gaussian_div_2d(
    mu0_a: float, mu1_a: float,
    c00_a: float, c01_a: float, c11_a: float,
    mu0_b: float, mu1_b: float,
    c00_b: float, c01_b: float, c11_b: float,
    prior_mu0: float, prior_mu1: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
) -> tuple:
    """
    Divide two 2D Gaussians: N(μ_a, Σ_a) / N(μ_b, Σ_b).

    Precision form: Π = Pa - Pb. Falls back to prior if result
    is degenerate.
    """
    if c00_b > INF_COV * 0.1:
        return mu0_a, mu1_a, c00_a, c01_a, c11_a
    if c00_a > INF_COV * 0.1:
        return prior_mu0, prior_mu1, prior_c00, prior_c01, prior_c11

    # Invert Σ_a
    det_a = c00_a * c11_a - c01_a * c01_a
    if abs(det_a) < 1e-30:
        return prior_mu0, prior_mu1, prior_c00, prior_c01, prior_c11
    inv_det_a = 1.0 / det_a
    pa00 = c11_a * inv_det_a
    pa01 = -c01_a * inv_det_a
    pa11 = c00_a * inv_det_a

    # Invert Σ_b
    det_b = c00_b * c11_b - c01_b * c01_b
    if abs(det_b) < 1e-30:
        return mu0_a, mu1_a, c00_a, c01_a, c11_a
    inv_det_b = 1.0 / det_b
    pb00 = c11_b * inv_det_b
    pb01 = -c01_b * inv_det_b
    pb11 = c00_b * inv_det_b

    # Π = Pa - Pb
    pi00 = pa00 - pb00
    pi01 = pa01 - pb01
    pi11 = pa11 - pb11

    # Check positive definiteness
    det_pi = pi00 * pi11 - pi01 * pi01
    if pi00 < 1e-10 or det_pi < 1e-10:
        return prior_mu0, prior_mu1, prior_c00, prior_c01, prior_c11

    # τ = Pa·μ_a - Pb·μ_b
    tau0 = (pa00 * mu0_a + pa01 * mu1_a) - (pb00 * mu0_b + pb01 * mu1_b)
    tau1 = (pa01 * mu0_a + pa11 * mu1_a) - (pb01 * mu0_b + pb11 * mu1_b)

    # Σ = Π^{-1}
    inv_det_pi = 1.0 / det_pi
    cov00 = pi11 * inv_det_pi
    cov01 = -pi01 * inv_det_pi
    cov11 = pi00 * inv_det_pi

    # μ = Σ · τ
    mu0 = cov00 * tau0 + cov01 * tau1
    mu1 = cov01 * tau0 + cov11 * tau1

    return mu0, mu1, cov00, cov01, cov11


@njit(cache=True, fastmath=True)
def extract_scalar_marginal(
    mu0: float, mu1: float,
    cov00: float, cov01: float, cov11: float,
) -> tuple:
    """
    Extract skill marginal from 2D state: (mu[0], sqrt(cov[0,0])).
    This is the projection H = [1, 0] applied to the 2D Gaussian.
    """
    sigma = math.sqrt(max(cov00, 1e-20))
    return mu0, sigma


# =============================================================================
# Batch game processing for Matérn TTT
# =============================================================================

@njit(cache=True)
def process_batch_games_matern(
    batch_idx: int,
    batch_offsets: np.ndarray,
    game_p1: np.ndarray,
    game_p2: np.ndarray,
    game_scores: np.ndarray,
    # 2D forward state (temp, by player)
    player_fwd_mu0: np.ndarray,
    player_fwd_mu1: np.ndarray,
    player_fwd_c00: np.ndarray,
    player_fwd_c01: np.ndarray,
    player_fwd_c11: np.ndarray,
    # 2D backward state (temp, by player)
    player_bwd_mu0: np.ndarray,
    player_bwd_mu1: np.ndarray,
    player_bwd_c00: np.ndarray,
    player_bwd_c01: np.ndarray,
    player_bwd_c11: np.ndarray,
    # Scalar likelihood (temp, by player)
    player_lik_mu: np.ndarray,
    player_lik_sigma: np.ndarray,
    # Parameters
    beta: float,
    prior_mu: float,
    prior_sigma: float,
) -> None:
    """
    Process all games in a batch, computing scalar likelihoods.

    Extracts scalar skill marginals from 2D forward×backward,
    then feeds to standard compute_game_likelihoods().
    """
    game_start = batch_offsets[batch_idx]
    game_end = batch_offsets[batch_idx + 1]

    # Reset likelihoods
    for g in range(game_start, game_end):
        p1, p2 = game_p1[g], game_p2[g]
        player_lik_mu[p1] = 0.0
        player_lik_sigma[p1] = INF_SIGMA
        player_lik_mu[p2] = 0.0
        player_lik_sigma[p2] = INF_SIGMA

    for g in range(game_start, game_end):
        p1, p2 = game_p1[g], game_p2[g]
        p1_wins = game_scores[g] > 0.5

        # Combine forward × backward → 2D posterior, then extract scalar
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

        # Compute scalar likelihoods (reused from TTT)
        lik1_mu, lik1_sigma, lik2_mu, lik2_sigma = compute_game_likelihoods(
            p1_mu, p1_sigma, p2_mu, p2_sigma, p1_wins, beta
        )

        player_lik_mu[p1], player_lik_sigma[p1] = gaussian_mul(
            player_lik_mu[p1], player_lik_sigma[p1], lik1_mu, lik1_sigma
        )
        player_lik_mu[p2], player_lik_sigma[p2] = gaussian_mul(
            player_lik_mu[p2], player_lik_sigma[p2], lik2_mu, lik2_sigma
        )


# =============================================================================
# Forward/backward sweeps with 2D Matérn state
# =============================================================================

@njit(cache=True)
def initial_forward_pass_matern(
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
    # 2D forward state (per appearance)
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    # 2D backward state (per appearance)
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    # Scalar likelihood (per appearance)
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    # Temp arrays (per player)
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
    beta: float,
    lambda_: float,
    sigma_sq: float,
    start_batch: int,
) -> None:
    """Forward pass with 2D Matérn state."""
    for b in range(start_batch, num_batches):
        a_start = app_offsets[b]
        a_end = app_offsets[b + 1]
        batch_time = batch_times[b]

        for a in range(a_start, a_end):
            p = app_player[a]
            prev_a = app_prev[a]

            if prev_a < 0:
                # First appearance: use stationary prior
                fwd_mu0[a] = prior_mu
                fwd_mu1[a] = 0.0
                fwd_c00[a] = prior_c00
                fwd_c01[a] = prior_c01
                fwd_c11[a] = prior_c11
            else:
                # Combine forward × likelihood at prev, then transition
                prev_time = batch_times[app_batch[prev_a]]

                # Embed scalar likelihood into 2D via rank-1 update
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

            # Backward uninformative
            bwd_mu0[a] = 0.0
            bwd_mu1[a] = 0.0
            bwd_c00[a] = INF_COV
            bwd_c01[a] = 0.0
            bwd_c11[a] = INF_COV

            # Copy to temp
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

        # Process games
        process_batch_games_matern(
            b, batch_offsets, game_p1, game_p2, game_scores,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            beta, prior_mu, prior_sigma,
        )

        # Copy likelihoods back
        for a in range(a_start, a_end):
            p = app_player[a]
            lik_mu[a] = t_lik_mu[p]
            lik_sigma[a] = t_lik_sigma[p]


@njit(cache=True, fastmath=True)
def mul_2d_with_scalar_lik(
    mu0: float, mu1: float,
    c00: float, c01: float, c11: float,
    lik_mu: float, lik_sigma: float,
) -> tuple:
    """
    Rank-1 precision update: embed scalar likelihood into 2D state.

    The observation model is H = [1, 0], so the scalar likelihood
    N(lik_mu, lik_sigma²) updates only the (0,0) precision entry.

    Equivalent to multiplying the 2D Gaussian by a 2D Gaussian with
    infinite variance in the velocity dimension.
    """
    if lik_sigma > INF_SIGMA * 0.1:
        return mu0, mu1, c00, c01, c11
    if c00 > INF_COV * 0.1:
        # State is uninformative, return likelihood embedded in 2D
        return lik_mu, 0.0, lik_sigma * lik_sigma, 0.0, INF_COV,

    # Precision of state
    det = c00 * c11 - c01 * c01
    if abs(det) < 1e-30:
        return lik_mu, 0.0, lik_sigma * lik_sigma, 0.0, INF_COV

    inv_det = 1.0 / det
    p00 = c11 * inv_det
    p01 = -c01 * inv_det
    p11 = c00 * inv_det

    # Add scalar likelihood precision to (0,0) entry
    lik_prec = 1.0 / (lik_sigma * lik_sigma)
    p00_new = p00 + lik_prec

    # Natural parameters
    tau0 = p00 * mu0 + p01 * mu1 + lik_prec * lik_mu
    tau1 = p01 * mu0 + p11 * mu1

    # Invert new precision
    det_new = p00_new * p11 - p01 * p01
    if abs(det_new) < 1e-30:
        return mu0, mu1, c00, c01, c11

    inv_det_new = 1.0 / det_new
    nc00 = p11 * inv_det_new
    nc01 = -p01 * inv_det_new
    nc11 = p00_new * inv_det_new

    nm0 = nc00 * tau0 + nc01 * tau1
    nm1 = nc01 * tau0 + nc11 * tau1

    return nm0, nm1, nc00, nc01, nc11


@njit(cache=True)
def backward_sweep_matern(
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
    # 2D state (per appearance)
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    # Temp arrays (per player)
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    beta: float,
    lambda_: float,
    sigma_sq: float,
) -> float:
    """Backward sweep with 2D Matérn state. Returns max change."""
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
                # Combine likelihood × backward at next, then reverse-transition
                bl_mu0, bl_mu1, bl_c00, bl_c01, bl_c11 = \
                    mul_2d_with_scalar_lik(
                        bwd_mu0[next_a], bwd_mu1[next_a],
                        bwd_c00[next_a], bwd_c01[next_a], bwd_c11[next_a],
                        lik_mu[next_a], lik_sigma[next_a],
                    )

                next_time = batch_times[app_batch[next_a]]
                elapsed = next_time - batch_time

                # Backward transition: propagate from next back to this.
                # Uses F^{-1}(dt) instead of F(dt).
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

            # Copy to temp
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

        # Recompute likelihoods
        process_batch_games_matern(
            b, batch_offsets, game_p1, game_p2, game_scores,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            beta, prior_mu, prior_sigma,
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            lik_mu[a] = t_lik_mu[p]
            lik_sigma[a] = t_lik_sigma[p]

    return max_change


@njit(cache=True)
def forward_sweep_matern(
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
    # 2D state (per appearance)
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    # Temp arrays (per player)
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
    beta: float,
    lambda_: float,
    sigma_sq: float,
) -> float:
    """Forward sweep with 2D Matérn state. Returns max change."""
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

            # Copy to temp
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

        # Recompute likelihoods
        process_batch_games_matern(
            b, batch_offsets, game_p1, game_p2, game_scores,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            beta, prior_mu, prior_sigma,
        )

        for a in range(a_start, a_end):
            p = app_player[a]
            lik_mu[a] = t_lik_mu[p]
            lik_sigma[a] = t_lik_sigma[p]

    return max_change


@njit(cache=True)
def run_convergence_matern(
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
    # 2D state (per appearance)
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    # Temp arrays (per player)
    t_fwd_mu0: np.ndarray, t_fwd_mu1: np.ndarray,
    t_fwd_c00: np.ndarray, t_fwd_c01: np.ndarray, t_fwd_c11: np.ndarray,
    t_bwd_mu0: np.ndarray, t_bwd_mu1: np.ndarray,
    t_bwd_c00: np.ndarray, t_bwd_c01: np.ndarray, t_bwd_c11: np.ndarray,
    t_lik_mu: np.ndarray, t_lik_sigma: np.ndarray,
    # Parameters
    prior_mu: float,
    prior_sigma: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
    beta: float,
    lambda_: float,
    sigma_sq: float,
    max_iterations: int,
    epsilon: float,
) -> int:
    """Run forward-backward iterations until convergence."""
    for iteration in range(max_iterations):
        bwd_change = backward_sweep_matern(
            num_batches, batch_offsets, batch_times,
            game_p1, game_p2, game_scores, num_players,
            app_offsets, app_player, app_next, app_batch,
            fwd_mu0, fwd_mu1, fwd_c00, fwd_c01, fwd_c11,
            bwd_mu0, bwd_mu1, bwd_c00, bwd_c01, bwd_c11,
            lik_mu, lik_sigma,
            t_fwd_mu0, t_fwd_mu1, t_fwd_c00, t_fwd_c01, t_fwd_c11,
            t_bwd_mu0, t_bwd_mu1, t_bwd_c00, t_bwd_c01, t_bwd_c11,
            t_lik_mu, t_lik_sigma,
            prior_mu, prior_sigma, beta, lambda_, sigma_sq,
        )

        fwd_change = forward_sweep_matern(
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
            beta, lambda_, sigma_sq,
        )

        max_change = max(bwd_change, fwd_change)
        if max_change < epsilon:
            return iteration + 1

    return max_iterations


@njit(cache=True)
def extract_final_ratings_matern(
    num_players: int,
    player_last_app: np.ndarray,
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    ratings_out: np.ndarray,
    rd_out: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    display_scale: float,
    display_offset: float,
) -> None:
    """
    Extract the most recent rating for each player.

    Posterior = forward × backward × likelihood at last appearance.
    Then extract skill marginal (component 0) for rating/RD.
    """
    for p in range(num_players):
        a = player_last_app[p]
        if a < 0:
            ratings_out[p] = display_offset
            rd_out[p] = prior_sigma * display_scale
        else:
            # Combine forward × backward (2D)
            fb_mu0, fb_mu1, fb_c00, fb_c01, fb_c11 = gaussian_mul_2d(
                fwd_mu0[a], fwd_mu1[a],
                fwd_c00[a], fwd_c01[a], fwd_c11[a],
                bwd_mu0[a], bwd_mu1[a],
                bwd_c00[a], bwd_c01[a], bwd_c11[a],
            )
            # Multiply with scalar likelihood (rank-1 update on skill dim)
            post_mu0, post_mu1, post_c00, post_c01, post_c11 = \
                mul_2d_with_scalar_lik(
                    fb_mu0, fb_mu1, fb_c00, fb_c01, fb_c11,
                    lik_mu[a], lik_sigma[a],
                )
            # Extract skill marginal
            skill_mu, skill_sigma = extract_scalar_marginal(
                post_mu0, post_mu1, post_c00, post_c01, post_c11
            )
            ratings_out[p] = skill_mu * display_scale + display_offset
            rd_out[p] = skill_sigma * display_scale


# =============================================================================
# Time-aware prediction for Matérn 3/2
# =============================================================================

@njit(cache=True, parallel=True)
def predict_proba_batch_at_day_matern(
    player1: np.ndarray,
    player2: np.ndarray,
    # Per-player 2D posterior at last appearance
    p_mu0: np.ndarray, p_mu1: np.ndarray,
    p_c00: np.ndarray, p_c01: np.ndarray, p_c11: np.ndarray,
    player_last_day: np.ndarray,
    day: int,
    beta: float,
    lambda_: float,
    sigma_sq: float,
    display_scale: float,
    display_offset: float,
) -> np.ndarray:
    """Predict win probs with 2D state propagated forward to prediction day."""
    n = len(player1)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        p1, p2 = player1[i], player2[i]

        # Propagate player 1 state to current day
        dt1 = max(0, day - player_last_day[p1])
        m0_1, m1_1, c00_1, c01_1, c11_1 = matern32_transition(
            p_mu0[p1], p_mu1[p1],
            p_c00[p1], p_c01[p1], p_c11[p1],
            float(dt1), lambda_, sigma_sq,
        )

        # Propagate player 2 state
        dt2 = max(0, day - player_last_day[p2])
        m0_2, m1_2, c00_2, c01_2, c11_2 = matern32_transition(
            p_mu0[p2], p_mu1[p2],
            p_c00[p2], p_c01[p2], p_c11[p2],
            float(dt2), lambda_, sigma_sq,
        )

        # Extract skill marginals
        mu1_val = m0_1
        sigma1 = math.sqrt(max(c00_1, 1e-20))
        mu2_val = m0_2
        sigma2 = math.sqrt(max(c00_2, 1e-20))

        diff_mu = mu1_val - mu2_val
        diff_sigma = math.sqrt(sigma1*sigma1 + sigma2*sigma2 + 2.0*beta*beta)
        result[i] = norm_cdf(diff_mu / diff_sigma)

    return result


@njit(cache=True)
def predict_single_at_day_matern(
    mu0_1: float, mu1_1: float,
    c00_1: float, c01_1: float, c11_1: float,
    mu0_2: float, mu1_2: float,
    c00_2: float, c01_2: float, c11_2: float,
    last_day1: int, last_day2: int,
    day: int,
    beta: float,
    lambda_: float,
    sigma_sq: float,
) -> float:
    """Predict single win prob with 2D state propagated to day."""
    dt1 = max(0, day - last_day1)
    m0_1, _, c00_1p, _, _ = matern32_transition(
        mu0_1, mu1_1, c00_1, c01_1, c11_1,
        float(dt1), lambda_, sigma_sq,
    )

    dt2 = max(0, day - last_day2)
    m0_2, _, c00_2p, _, _ = matern32_transition(
        mu0_2, mu1_2, c00_2, c01_2, c11_2,
        float(dt2), lambda_, sigma_sq,
    )

    sigma1 = math.sqrt(max(c00_1p, 1e-20))
    sigma2 = math.sqrt(max(c00_2p, 1e-20))

    diff_mu = m0_1 - m0_2
    diff_sigma = math.sqrt(sigma1*sigma1 + sigma2*sigma2 + 2.0*beta*beta)
    return norm_cdf(diff_mu / diff_sigma)


# =============================================================================
# Store per-player 2D posterior (for time-aware prediction)
# =============================================================================

@njit(cache=True)
def extract_player_posteriors_matern(
    num_players: int,
    player_last_app: np.ndarray,
    fwd_mu0: np.ndarray, fwd_mu1: np.ndarray,
    fwd_c00: np.ndarray, fwd_c01: np.ndarray, fwd_c11: np.ndarray,
    bwd_mu0: np.ndarray, bwd_mu1: np.ndarray,
    bwd_c00: np.ndarray, bwd_c01: np.ndarray, bwd_c11: np.ndarray,
    lik_mu: np.ndarray, lik_sigma: np.ndarray,
    # Output: per-player 2D posterior
    out_mu0: np.ndarray, out_mu1: np.ndarray,
    out_c00: np.ndarray, out_c01: np.ndarray, out_c11: np.ndarray,
    prior_mu: float,
    prior_c00: float, prior_c01: float, prior_c11: float,
) -> None:
    """Extract 2D posterior at last appearance for each player."""
    for p in range(num_players):
        a = player_last_app[p]
        if a < 0:
            out_mu0[p] = prior_mu
            out_mu1[p] = 0.0
            out_c00[p] = prior_c00
            out_c01[p] = prior_c01
            out_c11[p] = prior_c11
        else:
            fb_mu0, fb_mu1, fb_c00, fb_c01, fb_c11 = gaussian_mul_2d(
                fwd_mu0[a], fwd_mu1[a],
                fwd_c00[a], fwd_c01[a], fwd_c11[a],
                bwd_mu0[a], bwd_mu1[a],
                bwd_c00[a], bwd_c01[a], bwd_c11[a],
            )
            pm0, pm1, pc00, pc01, pc11 = mul_2d_with_scalar_lik(
                fb_mu0, fb_mu1, fb_c00, fb_c01, fb_c11,
                lik_mu[a], lik_sigma[a],
            )
            out_mu0[p] = pm0
            out_mu1[p] = pm1
            out_c00[p] = pc00
            out_c01[p] = pc01
            out_c11[p] = pc11
