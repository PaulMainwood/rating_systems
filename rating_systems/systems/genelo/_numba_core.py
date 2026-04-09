"""
Numba-accelerated core functions for GenElo (Generalised Elo).

Implements the Bayesian Elo update from Ingram (2021), "How to extend Elo:
a Bayesian perspective". The key difference from standard Elo is that the
k-factor depends on the prior skill difference (Eq. 17 in the paper):

    k = (b/2) / (1/σ²_δ + b² γ(bμ_δ)(1 - γ(bμ_δ)))

For the margin-of-victory extension (Eq. 24-27), the update decomposes into
a win component and a margin component with a shared k.

Prediction uses the approximate marginal likelihood (Eq. 33):
    P(win) ≈ γ(b μ_δ / α)  where α = sqrt(1 + π σ²_δ b² / 8)
"""

import numpy as np
from numba import njit, prange

# Sentinel value for missing margins (fastmath=True breaks np.isnan)
MARGIN_MISSING = 1e30


@njit(cache=True, fastmath=True, inline="always")
def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


# ---------------------------------------------------------------------------
# Win/loss only (Section 2.3)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def fit_all_days_genelo(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    day_offsets: np.ndarray,
    ratings: np.ndarray,
    sigma_delta_sq: float,
    b: float,
    use_taylor_k: bool,
) -> None:
    """
    Fit GenElo ratings for ALL days in a single Numba call (win/loss only).

    With use_taylor_k=True, uses the Bayesian k-factor from Eq. 17.
    With use_taylor_k=False, uses a constant k = b * sigma_delta_sq / 2,
    which recovers standard Elo.

    Args:
        player1: All player1 IDs (sorted by day)
        player2: All player2 IDs (sorted by day)
        scores: All scores (sorted by day), 1.0=P1 wins, 0.0=P2 wins
        day_offsets: Start index for each day (length = num_days + 1)
        ratings: Ratings array to update in-place
        sigma_delta_sq: Prior variance of the skill difference (2σ²)
        b: Elo scale factor, log(10)/scale
        use_taylor_k: Whether to use Bayesian k or constant k
    """
    n_days = len(day_offsets) - 1
    b_sq = b * b

    # Constant k (used when use_taylor_k=False)
    k_const = b * sigma_delta_sq / 2.0
    inv_sigma_delta_sq = 1.0 / sigma_delta_sq

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            score = scores[i]

            r1 = ratings[p1]
            r2 = ratings[p2]
            delta_mu = r1 - r2

            gamma = _sigmoid(b * delta_mu)

            if use_taylor_k:
                # Bayesian k-factor (Eq. 17)
                eta = gamma * (1.0 - gamma)
                k = (b / 2.0) / (inv_sigma_delta_sq + b_sq * eta)
            else:
                k = k_const / 2.0

            # Update (Eq. 18-19): each player gets k × residual
            residual = score - gamma
            update = 2.0 * k * residual
            ratings[p1] = r1 + update / 2.0
            ratings[p2] = r2 - update / 2.0


# ---------------------------------------------------------------------------
# Win/loss + margin of victory (Section 2.4)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def fit_all_days_genelo_margin(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    margins: np.ndarray,
    day_offsets: np.ndarray,
    ratings: np.ndarray,
    sigma_delta_sq: float,
    b: float,
    c1: float,
    c2: float,
    sigma_obs_sq: float,
) -> None:
    """
    Fit GenElo ratings with margin of victory for ALL days.

    Implements Eq. 24-27. The update has two components:
      - Win component: k_win × (score - γ)
      - Margin component: k_margin × (margin - s_pred)

    Args:
        player1: All player1 IDs (sorted by day)
        player2: All player2 IDs (sorted by day)
        scores: All scores (sorted by day), 1.0=P1 wins, 0.0=P2 wins
        margins: Margin of victory for each game (from P1's perspective)
        day_offsets: Start index for each day (length = num_days + 1)
        ratings: Ratings array to update in-place
        sigma_delta_sq: Prior variance of the skill difference (2σ²)
        b: Elo scale factor, log(10)/scale
        c1: Margin slope parameter
        c2: Margin offset parameter (winner's expected margin advantage)
        sigma_obs_sq: Observation noise variance for margins
    """
    n_days = len(day_offsets) - 1
    b_sq = b * b
    c1_sq = c1 * c1
    inv_sigma_delta_sq = 1.0 / sigma_delta_sq
    c1_over_sigma_obs_sq = c1 / sigma_obs_sq

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            score = scores[i]
            margin = margins[i]

            r1 = ratings[p1]
            r2 = ratings[p2]
            delta_mu = r1 - r2

            gamma = _sigmoid(b * delta_mu)
            eta = gamma * (1.0 - gamma)

            # MARGIN_MISSING sentinel → win-only update (skip margin component)
            if margin >= MARGIN_MISSING:
                k = (b / 2.0) / (inv_sigma_delta_sq + b_sq * eta)
                update = 2.0 * k * (score - gamma)
            else:
                # Shared k denominator (Eq. 27)
                denom = inv_sigma_delta_sq + b_sq * eta + c1_sq / sigma_obs_sq
                k_shared = 0.5 / denom

                # Win component (Eq. 26)
                k_win = b * k_shared
                win_residual = score - gamma

                # Margin component (Eq. 26)
                # s_pred: expected margin from P1's perspective
                # If P1 won (score=1): s_pred = c1*delta_mu + c2
                # If P2 won (score=0): s_pred = c1*delta_mu - c2
                # Combined: s_pred = c1*delta_mu + c2*(2*score - 1)
                s_pred = c1 * delta_mu + c2 * (2.0 * score - 1.0)
                k_margin = c1_over_sigma_obs_sq * k_shared
                margin_residual = margin - s_pred

                # Combined update for each player
                update = 2.0 * k_win * win_residual + 2.0 * k_margin * margin_residual
            ratings[p1] = r1 + update / 2.0
            ratings[p2] = r2 - update / 2.0


# ---------------------------------------------------------------------------
# Prediction (Section 2.7, Appendix E)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def predict_single_genelo(
    r1: float,
    r2: float,
    b: float,
    sigma_delta_sq: float,
) -> float:
    """
    Predict P(player1 wins) using approximate marginal likelihood (Eq. 33).

    P(win) ≈ γ(b μ_δ / α) where α = sqrt(1 + π σ²_δ b² / 8)
    """
    delta_mu = r1 - r2
    alpha = np.sqrt(1.0 + np.pi * sigma_delta_sq * b * b / 8.0)
    return _sigmoid(b * delta_mu / alpha)


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_genelo_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    ratings: np.ndarray,
    b: float,
    sigma_delta_sq: float,
) -> np.ndarray:
    """
    Predict win probabilities for a batch of matchups (fully parallel).

    Uses the approximate marginal likelihood from Eq. 33 / Appendix E.
    """
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)
    alpha = np.sqrt(1.0 + np.pi * sigma_delta_sq * b * b / 8.0)
    b_over_alpha = b / alpha

    for i in prange(n):
        r1 = ratings[player1[i]]
        r2 = ratings[player2[i]]
        proba[i] = _sigmoid((r1 - r2) * b_over_alpha)

    return proba


# ---------------------------------------------------------------------------
# Log marginal likelihood (for parameter estimation, Eq. 33)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def log_marginal_likelihood_scalar(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    margins: np.ndarray,
    day_offsets: np.ndarray,
    ratings: np.ndarray,
    sigma_delta_sq: float,
    b: float,
    c1: float,
    c2: float,
    sigma_obs_sq: float,
    use_margin: bool,
    use_taylor_k: bool,
) -> float:
    """
    Compute the log marginal likelihood over a dataset (Eq. 33).

    This runs a forward pass: for each match, compute the pre-match
    prediction probability, accumulate its log, then update ratings.
    Returns the total log-likelihood.

    Note: This modifies ratings in-place (performs a fit).
    """
    n_days = len(day_offsets) - 1
    b_sq = b * b
    inv_sigma_delta_sq = 1.0 / sigma_delta_sq
    alpha = np.sqrt(1.0 + np.pi * sigma_delta_sq * b_sq / 8.0)
    b_over_alpha = b / alpha

    total_ll = 0.0
    n_matches = 0

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            score = scores[i]

            r1 = ratings[p1]
            r2 = ratings[p2]
            delta_mu = r1 - r2

            # Log-likelihood of the win outcome (Eq. 33, first term)
            p_win = _sigmoid(b_over_alpha * delta_mu)
            if score > 0.5:
                ll_win = np.log(max(p_win, 1e-15))
            else:
                ll_win = np.log(max(1.0 - p_win, 1e-15))
            total_ll += ll_win

            # Log-likelihood of the margin (Eq. 33, second term)
            if use_margin:
                margin = margins[i]
                if margin < MARGIN_MISSING:
                    s_pred = c1 * delta_mu + c2 * (2.0 * score - 1.0)
                    var_margin = sigma_obs_sq + c1 * c1 * sigma_delta_sq
                    ll_margin = -0.5 * np.log(2.0 * np.pi * var_margin) - \
                        0.5 * (margin - s_pred) ** 2 / var_margin
                    total_ll += ll_margin

            n_matches += 1

            # Now update ratings (so subsequent predictions use updated values)
            gamma = _sigmoid(b * delta_mu)
            eta = gamma * (1.0 - gamma)

            if use_margin and margins[i] < MARGIN_MISSING:
                c1_sq = c1 * c1
                denom = inv_sigma_delta_sq + b_sq * eta + c1_sq / sigma_obs_sq
                k_shared = 0.5 / denom
                k_win = b * k_shared
                k_margin_val = (c1 / sigma_obs_sq) * k_shared
                s_pred_upd = c1 * delta_mu + c2 * (2.0 * score - 1.0)
                update = (2.0 * k_win * (score - gamma) +
                          2.0 * k_margin_val * (margins[i] - s_pred_upd))
            elif use_taylor_k:
                k = (b / 2.0) / (inv_sigma_delta_sq + b_sq * eta)
                update = 2.0 * k * (score - gamma)
            else:
                k_const = b * sigma_delta_sq / 2.0
                update = k_const * (score - gamma)

            ratings[p1] = r1 + update / 2.0
            ratings[p2] = r2 - update / 2.0

    return total_ll
