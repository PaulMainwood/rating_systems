"""
Parameter estimation for GenElo models.

Estimates hyperparameters by maximising the log marginal likelihood (Eq. 33).
Uses scipy.optimize for numerical optimisation.

The log marginal likelihood for a single match is (Eq. 33):
    log p(y=1, s) ≈ log γ(b μ_δ / α) + log N(s | c₁μ_δ + c₂, σ²_obs + c₁²σ²_δ)
where α = sqrt(1 + π σ²_δ b² / 8).

The full likelihood sums over all matches, computed during a forward pass
(predict then update for each match).
"""

import numpy as np
from scipy.optimize import minimize

from ...data import GameDataset
from ._numba_core import log_marginal_likelihood_scalar


def estimate_params_scalar(
    dataset: GameDataset,
    margins: np.ndarray = None,
    scale: float = 400.0,
    initial_rating: float = 1500.0,
    use_margin: bool = False,
    initial_sigma: float = 84.0,
    initial_c1: float = 0.000131,
    initial_c2: float = 0.102,
    initial_sigma_obs: float = 0.085,
    verbose: bool = False,
) -> dict:
    """
    Estimate scalar GenElo parameters by maximising log marginal likelihood.

    Args:
        dataset: Training dataset.
        margins: Per-game margin array (required if use_margin=True).
        scale: Elo scale (fixed, not optimised).
        initial_rating: Starting rating (fixed).
        use_margin: Whether to estimate margin parameters.
        initial_sigma: Starting value for sigma optimisation.
        initial_c1: Starting value for c1 (margin slope).
        initial_c2: Starting value for c2 (margin offset).
        initial_sigma_obs: Starting value for sigma_obs (margin noise).
        verbose: Print progress.

    Returns:
        Dictionary with estimated parameters: {sigma, c1, c2, sigma_obs, log_likelihood}.
    """
    b = np.log(10.0) / scale
    player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()
    n_players = dataset.num_players

    if use_margin and margins is None:
        raise ValueError("margins required when use_margin=True")

    if margins is not None:
        margins = np.ascontiguousarray(margins, dtype=np.float64)
    else:
        margins = np.zeros(len(player1), dtype=np.float64)

    def neg_ll(params):
        if use_margin:
            sigma, c1, c2, sigma_obs = params
            if sigma <= 0 or sigma_obs <= 0:
                return 1e10
        else:
            sigma = params[0]
            c1, c2, sigma_obs = 0.0, 0.0, 1.0
            if sigma <= 0:
                return 1e10

        sigma_delta_sq = 2.0 * sigma * sigma
        sigma_obs_sq = sigma_obs * sigma_obs

        # Fresh ratings for this evaluation
        ratings = np.full(n_players, initial_rating, dtype=np.float64)

        ll = log_marginal_likelihood_scalar(
            player1, player2, scores, margins, day_offsets,
            ratings, sigma_delta_sq, b,
            c1, c2, sigma_obs_sq,
            use_margin, True,
        )
        if verbose:
            print(f"  sigma={sigma:.2f}", end="")
            if use_margin:
                print(f" c1={c1:.6f} c2={c2:.4f} sigma_obs={sigma_obs:.4f}", end="")
            print(f" => LL={ll:.4f} (mean={ll/len(player1):.6f})")
        return -ll

    if use_margin:
        x0 = np.array([initial_sigma, initial_c1, initial_c2, initial_sigma_obs])
        bounds = [(10.0, 300.0), (1e-7, 0.01), (0.001, 0.5), (0.01, 0.5)]
    else:
        x0 = np.array([initial_sigma])
        bounds = [(10.0, 300.0)]

    result = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)

    if use_margin:
        sigma, c1, c2, sigma_obs = result.x
        return {
            "sigma": sigma,
            "c1": c1,
            "c2": c2,
            "sigma_obs": sigma_obs,
            "log_likelihood": -result.fun,
            "mean_log_likelihood": -result.fun / len(player1),
            "converged": result.success,
        }
    else:
        sigma = result.x[0]
        return {
            "sigma": sigma,
            "log_likelihood": -result.fun,
            "mean_log_likelihood": -result.fun / len(player1),
            "converged": result.success,
        }
