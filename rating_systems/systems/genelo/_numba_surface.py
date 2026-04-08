"""
Numba-accelerated core functions for GenEloSurface (multivariate GenElo).

Implements the correlated-skills extension from Ingram (2021), Section 2.5-2.6.
Each player has a vector of skills (one per surface + optional tournament
additions). The update uses the multivariate Newton-Raphson step (Eq. 29):

    μ'_θ = μ_θ - H⁻¹(μ_θ) j(μ_θ)

The key insight is that the Hessian is a rank-1 update of -Σ⁻¹_θ, so
Sherman-Morrison gives H⁻¹ in closed form. No matrix inversion is needed:

    μ'_w = μ_w + (β / (1 + α q)) × Σ a_w
    μ'_l = μ_l - (β / (1 + α q)) × Σ a_l

where:
    β = b_eff(1 - γ) + (c₁/σ²_obs)(s - s_pred)  [combined residual]
    α = η b²_eff + c₁²/σ²_obs                     [curvature]
    q = a_w^T Σ a_w + a_l^T Σ a_l                  [quadratic form]
    Σ a = matrix-vector product (or column selection for indicators)
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, inline="always")
def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


# ---------------------------------------------------------------------------
# Surface constants
# ---------------------------------------------------------------------------
SURFACE_HARD = 0
SURFACE_CLAY = 1
SURFACE_GRASS = 2
SURFACE_INDOOR_HARD = 3
NUM_SURFACES = 4

# Tournament levels
TOURNAMENT_250_500 = 0
TOURNAMENT_MASTERS = 1
TOURNAMENT_SLAM = 2
NUM_TOURNAMENT_ADDITIONS = 2  # Masters and Slam (250/500 is baseline)


@njit(cache=True, fastmath=True, inline="always")
def _compute_sigma_a(
    cov_matrix: np.ndarray,
    surface_id: int,
    tournament_level: int,
    n_skills: int,
) -> np.ndarray:
    """
    Compute Σ a where a is the indicator vector for surface + tournament.

    a has 1.0 at the surface index, and optionally 1.0 at the tournament
    index (offset by NUM_SURFACES). For 250/500 tournaments, only the
    surface component is active.

    Returns Σ a as a vector of length n_skills.
    """
    result = np.empty(n_skills, dtype=np.float64)
    # Start with the surface column of Σ
    for k in range(n_skills):
        result[k] = cov_matrix[k, surface_id]
    # Add tournament column if Masters or Slam
    if tournament_level >= TOURNAMENT_MASTERS:
        tourn_idx = NUM_SURFACES + (tournament_level - TOURNAMENT_MASTERS)
        for k in range(n_skills):
            result[k] += cov_matrix[k, tourn_idx]
    return result


@njit(cache=True, fastmath=True, inline="always")
def _compute_q(
    cov_matrix: np.ndarray,
    surface_id: int,
    tournament_level: int,
) -> float:
    """
    Compute q = a_w^T Σ a_w + a_l^T Σ a_l = 2 a^T Σ a (when a_w = a_l).

    For indicator vector a with 1s at surface_id and optionally tournament:
        a^T Σ a = Σ[s,s] + 2*Σ[s,t] + Σ[t,t]  (if tournament active)
        a^T Σ a = Σ[s,s]                        (if no tournament addition)
    """
    q_half = cov_matrix[surface_id, surface_id]
    if tournament_level >= TOURNAMENT_MASTERS:
        tourn_idx = NUM_SURFACES + (tournament_level - TOURNAMENT_MASTERS)
        q_half += 2.0 * cov_matrix[surface_id, tourn_idx]
        q_half += cov_matrix[tourn_idx, tourn_idx]
    return 2.0 * q_half


@njit(cache=True, fastmath=True, inline="always")
def _compute_delta_mu(
    skill_ratings: np.ndarray,
    p1: int,
    p2: int,
    surface_id: int,
    tournament_level: int,
) -> float:
    """
    Compute δ_μ = a^T μ_w - a^T μ_l.

    This is the predicted skill difference on the given surface/tournament.
    """
    delta = skill_ratings[p1, surface_id] - skill_ratings[p2, surface_id]
    if tournament_level >= TOURNAMENT_MASTERS:
        tourn_idx = NUM_SURFACES + (tournament_level - TOURNAMENT_MASTERS)
        delta += skill_ratings[p1, tourn_idx] - skill_ratings[p2, tourn_idx]
    return delta


# ---------------------------------------------------------------------------
# Full update: surface + margin + format (Section 2.5-2.6)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def fit_all_days_surface(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    margins: np.ndarray,
    surfaces: np.ndarray,
    tournament_levels: np.ndarray,
    is_bo5: np.ndarray,
    day_offsets: np.ndarray,
    skill_ratings: np.ndarray,
    cov_matrix: np.ndarray,
    b: float,
    c1: float,
    c2: float,
    sigma_obs_sq: float,
    sigma_obs_bo5_sq: float,
    bo5_factor: float,
    use_margin: bool,
) -> None:
    """
    Fit GenEloSurface for ALL days in a single Numba call.

    Implements the multivariate update via Sherman-Morrison (no matrix inverse).

    Args:
        player1: Player 1 IDs (sorted by day)
        player2: Player 2 IDs (sorted by day)
        scores: Scores (1.0=P1 wins, 0.0=P2 wins)
        margins: Margin of victory (from P1's perspective). Ignored if use_margin=False.
        surfaces: Surface ID per game (0=hard, 1=clay, 2=grass, 3=indoor_hard)
        tournament_levels: Tournament level per game (0=250/500, 1=Masters, 2=Slam)
        is_bo5: Whether each game is best-of-5 (0 or 1)
        day_offsets: Start index for each day (length = num_days + 1)
        skill_ratings: Skill ratings array (num_players, n_skills) to update in-place
        cov_matrix: Prior covariance matrix Σ (n_skills, n_skills)
        b: Base Elo scale factor, log(10)/scale
        c1: Margin slope
        c2: Margin offset (winner's advantage)
        sigma_obs_sq: Margin observation variance (bo3)
        sigma_obs_bo5_sq: Margin observation variance (bo5)
        bo5_factor: Multiplier m for bo5 format (Eq. 32)
        use_margin: Whether to use margin updates
    """
    n_days = len(day_offsets) - 1
    n_skills = skill_ratings.shape[1]
    c1_sq = c1 * c1

    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]

        for i in range(start, end):
            p1 = player1[i]
            p2 = player2[i]
            score = scores[i]
            surface_id = surfaces[i]
            tourn_level = tournament_levels[i]
            bo5 = is_bo5[i]

            # Effective b: b_eff = b * (1 + I_bo5 * m) [Eq. 32]
            b_eff = b * (1.0 + bo5 * bo5_factor)
            b_eff_sq = b_eff * b_eff

            # Predicted skill difference
            delta_mu = _compute_delta_mu(
                skill_ratings, p1, p2, surface_id, tourn_level
            )

            # Win probability
            gamma = _sigmoid(b_eff * delta_mu)
            eta = gamma * (1.0 - gamma)

            # Quadratic form q = 2 a^T Σ a
            q = _compute_q(cov_matrix, surface_id, tourn_level)

            # Select observation variance based on format
            obs_var = sigma_obs_bo5_sq if bo5 else sigma_obs_sq

            # Compute α (curvature) and β (combined residual)
            alpha = b_eff_sq * eta
            beta = b_eff * (score - gamma)

            if use_margin:
                margin = margins[i]
                alpha += c1_sq / obs_var
                s_pred = c1 * delta_mu + c2 * (2.0 * score - 1.0)
                beta += (c1 / obs_var) * (margin - s_pred)

            # Update factor: β / (1 + α q)
            factor = beta / (1.0 + alpha * q)

            # Compute Σ a (same for both players since a_w = a_l)
            sigma_a = _compute_sigma_a(cov_matrix, surface_id, tourn_level, n_skills)

            # Apply update to both players
            for k in range(n_skills):
                skill_ratings[p1, k] += factor * sigma_a[k]
                skill_ratings[p2, k] -= factor * sigma_a[k]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def predict_single_surface(
    skill_ratings: np.ndarray,
    p1: int,
    p2: int,
    surface_id: int,
    tournament_level: int,
    is_bo5: int,
    cov_matrix: np.ndarray,
    b: float,
    bo5_factor: float,
) -> float:
    """
    Predict P(player1 wins) for a single matchup with surface/tournament info.

    Uses the approximate marginal likelihood:
        P(win) ≈ γ(b_eff μ_δ / α)
    where α = sqrt(1 + π q b²_eff / 8) and q = 2 a^T Σ a.
    """
    b_eff = b * (1.0 + is_bo5 * bo5_factor)
    delta_mu = _compute_delta_mu(
        skill_ratings, p1, p2, surface_id, tournament_level
    )
    q = _compute_q(cov_matrix, surface_id, tournament_level)
    alpha = np.sqrt(1.0 + np.pi * q * b_eff * b_eff / 8.0)
    return _sigmoid(b_eff * delta_mu / alpha)


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_surface_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    surfaces: np.ndarray,
    tournament_levels: np.ndarray,
    is_bo5: np.ndarray,
    skill_ratings: np.ndarray,
    cov_matrix: np.ndarray,
    b: float,
    bo5_factor: float,
) -> np.ndarray:
    """Predict win probabilities for a batch of matchups (fully parallel)."""
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)

    for i in prange(n):
        proba[i] = predict_single_surface(
            skill_ratings,
            player1[i], player2[i],
            surfaces[i], tournament_levels[i], is_bo5[i],
            cov_matrix, b, bo5_factor,
        )

    return proba


@njit(cache=True, fastmath=True, parallel=True)
def predict_proba_surface_batch_scalar(
    player1: np.ndarray,
    player2: np.ndarray,
    surface_id: int,
    tournament_level: int,
    is_bo5_val: int,
    skill_ratings: np.ndarray,
    cov_matrix: np.ndarray,
    b: float,
    bo5_factor: float,
) -> np.ndarray:
    """Predict batch with a single surface/tournament (common case)."""
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)

    b_eff = b * (1.0 + is_bo5_val * bo5_factor)
    q = _compute_q(cov_matrix, surface_id, tournament_level)
    alpha = np.sqrt(1.0 + np.pi * q * b_eff * b_eff / 8.0)
    b_over_alpha = b_eff / alpha

    for i in prange(n):
        delta_mu = _compute_delta_mu(
            skill_ratings, player1[i], player2[i],
            surface_id, tournament_level,
        )
        proba[i] = _sigmoid(b_over_alpha * delta_mu)

    return proba
