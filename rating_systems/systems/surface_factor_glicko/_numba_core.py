"""Numba-accelerated core for Surface-Factor Weighted Glicko.

TrueSkill 2-style rank-1 factor model:

    base_i             ~ N(0, sigma_base^2)
    offset_{i, s}      ~ N(0, sigma_s^2)        for s in {0=Hard, 1=Clay, 2=Grass}
    skill_{i, s}       = w_s * base_i + offset_{i, s}

The effective per-surface rating used in the Bradley-Terry / Glicko logit is
the linear combination above. Each match updates the effective rating with a
standard weighted-Glicko step, then the change is propagated back to the
base and the on-surface offset via Kalman gains proportional to their prior
variance contributions to the effective variance.

Time decay: base RD grows by c_base^2 * days_inactive between matches; each
offset RD grows by c_offset_s^2 * days_inactive. Tighter offset drifts make
surface-specific skills more stable than overall form, in keeping with the
Halo 5 finding that cross-mode correlation is large but imperfect (r ≈ 0.6).
"""

import math

import numpy as np
from numba import njit

from ..glicko._numba_core import Q, Q_SQUARED, _g, _expected_score


N_SURFACES = 3


@njit(cache=True, fastmath=True)
def _effective_rating(base_rating, offset_rating, w_s):
    """Effective surface-conditioned rating: R_eff = w_s * R_base + R_offset_s."""
    return w_s * base_rating + offset_rating


@njit(cache=True, fastmath=True)
def _effective_rd_sq(base_rd, offset_rd, w_s):
    """Effective surface RD^2 = w_s^2 * RD_base^2 + RD_offset^2 (independent prior)."""
    return (w_s * base_rd) ** 2 + offset_rd * offset_rd


@njit(cache=True, fastmath=True)
def _grow_rd_for_inactivity(
    base_rd,
    offset_rd,
    last_played,
    active_players,
    current_day,
    c_base,
    c_offset,
    min_rd,
    max_rd,
):
    """Grow base and per-surface offset RDs by their respective drift rates."""
    c_base_sq = c_base * c_base
    n_offset_surfaces = offset_rd.shape[1]

    for k in range(active_players.shape[0]):
        p = active_players[k]
        days = current_day - last_played[p]
        if days <= 0:
            continue
        b_rd = math.sqrt(base_rd[p] * base_rd[p] + c_base_sq * days)
        base_rd[p] = min(max(b_rd, min_rd), max_rd)
        for s in range(n_offset_surfaces):
            c_s = c_offset[s]
            o_rd = math.sqrt(offset_rd[p, s] * offset_rd[p, s] + c_s * c_s * days)
            offset_rd[p, s] = min(max(o_rd, min_rd), max_rd)


@njit(cache=True, fastmath=True)
def predict_single_factor(
    base_rating_i, base_rd_i, offset_rating_i, offset_rd_i,
    base_rating_j, base_rd_j, offset_rating_j, offset_rd_j,
    w_s,
):
    """Predict P(i beats j) on surface s, accounting for both players' uncertainty."""
    r_i = _effective_rating(base_rating_i, offset_rating_i, w_s)
    r_j = _effective_rating(base_rating_j, offset_rating_j, w_s)
    rd_i_sq = _effective_rd_sq(base_rd_i, offset_rd_i, w_s)
    rd_j_sq = _effective_rd_sq(base_rd_j, offset_rd_j, w_s)
    rd_combined = math.sqrt(rd_i_sq + rd_j_sq)
    g = _g(rd_combined)
    return _expected_score(r_i, r_j, g)


@njit(cache=True, fastmath=True)
def predict_proba_batch_factor(
    p1, p2, surfaces, w_surfaces,
    base_rating, base_rd, offset_rating, offset_rd,
):
    """Batched P(p1 beats p2) on each game's surface."""
    n = p1.shape[0]
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        s = surfaces[k]
        w_s = w_surfaces[s]
        i = p1[k]
        j = p2[k]
        out[k] = predict_single_factor(
            base_rating[i], base_rd[i], offset_rating[i, s], offset_rd[i, s],
            base_rating[j], base_rd[j], offset_rating[j, s], offset_rd[j, s],
            w_s,
        )
    return out


@njit(cache=True, fastmath=True)
def update_ratings_batch_factor(
    player1,
    player2,
    scores,
    weights,
    surfaces,
    w_surfaces,
    base_rating,
    base_rd,
    offset_rating,
    offset_rd,
    last_played,
    current_day,
    c_base,
    c_offset,
    min_rd,
    max_rd,
):
    """One rating period: grow inactivity, compute per-player updates, distribute.

    All games are treated as simultaneous (pre-period ratings used). For each
    active player, accumulate weighted Glicko information across their games,
    then map the effective-rating posterior back onto base and the surfaces
    actually played via prior-variance-weighted Kalman gains.

    Returns the number of distinct players updated.
    """
    n_games = player1.shape[0]
    if n_games == 0:
        return 0

    # Find unique active players.
    players_set = set()
    for k in range(n_games):
        players_set.add(player1[k])
        players_set.add(player2[k])
    active_players = np.array(list(players_set), dtype=np.int64)
    n_players = active_players.shape[0]

    _grow_rd_for_inactivity(
        base_rd, offset_rd, last_played, active_players, current_day,
        c_base, c_offset, min_rd, max_rd,
    )

    # Snapshot for simultaneous updates within the rating period.
    pre_base_rating = base_rating.copy()
    pre_base_rd = base_rd.copy()
    pre_offset_rating = offset_rating.copy()
    pre_offset_rd = offset_rd.copy()

    for idx in range(n_players):
        p = active_players[idx]
        b_rating = pre_base_rating[p]
        b_rd = pre_base_rd[p]
        b_rd_sq = b_rd * b_rd

        # Per-surface accumulators. Each surface is updated independently below
        # because the effective rating depends on the offset for *that* surface.
        sum_g_sq_e = np.zeros(N_SURFACES, dtype=np.float64)
        sum_g_diff = np.zeros(N_SURFACES, dtype=np.float64)
        games_per_surface = np.zeros(N_SURFACES, dtype=np.int64)

        for k in range(n_games):
            if player1[k] == p:
                opp = player2[k]
                score = scores[k]
            elif player2[k] == p:
                opp = player1[k]
                score = 1.0 - scores[k]
            else:
                continue

            w_j = weights[k]
            s = surfaces[k]
            w_s = w_surfaces[s]

            r_p = _effective_rating(b_rating, pre_offset_rating[p, s], w_s)
            rd_p_sq = _effective_rd_sq(b_rd, pre_offset_rd[p, s], w_s)

            r_o = _effective_rating(pre_base_rating[opp], pre_offset_rating[opp, s], w_s)
            rd_o_sq = _effective_rd_sq(pre_base_rd[opp], pre_offset_rd[opp, s], w_s)

            # Use the opponent's effective RD in the standard Glicko g(.) factor;
            # this matches the original Glicko derivation (player's own variance
            # appears only via the posterior update below).
            g = _g(math.sqrt(rd_o_sq))
            e = _expected_score(r_p, r_o, g)

            sum_g_sq_e[s] += w_j * g * g * e * (1.0 - e)
            sum_g_diff[s] += w_j * g * (score - e)
            games_per_surface[s] += 1

            # Pull r_p, rd_p back into the accumulators only via the standard
            # Glicko mechanism — they are not used directly here.

        # Apply each surface's worth of evidence in turn. Updating surfaces
        # sequentially within one period is approximate (information from
        # surface A would in principle move base, which would shift the prior
        # for surface B). To stay close to a one-pass Glicko-style update we
        # accept that approximation; in practice players rarely play multiple
        # surfaces on the same day, so the approximation has near-zero impact.
        for s in range(N_SURFACES):
            if games_per_surface[s] == 0:
                continue

            w_s = w_surfaces[s]
            o_rd_sq = pre_offset_rd[p, s] * pre_offset_rd[p, s]
            rd_eff_prior_sq = (w_s * w_s) * b_rd_sq + o_rd_sq

            info_likelihood = Q_SQUARED * sum_g_sq_e[s]
            if info_likelihood <= 1e-10:
                continue
            d_sq = 1.0 / info_likelihood

            # Residual at the effective-rating level. In Glicko's Gaussian
            # approximation, the likelihood is centred at
            #     R_obs - R_eff_prior = (Q * sum_g_diff) / (Q^2 * sum_g_sq_e)
            #                         = Q * sum_g_diff * d^2,
            # with observation variance d^2. This is the *innovation* the
            # Kalman update should propagate; with w_s=1 and zero offset
            # variance the resulting update reproduces standard WGlicko.
            residual = Q * sum_g_diff[s] * d_sq

            innovation_var = rd_eff_prior_sq + d_sq
            k_base = (w_s * b_rd_sq) / innovation_var
            k_offset = o_rd_sq / innovation_var

            new_base_rating = b_rating + k_base * residual
            new_offset_rating = pre_offset_rating[p, s] + k_offset * residual

            # Diagonal-approximated variance shrinkage. Joseph form would be
            # more numerically stable in extreme regimes, but standard form
            # is sufficient with the [min_rd, max_rd] clamps below.
            new_base_rd_sq = b_rd_sq * (1.0 - k_base * w_s)
            new_offset_rd_sq = o_rd_sq * (1.0 - k_offset)

            new_base_rd = math.sqrt(max(new_base_rd_sq, min_rd * min_rd))
            new_offset_rd = math.sqrt(max(new_offset_rd_sq, min_rd * min_rd))

            new_base_rd = min(max(new_base_rd, min_rd), max_rd)
            new_offset_rd = min(max(new_offset_rd, min_rd), max_rd)

            base_rating[p] = new_base_rating
            base_rd[p] = new_base_rd
            offset_rating[p, s] = new_offset_rating
            offset_rd[p, s] = new_offset_rd

            # Compound base updates within the period: subsequent surfaces for
            # the same player should see the freshly-updated base value.
            b_rating = new_base_rating
            b_rd = new_base_rd
            b_rd_sq = b_rd * b_rd

        last_played[p] = current_day

    return n_players


@njit(cache=True, fastmath=True)
def fit_all_days_factor(
    player1,
    player2,
    scores,
    weights,
    surfaces,
    w_surfaces,
    day_indices,
    day_offsets,
    base_rating,
    base_rd,
    offset_rating,
    offset_rd,
    last_played,
    c_base,
    c_offset,
    min_rd,
    max_rd,
):
    """Run the per-period update routine across every day in the dataset."""
    n_days = day_offsets.shape[0] - 1
    total = 0
    for day_idx in range(n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]
        if end <= start:
            continue
        cur_day = day_indices[day_idx]
        total += update_ratings_batch_factor(
            player1[start:end],
            player2[start:end],
            scores[start:end],
            weights[start:end],
            surfaces[start:end],
            w_surfaces,
            base_rating,
            base_rd,
            offset_rating,
            offset_rd,
            last_played,
            cur_day,
            c_base,
            c_offset,
            min_rd,
            max_rd,
        )
    return total


@njit(cache=True, fastmath=True)
def walk_forward_predict_update_factor(
    player1,
    player2,
    scores,
    surfaces,
    w_surfaces,
    day_indices,
    day_offsets,
    weights,
    base_rating,
    base_rd,
    offset_rating,
    offset_rd,
    last_played,
    c_base,
    c_offset,
    min_rd,
    max_rd,
    n_train_days,
):
    """Fused walk-forward: train on first n_train_days, then per-day predict-then-update.

    Returns a per-game prediction array; entries for the training period are NaN.
    """
    n_games = player1.shape[0]
    preds = np.full(n_games, np.nan, dtype=np.float64)
    n_days = day_offsets.shape[0] - 1

    # Training phase.
    train_days = min(n_train_days, n_days)
    for day_idx in range(train_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]
        if end <= start:
            continue
        cur_day = day_indices[day_idx]
        update_ratings_batch_factor(
            player1[start:end],
            player2[start:end],
            scores[start:end],
            weights[start:end],
            surfaces[start:end],
            w_surfaces,
            base_rating,
            base_rd,
            offset_rating,
            offset_rd,
            last_played,
            cur_day,
            c_base,
            c_offset,
            min_rd,
            max_rd,
        )

    # Test phase: predict, then update.
    for day_idx in range(train_days, n_days):
        start = day_offsets[day_idx]
        end = day_offsets[day_idx + 1]
        if end <= start:
            continue
        cur_day = day_indices[day_idx]

        # Grow RDs forward without applying observations, just for prediction.
        # Predictions use the *pre-period* ratings but with RD already inflated.
        # To match wglicko's behaviour we'll inflate, predict, then call update
        # which re-inflates only the not-yet-played players (no-op here).
        # Active players on this day:
        active_set = set()
        for k in range(start, end):
            active_set.add(player1[k])
            active_set.add(player2[k])
        active_players = np.array(list(active_set), dtype=np.int64)
        _grow_rd_for_inactivity(
            base_rd, offset_rd, last_played, active_players, cur_day,
            c_base, c_offset, min_rd, max_rd,
        )

        # Predict.
        for k in range(start, end):
            i = player1[k]
            j = player2[k]
            s = surfaces[k]
            preds[k] = predict_single_factor(
                base_rating[i], base_rd[i], offset_rating[i, s], offset_rd[i, s],
                base_rating[j], base_rd[j], offset_rating[j, s], offset_rd[j, s],
                w_surfaces[s],
            )

        # Mark these players as having "played today" so the update routine
        # doesn't double-inflate their RDs.
        for idx in range(active_players.shape[0]):
            last_played[active_players[idx]] = cur_day

        update_ratings_batch_factor(
            player1[start:end],
            player2[start:end],
            scores[start:end],
            weights[start:end],
            surfaces[start:end],
            w_surfaces,
            base_rating,
            base_rd,
            offset_rating,
            offset_rd,
            last_played,
            cur_day,
            c_base,
            c_offset,
            min_rd,
            max_rd,
        )

    return preds
