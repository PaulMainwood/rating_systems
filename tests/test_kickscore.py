"""Tests for the KickScore rating system.

The EP site update is the delicate part: it is easy to get the winner's site
right and the loser's wrong, because the two differ only by the sign of the
coefficient (+1 for the winner, -1 for the loser) and the coefficient appears
*squared* on one term. See `_ep_site` in `_numba_core.py`.

Note on test design: the classic failure mode of this update is an error term
proportional to the cavity mean, which VANISHES when the cavity mean is zero.
A single-match test, or a balanced round-robin (whose symmetric fixed point has
every mean at 0), passes even on broken code. Every test here therefore uses
players with several matches and >= 2 EP iterations, so cavity means are
non-zero by the time the loser's site is computed.
"""

import numpy as np
import pandas as pd
import pytest

from rating_systems import GameDataset
from rating_systems.systems.kickscore_rating import KickScoreRating
from rating_systems.systems.kickscore_rating._numba_core import (
    _ep_site, kalman_forward_backward, kalman_forward_backward_ref,
)

VAR_C, VAR_M, LSCALE = 0.05, 0.15, 365.0


# ── the scalarised Kalman must equal the readable reference ───────────────
#
# kalman_forward_backward is kalman_forward_backward_ref with every 3x3
# matrix op hand-expanded into scalars, for speed. That is precisely the kind
# of code that grows a silent transcription error, so pin the two together
# across a range of realistic and awkward inputs.

def _run_both(ts, ns, xs, var_c=VAR_C, var_m=VAR_M, lscale=LSCALE):
    lam = np.sqrt(3.0) / lscale
    n = len(ts)
    out = []
    for fn in (kalman_forward_backward_ref, kalman_forward_backward):
        arrs = (np.zeros((n, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3)),
                np.zeros((n, 3, 3)), np.zeros((n, 3)), np.zeros((n, 3, 3)))
        ms, vs = np.zeros(n), np.zeros(n)
        fn(ts, ns, xs, var_c, var_m, lam, *arrs, ms, vs)
        out.append((ms, vs) + arrs)
    return out


@pytest.mark.parametrize("n,seed", [(1, 0), (2, 1), (5, 2), (40, 3), (400, 4)])
def test_scalar_kalman_matches_reference(n, seed):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.uniform(0, 3000, n))
    ns = rng.normal(0, 0.5, n)
    xs = rng.uniform(0.01, 3.0, n)
    ref, fast = _run_both(ts, ns, xs)
    for name, a, b in zip(("out_ms", "out_vs", "m_p", "P_p", "m_f", "P_f",
                           "m_s", "P_s"), ref, fast):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12), f"{name} diverged"


def test_scalar_kalman_matches_reference_on_awkward_inputs():
    """Duplicate timestamps (dt=0), huge gaps, and tiny/large site precisions."""
    cases = [
        # same-day matches -> dt = 0 -> A = I, Q = 0
        (np.array([0.0, 0.0, 0.0, 5.0]), np.array([0.3, -0.2, 0.5, 0.1]),
         np.array([0.5, 0.5, 0.5, 0.5])),
        # multi-year gaps, as a returning player produces
        (np.array([0.0, 2000.0, 4000.0, 9000.0]), np.array([0.4, -0.3, 0.2, 0.1]),
         np.array([1.0, 0.8, 1.2, 0.9])),
        # near-zero precision (uninformative site) and a very sharp one
        (np.array([0.0, 10.0, 20.0]), np.array([0.1, 0.2, -0.1]),
         np.array([1e-8, 50.0, 1e-8])),
    ]
    for ts, ns, xs in cases:
        for lscale in (365.0, 2728.0):
            ref, fast = _run_both(ts, ns, xs, var_c=1.875, var_m=5.625,
                                  lscale=lscale)
            for name, a, b in zip(("out_ms", "out_vs", "m_p", "P_p", "m_f",
                                   "P_f", "m_s", "P_s"), ref, fast):
                assert np.allclose(a, b, rtol=1e-8, atol=1e-11), (
                    f"{name} diverged at lscale={lscale}, ts={ts}"
                )


def test_scalar_kalman_keeps_covariances_symmetric():
    rng = np.random.default_rng(7)
    n = 60
    ts = np.sort(rng.uniform(0, 5000, n))
    ns = rng.normal(0, 0.5, n)
    xs = rng.uniform(0.01, 3.0, n)
    _ref, fast = _run_both(ts, ns, xs, var_c=1.875, var_m=5.625, lscale=2728.0)
    _ms, _vs, _m_p, P_p, _m_f, P_f, _m_s, P_s = fast
    for name, P in (("P_p", P_p), ("P_f", P_f), ("P_s", P_s)):
        assert np.allclose(P, np.transpose(P, (0, 2, 1)), atol=1e-12), (
            f"{name} is not symmetric"
        )
    assert np.all(P_s[:, 0, 0] > 0) and np.all(P_s[:, 1, 1] > 0)


# ── the EP site update, in isolation ──────────────────────────────────────
#
# For a GAUSSIAN likelihood the EP site is exact and closed-form, and both
# cavity terms cancel: x_site = coeff^2 / var_obs, n_site = coeff * diff /
# var_obs -- independent of the cavity. That makes it a razor-sharp unit test
# needing no dataset, no Kalman pass and no reference library.

@pytest.mark.parametrize("coeff", [1.0, -1.0])
@pytest.mark.parametrize("n_cav,x_cav", [
    (0.0, 4.0),    # zero cavity mean: passes even on a broken site — kept
    (2.3, 4.0),    # deliberately as a reminder of why it proves nothing.
    (-1.7, 0.6),
])
@pytest.mark.parametrize("diff,var_obs", [(0.8, 0.5), (-1.4, 2.0)])
def test_gaussian_site_is_exact_and_cavity_independent(coeff, n_cav, x_cav,
                                                       diff, var_obs):
    # Gaussian moment matching at this cavity.
    mean_cav = coeff * n_cav / x_cav
    var_cav = coeff * coeff / x_cav
    s = var_obs + var_cav
    dlogpart = (diff - mean_cav) / s
    d2logpart = -1.0 / s

    n, x, ok = _ep_site(coeff, dlogpart, d2logpart, n_cav, x_cav)
    assert ok
    assert x == pytest.approx(coeff * coeff / var_obs, rel=1e-12)
    assert n == pytest.approx(coeff * diff / var_obs, rel=1e-12)


def test_ep_site_matches_reference_library_formula():
    """Pin `_ep_site` to kickscore's Observation.ep_update expression."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        coeff = float(rng.choice([1.0, -1.0]))
        dlogpart = rng.uniform(-1.0, 1.0)
        d2logpart = rng.uniform(-0.5, -0.01)
        n_cav = rng.uniform(-3.0, 3.0)
        x_cav = rng.uniform(0.5, 5.0)

        denom = 1.0 + coeff * coeff * d2logpart / x_cav
        x_ref = -coeff * coeff * d2logpart / denom
        n_ref = coeff * (dlogpart - coeff * (n_cav / x_cav) * d2logpart) / denom

        n, x, ok = _ep_site(coeff, dlogpart, d2logpart, n_cav, x_cav)
        assert ok
        assert n == pytest.approx(n_ref, rel=1e-12)
        assert x == pytest.approx(x_ref, rel=1e-12)


def test_ep_site_winner_and_loser_are_mirror_images():
    """The two sites must be exact mirrors under a negated cavity mean.

    This is the site-level statement of the antisymmetry that the end-to-end
    flip test checks: the loser's site at cavity mean +m must equal the
    negation of the winner's site at cavity mean -m.
    """
    dlogpart, d2logpart, x_cav = 0.35, -0.20, 0.4
    for m_cav in (0.5, 2.0, -1.3):
        n_cav = m_cav * x_cav
        n_win, x_win, _ = _ep_site(1.0, dlogpart, d2logpart, -n_cav, x_cav)
        n_lose, x_lose, _ = _ep_site(-1.0, dlogpart, d2logpart, n_cav, x_cav)
        assert n_lose == pytest.approx(-n_win, rel=1e-12)
        assert x_lose == pytest.approx(x_win, rel=1e-12)


def synth_matches(num_players=8, num_matches=120, num_days=30, seed=1):
    """Synthetic matches with skill-driven outcomes.

    Deliberately gives every player many matches so that cavity means are
    non-zero — see the module docstring.
    """
    rng = np.random.default_rng(seed)
    true_skill = np.linspace(-1.0, 1.0, num_players)

    p1 = rng.integers(0, num_players, num_matches)
    p2 = rng.integers(0, num_players, num_matches)
    while (p1 == p2).any():
        mask = p1 == p2
        p2[mask] = rng.integers(0, num_players, mask.sum())

    win_prob = 1.0 / (1.0 + np.exp(-(true_skill[p1] - true_skill[p2]) * 2.0))
    scores = (rng.random(num_matches) < win_prob).astype(float)
    days = np.sort(rng.integers(0, num_days, num_matches)).astype(float)
    return p1, p2, scores, days


def fit_native(p1, p2, scores, days, max_iter=200, tol=1e-8):
    df = pd.DataFrame(
        {"Player1": p1, "Player2": p2, "Score": scores, "Day": days.astype(int)}
    )
    system = KickScoreRating(
        var_constant=VAR_C, var_matern=VAR_M, lscale=LSCALE,
        max_iter=max_iter, tol=tol,
    )
    system.fit(GameDataset.from_dataframe(df))
    return system


# ── metamorphic: flipping every result must mirror the posterior ──────────
#
# The prior is a zero-mean GP and the likelihood is sigma(s_winner - s_loser),
# so the model is exactly antisymmetric: p(s | flipped results) = p(-s | results).
# Flipping every result must therefore negate every posterior mean and leave
# every variance untouched. This needs no reference implementation and would
# have caught the original loser-site sign error on day one.

def test_flipping_all_results_negates_posterior_means():
    p1, p2, scores, days = synth_matches()
    fwd = fit_native(p1, p2, scores, days)
    rev = fit_native(p1, p2, 1.0 - scores, days)

    t = float(days.max() + 1)
    players = sorted(set(p1.tolist()) | set(p2.tolist()))
    assert players, "synthetic data produced no players"

    for pid in players:
        m_f, v_f = fwd._get_score(int(pid), t)
        m_r, v_r = rev._get_score(int(pid), t)
        assert m_r == pytest.approx(-m_f, abs=1e-9), (
            f"player {pid}: flipping results gave mean {m_r}, expected {-m_f}"
        )
        assert v_r == pytest.approx(v_f, abs=1e-9), (
            f"player {pid}: flipping results changed variance {v_f} -> {v_r}"
        )


def test_flipping_all_results_inverts_predictions():
    p1, p2, scores, days = synth_matches()
    fwd = fit_native(p1, p2, scores, days)
    rev = fit_native(p1, p2, 1.0 - scores, days)

    t = float(days.max() + 1)
    for a, b in ((0, 1), (2, 5), (3, 7)):
        assert fwd.predict_proba(a, b, day=t) == pytest.approx(
            1.0 - rev.predict_proba(a, b, day=t), abs=1e-9
        )


def test_synthetic_data_produces_nonzero_cavity_means():
    """Guard the guard: if this fails, the tests above are blind to the bug."""
    p1, p2, scores, days = synth_matches()
    system = fit_native(p1, p2, scores, days)
    t = float(days.max() + 1)
    means = [system._get_score(int(p), t)[0]
             for p in sorted(set(p1.tolist()) | set(p2.tolist()))]
    assert max(abs(m) for m in means) > 0.05, (
        "posterior means are all ~0; the symmetry test cannot detect a "
        "cavity-mean-proportional error on this data"
    )


def test_stronger_players_rate_higher():
    """Sanity: skill ordering should broadly survive fitting."""
    p1, p2, scores, days = synth_matches(num_players=6, num_matches=400, seed=3)
    system = fit_native(p1, p2, scores, days)
    t = float(days.max() + 1)
    m_weak, _ = system._get_score(0, t)     # true_skill = -1.0
    m_strong, _ = system._get_score(5, t)   # true_skill = +1.0
    assert m_strong > m_weak


# ── parity against the reference `kickscore` library ──────────────────────
#
# `kickscore` (Maystre) is the implementation this system reimplements natively
# for numba speed. Its Constant + Matern32 composite kernel matches our
# hardcoded state space exactly (measurement vector [1, 1, 0]; state covariance
# diag(var_c, var_m, var_m * lambda^2)), so any disagreement here is EP, not a
# kernel mismatch.
#
# Tolerance is 1e-6 rather than machine epsilon: our log-Phi constants are
# 9-significant-figure roundings of the library's exact expressions, and we add
# 1e-10 jitter to the RTS solve that the library does not. Both are ~1e-9
# relative. A real site-update error moves posteriors by O(0.1-1.0).

def _fit_library(p1, p2, scores, days):
    import kickscore as ks

    model = ks.BinaryModel(obs_type="logit")
    kern = ks.kernel.Constant(var=VAR_C) + ks.kernel.Matern32(var=VAR_M, lscale=LSCALE)
    for pid in sorted(set(p1.tolist()) | set(p2.tolist())):
        model.add_item(str(pid), kernel=kern, fitter="recursive")
    # observe() rejects out-of-order timestamps; `days` is sorted by synth_matches.
    for a, b, s, t in zip(p1, p2, scores, days):
        w, l = (a, b) if s >= 0.5 else (b, a)
        model.observe(winners=[str(w)], losers=[str(l)], t=float(t))
    converged = model.fit(method="ep", lr=1.0, tol=1e-8, max_iter=500)
    assert converged, "reference library failed to converge — test is inconclusive"
    return model


# ── likelihood selection ──────────────────────────────────────────────────

def test_obs_type_rejects_unknown_likelihood():
    """obs_type was silently ignored for months; it must now fail loudly."""
    with pytest.raises(ValueError, match="obs_type"):
        KickScoreRating(obs_type="cauchy")


def test_probit_is_wired_through_and_differs_from_logit():
    p1, p2, scores, days = synth_matches()
    df = pd.DataFrame({"Player1": p1, "Player2": p2, "Score": scores,
                       "Day": days.astype(int)})

    fitted = {}
    for obs in ("logit", "probit"):
        system = KickScoreRating(var_constant=VAR_C, var_matern=VAR_M,
                                 lscale=LSCALE, obs_type=obs, max_iter=200)
        system.fit(GameDataset.from_dataframe(df))
        fitted[obs] = system

    t = float(days.max() + 1)
    p_logit = fitted["logit"].predict_proba(0, 7, day=t)
    p_probit = fitted["probit"].predict_proba(0, 7, day=t)

    # Both sane, on the same side of the coin flip, but genuinely distinct
    # likelihoods — if probit were still silently running logit these would
    # be bit-identical.
    for p in (p_logit, p_probit):
        assert 0.0 < p < 1.0
    assert (p_logit - 0.5) * (p_probit - 0.5) > 0
    assert abs(p_logit - p_probit) > 1e-6


# ── refit_interval ────────────────────────────────────────────────────────

def _run_walk_forward(refit_interval, num_days=40):
    """Fit on the first half, then update() day by day over the second."""
    p1, p2, scores, days = synth_matches(num_players=10, num_matches=300,
                                         num_days=num_days, seed=7)
    split = num_days // 2
    train = days < split
    df = pd.DataFrame({"Player1": p1[train], "Player2": p2[train],
                       "Score": scores[train], "Day": days[train].astype(int)})
    system = KickScoreRating(var_constant=VAR_C, var_matern=VAR_M, lscale=LSCALE,
                             max_iter=100, refit_max_iterations=2,
                             refit_interval=refit_interval)
    system.fit(GameDataset.from_dataframe(df))

    from rating_systems.data import GameBatch
    for d in range(split, num_days):
        mask = days == d
        if not mask.any():
            continue
        system.update(GameBatch(
            player1=p1[mask].astype(np.int64), player2=p2[mask].astype(np.int64),
            scores=scores[mask], day=d,
        ))
    return system, float(num_days)


def test_refit_interval_tracks_daily_refits_closely():
    """A weekly refit should stay close to a daily one, not diverge."""
    daily, t = _run_walk_forward(refit_interval=1)
    weekly, _ = _run_walk_forward(refit_interval=7)
    for pid in range(10):
        m_d, _ = daily._get_score(pid, t)
        m_w, _ = weekly._get_score(pid, t)
        assert m_w == pytest.approx(m_d, abs=0.25), f"player {pid} drifted too far"


def test_refit_interval_still_advances_the_clock():
    """Evidence may lag between refits, but predictions must stay time-aware."""
    system, t = _run_walk_forward(refit_interval=7)
    assert system._current_day == 39
    # Variance must grow as we extrapolate further past the last observation:
    # the Matern form component decays toward the constant baseline.
    _, v_near = system._get_score(0, t)
    _, v_far = system._get_score(0, t + 500.0)
    assert v_far > v_near


def test_state_roundtrip_preserves_obs_type(tmp_path):
    p1, p2, scores, days = synth_matches()
    df = pd.DataFrame({"Player1": p1, "Player2": p2, "Score": scores,
                       "Day": days.astype(int)})
    system = KickScoreRating(var_constant=VAR_C, var_matern=VAR_M, lscale=LSCALE,
                             obs_type="probit", max_iter=100)
    system.fit(GameDataset.from_dataframe(df))
    t = float(days.max() + 1)
    before = system.predict_proba(0, 7, day=t)

    path = str(tmp_path / "ks.npz")
    system.save_state(path)
    restored = KickScoreRating(obs_type="logit")  # deliberately wrong; load must win
    restored.load_state(path)

    assert restored.config.obs_type == "probit"
    assert restored.predict_proba(0, 7, day=t) == pytest.approx(before, abs=1e-9)


def test_posteriors_match_reference_library():
    pytest.importorskip("kickscore")
    p1, p2, scores, days = synth_matches(num_players=12, num_matches=200, num_days=40,
                                         seed=0)
    native = fit_native(p1, p2, scores, days, max_iter=500)
    model = _fit_library(p1, p2, scores, days)

    t = float(days.max() + 1)
    for pid in sorted(set(p1.tolist()) | set(p2.tolist())):
        m_lib, v_lib = model.item[str(pid)].predict(np.array([t]))
        m_nat, v_nat = native._get_score(int(pid), t)
        assert m_nat == pytest.approx(m_lib[0], abs=1e-6), f"mean mismatch, player {pid}"
        assert v_nat == pytest.approx(v_lib[0], abs=1e-6), f"var mismatch, player {pid}"


def test_predictions_match_reference_library():
    pytest.importorskip("kickscore")
    p1, p2, scores, days = synth_matches(num_players=12, num_matches=200, num_days=40,
                                         seed=0)
    native = fit_native(p1, p2, scores, days, max_iter=500)
    model = _fit_library(p1, p2, scores, days)

    t = float(days.max() + 1)
    for a, b in ((0, 1), (3, 9), (5, 11)):
        p_lib, _ = model.probabilities(team1=[str(a)], team2=[str(b)], t=t)
        assert native.predict_proba(a, b, day=t) == pytest.approx(p_lib, abs=1e-6)
