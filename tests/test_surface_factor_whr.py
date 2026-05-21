"""Tests for the Surface-Factor Whole-History Rating system."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rating_systems import GameDataset, SurfaceFactorWHR, WHR, WSurfaceFactorWHR
from rating_systems.systems.surface_factor_whr import (
    SURFACE_CLAY,
    SURFACE_GRASS,
    SURFACE_HARD,
)


def _make_dataset(
    n_players: int = 20,
    n_games: int = 600,
    n_days: int = 60,
    seed: int = 0,
) -> tuple[GameDataset, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_base = rng.normal(0, 1, n_players)
    offsets = rng.normal(0, 0.5, (n_players, 3))

    p1 = rng.integers(0, n_players, n_games)
    p2 = rng.integers(0, n_players, n_games)
    while (p1 == p2).any():
        mask = p1 == p2
        p2[mask] = rng.integers(0, n_players, mask.sum())

    surfaces = rng.integers(0, 3, n_games).astype(np.int32)
    diff = (
        true_base[p1] + offsets[p1, surfaces]
        - true_base[p2] - offsets[p2, surfaces]
    )
    win_prob = 1.0 / (1.0 + np.exp(-diff * 1.5))
    scores = (rng.random(n_games) < win_prob).astype(np.float64)

    days = np.sort(rng.integers(0, n_days, n_games)).astype(np.int64)
    df = pl.DataFrame({"Player1": p1, "Player2": p2, "Score": scores, "Day": days})
    return GameDataset(df), surfaces


def test_fit_runs_and_arrays_are_finite():
    ds, surfaces = _make_dataset()
    sys = SurfaceFactorWHR(max_iterations=20)
    sys.fit(ds, surfaces=surfaces)
    assert sys.is_fitted
    assert np.all(np.isfinite(sys._base_rating))
    assert np.all(np.isfinite(sys._offset_rating))


def test_predict_proba_is_symmetric_complement():
    ds, surfaces = _make_dataset(seed=1)
    sys = SurfaceFactorWHR(max_iterations=20)
    sys.fit(ds, surfaces=surfaces)
    for surface in (SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS):
        p_ab = sys.predict_proba(0, 1, surface)
        p_ba = sys.predict_proba(1, 0, surface)
        assert p_ab + p_ba == pytest.approx(1.0, abs=1e-9)


def test_batch_predict_matches_single_predict():
    ds, surfaces = _make_dataset(seed=2)
    sys = SurfaceFactorWHR(max_iterations=20)
    sys.fit(ds, surfaces=surfaces)
    p1 = np.array([0, 1, 2, 0])
    p2 = np.array([1, 2, 3, 3])
    surf = np.array([SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS, SURFACE_CLAY], dtype=np.int32)
    batch = sys.predict_proba(p1, p2, surf)
    singles = np.array([
        sys.predict_proba(int(a), int(b), int(s)) for a, b, s in zip(p1, p2, surf)
    ])
    np.testing.assert_allclose(batch, singles, atol=1e-12)


def test_invalid_surface_index_raises():
    ds, surfaces = _make_dataset(n_games=50, seed=3)
    bad = surfaces.copy()
    bad[0] = 5
    sys = SurfaceFactorWHR(max_iterations=5)
    with pytest.raises(ValueError):
        sys.fit(ds, surfaces=bad)


def test_negative_w_rejected():
    with pytest.raises(ValueError):
        SurfaceFactorWHR(w_surfaces=(-0.1, 1.0, 1.0))


def test_single_surface_w_one_strong_anchor_recovers_plain_whr_order():
    """When all games are on a single surface and ``w_s = 1`` with a strong
    anchor on the offset, the base trajectory carries the rating signal and
    plain WHR should produce the same *ranking* (numeric equality is not
    expected because the offset still absorbs some variance).
    """
    ds, _ = _make_dataset(n_games=400, seed=4)
    n_games = ds.num_games
    surfaces = np.zeros(n_games, dtype=np.int32)  # everything on Hard

    sys_factor = SurfaceFactorWHR(
        w_surfaces=(1.0, 1.0, 1.0),
        w2_off=(1e-6, 1e-6, 1e-6),  # offsets nearly frozen
        offset_anchor=1.0,           # strong pull toward zero
        max_iterations=50,
    )
    sys_factor.fit(ds, surfaces=surfaces)

    sys_whr = WHR(max_iterations=50)
    sys_whr.fit(ds.dataset if hasattr(ds, "dataset") else ds)

    rank_factor = np.argsort(-sys_factor._base_rating)
    rank_whr = np.argsort(-sys_whr._ratings.ratings)

    # Kendall's tau-like agreement on the top half of the ranking.
    top_n = ds.num_players // 2
    overlap = len(set(rank_factor[:top_n]) & set(rank_whr[:top_n])) / top_n
    assert overlap >= 0.7, f"Top-{top_n} overlap was only {overlap:.0%}"


def test_get_effective_rating_matches_formula():
    ds, surfaces = _make_dataset(seed=5)
    sys = SurfaceFactorWHR(
        w_surfaces=(0.7, 0.9, 1.1),
        max_iterations=15,
    )
    sys.fit(ds, surfaces=surfaces)

    for pid in range(4):
        for s in range(3):
            eff = sys.get_effective_rating(pid, s)
            expected = (
                sys._w_surfaces_arr[s]
                * (sys._base_rating[pid] - sys.config.initial_rating)
                + sys._offset_rating[pid, s]
                + sys.config.initial_rating
            )
            assert eff == pytest.approx(expected, abs=1e-9)


def test_unit_weights_match_unweighted_fit():
    """Weighted system with all-ones weights must equal the unweighted system."""
    ds, surfaces = _make_dataset(seed=10)
    n = ds.num_games
    weights = np.ones(n, dtype=np.float64)

    sys_unw = SurfaceFactorWHR(max_iterations=15)
    sys_unw.fit(ds, surfaces=surfaces)

    sys_w = WSurfaceFactorWHR(max_iterations=15)
    sys_w.fit(ds, surfaces=surfaces, weights=weights)

    np.testing.assert_allclose(
        sys_unw._base_rating, sys_w._base_rating, atol=1e-9,
    )
    np.testing.assert_allclose(
        sys_unw._offset_rating, sys_w._offset_rating, atol=1e-9,
    )


def test_weights_scale_information():
    """Doubling every game's weight should produce sharper (more confident)
    ratings: the most recent base rating for the top player should move
    further from the prior than under unit weights."""
    ds, surfaces = _make_dataset(n_games=200, seed=11)
    n = ds.num_games

    sys_unw = WSurfaceFactorWHR(max_iterations=20)
    sys_unw.fit(ds, surfaces=surfaces, weights=np.ones(n))

    sys_heavy = WSurfaceFactorWHR(max_iterations=20)
    sys_heavy.fit(ds, surfaces=surfaces, weights=np.full(n, 2.0))

    # Heavy weights should produce more extreme ratings than unit weights.
    spread_unw = sys_unw._base_rating.std()
    spread_heavy = sys_heavy._base_rating.std()
    assert spread_heavy > spread_unw, (
        f"Doubled weights should widen rating spread; "
        f"unit std={spread_unw:.2f}, heavy std={spread_heavy:.2f}"
    )


def test_weighted_zero_weights_collapse_to_prior():
    """Zero-weighted games carry no information; ratings should stay at prior."""
    ds, surfaces = _make_dataset(n_games=100, seed=12)
    n = ds.num_games
    sys = WSurfaceFactorWHR(max_iterations=20)
    sys.fit(ds, surfaces=surfaces, weights=np.zeros(n))
    # All base ratings should sit at the prior (1500), modulo the virtual
    # anchor pulling player 0 toward 0 on their first pd. The spread should
    # be very small.
    assert sys._base_rating.std() < 50.0


def test_update_refits_with_new_games():
    """After update() the most-recent ratings should move in the right direction."""
    ds, surfaces = _make_dataset(n_games=100, seed=6)
    sys = SurfaceFactorWHR(max_iterations=20, refit_max_iterations=10, refit_interval=1)
    sys.fit(ds, surfaces=surfaces)

    eff_clay_0_before = sys.get_effective_rating(0, SURFACE_CLAY)
    eff_clay_1_before = sys.get_effective_rating(1, SURFACE_CLAY)

    # Add a strong signal that player 0 dominates player 1 on Clay.
    from rating_systems import GameBatch
    new_p1 = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    new_p2 = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    new_scores = np.ones(5, dtype=np.float64)
    new_day = int(sys._stored_days.max()) + 30
    batch = GameBatch(new_p1, new_p2, new_scores, new_day)
    sys.update(batch, surfaces=np.full(5, SURFACE_CLAY, dtype=np.int32))

    eff_clay_0_after = sys.get_effective_rating(0, SURFACE_CLAY)
    eff_clay_1_after = sys.get_effective_rating(1, SURFACE_CLAY)

    # The gap between player 0 and player 1 on Clay should have moved in
    # P0's favour.
    gap_before = eff_clay_0_before - eff_clay_1_before
    gap_after = eff_clay_0_after - eff_clay_1_after
    assert gap_after > gap_before, (
        f"Clay gap (P0 - P1) should have increased; got {gap_before:.1f} -> {gap_after:.1f}"
    )
