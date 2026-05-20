"""Tests for the Surface-Factor Weighted Glicko system."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from rating_systems import GameDataset, WGlicko, WSurfaceFactorGlicko
from rating_systems.systems.surface_factor_glicko import (
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
    """Generate a synthetic dataset with rotating surfaces and skill-driven outcomes."""
    rng = np.random.default_rng(seed)
    true_base = rng.normal(0, 1, n_players)
    # Per-surface offsets so each surface has a different ranking.
    offsets = rng.normal(0, 0.5, (n_players, 3))

    p1 = rng.integers(0, n_players, n_games)
    p2 = rng.integers(0, n_players, n_games)
    while (p1 == p2).any():
        mask = p1 == p2
        p2[mask] = rng.integers(0, n_players, mask.sum())

    surfaces = rng.integers(0, 3, n_games).astype(np.int32)
    effective_diff = (
        true_base[p1] + offsets[p1, surfaces]
        - true_base[p2] - offsets[p2, surfaces]
    )
    win_prob = 1.0 / (1.0 + np.exp(-effective_diff * 1.5))
    scores = (rng.random(n_games) < win_prob).astype(np.float64)

    days = np.sort(rng.integers(0, n_days, n_games)).astype(np.int64)

    df = pl.DataFrame({
        "Player1": p1,
        "Player2": p2,
        "Score": scores,
        "Day": days,
    })
    return GameDataset(df), surfaces


def test_fit_runs_and_arrays_are_finite():
    ds, surfaces = _make_dataset()
    sys = WSurfaceFactorGlicko()
    sys.fit(ds, surfaces=surfaces)
    assert sys.is_fitted
    assert np.all(np.isfinite(sys._ratings.ratings))
    assert np.all(np.isfinite(sys._ratings.rd))
    assert np.all(np.isfinite(sys._offset_rating))
    assert np.all(np.isfinite(sys._offset_rd))


def test_uniform_offsets_zero_when_no_data():
    """With no matches the offsets should never have moved from their prior."""
    df = pl.DataFrame({"Player1": [0], "Player2": [1], "Score": [1.0], "Day": [0]})
    ds = GameDataset(df)
    sys = WSurfaceFactorGlicko(initial_offset_rating=0.0)
    sys.fit(ds, surfaces=np.array([0], dtype=np.int32))
    # Player 2 plays only on Hard so Clay/Grass offsets stay at prior.
    assert sys._offset_rating[0, SURFACE_CLAY] == pytest.approx(0.0)
    assert sys._offset_rating[0, SURFACE_GRASS] == pytest.approx(0.0)
    assert sys._offset_rating[1, SURFACE_CLAY] == pytest.approx(0.0)
    assert sys._offset_rating[1, SURFACE_GRASS] == pytest.approx(0.0)
    # The hard-court offset should have moved.
    assert sys._offset_rating[0, SURFACE_HARD] != 0.0


def test_w_zero_isolates_surfaces():
    """w_s = 0 makes each surface essentially independent (info doesn't reach base)."""
    ds, surfaces = _make_dataset(n_games=200, seed=1)
    sys = WSurfaceFactorGlicko(
        w_surfaces=(0.0, 0.0, 0.0),
        initial_base_rd=0.01,  # near-zero base uncertainty => no base updates
    )
    sys.fit(ds, surfaces=surfaces)
    # Base should be untouched because k_base = w_s * RD_base^2 / denom ≈ 0.
    base_change = sys._ratings.ratings - sys.config.initial_base_rating
    assert np.allclose(base_change, 0.0, atol=1e-6)


def test_single_surface_w_one_zero_offset_reduces_to_wglicko():
    """On a single-surface dataset with w_s = 1 and near-zero offset variance,
    the factor system must reduce numerically to plain weighted Glicko on the
    base rating.

    (Multi-surface periods can't satisfy this exactly because the factor system
    processes each surface in turn within a rating period, whereas WGlicko
    aggregates all of a player's games in one shot.)
    """
    ds, _ = _make_dataset(n_games=300, seed=2)
    n_games = ds.num_games
    surfaces = np.zeros(n_games, dtype=np.int32)  # everything on Hard

    sys_factor = WSurfaceFactorGlicko(
        w_surfaces=(1.0, 1.0, 1.0),
        initial_offset_rd=1e-9,
        c_offset=(0.0, 0.0, 0.0),
        c_base=0.0,
        initial_base_rd=350.0,
        min_rd=1e-12,   # bypass the global RD floor so the offset stays tiny
        max_rd=1000.0,
    )
    sys_factor.fit(ds, surfaces=surfaces)

    sys_wglicko = WGlicko(initial_rd=350.0, c=0.0, min_rd=1e-12, max_rd=1000.0)
    sys_wglicko.fit(ds)

    diff = sys_factor._ratings.ratings - sys_wglicko._ratings.ratings
    np.testing.assert_allclose(diff, 0.0, atol=1e-9)


def test_predict_proba_is_symmetric_complement():
    ds, surfaces = _make_dataset(seed=3)
    sys = WSurfaceFactorGlicko()
    sys.fit(ds, surfaces=surfaces)
    for surface in (SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS):
        p_ab = sys.predict_proba(0, 1, surface)
        p_ba = sys.predict_proba(1, 0, surface)
        assert p_ab + p_ba == pytest.approx(1.0, abs=1e-9)


def test_batch_predict_matches_single_predict():
    ds, surfaces = _make_dataset(seed=4)
    sys = WSurfaceFactorGlicko()
    sys.fit(ds, surfaces=surfaces)

    p1 = np.array([0, 1, 2, 0])
    p2 = np.array([1, 2, 3, 3])
    surf = np.array([SURFACE_HARD, SURFACE_CLAY, SURFACE_GRASS, SURFACE_CLAY], dtype=np.int32)
    batch = sys.predict_proba(p1, p2, surf)
    singles = np.array([
        sys.predict_proba(int(a), int(b), int(s)) for a, b, s in zip(p1, p2, surf)
    ])
    np.testing.assert_allclose(batch, singles, atol=1e-12)


def test_get_rating_effective_matches_explicit_formula():
    ds, surfaces = _make_dataset(seed=5)
    sys = WSurfaceFactorGlicko(w_surfaces=(0.7, 0.9, 1.1))
    sys.fit(ds, surfaces=surfaces)

    pid = 0
    w_arr = sys._w_surfaces_arr
    for s in range(3):
        eff_r, eff_rd = sys.get_rating(pid, surface=s)
        expected_r = w_arr[s] * sys._ratings.ratings[pid] + sys._offset_rating[pid, s]
        expected_rd = math.sqrt(
            (w_arr[s] * sys._ratings.rd[pid]) ** 2 + sys._offset_rd[pid, s] ** 2
        )
        assert eff_r == pytest.approx(expected_r, abs=1e-9)
        assert eff_rd == pytest.approx(expected_rd, abs=1e-9)


def test_invalid_surface_index_raises():
    ds, surfaces = _make_dataset(n_games=50, seed=6)
    bad = surfaces.copy()
    bad[0] = 5
    sys = WSurfaceFactorGlicko()
    with pytest.raises(ValueError):
        sys.fit(ds, surfaces=bad)


def test_negative_w_rejected():
    with pytest.raises(ValueError):
        WSurfaceFactorGlicko(w_surfaces=(-0.1, 1.0, 1.0))


def test_time_decay_grows_offset_rd_with_inactivity():
    """Pre/post RD: after a long gap a player's RDs should have widened."""
    df = pl.DataFrame({
        "Player1": [0, 0],
        "Player2": [1, 1],
        "Score": [1.0, 1.0],
        "Day": [1, 100],  # 99 days apart
    })
    ds = GameDataset(df)
    sys = WSurfaceFactorGlicko(c_base=5.0, c_offset=(2.0, 2.0, 2.0))
    sys.fit(ds, surfaces=np.array([0, 0], dtype=np.int32))
    # Base RD after the second match should reflect the inactivity inflation
    # before the new game shrinks it again.
    assert sys._ratings.rd[0] > 0  # sanity
    assert sys._ratings.last_played[0] == 100


def test_fused_walk_forward_returns_predictions_for_test_period_only():
    ds, surfaces = _make_dataset(n_games=400, n_days=80, seed=7)
    sys = WSurfaceFactorGlicko()
    sys._ensure_capacity(ds.num_players)
    p1, p2, scores, day_indices, day_offsets = ds.get_batched_arrays()
    # Build a per-game day array to align surfaces, then reorder surfaces too.
    days_per_game = np.empty(len(p1), dtype=np.int64)
    for d in range(len(day_indices)):
        days_per_game[day_offsets[d]:day_offsets[d + 1]] = day_indices[d]
    # GameDataset already sorts by day, so surfaces is already aligned.
    weights = np.ones(len(p1), dtype=np.float64)
    n_train_days = len(day_indices) // 2
    preds = sys.fused_walk_forward(
        p1, p2, scores, surfaces, day_indices, day_offsets, weights, n_train_days,
    )
    train_cutoff = day_indices[n_train_days - 1]
    train_mask = days_per_game <= train_cutoff
    assert np.all(np.isnan(preds[train_mask]))
    assert np.all((preds[~train_mask] >= 0.0) & (preds[~train_mask] <= 1.0))
    # Mark the system as fitted so subsequent predict_proba calls work.
    sys._fitted = True
