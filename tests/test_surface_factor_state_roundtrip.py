"""Regression tests: surface-factor systems must round-trip through
save_state/load_state predicting bit-identically.

The base ``save_state`` persists only the core ``ratings``/``rd``; the
rank-1 surface factor lives in system-specific arrays
(``_offset_rating``/``_offset_rd`` for glicko, ``_base_rating``/
``_offset_rating`` for WHR). Before these systems overrode save/load, a
reloaded model lost those arrays and either raised in the njit predict
kernel (glicko) or reported "Model not fitted" (WHR). These tests lock in
the fix so the deployed prediction pipeline can restore a working model.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rating_systems import (
    GameDataset,
    WSurfaceFactorGlicko,
    WSurfaceFactorWHR,
)


def _make_dataset(n_players=24, n_games=800, n_days=80, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n_players)
    offs = rng.normal(0, 0.5, (n_players, 3))
    p1 = rng.integers(0, n_players, n_games)
    p2 = rng.integers(0, n_players, n_games)
    while (p1 == p2).any():
        m = p1 == p2
        p2[m] = rng.integers(0, n_players, m.sum())
    surf = rng.integers(0, 3, n_games).astype(np.int32)
    diff = base[p1] + offs[p1, surf] - base[p2] - offs[p2, surf]
    scores = (rng.random(n_games) < 1 / (1 + np.exp(-1.5 * diff))).astype(float)
    days = np.sort(rng.integers(0, n_days, n_games)).astype(np.int64)
    df = pl.DataFrame({"Player1": p1, "Player2": p2, "Score": scores,
                       "Day": days})
    return GameDataset.from_dataframe(df), surf


@pytest.mark.parametrize("cls", [WSurfaceFactorGlicko, WSurfaceFactorWHR])
def test_state_roundtrip_predicts_identically(cls, tmp_path):
    ds, surf = _make_dataset()
    n = 24
    sys = cls(num_players=n)
    sys.fit(ds, surfaces=surf)

    rng = np.random.default_rng(7)
    tp1 = rng.integers(0, n, 200).astype(np.int64)
    tp2 = rng.integers(0, n, 200).astype(np.int64)
    tsurf = rng.integers(0, 3, 200).astype(np.int32)
    before = np.asarray(sys.predict_proba(tp1, tp2, surfaces=tsurf))

    path = str(tmp_path / "state.npz")
    sys.save_state(path)

    # Mirror production: construct with no player count (genuinely unfitted,
    # as create_system does), then restore purely from the state file.
    fresh = cls()
    with pytest.raises(Exception):
        fresh.predict_proba(tp1, tp2, surfaces=tsurf)

    fresh.load_state(path)
    assert fresh._fitted
    after = np.asarray(fresh.predict_proba(tp1, tp2, surfaces=tsurf))

    assert np.array_equal(before, after)
    assert np.all(np.isfinite(after))
    assert after.min() >= 0.0 and after.max() <= 1.0
    assert after.std() > 0.02  # non-degenerate


@pytest.mark.parametrize("cls", [WSurfaceFactorGlicko, WSurfaceFactorWHR])
def test_state_file_carries_surface_arrays(cls, tmp_path):
    """The .npz must physically contain the surface-factor arrays."""
    ds, surf = _make_dataset()
    sys = cls(num_players=24)
    sys.fit(ds, surfaces=surf)
    path = str(tmp_path / "state.npz")
    sys.save_state(path)
    keys = set(np.load(path).keys())
    assert "offset_rating" in keys
    if cls is WSurfaceFactorGlicko:
        assert "offset_rd" in keys
    else:
        assert "base_rating" in keys
