"""The three serve-and-return systems answer the same call.

tennis_ratings drives the whole family through one function, which passes the
surface of each match. ``ServeReturnElo`` has no surface term and used to
reject the argument, so it could not be walked forward at all.
"""

import numpy as np
import polars as pl
import pytest

from rating_systems.data import GameDataset
from rating_systems.systems import ServeReturnElo, ServeReturnGlicko, ServeReturnGlickoPoints


def _fitted(cls):
    rng = np.random.default_rng(3)
    n = 400
    p1 = rng.integers(0, 12, n)
    p2 = (p1 + rng.integers(1, 12, n)) % 12
    total = rng.integers(50, 90, (2, n)).astype(float)
    stats = {
        "p1_serve_won": np.floor(total[0] * rng.uniform(0.5, 0.75, n)),
        "p1_serve_total": total[0],
        "p2_serve_won": np.floor(total[1] * rng.uniform(0.5, 0.75, n)),
        "p2_serve_total": total[1],
        "surface_ids": rng.integers(0, 4, n).astype(np.int32),
    }
    games = pl.DataFrame({"Player1": p1, "Player2": p2,
                          "Score": rng.integers(0, 2, n).astype(float),
                          "Day": np.sort(rng.integers(0, 60, n))})
    return cls().fit(GameDataset.from_dataframe(games), serve_stats=stats)


@pytest.mark.parametrize("cls", [ServeReturnElo, ServeReturnGlicko, ServeReturnGlickoPoints])
def test_each_takes_the_surface_and_gives_probabilities(cls):
    system = _fitted(cls)
    a, b = np.array([0, 1, 2]), np.array([3, 4, 5])
    p = system.predict_proba(a, b, best_of=np.array([3, 5, 3]), day=61,
                             surfaces=np.array([0, 1, 2], dtype=np.int32))
    assert p.shape == (3,) and np.all((p > 0) & (p < 1))
    q = system.predict_proba(b, a, best_of=np.array([3, 5, 3]), day=61,
                             surfaces=np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_allclose(p + q, 1.0, atol=1e-9)
