"""Whole-history systems must give the same answer on every run.

WeightedWHR and WHR update players in parallel. Until November 2026 the
parallel active-set update changed the active set inside the parallel loop,
which raced: two identical runs produced ratings that differed by up to about
1e-3 in predicted probability. These tests run the same fit and the same
sequence of daily updates twice and require identical ratings.

A race does not show on every run, so each test repeats the comparison several
times on a problem large enough to occupy several threads.
"""

import numpy as np
import polars as pl
import pytest

from rating_systems import GameDataset, WHR, WeightedWHR
from rating_systems.data import GameBatch

N_PLAYERS = 400
N_DAYS = 120
GAMES_PER_DAY = 60
TRAIN_DAYS = 60
REPEATS = 4


def _games(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    strength = rng.normal(0.0, 1.0, N_PLAYERS)
    rows = []
    for day in range(N_DAYS):
        a = rng.integers(0, N_PLAYERS, GAMES_PER_DAY)
        b = (a + rng.integers(1, N_PLAYERS, GAMES_PER_DAY)) % N_PLAYERS
        p = 1.0 / (1.0 + np.exp(-(strength[a] - strength[b])))
        score = (rng.random(GAMES_PER_DAY) < p).astype(float)
        for i in range(GAMES_PER_DAY):
            rows.append((int(a[i]), int(b[i]), float(score[i]), day))
    return pl.DataFrame(rows, schema=["Player1", "Player2", "Score", "Day"], orient="row")


def _run_weighted(df: pl.DataFrame) -> np.ndarray:
    system = WeightedWHR(num_players=N_PLAYERS)
    train = df.filter(pl.col("Day") < TRAIN_DAYS)
    system.fit(GameDataset.from_dataframe(train),
               weights=np.ones(len(train), dtype=np.float64))
    for day in range(TRAIN_DAYS, N_DAYS):
        d = df.filter(pl.col("Day") == day)
        batch = GameBatch(player1=d["Player1"].to_numpy().astype(np.int64),
                          player2=d["Player2"].to_numpy().astype(np.int64),
                          scores=d["Score"].to_numpy().astype(np.float64), day=day)
        system.update_weighted(batch, np.ones(len(d), dtype=np.float64))
    return system._ratings.ratings.copy()


def _run_base(df: pl.DataFrame) -> np.ndarray:
    system = WHR(num_players=N_PLAYERS)
    train = df.filter(pl.col("Day") < TRAIN_DAYS)
    system.fit(GameDataset.from_dataframe(train))
    for day in range(TRAIN_DAYS, N_DAYS):
        d = df.filter(pl.col("Day") == day)
        system.update(GameBatch(
            player1=d["Player1"].to_numpy().astype(np.int64),
            player2=d["Player2"].to_numpy().astype(np.int64),
            scores=d["Score"].to_numpy().astype(np.float64), day=day))
    return system._ratings.ratings.copy()


@pytest.mark.parametrize("run", [_run_weighted, _run_base],
                         ids=["WeightedWHR", "WHR"])
def test_identical_runs_give_identical_ratings(run):
    df = _games()
    first = run(df)
    assert np.isfinite(first).all()
    for _ in range(REPEATS):
        np.testing.assert_array_equal(run(df), first)
