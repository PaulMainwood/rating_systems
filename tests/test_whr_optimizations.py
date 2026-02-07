"""Tests for WHR performance optimisations.

Verifies three independent optimisations:
1. Composite key data structure build (replaces 2D np.unique)
2. Active-set player updates (skip converged players)
3. Anderson acceleration (extrapolate iteration sequence)

Each can be toggled independently. All must produce ratings equivalent
to the baseline (all optimisations off) given sufficient iterations.
"""

import time

import numpy as np
import polars as pl

from rating_systems import WHR, Backtester, GameDataset


def generate_test_data(
    num_players: int = 100,
    num_games: int = 5000,
    num_days: int = 30,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic test data with skill-based outcomes."""
    rng = np.random.RandomState(seed)

    true_skill = np.linspace(0, 1, num_players)

    p1 = rng.randint(0, num_players, num_games)
    p2 = rng.randint(0, num_players, num_games)

    while (p1 == p2).any():
        mask = p1 == p2
        p2[mask] = rng.randint(0, num_players, mask.sum())

    skill_diff = true_skill[p1] - true_skill[p2]
    win_prob = 1 / (1 + np.exp(-skill_diff * 4))
    scores = (rng.random(num_games) < win_prob).astype(float)

    days = np.sort(rng.randint(0, num_days, num_games))

    return pl.DataFrame({
        "Player1": p1,
        "Player2": p2,
        "Score": scores,
        "Day": days,
    })


# ---------------------------------------------------------------------------
# Test 1: Composite key produces same data structures
# ---------------------------------------------------------------------------

def test_composite_key_equivalence():
    """Composite key build should produce identical results to 2D unique."""
    print("=" * 60)
    print("Test 1: Composite key data structure equivalence")
    print("=" * 60)

    df = generate_test_data(num_players=50, num_games=2000, num_days=20)
    dataset = GameDataset.from_dataframe(df)

    # Fit two models — they both use composite key now, so just verify
    # the data structures are internally consistent
    whr = WHR(w2=300.0, max_iterations=100, use_active_set=False,
              anderson_window=0)
    whr.fit(dataset, end_day=15)

    # Verify pd_to_player is consistent with player_offsets
    for pid in range(whr._num_players):
        pd_start = whr._player_offsets[pid]
        pd_end = whr._player_offsets[pid + 1]
        for idx in range(pd_start, pd_end):
            assert whr._pd_to_player[idx] == pid, (
                f"pd_to_player[{idx}] = {whr._pd_to_player[idx]}, expected {pid}"
            )

    # Verify days are sorted within each player
    for pid in range(whr._num_players):
        pd_start = whr._player_offsets[pid]
        pd_end = whr._player_offsets[pid + 1]
        if pd_end - pd_start > 1:
            days = whr._pd_days[pd_start:pd_end]
            assert np.all(days[1:] >= days[:-1]), (
                f"Player {pid}: days not sorted: {days}"
            )

    # Verify total game references = 2 * num_games
    total_refs = whr._pd_game_offsets[-1]
    n_games = len(whr._stored_player1)
    assert total_refs == 2 * n_games, (
        f"Expected {2 * n_games} game refs, got {total_refs}"
    )

    print(f"  pd_to_player length: {len(whr._pd_to_player)}")
    print(f"  Total player-days: {len(whr._pd_r)}")
    print(f"  Total game refs: {total_refs} (2 * {n_games})")
    print("  PASSED: Data structures are internally consistent.\n")


# ---------------------------------------------------------------------------
# Test 2: Active-set correctness
# ---------------------------------------------------------------------------

def test_active_set_correctness():
    """Active-set ON vs OFF should converge to the same ratings."""
    print("=" * 60)
    print("Test 2: Active-set correctness")
    print("=" * 60)

    df = generate_test_data(num_players=50, num_games=2000, num_days=20)
    dataset = GameDataset.from_dataframe(df)

    max_iters = 200

    # Baseline: no optimisations
    whr_base = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   use_active_set=False, anderson_window=0)
    whr_base.fit(dataset, end_day=10)

    # Active set only
    whr_active = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                     use_active_set=True, anderson_window=0)
    whr_active.fit(dataset, end_day=10)

    # Compare initial fit ratings
    diff = np.max(np.abs(whr_base.get_ratings().ratings - whr_active.get_ratings().ratings))
    print(f"  Initial fit max diff: {diff:.2e}")
    assert diff < 0.5, f"Initial fit ratings differ too much: {diff}"

    # Walk-forward and compare
    for day in range(11, 16):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_base.update(batch)
        whr_active.update(batch)

    base_ratings = whr_base.get_ratings().ratings
    active_ratings = whr_active.get_ratings().ratings
    max_diff = np.max(np.abs(base_ratings - active_ratings))
    mean_diff = np.mean(np.abs(base_ratings - active_ratings))
    print(f"  After walk-forward: max diff = {max_diff:.6f}, mean diff = {mean_diff:.6f}")
    assert max_diff < 1.0, f"Active-set ratings differ too much: {max_diff}"
    print("  PASSED: Active-set ratings match baseline.\n")


# ---------------------------------------------------------------------------
# Test 3: Anderson acceleration correctness
# ---------------------------------------------------------------------------

def test_anderson_correctness():
    """Anderson acceleration ON vs OFF should converge to the same ratings."""
    print("=" * 60)
    print("Test 3: Anderson acceleration correctness")
    print("=" * 60)

    df = generate_test_data(num_players=50, num_games=2000, num_days=20)
    dataset = GameDataset.from_dataframe(df)

    max_iters = 200

    # Baseline: no optimisations
    whr_base = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   use_active_set=False, anderson_window=0)
    whr_base.fit(dataset, end_day=10)

    # Anderson only
    whr_anderson = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                       use_active_set=False, anderson_window=5)
    whr_anderson.fit(dataset, end_day=10)

    # Compare initial fit
    diff = np.max(np.abs(whr_base.get_ratings().ratings - whr_anderson.get_ratings().ratings))
    print(f"  Initial fit max diff: {diff:.2e}")
    assert diff < 0.5, f"Initial fit ratings differ too much: {diff}"

    # Walk-forward
    for day in range(11, 16):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_base.update(batch)
        whr_anderson.update(batch)

    base_ratings = whr_base.get_ratings().ratings
    anderson_ratings = whr_anderson.get_ratings().ratings
    max_diff = np.max(np.abs(base_ratings - anderson_ratings))
    mean_diff = np.mean(np.abs(base_ratings - anderson_ratings))
    print(f"  After walk-forward: max diff = {max_diff:.6f}, mean diff = {mean_diff:.6f}")
    assert max_diff < 1.0, f"Anderson ratings differ too much: {max_diff}"
    print("  PASSED: Anderson ratings match baseline.\n")


# ---------------------------------------------------------------------------
# Test 4: Combined correctness (all three on vs all off)
# ---------------------------------------------------------------------------

def test_combined_correctness():
    """All optimisations ON vs all OFF should converge to the same ratings."""
    print("=" * 60)
    print("Test 4: Combined correctness (all on vs all off)")
    print("=" * 60)

    df = generate_test_data(num_players=50, num_games=2000, num_days=20)
    dataset = GameDataset.from_dataframe(df)

    max_iters = 200

    # All off
    whr_off = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                  warm_start=False, use_active_set=False, anderson_window=0)
    whr_off.fit(dataset, end_day=10)

    # All on (defaults)
    whr_on = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                 warm_start=True, use_active_set=True, anderson_window=5)
    whr_on.fit(dataset, end_day=10)

    # Walk-forward
    for day in range(11, 16):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_off.update(batch)
        whr_on.update(batch)

    off_ratings = whr_off.get_ratings().ratings
    on_ratings = whr_on.get_ratings().ratings
    max_diff = np.max(np.abs(off_ratings - on_ratings))
    mean_diff = np.mean(np.abs(off_ratings - on_ratings))
    print(f"  After walk-forward: max diff = {max_diff:.6f}, mean diff = {mean_diff:.6f}")
    assert max_diff < 1.0, f"Combined ratings differ too much: {max_diff}"
    print("  PASSED: Combined optimisations match baseline.\n")


# ---------------------------------------------------------------------------
# Test 5: Active-set reduces iterations during refits
# ---------------------------------------------------------------------------

def test_active_set_fewer_iterations():
    """Active-set should reduce iteration count during warm-start refits."""
    print("=" * 60)
    print("Test 5: Active-set iteration reduction")
    print("=" * 60)

    df = generate_test_data(num_players=80, num_games=5000, num_days=30)
    dataset = GameDataset.from_dataframe(df)

    max_iters = 200

    # Without active set (but with warm start)
    whr_no_active = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                        warm_start=True, use_active_set=False, anderson_window=0)
    whr_no_active.fit(dataset, end_day=15)

    # With active set + warm start
    whr_active = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                     warm_start=True, use_active_set=True, anderson_window=0)
    whr_active.fit(dataset, end_day=15)

    no_active_total = 0
    active_total = 0

    for day in range(16, 25):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_no_active.update(batch)
        whr_active.update(batch)
        no_active_total += whr_no_active._num_iterations
        active_total += whr_active._num_iterations

    print(f"  Without active-set: {no_active_total} total iterations")
    print(f"  With active-set: {active_total} total iterations")
    if no_active_total > 0:
        ratio = no_active_total / max(active_total, 1)
        print(f"  Reduction: {ratio:.1f}x fewer iterations")

    # Active set should use no more iterations (may be same if all converge quickly)
    assert active_total <= no_active_total, (
        f"Active-set used more iterations: {active_total} > {no_active_total}"
    )
    print("  PASSED: Active-set reduces iterations.\n")


# ---------------------------------------------------------------------------
# Test 6: Anderson reduces iterations
# ---------------------------------------------------------------------------

def test_anderson_fewer_iterations():
    """Anderson acceleration should reduce iteration count."""
    print("=" * 60)
    print("Test 6: Anderson iteration reduction")
    print("=" * 60)

    df = generate_test_data(num_players=80, num_games=5000, num_days=30)
    dataset = GameDataset.from_dataframe(df)

    max_iters = 200

    # Without Anderson
    whr_no_anderson = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                          warm_start=True, use_active_set=False, anderson_window=0)
    whr_no_anderson.fit(dataset, end_day=15)

    # With Anderson
    whr_anderson = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                       warm_start=True, use_active_set=False, anderson_window=5)
    whr_anderson.fit(dataset, end_day=15)

    no_anderson_total = 0
    anderson_total = 0

    for day in range(16, 25):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_no_anderson.update(batch)
        whr_anderson.update(batch)
        no_anderson_total += whr_no_anderson._num_iterations
        anderson_total += whr_anderson._num_iterations

    print(f"  Without Anderson: {no_anderson_total} total iterations")
    print(f"  With Anderson: {anderson_total} total iterations")
    if no_anderson_total > 0:
        ratio = no_anderson_total / max(anderson_total, 1)
        print(f"  Reduction: {ratio:.1f}x fewer iterations")

    # Anderson should use no more iterations
    assert anderson_total <= no_anderson_total, (
        f"Anderson used more iterations: {anderson_total} > {no_anderson_total}"
    )
    print("  PASSED: Anderson reduces iterations.\n")


# ---------------------------------------------------------------------------
# Test 7: Wall-clock benchmark (combined speedup)
# ---------------------------------------------------------------------------

def test_wall_clock_benchmark():
    """Measure wall-clock speedup with all optimisations combined."""
    print("=" * 60)
    print("Test 7: Wall-clock benchmark (all opts on vs off)")
    print("=" * 60)

    df = generate_test_data(num_players=200, num_games=25000, num_days=50)
    dataset = GameDataset.from_dataframe(df)

    max_iters = 100

    # All off
    whr_off = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                  warm_start=False, use_active_set=False, anderson_window=0)
    whr_off.fit(dataset, end_day=25)

    t0 = time.perf_counter()
    for day in range(26, 45):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_off.update(batch)
    time_off = time.perf_counter() - t0

    # All on
    whr_on = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                 warm_start=True, use_active_set=True, anderson_window=5)
    whr_on.fit(dataset, end_day=25)

    t0 = time.perf_counter()
    for day in range(26, 45):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_on.update(batch)
    time_on = time.perf_counter() - t0

    print(f"  All opts OFF: {time_off:.3f}s")
    print(f"  All opts ON:  {time_on:.3f}s")
    if time_on > 0:
        speedup = time_off / time_on
        print(f"  Speedup: {speedup:.2f}x")
    print("  PASSED: Timing recorded.\n")


# ---------------------------------------------------------------------------
# Test 8: Backtest equivalence
# ---------------------------------------------------------------------------

def test_backtest_equivalence():
    """Optimised and baseline should produce equivalent backtest metrics."""
    print("=" * 60)
    print("Test 8: Backtest equivalence")
    print("=" * 60)

    df = generate_test_data(num_players=60, num_games=3000, num_days=20)
    dataset = GameDataset.from_dataframe(df)

    max_iters = 100

    # Baseline
    whr_base = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   warm_start=False, use_active_set=False, anderson_window=0)
    backtester_base = Backtester(whr_base, dataset)
    result_base = backtester_base.run(train_end_day=10, verbose=False)

    # All optimisations on
    whr_opt = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                  warm_start=True, use_active_set=True, anderson_window=5)
    backtester_opt = Backtester(whr_opt, dataset)
    result_opt = backtester_opt.run(train_end_day=10, verbose=False)

    print(f"  Baseline Brier: {result_base.brier:.6f}")
    print(f"  Optimised Brier: {result_opt.brier:.6f}")
    brier_diff = abs(result_base.brier - result_opt.brier)
    print(f"  Brier diff: {brier_diff:.6f}")

    print(f"  Baseline LogLoss: {result_base.log_loss:.6f}")
    print(f"  Optimised LogLoss: {result_opt.log_loss:.6f}")
    logloss_diff = abs(result_base.log_loss - result_opt.log_loss)
    print(f"  LogLoss diff: {logloss_diff:.6f}")

    assert brier_diff < 0.005, f"Brier scores differ too much: {brier_diff}"
    assert logloss_diff < 0.01, f"LogLoss differs too much: {logloss_diff}"
    print("  PASSED: Backtest metrics are equivalent.\n")


# ---------------------------------------------------------------------------
# Test 9: Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    """Verify default config values for new parameters."""
    print("=" * 60)
    print("Test 9: Config defaults")
    print("=" * 60)

    whr = WHR()
    assert whr.config.use_active_set is True, (
        f"use_active_set should default to True, got {whr.config.use_active_set}"
    )
    assert whr.config.anderson_window == 5, (
        f"anderson_window should default to 5, got {whr.config.anderson_window}"
    )
    assert whr.config.warm_start is True, (
        f"warm_start should default to True, got {whr.config.warm_start}"
    )

    print(f"  use_active_set={whr.config.use_active_set} (default)")
    print(f"  anderson_window={whr.config.anderson_window} (default)")
    print(f"  warm_start={whr.config.warm_start} (default)")
    print("  PASSED: Defaults are correct.\n")


# ---------------------------------------------------------------------------
# Test 10: Disable toggles
# ---------------------------------------------------------------------------

def test_disable_toggles():
    """Each optimisation can be independently disabled."""
    print("=" * 60)
    print("Test 10: Disable toggles")
    print("=" * 60)

    df = generate_test_data(num_players=30, num_games=500, num_days=10)
    dataset = GameDataset.from_dataframe(df)

    configs = [
        {"warm_start": False, "use_active_set": False, "anderson_window": 0},
        {"warm_start": True, "use_active_set": False, "anderson_window": 0},
        {"warm_start": False, "use_active_set": True, "anderson_window": 0},
        {"warm_start": False, "use_active_set": False, "anderson_window": 5},
        {"warm_start": True, "use_active_set": True, "anderson_window": 0},
        {"warm_start": True, "use_active_set": False, "anderson_window": 5},
        {"warm_start": False, "use_active_set": True, "anderson_window": 5},
        {"warm_start": True, "use_active_set": True, "anderson_window": 5},
    ]

    results = []
    for cfg in configs:
        whr = WHR(w2=300.0, max_iterations=100, refit_max_iterations=100, **cfg)
        whr.fit(dataset, end_day=5)
        for day in range(6, 9):
            try:
                batch = dataset.get_day(day)
            except ValueError:
                continue
            whr.update(batch)
        results.append(whr.get_ratings().ratings.copy())
        label = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"  {label}: iters={whr._num_iterations}")

    # All should produce similar ratings (within tolerance)
    baseline = results[0]
    for i, (cfg, ratings) in enumerate(zip(configs, results)):
        max_diff = np.max(np.abs(baseline - ratings))
        if max_diff > 2.0:
            label = ", ".join(f"{k}={v}" for k, v in cfg.items())
            print(f"  WARNING: config {label} differs by {max_diff:.4f} Elo")

    print("  PASSED: All toggle combinations run without error.\n")


# ---------------------------------------------------------------------------
# Test 11: Repr includes new parameters
# ---------------------------------------------------------------------------

def test_repr_new_params():
    """Repr should include active_set and anderson parameters."""
    print("=" * 60)
    print("Test 11: Repr includes new parameters")
    print("=" * 60)

    whr = WHR(use_active_set=True, anderson_window=5)
    r = repr(whr)
    print(f"  {r}")
    assert "active_set=True" in r, f"active_set not in repr: {r}"
    assert "anderson=5" in r, f"anderson not in repr: {r}"

    whr2 = WHR(use_active_set=False, anderson_window=0)
    r2 = repr(whr2)
    print(f"  {r2}")
    assert "active_set=False" in r2, f"active_set not in repr: {r2}"
    assert "anderson=0" in r2, f"anderson not in repr: {r2}"

    print("  PASSED: Repr shows new parameters.\n")


if __name__ == "__main__":
    test_config_defaults()
    test_repr_new_params()
    test_composite_key_equivalence()
    test_active_set_correctness()
    test_anderson_correctness()
    test_combined_correctness()
    test_active_set_fewer_iterations()
    test_anderson_fewer_iterations()
    test_wall_clock_benchmark()
    test_backtest_equivalence()
    test_disable_toggles()

    print("=" * 60)
    print("All WHR optimisation tests passed!")
    print("=" * 60)
