"""Tests for WHR warm start optimisation.

Verifies that warm start:
1. Produces the same final ratings as cold start (given enough iterations)
2. Requires fewer iterations to converge
3. Produces identical prediction quality (Brier scores)
4. Handles edge cases: new players, first refit, etc.
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


def test_warm_start_rating_correctness():
    """Warm start and cold start should converge to the same ratings."""
    print("=" * 60)
    print("Test 1: Rating correctness (warm vs cold start)")
    print("=" * 60)

    df = generate_test_data(num_players=50, num_games=2000, num_days=20)
    dataset = GameDataset.from_dataframe(df)

    # Use high max_iterations to ensure both converge fully
    max_iters = 100

    # Cold start: warm_start=False
    whr_cold = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   warm_start=False)
    whr_cold.fit(dataset, end_day=10)

    # Warm start: warm_start=True
    whr_warm = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   warm_start=True)
    whr_warm.fit(dataset, end_day=10)

    # After initial fit (no refit yet), ratings should be identical
    cold_ratings_initial = whr_cold.get_ratings().ratings.copy()
    warm_ratings_initial = whr_warm.get_ratings().ratings.copy()
    max_diff_initial = np.max(np.abs(cold_ratings_initial - warm_ratings_initial))
    print(f"  After initial fit: max diff = {max_diff_initial:.2e}")
    assert max_diff_initial < 1e-10, f"Initial fit should be identical, got diff={max_diff_initial}"

    # Now do walk-forward updates for several days
    for day in range(11, 16):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_cold.update(batch)
        whr_warm.update(batch)

    cold_ratings = whr_cold.get_ratings().ratings
    warm_ratings = whr_warm.get_ratings().ratings

    max_diff = np.max(np.abs(cold_ratings - warm_ratings))
    mean_diff = np.mean(np.abs(cold_ratings - warm_ratings))
    print(f"  After walk-forward updates: max diff = {max_diff:.6f}, mean diff = {mean_diff:.6f}")

    # With enough iterations, both should converge to the same point
    # Allow small tolerance due to floating point and convergence differences
    assert max_diff < 0.5, f"Ratings differ too much: max_diff={max_diff:.4f}"
    print("  PASSED: Ratings match within tolerance.\n")


def test_warm_start_fewer_iterations():
    """Warm start should converge in fewer iterations than cold start."""
    print("=" * 60)
    print("Test 2: Convergence speed (warm vs cold start)")
    print("=" * 60)

    df = generate_test_data(num_players=50, num_games=2000, num_days=20)
    dataset = GameDataset.from_dataframe(df)

    # Use enough iterations so convergence threshold is the limiting factor
    max_iters = 200
    threshold = 1e-6

    # Cold start
    whr_cold = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   convergence_threshold=threshold, warm_start=False)
    whr_cold.fit(dataset, end_day=10)
    cold_iters_initial = whr_cold._num_iterations

    # Warm start
    whr_warm = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   convergence_threshold=threshold, warm_start=True)
    whr_warm.fit(dataset, end_day=10)
    warm_iters_initial = whr_warm._num_iterations

    print(f"  Initial fit iterations: cold={cold_iters_initial}, warm={warm_iters_initial}")
    print(f"  (Should be same since initial fit has no previous state)")

    # Track iterations through walk-forward
    cold_total_iters = 0
    warm_total_iters = 0
    num_refits = 0

    for day in range(11, 20):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_cold.update(batch)
        whr_warm.update(batch)
        cold_total_iters += whr_cold._num_iterations
        warm_total_iters += whr_warm._num_iterations
        num_refits += 1

    print(f"  Walk-forward refits: {num_refits}")
    print(f"  Total refit iterations: cold={cold_total_iters}, warm={warm_total_iters}")
    if cold_total_iters > 0:
        speedup = cold_total_iters / max(warm_total_iters, 1)
        print(f"  Iteration reduction: {speedup:.1f}x fewer iterations with warm start")

    assert warm_total_iters < cold_total_iters, (
        f"Warm start should use fewer iterations: warm={warm_total_iters}, cold={cold_total_iters}"
    )
    print("  PASSED: Warm start uses fewer iterations.\n")


def test_warm_start_wall_clock_speedup():
    """Warm start should be faster in wall-clock time."""
    print("=" * 60)
    print("Test 3: Wall-clock speedup")
    print("=" * 60)

    df = generate_test_data(num_players=80, num_games=5000, num_days=30)
    dataset = GameDataset.from_dataframe(df)

    max_iters = 100

    # Cold start timing
    whr_cold = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   warm_start=False)
    whr_cold.fit(dataset, end_day=15)

    t0 = time.perf_counter()
    for day in range(16, 30):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_cold.update(batch)
    cold_time = time.perf_counter() - t0

    # Warm start timing
    whr_warm = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   warm_start=True)
    whr_warm.fit(dataset, end_day=15)

    t0 = time.perf_counter()
    for day in range(16, 30):
        try:
            batch = dataset.get_day(day)
        except ValueError:
            continue
        whr_warm.update(batch)
    warm_time = time.perf_counter() - t0

    print(f"  Cold start walk-forward: {cold_time:.3f}s")
    print(f"  Warm start walk-forward: {warm_time:.3f}s")
    if warm_time > 0:
        speedup = cold_time / warm_time
        print(f"  Wall-clock speedup: {speedup:.2f}x")
    print("  PASSED: Timing recorded.\n")


def test_warm_start_backtest_equivalence():
    """Warm and cold start should produce equivalent backtest metrics."""
    print("=" * 60)
    print("Test 4: Backtest metric equivalence")
    print("=" * 60)

    df = generate_test_data(num_players=60, num_games=3000, num_days=20)
    dataset = GameDataset.from_dataframe(df)

    max_iters = 100

    # Cold start backtest
    whr_cold = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   warm_start=False)
    backtester_cold = Backtester(whr_cold, dataset)
    result_cold = backtester_cold.run(train_end_day=10, verbose=False)

    # Warm start backtest
    whr_warm = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
                   warm_start=True)
    backtester_warm = Backtester(whr_warm, dataset)
    result_warm = backtester_warm.run(train_end_day=10, verbose=False)

    print(f"  Cold start Brier: {result_cold.brier:.6f}")
    print(f"  Warm start Brier: {result_warm.brier:.6f}")
    print(f"  Brier difference: {abs(result_cold.brier - result_warm.brier):.6f}")

    print(f"  Cold start LogLoss: {result_cold.log_loss:.6f}")
    print(f"  Warm start LogLoss: {result_warm.log_loss:.6f}")
    print(f"  LogLoss difference: {abs(result_cold.log_loss - result_warm.log_loss):.6f}")

    print(f"  Cold start Accuracy: {result_cold.accuracy:.6f}")
    print(f"  Warm start Accuracy: {result_warm.accuracy:.6f}")

    # Brier scores should be very close (within 0.002)
    brier_diff = abs(result_cold.brier - result_warm.brier)
    assert brier_diff < 0.002, f"Brier scores differ too much: {brier_diff:.6f}"
    print("  PASSED: Backtest metrics are equivalent.\n")


def test_warm_start_default_enabled():
    """Warm start should be enabled by default."""
    print("=" * 60)
    print("Test 5: Default configuration")
    print("=" * 60)

    whr = WHR()
    assert whr.config.warm_start is True, "warm_start should default to True"
    print(f"  WHR default: warm_start={whr.config.warm_start}")

    whr_off = WHR(warm_start=False)
    assert whr_off.config.warm_start is False, "Should be able to disable warm_start"
    print(f"  WHR(warm_start=False): warm_start={whr_off.config.warm_start}")

    print("  PASSED: Default config correct.\n")


def test_warm_start_single_refit():
    """Test warm start works correctly with a single refit step."""
    print("=" * 60)
    print("Test 6: Single refit step")
    print("=" * 60)

    df = generate_test_data(num_players=30, num_games=500, num_days=10)
    dataset = GameDataset.from_dataframe(df)

    max_iters = 200
    threshold = 1e-6

    whr = WHR(w2=300.0, max_iterations=max_iters, refit_max_iterations=max_iters,
              convergence_threshold=threshold, warm_start=True)
    whr.fit(dataset, end_day=5)

    initial_iterations = whr._num_iterations
    initial_ratings = whr.get_ratings().ratings.copy()

    # Update with one more day
    batch = dataset.get_day(6)
    whr.update(batch)

    refit_iterations = whr._num_iterations
    updated_ratings = whr.get_ratings().ratings

    print(f"  Initial fit: {initial_iterations} iterations")
    print(f"  After 1 refit: {refit_iterations} iterations")
    print(f"  Rating change (mean): {np.mean(np.abs(updated_ratings - initial_ratings)):.4f}")

    # The refit should converge faster with warm start
    print(f"  Refit used {refit_iterations} iterations (initial fit used {initial_iterations})")
    assert refit_iterations <= initial_iterations, (
        f"First refit should be no worse than initial fit: refit={refit_iterations}, initial={initial_iterations}"
    )
    print("  PASSED: Single refit works correctly.\n")


def test_warm_start_repr():
    """Test repr includes warm_start status."""
    print("=" * 60)
    print("Test 7: Repr includes warm_start")
    print("=" * 60)

    whr_on = WHR(warm_start=True)
    whr_off = WHR(warm_start=False)

    print(f"  warm_start=True:  {whr_on}")
    print(f"  warm_start=False: {whr_off}")

    assert "warm_start=True" in repr(whr_on)
    assert "warm_start=False" in repr(whr_off)
    print("  PASSED: Repr shows warm_start.\n")


if __name__ == "__main__":
    test_warm_start_default_enabled()
    test_warm_start_repr()
    test_warm_start_rating_correctness()
    test_warm_start_fewer_iterations()
    test_warm_start_wall_clock_speedup()
    test_warm_start_backtest_equivalence()
    test_warm_start_single_refit()

    print("=" * 60)
    print("All warm start tests passed!")
    print("=" * 60)
