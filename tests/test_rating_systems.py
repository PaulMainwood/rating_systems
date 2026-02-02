"""Test script for rating systems package."""

import numpy as np
import pandas as pd

from rating_systems import (
    GameDataset,
    Elo,
    Glicko,
    Glicko2,
    Stephenson,
    WHR,
    TrueSkillThroughTime,
    Backtester,
    compare_systems,
)


def generate_test_data(
    num_players: int = 100,
    num_games: int = 10000,
    num_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic test data with skill-based outcomes."""
    np.random.seed(seed)

    # Assign true skill to each player
    true_skill = np.linspace(0, 1, num_players)

    p1 = np.random.randint(0, num_players, num_games)
    p2 = np.random.randint(0, num_players, num_games)

    # Ensure players don't play themselves
    while (p1 == p2).any():
        mask = p1 == p2
        p2[mask] = np.random.randint(0, num_players, mask.sum())

    # Generate scores based on true skill difference
    skill_diff = true_skill[p1] - true_skill[p2]
    win_prob = 1 / (1 + np.exp(-skill_diff * 4))  # Sigmoid with steepness 4
    scores = (np.random.random(num_games) < win_prob).astype(float)

    days = np.sort(np.random.randint(0, num_days, num_games))

    return pd.DataFrame({
        "Player1": p1,
        "Player2": p2,
        "Score": scores,
        "Day": days,
    })


def test_basic_functionality():
    """Test basic fit/predict/update functionality."""
    print("=" * 60)
    print("Testing basic functionality...")
    print("=" * 60)

    # Generate data
    df = generate_test_data(num_players=50, num_games=1000, num_days=10)
    dataset = GameDataset.from_dataframe(df)

    print(f"Dataset: {dataset}")

    # Test online systems
    for SystemClass in [Elo, Glicko, Glicko2, Stephenson]:
        print(f"\nTesting {SystemClass.__name__}...")

        system = SystemClass()
        print(f"  Before fit: {system}")

        # Fit
        system.fit(dataset, end_day=5)
        print(f"  After fit (day 5): {system}")

        # Get ratings
        ratings = system.get_ratings()
        print(f"  Ratings shape: {ratings.ratings.shape}")
        print(f"  Rating range: [{ratings.ratings.min():.1f}, {ratings.ratings.max():.1f}]")

        # Predict
        batch = dataset.get_day(6)
        predictions = system.predict(batch)
        print(f"  Predictions for day 6: {len(predictions)} games")
        print(f"  Prediction range: [{predictions.predicted_proba.min():.3f}, {predictions.predicted_proba.max():.3f}]")

        # Update
        system.update(batch)
        print(f"  After update: current_day = {system.current_day}")

    print("\nBasic functionality tests passed!")


def test_whr():
    """Test WHR-specific functionality."""
    print("\n" + "=" * 60)
    print("Testing WHR (Whole History Rating)...")
    print("=" * 60)

    # Generate data
    df = generate_test_data(num_players=30, num_games=500, num_days=10)
    dataset = GameDataset.from_dataframe(df)

    print(f"Dataset: {dataset}")

    # Create WHR system
    whr = WHR(w2=300.0, max_iterations=50)
    print(f"Before fit: {whr}")

    # Fit on first 5 days
    whr.fit(dataset, end_day=5)
    print(f"After fit (day 5): {whr}")

    # Get ratings
    ratings = whr.get_ratings()
    print(f"Ratings shape: {ratings.ratings.shape}")
    print(f"Rating range: [{ratings.ratings.min():.1f}, {ratings.ratings.max():.1f}]")
    if ratings.rd is not None:
        print(f"Uncertainty range: [{ratings.rd.min():.1f}, {ratings.rd.max():.1f}]")

    # Test rating history for a player
    history = whr.get_rating_history(0)
    if history:
        print(f"Player 0 history: {len(history['days'])} days")
        print(f"  Days: {history['days'][:5]}...")
        print(f"  Ratings: {[f'{r:.1f}' for r in history['ratings'][:5]]}...")

    # Predict
    batch = dataset.get_day(6)
    predictions = whr.predict(batch)
    print(f"Predictions for day 6: {len(predictions)} games")
    print(f"Prediction range: [{predictions.predicted_proba.min():.3f}, {predictions.predicted_proba.max():.3f}]")

    # Update (WHR refits on all data)
    whr.update(batch)
    print(f"After update: current_day = {whr.current_day}")

    # Verify ratings changed
    new_ratings = whr.get_ratings()
    print(f"Rating range after update: [{new_ratings.ratings.min():.1f}, {new_ratings.ratings.max():.1f}]")

    print("\nWHR tests passed!")


def test_ttt():
    """Test TrueSkill Through Time functionality."""
    print("\n" + "=" * 60)
    print("Testing TrueSkill Through Time...")
    print("=" * 60)

    # Generate smaller data (TTT is computationally intensive)
    df = generate_test_data(num_players=20, num_games=200, num_days=5)
    dataset = GameDataset.from_dataframe(df)

    print(f"Dataset: {dataset}")

    # Create TTT system (uses internal scale, displayed in Elo-like scale)
    ttt = TrueSkillThroughTime(max_iterations=30)
    print(f"Before fit: {ttt}")

    # Fit on first 5 days
    ttt.fit(dataset, end_day=5)
    print(f"After fit (day 5): {ttt}")

    # Get ratings
    ratings = ttt.get_ratings()
    print(f"Ratings shape: {ratings.ratings.shape}")
    print(f"Rating range: [{ratings.ratings.min():.1f}, {ratings.ratings.max():.1f}]")
    if ratings.rd is not None:
        print(f"Uncertainty range: [{ratings.rd.min():.1f}, {ratings.rd.max():.1f}]")

    # Test rating history
    history = ttt.get_rating_history(0)
    if history:
        print(f"Player 0 history: {len(history['days'])} days")
        print(f"  Ratings: {[f'{r:.1f}' for r in history['ratings'][:5]]}...")

    # Predict on last day
    last_day = dataset.max_day
    batch = dataset.get_day(last_day)
    predictions = ttt.predict(batch)
    print(f"Predictions for day {last_day}: {len(predictions)} games")
    print(f"Prediction range: [{predictions.predicted_proba.min():.3f}, {predictions.predicted_proba.max():.3f}]")

    print("\nTTT tests passed!")


def test_backtesting():
    """Test backtesting functionality."""
    print("\n" + "=" * 60)
    print("Testing backtesting...")
    print("=" * 60)

    # Generate data
    df = generate_test_data(num_players=100, num_games=5000, num_days=30)
    dataset = GameDataset.from_dataframe(df)

    print(f"Dataset: {dataset}")

    # Create systems with different parameters
    elo = Elo(k_factor=32)

    # Run backtest
    backtester = Backtester(elo, dataset)
    result = backtester.run(train_end_day=15, verbose=True)

    print("\nDaily results (first 5 days):")
    print(result.to_dataframe().head())

    print("\nBacktesting tests passed!")


def test_system_comparison():
    """Test comparing multiple systems."""
    print("\n" + "=" * 60)
    print("Comparing rating systems...")
    print("=" * 60)

    # Generate data
    df = generate_test_data(num_players=100, num_games=5000, num_days=30)
    dataset = GameDataset.from_dataframe(df)

    # Create systems (exclude TTT from comparison - too slow for large datasets)
    systems = [
        Elo(k_factor=16),
        Elo(k_factor=32),
        Glicko(),
        Glicko2(tau=0.5),
        Stephenson(),
        WHR(w2=300.0, max_iterations=30),
        # TrueSkillThroughTime excluded - O(n²) per iteration is slow
    ]

    # Compare
    comparison = compare_systems(systems, dataset, train_end_day=15, verbose=False)

    print("\nComparison results:")
    print(comparison.to_string(index=False))

    print("\nSystem comparison tests passed!")


def test_dataset_operations():
    """Test dataset filtering and splitting."""
    print("\n" + "=" * 60)
    print("Testing dataset operations...")
    print("=" * 60)

    df = generate_test_data(num_players=50, num_games=2000, num_days=20)
    dataset = GameDataset.from_dataframe(df)

    print(f"Full dataset: {dataset}")
    print(f"Days: {dataset.min_day} to {dataset.max_day}")

    # Filter
    filtered = dataset.filter_days(start_day=5, end_day=15)
    print(f"Filtered [5, 15]: {filtered}")

    # Split
    train, test = dataset.split_by_day(split_day=10)
    print(f"Train (< day 10): {train}")
    print(f"Test (>= day 10): {test}")

    # Iterate by day
    day_counts = []
    for batch in dataset.iter_days():
        day_counts.append((batch.day, len(batch)))

    print(f"Days with games: {len(day_counts)}")
    print(f"Games per day: min={min(c for _, c in day_counts)}, max={max(c for _, c in day_counts)}")

    print("\nDataset operations tests passed!")


if __name__ == "__main__":
    np.random.seed(42)

    test_basic_functionality()
    test_whr()
    test_ttt()
    test_dataset_operations()
    test_backtesting()
    test_system_comparison()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
