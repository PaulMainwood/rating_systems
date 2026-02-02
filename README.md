# Rating Systems

High-performance implementations of popular rating systems using Numba JIT compilation.

## Features

- **Fast**: Numba-accelerated implementations (3-15x faster than pure Python)
- **Multiple Systems**: Elo, Glicko, Glicko-2, Stephenson, TrueSkill, Yuksel, WHR, TrueSkill Through Time
- **Unified Interface**: All systems share `fit()`, `predict_proba()`, `update()` methods
- **Backtesting**: Walk-forward validation with Brier score, log loss, and accuracy metrics
- **Time-varying Ratings**: WHR and TTT track skill evolution over time

## Installation

```bash
# Core installation
pip install -e .

# Development installation (includes pytest, black, etc.)
pip install -e ".[dev]"
```

## Quick Start

```python
from rating_systems import GameDataset, Elo, Glicko2, Backtester

# Load data (parquet with columns: Player1, Player2, Score, Day)
dataset = GameDataset.from_parquet("games.parquet")

# Create and fit a rating system
elo = Elo(k_factor=32)
elo.fit(dataset)

# Get ratings
ratings = elo.get_ratings()
print(ratings.to_dataframe())

# Predict outcomes
proba = elo.predict_proba(player1_ids, player2_ids)

# Backtest
backtester = Backtester(elo, dataset)
results = backtester.run(train_end_day=100)
print(f"Brier Score: {results.brier:.4f}")
```

## Rating Systems

### Online Systems (Incremental Updates)

These systems can update ratings incrementally as new games arrive.

| System | Description | Key Parameters |
|--------|-------------|----------------|
| **Elo** | Classic rating system | `k_factor` (32), `initial_rating` (1500) |
| **Glicko** | Adds rating deviation (RD) | `initial_rd` (350), `c` (34.6) |
| **Glicko2** | Adds volatility parameter | `tau` (0.5), `initial_volatility` (0.06) |
| **Stephenson** | Extended Glicko with neighbourhood parameter | `hval` (10), `lambda_param` (2) |
| **TrueSkill** | Bayesian skill estimation | `initial_sigma` (8.33), `beta` (4.17) |
| **Yuksel** | Adaptive with uncertainty tracking | `delta_r_max` (32), `alpha` (0.1) |

### Batch Systems (Full History Refit)

These systems must refit on all historical data when new games arrive.

| System | Description | Key Parameters |
|--------|-------------|----------------|
| **WHR** | Whole History Rating - Bayesian with time-varying skill | `w2` (300), `max_iterations` (50) |
| **TrueSkillThroughTime** | Gaussian belief propagation | `sigma` (6), `beta` (1), `gamma` (0.03) |
| **SurfaceTTT** | Surface-specific TTT (e.g., for tennis) | `base_weight` (0.6), `surface_sigma` (3) |

## Data Format

Input parquet files should have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `Player1` | Int | Player 1 ID (0-indexed) |
| `Player2` | Int | Player 2 ID (0-indexed) |
| `Score` | Float | 1.0 = Player1 wins, 0.0 = Player2 wins |
| `Day` | Int | Time period / rating period |

## Backtesting

Walk-forward validation that simulates real-world usage:

```python
from rating_systems import compare_systems

systems = [
    Elo(k_factor=16),
    Elo(k_factor=32),
    Glicko2(),
    Stephenson(),
]

comparison = compare_systems(systems, dataset, train_end_day=100)
print(comparison)
```

## Project Structure

```
rating_systems/
├── data/           # GameDataset, GameBatch
├── base/           # RatingSystem base class, PlayerRatings
├── systems/
│   ├── elo/        # Elo
│   ├── glicko/     # Glicko
│   ├── glicko2/    # Glicko-2
│   ├── stephenson/ # Stephenson
│   ├── trueskill/  # TrueSkill
│   ├── yuksel/     # Yuksel
│   ├── whr/        # Whole History Rating
│   └── trueskill_through_time/  # TTT and SurfaceTTT
├── evaluation/     # Backtester, metrics
└── results/        # FittedRatings classes
```

## License

MIT
