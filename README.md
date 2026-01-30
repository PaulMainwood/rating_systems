# Rating Systems

PyTorch-based implementations of popular rating systems with GPU acceleration support.

## Features

- **Multiple Rating Systems**: Elo, Glicko, Glicko-2, Whole History Rating (WHR), TrueSkill Through Time
- **GPU Acceleration**: Automatic device detection (CUDA > MPS > CPU)
- **Unified Interface**: All systems share `fit()`, `predict_proba()`, `update()` methods
- **Backtesting**: Walk-forward validation with Brier score, log loss, and accuracy metrics
- **Time-varying Ratings**: WHR and TTT track skill evolution over time

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from rating_systems import GameDataset, Elo, Glicko2, WHR, Backtester

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

| System | Description | Key Parameters |
|--------|-------------|----------------|
| **Elo** | Classic rating system | `k_factor` (default: 32) |
| **Glicko** | Adds rating deviation (RD) | `initial_rd`, `c` |
| **Glicko-2** | Adds volatility parameter | `tau`, `initial_volatility` |

### Batch Systems (Full Refit)

| System | Description | Key Parameters |
|--------|-------------|----------------|
| **WHR** | Whole History Rating - Bayesian with time-varying skill | `w2` (drift variance) |
| **TrueSkillThroughTime** | Gaussian belief propagation | `sigma`, `beta`, `gamma` |

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
    WHR(w2=300),
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
│   ├── elo/
│   ├── glicko/
│   ├── glicko2/
│   ├── whr/
│   └── trueskill_through_time/
├── evaluation/     # Backtester, metrics
└── utils/          # Device detection
```

## License

MIT
