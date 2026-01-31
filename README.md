# Rating Systems

High-performance implementations of popular rating systems using Numba JIT compilation.

## Features

- **Fast by Default**: Numba-accelerated implementations (3-15x faster than pure Python)
- **Multiple Rating Systems**: Elo, Glicko, Glicko-2, Whole History Rating (WHR), TrueSkill Through Time
- **Optional GPU Support**: PyTorch-based variants available for GPU acceleration on very large datasets
- **Unified Interface**: All systems share `fit()`, `predict_proba()`, `update()` methods
- **Backtesting**: Walk-forward validation with Brier score, log loss, and accuracy metrics
- **Time-varying Ratings**: WHR and TTT track skill evolution over time

## Installation

```bash
# Core installation (Numba-based, no GPU)
pip install -e .

# With PyTorch support for GPU acceleration
pip install -e ".[torch]"

# Development installation
pip install -e ".[dev]"
```

## Quick Start

```python
from rating_systems import GameDataset, Elo, Glicko2, Backtester

# Load data (parquet with columns: Player1, Player2, Score, Day)
dataset = GameDataset.from_parquet("games.parquet")

# Create and fit a rating system (uses fast Numba backend)
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

| System | Description | Key Parameters | Backend |
|--------|-------------|----------------|---------|
| **Elo** | Classic rating system | `k_factor` (default: 32) | Numba |
| **Glicko** | Adds rating deviation (RD) | `initial_rd`, `c` | Numba |
| **Glicko2** | Adds volatility parameter | `tau`, `initial_volatility` | Numba |
| **EloTorch** | Elo with GPU support | Same as Elo | PyTorch |
| **GlickoTorch** | Glicko with GPU support | Same as Glicko | PyTorch |
| **Glicko2Torch** | Glicko-2 with GPU support | Same as Glicko2 | PyTorch |

### Batch Systems (Full Refit)

| System | Description | Key Parameters |
|--------|-------------|----------------|
| **WHR** | Whole History Rating - Bayesian with time-varying skill | `w2` (drift variance) |
| **TrueSkillThroughTime** | Gaussian belief propagation | `sigma`, `beta`, `gamma` |

## Performance

Numba implementations provide significant speedups over PyTorch for sequential game processing:

| System | Numba | PyTorch | Speedup |
|--------|-------|---------|---------|
| Elo | 0.66s | 2.52s | 3.8x |
| Glicko | 1.38s | 21.9s | 15.8x |
| Glicko-2 | 1.59s | 15.3s | 9.6x |

*Benchmark: 50,000 games, 1,000 players, 500 days on CPU*

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
]

comparison = compare_systems(systems, dataset, train_end_day=100)
print(comparison)
```

## Using PyTorch Backends

For GPU acceleration on very large datasets:

```python
from rating_systems import EloTorch, GlickoTorch, Glicko2Torch
import torch

# Uses CUDA if available, otherwise MPS (Apple Silicon) or CPU
elo_gpu = EloTorch(k_factor=32)
elo_gpu.fit(dataset)

# Explicitly set device
elo_cuda = EloTorch(k_factor=32, device=torch.device("cuda"))
```

## Project Structure

```
rating_systems/
├── data/           # GameDataset, GameBatch
├── base/           # RatingSystem base class, PlayerRatings
├── systems/
│   ├── elo/        # Elo (Numba) + EloTorch
│   ├── glicko/     # Glicko (Numba) + GlickoTorch
│   ├── glicko2/    # Glicko2 (Numba) + Glicko2Torch
│   ├── whr/        # Whole History Rating
│   └── trueskill_through_time/
├── evaluation/     # Backtester, metrics
└── utils/          # Device detection utilities
```

## License

MIT
