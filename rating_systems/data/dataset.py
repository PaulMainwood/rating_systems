"""Dataset classes for loading and managing game data.

Uses Polars for high-performance data manipulation and batching.
"""

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from .types import GameBatch


class GameDataset:
    """
    Container for game data loaded from parquet files.

    Uses Polars internally for fast group-by operations when iterating by day.
    Falls back to pandas if Polars is not installed.

    Provides methods for:
    - Loading data from parquet files
    - Filtering by day range
    - Splitting into train/test sets
    - Iterating over games by day (rating period) - FAST with Polars
    """

    def __init__(self, df=None):
        """
        Initialize dataset.

        Args:
            df: DataFrame with columns Player1, Player2, Score, Day
                (can be pandas or polars DataFrame)
        """
        self._num_players: Optional[int] = None
        self._days: Optional[np.ndarray] = None

        # Internal storage: pre-grouped numpy arrays for fast iteration
        self._day_indices: Optional[np.ndarray] = None  # Day values
        self._day_offsets: Optional[np.ndarray] = None  # Start index for each day
        self._player1: Optional[np.ndarray] = None
        self._player2: Optional[np.ndarray] = None
        self._scores: Optional[np.ndarray] = None

        if df is not None:
            self._load_dataframe(df)

    def _load_dataframe(self, df) -> None:
        """Load and preprocess DataFrame for fast iteration."""
        # Convert to polars if not already
        if not isinstance(df, pl.DataFrame):
            # Assume pandas DataFrame - convert to polars
            df = pl.from_pandas(df)
        self._load_polars(df)

    def _load_polars(self, df: "pl.DataFrame") -> None:
        """Load using Polars (fast path)."""
        # Validate columns
        required = {"Player1", "Player2", "Score", "Day"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Sort by day for sequential access
        df = df.sort("Day")

        # Extract arrays (zero-copy where possible)
        self._player1 = df["Player1"].to_numpy().astype(np.int64)
        self._player2 = df["Player2"].to_numpy().astype(np.int64)
        self._scores = df["Score"].to_numpy().astype(np.float64)
        days_arr = df["Day"].to_numpy()

        # Compute day boundaries using Polars group-by
        day_groups = df.group_by("Day", maintain_order=True).agg(pl.len().alias("count"))
        self._day_indices = day_groups["Day"].to_numpy().astype(np.int32)
        counts = day_groups["count"].to_numpy()

        # Compute offsets (cumsum of counts, starting at 0)
        self._day_offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=self._day_offsets[1:])

        # Metadata
        self._num_players = int(max(self._player1.max(), self._player2.max())) + 1
        self._days = self._day_indices

    @classmethod
    def from_parquet(cls, path: Union[str, Path]) -> "GameDataset":
        """Load dataset from a parquet file."""
        df = pl.read_parquet(path)
        return cls(df=df)

    @classmethod
    def from_dataframe(cls, df) -> "GameDataset":
        """Create dataset from a DataFrame (pandas or polars)."""
        return cls(df=df)

    @property
    def num_players(self) -> int:
        """Number of unique players (max player ID + 1)."""
        if self._num_players is None:
            raise ValueError("No data loaded")
        return self._num_players

    @property
    def num_games(self) -> int:
        """Total number of games."""
        if self._player1 is None:
            return 0
        return len(self._player1)

    @property
    def num_days(self) -> int:
        """Number of unique days."""
        if self._day_indices is None:
            return 0
        return len(self._day_indices)

    @property
    def days(self) -> List[int]:
        """List of unique days in sorted order."""
        if self._day_indices is None:
            return []
        return self._day_indices.tolist()

    @property
    def min_day(self) -> int:
        """First day in the dataset."""
        if self._day_indices is None or len(self._day_indices) == 0:
            raise ValueError("No data loaded")
        return int(self._day_indices[0])

    @property
    def max_day(self) -> int:
        """Last day in the dataset."""
        if self._day_indices is None or len(self._day_indices) == 0:
            raise ValueError("No data loaded")
        return int(self._day_indices[-1])

    def filter_days(
        self,
        start_day: Optional[int] = None,
        end_day: Optional[int] = None,
    ) -> "GameDataset":
        """
        Create a new dataset filtered to a day range.

        Args:
            start_day: First day to include (inclusive)
            end_day: Last day to include (inclusive)

        Returns:
            New GameDataset with filtered data
        """
        if self._day_indices is None:
            raise ValueError("No data loaded")

        # Find day index range
        start_idx = 0
        end_idx = len(self._day_indices)

        if start_day is not None:
            start_idx = np.searchsorted(self._day_indices, start_day, side='left')
        if end_day is not None:
            end_idx = np.searchsorted(self._day_indices, end_day, side='right')

        # Get game index range
        game_start = self._day_offsets[start_idx]
        game_end = self._day_offsets[end_idx]

        # Create new dataset with sliced data
        new_dataset = GameDataset()
        new_dataset._player1 = self._player1[game_start:game_end].copy()
        new_dataset._player2 = self._player2[game_start:game_end].copy()
        new_dataset._scores = self._scores[game_start:game_end].copy()
        new_dataset._day_indices = self._day_indices[start_idx:end_idx].copy()

        # Recompute offsets relative to new start
        new_dataset._day_offsets = self._day_offsets[start_idx:end_idx + 1].copy()
        new_dataset._day_offsets -= game_start

        new_dataset._num_players = self._num_players
        new_dataset._days = new_dataset._day_indices

        return new_dataset

    def split_by_day(
        self,
        split_day: int,
    ) -> Tuple["GameDataset", "GameDataset"]:
        """
        Split dataset into train and test sets by day.

        Args:
            split_day: First day of test set. Train includes all days < split_day.

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        train = self.filter_days(end_day=split_day - 1)
        test = self.filter_days(start_day=split_day)
        return train, test

    def iter_days(self) -> Iterator[GameBatch]:
        """
        Iterate over games grouped by day.

        This is highly optimized - data is pre-sorted and pre-indexed,
        so iteration is just slicing pre-computed arrays.
        """
        if self._day_indices is None:
            return

        for i in range(len(self._day_indices)):
            start = self._day_offsets[i]
            end = self._day_offsets[i + 1]

            yield GameBatch(
                player1=self._player1[start:end],
                player2=self._player2[start:end],
                scores=self._scores[start:end],
                day=int(self._day_indices[i]),
            )

    def get_all_games(self) -> GameBatch:
        """Get all games as a single batch."""
        if self._player1 is None:
            raise ValueError("No data loaded")

        return GameBatch(
            player1=self._player1,
            player2=self._player2,
            scores=self._scores,
            day=-1,
        )

    def get_day(self, day: int) -> GameBatch:
        """Get all games for a specific day."""
        if self._day_indices is None:
            raise ValueError("No data loaded")

        idx = np.searchsorted(self._day_indices, day)
        if idx >= len(self._day_indices) or self._day_indices[idx] != day:
            raise ValueError(f"No games found for day {day}")

        start = self._day_offsets[idx]
        end = self._day_offsets[idx + 1]

        return GameBatch(
            player1=self._player1[start:end],
            player2=self._player2[start:end],
            scores=self._scores[start:end],
            day=day,
        )

    def get_batched_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all data as pre-batched arrays for direct Numba processing.

        Returns:
            (player1, player2, scores, day_indices, day_offsets)

        This allows rating systems to process all data without Python iteration.
        """
        return (
            self._player1,
            self._player2,
            self._scores,
            self._day_indices,
            self._day_offsets,
        )

    def __len__(self) -> int:
        return self.num_games

    def __repr__(self) -> str:
        if self._player1 is None:
            return "GameDataset(empty)"
        return (
            f"GameDataset(games={self.num_games:,}, players={self.num_players:,}, "
            f"days={self.num_days:,}, range=[{self.min_day}, {self.max_day}])"
        )
