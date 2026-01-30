"""Dataset classes for loading and managing game data."""

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import pandas as pd
import torch

from .types import GameBatch


class GameDataset:
    """
    Container for game data loaded from parquet files.

    Provides methods for:
    - Loading data from parquet files
    - Filtering by day range
    - Splitting into train/test sets
    - Iterating over games by day (rating period)
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize dataset.

        Args:
            df: DataFrame with columns Player1, Player2, Score, Day
            device: PyTorch device for tensors
        """
        from ..utils import get_device

        self.device = device or get_device()
        self._df: Optional[pd.DataFrame] = None
        self._num_players: Optional[int] = None

        if df is not None:
            self._load_dataframe(df)

    def _load_dataframe(self, df: pd.DataFrame) -> None:
        """Load and validate a DataFrame."""
        required_cols = {"Player1", "Player2", "Score", "Day"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self._df = df.sort_values("Day").reset_index(drop=True)
        self._num_players = max(df["Player1"].max(), df["Player2"].max()) + 1

    @classmethod
    def from_parquet(
        cls,
        path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> "GameDataset":
        """Load dataset from a parquet file."""
        df = pd.read_parquet(path)
        return cls(df=df, device=device)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        device: Optional[torch.device] = None,
    ) -> "GameDataset":
        """Create dataset from a pandas DataFrame."""
        return cls(df=df, device=device)

    @property
    def num_players(self) -> int:
        """Number of unique players (max player ID + 1)."""
        if self._num_players is None:
            raise ValueError("No data loaded")
        return self._num_players

    @property
    def num_games(self) -> int:
        """Total number of games."""
        if self._df is None:
            return 0
        return len(self._df)

    @property
    def days(self) -> List[int]:
        """List of unique days in sorted order."""
        if self._df is None:
            return []
        return sorted(self._df["Day"].unique().tolist())

    @property
    def min_day(self) -> int:
        """First day in the dataset."""
        if self._df is None:
            raise ValueError("No data loaded")
        return int(self._df["Day"].min())

    @property
    def max_day(self) -> int:
        """Last day in the dataset."""
        if self._df is None:
            raise ValueError("No data loaded")
        return int(self._df["Day"].max())

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
        if self._df is None:
            raise ValueError("No data loaded")

        mask = pd.Series(True, index=self._df.index)
        if start_day is not None:
            mask &= self._df["Day"] >= start_day
        if end_day is not None:
            mask &= self._df["Day"] <= end_day

        new_dataset = GameDataset(device=self.device)
        new_dataset._df = self._df[mask].reset_index(drop=True)
        new_dataset._num_players = self._num_players
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
        """Iterate over games grouped by day."""
        if self._df is None:
            return

        for day in self.days:
            day_df = self._df[self._df["Day"] == day]
            batch = GameBatch(
                player1=torch.tensor(day_df["Player1"].values, dtype=torch.long),
                player2=torch.tensor(day_df["Player2"].values, dtype=torch.long),
                scores=torch.tensor(day_df["Score"].values, dtype=torch.float32),
                day=day,
            )
            yield batch.to(self.device)

    def get_all_games(self) -> GameBatch:
        """Get all games as a single batch."""
        if self._df is None:
            raise ValueError("No data loaded")

        return GameBatch(
            player1=torch.tensor(self._df["Player1"].values, dtype=torch.long, device=self.device),
            player2=torch.tensor(self._df["Player2"].values, dtype=torch.long, device=self.device),
            scores=torch.tensor(self._df["Score"].values, dtype=torch.float32, device=self.device),
            day=-1,  # Mixed days
        )

    def get_day(self, day: int) -> GameBatch:
        """Get all games for a specific day."""
        if self._df is None:
            raise ValueError("No data loaded")

        day_df = self._df[self._df["Day"] == day]
        if len(day_df) == 0:
            raise ValueError(f"No games found for day {day}")

        return GameBatch(
            player1=torch.tensor(day_df["Player1"].values, dtype=torch.long),
            player2=torch.tensor(day_df["Player2"].values, dtype=torch.long),
            scores=torch.tensor(day_df["Score"].values, dtype=torch.float32),
            day=day,
        ).to(self.device)

    def __len__(self) -> int:
        return self.num_games

    def __repr__(self) -> str:
        if self._df is None:
            return "GameDataset(empty)"
        return (
            f"GameDataset(games={self.num_games}, players={self.num_players}, "
            f"days={len(self.days)}, range=[{self.min_day}, {self.max_day}])"
        )
