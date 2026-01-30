"""Abstract base class for rating systems."""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional

import torch

from ..data import GameBatch, GameDataset
from ..data.types import PredictionResult
from .player_ratings import PlayerRatings


class RatingSystemType(Enum):
    """Type of rating system update mechanism."""

    ONLINE = auto()    # Can update incrementally (Elo, Glicko, Glicko-2)
    BATCH = auto()     # Must refit on all historical data each update


class RatingSystem(ABC):
    """
    Abstract base class for all rating systems.

    Subclasses must implement:
    - _initialize_ratings(): Create initial player ratings
    - _update_ratings(): Process a batch of games and update ratings
    - predict_proba(): Predict win probability for player 1

    The base class provides:
    - fit(): Fit the model on a dataset
    - update(): Incremental update with new games
    - reset(): Reset to initial state
    - get_ratings(): Get current player ratings
    """

    system_type: RatingSystemType = RatingSystemType.ONLINE

    def __init__(
        self,
        num_players: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize rating system.

        Args:
            num_players: Number of players (can be set later via fit)
            device: PyTorch device for computations
        """
        from ..utils import get_device

        self.device = device or get_device()
        self._num_players = num_players
        self._ratings: Optional[PlayerRatings] = None
        self._current_day: int = -1
        self._fitted: bool = False

        if num_players is not None:
            self._ratings = self._initialize_ratings(num_players)

    @property
    def num_players(self) -> int:
        """Number of players in the system."""
        if self._num_players is None:
            raise ValueError("Number of players not set. Call fit() first.")
        return self._num_players

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._fitted

    @property
    def current_day(self) -> int:
        """The last day processed."""
        return self._current_day

    @abstractmethod
    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """
        Create initial ratings for all players.

        Args:
            num_players: Number of players

        Returns:
            PlayerRatings with initial values
        """
        pass

    @abstractmethod
    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """
        Update ratings based on a batch of games.

        This method should modify ratings in-place.

        Args:
            batch: Batch of games to process
            ratings: Current player ratings (modified in-place)
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        player1: torch.Tensor,
        player2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict probability that player1 beats player2.

        Args:
            player1: (N,) Player 1 IDs
            player2: (N,) Player 2 IDs

        Returns:
            (N,) Probabilities that player1 wins
        """
        pass

    def fit(
        self,
        dataset: GameDataset,
        end_day: Optional[int] = None,
    ) -> "RatingSystem":
        """
        Fit the rating system on a dataset.

        Args:
            dataset: Game dataset to fit on
            end_day: Last day to include (inclusive). If None, uses all data.

        Returns:
            self (for method chaining)
        """
        # Filter dataset if end_day specified
        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        # Initialize if needed
        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        # Process all games by day
        for batch in dataset.iter_days():
            self._update_ratings(batch, self._ratings)
            self._current_day = batch.day

        self._fitted = True
        return self

    def update(self, batch: GameBatch) -> "RatingSystem":
        """
        Incrementally update ratings with a new batch of games.

        For ONLINE systems, this processes the batch directly.
        For BATCH systems, this must be overridden to refit on all data.

        Args:
            batch: New games to process

        Returns:
            self (for method chaining)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before updating. Call fit() first.")

        if self.system_type == RatingSystemType.BATCH:
            raise NotImplementedError(
                "Batch systems must override update() to refit on all historical data"
            )

        batch = batch.to(self.device)
        self._update_ratings(batch, self._ratings)
        self._current_day = batch.day
        return self

    def predict(self, batch: GameBatch) -> PredictionResult:
        """
        Predict outcomes for a batch of games.

        Args:
            batch: Games to predict

        Returns:
            PredictionResult with predicted probabilities
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before predicting. Call fit() first.")

        batch = batch.to(self.device)
        proba = self.predict_proba(batch.player1, batch.player2)

        return PredictionResult(
            player1=batch.player1,
            player2=batch.player2,
            predicted_proba=proba,
            actual_scores=batch.scores,
        )

    def reset(self) -> "RatingSystem":
        """
        Reset the rating system to initial state.

        Returns:
            self (for method chaining)
        """
        if self._num_players is not None:
            self._ratings = self._initialize_ratings(self._num_players)
        self._current_day = -1
        self._fitted = False
        return self

    def get_ratings(self) -> PlayerRatings:
        """
        Get current player ratings.

        Returns:
            PlayerRatings object with current ratings
        """
        if self._ratings is None:
            raise ValueError("No ratings available. Call fit() first.")
        return self._ratings.clone()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return f"{self.__class__.__name__}(players={players}, {status})"
