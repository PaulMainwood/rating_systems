"""
G-Elo (Generalized Elo) rating system with ordinal margin-of-victory.

Extends standard Elo to use match scoreline (set counts) via a cumulative
logistic model. For J ordinal outcome categories:

    P(MOV ≤ j | Δ) = σ(θ_j - Δ)

The update has the SAME FORM as standard Elo but with generalized scores:
    Δr = K × (S_gen - E_gen)

where S_gen is the observed ordinal level and E_gen is its expectation.

For tennis best-of-3 (J=4 categories: {0-2, 1-2, 2-1, 2-0}):
    thresholds = [-θ2, -θ1, θ1, θ2]  (symmetric around 0... but we only
    need J-1=3 thresholds)

The prediction for win probability uses the standard Elo sigmoid (the
ordinal model collapses to binary Elo for the win/loss boundary when
thresholds are symmetric).

Accepts set counts as `margins` array (ordinal categories) alongside
the binary scores. When margins aren't provided, falls back to standard
Elo behaviour (margin categories inferred from binary score only).

Reference: Szczecinski, "G-Elo: Generalization of the Elo Algorithm by
Modeling the Discretized Margin of Victory", JQAS 2022.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

from ...base import PlayerRatings, RatingSystem, RatingSystemType
from ...data import GameBatch, GameDataset
from ._numba_core import (
    update_ratings_sequential,
    predict_proba_batch,
    predict_single,
    fit_all_days,
    scores_to_margins_bo3,
)


# Default thresholds for tennis.
# For Bo3 (J=4): 3 thresholds separating {0-2, 1-2, 2-1, 2-0}
# These should be symmetric: θ = [-d2, 0, d2] where d2 controls the
# spacing between margin categories.
# With scale=400, θ1=0 is the win/loss boundary, θ0 and θ2 control
# how much extra rating advantage is needed for a straight-set win.
DEFAULT_THRESHOLDS_BO3 = np.array([-0.5, 0.0, 0.5], dtype=np.float64)


@dataclass
class GEloConfig:
    """Configuration for G-Elo rating system."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0
    # Ordinal thresholds (J-1 values for J categories).
    # Default: symmetric 3-threshold for Bo3 tennis.
    # The middle threshold (0.0) is the win/loss boundary.
    # Outer thresholds control margin sensitivity.
    threshold_spread: float = 0.5  # distance from center to outer thresholds


class GElo(RatingSystem):
    """
    G-Elo rating system with Numba acceleration.

    Extends standard Elo with ordinal margin-of-victory modelling.
    Uses set score differences in tennis to provide richer signal than
    binary win/loss.

    When no margin data is provided, automatically infers margins from
    the binary score (defaulting to a middle-margin win/loss). When
    Player1Sets/Player2Sets columns are present in the parquet, full
    ordinal margins are used.

    The prediction model is identical to standard Elo — G-Elo only
    changes how updates are scaled by margin. With symmetric thresholds,
    the win probability prediction is the standard logistic.

    Parameters:
        initial_rating: Starting rating (default: 1500)
        k_factor: Maximum rating change per game (default: 32)
        scale: Logistic scale (default: 400)
        threshold_spread: Distance between ordinal thresholds (default: 0.5)
    """

    system_type = RatingSystemType.ONLINE

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        scale: float = 400.0,
        threshold_spread: float = 0.5,
        num_players: Optional[int] = None,
    ):
        self.config = GEloConfig(
            initial_rating=initial_rating,
            k_factor=k_factor,
            scale=scale,
            threshold_spread=threshold_spread,
        )
        # Build symmetric thresholds for Bo3: [-spread, 0, spread]
        self._thresholds = np.array(
            [-threshold_spread, 0.0, threshold_spread], dtype=np.float64)
        self._margins: Optional[np.ndarray] = None
        self._num_games_fitted = 0
        self._player_names: Optional[Dict[int, str]] = None
        super().__init__(num_players=num_players)

    def _initialize_ratings(self, num_players: int) -> PlayerRatings:
        """Create initial G-Elo ratings."""
        return PlayerRatings(
            ratings=np.full(num_players, self.config.initial_rating, dtype=np.float64),
            metadata={"system": "gelo", "config": self.config},
        )

    def _update_ratings(self, batch: GameBatch, ratings: PlayerRatings) -> None:
        """Update ratings with a batch of games using stored margins."""
        if len(batch) == 0:
            return

        # Get margins for this batch
        start = self._num_games_fitted
        end = start + len(batch)
        if self._margins is not None and end <= len(self._margins):
            margins = self._margins[start:end]
        else:
            # Fallback: infer from binary scores (no set data)
            margins = np.where(batch.scores > 0.5, 2.0, 1.0)

        update_ratings_sequential(
            batch.player1,
            batch.player2,
            margins,
            ratings.ratings,
            self.config.k_factor,
            self.config.scale,
            self._thresholds,
        )
        self._num_games_fitted += len(batch)

    def predict_proba(
        self,
        player1: Union[int, np.ndarray, List[int]],
        player2: Union[int, np.ndarray, List[int]],
        day: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Predict P(player1 wins). Same as standard Elo (symmetric thresholds)."""
        if self._ratings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(player1, (int, np.integer)) and isinstance(player2, (int, np.integer)):
            return predict_single(
                self._ratings.ratings[int(player1)],
                self._ratings.ratings[int(player2)],
                self.config.scale,
            )

        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(p1, p2, self._ratings.ratings, self.config.scale)

    def fit(
        self,
        dataset: GameDataset,
        margins: Optional[np.ndarray] = None,
        end_day: Optional[int] = None,
        player_names: Optional[Dict[int, str]] = None,
    ) -> "GElo":
        """
        Fit G-Elo on a dataset.

        Args:
            dataset: Game dataset to fit on
            margins: Per-game ordinal margin categories (0-indexed, float64).
                For Bo3: 0=lost 0-2, 1=lost 1-2, 2=won 2-1, 3=won 2-0.
                If None, tries to read Player1Sets/Player2Sets from the
                dataset's underlying parquet, falling back to binary.
            end_day: Last day to include (inclusive)
            player_names: Optional mapping of player_id -> name

        Returns:
            self (for method chaining)
        """
        self._player_names = player_names

        if end_day is not None:
            dataset = dataset.filter_days(end_day=end_day)

        if self._num_players is None or self._num_players < dataset.num_players:
            self._num_players = dataset.num_players
            self._ratings = self._initialize_ratings(self._num_players)

        player1, player2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

        if player1 is None or len(player1) == 0:
            self._num_games_fitted = 0
            self._fitted = True
            return self

        # Compute margins
        if margins is not None:
            m = np.ascontiguousarray(margins, dtype=np.float64)
        else:
            # Default: infer from binary scores (middle-margin)
            m = np.where(scores > 0.5, 2.0, 1.0)

        self._margins = m

        # Process all days in single Numba call
        fit_all_days(
            player1,
            player2,
            m,
            day_offsets,
            self._ratings.ratings,
            self.config.k_factor,
            self.config.scale,
            self._thresholds,
        )
        self._num_games_fitted = len(player1)
        self._current_day = int(day_indices[-1]) if len(day_indices) > 0 else None

        self._fitted = True
        return self

    def reset(self) -> "GElo":
        """Reset to initial state."""
        self._margins = None
        self._num_games_fitted = 0
        return super().reset()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        players = self._num_players or "?"
        return (
            f"GElo(k_factor={self.config.k_factor}, "
            f"threshold_spread={self.config.threshold_spread}, "
            f"players={players}, {status})"
        )
