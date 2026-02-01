"""
Fitted ratings objects for querying without refitting.

These classes wrap fitted rating system results and provide
rich query interfaces for analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from ..systems.elo._numba_core import (
    predict_single,
    predict_proba_batch,
    get_top_n_indices,
    get_bottom_n_indices,
    compute_all_vs_all_matrix,
    compute_expected_scores_against_field,
)


def _compute_ranks(ratings: np.ndarray) -> np.ndarray:
    """
    Compute ranks for all players efficiently in O(n log n).

    Returns array where ranks[i] = rank of player i (1 = highest).
    """
    n = len(ratings)
    # argsort descending gives indices sorted by rating high to low
    sorted_indices = np.argsort(-ratings)
    ranks = np.empty(n, dtype=np.int32)
    ranks[sorted_indices] = np.arange(1, n + 1)
    return ranks


def _get_names_vectorized(
    indices: np.ndarray,
    player_names: Optional[Dict[int, str]],
) -> List[str]:
    """Get names for indices, with fast path when no names dict."""
    if player_names is None:
        return [f"Player_{i}" for i in indices]
    return [player_names.get(int(i), f"Player_{i}") for i in indices]


@dataclass
class FittedEloRatings:
    """
    Queryable fitted Elo ratings.

    Provides efficient methods for:
    - Getting top/bottom N players
    - Predicting matchup outcomes
    - Computing head-to-head matrices
    - Exporting to various formats

    Attributes:
        ratings: Array of player ratings (index = player_id)
        scale: Elo scale parameter (default 400)
        initial_rating: Initial rating value
        k_factor: K-factor used for fitting
        num_games_fitted: Number of games used to fit
        last_day: Last day in training data
        player_names: Optional mapping of player_id -> name
    """

    ratings: np.ndarray
    scale: float = 400.0
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    num_games_fitted: int = 0
    last_day: int = -1
    player_names: Optional[Dict[int, str]] = None

    # Cached computed values (lazy)
    _ranks: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure ratings array is contiguous."""
        self.ratings = np.ascontiguousarray(self.ratings, dtype=np.float64)
        self._ranks = None  # Reset cache

    @property
    def num_players(self) -> int:
        """Number of players in the system."""
        return len(self.ratings)

    @property
    def ranks(self) -> np.ndarray:
        """Lazily computed ranks array (1 = highest rated)."""
        if self._ranks is None:
            self._ranks = _compute_ranks(self.ratings)
        return self._ranks

    def get_rating(self, player_id: int) -> float:
        """Get rating for a single player."""
        return float(self.ratings[player_id])

    def get_ratings(self, player_ids: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """Get ratings for multiple players."""
        if isinstance(player_ids, int):
            return np.array([self.ratings[player_ids]])
        return self.ratings[np.asarray(player_ids)]

    def get_name(self, player_id: int) -> str:
        """Get player name (or ID string if no names loaded)."""
        if self.player_names and player_id in self.player_names:
            return self.player_names[player_id]
        return f"Player_{player_id}"

    # =========================================================================
    # Top/Bottom Players
    # =========================================================================

    def top(self, n: int = 10) -> pl.DataFrame:
        """
        Get top N rated players.

        Returns DataFrame with columns: rank, player_id, name, rating
        """
        indices = get_top_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def bottom(self, n: int = 10) -> pl.DataFrame:
        """Get bottom N rated players."""
        indices = get_bottom_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def rank(self, player_id: int) -> int:
        """Get rank of a specific player (1 = highest rated)."""
        return int(self.ranks[player_id])

    def _indices_to_dataframe(self, indices: np.ndarray) -> pl.DataFrame:
        """Convert player indices to a formatted DataFrame."""
        return pl.DataFrame({
            "rank": self.ranks[indices],
            "player_id": indices,
            "name": _get_names_vectorized(indices, self.player_names),
            "rating": self.ratings[indices],
        })

    # =========================================================================
    # Matchup Predictions
    # =========================================================================

    def predict(self, player1: int, player2: int) -> float:
        """
        Predict probability that player1 beats player2.

        Args:
            player1: Player 1 ID
            player2: Player 2 ID

        Returns:
            Probability of player1 winning (0 to 1)
        """
        return predict_single(
            self.ratings[player1],
            self.ratings[player2],
            self.scale,
        )

    def predict_batch(
        self,
        player1: Union[List[int], np.ndarray],
        player2: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        """Predict outcomes for multiple matchups (vectorized)."""
        p1 = np.ascontiguousarray(player1, dtype=np.int64)
        p2 = np.ascontiguousarray(player2, dtype=np.int64)
        return predict_proba_batch(p1, p2, self.ratings, self.scale)

    def matchup(self, player1: int, player2: int) -> pl.DataFrame:
        """
        Get detailed matchup analysis between two players.

        Returns DataFrame with both perspectives and rating difference.
        """
        r1 = self.ratings[player1]
        r2 = self.ratings[player2]
        p1_wins = self.predict(player1, player2)

        return pl.DataFrame({
            "player_id": [player1, player2],
            "name": [self.get_name(player1), self.get_name(player2)],
            "rating": [r1, r2],
            "win_prob": [p1_wins, 1.0 - p1_wins],
            "rating_diff": [r1 - r2, r2 - r1],
        })

    def head_to_head_matrix(
        self,
        player_ids: Optional[Union[List[int], np.ndarray]] = None,
        top_n: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Compute head-to-head win probability matrix.

        Args:
            player_ids: Specific players to include (default: top_n or all)
            top_n: Use top N players (ignored if player_ids provided)

        Returns:
            DataFrame with win probabilities (row player beats column player)
        """
        if player_ids is not None:
            indices = np.asarray(player_ids, dtype=np.int64)
        elif top_n is not None:
            indices = get_top_n_indices(self.ratings, top_n)
        else:
            indices = np.arange(self.num_players, dtype=np.int64)

        matrix = compute_all_vs_all_matrix(self.ratings, indices, self.scale)
        names = _get_names_vectorized(indices, self.player_names)

        # Build DataFrame with player name as first column, then probability columns
        data = {"player": names}
        for i, name in enumerate(names):
            data[name] = matrix[i, :]
        return pl.DataFrame(data)

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def rating_distribution(self, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Get histogram of rating distribution."""
        counts, edges = np.histogram(self.ratings, bins=bins)
        return counts, edges

    def percentile(self, player_id: int) -> float:
        """Get percentile rank of a player (100 = highest)."""
        return 100.0 * (1.0 - (self.ranks[player_id] - 1) / self.num_players)

    def players_in_range(
        self,
        min_rating: float,
        max_rating: float,
    ) -> pl.DataFrame:
        """Get all players within a rating range."""
        mask = (self.ratings >= min_rating) & (self.ratings <= max_rating)
        indices = np.where(mask)[0]
        # Sort by rating descending
        sorted_idx = indices[np.argsort(-self.ratings[indices])]
        return self._indices_to_dataframe(sorted_idx)

    def expected_score_vs_field(
        self,
        player_id: int,
        field: Optional[Union[List[int], np.ndarray]] = None,
    ) -> float:
        """
        Compute expected score for a player against a field.

        Args:
            player_id: The player to evaluate
            field: Opponents to evaluate against (default: all other players)

        Returns:
            Expected score (0 to 1, where 0.5 = even)
        """
        if field is None:
            # Efficiently create field excluding player_id
            field = np.concatenate([
                np.arange(player_id, dtype=np.int64),
                np.arange(player_id + 1, self.num_players, dtype=np.int64)
            ])
        else:
            field = np.asarray(field, dtype=np.int64)

        expected = compute_expected_scores_against_field(self.ratings, field)
        return float(expected[player_id])

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_dataframe(self, include_rank: bool = True) -> pl.DataFrame:
        """
        Export all ratings to a DataFrame.

        Args:
            include_rank: Whether to include rank column (default True)
        """
        player_ids = np.arange(self.num_players)

        data = {
            "player_id": player_ids,
            "rating": self.ratings,
        }

        if include_rank:
            data["rank"] = self.ranks

        # Add names if available (vectorized)
        if self.player_names:
            data["name"] = _get_names_vectorized(player_ids, self.player_names)

        df = pl.DataFrame(data)
        return df.sort("rating", descending=True)

    def to_dict(self) -> Dict:
        """Export to dictionary format (metadata only, no large arrays)."""
        return {
            "scale": self.scale,
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "num_games_fitted": self.num_games_fitted,
            "last_day": self.last_day,
            "num_players": self.num_players,
        }

    def save(self, path: str, include_rank: bool = False) -> None:
        """
        Save fitted ratings to a parquet file.

        Args:
            path: Output file path
            include_rank: Whether to include rank column (slower for large datasets)
        """
        df = self.to_dataframe(include_rank=include_rank)
        df.write_parquet(path)

    def save_compact(self, path: str) -> None:
        """Save just the ratings array (most efficient)."""
        np.save(path, self.ratings)

    @classmethod
    def load(cls, path: str, scale: float = 400.0) -> "FittedEloRatings":
        """Load fitted ratings from a parquet file."""
        df = pl.read_parquet(path)
        ratings = df["rating"].to_numpy()
        names = None
        if "name" in df.columns:
            names = dict(zip(df["player_id"].to_list(), df["name"].to_list()))
        return cls(ratings=ratings, scale=scale, player_names=names)

    @classmethod
    def load_compact(cls, path: str, scale: float = 400.0) -> "FittedEloRatings":
        """Load from compact numpy format."""
        ratings = np.load(path)
        return cls(ratings=ratings, scale=scale)

    def __repr__(self) -> str:
        return (
            f"FittedEloRatings(players={self.num_players}, "
            f"games={self.num_games_fitted}, k={self.k_factor})"
        )

    def __str__(self) -> str:
        """Pretty string representation."""
        lines = [
            f"Fitted Elo Ratings",
            f"  Players: {self.num_players:,}",
            f"  Games fitted: {self.num_games_fitted:,}",
            f"  K-factor: {self.k_factor}",
            f"  Rating range: {self.ratings.min():.1f} - {self.ratings.max():.1f}",
            f"  Mean rating: {self.ratings.mean():.1f}",
        ]
        return "\n".join(lines)


@dataclass
class FittedGlickoRatings:
    """
    Queryable fitted Glicko ratings.

    Similar to FittedEloRatings but includes Rating Deviation (RD).
    """

    ratings: np.ndarray
    rd: np.ndarray
    last_played: Optional[np.ndarray] = None
    q: float = 0.00575646273  # ln(10)/400
    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    num_games_fitted: int = 0
    last_day: int = -1
    player_names: Optional[Dict[int, str]] = None

    # Cached
    _ranks: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure arrays are contiguous."""
        self.ratings = np.ascontiguousarray(self.ratings, dtype=np.float64)
        self.rd = np.ascontiguousarray(self.rd, dtype=np.float64)
        if self.last_played is not None:
            self.last_played = np.ascontiguousarray(self.last_played, dtype=np.int32)
        self._ranks = None

    @property
    def num_players(self) -> int:
        return len(self.ratings)

    @property
    def ranks(self) -> np.ndarray:
        """Lazily computed ranks array."""
        if self._ranks is None:
            self._ranks = _compute_ranks(self.ratings)
        return self._ranks

    def get_rating(self, player_id: int) -> Tuple[float, float]:
        """Get (rating, rd) for a single player."""
        return float(self.ratings[player_id]), float(self.rd[player_id])

    def get_name(self, player_id: int) -> str:
        if self.player_names and player_id in self.player_names:
            return self.player_names[player_id]
        return f"Player_{player_id}"

    def _g(self, rd: float) -> float:
        """Glicko g function."""
        return 1.0 / np.sqrt(1.0 + 3.0 * (self.q ** 2) * (rd ** 2) / (np.pi ** 2))

    def predict(self, player1: int, player2: int) -> float:
        """Predict probability that player1 beats player2."""
        r1, rd1 = self.get_rating(player1)
        r2, rd2 = self.get_rating(player2)

        combined_rd = np.sqrt(rd1 ** 2 + rd2 ** 2)
        g = self._g(combined_rd)
        exponent = -g * (r1 - r2) / 400.0
        return 1.0 / (1.0 + 10.0 ** exponent)

    def top(self, n: int = 10) -> pl.DataFrame:
        """Get top N rated players with RD."""
        indices = get_top_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def bottom(self, n: int = 10) -> pl.DataFrame:
        """Get bottom N rated players."""
        indices = get_bottom_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def rank(self, player_id: int) -> int:
        """Get rank of a specific player."""
        return int(self.ranks[player_id])

    def _indices_to_dataframe(self, indices: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({
            "rank": self.ranks[indices],
            "player_id": indices,
            "name": _get_names_vectorized(indices, self.player_names),
            "rating": self.ratings[indices],
            "rd": self.rd[indices],
        })

    def matchup(self, player1: int, player2: int) -> pl.DataFrame:
        """Get detailed matchup analysis between two players."""
        r1, rd1 = self.get_rating(player1)
        r2, rd2 = self.get_rating(player2)
        p1_wins = self.predict(player1, player2)

        return pl.DataFrame({
            "player_id": [player1, player2],
            "name": [self.get_name(player1), self.get_name(player2)],
            "rating": [r1, r2],
            "rd": [rd1, rd2],
            "win_prob": [p1_wins, 1.0 - p1_wins],
        })

    def confident_players(self, max_rd: float = 100.0, n: int = 10) -> pl.DataFrame:
        """Get top players with low RD (well-established ratings)."""
        mask = self.rd <= max_rd
        filtered_ratings = np.where(mask, self.ratings, -np.inf)
        indices = get_top_n_indices(filtered_ratings, n)
        indices = indices[mask[indices]]
        return self._indices_to_dataframe(indices)

    def to_dataframe(self, include_rank: bool = True) -> pl.DataFrame:
        """Export all ratings to DataFrame."""
        player_ids = np.arange(self.num_players)

        data = {
            "player_id": player_ids,
            "rating": self.ratings,
            "rd": self.rd,
        }

        if include_rank:
            data["rank"] = self.ranks

        if self.player_names:
            data["name"] = _get_names_vectorized(player_ids, self.player_names)

        if self.last_played is not None:
            data["last_played"] = self.last_played

        return pl.DataFrame(data).sort("rating", descending=True)

    def save(self, path: str, include_rank: bool = False) -> None:
        """Save fitted ratings to parquet."""
        self.to_dataframe(include_rank=include_rank).write_parquet(path)

    def __repr__(self) -> str:
        return (
            f"FittedGlickoRatings(players={self.num_players}, "
            f"games={self.num_games_fitted})"
        )

    def __str__(self) -> str:
        lines = [
            f"Fitted Glicko Ratings",
            f"  Players: {self.num_players:,}",
            f"  Games fitted: {self.num_games_fitted:,}",
            f"  Rating range: {self.ratings.min():.1f} - {self.ratings.max():.1f}",
            f"  Mean RD: {self.rd.mean():.1f}",
        ]
        return "\n".join(lines)


@dataclass
class FittedGlicko2Ratings:
    """
    Queryable fitted Glicko-2 ratings.

    Similar to FittedGlickoRatings but includes volatility parameter.
    """

    ratings: np.ndarray
    rd: np.ndarray
    volatility: np.ndarray
    last_played: Optional[np.ndarray] = None
    scale: float = 173.7178  # Glicko-2 scale factor
    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    initial_volatility: float = 0.06
    tau: float = 0.5
    num_games_fitted: int = 0
    last_day: int = -1
    player_names: Optional[Dict[int, str]] = None

    # Cached
    _ranks: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure arrays are contiguous."""
        self.ratings = np.ascontiguousarray(self.ratings, dtype=np.float64)
        self.rd = np.ascontiguousarray(self.rd, dtype=np.float64)
        self.volatility = np.ascontiguousarray(self.volatility, dtype=np.float64)
        if self.last_played is not None:
            self.last_played = np.ascontiguousarray(self.last_played, dtype=np.int32)
        self._ranks = None

    @property
    def num_players(self) -> int:
        return len(self.ratings)

    @property
    def ranks(self) -> np.ndarray:
        """Lazily computed ranks array."""
        if self._ranks is None:
            self._ranks = _compute_ranks(self.ratings)
        return self._ranks

    def get_rating(self, player_id: int) -> Tuple[float, float, float]:
        """Get (rating, rd, volatility) for a single player."""
        return (
            float(self.ratings[player_id]),
            float(self.rd[player_id]),
            float(self.volatility[player_id]),
        )

    def get_name(self, player_id: int) -> str:
        if self.player_names and player_id in self.player_names:
            return self.player_names[player_id]
        return f"Player_{player_id}"

    def _g(self, phi: float) -> float:
        """Glicko-2 g function (using Glicko-scale RD)."""
        # Convert RD to phi (Glicko-2 scale)
        phi_g2 = phi / self.scale
        return 1.0 / np.sqrt(1.0 + 3.0 * (phi_g2 ** 2) / (np.pi ** 2))

    def predict(self, player1: int, player2: int) -> float:
        """Predict probability that player1 beats player2."""
        r1, rd1, _ = self.get_rating(player1)
        r2, rd2, _ = self.get_rating(player2)

        # Convert to Glicko-2 scale for calculation
        mu1 = (r1 - self.initial_rating) / self.scale
        mu2 = (r2 - self.initial_rating) / self.scale
        phi1 = rd1 / self.scale
        phi2 = rd2 / self.scale

        combined_phi = np.sqrt(phi1 ** 2 + phi2 ** 2)
        g_combined = 1.0 / np.sqrt(1.0 + 3.0 * (combined_phi ** 2) / (np.pi ** 2))

        return 1.0 / (1.0 + np.exp(-g_combined * (mu1 - mu2)))

    def top(self, n: int = 10) -> pl.DataFrame:
        """Get top N rated players with RD and volatility."""
        indices = get_top_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def bottom(self, n: int = 10) -> pl.DataFrame:
        """Get bottom N rated players."""
        indices = get_bottom_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def rank(self, player_id: int) -> int:
        """Get rank of a specific player."""
        return int(self.ranks[player_id])

    def _indices_to_dataframe(self, indices: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({
            "rank": self.ranks[indices],
            "player_id": indices,
            "name": _get_names_vectorized(indices, self.player_names),
            "rating": self.ratings[indices],
            "rd": self.rd[indices],
            "volatility": self.volatility[indices],
        })

    def matchup(self, player1: int, player2: int) -> pl.DataFrame:
        """Get detailed matchup analysis between two players."""
        r1, rd1, vol1 = self.get_rating(player1)
        r2, rd2, vol2 = self.get_rating(player2)
        p1_wins = self.predict(player1, player2)

        return pl.DataFrame({
            "player_id": [player1, player2],
            "name": [self.get_name(player1), self.get_name(player2)],
            "rating": [r1, r2],
            "rd": [rd1, rd2],
            "volatility": [vol1, vol2],
            "win_prob": [p1_wins, 1.0 - p1_wins],
        })

    def confident_players(self, max_rd: float = 100.0, n: int = 10) -> pl.DataFrame:
        """Get top players with low RD (well-established ratings)."""
        mask = self.rd <= max_rd
        filtered_ratings = np.where(mask, self.ratings, -np.inf)
        indices = get_top_n_indices(filtered_ratings, n)
        indices = indices[mask[indices]]
        return self._indices_to_dataframe(indices)

    def stable_players(self, max_volatility: float = 0.05, n: int = 10) -> pl.DataFrame:
        """Get top players with low volatility (stable ratings)."""
        mask = self.volatility <= max_volatility
        filtered_ratings = np.where(mask, self.ratings, -np.inf)
        indices = get_top_n_indices(filtered_ratings, n)
        indices = indices[mask[indices]]
        return self._indices_to_dataframe(indices)

    def to_dataframe(self, include_rank: bool = True) -> pl.DataFrame:
        """Export all ratings to DataFrame."""
        player_ids = np.arange(self.num_players)

        data = {
            "player_id": player_ids,
            "rating": self.ratings,
            "rd": self.rd,
            "volatility": self.volatility,
        }

        if include_rank:
            data["rank"] = self.ranks

        if self.player_names:
            data["name"] = _get_names_vectorized(player_ids, self.player_names)

        if self.last_played is not None:
            data["last_played"] = self.last_played

        return pl.DataFrame(data).sort("rating", descending=True)

    def save(self, path: str, include_rank: bool = False) -> None:
        """Save fitted ratings to parquet."""
        self.to_dataframe(include_rank=include_rank).write_parquet(path)

    def __repr__(self) -> str:
        return (
            f"FittedGlicko2Ratings(players={self.num_players}, "
            f"games={self.num_games_fitted}, tau={self.tau})"
        )

    def __str__(self) -> str:
        lines = [
            f"Fitted Glicko-2 Ratings",
            f"  Players: {self.num_players:,}",
            f"  Games fitted: {self.num_games_fitted:,}",
            f"  Tau: {self.tau}",
            f"  Rating range: {self.ratings.min():.1f} - {self.ratings.max():.1f}",
            f"  Mean RD: {self.rd.mean():.1f}",
            f"  Mean volatility: {self.volatility.mean():.4f}",
        ]
        return "\n".join(lines)


@dataclass
class FittedWHRRatings:
    """
    Queryable fitted WHR (Whole History Rating) ratings.

    Provides access to:
    - Current ratings and uncertainties for all players
    - Full rating history over time for each player
    - Prediction methods
    - Top/bottom player queries

    WHR is unique in that it maintains a complete rating history,
    not just the current rating. This allows querying a player's
    rating at any point in time.
    """

    ratings: np.ndarray  # Current (most recent) rating per player
    rd: np.ndarray  # Current uncertainty per player
    w2: float = 300.0  # Wiener variance parameter
    initial_rating: float = 1500.0
    num_games_fitted: int = 0
    num_iterations: int = 0
    last_day: Optional[int] = None
    player_names: Optional[Dict[int, str]] = None
    rating_history: Optional[Dict[int, Dict]] = None  # player_id -> {days, ratings, uncertainties}

    # Cached
    _ranks: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure arrays are contiguous."""
        self.ratings = np.ascontiguousarray(self.ratings, dtype=np.float64)
        self.rd = np.ascontiguousarray(self.rd, dtype=np.float64)
        self._ranks = None

    @property
    def num_players(self) -> int:
        return len(self.ratings)

    @property
    def ranks(self) -> np.ndarray:
        """Lazily computed ranks array."""
        if self._ranks is None:
            self._ranks = _compute_ranks(self.ratings)
        return self._ranks

    def get_rating(self, player_id: int) -> Tuple[float, float]:
        """Get (rating, uncertainty) for a single player."""
        return float(self.ratings[player_id]), float(self.rd[player_id])

    def get_name(self, player_id: int) -> str:
        if self.player_names and player_id in self.player_names:
            return self.player_names[player_id]
        return f"Player_{player_id}"

    def get_history(self, player_id: int) -> Optional[Dict]:
        """
        Get the full rating history for a player.

        Returns:
            Dict with 'days', 'ratings', 'uncertainties' arrays,
            or None if player has no history.
        """
        if self.rating_history is None:
            return None
        return self.rating_history.get(player_id)

    def get_rating_at_day(self, player_id: int, day: int) -> Optional[float]:
        """
        Get a player's rating at a specific day.

        Returns the rating from the most recent game on or before the given day,
        or None if the player has no games before that day.
        """
        history = self.get_history(player_id)
        if history is None:
            return None

        days = history["days"]
        ratings = history["ratings"]

        # Find most recent day <= target day
        for i in range(len(days) - 1, -1, -1):
            if days[i] <= day:
                return float(ratings[i])

        return None

    def predict(self, player1: int, player2: int) -> float:
        """Predict probability that player1 beats player2."""
        import math
        LN10_400 = math.log(10) / 400.0

        r1 = (self.ratings[player1] - self.initial_rating) * LN10_400
        r2 = (self.ratings[player2] - self.initial_rating) * LN10_400

        diff = r1 - r2
        if diff > 20:
            return 1.0 - 1e-9
        elif diff < -20:
            return 1e-9
        return 1.0 / (1.0 + math.exp(-diff))

    def predict_batch(
        self,
        player1: Union[List[int], np.ndarray],
        player2: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        """Predict outcomes for multiple matchups."""
        import math
        LN10_400 = math.log(10) / 400.0

        p1 = np.asarray(player1)
        p2 = np.asarray(player2)

        r1 = (self.ratings[p1] - self.initial_rating) * LN10_400
        r2 = (self.ratings[p2] - self.initial_rating) * LN10_400

        diff = r1 - r2
        return 1.0 / (1.0 + np.exp(-np.clip(diff, -20, 20)))

    def top(self, n: int = 10) -> pl.DataFrame:
        """Get top N rated players."""
        indices = get_top_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def bottom(self, n: int = 10) -> pl.DataFrame:
        """Get bottom N rated players."""
        indices = get_bottom_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def rank(self, player_id: int) -> int:
        """Get rank of a specific player."""
        return int(self.ranks[player_id])

    def _indices_to_dataframe(self, indices: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({
            "rank": self.ranks[indices],
            "player_id": indices,
            "name": _get_names_vectorized(indices, self.player_names),
            "rating": self.ratings[indices],
            "rd": self.rd[indices],
        })

    def matchup(self, player1: int, player2: int) -> pl.DataFrame:
        """Get detailed matchup analysis between two players."""
        r1, rd1 = self.get_rating(player1)
        r2, rd2 = self.get_rating(player2)
        p1_wins = self.predict(player1, player2)

        return pl.DataFrame({
            "player_id": [player1, player2],
            "name": [self.get_name(player1), self.get_name(player2)],
            "rating": [r1, r2],
            "rd": [rd1, rd2],
            "win_prob": [p1_wins, 1.0 - p1_wins],
        })

    def confident_players(self, max_rd: float = 100.0, n: int = 10) -> pl.DataFrame:
        """Get top players with low RD (well-established ratings)."""
        mask = self.rd <= max_rd
        filtered_ratings = np.where(mask, self.ratings, -np.inf)
        indices = get_top_n_indices(filtered_ratings, n)
        indices = indices[mask[indices]]
        return self._indices_to_dataframe(indices)

    def active_players(self, min_days: int = 5, n: int = 10) -> pl.DataFrame:
        """Get top players with at least min_days of activity."""
        if self.rating_history is None:
            return self.top(n)

        mask = np.array([
            len(self.rating_history.get(i, {}).get("days", [])) >= min_days
            for i in range(self.num_players)
        ])
        filtered_ratings = np.where(mask, self.ratings, -np.inf)
        indices = get_top_n_indices(filtered_ratings, n)
        indices = indices[mask[indices]]
        return self._indices_to_dataframe(indices)

    def to_dataframe(self, include_rank: bool = True) -> pl.DataFrame:
        """Export all ratings to DataFrame."""
        player_ids = np.arange(self.num_players)

        data = {
            "player_id": player_ids,
            "rating": self.ratings,
            "rd": self.rd,
        }

        if include_rank:
            data["rank"] = self.ranks

        if self.player_names:
            data["name"] = _get_names_vectorized(player_ids, self.player_names)

        return pl.DataFrame(data).sort("rating", descending=True)

    def history_to_dataframe(self, player_id: int) -> Optional[pl.DataFrame]:
        """Export a player's rating history to DataFrame."""
        history = self.get_history(player_id)
        if history is None:
            return None

        return pl.DataFrame({
            "day": history["days"],
            "rating": history["ratings"],
            "uncertainty": history["uncertainties"],
        })

    def save(self, path: str, include_rank: bool = False) -> None:
        """Save fitted ratings to parquet."""
        self.to_dataframe(include_rank=include_rank).write_parquet(path)

    def __repr__(self) -> str:
        return (
            f"FittedWHRRatings(players={self.num_players}, "
            f"games={self.num_games_fitted}, w2={self.w2}, "
            f"iterations={self.num_iterations})"
        )

    def __str__(self) -> str:
        lines = [
            f"Fitted WHR Ratings",
            f"  Players: {self.num_players:,}",
            f"  Games fitted: {self.num_games_fitted:,}",
            f"  W2 (Wiener variance): {self.w2}",
            f"  Iterations: {self.num_iterations}",
            f"  Rating range: {self.ratings.min():.1f} - {self.ratings.max():.1f}",
            f"  Mean RD: {self.rd.mean():.1f}",
        ]
        return "\n".join(lines)


@dataclass
class FittedTTTRatings:
    """
    Queryable fitted TrueSkill Through Time (TTT) ratings.

    Provides access to:
    - Current ratings and uncertainties for all players
    - Full rating history over time for each player
    - Prediction methods accounting for uncertainty
    - Top/bottom player queries

    TTT models skill as evolving over time with Gaussian belief propagation,
    providing globally consistent historical ratings.
    """

    ratings: np.ndarray  # Current (most recent) rating per player
    rd: np.ndarray  # Current uncertainty per player
    sigma: float = 1.5  # Prior skill std dev
    beta: float = 0.5  # Performance variability
    gamma: float = 0.01  # Skill drift rate
    display_scale: float = 266.67  # Scale factor for display
    display_offset: float = 1500.0  # Offset for display
    num_games_fitted: int = 0
    num_iterations: int = 0
    last_day: Optional[int] = None
    player_names: Optional[Dict[int, str]] = None
    rating_history: Optional[Dict[int, Dict]] = None  # player_id -> {days, ratings, uncertainties}

    # Cached
    _ranks: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure arrays are contiguous."""
        self.ratings = np.ascontiguousarray(self.ratings, dtype=np.float64)
        self.rd = np.ascontiguousarray(self.rd, dtype=np.float64)
        self._ranks = None

    @property
    def num_players(self) -> int:
        return len(self.ratings)

    @property
    def ranks(self) -> np.ndarray:
        """Lazily computed ranks array."""
        if self._ranks is None:
            self._ranks = _compute_ranks(self.ratings)
        return self._ranks

    def get_rating(self, player_id: int) -> Tuple[float, float]:
        """Get (rating, uncertainty) for a single player."""
        return float(self.ratings[player_id]), float(self.rd[player_id])

    def get_name(self, player_id: int) -> str:
        if self.player_names and player_id in self.player_names:
            return self.player_names[player_id]
        return f"Player_{player_id}"

    def get_history(self, player_id: int) -> Optional[Dict]:
        """
        Get the full rating history for a player.

        Returns:
            Dict with 'days', 'ratings', 'uncertainties' arrays,
            or None if player has no history.
        """
        if self.rating_history is None:
            return None
        return self.rating_history.get(player_id)

    def get_rating_at_day(self, player_id: int, day: int) -> Optional[float]:
        """
        Get a player's rating at a specific day.

        Returns the rating from the most recent game on or before the given day,
        or None if the player has no games before that day.
        """
        history = self.get_history(player_id)
        if history is None:
            return None

        days = history["days"]
        ratings = history["ratings"]

        # Find most recent day <= target day
        for i in range(len(days) - 1, -1, -1):
            if days[i] <= day:
                return float(ratings[i])

        return None

    def predict(self, player1: int, player2: int) -> float:
        """
        Predict probability that player1 beats player2.

        Uses the TrueSkill prediction formula accounting for
        both skill and uncertainty.
        """
        import math

        r1 = (self.ratings[player1] - self.display_offset) / self.display_scale
        r2 = (self.ratings[player2] - self.display_offset) / self.display_scale
        rd1 = self.rd[player1] / self.display_scale
        rd2 = self.rd[player2] / self.display_scale

        # Combined variance: 2*beta^2 + sigma1^2 + sigma2^2
        c = math.sqrt(2.0 * self.beta ** 2 + rd1 ** 2 + rd2 ** 2)

        diff = (r1 - r2) / c
        # Use logistic approximation of probit
        return 1.0 / (1.0 + math.exp(-1.7 * diff))

    def predict_batch(
        self,
        player1: Union[List[int], np.ndarray],
        player2: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        """Predict outcomes for multiple matchups."""
        p1 = np.asarray(player1)
        p2 = np.asarray(player2)

        r1 = (self.ratings[p1] - self.display_offset) / self.display_scale
        r2 = (self.ratings[p2] - self.display_offset) / self.display_scale
        rd1 = self.rd[p1] / self.display_scale
        rd2 = self.rd[p2] / self.display_scale

        c = np.sqrt(2.0 * self.beta ** 2 + rd1 ** 2 + rd2 ** 2)
        diff = (r1 - r2) / c

        return 1.0 / (1.0 + np.exp(-1.7 * np.clip(diff, -20, 20)))

    def top(self, n: int = 10) -> pl.DataFrame:
        """Get top N rated players."""
        indices = get_top_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def bottom(self, n: int = 10) -> pl.DataFrame:
        """Get bottom N rated players."""
        indices = get_bottom_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def rank(self, player_id: int) -> int:
        """Get rank of a specific player."""
        return int(self.ranks[player_id])

    def _indices_to_dataframe(self, indices: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({
            "rank": self.ranks[indices],
            "player_id": indices,
            "name": _get_names_vectorized(indices, self.player_names),
            "rating": self.ratings[indices],
            "rd": self.rd[indices],
        })

    def matchup(self, player1: int, player2: int) -> pl.DataFrame:
        """Get detailed matchup analysis between two players."""
        r1, rd1 = self.get_rating(player1)
        r2, rd2 = self.get_rating(player2)
        p1_wins = self.predict(player1, player2)

        return pl.DataFrame({
            "player_id": [player1, player2],
            "name": [self.get_name(player1), self.get_name(player2)],
            "rating": [r1, r2],
            "rd": [rd1, rd2],
            "win_prob": [p1_wins, 1.0 - p1_wins],
        })

    def confident_players(self, max_rd: float = 100.0, n: int = 10) -> pl.DataFrame:
        """Get top players with low uncertainty (well-established ratings)."""
        mask = self.rd <= max_rd
        filtered_ratings = np.where(mask, self.ratings, -np.inf)
        indices = get_top_n_indices(filtered_ratings, n)
        indices = indices[mask[indices]]
        return self._indices_to_dataframe(indices)

    def active_players(self, min_days: int = 5, n: int = 10) -> pl.DataFrame:
        """Get top players with at least min_days of activity."""
        if self.rating_history is None:
            return self.top(n)

        mask = np.array([
            len(self.rating_history.get(i, {}).get("days", [])) >= min_days
            for i in range(self.num_players)
        ])
        filtered_ratings = np.where(mask, self.ratings, -np.inf)
        indices = get_top_n_indices(filtered_ratings, n)
        indices = indices[mask[indices]]
        return self._indices_to_dataframe(indices)

    def to_dataframe(self, include_rank: bool = True) -> pl.DataFrame:
        """Export all ratings to DataFrame."""
        player_ids = np.arange(self.num_players)

        data = {
            "player_id": player_ids,
            "rating": self.ratings,
            "rd": self.rd,
        }

        if include_rank:
            data["rank"] = self.ranks

        if self.player_names:
            data["name"] = _get_names_vectorized(player_ids, self.player_names)

        return pl.DataFrame(data).sort("rating", descending=True)

    def history_to_dataframe(self, player_id: int) -> Optional[pl.DataFrame]:
        """Export a player's rating history to DataFrame."""
        history = self.get_history(player_id)
        if history is None:
            return None

        return pl.DataFrame({
            "day": history["days"],
            "rating": history["ratings"],
            "uncertainty": history["uncertainties"],
        })

    def save(self, path: str, include_rank: bool = False) -> None:
        """Save fitted ratings to parquet."""
        self.to_dataframe(include_rank=include_rank).write_parquet(path)

    def __repr__(self) -> str:
        return (
            f"FittedTTTRatings(players={self.num_players}, "
            f"games={self.num_games_fitted}, beta={self.beta}, "
            f"iterations={self.num_iterations})"
        )

    def __str__(self) -> str:
        lines = [
            f"Fitted TrueSkill Through Time Ratings",
            f"  Players: {self.num_players:,}",
            f"  Games fitted: {self.num_games_fitted:,}",
            f"  Sigma: {self.sigma}",
            f"  Beta: {self.beta}",
            f"  Gamma: {self.gamma}",
            f"  Iterations: {self.num_iterations}",
            f"  Rating range: {self.ratings.min():.1f} - {self.ratings.max():.1f}",
            f"  Mean RD: {self.rd.mean():.1f}",
        ]
        return "\n".join(lines)


@dataclass
class FittedTrueSkillRatings:
    """
    Queryable fitted TrueSkill ratings.

    TrueSkill uses Gaussian skill beliefs N(mu, sigma^2) for each player.
    This class provides both internal scale values (mu, sigma) and
    display-scale values for Elo-like readability.

    Conservative rating (mu - k*sigma) represents a lower bound on skill
    and is useful for ranking when you want high confidence.

    Attributes:
        mu: Array of skill means (internal scale)
        sigma: Array of skill uncertainties (internal scale)
        beta: Performance variability parameter
        display_scale: Scale factor for Elo-like display
        display_offset: Offset for Elo-like display
    """

    mu: np.ndarray  # Internal scale skill means
    sigma: np.ndarray  # Internal scale uncertainties
    beta: float = 4.166666667
    initial_mu: float = 25.0
    initial_sigma: float = 8.333333333
    display_scale: float = 133.333333  # 400/3
    display_offset: float = 1500.0
    num_games_fitted: int = 0
    last_day: int = -1
    player_names: Optional[Dict[int, str]] = None

    # Cached
    _ranks: Optional[np.ndarray] = field(default=None, repr=False)
    _conservative_ranks: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure arrays are contiguous."""
        self.mu = np.ascontiguousarray(self.mu, dtype=np.float64)
        self.sigma = np.ascontiguousarray(self.sigma, dtype=np.float64)
        self._ranks = None
        self._conservative_ranks = None

    @property
    def num_players(self) -> int:
        return len(self.mu)

    @property
    def ratings(self) -> np.ndarray:
        """Display-scale ratings (mu * scale + offset)."""
        return self.mu * self.display_scale + self.display_offset

    @property
    def rd(self) -> np.ndarray:
        """Display-scale uncertainties (sigma * scale)."""
        return self.sigma * self.display_scale

    @property
    def ranks(self) -> np.ndarray:
        """Lazily computed ranks by mu (1 = highest)."""
        if self._ranks is None:
            self._ranks = _compute_ranks(self.mu)
        return self._ranks

    def conservative_rating(self, k: float = 3.0) -> np.ndarray:
        """
        Compute conservative ratings: mu - k*sigma (internal scale).

        Args:
            k: Number of standard deviations (default 3 for ~99.7% confidence)
        """
        return self.mu - k * self.sigma

    def conservative_display_rating(self, k: float = 3.0) -> np.ndarray:
        """Compute conservative ratings in display scale."""
        return self.conservative_rating(k) * self.display_scale + self.display_offset

    def get_rating(self, player_id: int) -> Tuple[float, float]:
        """Get (mu, sigma) for a player in internal scale."""
        return float(self.mu[player_id]), float(self.sigma[player_id])

    def get_display_rating(self, player_id: int) -> Tuple[float, float]:
        """Get (rating, rd) for a player in display scale."""
        return (
            float(self.mu[player_id] * self.display_scale + self.display_offset),
            float(self.sigma[player_id] * self.display_scale),
        )

    def get_name(self, player_id: int) -> str:
        if self.player_names and player_id in self.player_names:
            return self.player_names[player_id]
        return f"Player_{player_id}"

    def predict(self, player1: int, player2: int) -> float:
        """
        Predict probability that player1 beats player2.

        P(p1 wins) = Phi((mu1 - mu2) / sqrt(2*beta^2 + sigma1^2 + sigma2^2))
        """
        import math

        mu1, sigma1 = self.mu[player1], self.sigma[player1]
        mu2, sigma2 = self.mu[player2], self.sigma[player2]

        c = math.sqrt(2.0 * self.beta ** 2 + sigma1 ** 2 + sigma2 ** 2)
        t = (mu1 - mu2) / c

        # Standard normal CDF
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

    def predict_batch(
        self,
        player1: Union[List[int], np.ndarray],
        player2: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        """Predict outcomes for multiple matchups."""
        from scipy.special import erf

        p1 = np.asarray(player1)
        p2 = np.asarray(player2)

        mu1, sigma1 = self.mu[p1], self.sigma[p1]
        mu2, sigma2 = self.mu[p2], self.sigma[p2]

        c = np.sqrt(2.0 * self.beta ** 2 + sigma1 ** 2 + sigma2 ** 2)
        t = (mu1 - mu2) / c

        return 0.5 * (1.0 + erf(t / np.sqrt(2.0)))

    def top(self, n: int = 10) -> pl.DataFrame:
        """Get top N rated players by mu."""
        indices = get_top_n_indices(self.mu, n)
        return self._indices_to_dataframe(indices)

    def conservative_top(self, n: int = 10, k: float = 3.0) -> pl.DataFrame:
        """
        Get top N players by conservative rating (mu - k*sigma).

        More appropriate for ranking when confidence matters.
        """
        conservative = self.conservative_rating(k)
        indices = get_top_n_indices(conservative, n)
        return self._indices_to_dataframe(indices, include_conservative=True, k=k)

    def bottom(self, n: int = 10) -> pl.DataFrame:
        """Get bottom N rated players."""
        indices = get_bottom_n_indices(self.mu, n)
        return self._indices_to_dataframe(indices)

    def rank(self, player_id: int) -> int:
        """Get rank of a specific player by mu (1 = highest)."""
        return int(self.ranks[player_id])

    def _indices_to_dataframe(
        self,
        indices: np.ndarray,
        include_conservative: bool = False,
        k: float = 3.0,
    ) -> pl.DataFrame:
        data = {
            "rank": self.ranks[indices],
            "player_id": indices,
            "name": _get_names_vectorized(indices, self.player_names),
            "rating": self.ratings[indices],  # Display scale
            "rd": self.rd[indices],  # Display scale
            "mu": self.mu[indices],  # Internal scale
            "sigma": self.sigma[indices],  # Internal scale
        }
        if include_conservative:
            data["conservative"] = self.conservative_display_rating(k)[indices]
        return pl.DataFrame(data)

    def matchup(self, player1: int, player2: int) -> pl.DataFrame:
        """Get detailed matchup analysis between two players."""
        r1, rd1 = self.get_display_rating(player1)
        r2, rd2 = self.get_display_rating(player2)
        mu1, sigma1 = self.get_rating(player1)
        mu2, sigma2 = self.get_rating(player2)
        p1_wins = self.predict(player1, player2)

        return pl.DataFrame({
            "player_id": [player1, player2],
            "name": [self.get_name(player1), self.get_name(player2)],
            "rating": [r1, r2],
            "rd": [rd1, rd2],
            "mu": [mu1, mu2],
            "sigma": [sigma1, sigma2],
            "win_prob": [p1_wins, 1.0 - p1_wins],
        })

    def confident_players(self, max_sigma: float = 2.0, n: int = 10) -> pl.DataFrame:
        """
        Get top players with low uncertainty (well-established ratings).

        Args:
            max_sigma: Maximum sigma in internal scale (default 2.0)
            n: Number of players to return
        """
        mask = self.sigma <= max_sigma
        filtered_mu = np.where(mask, self.mu, -np.inf)
        indices = get_top_n_indices(filtered_mu, n)
        indices = indices[mask[indices]]
        return self._indices_to_dataframe(indices)

    def to_dataframe(self, include_rank: bool = True) -> pl.DataFrame:
        """Export all ratings to DataFrame."""
        player_ids = np.arange(self.num_players)

        data = {
            "player_id": player_ids,
            "rating": self.ratings,  # Display scale
            "rd": self.rd,  # Display scale
            "mu": self.mu,
            "sigma": self.sigma,
        }

        if include_rank:
            data["rank"] = self.ranks

        if self.player_names:
            data["name"] = _get_names_vectorized(player_ids, self.player_names)

        return pl.DataFrame(data).sort("rating", descending=True)

    def save(self, path: str, include_rank: bool = False) -> None:
        """Save fitted ratings to parquet."""
        self.to_dataframe(include_rank=include_rank).write_parquet(path)

    def __repr__(self) -> str:
        return (
            f"FittedTrueSkillRatings(players={self.num_players}, "
            f"games={self.num_games_fitted}, beta={self.beta:.3f})"
        )

    def __str__(self) -> str:
        lines = [
            f"Fitted TrueSkill Ratings",
            f"  Players: {self.num_players:,}",
            f"  Games fitted: {self.num_games_fitted:,}",
            f"  Beta: {self.beta:.4f}",
            f"  Rating range: {self.ratings.min():.1f} - {self.ratings.max():.1f}",
            f"  Mean mu: {self.mu.mean():.2f}",
            f"  Mean sigma: {self.sigma.mean():.2f}",
        ]
        return "\n".join(lines)


@dataclass
class FittedYukselRatings:
    """
    Queryable fitted Yuksel ratings.

    The Yuksel method tracks uncertainty (phi) via running variance.
    phi = sqrt(V/W) represents the standard deviation of the rating history.

    Attributes:
        ratings: Array of player ratings (Elo scale)
        phi: Array of uncertainty values (standard deviation of rating)
        delta_r_max: Maximum rating change per game
        alpha: Uncertainty decay factor
        scaling_factor: Update scaling factor
        initial_rating: Initial rating value
    """

    ratings: np.ndarray
    phi: np.ndarray  # Uncertainty (sqrt(V/W))
    delta_r_max: float = 350.0
    alpha: float = 2.0
    scaling_factor: float = 0.9
    initial_rating: float = 1500.0
    num_games_fitted: int = 0
    last_day: int = -1
    player_names: Optional[Dict[int, str]] = None

    # Cached
    _ranks: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure arrays are contiguous."""
        self.ratings = np.ascontiguousarray(self.ratings, dtype=np.float64)
        self.phi = np.ascontiguousarray(self.phi, dtype=np.float64)
        self._ranks = None

    @property
    def num_players(self) -> int:
        return len(self.ratings)

    @property
    def ranks(self) -> np.ndarray:
        """Lazily computed ranks array (1 = highest rated)."""
        if self._ranks is None:
            self._ranks = _compute_ranks(self.ratings)
        return self._ranks

    def get_rating(self, player_id: int) -> Tuple[float, float]:
        """Get (rating, phi) for a single player."""
        return float(self.ratings[player_id]), float(self.phi[player_id])

    def get_name(self, player_id: int) -> str:
        if self.player_names and player_id in self.player_names:
            return self.player_names[player_id]
        return f"Player_{player_id}"

    def predict(self, player1: int, player2: int) -> float:
        """
        Predict probability that player1 beats player2.

        Uses Elo-style prediction: P = 1 / (1 + 10^((r2-r1)/400))
        """
        import math
        Q = math.log(10) / 400.0
        diff = Q * (self.ratings[player1] - self.ratings[player2])
        if diff > 20:
            return 1.0 - 1e-9
        elif diff < -20:
            return 1e-9
        return 1.0 / (1.0 + math.exp(-diff))

    def predict_batch(
        self,
        player1: Union[List[int], np.ndarray],
        player2: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        """Predict outcomes for multiple matchups."""
        Q = np.log(10) / 400.0
        p1 = np.asarray(player1)
        p2 = np.asarray(player2)
        diff = Q * (self.ratings[p1] - self.ratings[p2])
        return 1.0 / (1.0 + np.exp(-np.clip(diff, -20, 20)))

    def top(self, n: int = 10) -> pl.DataFrame:
        """Get top N rated players with uncertainty."""
        indices = get_top_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def bottom(self, n: int = 10) -> pl.DataFrame:
        """Get bottom N rated players."""
        indices = get_bottom_n_indices(self.ratings, n)
        return self._indices_to_dataframe(indices)

    def rank(self, player_id: int) -> int:
        """Get rank of a specific player (1 = highest rated)."""
        return int(self.ranks[player_id])

    def _indices_to_dataframe(self, indices: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({
            "rank": self.ranks[indices],
            "player_id": indices,
            "name": _get_names_vectorized(indices, self.player_names),
            "rating": self.ratings[indices],
            "phi": self.phi[indices],
        })

    def matchup(self, player1: int, player2: int) -> pl.DataFrame:
        """Get detailed matchup analysis between two players."""
        r1, phi1 = self.get_rating(player1)
        r2, phi2 = self.get_rating(player2)
        p1_wins = self.predict(player1, player2)

        return pl.DataFrame({
            "player_id": [player1, player2],
            "name": [self.get_name(player1), self.get_name(player2)],
            "rating": [r1, r2],
            "phi": [phi1, phi2],
            "win_prob": [p1_wins, 1.0 - p1_wins],
        })

    def confident_players(self, max_phi: float = 100.0, n: int = 10) -> pl.DataFrame:
        """
        Get top players with low uncertainty (well-established ratings).

        Args:
            max_phi: Maximum uncertainty (default 100.0)
            n: Number of players to return
        """
        mask = self.phi <= max_phi
        filtered_ratings = np.where(mask, self.ratings, -np.inf)
        indices = get_top_n_indices(filtered_ratings, n)
        indices = indices[mask[indices]]
        return self._indices_to_dataframe(indices)

    def to_dataframe(self, include_rank: bool = True) -> pl.DataFrame:
        """Export all ratings to DataFrame."""
        player_ids = np.arange(self.num_players)

        data = {
            "player_id": player_ids,
            "rating": self.ratings,
            "phi": self.phi,
        }

        if include_rank:
            data["rank"] = self.ranks

        if self.player_names:
            data["name"] = _get_names_vectorized(player_ids, self.player_names)

        return pl.DataFrame(data).sort("rating", descending=True)

    def save(self, path: str, include_rank: bool = False) -> None:
        """Save fitted ratings to parquet."""
        self.to_dataframe(include_rank=include_rank).write_parquet(path)

    def __repr__(self) -> str:
        return (
            f"FittedYukselRatings(players={self.num_players}, "
            f"games={self.num_games_fitted}, delta_r_max={self.delta_r_max})"
        )

    def __str__(self) -> str:
        lines = [
            f"Fitted Yuksel Ratings",
            f"  Players: {self.num_players:,}",
            f"  Games fitted: {self.num_games_fitted:,}",
            f"  Delta R Max: {self.delta_r_max}",
            f"  Alpha: {self.alpha}",
            f"  Rating range: {self.ratings.min():.1f} - {self.ratings.max():.1f}",
            f"  Mean phi: {self.phi.mean():.1f}",
        ]
        return "\n".join(lines)
