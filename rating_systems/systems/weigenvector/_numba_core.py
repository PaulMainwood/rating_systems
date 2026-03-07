"""Numba-accelerated kernels for weighted eigenvector centrality predictions."""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def predict_proba_with_handicap_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    centrality: np.ndarray,
    scale: float,
    handicaps: np.ndarray,
) -> np.ndarray:
    """Predict P(player1 wins) with per-game handicaps."""
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)
    log10 = np.log(10.0)
    for i in range(n):
        diff = centrality[player1[i]] - centrality[player2[i]] + handicaps[i]
        proba[i] = 1.0 / (1.0 + np.exp(-diff * log10 / scale))
    return proba


@nb.njit(cache=True)
def predict_single_with_handicap(
    c1: float,
    c2: float,
    scale: float,
    handicap: float,
) -> float:
    """Predict P(player1 wins) for a single matchup with handicap."""
    log10 = np.log(10.0)
    diff = c1 - c2 + handicap
    return 1.0 / (1.0 + np.exp(-diff * log10 / scale))
