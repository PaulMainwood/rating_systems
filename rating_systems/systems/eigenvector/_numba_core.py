"""Numba-accelerated kernels for eigenvector centrality predictions."""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def predict_proba_batch(
    player1: np.ndarray,
    player2: np.ndarray,
    centrality: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Predict P(player1 wins) using logistic on centrality difference."""
    n = len(player1)
    proba = np.empty(n, dtype=np.float64)
    log10 = np.log(10.0)
    for i in range(n):
        diff = centrality[player1[i]] - centrality[player2[i]]
        proba[i] = 1.0 / (1.0 + np.exp(-diff * log10 / scale))
    return proba


@nb.njit(cache=True)
def predict_single(
    c1: float,
    c2: float,
    scale: float,
) -> float:
    """Predict P(player1 wins) for a single matchup."""
    log10 = np.log(10.0)
    diff = c1 - c2
    return 1.0 / (1.0 + np.exp(-diff * log10 / scale))


@nb.njit(cache=True)
def decay_edges_inplace(
    weight: np.ndarray,
    day: np.ndarray,
    n: int,
    current_day: int,
    decay_rate: float,
) -> None:
    """Decay all edge weights to current_day, updating day stamps in-place."""
    for i in range(n):
        dt = current_day - day[i]
        if dt > 0:
            weight[i] *= np.exp(-decay_rate * dt)
            day[i] = current_day
