"""Evaluation metrics for rating system predictions (numpy-based).

All metrics optionally accept ``sample_weight`` — a per-match weight array
aligned to ``predictions`` / ``actuals``. When provided, metrics are computed
as a weighted mean rather than a plain mean. Passing zero-sum weights (or
an empty array) returns NaN.
"""

from typing import Optional

import numpy as np


def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0 or len(values) == 0:
        return float("nan")
    return float(np.sum(values * weights) / total)


def brier_score(
    predictions: np.ndarray,
    actuals: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Brier score (mean squared error of probability predictions).

    Lower is better. Range: [0, 1]. Perfect predictions: 0. Random (0.5): 0.25.

    Args:
        predictions: (N,) Predicted probabilities for player 1 winning
        actuals: (N,) Actual outcomes (1.0 = player 1 won, 0.0 = player 2 won)
        sample_weight: (N,) optional per-match weights. If given, returns a
            weighted mean; zero-sum weights yield NaN.
    """
    return _weighted_mean((predictions - actuals) ** 2, sample_weight)


def log_loss(
    predictions: np.ndarray,
    actuals: np.ndarray,
    eps: float = 1e-15,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Log loss (binary cross-entropy).

    Lower is better. Range: [0, inf). Perfect: 0. Random (0.5): ~0.693.

    Args:
        predictions: (N,) Predicted probabilities for player 1 winning
        actuals: (N,) Actual outcomes (1.0 = player 1 won, 0.0 = player 2 won)
        eps: Small value to avoid log(0)
        sample_weight: (N,) optional per-match weights.
    """
    predictions = np.clip(predictions, eps, 1 - eps)
    loss = -(actuals * np.log(predictions) + (1 - actuals) * np.log(1 - predictions))
    return _weighted_mean(loss, sample_weight)


def accuracy(
    predictions: np.ndarray,
    actuals: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Prediction accuracy (predicts P1 wins if probability > 0.5).

    Args:
        predictions: (N,) Predicted probabilities for player 1 winning
        actuals: (N,) Actual outcomes (1.0 = player 1 won, 0.0 = player 2 won)
        sample_weight: (N,) optional per-match weights.
    """
    predicted_wins = (predictions > 0.5).astype(float)
    actual_wins = (actuals > 0.5).astype(float)
    return _weighted_mean((predicted_wins == actual_wins).astype(float), sample_weight)


def calibration_error(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Expected calibration error.

    Measures how well predicted probabilities match actual frequencies.
    With ``sample_weight``, each bin's confidence and accuracy are weighted
    means and the overall ECE weights bins by the total weight in the bin.
    """
    if sample_weight is None:
        w = np.ones_like(predictions, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)

    total_w = float(w.sum())
    if total_w <= 0.0 or len(predictions) == 0:
        return float("nan")

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        bin_w = float(w[in_bin].sum())
        if bin_w <= 0.0:
            continue
        prop_in_bin = bin_w / total_w
        avg_confidence = float(np.sum(predictions[in_bin] * w[in_bin]) / bin_w)
        avg_accuracy = float(np.sum(actuals[in_bin] * w[in_bin]) / bin_w)
        ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)
