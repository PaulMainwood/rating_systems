"""Evaluation metrics for rating system predictions (numpy-based)."""

import numpy as np


def brier_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Brier score (mean squared error of probability predictions).

    Lower is better. Range: [0, 1]
    Perfect predictions: 0
    Random guessing (0.5): 0.25

    Args:
        predictions: (N,) Predicted probabilities for player 1 winning
        actuals: (N,) Actual outcomes (1.0 = player 1 won, 0.0 = player 2 won)

    Returns:
        Scalar Brier score
    """
    return float(np.mean((predictions - actuals) ** 2))


def log_loss(
    predictions: np.ndarray,
    actuals: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Calculate log loss (cross-entropy loss).

    Lower is better. Range: [0, inf)
    Perfect predictions: 0
    Random guessing (0.5): ~0.693

    Args:
        predictions: (N,) Predicted probabilities for player 1 winning
        actuals: (N,) Actual outcomes (1.0 = player 1 won, 0.0 = player 2 won)
        eps: Small value to avoid log(0)

    Returns:
        Scalar log loss
    """
    # Clamp predictions to avoid log(0)
    predictions = np.clip(predictions, eps, 1 - eps)

    loss = -(actuals * np.log(predictions) + (1 - actuals) * np.log(1 - predictions))
    return float(np.mean(loss))


def accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate prediction accuracy.

    Predicts player 1 wins if probability > 0.5.

    Args:
        predictions: (N,) Predicted probabilities for player 1 winning
        actuals: (N,) Actual outcomes (1.0 = player 1 won, 0.0 = player 2 won)

    Returns:
        Scalar accuracy [0, 1]
    """
    predicted_wins = (predictions > 0.5).astype(float)
    actual_wins = (actuals > 0.5).astype(float)
    return float(np.mean(predicted_wins == actual_wins))


def calibration_error(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate expected calibration error.

    Measures how well predicted probabilities match actual frequencies.

    Args:
        predictions: (N,) Predicted probabilities
        actuals: (N,) Actual outcomes
        n_bins: Number of bins for calibration

    Returns:
        Scalar expected calibration error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(predictions[in_bin])
            avg_accuracy = np.mean(actuals[in_bin])
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)
