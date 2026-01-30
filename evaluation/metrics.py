"""Evaluation metrics for rating system predictions."""

import torch


def brier_score(predictions: torch.Tensor, actuals: torch.Tensor) -> torch.Tensor:
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
    return torch.mean((predictions - actuals) ** 2)


def log_loss(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    eps: float = 1e-15,
) -> torch.Tensor:
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
    predictions = torch.clamp(predictions, eps, 1 - eps)

    loss = -(actuals * torch.log(predictions) + (1 - actuals) * torch.log(1 - predictions))
    return torch.mean(loss)


def accuracy(predictions: torch.Tensor, actuals: torch.Tensor) -> torch.Tensor:
    """
    Calculate prediction accuracy.

    Predicts player 1 wins if probability > 0.5.

    Args:
        predictions: (N,) Predicted probabilities for player 1 winning
        actuals: (N,) Actual outcomes (1.0 = player 1 won, 0.0 = player 2 won)

    Returns:
        Scalar accuracy [0, 1]
    """
    predicted_wins = (predictions > 0.5).float()
    actual_wins = (actuals > 0.5).float()
    return torch.mean((predicted_wins == actual_wins).float())


def calibration_error(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
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
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=predictions.device)
    ece = torch.tensor(0.0, device=predictions.device)

    for i in range(n_bins):
        in_bin = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            avg_confidence = predictions[in_bin].mean()
            avg_accuracy = actuals[in_bin].mean()
            ece += prop_in_bin * torch.abs(avg_accuracy - avg_confidence)

    return ece
