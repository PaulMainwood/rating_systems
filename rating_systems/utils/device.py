"""Device utilities for PyTorch acceleration (optional)."""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def get_device() -> Optional["torch.device"]:
    """
    Get the best available device for computation.

    Returns None if PyTorch is not installed.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return None


def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False
