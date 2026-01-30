"""
Gaussian distribution utilities for TrueSkill Through Time.

Provides both scalar and vectorized (PyTorch tensor) operations.
Uses precision form (pi, tau) for numerical stability in message passing.
"""

import math
from typing import Tuple, Union

import torch

# Constants
SQRT2 = math.sqrt(2)
SQRT2PI = math.sqrt(2 * math.pi)
INF = float("inf")


def pdf(x: torch.Tensor, mu: torch.Tensor = None, sigma: torch.Tensor = None) -> torch.Tensor:
    """Standard normal PDF (or general normal if mu, sigma provided)."""
    if mu is None:
        mu = torch.zeros_like(x)
    if sigma is None:
        sigma = torch.ones_like(x)

    z = (x - mu) / sigma
    return torch.exp(-0.5 * z * z) / (sigma * SQRT2PI)


def cdf(x: torch.Tensor, mu: torch.Tensor = None, sigma: torch.Tensor = None) -> torch.Tensor:
    """Standard normal CDF (or general normal if mu, sigma provided)."""
    if mu is None:
        mu = torch.zeros_like(x)
    if sigma is None:
        sigma = torch.ones_like(x)

    z = (x - mu) / sigma
    return 0.5 * (1 + torch.erf(z / SQRT2))


def ppf(p: torch.Tensor) -> torch.Tensor:
    """Inverse CDF (percent point function) of standard normal."""
    # Use erfinv: if Phi(x) = p, then x = sqrt(2) * erfinv(2p - 1)
    return SQRT2 * torch.erfinv(2 * p - 1)


def v_win(t: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute v for win/loss outcome (non-draw).

    v = pdf(-t) / cdf(-t) = pdf(t) / cdf(-t)

    This is the mean correction factor for a truncated Gaussian.
    """
    # cdf(-t) = 1 - cdf(t)
    denom = cdf(-t)
    denom = torch.clamp(denom, min=eps)
    return pdf(t) / denom


def w_win(t: torch.Tensor, v: torch.Tensor = None, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute w for win/loss outcome.

    w = v * (v + t)

    This is the variance correction factor.
    """
    if v is None:
        v = v_win(t, eps)
    return v * (v + t)


def v_draw(t: torch.Tensor, margin: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute v for draw outcome.

    v = (pdf(alpha) - pdf(beta)) / (cdf(beta) - cdf(alpha))
    where alpha = (-margin - t), beta = (margin - t)
    """
    alpha = -margin - t
    beta = margin - t

    pdf_diff = pdf(alpha) - pdf(beta)
    cdf_diff = cdf(beta) - cdf(alpha)
    cdf_diff = torch.clamp(cdf_diff, min=eps)

    return pdf_diff / cdf_diff


def w_draw(t: torch.Tensor, margin: torch.Tensor, v: torch.Tensor = None, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute w for draw outcome.
    """
    alpha = -margin - t
    beta = margin - t

    if v is None:
        v = v_draw(t, margin, eps)

    cdf_diff = cdf(beta) - cdf(alpha)
    cdf_diff = torch.clamp(cdf_diff, min=eps)

    u = (alpha * pdf(alpha) - beta * pdf(beta)) / cdf_diff
    return -(u - v * v)


class GaussianTensor:
    """
    Batch of Gaussian distributions represented as PyTorch tensors.

    Uses precision form internally for stable message passing:
    - pi = 1/sigma^2 (precision)
    - tau = mu/sigma^2 (precision-weighted mean)

    Supports vectorized operations across batches of distributions.
    """

    def __init__(
        self,
        mu: torch.Tensor = None,
        sigma: torch.Tensor = None,
        pi: torch.Tensor = None,
        tau: torch.Tensor = None,
        device: torch.device = None,
    ):
        """
        Initialize from either (mu, sigma) or (pi, tau).
        """
        if device is None:
            if mu is not None:
                device = mu.device
            elif pi is not None:
                device = pi.device
            else:
                device = torch.device("cpu")

        self.device = device

        if pi is not None and tau is not None:
            self._pi = pi.to(device)
            self._tau = tau.to(device)
        elif mu is not None and sigma is not None:
            mu = mu.to(device)
            sigma = sigma.to(device)
            # Avoid division by zero
            sigma_sq = sigma * sigma
            sigma_sq = torch.where(sigma_sq > 1e-10, sigma_sq, torch.full_like(sigma_sq, 1e-10))
            self._pi = 1.0 / sigma_sq
            self._tau = mu / sigma_sq
        else:
            # Default: uninformative prior (infinite variance)
            self._pi = torch.tensor([0.0], device=device)
            self._tau = torch.tensor([0.0], device=device)

    @property
    def pi(self) -> torch.Tensor:
        return self._pi

    @property
    def tau(self) -> torch.Tensor:
        return self._tau

    @property
    def mu(self) -> torch.Tensor:
        """Mean of the distribution."""
        # mu = tau / pi, handle pi=0 case
        return torch.where(self._pi > 1e-10, self._tau / self._pi, torch.zeros_like(self._tau))

    @property
    def sigma(self) -> torch.Tensor:
        """Standard deviation of the distribution."""
        # sigma = 1/sqrt(pi), handle pi=0 case
        return torch.where(self._pi > 1e-10, 1.0 / torch.sqrt(self._pi), torch.full_like(self._pi, INF))

    @property
    def sigma_sq(self) -> torch.Tensor:
        """Variance of the distribution."""
        return torch.where(self._pi > 1e-10, 1.0 / self._pi, torch.full_like(self._pi, INF))

    def __mul__(self, other: "GaussianTensor") -> "GaussianTensor":
        """
        Multiply two Gaussians (product of PDFs, normalized).
        In precision form: pi_new = pi1 + pi2, tau_new = tau1 + tau2
        """
        return GaussianTensor(
            pi=self._pi + other._pi,
            tau=self._tau + other._tau,
            device=self.device,
        )

    def __truediv__(self, other: "GaussianTensor") -> "GaussianTensor":
        """
        Divide two Gaussians (remove a message).
        In precision form: pi_new = pi1 - pi2, tau_new = tau1 - tau2
        """
        new_pi = self._pi - other._pi
        new_tau = self._tau - other._tau
        # Clamp pi to avoid negative precision
        new_pi = torch.clamp(new_pi, min=0.0)
        return GaussianTensor(pi=new_pi, tau=new_tau, device=self.device)

    def __add__(self, other: "GaussianTensor") -> "GaussianTensor":
        """
        Add two Gaussians (sum of random variables).
        mu_new = mu1 + mu2, sigma_new^2 = sigma1^2 + sigma2^2
        """
        new_sigma_sq = self.sigma_sq + other.sigma_sq
        new_mu = self.mu + other.mu
        new_sigma = torch.sqrt(new_sigma_sq)
        return GaussianTensor(mu=new_mu, sigma=new_sigma, device=self.device)

    def clone(self) -> "GaussianTensor":
        return GaussianTensor(pi=self._pi.clone(), tau=self._tau.clone(), device=self.device)

    @staticmethod
    def uninformative(shape: Tuple[int, ...], device: torch.device) -> "GaussianTensor":
        """Create uninformative (infinite variance) Gaussians."""
        return GaussianTensor(
            pi=torch.zeros(shape, device=device),
            tau=torch.zeros(shape, device=device),
            device=device,
        )

    @staticmethod
    def from_mu_sigma(mu: torch.Tensor, sigma: torch.Tensor) -> "GaussianTensor":
        return GaussianTensor(mu=mu, sigma=sigma)

    def to(self, device: torch.device) -> "GaussianTensor":
        return GaussianTensor(pi=self._pi.to(device), tau=self._tau.to(device), device=device)


def truncate_gaussian(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    margin: torch.Tensor,
    is_win: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute truncated Gaussian parameters after observing outcome.

    Args:
        mu: Mean of performance difference
        sigma: Std dev of performance difference
        margin: Draw margin (0 for no draws)
        is_win: Boolean tensor, True if player 1 won (else draw)

    Returns:
        (mu_new, sigma_new) after truncation
    """
    t = mu / sigma

    # For wins: use v_win, w_win
    v_w = v_win(t)
    w_w = w_win(t, v_w)

    # For draws: use v_draw, w_draw
    v_d = v_draw(t, margin / sigma)
    w_d = w_draw(t, margin / sigma, v_d)

    v = torch.where(is_win, v_w, v_d)
    w = torch.where(is_win, w_w, w_d)

    # Clamp w to valid range
    w = torch.clamp(w, min=1e-10, max=1.0 - 1e-10)

    mu_new = mu + sigma * v
    sigma_new = sigma * torch.sqrt(1 - w)

    return mu_new, sigma_new
