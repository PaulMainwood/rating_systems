"""
Hyperparameter optimization for rating systems.

Uses scipy.optimize to find optimal parameters by minimizing log loss
on a held-out validation set via walk-forward backtesting.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from scipy.optimize import differential_evolution, minimize

from ..base import RatingSystem
from ..data import GameDataset
from .backtester import Backtester


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""

    system_class: str
    best_params: Dict[str, float]
    best_brier: float
    best_log_loss: float
    best_accuracy: float
    n_evaluations: int
    total_time: float
    history: List[Dict[str, Any]]
    fixed_params: Dict[str, Any] = None  # Parameters that were held fixed during optimization

    def summary(self) -> str:
        """Return a summary string."""
        params_str = ", ".join(f"{k}={v:.4f}" for k, v in self.best_params.items())
        return (
            f"\n{'='*60}\n"
            f"Optimization Results for {self.system_class}\n"
            f"{'='*60}\n"
            f"Best Parameters: {params_str}\n"
            f"Best Brier Score: {self.best_brier:.6f}\n"
            f"Best Log Loss: {self.best_log_loss:.6f}\n"
            f"Best Accuracy: {self.best_accuracy:.1%}\n"
            f"Evaluations: {self.n_evaluations}\n"
            f"Total Time: {self.total_time:.1f}s\n"
            f"{'='*60}"
        )


class RatingSystemOptimizer:
    """
    Hyperparameter optimizer for rating systems.

    Uses differential evolution (global optimization) or L-BFGS-B
    (local optimization) to find parameters that minimize log loss.

    Example:
        >>> optimizer = RatingSystemOptimizer(Elo, dataset)
        >>> result = optimizer.optimize(
        ...     param_bounds={'k_factor': (8, 64), 'scale': (200, 600)},
        ...     verbose=True
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        system_class: Type[RatingSystem],
        dataset: GameDataset,
        train_ratio: float = 0.7,
        fixed_params: Optional[Dict[str, Any]] = None,
        max_test_days: Optional[int] = None,
    ):
        """
        Initialize optimizer.

        Args:
            system_class: Rating system class to optimize
            dataset: Full game dataset
            train_ratio: Fraction of days for initial training (default 0.7)
            fixed_params: Parameters to keep fixed during optimization
            max_test_days: Limit test period to this many days (for faster optimization)
        """
        self.system_class = system_class
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.fixed_params = fixed_params or {}
        self.max_test_days = max_test_days

        # Compute train/test split day
        days = dataset.days
        split_idx = int(len(days) * train_ratio)
        self.train_end_day = days[split_idx - 1] if split_idx > 0 else days[0]

        # Optionally limit test period
        self.test_end_day = None
        if max_test_days is not None:
            test_days = [d for d in days if d > self.train_end_day]
            if len(test_days) > max_test_days:
                self.test_end_day = test_days[max_test_days - 1]

        # Tracking
        self._eval_count = 0
        self._history: List[Dict[str, Any]] = []
        self._best_log_loss = float('inf')
        self._best_params: Dict[str, float] = {}
        self._best_result: Optional[Any] = None
        self._start_time = 0.0
        self._verbose = False
        self._param_names: List[str] = []
        self._bounds: Optional[List[Tuple[float, float]]] = None

    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function for optimization (minimizes log loss).

        Args:
            x: Parameter values as array

        Returns:
            Log loss (to minimize)
        """
        # Clamp parameters to bounds (safety net for methods that may violate)
        if self._bounds is not None:
            x = np.clip(x, [b[0] for b in self._bounds], [b[1] for b in self._bounds])

        # Convert array to parameter dict
        params = {name: val for name, val in zip(self._param_names, x)}
        params.update(self.fixed_params)

        try:
            # Create system with these parameters
            system = self.system_class(**params)

            # Run backtest
            backtester = Backtester(system, self.dataset)
            result = backtester.run(
                train_end_day=self.train_end_day,
                test_end_day=self.test_end_day,
                verbose=False,
            )

            brier = result.brier
            log_loss = result.log_loss
            accuracy = result.accuracy

        except Exception as e:
            # Return high penalty for invalid parameters
            if self._verbose:
                print(f"  [!] Error with params {params}: {e}")
            return 10.0  # High log loss penalty

        self._eval_count += 1

        # Track history
        entry = {
            "eval": self._eval_count,
            "params": params.copy(),
            "brier": brier,
            "log_loss": log_loss,
            "accuracy": accuracy,
            "time": time.time() - self._start_time,
        }
        self._history.append(entry)

        # Update best
        is_best = log_loss < self._best_log_loss
        if is_best:
            self._best_log_loss = log_loss
            self._best_params = params.copy()
            self._best_result = result

        # Always print every iteration with full details
        if self._verbose:
            elapsed = time.time() - self._start_time
            params_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items()
                                   if k in self._param_names)
            best_marker = " *BEST*" if is_best else ""
            print(f"  [{self._eval_count:3d}] LogLoss={log_loss:.6f} Brier={brier:.6f} "
                  f"Acc={accuracy:.4f} | {params_str} ({elapsed:.1f}s){best_marker}")

        return log_loss

    def optimize(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        method: str = "differential_evolution",
        maxiter: int = 50,
        verbose: bool = True,
        x0: Optional[np.ndarray] = None,
        ftol: Optional[float] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize hyperparameters.

        Args:
            param_bounds: Dict mapping parameter names to (min, max) bounds
            method: Optimization method:
                - "differential_evolution": Global optimization (recommended)
                - "L-BFGS-B": Local gradient-based optimization
                - "Nelder-Mead": Simplex method (no gradients)
                - "Powell": Direction set method
            maxiter: Maximum iterations/generations
            verbose: Whether to print progress
            x0: Initial parameter values for local methods. If None, uses
                midpoint of bounds. Ignored for differential_evolution.
            **kwargs: Additional arguments passed to optimizer

        Returns:
            OptimizationResult with best parameters and metrics
        """
        self._verbose = verbose
        self._eval_count = 0
        self._history = []
        self._best_log_loss = float('inf')
        self._best_params = {}
        self._start_time = time.time()

        self._param_names = list(param_bounds.keys())
        bounds = [param_bounds[name] for name in self._param_names]
        self._bounds = bounds

        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimizing {self.system_class.__name__}")
            print(f"{'='*60}")
            print(f"Method: {method}")
            print(f"Parameters: {self._param_names}")
            print(f"Bounds: {bounds}")
            print(f"Max iterations: {maxiter}")
            print(f"Train/test split at day: {self.train_end_day}")
            if self.test_end_day:
                test_days = len([d for d in self.dataset.days
                                if self.train_end_day < d <= self.test_end_day])
                print(f"Test period limited to: {test_days} days (ending day {self.test_end_day})")
            if self.fixed_params:
                print(f"Fixed params: {self.fixed_params}")
            print(f"Dataset: {self.dataset.num_games:,} games, "
                  f"{self.dataset.num_players:,} players")
            print(f"{'='*60}\n")

        if method == "differential_evolution":
            # Global optimization - good for finding global minimum
            result = differential_evolution(
                self._objective,
                bounds,
                maxiter=maxiter,
                polish=True,  # Refine with L-BFGS-B at end
                disp=verbose,
                seed=42,
                workers=1,  # Single-threaded for compatibility
                updating='deferred',
                **kwargs,
            )
            best_x = result.x
        else:
            # Local optimization methods
            if x0 is None:
                x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
            else:
                x0 = np.asarray(x0, dtype=np.float64)

            if verbose:
                x0_dict = {name: val for name, val in zip(self._param_names, x0)}
                print(f"Starting point: {x0_dict}\n")

            # For L-BFGS-B, use larger finite-difference step size
            # Default eps (~1e-8) is too small for noisy log loss objectives
            options = {"maxiter": maxiter, "disp": False}
            if ftol is not None:
                options["ftol"] = ftol
                if method == "Powell":
                    options["xtol"] = ftol
            if method == "L-BFGS-B":
                # Scale eps based on parameter ranges
                # Use ~2% of parameter range for gradient estimation
                eps_values = np.array([(b[1] - b[0]) * 0.02 for b in bounds])
                options["eps"] = eps_values

            # Powell and Nelder-Mead support bounds since scipy 1.7.0
            supports_bounds = {"L-BFGS-B", "SLSQP", "Powell", "Nelder-Mead"}
            result = minimize(
                self._objective,
                x0,
                method=method,
                bounds=bounds if method in supports_bounds else None,
                options=options,
                **kwargs,
            )
            best_x = result.x

        total_time = time.time() - self._start_time

        # Build final params dict
        final_params = {name: val for name, val in zip(self._param_names, best_x)}

        # Get metrics from best result
        best_brier = self._best_result.brier if self._best_result else 0.0
        best_accuracy = self._best_result.accuracy if self._best_result else 0.0

        opt_result = OptimizationResult(
            system_class=self.system_class.__name__,
            best_params=final_params,
            best_brier=best_brier,
            best_log_loss=self._best_log_loss,
            best_accuracy=best_accuracy,
            n_evaluations=self._eval_count,
            total_time=total_time,
            history=self._history,
            fixed_params=self.fixed_params,
        )

        if verbose:
            print(opt_result.summary())

        return opt_result


def optimize_elo(
    dataset: GameDataset,
    k_bounds: Tuple[float, float] = (4, 100),
    initial_rating: float = 1500.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize Elo parameters (k_factor only; scale fixed at 400).

    Args:
        dataset: Game dataset
        k_bounds: Bounds for K-factor
        initial_rating: Fixed initial rating
        maxiter: Maximum optimization iterations
        train_ratio: Fraction of days for initial training (default 0.7)
        method: Optimization method ("differential_evolution" or "L-BFGS-B")
        verbose: Whether to print progress

    Returns:
        OptimizationResult
    """
    from ..systems.elo import Elo

    optimizer = RatingSystemOptimizer(
        Elo,
        dataset,
        train_ratio=train_ratio,
        fixed_params={"initial_rating": initial_rating},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "k_factor": k_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_glicko(
    dataset: GameDataset,
    c_bounds: Tuple[float, float] = (20, 100),
    initial_rating: float = 1500.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize Glicko parameters.

    Args:
        dataset: Game dataset
        c_bounds: Bounds for c parameter (RD increase per period)
        initial_rating: Fixed initial rating
        maxiter: Maximum optimization iterations
        train_ratio: Fraction of days for initial training (default 0.7)
        method: Optimization method ("differential_evolution" or "L-BFGS-B")
        verbose: Whether to print progress

    Returns:
        OptimizationResult
    """
    from ..systems.glicko import Glicko

    optimizer = RatingSystemOptimizer(
        Glicko,
        dataset,
        train_ratio=train_ratio,
        fixed_params={"initial_rating": initial_rating},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "c": c_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_glicko2(
    dataset: GameDataset,
    initial_volatility_bounds: Tuple[float, float] = (0.03, 0.15),
    tau_bounds: Tuple[float, float] = (-1.0, 5.0),
    initial_rating: float = 1500.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize Glicko-2 parameters.

    Args:
        dataset: Game dataset
        initial_volatility_bounds: Bounds for initial volatility
        tau_bounds: Bounds for tau (system constant)
        initial_rating: Fixed initial rating
        maxiter: Maximum optimization iterations
        train_ratio: Fraction of days for initial training (default 0.7)
        method: Optimization method ("differential_evolution" or "L-BFGS-B")
        verbose: Whether to print progress

    Returns:
        OptimizationResult
    """
    from ..systems.glicko2 import Glicko2

    optimizer = RatingSystemOptimizer(
        Glicko2,
        dataset,
        train_ratio=train_ratio,
        fixed_params={"initial_rating": initial_rating},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "initial_volatility": initial_volatility_bounds,
            "tau": tau_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_stephenson(
    dataset: GameDataset,
    cval_bounds: Tuple[float, float] = (5, 50),
    hval_bounds: Tuple[float, float] = (0, 30),
    lambda_bounds: Tuple[float, float] = (0, 10),
    initial_rating: float = 1500.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize Stephenson parameters.

    Args:
        dataset: Game dataset
        cval_bounds: Bounds for RD increase per period of inactivity
        hval_bounds: Bounds for additional RD increase per game
        lambda_bounds: Bounds for neighbourhood shrinkage parameter
        initial_rating: Fixed initial rating
        maxiter: Maximum optimization iterations
        train_ratio: Fraction of days for initial training (default 0.7)
        method: Optimization method ("differential_evolution" or "L-BFGS-B")
        verbose: Whether to print progress

    Returns:
        OptimizationResult
    """
    from ..systems.stephenson import Stephenson

    optimizer = RatingSystemOptimizer(
        Stephenson,
        dataset,
        train_ratio=train_ratio,
        fixed_params={"initial_rating": initial_rating},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "cval": cval_bounds,
            "hval": hval_bounds,
            "lambda_param": lambda_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_whr(
    dataset: GameDataset,
    w2_bounds: Tuple[float, float] = (10, 1000),
    maxiter: int = 20,
    max_test_days: Optional[int] = None,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
    fixed_params: Optional[Dict[str, Any]] = None,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize WHR parameters.

    Args:
        dataset: Game dataset
        w2_bounds: Bounds for Wiener variance (skill drift)
        maxiter: Maximum optimization iterations
        max_test_days: Limit test period to this many days (for faster optimization)
        train_ratio: Fraction of days for initial training
        method: Optimization method ("differential_evolution" or "L-BFGS-B")
        verbose: Whether to print progress
        fixed_params: Parameters held fixed during optimization
            (e.g. max_iterations, refit_max_iterations, refit_interval).
            If None, WHR class defaults apply.

    Returns:
        OptimizationResult
    """
    from ..systems.whr import WHR

    optimizer = RatingSystemOptimizer(
        WHR,
        dataset,
        train_ratio=train_ratio,
        fixed_params=fixed_params or {},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "w2": w2_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_ttt(
    dataset: GameDataset,
    sigma_bounds: Tuple[float, float] = (0.5, 3.0),
    beta_bounds: Tuple[float, float] = (0.1, 1.5),
    gamma_bounds: Tuple[float, float] = (0.001, 0.1),
    maxiter: int = 20,
    max_test_days: Optional[int] = None,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
    fixed_params: Optional[Dict[str, Any]] = None,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize TrueSkill Through Time parameters.

    Args:
        dataset: Game dataset
        sigma_bounds: Bounds for prior skill std dev
        beta_bounds: Bounds for performance variability
        gamma_bounds: Bounds for skill drift rate
        maxiter: Maximum optimization iterations
        max_test_days: Limit test period to this many days (for faster optimization)
        train_ratio: Fraction of days for initial training
        method: Optimization method ("differential_evolution" or "L-BFGS-B")
        verbose: Whether to print progress
        fixed_params: Parameters held fixed during optimization
            (e.g. max_iterations, refit_max_iterations, refit_interval).
            If None, TTT class defaults apply.

    Returns:
        OptimizationResult
    """
    from ..systems.trueskill_through_time import TrueSkillThroughTime

    optimizer = RatingSystemOptimizer(
        TrueSkillThroughTime,
        dataset,
        train_ratio=train_ratio,
        fixed_params=fixed_params or {},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "sigma": sigma_bounds,
            "beta": beta_bounds,
            "gamma": gamma_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_matern_ttt(
    dataset: GameDataset,
    sigma_bounds: Tuple[float, float] = (0.5, 10.0),
    beta_bounds: Tuple[float, float] = (0.1, 3.0),
    lengthscale_bounds: Tuple[float, float] = (50, 2000),
    maxiter: int = 20,
    max_test_days: Optional[int] = None,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
    fixed_params: Optional[Dict[str, Any]] = None,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize Matérn 3/2 TrueSkill Through Time parameters.

    Args:
        dataset: Game dataset
        sigma_bounds: Bounds for marginal std dev
        beta_bounds: Bounds for performance noise
        lengthscale_bounds: Bounds for temporal correlation length (days)
        maxiter: Maximum optimization iterations
        max_test_days: Limit test period
        train_ratio: Fraction of days for initial training
        method: Optimization method
        verbose: Whether to print progress
        fixed_params: Parameters held fixed during optimization

    Returns:
        OptimizationResult
    """
    from ..systems.matern_ttt import MaternTTT

    optimizer = RatingSystemOptimizer(
        MaternTTT,
        dataset,
        train_ratio=train_ratio,
        fixed_params=fixed_params or {},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "sigma": sigma_bounds,
            "beta": beta_bounds,
            "lengthscale": lengthscale_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_margin_ttt(
    dataset: GameDataset,
    sigma_bounds: Tuple[float, float] = (0.5, 5.0),
    beta_bounds: Tuple[float, float] = (0.1, 3.0),
    gamma_bounds: Tuple[float, float] = (0.001, 0.3),
    margin_scale_bounds: Tuple[float, float] = (1.0, 30.0),
    sigma_margin_bounds: Tuple[float, float] = (0.5, 10.0),
    maxiter: int = 20,
    max_test_days: Optional[int] = None,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
    fixed_params: Optional[Dict[str, Any]] = None,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize MarginTTT parameters.

    Args:
        dataset: Game dataset
        sigma_bounds: Bounds for prior skill std dev
        beta_bounds: Bounds for performance variability
        gamma_bounds: Bounds for skill drift rate
        margin_scale_bounds: Bounds for margin → skill unit conversion
        sigma_margin_bounds: Bounds for observation noise
        maxiter: Maximum optimization iterations
        max_test_days: Limit test period
        train_ratio: Fraction of days for initial training
        method: Optimization method
        verbose: Whether to print progress
        fixed_params: Parameters held fixed during optimization

    Returns:
        OptimizationResult
    """
    from ..systems.margin_ttt import MarginTTT

    optimizer = RatingSystemOptimizer(
        MarginTTT,
        dataset,
        train_ratio=train_ratio,
        fixed_params=fixed_params or {},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "sigma": sigma_bounds,
            "beta": beta_bounds,
            "gamma": gamma_bounds,
            "margin_scale": margin_scale_bounds,
            "sigma_margin": sigma_margin_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_trueskill(
    dataset: GameDataset,
    initial_mu_bounds: Tuple[float, float] = (20, 30),
    initial_sigma_bounds: Tuple[float, float] = (5, 12),
    beta_bounds: Tuple[float, float] = (2, 8),
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize TrueSkill parameters.

    Args:
        dataset: Game dataset
        initial_mu_bounds: Bounds for initial skill mean
        initial_sigma_bounds: Bounds for initial skill uncertainty
        beta_bounds: Bounds for performance variability
        maxiter: Maximum optimization iterations
        train_ratio: Fraction of days for initial training (default 0.7)
        method: Optimization method ("differential_evolution" or "L-BFGS-B")
        verbose: Whether to print progress

    Returns:
        OptimizationResult
    """
    from ..systems.trueskill import TrueSkill

    optimizer = RatingSystemOptimizer(
        TrueSkill,
        dataset,
        train_ratio=train_ratio,
        fixed_params={},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "initial_mu": initial_mu_bounds,
            "initial_sigma": initial_sigma_bounds,
            "beta": beta_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_yuksel(
    dataset: GameDataset,
    delta_r_max_bounds: Tuple[float, float] = (50, 500),
    alpha_bounds: Tuple[float, float] = (0.5, 5.0),
    scaling_factor_bounds: Tuple[float, float] = (0.5, 1.0),
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize Yuksel parameters.

    Args:
        dataset: Game dataset
        delta_r_max_bounds: Bounds for max rating change per game
        alpha_bounds: Bounds for uncertainty decay factor
        scaling_factor_bounds: Bounds for update scaling factor
        maxiter: Maximum optimization iterations
        train_ratio: Fraction of days for initial training (default 0.7)
        method: Optimization method ("differential_evolution" or "L-BFGS-B")
        verbose: Whether to print progress

    Returns:
        OptimizationResult
    """
    from ..systems.yuksel import Yuksel

    optimizer = RatingSystemOptimizer(
        Yuksel,
        dataset,
        train_ratio=train_ratio,
        fixed_params={"initial_rating": 1500.0},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "delta_r_max": delta_r_max_bounds,
            "alpha": alpha_bounds,
            "scaling_factor": scaling_factor_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_melo(
    dataset: GameDataset,
    k_bounds: Tuple[float, float] = (4, 100),
    c_factor_bounds: Tuple[float, float] = (0.5, 64),
    initial_rating: float = 1500.0,
    k_dim: int = 1,
    initial_c_std: float = 0.01,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize mElo parameters (k_factor + c_factor; scale fixed at 400).

    Args:
        dataset: Game dataset
        k_bounds: Bounds for K-factor (scalar rating update)
        c_factor_bounds: Bounds for style vector learning rate
        initial_rating: Fixed initial rating
        k_dim: Fixed intransitivity dimension (default: 1)
        initial_c_std: Fixed initial style vector std (default: 0.01)
        maxiter: Maximum optimization iterations
        train_ratio: Fraction of days for initial training
        method: Optimization method
        verbose: Whether to print progress

    Returns:
        OptimizationResult
    """
    from ..systems.melo import MElo

    optimizer = RatingSystemOptimizer(
        MElo,
        dataset,
        train_ratio=train_ratio,
        fixed_params={
            "initial_rating": initial_rating,
            "k_dim": k_dim,
            "initial_c_std": initial_c_std,
        },
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "k_factor": k_bounds,
            "c_factor": c_factor_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_eigenvector(
    dataset: GameDataset,
    half_life_bounds: Tuple[float, float] = (90, 730),
    scale_bounds: Tuple[float, float] = (50, 500),
    centrality_scale_bounds: Tuple[float, float] = (1000, 50000),
    maxiter: int = 20,
    max_test_days: Optional[int] = None,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
    fixed_params: Optional[Dict[str, Any]] = None,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize EigenvectorCentrality parameters.

    Args:
        dataset: Game dataset
        half_life_bounds: Bounds for edge decay half-life (days)
        scale_bounds: Bounds for logistic scale
        centrality_scale_bounds: Bounds for centrality-to-rating multiplier
        maxiter: Maximum optimization iterations
        max_test_days: Limit test period
        train_ratio: Fraction of days for initial training
        method: Optimization method
        verbose: Whether to print progress
        fixed_params: Parameters held fixed during optimization

    Returns:
        OptimizationResult
    """
    from ..systems.eigenvector import EigenvectorCentrality

    optimizer = RatingSystemOptimizer(
        EigenvectorCentrality,
        dataset,
        train_ratio=train_ratio,
        fixed_params=fixed_params or {"recompute_interval": 7, "min_edges": 10},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "half_life": half_life_bounds,
            "scale": scale_bounds,
            "centrality_scale": centrality_scale_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_gelo(
    dataset: GameDataset,
    k_bounds: Tuple[float, float] = (4, 100),
    threshold_spread_bounds: Tuple[float, float] = (2.0, 20.0),
    initial_rating: float = 1500.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize G-Elo parameters (k_factor + threshold_spread; scale fixed at 400).

    Args:
        dataset: Game dataset
        k_bounds: Bounds for K-factor
        threshold_spread_bounds: Bounds for ordinal threshold spacing
        initial_rating: Fixed initial rating
        maxiter: Maximum optimization iterations
        train_ratio: Fraction of days for initial training
        method: Optimization method
        verbose: Whether to print progress

    Returns:
        OptimizationResult
    """
    from ..systems.gelo import GElo

    optimizer = RatingSystemOptimizer(
        GElo,
        dataset,
        train_ratio=train_ratio,
        fixed_params={"initial_rating": initial_rating},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "k_factor": k_bounds,
            "threshold_spread": threshold_spread_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_genelo(
    dataset: GameDataset,
    sigma_bounds: Tuple[float, float] = (30, 200),
    initial_rating: float = 1500.0,
    scale: float = 400.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """Optimise GenElo sigma (Bayesian k-factor); scale and initial_rating fixed."""
    from ..systems.genelo import GenElo

    optimizer = RatingSystemOptimizer(
        GenElo,
        dataset,
        train_ratio=train_ratio,
        fixed_params={
            "initial_rating": initial_rating,
            "scale": scale,
            "use_taylor_k": True,
        },
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={"sigma": sigma_bounds},
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_disc(
    dataset: GameDataset,
    k_bounds: Tuple[float, float] = (4, 100),
    consistency_lr_bounds: Tuple[float, float] = (0.01, 1.0),
    initial_rating: float = 1500.0,
    initial_consistency: float = 1.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """Optimise Disc parameters (k_factor + consistency_lr; scale fixed at 400)."""
    from ..systems.disc import Disc

    optimizer = RatingSystemOptimizer(
        Disc,
        dataset,
        train_ratio=train_ratio,
        fixed_params={
            "initial_rating": initial_rating,
            "initial_consistency": initial_consistency,
        },
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "k_factor": k_bounds,
            "consistency_lr": consistency_lr_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_kickscore(
    dataset: GameDataset,
    var_constant_bounds: Tuple[float, float] = (0.01, 0.5),
    var_matern_bounds: Tuple[float, float] = (0.01, 1.0),
    lscale_bounds: Tuple[float, float] = (30, 2000),
    maxiter: int = 20,
    max_test_days: Optional[int] = None,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
    fixed_params: Optional[Dict[str, Any]] = None,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """Optimize KickScore (GP paired comparisons) parameters.

    Args:
        dataset: Game dataset
        var_constant_bounds: Bounds for baseline skill variance
        var_matern_bounds: Bounds for form-variation variance
        lscale_bounds: Bounds for temporal lengthscale (days)
        maxiter: Maximum optimization iterations
        max_test_days: Limit test period
        train_ratio: Fraction of days for initial training
        method: Optimization method
        verbose: Whether to print progress
        fixed_params: Parameters held fixed during optimization

    Returns:
        OptimizationResult
    """
    from ..systems.kickscore_rating import KickScoreRating

    fp = fixed_params or {}
    fp.setdefault("max_iter", 30)
    fp.setdefault("refit_max_iterations", 1)
    fp.setdefault("tol", 1e-3)
    fp.setdefault("ep_lr", 1.0)

    optimizer = RatingSystemOptimizer(
        KickScoreRating,
        dataset,
        train_ratio=train_ratio,
        fixed_params=fp,
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "var_constant": var_constant_bounds,
            "var_matern": var_matern_bounds,
            "lscale": lscale_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
        x0=x0,
        **kwargs,
    )


def optimize_genelo_surface(
    dataset: GameDataset,
    surfaces: np.ndarray,
    tournament_levels: Optional[np.ndarray] = None,
    is_bo5: Optional[np.ndarray] = None,
    margins: Optional[np.ndarray] = None,
    sigma_hard_bounds: Tuple[float, float] = (30, 200),
    sigma_clay_bounds: Tuple[float, float] = (30, 200),
    sigma_grass_bounds: Tuple[float, float] = (30, 200),
    sigma_indoor_bounds: Tuple[float, float] = (30, 200),
    rho_hard_clay_bounds: Tuple[float, float] = (0.1, 0.99),
    rho_hard_grass_bounds: Tuple[float, float] = (0.1, 0.99),
    rho_hard_indoor_bounds: Tuple[float, float] = (0.1, 0.99),
    rho_clay_grass_bounds: Tuple[float, float] = (0.1, 0.99),
    rho_clay_indoor_bounds: Tuple[float, float] = (0.1, 0.99),
    rho_grass_indoor_bounds: Tuple[float, float] = (0.1, 0.99),
    c1_bounds: Optional[Tuple[float, float]] = None,
    c2_bounds: Optional[Tuple[float, float]] = None,
    sigma_obs_bounds: Optional[Tuple[float, float]] = None,
    sigma_obs_bo5_bounds: Optional[Tuple[float, float]] = None,
    initial_rating: float = 1500.0,
    scale: float = 400.0,
    tournament_sigmas: Optional[Tuple[float, ...]] = None,
    bo5_factor: float = 0.432,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    max_test_days: Optional[int] = None,
    method: str = "differential_evolution",
    verbose: bool = True,
    x0: Optional[np.ndarray] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Optimise GenEloSurface parameters via custom walk-forward backtest.

    Always optimises 10 structural parameters (4 sigmas + 6 correlations).
    When margins are provided and margin bounds are set, also optimises
    up to 4 margin parameters (c1, c2, sigma_obs, sigma_obs_bo5).

    Args:
        dataset: Game dataset.
        surfaces: Surface ID per game (0=hard, 1=clay, 2=grass, 3=indoor_hard).
        tournament_levels: Tournament level per game.
        is_bo5: Best-of-5 indicator per game.
        margins: Margin per game (MARGIN_MISSING sentinel for missing). If None,
            margins are disabled regardless of bounds.
        sigma_hard_bounds..rho_grass_indoor_bounds: Bounds for structural params.
        c1_bounds: Bounds for margin slope. If set (with margins), c1 is optimised.
        c2_bounds: Bounds for margin offset. If set (with margins), c2 is optimised.
        sigma_obs_bounds: Bounds for bo3 margin noise. If set, optimised.
        sigma_obs_bo5_bounds: Bounds for bo5 margin noise. If set, optimised.
        initial_rating: Fixed initial rating.
        scale: Fixed Elo scale.
        tournament_sigmas: Fixed tournament sigmas.
        bo5_factor: Fixed bo5 format multiplier.
        maxiter: Maximum optimisation iterations.
        train_ratio: Fraction of days for initial training.
        max_test_days: Limit test period.
        method: Optimisation method.
        verbose: Whether to print progress.
        x0: Initial parameter guess.

    Returns:
        OptimizationResult with best parameters.
    """
    from ..systems.genelo import GenEloSurface
    from .metrics import brier_score as _brier, log_loss as _ll, accuracy as _acc

    surfaces = np.ascontiguousarray(surfaces, dtype=np.int64)
    n_games = dataset.num_games
    if tournament_levels is None:
        tournament_levels = np.zeros(n_games, dtype=np.int64)
    else:
        tournament_levels = np.ascontiguousarray(tournament_levels, dtype=np.int64)
    if is_bo5 is None:
        is_bo5 = np.zeros(n_games, dtype=np.int64)
    else:
        is_bo5 = np.ascontiguousarray(is_bo5, dtype=np.int64)
    if margins is not None:
        margins = np.ascontiguousarray(margins, dtype=np.float64)

    # Determine which margin params to optimise
    use_margins = margins is not None
    opt_c1 = use_margins and c1_bounds is not None
    opt_c2 = use_margins and c2_bounds is not None
    opt_sobs = use_margins and sigma_obs_bounds is not None
    opt_sobs5 = use_margins and sigma_obs_bo5_bounds is not None

    # Compute train/test split
    days = dataset.days
    split_idx = int(len(days) * train_ratio)
    train_end_day = days[split_idx - 1] if split_idx > 0 else days[0]
    test_end_day = None
    if max_test_days is not None:
        test_days_list = [d for d in days if d > train_end_day]
        if len(test_days_list) > max_test_days:
            test_end_day = test_days_list[max_test_days - 1]

    p1, p2, scores, day_indices, day_offsets = dataset.get_batched_arrays()

    train_dataset = dataset.filter_days(end_day=train_end_day)
    n_train = train_dataset.num_games
    train_surfaces = surfaces[:n_train]
    train_tourn = tournament_levels[:n_train]
    train_bo5 = is_bo5[:n_train]
    train_margins = margins[:n_train] if use_margins else None

    test_days = [d for d in days if d > train_end_day]
    if test_end_day is not None:
        test_days = [d for d in test_days if d <= test_end_day]

    # Build fixed config kwargs
    fixed_kw = dict(
        initial_rating=initial_rating,
        scale=scale,
        bo5_factor=bo5_factor,
    )
    if tournament_sigmas is not None:
        fixed_kw["tournament_sigmas"] = tournament_sigmas

    # Build param names and bounds — structural (always) + margin (optional)
    param_names = [
        "sigma_hard", "sigma_clay", "sigma_grass", "sigma_indoor",
        "rho_hard_clay", "rho_hard_grass", "rho_hard_indoor",
        "rho_clay_grass", "rho_clay_indoor", "rho_grass_indoor",
    ]
    bounds = [
        sigma_hard_bounds, sigma_clay_bounds, sigma_grass_bounds, sigma_indoor_bounds,
        rho_hard_clay_bounds, rho_hard_grass_bounds, rho_hard_indoor_bounds,
        rho_clay_grass_bounds, rho_clay_indoor_bounds, rho_grass_indoor_bounds,
    ]
    if opt_c1:
        param_names.append("c1"); bounds.append(c1_bounds)
    if opt_c2:
        param_names.append("c2"); bounds.append(c2_bounds)
    if opt_sobs:
        param_names.append("sigma_obs"); bounds.append(sigma_obs_bounds)
    if opt_sobs5:
        param_names.append("sigma_obs_bo5"); bounds.append(sigma_obs_bo5_bounds)

    n_structural = 10
    eval_count = 0
    best_ll = float("inf")
    best_params = {}
    best_brier = 1.0
    best_acc = 0.0
    history = []
    start_time = time.time()

    def objective(x):
        nonlocal eval_count, best_ll, best_params, best_brier, best_acc

        x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
        s_hard, s_clay, s_grass, s_indoor = x[:4]
        rho_hc, rho_hg, rho_hi, rho_cg, rho_ci, rho_gi = x[4:10]

        corr_dict = {
            (0, 1): rho_hc, (0, 2): rho_hg, (0, 3): rho_hi,
            (1, 2): rho_cg, (1, 3): rho_ci, (2, 3): rho_gi,
        }

        # Extract margin params from x (after the 10 structural params)
        margin_kw = {}
        idx = n_structural
        if opt_c1:
            margin_kw["c1"] = float(x[idx]); idx += 1
        if opt_c2:
            margin_kw["c2"] = float(x[idx]); idx += 1
        if opt_sobs:
            margin_kw["sigma_obs"] = float(x[idx]); idx += 1
        if opt_sobs5:
            margin_kw["sigma_obs_bo5"] = float(x[idx]); idx += 1

        # If optimising some margin params, fill in defaults for the rest
        if margin_kw:
            margin_kw.setdefault("c1", 0.000144)
            margin_kw.setdefault("c2", 0.0998)
            margin_kw.setdefault("sigma_obs", 0.087)
            margin_kw.setdefault("sigma_obs_bo5", 0.071)
        else:
            margin_kw = dict(c1=None, c2=None, sigma_obs=None, sigma_obs_bo5=None)

        system = GenEloSurface(
            surface_sigmas=(s_hard, s_clay, s_grass, s_indoor),
            surface_correlations=corr_dict,
            **fixed_kw, **margin_kw,
        )
        system.fit(
            train_dataset,
            surfaces=train_surfaces,
            tournament_levels=train_tourn,
            is_bo5=train_bo5,
            margins=train_margins,
        )

        # Walk-forward
        all_preds = []
        all_acts = []
        game_idx = n_train

        for day in test_days:
            try:
                batch = dataset.get_day(day)
            except ValueError:
                continue

            n_day = len(batch)
            day_surf = surfaces[game_idx:game_idx + n_day]
            day_tourn = tournament_levels[game_idx:game_idx + n_day]
            day_bo5 = is_bo5[game_idx:game_idx + n_day]

            preds = system.predict_proba_with_covariates(
                batch.player1, batch.player2,
                day_surf, day_tourn, day_bo5,
            )
            all_preds.append(preds)
            all_acts.append(batch.scores)

            update_kw = dict(surfaces=day_surf, tournament_levels=day_tourn,
                             is_bo5=day_bo5)
            if use_margins:
                update_kw["margins"] = margins[game_idx:game_idx + n_day]
            system.update(batch, **update_kw)
            game_idx += n_day

        if not all_preds:
            return 10.0

        preds = np.concatenate(all_preds)
        acts = np.concatenate(all_acts)
        ll = _ll(preds, acts)
        brier = _brier(preds, acts)
        acc = _acc(preds, acts)

        eval_count += 1
        is_best = ll < best_ll
        if is_best:
            best_ll = ll
            best_params = dict(zip(param_names, x))
            best_brier = brier
            best_acc = acc

        entry = {
            "eval": eval_count,
            "params": dict(zip(param_names, x)),
            "brier": brier,
            "log_loss": ll,
            "accuracy": acc,
            "time": time.time() - start_time,
        }
        history.append(entry)

        if verbose:
            elapsed = time.time() - start_time
            p_str = ", ".join(f"{k}={v:.4f}" if "c1" in k or "sigma_obs" in k
                              else f"{k}={v:.2f}"
                              for k, v in zip(param_names, x))
            marker = " *BEST*" if is_best else ""
            print(
                f"  [{eval_count:3d}] LogLoss={ll:.6f} Brier={brier:.6f} "
                f"Acc={acc:.4f} | {p_str} ({elapsed:.1f}s){marker}"
            )

        return ll

    if verbose:
        n_params = len(param_names)
        n_test = len(test_days)
        print(f"\n{'='*60}")
        print(f"Optimizing GenEloSurface ({n_params} params)")
        print(f"{'='*60}")
        print(f"Method: {method}")
        print(f"Parameters: {param_names}")
        print(f"Max iterations: {maxiter}")
        print(f"Train/test split at day: {train_end_day}")
        print(f"Fixed params: scale={scale}, bo5_factor={bo5_factor}")
        if use_margins:
            from ..systems.genelo import MARGIN_MISSING
            n_valid = int(np.sum(margins < MARGIN_MISSING))
            print(f"Margins: {n_valid}/{n_games} valid ({n_valid/n_games*100:.0f}%)")
        else:
            print(f"Margins: disabled")
        print(f"Dataset: {n_games:,} games, {dataset.num_players:,} players")
        print(f"{'='*60}\n")

    if method == "differential_evolution":
        de_kw = dict(
            bounds=bounds,
            maxiter=maxiter,
            seed=42,
            tol=0.0001,
            polish=True,
            disp=verbose,
        )
        if x0 is not None:
            de_kw["x0"] = x0
        de_kw.update(kwargs)
        result = differential_evolution(objective, **de_kw)
    else:
        if x0 is None:
            x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])
        result = minimize(
            objective, x0, method=method, bounds=bounds,
            options={"maxiter": maxiter, "disp": verbose},
        )

    opt_result = OptimizationResult(
        system_class="GenEloSurface",
        best_params=best_params,
        best_brier=best_brier,
        best_log_loss=best_ll,
        best_accuracy=best_acc,
        n_evaluations=eval_count,
        total_time=time.time() - start_time,
        history=history,
        fixed_params=fixed_kw,
    )

    if verbose:
        print(opt_result.summary())

    return opt_result


def optimize_all(
    dataset: GameDataset,
    systems: Optional[List[str]] = None,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    verbose: bool = True,
) -> Dict[str, OptimizationResult]:
    """
    Optimize all rating systems.

    Args:
        dataset: Game dataset
        systems: List of systems to optimize. Options:
            ["elo", "glicko", "glicko2", "stephenson", "trueskill", "yuksel", "whr", "ttt"]
            If None, optimizes all.
        maxiter: Maximum optimization iterations per system
        train_ratio: Fraction of days for initial training (default 0.7)
        verbose: Whether to print progress

    Returns:
        Dict mapping system name to OptimizationResult
    """
    if systems is None:
        systems = ["elo", "glicko", "glicko2", "stephenson", "trueskill", "yuksel", "whr", "ttt", "melo"]

    results = {}

    optimizers = {
        "elo": optimize_elo,
        "glicko": optimize_glicko,
        "glicko2": optimize_glicko2,
        "stephenson": optimize_stephenson,
        "trueskill": optimize_trueskill,
        "yuksel": optimize_yuksel,
        "whr": optimize_whr,
        "ttt": optimize_ttt,
        "melo": optimize_melo,
    }

    for system in systems:
        if system not in optimizers:
            print(f"Unknown system: {system}")
            continue

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# OPTIMIZING {system.upper()}")
            print(f"{'#'*60}")

        results[system] = optimizers[system](
            dataset,
            maxiter=maxiter,
            train_ratio=train_ratio,
            verbose=verbose,
        )

    return results
