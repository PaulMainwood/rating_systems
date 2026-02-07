"""
Hyperparameter optimization for rating systems.

Uses scipy.optimize to find optimal parameters by minimizing Brier score
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
    (local optimization) to find parameters that minimize Brier score.

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
        self._best_brier = float('inf')
        self._best_params: Dict[str, float] = {}
        self._best_result: Optional[Any] = None
        self._start_time = 0.0
        self._verbose = False
        self._param_names: List[str] = []

    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function for optimization (minimizes Brier score).

        Args:
            x: Parameter values as array

        Returns:
            Brier score (to minimize)
        """
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
            return 1.0  # Max Brier score

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
        is_best = brier < self._best_brier
        if is_best:
            self._best_brier = brier
            self._best_params = params.copy()
            self._best_result = result

        # Always print every iteration with full details
        if self._verbose:
            elapsed = time.time() - self._start_time
            params_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items()
                                   if k in self._param_names)
            best_marker = " *BEST*" if is_best else ""
            print(f"  [{self._eval_count:3d}] Brier={brier:.6f} LogLoss={log_loss:.6f} "
                  f"Acc={accuracy:.4f} | {params_str} ({elapsed:.1f}s){best_marker}")

        return brier

    def optimize(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        method: str = "differential_evolution",
        maxiter: int = 50,
        verbose: bool = True,
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
            **kwargs: Additional arguments passed to optimizer

        Returns:
            OptimizationResult with best parameters and metrics
        """
        self._verbose = verbose
        self._eval_count = 0
        self._history = []
        self._best_brier = float('inf')
        self._best_params = {}
        self._start_time = time.time()

        self._param_names = list(param_bounds.keys())
        bounds = [param_bounds[name] for name in self._param_names]

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
            # Start from middle of bounds
            x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])

            # For L-BFGS-B, use larger finite-difference step size
            # Default eps (~1e-8) is too small for noisy Brier score objectives
            options = {"maxiter": maxiter, "disp": False}
            if method == "L-BFGS-B":
                # Scale eps based on parameter ranges
                # Use ~2% of parameter range for gradient estimation
                eps_values = np.array([(b[1] - b[0]) * 0.02 for b in bounds])
                options["eps"] = eps_values

            result = minimize(
                self._objective,
                x0,
                method=method,
                bounds=bounds if method in ["L-BFGS-B", "SLSQP"] else None,
                options=options,
                **kwargs,
            )
            best_x = result.x

        total_time = time.time() - self._start_time

        # Build final params dict
        final_params = {name: val for name, val in zip(self._param_names, best_x)}

        # Get metrics from best result
        best_log_loss = self._best_result.log_loss if self._best_result else 0.0
        best_accuracy = self._best_result.accuracy if self._best_result else 0.0

        opt_result = OptimizationResult(
            system_class=self.system_class.__name__,
            best_params=final_params,
            best_brier=self._best_brier,
            best_log_loss=best_log_loss,
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
    k_bounds: Tuple[float, float] = (4, 64),
    scale_bounds: Tuple[float, float] = (200, 600),
    initial_rating: float = 1500.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize Elo parameters.

    Args:
        dataset: Game dataset
        k_bounds: Bounds for K-factor
        scale_bounds: Bounds for scale (logistic parameter)
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
    )

    return optimizer.optimize(
        param_bounds={
            "k_factor": k_bounds,
            "scale": scale_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
    )


def optimize_glicko(
    dataset: GameDataset,
    initial_rd_bounds: Tuple[float, float] = (200, 500),
    rd_decay_bounds: Tuple[float, float] = (10, 100),
    c_bounds: Tuple[float, float] = (20, 100),
    initial_rating: float = 1500.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize Glicko parameters.

    Args:
        dataset: Game dataset
        initial_rd_bounds: Bounds for initial rating deviation
        rd_decay_bounds: Bounds for RD decay rate
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
    )

    return optimizer.optimize(
        param_bounds={
            "initial_rd": initial_rd_bounds,
            "c": c_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
    )


def optimize_glicko2(
    dataset: GameDataset,
    initial_rd_bounds: Tuple[float, float] = (200, 500),
    initial_volatility_bounds: Tuple[float, float] = (0.03, 0.09),
    tau_bounds: Tuple[float, float] = (0.3, 1.2),
    initial_rating: float = 1500.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize Glicko-2 parameters.

    Args:
        dataset: Game dataset
        initial_rd_bounds: Bounds for initial rating deviation
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
    )

    return optimizer.optimize(
        param_bounds={
            "initial_rd": initial_rd_bounds,
            "initial_volatility": initial_volatility_bounds,
            "tau": tau_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
    )


def optimize_stephenson(
    dataset: GameDataset,
    initial_rd_bounds: Tuple[float, float] = (200, 500),
    cval_bounds: Tuple[float, float] = (5, 50),
    hval_bounds: Tuple[float, float] = (0, 30),
    lambda_bounds: Tuple[float, float] = (0, 10),
    initial_rating: float = 1500.0,
    maxiter: int = 30,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize Stephenson parameters.

    Args:
        dataset: Game dataset
        initial_rd_bounds: Bounds for initial rating deviation
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
    )

    return optimizer.optimize(
        param_bounds={
            "initial_rd": initial_rd_bounds,
            "cval": cval_bounds,
            "hval": hval_bounds,
            "lambda_param": lambda_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
    )


def optimize_whr(
    dataset: GameDataset,
    w2_bounds: Tuple[float, float] = (10, 1000),
    maxiter: int = 20,
    max_test_days: Optional[int] = None,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
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

    Returns:
        OptimizationResult
    """
    from ..systems.whr import WHR

    optimizer = RatingSystemOptimizer(
        WHR,
        dataset,
        train_ratio=train_ratio,
        fixed_params={"max_iterations": 100, "refit_max_iterations": 60},
        max_test_days=max_test_days,
    )

    return optimizer.optimize(
        param_bounds={
            "w2": w2_bounds,
        },
        method=method,
        maxiter=maxiter,
        verbose=verbose,
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

    Returns:
        OptimizationResult
    """
    from ..systems.trueskill_through_time import TrueSkillThroughTime

    optimizer = RatingSystemOptimizer(
        TrueSkillThroughTime,
        dataset,
        train_ratio=train_ratio,
        fixed_params={
            "max_iterations": 10,
            "refit_max_iterations": 1,
            "refit_interval": 1,  # Refit daily for accurate evaluation
        },
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
    )


def optimize_trueskill(
    dataset: GameDataset,
    initial_mu_bounds: Tuple[float, float] = (20, 30),
    initial_sigma_bounds: Tuple[float, float] = (5, 12),
    beta_bounds: Tuple[float, float] = (2, 8),
    maxiter: int = 30,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
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
    )


def optimize_yuksel(
    dataset: GameDataset,
    delta_r_max_bounds: Tuple[float, float] = (50, 500),
    alpha_bounds: Tuple[float, float] = (0.5, 5.0),
    scaling_factor_bounds: Tuple[float, float] = (0.5, 1.0),
    maxiter: int = 30,
    train_ratio: float = 0.7,
    method: str = "differential_evolution",
    verbose: bool = True,
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
    )


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
        systems = ["elo", "glicko", "glicko2", "stephenson", "trueskill", "yuksel", "whr", "ttt"]

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
