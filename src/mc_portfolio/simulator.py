"""Monte Carlo simulation engines."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _validate_inputs(S0: float, sigma_annual: float, T_years: float, n_steps: int, n_sims: int) -> None:
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if sigma_annual < 0:
        raise ValueError("sigma_annual cannot be negative")
    if T_years <= 0 or n_steps <= 0 or n_sims <= 0:
        raise ValueError("T_years, n_steps and n_sims must be positive")


def simulate_gbm(
    S0: float,
    mu_annual: float,
    sigma_annual: float,
    T_years: float,
    n_steps: int,
    n_sims: int,
    random_seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Vectorized Geometric Brownian Motion simulator.

    Returns an array shaped ``(n_steps + 1, n_sims)``. Use ``dtype=np.float32`` from
    the UI for larger runs with lower memory usage.
    """
    _validate_inputs(S0, sigma_annual, T_years, n_steps, n_sims)
    rng = np.random.default_rng(random_seed)
    dt = T_years / n_steps
    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    diffusion = sigma_annual * np.sqrt(dt)
    increments = rng.normal(drift, diffusion, size=(n_steps, n_sims)).astype(dtype, copy=False)
    log_paths = np.empty((n_steps + 1, n_sims), dtype=dtype)
    log_paths[0] = 0.0
    np.cumsum(increments, axis=0, out=log_paths[1:])
    return (S0 * np.exp(log_paths)).astype(dtype, copy=False)


def simulate_gbm_by_steps_per_year(
    S0: float,
    mu_annual: float,
    sigma_annual: float,
    years: float,
    steps_per_year: int,
    n_sims: int,
    random_seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, int]:
    n_steps = max(1, int(round(steps_per_year * years)))
    paths = simulate_gbm(S0, mu_annual, sigma_annual, years, n_steps, n_sims, random_seed, dtype=dtype)
    return paths, n_steps


def simulate_correlated_portfolio(
    initial_prices: np.ndarray,
    weights: np.ndarray,
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    years: float,
    steps_per_year: int,
    n_sims: int,
    initial_value: float = 10_000.0,
    random_seed: Optional[int] = None,
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, int]:
    """Simulate a multi-asset buy-and-hold portfolio with correlated GBM returns."""
    initial_prices = np.asarray(initial_prices, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mu_annual = np.asarray(mu_annual, dtype=float)
    cov_annual = np.asarray(cov_annual, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()
    n_assets = len(weights)
    if initial_prices.shape[0] != n_assets or mu_annual.shape[0] != n_assets or cov_annual.shape != (n_assets, n_assets):
        raise ValueError("initial_prices, weights, mu_annual and cov_annual have incompatible shapes")

    n_steps = max(1, int(round(steps_per_year * years)))
    dt = years / n_steps
    rng = np.random.default_rng(random_seed)
    chol = np.linalg.cholesky(cov_annual * dt + np.eye(n_assets) * 1e-12)
    drift = (mu_annual - 0.5 * np.diag(cov_annual)) * dt
    z = rng.standard_normal(size=(n_steps, n_sims, n_assets))
    increments = drift + np.einsum("tSA,BA->tSB", z, chol).astype(dtype, copy=False)
    log_paths = np.concatenate([np.zeros((1, n_sims, n_assets), dtype=dtype), np.cumsum(increments, axis=0)])
    price_paths = initial_prices * np.exp(log_paths)
    shares = initial_value * weights / initial_prices
    values = np.einsum("tsa,a->ts", price_paths, shares).astype(dtype, copy=False)
    return values, n_steps
