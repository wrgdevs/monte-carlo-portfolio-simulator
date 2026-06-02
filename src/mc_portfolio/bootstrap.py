"""Historical bootstrap simulator that resamples real returns instead of assuming GBM."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .params_estimator import log_returns


def simulate_bootstrap_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
    initial_value: float = 10_000.0,
    years: float = 1.0,
    steps_per_year: int = 252,
    n_sims: int = 10_000,
    random_seed: int | None = 42,
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, int]:
    """Resample historical daily portfolio log returns to create future paths."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    returns = log_returns(prices).dropna()
    portfolio_log_returns = returns.to_numpy() @ weights
    n_steps = max(1, int(round(years * steps_per_year)))
    rng = np.random.default_rng(random_seed)
    sampled = rng.choice(portfolio_log_returns, size=(n_steps, n_sims), replace=True).astype(dtype, copy=False)
    log_paths = np.empty((n_steps + 1, n_sims), dtype=dtype)
    log_paths[0] = 0.0
    np.cumsum(sampled, axis=0, out=log_paths[1:])
    return (initial_value * np.exp(log_paths)).astype(dtype, copy=False), n_steps
