"""Risk and performance metrics for Monte Carlo price paths."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def final_returns(paths: np.ndarray, initial_value: float) -> np.ndarray:
    return paths[-1, :] / initial_value - 1.0


def expected_final_price(paths: np.ndarray) -> float:
    return float(np.mean(paths[-1, :]))


def prob_of_loss(paths: np.ndarray, initial_value: float) -> float:
    return float(np.mean(paths[-1, :] < initial_value))


def std_final_prices(paths: np.ndarray) -> float:
    return float(np.std(paths[-1, :], ddof=1))


def var_historical(values: np.ndarray, alpha: float = 0.05) -> float:
    """Lower-tail percentile of simulated final values."""
    return float(np.percentile(values, 100 * alpha))


def cvar_historical(values: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional VaR / expected shortfall: average of the worst alpha outcomes."""
    if values.size == 0:
        return 0.0
    cutoff = np.percentile(values, 100 * alpha)
    tail = values[values <= cutoff]
    return float(np.mean(tail)) if tail.size else float(cutoff)


def sharpe_ratio(mu_annual: float, sigma_annual: float, rf: float = 0.04) -> float:
    return float((mu_annual - rf) / sigma_annual) if sigma_annual > 0 else 0.0


def sortino_ratio(returns: np.ndarray, rf_per_period: float = 0.0) -> float:
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    downside_std = np.std(downside, ddof=1) if downside.size > 1 else 0.0
    return float(np.mean(excess) / downside_std) if downside_std > 0 else 0.0


def max_drawdown(path: np.ndarray) -> float:
    running_max = np.maximum.accumulate(path)
    drawdowns = (path - running_max) / running_max
    return float(np.min(drawdowns))


def max_drawdown_per_sim(paths: np.ndarray) -> np.ndarray:
    """Vectorized max drawdown for each simulation path.

    This replaces a Python loop and is much faster for large simulation counts.
    """
    running_max = np.maximum.accumulate(paths, axis=0)
    drawdowns = (paths - running_max) / running_max
    return np.min(drawdowns, axis=0)


def summarize_simulation(
    paths: np.ndarray,
    S0: float,
    alpha: float = 0.05,
    mu_annual: Optional[float] = None,
    sigma_annual: Optional[float] = None,
    rf: float = 0.04,
) -> Dict:
    finals = paths[-1, :]
    returns = final_returns(paths, S0)
    drawdowns = max_drawdown_per_sim(paths)
    var_value = var_historical(finals, alpha=alpha)
    cvar_value = cvar_historical(finals, alpha=alpha)
    sharpe = sharpe_ratio(mu_annual, sigma_annual, rf) if mu_annual is not None and sigma_annual is not None else 0.0

    return {
        "expected_final_price": expected_final_price(paths),
        "expected_return": float(np.mean(returns)),
        "probability_of_loss": prob_of_loss(paths, S0),
        "std_final_price": std_final_prices(paths),
        "median_final_price": float(np.median(finals)),
        "var_historical": var_value,
        "cvar_95": cvar_value,
        "var_loss_pct": float((S0 - var_value) / S0),
        "cvar_loss_pct": float((S0 - cvar_value) / S0),
        "upside_probability_20pct": float(np.mean(returns >= 0.20)),
        "percentiles": {
            "1": float(np.percentile(finals, 1)),
            "5": float(np.percentile(finals, 5)),
            "25": float(np.percentile(finals, 25)),
            "50": float(np.percentile(finals, 50)),
            "75": float(np.percentile(finals, 75)),
            "95": float(np.percentile(finals, 95)),
            "99": float(np.percentile(finals, 99)),
        },
        "max_drawdown": float(np.min(drawdowns)),
        "mean_max_drawdown": float(np.mean(drawdowns)),
        "worst_max_drawdown": float(np.min(drawdowns)),
        "std_max_drawdown": float(np.std(drawdowns, ddof=1)),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino_ratio(returns, rf_per_period=rf),
    }
