"""Portfolio optimization utilities for Modern Portfolio Theory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class PortfolioPoint:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    return weights / total


def portfolio_return(weights: np.ndarray, mu_annual: np.ndarray) -> float:
    weights = _normalize_weights(weights)
    return float(weights @ np.asarray(mu_annual, dtype=float))


def portfolio_volatility(weights: np.ndarray, cov_annual: np.ndarray) -> float:
    weights = _normalize_weights(weights)
    cov_annual = np.asarray(cov_annual, dtype=float)
    variance = float(weights @ cov_annual @ weights)
    return float(np.sqrt(max(variance, 0.0)))


def portfolio_sharpe(weights: np.ndarray, mu_annual: np.ndarray, cov_annual: np.ndarray, rf: float = 0.04) -> float:
    vol = portfolio_volatility(weights, cov_annual)
    return float((portfolio_return(weights, mu_annual) - rf) / vol) if vol > 0 else 0.0


def random_long_only_portfolios(
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    n_portfolios: int = 5000,
    rf: float = 0.04,
    seed: int | None = 42,
) -> Dict[str, np.ndarray]:
    """Generate random long-only portfolios for an efficient-frontier cloud.

    Uses a Dirichlet distribution so every portfolio is fully invested and weights sum to 1.
    """
    mu_annual = np.asarray(mu_annual, dtype=float)
    cov_annual = np.asarray(cov_annual, dtype=float)
    if mu_annual.ndim != 1 or cov_annual.shape != (mu_annual.size, mu_annual.size):
        raise ValueError("mu_annual and cov_annual have incompatible shapes")
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(mu_annual.size), size=n_portfolios)
    returns = weights @ mu_annual
    vols = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", weights, cov_annual, weights), 0.0))
    sharpes = np.divide(returns - rf, vols, out=np.zeros_like(returns), where=vols > 0)
    return {"weights": weights, "returns": returns, "volatility": vols, "sharpe": sharpes}


def best_portfolios(frontier: Dict[str, np.ndarray]) -> Dict[str, PortfolioPoint]:
    """Pick max-Sharpe and min-volatility portfolios from generated candidates."""
    max_sharpe_idx = int(np.nanargmax(frontier["sharpe"]))
    min_vol_idx = int(np.nanargmin(frontier["volatility"]))

    def point(i: int) -> PortfolioPoint:
        return PortfolioPoint(
            weights=frontier["weights"][i],
            expected_return=float(frontier["returns"][i]),
            volatility=float(frontier["volatility"][i]),
            sharpe=float(frontier["sharpe"][i]),
        )

    return {"max_sharpe": point(max_sharpe_idx), "min_volatility": point(min_vol_idx)}


def optimize_portfolio_monte_carlo(
    tickers: list[str],
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    n_portfolios: int = 8000,
    rf: float = 0.04,
    seed: int | None = 42,
) -> Dict:
    """Return an optimization summary suitable for JSON/Streamlit display."""
    frontier = random_long_only_portfolios(mu_annual, cov_annual, n_portfolios=n_portfolios, rf=rf, seed=seed)
    best = best_portfolios(frontier)

    def serialise(point: PortfolioPoint) -> Dict:
        return {
            "expected_return": point.expected_return,
            "volatility": point.volatility,
            "sharpe": point.sharpe,
            "weights": {ticker: float(weight) for ticker, weight in zip(tickers, point.weights)},
        }

    return {
        "n_portfolios": n_portfolios,
        "max_sharpe": serialise(best["max_sharpe"]),
        "min_volatility": serialise(best["min_volatility"]),
        "frontier": {
            "returns": frontier["returns"].tolist(),
            "volatility": frontier["volatility"].tolist(),
            "sharpe": frontier["sharpe"].tolist(),
        },
    }
