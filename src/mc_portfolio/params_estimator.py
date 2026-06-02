from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def estimate_params_from_prices(prices: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR) -> Dict:
    """Estimate annualized drift and volatility from log returns."""
    log_r = log_returns(prices)
    mu_daily = float(log_r.mean())
    sigma_daily = float(log_r.std(ddof=1))
    return {
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
        "mu_annual": mu_daily * trading_days,
        "sigma_annual": sigma_daily * np.sqrt(trading_days),
        "last_price": float(prices.iloc[-1]),
        "n_obs": int(log_r.shape[0]),
    }


def estimate_portfolio_params(prices: pd.DataFrame, trading_days: int = TRADING_DAYS_PER_YEAR) -> Dict:
    """Estimate per-asset expected returns plus annualized covariance/correlation."""
    log_r = log_returns(prices)
    return {
        "tickers": list(prices.columns),
        "last_prices": prices.iloc[-1].astype(float).to_numpy(),
        "mu_annual": (log_r.mean() * trading_days).astype(float).to_numpy(),
        "cov_annual": (log_r.cov() * trading_days).astype(float).to_numpy(),
        "corr": log_r.corr(),
        "n_obs": int(log_r.shape[0]),
    }
