"""Historical market data helpers."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import DEFAULT_HISTORY_PERIOD, MIN_HISTORY_DAYS


def fetch_price_history(ticker: str, period: str = DEFAULT_HISTORY_PERIOD, interval: str = "1d") -> Optional[pd.Series]:
    """Fetch adjusted close price series for one ticker."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    except Exception:
        return None
    if hist is None or hist.empty:
        return None
    col = "Adj Close" if "Adj Close" in hist.columns else "Close" if "Close" in hist.columns else None
    if col is None:
        return None
    prices = hist[col].dropna().astype(float)
    if prices.shape[0] < MIN_HISTORY_DAYS:
        return None
    prices.name = ticker.upper()
    return prices


def fetch_price_matrix(tickers: list[str], period: str = DEFAULT_HISTORY_PERIOD, interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch adjusted close prices for multiple tickers as one aligned DataFrame."""
    clean = [t.strip().upper() for t in tickers if t.strip()]
    if not clean:
        return None
    try:
        import yfinance as yf
        raw = yf.download(clean, period=period, interval=interval, auto_adjust=False, progress=False, group_by="column")
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"] if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"]
    else:
        prices = raw[["Adj Close"]] if "Adj Close" in raw.columns else raw[["Close"]]
        prices.columns = clean[:1]
    prices = prices.dropna(how="all").ffill().dropna()
    if prices.shape[0] < MIN_HISTORY_DAYS:
        return None
    return prices.astype(float)
