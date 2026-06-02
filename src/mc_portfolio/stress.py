"""Stress-testing utilities for simulated portfolio paths."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .metrics import summarize_simulation


def apply_instant_shock(paths: np.ndarray, shock_pct: float, shock_step: int = 1) -> np.ndarray:
    """Apply an instantaneous market shock from a given step onward.

    ``shock_pct=-0.30`` represents a 30% drop; ``shock_pct=0.10`` represents a 10% jump.
    """
    if shock_pct <= -1:
        raise ValueError("shock_pct must be greater than -100%")
    stressed = np.array(paths, copy=True)
    shock_step = int(np.clip(shock_step, 1, stressed.shape[0] - 1))
    stressed[shock_step:, :] *= 1.0 + shock_pct
    return stressed


def stress_test_paths(paths: np.ndarray, initial_value: float, rf: float = 0.04) -> Dict[str, Dict]:
    """Run common downside/upside stress scenarios against existing simulated paths."""
    scenarios = {
        "base_case": paths,
        "market_crash_20pct": apply_instant_shock(paths, -0.20),
        "market_crash_35pct": apply_instant_shock(paths, -0.35),
        "relief_rally_10pct": apply_instant_shock(paths, 0.10),
    }
    return {name: summarize_simulation(p, initial_value, rf=rf) for name, p in scenarios.items()}
