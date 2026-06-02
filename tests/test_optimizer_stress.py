import numpy as np

from mc_portfolio.optimizer import optimize_portfolio_monte_carlo, portfolio_volatility
from mc_portfolio.stress import apply_instant_shock


def test_optimizer_weights_sum_to_one():
    mu = np.array([0.08, 0.12, 0.04])
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.01]])
    result = optimize_portfolio_monte_carlo(["A", "B", "C"], mu, cov, n_portfolios=500, rf=0.03, seed=1)
    weights = result["max_sharpe"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert result["min_volatility"]["volatility"] <= max(result["frontier"]["volatility"])


def test_portfolio_volatility_positive():
    vol = portfolio_volatility(np.array([0.5, 0.5]), np.array([[0.04, 0.01], [0.01, 0.09]]))
    assert vol > 0


def test_apply_instant_shock():
    paths = np.array([[100, 100], [110, 90], [120, 80]], dtype=float)
    stressed = apply_instant_shock(paths, -0.20, shock_step=1)
    assert np.allclose(stressed[0], [100, 100])
    assert np.allclose(stressed[1], [88, 72])
    assert np.allclose(stressed[2], [96, 64])
