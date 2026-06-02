import numpy as np

from mc_portfolio.metrics import max_drawdown_per_sim, summarize_simulation


def test_vectorized_drawdown():
    paths = np.array([[100, 100], [120, 80], [90, 70], [130, 90]], dtype=float)
    dd = max_drawdown_per_sim(paths)
    assert np.allclose(dd, [-0.25, -0.30])


def test_summary_contains_new_metrics():
    paths = np.array([[100, 100, 100], [110, 90, 120], [115, 80, 130]], dtype=float)
    summary = summarize_simulation(paths, 100, mu_annual=0.08, sigma_annual=0.2)
    assert "sortino_ratio" in summary
    assert "var_loss_pct" in summary
    assert summary["expected_final_price"] > 0
