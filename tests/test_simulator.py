import numpy as np
from mc_portfolio.simulator import simulate_gbm


def test_simulate_basic_shape():
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 252
    n_sims = 100
    paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_sims, random_seed=123)
    assert paths.shape == (n_steps + 1, n_sims)
    assert np.all(paths > 0)