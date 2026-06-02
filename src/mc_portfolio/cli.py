import argparse
import json
import os
import numpy as np
from typing import Optional

from .data_fetcher import fetch_price_history, fetch_price_matrix
from .params_estimator import estimate_params_from_prices, estimate_portfolio_params
from .simulator import simulate_gbm_by_steps_per_year, simulate_correlated_portfolio
from .metrics import summarize_simulation
from .optimizer import optimize_portfolio_monte_carlo
from .stress import stress_test_paths
from .bootstrap import simulate_bootstrap_portfolio
from .report import write_html_report
from .viz import plot_sample_paths, plot_final_histogram, plot_max_drawdown_histogram
from .utils import validate_ticker, ensure_dir
from .config import (
    DEFAULT_N_SIMS,
    DEFAULT_N_STEPS_PER_YEAR,
    OUTPUT_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_BENCHMARK,
)


def prompt_manual_params() -> dict:
    """Prompt user for manual parameters (used only in CLI interactive mode)."""
    print("Enter manual parameters (press Enter to use defaults where shown):")
    S0 = float(input("Initial price S0: "))
    mu_a = float(input("Annual expected return mu (e.g. 0.08 for 8%): "))
    sigma_a = float(input("Annual volatility sigma (e.g. 0.25 for 25%): "))
    years = float(input("Time horizon in years (e.g. 1.0): "))
    sims = int(input("Number of simulations (e.g. 5000): "))
    steps_per_year = int(input("Steps per year (e.g. 252): "))
    return {
        "S0": S0,
        "mu_annual": mu_a,
        "sigma_annual": sigma_a,
        "years": years,
        "sims": sims,
        "steps_per_year": steps_per_year,
    }


def run_from_ticker(
    ticker: str,
    sims: int = DEFAULT_N_SIMS,
    years: float = 1.0,
    steps_per_year: int = DEFAULT_N_STEPS_PER_YEAR,
    seed: Optional[int] = DEFAULT_RANDOM_SEED,
    rf: float = DEFAULT_RISK_FREE_RATE,
    benchmark: Optional[str] = DEFAULT_BENCHMARK,
):
    """Run full simulation using real historical data from a ticker."""
    prices = fetch_price_history(ticker)
    if prices is None:
        return None, "fetch_failed"

    params = estimate_params_from_prices(prices)
    S0 = params["last_price"]
    mu_annual = params["mu_annual"]
    sigma_annual = params["sigma_annual"]

    paths, n_steps = simulate_gbm_by_steps_per_year(
        S0, mu_annual, sigma_annual, years, steps_per_year, sims, random_seed=seed
    )

    # Benchmark comparison (only in ticker mode)
    benchmark_info = None
    if benchmark and benchmark.upper() != ticker.upper():
        bench_prices = fetch_price_history(benchmark)
        if bench_prices is not None:
            bench_params = estimate_params_from_prices(bench_prices)
            benchmark_info = {
                "ticker": benchmark,
                "mu_annual": bench_params["mu_annual"],
                "sigma_annual": bench_params["sigma_annual"],
            }

    ensure_dir(OUTPUT_DIR)
    p1 = plot_sample_paths(paths, title=f"{ticker} Simulated Price Paths")
    p2 = plot_final_histogram(paths, title=f"{ticker} Final Price Distribution")
    p3 = plot_max_drawdown_histogram(paths, title=f"{ticker} Max Drawdown Distribution")

    summary = summarize_simulation(
        paths, S0, mu_annual=mu_annual, sigma_annual=sigma_annual, rf=rf
    )
    metadata = {
        "ticker": ticker,
        "S0": S0,
        "mu_annual": mu_annual,
        "sigma_annual": sigma_annual,
        "n_obs": params["n_obs"],
        "n_steps": n_steps,
        "sims": sims,
        "plots": [p1, p2, p3],
        "benchmark": benchmark_info,
        "rf_rate": rf,
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump({"metadata": metadata, "summary": summary}, f, indent=2)

    return {"metadata": metadata, "summary": summary}, "ok"


def run_manual(
    S0: float,
    mu_annual: float,
    sigma_annual: float,
    years: float = 1.0,
    sims: int = DEFAULT_N_SIMS,
    steps_per_year: int = DEFAULT_N_STEPS_PER_YEAR,
    seed: Optional[int] = DEFAULT_RANDOM_SEED,
    rf: float = DEFAULT_RISK_FREE_RATE,
    benchmark: Optional[str] = None,  # ignored in manual mode
):
    """Run simulation with fully manual parameters (used by CLI and Streamlit)."""
    paths, n_steps = simulate_gbm_by_steps_per_year(
        S0, mu_annual, sigma_annual, years, steps_per_year, sims, random_seed=seed
    )

    ensure_dir(OUTPUT_DIR)
    p1 = plot_sample_paths(paths, title=f"Manual Simulation Paths (S0={S0:.2f})")
    p2 = plot_final_histogram(paths, title=f"Manual Final Price Distribution (S0={S0:.2f})")
    p3 = plot_max_drawdown_histogram(paths, title=f"Manual Max Drawdown Distribution (S0={S0:.2f})")

    summary = summarize_simulation(
        paths, S0, mu_annual=mu_annual, sigma_annual=sigma_annual, rf=rf
    )
    metadata = {
        "mode": "manual",
        "S0": S0,
        "mu_annual": mu_annual,
        "sigma_annual": sigma_annual,
        "years": years,
        "n_steps": n_steps,
        "sims": sims,
        "seed": seed,
        "plots": [p1, p2, p3],
        "benchmark": None,
        "rf_rate": rf,
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump({"metadata": metadata, "summary": summary}, f, indent=2)

    return {"metadata": metadata, "summary": summary}, "ok"


def run_portfolio(
    tickers: list[str],
    weights: list[float],
    initial_value: float = 10_000.0,
    sims: int = DEFAULT_N_SIMS,
    years: float = 1.0,
    steps_per_year: int = DEFAULT_N_STEPS_PER_YEAR,
    seed: Optional[int] = DEFAULT_RANDOM_SEED,
    rf: float = DEFAULT_RISK_FREE_RATE,
    include_optimization: bool = True,
    include_stress_tests: bool = True,
    include_bootstrap: bool = False,
    frontier_portfolios: int = 8000,
):
    """Run a correlated multi-asset portfolio simulation from historical data."""
    prices = fetch_price_matrix(tickers)
    if prices is None:
        return None, "fetch_failed"
    params = estimate_portfolio_params(prices)
    values, n_steps = simulate_correlated_portfolio(
        params["last_prices"],
        np.array(weights, dtype=float),
        params["mu_annual"],
        params["cov_annual"],
        years,
        steps_per_year,
        sims,
        initial_value=initial_value,
        random_seed=seed,
    )
    ensure_dir(OUTPUT_DIR)
    p1 = plot_sample_paths(values, title="Portfolio Simulated Value Paths")
    p2 = plot_final_histogram(values, title="Portfolio Final Value Distribution")
    p3 = plot_max_drawdown_histogram(values, title="Portfolio Max Drawdown Distribution")
    norm_weights = np.array(weights, dtype=float) / np.sum(weights)
    portfolio_mu = float(np.dot(norm_weights, params["mu_annual"]))
    portfolio_sigma = float(np.sqrt(norm_weights @ params["cov_annual"] @ norm_weights))
    summary = summarize_simulation(values, initial_value, mu_annual=portfolio_mu, sigma_annual=portfolio_sigma, rf=rf)
    metadata = {
        "mode": "portfolio",
        "tickers": params["tickers"],
        "weights": norm_weights.round(6).tolist(),
        "initial_value": initial_value,
        "mu_annual": portfolio_mu,
        "sigma_annual": portfolio_sigma,
        "asset_mu_annual": params["mu_annual"].tolist(),
        "n_obs": params["n_obs"],
        "n_steps": n_steps,
        "sims": sims,
        "seed": seed,
        "plots": [p1, p2, p3],
        "rf_rate": rf,
    }
    result = {"metadata": metadata, "summary": summary}
    if include_optimization and len(params["tickers"]) >= 2:
        result["optimization"] = optimize_portfolio_monte_carlo(
            params["tickers"], params["mu_annual"], params["cov_annual"], n_portfolios=frontier_portfolios, rf=rf, seed=seed
        )
    if include_stress_tests:
        result["stress_tests"] = stress_test_paths(values, initial_value, rf=rf)
    if include_bootstrap:
        bootstrap_paths, _ = simulate_bootstrap_portfolio(prices, norm_weights, initial_value, years, steps_per_year, sims, seed)
        result["bootstrap_summary"] = summarize_simulation(bootstrap_paths, initial_value, rf=rf)
    result["metadata"]["report"] = write_html_report(result)
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result, "ok"


def main(argv=None):
    parser = argparse.ArgumentParser("Monte Carlo Portfolio Simulator CLI")
    parser.add_argument("--ticker", type=str, help="Ticker symbol. Omit for manual mode.")
    parser.add_argument("--portfolio", type=str, help="Comma-separated tickers, e.g. AAPL,MSFT,SPY")
    parser.add_argument("--weights", type=str, help="Comma-separated weights, e.g. 0.4,0.4,0.2")
    parser.add_argument("--initial_value", type=float, default=10000.0)
    parser.add_argument("--sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--steps_per_year", type=int, default=DEFAULT_N_STEPS_PER_YEAR)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--rf", type=float, default=DEFAULT_RISK_FREE_RATE, help="Risk-free rate (e.g. 0.04)")
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK, help="Benchmark ticker for comparison (e.g. SPY)")
    parser.add_argument("--bootstrap", action="store_true", help="Also run historical bootstrap comparison in portfolio mode")
    parser.add_argument("--no_optimization", action="store_true", help="Skip efficient-frontier optimization in portfolio mode")
    parser.add_argument("--no_stress", action="store_true", help="Skip stress tests in portfolio mode")
    parser.add_argument("--frontier_portfolios", type=int, default=8000, help="Random portfolios sampled for efficient frontier")

    args = parser.parse_args(argv)
    ensure_dir(OUTPUT_DIR)

    if args.portfolio:
        tickers = [t.strip() for t in args.portfolio.split(",") if t.strip()]
        weights = [float(x) for x in (args.weights or "").split(",") if x.strip()]
        if len(weights) != len(tickers):
            weights = [1 / len(tickers)] * len(tickers)
        result, status = run_portfolio(
            tickers, weights, args.initial_value, args.sims, args.years, args.steps_per_year, args.seed,
            rf=args.rf, include_optimization=not args.no_optimization, include_stress_tests=not args.no_stress,
            include_bootstrap=args.bootstrap, frontier_portfolios=args.frontier_portfolios
        )
    elif args.ticker:
        if not validate_ticker(args.ticker):
            print(f"Invalid ticker: {args.ticker}")
            return
        result, status = run_from_ticker(
            args.ticker,
            args.sims,
            args.years,
            args.steps_per_year,
            args.seed,
            rf=args.rf,
            benchmark=args.benchmark,
        )
    else:
        print("Manual mode activated.")
        params = prompt_manual_params()
        result, status = run_manual(
            params["S0"],
            params["mu_annual"],
            params["sigma_annual"],
            params["years"],
            params["sims"],
            params["steps_per_year"],
            args.seed,
            rf=args.rf,
        )

    if status == "ok":
        print("✅ Simulation completed successfully!")
        print(json.dumps(result["summary"], indent=2))
        print(f"\nPlots and summary.json saved to: {OUTPUT_DIR}")
    else:
        print("❌ Failed to fetch data for ticker. Try another ticker or manual mode.")


if __name__ == "__main__":
    main()