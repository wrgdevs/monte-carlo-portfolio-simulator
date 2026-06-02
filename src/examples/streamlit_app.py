import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
try:
    import plotly.express as px
except Exception:  # Plotly is optional for non-dashboard use
    px = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_portfolio.cli import run_from_ticker, run_manual, run_portfolio
from mc_portfolio.config import OUTPUT_DIR

st.set_page_config(page_title="Monte Carlo Portfolio Simulator", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.16); padding: 14px; border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Monte Carlo Portfolio Simulator")
st.caption("Historical-data calibration • correlated portfolio simulation • VaR/CVaR • drawdown analysis • downloadable JSON")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Simulation Mode", ["Single ticker", "Portfolio", "Manual"], index=1)
    sims = st.slider("Simulations", 1_000, 100_000, 10_000, step=1_000, help="Higher values give smoother distributions but use more memory.")
    years = st.number_input("Time Horizon (years)", 0.1, 10.0, 1.0, step=0.1)
    steps_per_year = st.slider("Steps per Year", 12, 500, 252)
    seed = st.number_input("Random Seed", 0, 999_999, 42)
    rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 15.0, 4.0, step=0.1) / 100
    st.divider()

    if mode == "Single ticker":
        ticker = st.text_input("Ticker", "AAPL").upper().strip()
        benchmark = st.text_input("Benchmark", "SPY").upper().strip()
        run_button = st.button("Run Single-Ticker Simulation", type="primary", use_container_width=True)
    elif mode == "Portfolio":
        tickers_text = st.text_input("Tickers", "AAPL, MSFT, SPY")
        weights_text = st.text_input("Weights", "0.40, 0.40, 0.20")
        initial_value = st.number_input("Initial Portfolio Value", min_value=100.0, value=10_000.0, step=500.0)
        frontier_portfolios = st.slider("Efficient Frontier Samples", 500, 20_000, 8_000, step=500)
        include_bootstrap = st.checkbox("Run historical bootstrap comparison", value=False)
        include_stress = st.checkbox("Run stress tests", value=True)
        include_optimization = st.checkbox("Run portfolio optimization", value=True)
        benchmark = None
        run_button = st.button("Run Portfolio Simulation", type="primary", use_container_width=True)
    else:
        S0 = st.number_input("Initial Price / Value", min_value=0.01, value=100.0, step=1.0)
        mu = st.number_input("Annual Expected Return", value=0.08, step=0.01, format="%.3f")
        sigma = st.number_input("Annual Volatility", min_value=0.0, value=0.25, step=0.01, format="%.3f")
        benchmark = None
        run_button = st.button("Run Manual Simulation", type="primary", use_container_width=True)


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [x.strip().upper() for x in text.split(",") if x.strip()]


if run_button:
    try:
        with st.spinner("Running simulation and generating charts..."):
            if mode == "Single ticker":
                result, status = run_from_ticker(ticker, sims, years, steps_per_year, seed, rf=rf_rate, benchmark=benchmark)
            elif mode == "Portfolio":
                tickers = parse_csv_strings(tickers_text)
                weights = parse_csv_floats(weights_text)
                if len(tickers) != len(weights):
                    st.warning("Ticker/weight counts did not match, so equal weights were used.")
                    weights = [1 / len(tickers)] * len(tickers)
                result, status = run_portfolio(
                    tickers, weights, initial_value, sims, years, steps_per_year, seed, rf=rf_rate,
                    include_optimization=include_optimization, include_stress_tests=include_stress,
                    include_bootstrap=include_bootstrap, frontier_portfolios=frontier_portfolios
                )
            else:
                result, status = run_manual(S0, mu, sigma, years, sims, steps_per_year, seed, rf=rf_rate)
    except Exception as exc:
        result, status = None, "error"
        st.exception(exc)

    if status == "ok":
        meta = result["metadata"]
        summ = result["summary"]
        st.success("Simulation complete")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected Final", f"${summ['expected_final_price']:,.2f}", f"{summ['expected_return']*100:.1f}%")
        c2.metric("Probability of Loss", f"{summ['probability_of_loss']*100:.1f}%")
        c3.metric("VaR 95%", f"${summ['var_historical']:,.2f}", f"-{summ['var_loss_pct']*100:.1f}%")
        c4.metric("CVaR 95%", f"${summ['cvar_95']:,.2f}", f"-{summ['cvar_loss_pct']*100:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Sharpe", f"{summ['sharpe_ratio']:.2f}")
        c6.metric("Sortino", f"{summ['sortino_ratio']:.2f}")
        c7.metric("Mean Max Drawdown", f"{summ['mean_max_drawdown']*100:.1f}%")
        c8.metric("Worst Drawdown", f"{summ['worst_max_drawdown']*100:.1f}%")

        extra_tabs = []
        if result.get("optimization"):
            extra_tabs.append("Optimization")
        if result.get("stress_tests"):
            extra_tabs.append("Stress Tests")
        if result.get("bootstrap_summary"):
            extra_tabs.append("Bootstrap")
        tab_objs = st.tabs(["Paths", "Final Distribution", "Drawdowns", "Details"] + extra_tabs)
        tab1, tab2, tab3, tab4 = tab_objs[:4]
        with tab1:
            st.image(meta["plots"][0], use_container_width=True)
        with tab2:
            st.image(meta["plots"][1], use_container_width=True)
        with tab3:
            st.image(meta["plots"][2], use_container_width=True)
        with tab4:
            pct = pd.DataFrame.from_dict(summ["percentiles"], orient="index", columns=["Final value"])
            st.subheader("Percentiles")
            st.dataframe(pct.style.format("${:,.2f}"), use_container_width=True)
            st.subheader("Metadata")
            st.json(meta)
            if meta.get("benchmark"):
                b = meta["benchmark"]
                st.info(f"Benchmark {b['ticker']}: {b['mu_annual']*100:.1f}% annualized return, {b['sigma_annual']*100:.1f}% annualized volatility")
            if meta.get("report") and os.path.exists(meta["report"]):
                with open(meta["report"], "r", encoding="utf-8") as f:
                    st.download_button("Download HTML report", data=f.read(), file_name="monte_carlo_report.html", use_container_width=True)
            json_path = os.path.join(OUTPUT_DIR, "summary.json")
            with open(json_path, "r") as f:
                st.download_button("Download summary.json", data=f.read(), file_name="monte_carlo_summary.json", use_container_width=True)

        tab_index = 4
        if result.get("optimization"):
            with tab_objs[tab_index]:
                opt = result["optimization"]
                st.subheader("Efficient Frontier")
                if px is not None:
                    df = pd.DataFrame(opt["frontier"])
                    fig = px.scatter(df, x="volatility", y="returns", color="sharpe", labels={"volatility":"Volatility", "returns":"Expected Return", "sharpe":"Sharpe"})
                    st.plotly_chart(fig, use_container_width=True)
                c1, c2 = st.columns(2)
                c1.write("**Max Sharpe Portfolio**")
                c1.json(opt["max_sharpe"])
                c2.write("**Minimum Volatility Portfolio**")
                c2.json(opt["min_volatility"])
            tab_index += 1
        if result.get("stress_tests"):
            with tab_objs[tab_index]:
                st.subheader("Stress Test Summary")
                rows = []
                for name, data in result["stress_tests"].items():
                    rows.append({
                        "Scenario": name,
                        "Expected Final": data["expected_final_price"],
                        "Probability of Loss": data["probability_of_loss"],
                        "VaR 95": data["var_historical"],
                        "CVaR 95": data["cvar_95"],
                        "Mean Max Drawdown": data["mean_max_drawdown"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            tab_index += 1
        if result.get("bootstrap_summary"):
            with tab_objs[tab_index]:
                st.subheader("GBM vs Historical Bootstrap")
                st.write("Bootstrap resamples actual historical returns instead of assuming normally distributed GBM shocks.")
                st.json(result["bootstrap_summary"])
    elif status != "error":
        st.error("Could not fetch enough historical data. Try different tickers or Manual mode.")
else:
    st.info("Choose settings in the sidebar, then run a simulation.")
    st.markdown(
        """
        **Suggested demo:** Portfolio mode with `AAPL, MSFT, SPY` and weights `0.40, 0.40, 0.20`.
        This shows the project is more than a single-stock simulator: it estimates covariance and simulates correlated portfolio paths.
        """
    )
