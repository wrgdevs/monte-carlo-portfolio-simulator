"""Simple example that demonstrates CLI functions programmatically."""
from mc_portfolio.cli import run_from_ticker


def demo():
    # Example: try AAPL. If it fails, the function returns (None, "fetch_failed").
    result, status = run_from_ticker("AAPL", sims=3000, years=1.0)
    if status == "ok":
        print("Summary:")
        print(result["summary"])  # metadata + summary saved in results/summary.json
    else:
        print("Fetch failed for ticker AAPL. Try manual mode or another ticker.")


if __name__ == "__main__":
    demo()