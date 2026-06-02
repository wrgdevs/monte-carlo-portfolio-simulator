"""Matplotlib visualizations for saved CLI outputs."""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR
from .metrics import max_drawdown_per_sim


def _save(outpath: Optional[str], default_name: str) -> str:
    if outpath is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return os.path.join(OUTPUT_DIR, default_name)
    return outpath


def plot_sample_paths(paths: np.ndarray, outpath: Optional[str] = None, n_plot: int = 75, title: str = "Simulated Price Paths") -> str:
    n_steps, n_sims = paths.shape
    n_plot = min(n_plot, n_sims)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n_sims, size=n_plot, replace=False)

    plt.figure(figsize=(11, 6))
    plt.plot(paths[:, sample_idx], alpha=0.22, linewidth=0.8)
    median = np.median(paths, axis=1)
    p05 = np.percentile(paths, 5, axis=1)
    p25 = np.percentile(paths, 25, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    x = np.arange(n_steps)
    plt.fill_between(x, p05, p95, alpha=0.14, label="5-95% band")
    plt.fill_between(x, p25, p75, alpha=0.22, label="25-75% band")
    plt.plot(median, linewidth=2.4, label="median")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value / Price")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    outpath = _save(outpath, "simulated_paths.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


def plot_final_histogram(paths: np.ndarray, outpath: Optional[str] = None, title: str = "Final Distribution") -> str:
    finals = paths[-1, :]
    plt.figure(figsize=(9, 5.5))
    plt.hist(finals, bins=60, alpha=0.85)
    for value, label, style in [
        (np.percentile(finals, 5), "5th pct / VaR", ":"),
        (np.mean(finals), "mean", "--"),
        (np.median(finals), "median", "-"),
        (np.percentile(finals, 95), "95th pct", ":"),
    ]:
        plt.axvline(value, linestyle=style, linewidth=1.6, label=label)
    plt.title(title)
    plt.xlabel("Final Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    outpath = _save(outpath, "final_histogram.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


def plot_max_drawdown_histogram(paths: np.ndarray, outpath: Optional[str] = None, title: str = "Max Drawdown Distribution") -> str:
    drawdowns = -max_drawdown_per_sim(paths) * 100
    plt.figure(figsize=(9, 5.5))
    plt.hist(drawdowns, bins=50, alpha=0.85)
    plt.axvline(np.mean(drawdowns), linestyle="--", label="mean")
    plt.axvline(np.percentile(drawdowns, 95), linestyle=":", label="95th pct")
    plt.title(title)
    plt.xlabel("Maximum Drawdown (%)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    outpath = _save(outpath, "drawdown_histogram.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath
