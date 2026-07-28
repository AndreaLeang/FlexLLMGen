#!/usr/bin/env python3
"""
plot_pin_memory_results.py

Plots latency-vs-size and throughput-vs-size from the raw CSV produced by
pin_memory_finegrained_sweep.py (or any CSV with `size_bytes` and `wall_ms`
columns, one row per call).

Usage:
    python plot_pin_memory_results.py [path/to/raw.csv]
Defaults to pin_memory_finegrained_raw.csv in the current directory.

Requires: pandas, numpy, matplotlib.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GB = 1024 ** 3  # binary GB (GiB) throughout, consistent with the sweep script


def load_and_aggregate(csv_path):
    df = pd.read_csv(csv_path)
    df["size_gb"] = df["size_bytes"] / GB
    df["throughput_gbps"] = df["size_gb"] / (df["wall_ms"] / 1000)  # GiB/s

    agg = df.groupby("size_bytes").agg(
        size_gb=("size_gb", "first"),
        latency_mean=("wall_ms", "mean"),
        latency_std=("wall_ms", "std"),
        throughput_mean=("throughput_gbps", "mean"),
        throughput_std=("throughput_gbps", "std"),
    ).sort_values("size_gb").reset_index(drop=True)
    return df, agg


def plot_latency(df, agg, out_path="latency_vs_size.png"):
    fig, ax = plt.subplots(figsize=(9, 6))

    # faint raw scatter for texture/variance context behind the mean line
    ax.scatter(df["size_gb"], df["wall_ms"], s=6, alpha=0.15, color="steelblue", label="individual calls")

    lower = np.clip(agg["latency_mean"] - agg["latency_std"], a_min=1e-3, a_max=None)
    upper = agg["latency_mean"] + agg["latency_std"]
    ax.fill_between(agg["size_gb"], lower, upper, color="steelblue", alpha=0.25, label="±1 std")
    ax.plot(agg["size_gb"], agg["latency_mean"], color="steelblue", linewidth=2,
             marker="o", markersize=3, label="mean")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tensor size (GiB)")
    ax.set_ylabel("pin_memory() latency (ms)")
    ax.set_title("pin_memory() latency vs. tensor size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_throughput(df, agg, out_path="throughput_vs_size.png"):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(df["size_gb"], df["throughput_gbps"], s=6, alpha=0.15, color="darkorange", label="individual calls")

    lower = np.clip(agg["throughput_mean"] - agg["throughput_std"], a_min=0, a_max=None)
    upper = agg["throughput_mean"] + agg["throughput_std"]
    ax.fill_between(agg["size_gb"], lower, upper, color="darkorange", alpha=0.25, label="±1 std")
    ax.plot(agg["size_gb"], agg["throughput_mean"], color="darkorange", linewidth=2,
             marker="o", markersize=3, label="mean")

    ax.set_xscale("log")
    ax.set_xlabel("Tensor size (GiB)")
    ax.set_ylabel("Effective throughput (GiB/s)")
    ax.set_title("pin_memory() throughput vs. tensor size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "pin_memory_finegrained_raw.csv"
    df, agg = load_and_aggregate(csv_path)
    plot_latency(df, agg)
    plot_throughput(df, agg)


if __name__ == "__main__":
    main()