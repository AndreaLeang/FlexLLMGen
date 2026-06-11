#!/usr/bin/env python3
"""
Stacked bar plot of mha-gen + mlp latency breakdown per supergroup.

Reads the summary CSV produced by analyze_csv.py and generates a
stacked bar chart with the following segments (bottom to top):

  1. MHA CUDA          = mha-gen-compute-cuda-sum - mha-gen-recompute-cuda
  2. Recompute CUDA    = mha-gen-recompute-cuda
  3. Phase-1 segment:
       winner == load-hidden-compute-cudamemcpy -> "Recompute Load"
       winner == pin-memory-1                  -> "PinnedMemory CPU"
  4. Phase-2 segment:
       winner == A: "MLP CUDA" + "PinnedMemory CPU" (merged with step 3)
       winner == B: "KVCache Load"
       winner == C: "KVCache Store"
  5. Misc. CPU = (mha-gen-sum + mlp-sum) - (cuda-sum + phase-1 + phase-2)

Usage:
    python plot_latency.py <summary.csv> [--out output.png] [--dpi 150] [--show]
"""

import csv
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Colors and legend ordering
# ---------------------------------------------------------------------------

COLORS = {
    "MHA CUDA":         "#2196F3",   # blue
    "Recompute CUDA":   "#90CAF9",   # light blue
    "Recompute Load":   "#FF9800",   # orange
    "PinnedMemory CPU": "#FFC107",   # amber
    "MLP CUDA":         "#4CAF50",   # green
    "KVCache Load":     "#9C27B0",   # purple
    "KVCache Store":    "#E91E63",   # pink
    "Misc. CPU":        "#9E9E9E",   # grey
}

LEGEND_ORDER = [
    "MHA CUDA",
    "Recompute CUDA",
    "Recompute Load",
    "PinnedMemory CPU",
    "MLP CUDA",
    "KVCache Load",
    "KVCache Store",
    "Misc. CPU",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fv(row, key):
    """Return float value for a CSV cell, defaulting to 0.0."""
    v = row.get(key, "")
    try:
        return float(v) if v not in ("", None) else 0.0
    except (ValueError, TypeError):
        return 0.0


def build_segments(row):
    """
    Compute stack segments for one supergroup row.
    Returns a dict {legend_label: float_value_us}.
    PinnedMemory CPU accumulates contributions from phase-1 and/or phase-2.
    """
    cuda_sum  = fv(row, "mha-gen-compute-cuda-sum")
    recompute = fv(row, "mha-gen-recompute-cuda")
    mha_cuda  = cuda_sum - recompute

    mha_sum   = fv(row, "mha-gen-sum")
    mlp_sum   = fv(row, "mlp-sum")
    phase1    = fv(row, "mlp-phase-1")
    phase2    = fv(row, "mlp-phase-2")
    mlp_cuda  = fv(row, "mlp-compute-cuda-sum")
    pin2      = fv(row, "pin-memory-2")
    lc1       = fv(row, "load-cache-cudamemcpy-1")
    lc2       = fv(row, "load-cache-cudamemcpy-2")
    sc1       = fv(row, "store-cache-cudamemcpy-1")
    sc2       = fv(row, "store-cache-cudamemcpy-2")

    phase1_winner = row.get("mlp-phase-1-winner", "")
    phase2_winner = row.get("mlp-phase-2-winner", "")

    segs = {}

    # 1. MHA CUDA
    segs["MHA CUDA"] = mha_cuda

    # 2. Recompute CUDA
    segs["Recompute CUDA"] = recompute

    # 3. Phase-1
    if "load-hidden-compute-cudamemcpy" in phase1_winner:
        segs["Recompute Load"] = phase1
    else:
        segs["PinnedMemory CPU"] = segs.get("PinnedMemory CPU", 0.0) + phase1

    # 4. Phase-2
    if phase2_winner.startswith("mlp-compute-cuda"):      # winner A
        segs["MLP CUDA"] = mlp_cuda
        segs["PinnedMemory CPU"] = segs.get("PinnedMemory CPU", 0.0) + pin2
    elif phase2_winner.startswith("load-cache"):           # winner B
        segs["KVCache Load"] = lc1 + lc2
    else:                                                  # winner C
        segs["KVCache Store"] = sc1 + sc2

    # 5. Misc. CPU
    segs["Misc. CPU"] = (mha_sum + mlp_sum) - (cuda_sum + phase1 + phase2)

    return segs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(csv_path, out_path=None, dpi=150, show=False):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    all_segs = [build_segments(r) for r in rows]

    # x-axis labels
    labels = [f"SG{r['supergroup']}\n(tok {r['token']})" for r in rows]
    n = len(rows)

    # Only include legend entries that actually appear
    used_legends = [l for l in LEGEND_ORDER
                    if any(s.get(l, 0.0) > 0 for s in all_segs)]

    x = np.arange(n)
    bar_width = 0.5

    fig, ax = plt.subplots(figsize=(max(6, n * 2.4), 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bottoms = np.zeros(n)

    for legend in used_legends:
        vals = np.array([s.get(legend, 0.0) for s in all_segs])
        ax.bar(
            x, vals, bar_width,
            bottom=bottoms,
            color=COLORS[legend],
            label=legend,
            edgecolor="white",
            linewidth=0.5,
        )
        # Segment value label if tall enough
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 80:
                ax.text(
                    xi, b + v / 2,
                    f"{v:.0f}",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                )
        bottoms += vals

    # Total annotation above each bar
    max_total = float(bottoms.max())
    for xi, total in enumerate(bottoms):
        ax.text(
            xi, total + max_total * 0.01,
            f"{total:.0f} µs",
            ha="center", va="bottom",
            fontsize=8.5, color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("Latency (µs)", fontsize=11)
    ax.set_title(
        "mha-gen + mlp Latency Breakdown per Supergroup",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_ylim(0, max_total * 1.13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=COLORS[l], edgecolor="#cccccc", label=l)
        for l in used_legends
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right", fontsize=9,
        framealpha=0.9, edgecolor="#cccccc",
    )

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved to {out_path}", file=sys.stderr)

    if show:
        plt.show()

    # Sanity check
    for row, segs in zip(rows, all_segs):
        seg_total = sum(segs.values())
        ref_total = fv(row, "mha-gen-sum") + fv(row, "mlp-sum")
        diff = abs(seg_total - ref_total)
        status = "OK" if diff < 0.1 else f"MISMATCH (diff={diff:.2f})"
        print(
            f"  SG{row['supergroup']}: segments={seg_total:.1f} µs  "
            f"mha+mlp={ref_total:.1f} µs  [{status}]",
            file=sys.stderr,
        )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stacked bar plot of mha-gen/mlp latency breakdown."
    )
    parser.add_argument("csv", help="Summary CSV from analyze_csv.py.")
    parser.add_argument(
        "--out", default=None, metavar="OUTPUT.png",
        help="Output image path (default: <csv_stem>_latency.png).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output image DPI (default: 150).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Call plt.show() after saving (for interactive use).",
    )
    args = parser.parse_args()

    out = args.out or (Path(args.csv).stem + "_latency.png")
    plot(args.csv, out_path=out, dpi=args.dpi, show=args.show)


if __name__ == "__main__":
    main()