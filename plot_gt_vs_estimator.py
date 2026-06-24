#!/usr/bin/env python3
"""
plot_gt_vs_estimator.py
=======================
Grouped bar plot comparing ground-truth latency breakdown vs. estimator predictions.

Reads the comparison CSV produced by gt_vs_estimator.py and creates:
  • One x-tick per experiment (e.g., different batch sizes or recompute lengths).
  • Within each x-tick: one stacked bar for ground truth, one per estimator mode.
  • Bars are stacked by latency segment (PinnedMemory, KV Load, MHA CUDA, etc.).

Usage
-----
  python plot_gt_vs_estimator.py comparison_results.csv \\
      [--x-axis batch_size|recompute_len|experiment_id] \\
      [--out figure.png] [--dpi 150] [--show]

  The --x-axis flag controls what is displayed on the x-axis.  The default is
  "batch_size".  Use "experiment_id" for the full label when sweeping multiple
  dimensions simultaneously.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")        # headless default; --show will call plt.show()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Color palette: one colour per canonical segment
# ---------------------------------------------------------------------------

# Ground-truth segments
GT_COLORS = {
    "PinnedMemory CPU":   "#FFC107",   # amber
    "Recompute Load":     "#FF9800",   # orange
    "Recompute CUDA":     "#90CAF9",   # light blue
    "MHA CUDA":           "#2196F3",   # blue
    "KVCache Load":       "#9C27B0",   # purple
    "KVCache Store":      "#E91E63",   # pink
    "Misc. CPU":          "#9E9E9E",   # grey
}

# Estimator segments (slightly different shades to distinguish from GT)
EST_COLORS = {
    "PinnedMemory CPU (phase1)": "#FFE082",   # light amber
    "PinnedMemory CPU (phase2)": "#FFB300",   # dark amber
    "Recompute Load":            "#FB8C00",   # dark orange
    "Recompute CUDA":            "#64B5F6",   # medium blue
    "MHA CUDA":                  "#1565C0",   # dark blue
    "KVCache Load K":            "#7B1FA2",   # dark purple
    "KVCache Load V":            "#CE93D8",   # light purple
}

# Segment ordering for stacking (bottom → top)
GT_STACK_ORDER = [
    "Recompute Load",
    "Recompute CUDA",
    "MHA CUDA",
    "PinnedMemory CPU",
    "KVCache Load",
    "KVCache Store",
    "Misc. CPU",
]

EST_STACK_ORDER = [
    "Recompute Load",
    "Recompute CUDA",
    "MHA CUDA",
    "PinnedMemory CPU (phase1)",
    "PinnedMemory CPU (phase2)",
    "KVCache Load K",
    "KVCache Load V",
]

# Bar appearance
BAR_ALPHA = 0.92
EDGE_COLOR = "white"
EDGE_WIDTH = 0.5


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _fv(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (ValueError, TypeError):
        return default


def _detect_estimator_modes(fieldnames: List[str]) -> List[str]:
    """Infer estimator mode names from CSV columns named est_{mode}_total_us."""
    modes = []
    seen = set()
    for col in fieldnames:
        if col.startswith("est_") and col.endswith("_total_us"):
            mode = col[len("est_"):-len("_total_us")]
            if mode not in seen:
                modes.append(mode)
                seen.add(mode)
    return modes


def load_comparison_csv(csv_path: str) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Returns (rows, fieldnames, estimator_mode_names).
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    modes = _detect_estimator_modes(fieldnames)
    return rows, fieldnames, modes


def get_gt_segments(row: Dict) -> Dict[str, float]:
    mapping = {
        "PinnedMemory CPU":  "gt_PinnedMemory_CPU_us",
        "Recompute Load":    "gt_Recompute_Load_us",
        "Recompute CUDA":    "gt_Recompute_CUDA_us",
        "MHA CUDA":          "gt_MHA_CUDA_us",
        "KVCache Load":      "gt_KVCache_Load_us",
        "KVCache Store":     "gt_KVCache_Store_us",
        "Misc. CPU":         "gt_Misc_CPU_us",
    }
    return {seg: _fv(row, col) for seg, col in mapping.items()}


def get_est_segments(row: Dict, mode: str) -> Dict[str, float]:
    pfx = f"est_{mode}"
    mapping = {
        "PinnedMemory CPU (phase1)": f"{pfx}_PinnedMemory_CPU_phase1_us",
        "PinnedMemory CPU (phase2)": f"{pfx}_PinnedMemory_CPU_phase2_us",
        "Recompute Load":            f"{pfx}_Recompute_Load_us",
        "Recompute CUDA":            f"{pfx}_Recompute_CUDA_us",
        "MHA CUDA":                  f"{pfx}_MHA_CUDA_us",
        "KVCache Load K":            f"{pfx}_KVCache_Load_K_us",
        "KVCache Load V":            f"{pfx}_KVCache_Load_V_us",
    }
    return {seg: _fv(row, col) for seg, col in mapping.items()}


def make_x_label(row: Dict, x_axis: str) -> str:
    if x_axis == "batch_size":
        bs = row.get("batch_size", "?")
        nb = row.get("num_batches", "?")
        off = row.get("offload_percent", "?")
        return f"bs={bs}\nnb={nb}\noff={float(off):.0f}%"
    elif x_axis == "recompute_len":
        rc = row.get("recompute_len", "?")
        return f"rc={rc}"
    elif x_axis == "offload_percent":
        return f"off={float(row.get('offload_percent', 0)):.0f}%"
    else:
        return row.get("experiment_id", "?")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(
    csv_path: str,
    x_axis: str = "batch_size",
    out_path: Optional[str] = None,
    dpi: int = 150,
    show: bool = False,
    modes_to_plot: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create the grouped-bar comparison plot.

    Parameters
    ----------
    csv_path       Path to comparison_results.csv.
    x_axis         Column used for x-tick labels: batch_size | recompute_len |
                   offload_percent | experiment_id.
    out_path       Save figure here (None = don't save).
    dpi            DPI for saved figure.
    show           Call plt.show().
    modes_to_plot  Subset of estimator mode names to include (None = all).
    title          Figure title override.
    """
    rows, fieldnames, all_modes = load_comparison_csv(csv_path)
    if not rows:
        print("No rows in CSV.", file=sys.stderr)
        sys.exit(1)

    modes = modes_to_plot if modes_to_plot is not None else all_modes
    if not modes:
        print("No estimator modes found in CSV.", file=sys.stderr)
        sys.exit(1)

    # Number of bars per x-tick: 1 (GT) + len(modes)
    n_ticks = len(rows)
    n_bars = 1 + len(modes)
    group_width = 0.8
    bar_w = group_width / n_bars
    offsets = np.linspace(-(group_width / 2) + bar_w / 2,
                          (group_width / 2) - bar_w / 2,
                          n_bars)

    x = np.arange(n_ticks)
    fig_w = max(8, n_ticks * n_bars * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Collect all drawn patches for the legend (deduplicated by label)
    legend_patches: Dict[str, mpatches.Patch] = {}

    def add_patch(label: str, color: str):
        if label not in legend_patches:
            legend_patches[label] = mpatches.Patch(
                facecolor=color, edgecolor="#cccccc", alpha=BAR_ALPHA, label=label
            )

    # Track per-bar max for annotations
    all_totals = []

    # ------------------------------------------------------------------ GT --
    gt_bottoms = np.zeros(n_ticks)
    for seg in GT_STACK_ORDER:
        vals = np.array([get_gt_segments(r).get(seg, 0.0) for r in rows])
        color = GT_COLORS[seg]
        ax.bar(
            x + offsets[0], vals, bar_w,
            bottom=gt_bottoms,
            color=color, alpha=BAR_ALPHA,
            edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH,
        )
        for xi, (v, b) in enumerate(zip(vals, gt_bottoms)):
            if v > 60:
                ax.text(
                    xi + offsets[0], b + v / 2, f"{v:.0f}",
                    ha="center", va="center", fontsize=6.5,
                    color="white", fontweight="bold",
                )
        gt_bottoms += vals
        add_patch(f"GT: {seg}", color)

    all_totals.extend(gt_bottoms.tolist())

    # Annotate GT bar total
    for xi, total in enumerate(gt_bottoms):
        ax.text(
            xi + offsets[0], total + 10, f"{total:.0f}",
            ha="center", va="bottom", fontsize=7.5, color="#333333",
        )

    # --------------------------------------------------------- Estimators --
    mode_bar_styles = [
        # hatch, edge_color pairs to visually distinguish bars
        ("//",   "#1A237E"),
        ("\\\\", "#4A148C"),
        ("xx",   "#880E4F"),
        ("oo",   "#1B5E20"),
        ("++",   "#E65100"),
    ]
    for mi, mode in enumerate(modes):
        bar_idx = mi + 1
        hatch, edge_c = mode_bar_styles[mi % len(mode_bar_styles)]
        est_bottoms = np.zeros(n_ticks)
        for seg in EST_STACK_ORDER:
            vals = np.array([get_est_segments(r, mode).get(seg, 0.0) for r in rows])
            color = EST_COLORS[seg]
            ax.bar(
                x + offsets[bar_idx], vals, bar_w,
                bottom=est_bottoms,
                color=color, alpha=BAR_ALPHA,
                edgecolor=edge_c, linewidth=EDGE_WIDTH,
                hatch=hatch,
            )
            for xi, (v, b) in enumerate(zip(vals, est_bottoms)):
                if v > 60:
                    ax.text(
                        xi + offsets[bar_idx], b + v / 2, f"{v:.0f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold",
                    )
            est_bottoms += vals
            add_patch(f"Est [{mode}]: {seg}", color)

        all_totals.extend(est_bottoms.tolist())

        # Annotate est bar total
        for xi, total in enumerate(est_bottoms):
            ax.text(
                xi + offsets[bar_idx], total + 10, f"{total:.0f}",
                ha="center", va="bottom", fontsize=7.5, color="#555555",
                style="italic",
            )

    # ---------------------------------------------------------------- Axes --
    max_total = max(all_totals) if all_totals else 1.0
    ax.set_ylim(0, max_total * 1.20)
    ax.set_xlim(-0.65, n_ticks - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([make_x_label(r, x_axis) for r in rows], fontsize=9)
    ax.set_ylabel("Latency (µs)", fontsize=11)
    ax.set_xlabel(x_axis.replace("_", " ").title(), fontsize=10)

    default_title = "Ground Truth vs. Estimator: MHA Latency Breakdown"
    ax.set_title(title or default_title, fontsize=13, fontweight="bold", pad=14)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: two groups (GT / Estimator) sorted by segment order
    gt_handles = [p for lbl, p in legend_patches.items() if lbl.startswith("GT:")]
    est_handles = [p for lbl, p in legend_patches.items() if not lbl.startswith("GT:")]

    # Add a blank separator patch between GT and Est groups
    blank = mpatches.Patch(visible=False, label="")

    legend = ax.legend(
        handles=gt_handles + [blank] + est_handles,
        loc="upper right",
        fontsize=7.5,
        framealpha=0.92,
        edgecolor="#cccccc",
        ncol=max(1, (len(gt_handles) + len(est_handles)) // 7 + 1),
    )

    # Add "GT" / "Estimator" text annotations on the bars at x=0 for clarity
    if n_ticks > 0:
        ax.annotate(
            "GT",
            xy=(offsets[0], 0), xytext=(offsets[0], -max_total * 0.06),
            ha="center", va="top", fontsize=7, color="#333333", fontweight="bold",
        )
        for mi, mode in enumerate(modes):
            ax.annotate(
                mode,
                xy=(offsets[mi + 1], 0),
                xytext=(offsets[mi + 1], -max_total * 0.06),
                ha="center", va="top", fontsize=7, color="#555555",
            )

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to: {out_path}")

    if show:
        plt.show()

    # Sanity print
    print("\nSanity check — total latency (µs):")
    print(f"  {'Experiment':<35} {'GT':>10}", end="")
    for m in modes:
        print(f"  {'Est['+m+']':>15}", end="")
    print()
    for row, gt_tot, *est_tots in zip(
        rows,
        gt_bottoms,
        *[[get_est_segments(r, m) for r in rows] for m in modes],
    ):
        exp_id = row.get("experiment_id", "?")[:35]
        print(f"  {exp_id:<35} {gt_tot:>10.1f}", end="")
        for m, segs in zip(modes, est_tots):
            tot = sum(segs.values()) if isinstance(segs, dict) else segs
            print(f"  {tot:>15.1f}", end="")
        print()

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot GT vs estimator latency breakdown from comparison CSV."
    )
    parser.add_argument("csv", help="Path to comparison_results.csv")
    parser.add_argument(
        "--x-axis",
        choices=["batch_size", "recompute_len", "offload_percent", "experiment_id"],
        default="batch_size",
        help="Which column to use for x-tick labels.",
    )
    parser.add_argument("--out", default=None, metavar="OUTPUT.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--modes", nargs="+", default=None, metavar="MODE",
        help="Subset of estimator mode names to include (default: all).",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    out = args.out or (Path(args.csv).stem + "_comparison.png")
    plot_comparison(
        csv_path=args.csv,
        x_axis=args.x_axis,
        out_path=out,
        dpi=args.dpi,
        show=args.show,
        modes_to_plot=args.modes,
        title=args.title,
    )


if __name__ == "__main__":
    main()