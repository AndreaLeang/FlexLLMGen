#!/usr/bin/env python3
"""
Stacked bar plot of latency breakdown per group/supergroup.

Reads the summary CSV produced by analyze_csv.py (sep or nosep, auto-detected).

SEP mode (original behavior from plot_latency_single.py, unchanged):
  1. MHA CUDA          = mha-gen-compute-cuda-sum - mha-gen-recompute-cuda
  2. Recompute CUDA    = mha-gen-recompute-cuda
  3. Phase-1:
       winner == load-hidden-compute-cudamemcpy -> "Recompute Load"
       winner == pin-memory-1                   -> "PinnedMemory CPU"
  4. Phase-2:
       winner A ("mlp-compute-cuda-sum+pin-memory-2") -> "MLP CUDA" + "PinnedMemory CPU"
       winner B -> "KVCache Load"
       winner C -> "KVCache Store"
  5. Misc. CPU = (mha-gen-sum + mlp-sum) - (cuda-sum + phase-1 + phase-2)

NOSEP mode (--nosep flag or auto-detected):
  1. Phase-1:
       winner == load-hidden-compute-cudamemcpy -> "Recompute Load"
       winner == pin-memory-1                   -> "PinnedMemory CPU"
  2. Phase-2:
       winner A ("pin-memory-2+compute-cuda-sum") -> "Compute CUDA" + "PinnedMemory CPU"
       winner B -> "KVCache Load"
       winner C -> "KVCache Store"
  3. Misc. CPU = sum-all - phase-1 - phase-2

Usage:
    python plot_latency.py <summary.csv> [--out output.png] [--dpi 150] [--show] [--nosep]
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
    "MHA CUDA":         "#2196F3",   # blue       (sep only)
    "Recompute CUDA":   "#90CAF9",   # light blue (sep only)
    "Compute CUDA":     "#2196F3",   # blue       (nosep winner-A only)
    "Recompute Load":   "#FF9800",   # orange
    "PinnedMemory CPU": "#FFC107",   # amber
    "MLP CUDA":         "#4CAF50",   # green      (sep winner-A only)
    "KVCache Load":     "#9C27B0",   # purple
    "KVCache Store":    "#E91E63",   # pink
    "Misc. CPU":        "#9E9E9E",   # grey
}

SEP_LEGEND_ORDER = [
    "MHA CUDA",
    "Recompute CUDA",
    "Recompute Load",
    "PinnedMemory CPU",
    "MLP CUDA",
    "KVCache Load",
    "KVCache Store",
    "Misc. CPU",
]

NOSEP_LEGEND_ORDER = [
    "Compute CUDA",
    "Recompute Load",
    "PinnedMemory CPU",
    "KVCache Load",
    "KVCache Store",
    "Misc. CPU",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fv(row, key):
    v = row.get(key, "")
    try:
        return float(v) if v not in ("", None) else 0.0
    except (ValueError, TypeError):
        return 0.0


def detect_nosep(all_cols):
    return "load_weight" in all_cols and "mha-gen_load_weight" not in all_cols


# ---------------------------------------------------------------------------
# SEP segment builder (original logic from plot_latency_single.py)
# ---------------------------------------------------------------------------

def build_segments_sep(row):
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

    segs["MHA CUDA"] = mha_cuda
    segs["Recompute CUDA"] = recompute

    if "load-hidden-compute-cudamemcpy" in phase1_winner:
        segs["Recompute Load"] = phase1
    else:
        segs["PinnedMemory CPU"] = segs.get("PinnedMemory CPU", 0.0) + phase1

    if phase2_winner.startswith("mlp-compute-cuda"):      # winner A
        segs["MLP CUDA"] = mlp_cuda
        segs["PinnedMemory CPU"] = segs.get("PinnedMemory CPU", 0.0) + pin2
    elif phase2_winner.startswith("load-cache"):           # winner B
        segs["KVCache Load"] = lc1 + lc2
    else:                                                  # winner C
        segs["KVCache Store"] = sc1 + sc2

    segs["Misc. CPU"] = (mha_sum + mlp_sum) - (cuda_sum + phase1 + phase2)

    return segs


# ---------------------------------------------------------------------------
# NOSEP segment builder
# ---------------------------------------------------------------------------

def build_segments_nosep(row):
    compute_cuda_sum = fv(row, "compute-cuda-sum")
    pin2  = fv(row, "pin-memory-2")
    lc1   = fv(row, "load-cache-cudamemcpy-1")
    lc2   = fv(row, "load-cache-cudamemcpy-2")
    sc1   = fv(row, "store-cache-cudamemcpy-1")
    sc2   = fv(row, "store-cache-cudamemcpy-2")
    phase1        = fv(row, "phase-1")
    phase1_winner = row.get("phase-1-winner", "")
    phase2_winner = row.get("phase-2-winner", "")

    segs = {}

    if "load-hidden-compute-cudamemcpy" in phase1_winner:
        segs["Recompute Load"] = phase1
    else:
        segs["PinnedMemory CPU"] = segs.get("PinnedMemory CPU", 0.0) + phase1

    if phase2_winner.startswith("pin-memory-2"):           # winner A
        segs["Compute CUDA"] = compute_cuda_sum
        segs["PinnedMemory CPU"] = segs.get("PinnedMemory CPU", 0.0) + pin2
    elif phase2_winner.startswith("load-cache"):            # winner B
        segs["KVCache Load"] = lc1 + lc2
    else:                                                   # winner C
        segs["KVCache Store"] = sc1 + sc2

    segs["Misc. CPU"] = fv(row, "misc-cpu")
    return segs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(csv_path, out_path=None, dpi=150, show=False, nosep=None):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        all_cols = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("No rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    if nosep is None:
        nosep = detect_nosep(all_cols)
    mode = "nosep" if nosep else "sep"
    print(f"Mode: {mode}", file=sys.stderr)

    if nosep:
        id_col = "group"
        all_segs = [build_segments_nosep(r) for r in rows]
        legend_order = NOSEP_LEGEND_ORDER
        labels = [f"G{r[id_col]}\n(tok {r['token']})" for r in rows]
    else:
        id_col = "supergroup"
        all_segs = [build_segments_sep(r) for r in rows]
        legend_order = SEP_LEGEND_ORDER
        labels = [f"SG{r[id_col]}\n(tok {r['token']})" for r in rows]

    used_legends = [l for l in legend_order
                    if any(s.get(l, 0.0) > 0 for s in all_segs)]

    n = len(rows)
    x = np.arange(n)
    bar_width = 0.5

    fig, ax = plt.subplots(figsize=(max(6, n * 2.4), 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bottoms = np.zeros(n)
    for legend in used_legends:
        vals = np.array([s.get(legend, 0.0) for s in all_segs])
        ax.bar(x, vals, bar_width, bottom=bottoms,
               color=COLORS[legend], label=legend,
               edgecolor="white", linewidth=0.5)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 80:
                ax.text(xi, b + v / 2, f"{v:.0f}",
                        ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottoms += vals

    max_total = float(bottoms.max())
    for xi, total in enumerate(bottoms):
        ax.text(xi, total + max_total * 0.01, f"{total:.0f} µs",
                ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("Latency (µs)", fontsize=11)
    ax.set_title(
        ("Nosep" if nosep else "mha-gen + mlp") + " Latency Breakdown per Group",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_ylim(0, max_total * 1.13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=[mpatches.Patch(facecolor=COLORS[l], edgecolor="#cccccc", label=l)
                 for l in used_legends],
        loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc",
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
        if nosep:
            ref_total = fv(row, "sum-all")
        else:
            ref_total = fv(row, "mha-gen-sum") + fv(row, "mlp-sum")
        diff = abs(seg_total - ref_total)
        status = "OK" if diff < 0.1 else f"MISMATCH (diff={diff:.2f})"
        print(f"  {id_col}={row[id_col]}: segments={seg_total:.1f} µs  "
              f"ref={ref_total:.1f} µs  [{status}]", file=sys.stderr)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stacked bar plot of latency breakdown."
    )
    parser.add_argument("csv", help="Summary CSV from analyze_csv.py.")
    parser.add_argument("--out", default=None, metavar="OUTPUT.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true",
                        help="Call plt.show() for interactive display.")
    parser.add_argument("--nosep", action="store_true",
                        help="Force nosep mode (auto-detected if omitted).")
    args = parser.parse_args()

    out = args.out or (Path(args.csv).stem + "_latency.png")
    plot(args.csv, out_path=out, dpi=args.dpi, show=args.show,
         nosep=args.nosep if args.nosep else None)


if __name__ == "__main__":
    main()