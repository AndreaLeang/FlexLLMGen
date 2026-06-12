#!/usr/bin/env python3
"""
Multi-workload stacked bar plot.

Scans a folder for subdirectories named:
    prompt_{prompt_length}
    prompt_{prompt_length}_bs{batchsize}   (batchsize=1 if omitted)

In each subdirectory, finds the first file matching *_summary.csv,
reads only the first group/supergroup row, auto-detects sep vs nosep,
builds latency segments, and plots all workloads side-by-side.

X-ticks are labeled by subfolder name (prompt length and batch size).

Segments — sep mode (bottom to top):
  MHA CUDA, Recompute CUDA,
  Phase-1: Recompute Load or PinnedMemory CPU
  Phase-2: MLP CUDA + PinnedMemory CPU (A), KVCache Load (B), KVCache Store (C)
  Misc. CPU

Segments — nosep mode (bottom to top):
  Phase-1: Recompute Load or PinnedMemory CPU
  Phase-2: Compute CUDA + PinnedMemory CPU (A), KVCache Load (B), KVCache Store (C)
  Misc. CPU

Usage:
    python plot_multi_workload.py <folder> [--out output.png] [--dpi 150] [--show]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Colors and legend ordering (shared with plot_latency.py)
# ---------------------------------------------------------------------------

COLORS = {
    "MHA CUDA":         "#2196F3",
    "Recompute CUDA":   "#90CAF9",
    "Compute CUDA":     "#2196F3",
    "Recompute Load":   "#FF9800",
    "PinnedMemory CPU": "#FFC107",
    "MLP CUDA":         "#4CAF50",
    "KVCache Load":     "#9C27B0",
    "KVCache Store":    "#E91E63",
    "Misc. CPU":        "#9E9E9E",
}

SEP_LEGEND_ORDER = [
    "MHA CUDA", "Recompute CUDA",
    "Recompute Load", "PinnedMemory CPU", "MLP CUDA",
    "KVCache Load", "KVCache Store", "Misc. CPU",
]

NOSEP_LEGEND_ORDER = [
    "Compute CUDA",
    "Recompute Load", "PinnedMemory CPU",
    "KVCache Load", "KVCache Store", "Misc. CPU",
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
# Segment builders (identical logic to plot_latency.py)
# ---------------------------------------------------------------------------

def build_segments_sep(row):
    cuda_sum  = fv(row, "mha-gen-compute-cuda-sum")
    recompute = fv(row, "mha-gen-recompute-cuda")
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
    segs["MHA CUDA"] = cuda_sum - recompute
    segs["Recompute CUDA"] = recompute

    if "load-hidden-compute-cudamemcpy" in phase1_winner:
        segs["Recompute Load"] = phase1
    else:
        segs["PinnedMemory CPU"] = segs.get("PinnedMemory CPU", 0.0) + phase1

    if phase2_winner.startswith("mlp-compute-cuda"):
        segs["MLP CUDA"] = mlp_cuda
        segs["PinnedMemory CPU"] = segs.get("PinnedMemory CPU", 0.0) + pin2
    elif phase2_winner.startswith("load-cache"):
        segs["KVCache Load"] = lc1 + lc2
    else:
        segs["KVCache Store"] = sc1 + sc2

    segs["Misc. CPU"] = (mha_sum + mlp_sum) - (cuda_sum + phase1 + phase2)
    return segs


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

    if phase2_winner.startswith("pin-memory-2"):
        segs["Compute CUDA"] = compute_cuda_sum
        segs["PinnedMemory CPU"] = segs.get("PinnedMemory CPU", 0.0) + pin2
    elif phase2_winner.startswith("load-cache"):
        segs["KVCache Load"] = lc1 + lc2
    else:
        segs["KVCache Store"] = sc1 + sc2

    segs["Misc. CPU"] = fv(row, "misc-cpu")
    return segs


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------

# Base pattern: prompt_{N} followed by any combination of _bs{B} and _recompute_{R}
# in any order (e.g. prompt_2048_bs4_recompute_512 or prompt_2048_recompute_512_bs4)
SUBDIR_PATTERN  = re.compile(r"^prompt_(\d+)((?:_(?:bs\d+|recompute_\d+))*)$", re.IGNORECASE)
BS_PATTERN      = re.compile(r"_bs(\d+)", re.IGNORECASE)
RECOMPUTE_PATTERN = re.compile(r"_recompute_(\d+)", re.IGNORECASE)


def parse_subdir(name):
    """
    Parse a subdir name such as:
        prompt_2048
        prompt_2048_bs4
        prompt_2048_recompute_512
        prompt_2048_bs4_recompute_512
        prompt_2048_recompute_512_bs4
    Returns (prompt_length, batch_size, recompute_length) or None if no match.
    recompute_length is None when not present; batch_size defaults to 1.
    """
    m = SUBDIR_PATTERN.match(name)
    if not m:
        return None
    prompt_len = int(m.group(1))
    rest = m.group(2)
    bs_m = BS_PATTERN.search(rest)
    rc_m = RECOMPUTE_PATTERN.search(rest)
    batch_size = int(bs_m.group(1)) if bs_m else 1
    recompute  = int(rc_m.group(1)) if rc_m else None
    return prompt_len, batch_size, recompute


def find_summary_csv(subdir: Path):
    """Find the first *_summary.csv in a subdirectory."""
    candidates = sorted(subdir.glob("*_summary.csv"))
    return candidates[0] if candidates else None


def load_first_row(csv_path: Path):
    """Load only the first data row from a summary CSV."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            return row, cols
    return None, []


def make_label(prompt_len, batch_size, recompute):
    """
    Build a multi-line x-tick label showing all non-default dimensions.
      prompt length always shown as p{N}
      batch size shown as bs{B} only if > 1
      recompute shown as rc{R} only if present
    """
    parts = [f"p{prompt_len}"]
    if batch_size != 1:
        parts.append(f"bs{batch_size}")
    if recompute is not None:
        parts.append(f"rc{recompute}")
    return "\n".join(parts)


def scan_folder(folder: Path):
    """
    Scan folder for prompt_* subdirs, return list of:
      (label, prompt_len, batch_size, recompute, row, cols, nosep)
    sorted by (prompt_len, batch_size, recompute).
    """
    entries = []
    for subdir in sorted(folder.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = parse_subdir(subdir.name)
        if parsed is None:
            print(f"  Skipping '{subdir.name}' (no match)", file=sys.stderr)
            continue
        prompt_len, batch_size, recompute = parsed
        csv_path = find_summary_csv(subdir)
        if csv_path is None:
            print(f"  No *_summary.csv in '{subdir.name}', skipping", file=sys.stderr)
            continue
        row, cols = load_first_row(csv_path)
        if row is None:
            print(f"  Empty CSV in '{subdir.name}', skipping", file=sys.stderr)
            continue
        nosep = detect_nosep(cols)
        label = make_label(prompt_len, batch_size, recompute)
        entries.append((label, prompt_len, batch_size, recompute, row, cols, nosep))
        rc_str = f"  recompute={recompute}" if recompute is not None else ""
        print(f"  '{subdir.name}' -> {'nosep' if nosep else 'sep'}{rc_str}  csv={csv_path.name}",
              file=sys.stderr)

    # Sort by (prompt_len, batch_size, recompute) — None sorts before any int
    entries.sort(key=lambda e: (e[1], e[2], e[3] if e[3] is not None else -1))
    return entries


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(folder, out_path=None, dpi=150, show=False):
    folder = Path(folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning '{folder}' ...", file=sys.stderr)
    entries = scan_folder(folder)

    if not entries:
        print("No valid workload subdirectories found.", file=sys.stderr)
        sys.exit(1)

    # Check if all entries are same mode; warn if mixed
    modes = set(e[6] for e in entries)
    if len(modes) > 1:
        print("Warning: mixed sep and nosep subdirectories — using per-entry mode.",
              file=sys.stderr)

    # Build segments per entry
    labels   = []
    all_segs = []
    for label, _, _, _, row, cols, nosep in entries:
        labels.append(label)
        if nosep:
            all_segs.append(build_segments_nosep(row))
        else:
            all_segs.append(build_segments_sep(row))

    # Legend order: use nosep order if any nosep, else sep
    any_nosep = any(e[6] for e in entries)
    legend_order = NOSEP_LEGEND_ORDER if any_nosep else SEP_LEGEND_ORDER
    # If mixed, include both orders (deduped)
    if len(modes) > 1:
        seen = set()
        legend_order = []
        for l in SEP_LEGEND_ORDER + NOSEP_LEGEND_ORDER:
            if l not in seen:
                legend_order.append(l)
                seen.add(l)

    used_legends = [l for l in legend_order
                    if any(s.get(l, 0.0) > 0 for s in all_segs)]

    n = len(entries)
    x = np.arange(n)
    bar_width = min(0.55, max(0.3, 0.8 / max(n, 1)))

    fig, ax = plt.subplots(figsize=(max(6, n * 2.0), 7))
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
    ax.set_xlabel("Workload (prompt length / batch size)", fontsize=10)
    ax.set_title("Latency Breakdown across Workloads",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max_total * 1.13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=[mpatches.Patch(facecolor=COLORS[l], edgecolor="#cccccc", label=l)
                 for l in used_legends],
        loc="upper left", fontsize=9, framealpha=0.9, edgecolor="#cccccc",
    )

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved to {out_path}", file=sys.stderr)

    if show:
        plt.show()

    # Sanity check
    for (label, _, _, _, row, cols, nosep), segs in zip(entries, all_segs):
        seg_total = sum(segs.values())
        if nosep:
            ref = fv(row, "sum-all")
        else:
            ref = fv(row, "mha-gen-sum") + fv(row, "mlp-sum")
        diff = abs(seg_total - ref)
        status = "OK" if diff < 0.5 else f"MISMATCH diff={diff:.1f}"
        print(f"  {label}: bar_total={seg_total:.1f} µs  ref={ref:.1f} µs  [{status}]",
              file=sys.stderr)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot latency breakdown across multiple workload subdirectories."
    )
    parser.add_argument(
        "folder",
        help="Root folder containing prompt_* subdirectories.",
    )
    parser.add_argument(
        "--out", default=None, metavar="OUTPUT.png",
        help="Output image path (default: <folder_name>_multi.png).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true",
                        help="Call plt.show() for interactive display.")
    args = parser.parse_args()

    out = args.out or (Path(args.folder).name + "_multi.png")
    plot(args.folder, out_path=out, dpi=args.dpi, show=args.show)


if __name__ == "__main__":
    main()