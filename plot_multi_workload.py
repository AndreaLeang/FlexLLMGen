#!/usr/bin/env python3
"""
Multi-workload stacked bar plot.

Scans a folder for subdirectories named:
    prompt_{N}
    prompt_{N}_bs{B}
    prompt_{N}_recompute_{R}
    prompt_{N}_bs{B}_recompute_{R}
    prompt_{N}_bs{B}_sep
    prompt_{N}_bs{B}_recompute_{R}_sep
    (suffixes _bs{B}, _recompute_{R}, _sep in any order)

In each subdirectory, finds the first file matching *_summary.csv,
reads rows (averaged per --avg / --avg-n), auto-detects sep/nosep/batched,
builds latency segments, and plots all workloads side-by-side.

Usage:
    python plot_multi_workload.py <folder> [--out output.png] [--dpi 150] [--show]
                                           [--avg | --avg-n N] [--batched]
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
# Colors and legend ordering (shared with plot_latency_single.py)
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
    "Recompute CUDA",
    "Recompute Load",
    "PinnedMemory CPU",
    "KVCache Load",
    "KVCache Store",
    "Misc. CPU",
]

BATCHED_LEGEND_ORDER = [
    "Recompute Load",
    "Recompute CUDA",
    "MHA CUDA",
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


def detect_batched(all_cols, rows=None):
    """Batched CSVs are identified by the presence of the 'critical-path' column."""
    return "critical-path" in all_cols


# ---------------------------------------------------------------------------
# Segment builders (identical logic to plot_latency_single.py)
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
    recompute        = fv(row, "recompute-cuda")
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
        segs["Compute CUDA"]   = compute_cuda_sum - recompute
        segs["Recompute CUDA"] = recompute
        segs["PinnedMemory CPU"] = segs.get("PinnedMemory CPU", 0.0) + pin2
    elif phase2_winner.startswith("load-cache"):            # winner B
        segs["KVCache Load"] = lc1 + lc2
    else:                                                   # winner C
        segs["KVCache Store"] = sc1 + sc2

    segs["Misc. CPU"] = fv(row, "misc-cpu")
    return segs


def build_segments_batched(row):
    """Identical to plot_latency_single.py's build_segments_batched."""
    crit_winner  = row.get("critical-path-winner", "")
    inner_winner = row.get("path1-inner-winner", "")

    lhc       = fv(row, "load-hidden-compute-cudamemcpy")
    pm1       = fv(row, "pin-memory-1")
    pm2       = fv(row, "pin-memory-2")
    lc1       = fv(row, "load-cache-cudamemcpy-1")
    lc2       = fv(row, "load-cache-cudamemcpy-2")
    sc1       = fv(row, "store-cache-cudamemcpy-1")
    sc2       = fv(row, "store-cache-cudamemcpy-2")
    recompute = fv(row, "recompute-cuda")
    mha_cuda  = fv(row, "mha-gen-cuda")
    sum_all   = fv(row, "sum-all")
    crit      = fv(row, "critical-path")

    segs = {}

    if crit_winner == "path1":
        if inner_winner == "subpath1":
            segs["PinnedMemory CPU"] = pm1 + pm2
            segs["KVCache Load"]     = lc2
        elif inner_winner == "subpath2":
            segs["PinnedMemory CPU"] = pm1
            segs["KVCache Load"]     = lc1 + lc2
        else:  # subpath3
            segs["Recompute Load"] = lhc
            segs["KVCache Load"]   = lc1 + lc2
    elif crit_winner == "path2":
        segs["Recompute Load"] = lhc
        segs["Recompute CUDA"] = recompute
        segs["MHA CUDA"]       = mha_cuda
    else:  # path3
        segs["PinnedMemory CPU"] = pm1 + pm2
        segs["KVCache Store"]    = sc1 + sc2

    segs["Misc. CPU"] = sum_all - crit
    return segs


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------

# Base pattern: prompt_{N} followed by any combination of _bs{B}, _nbs{B},
# _recompute_{R}, _sep in any order.
SUBDIR_PATTERN    = re.compile(r"^prompt_(\d+)((?:_(?:nbs\d+|bs\d+|recompute_\d+|sep))*)$", re.IGNORECASE)
BS_PATTERN        = re.compile(r"_bs(\d+)", re.IGNORECASE)
NBS_PATTERN       = re.compile(r"_nbs(\d+)", re.IGNORECASE)
RECOMPUTE_PATTERN = re.compile(r"_recompute_(\d+)", re.IGNORECASE)
SEP_PATTERN       = re.compile(r"_sep(?:_|$)", re.IGNORECASE)


def parse_subdir(name):
    """
    Parse a subdir name such as:
        prompt_2048
        prompt_2048_bs4
        prompt_2048_nbs2
        prompt_2048_recompute_512
        prompt_2048_bs4_recompute_512
        prompt_2048_bs4_sep
        prompt_2048_nbs2_sep
        prompt_2048_bs4_recompute_512_sep
    Returns (prompt_length, batch_size, num_batches, recompute_length, is_sep) or None if no match.
    batch_size defaults to 1; num_batches defaults to 1; recompute_length is None when absent; is_sep is bool.
    """
    m = SUBDIR_PATTERN.match(name)
    if not m:
        return None
    prompt_len = int(m.group(1))
    rest = m.group(2)
    bs_m  = BS_PATTERN.search(rest)
    nbs_m = NBS_PATTERN.search(rest)
    rc_m  = RECOMPUTE_PATTERN.search(rest)
    batch_size  = int(bs_m.group(1))  if bs_m  else 1
    num_batches = int(nbs_m.group(1)) if nbs_m else 1
    recompute   = int(rc_m.group(1))  if rc_m  else None
    is_sep      = bool(SEP_PATTERN.search(rest))
    return prompt_len, batch_size, num_batches, recompute, is_sep


def find_summary_csv(subdir: Path):
    """Find the first *_summary.csv in a subdirectory."""
    candidates = sorted(subdir.glob("*_summary.csv"))
    return candidates[0] if candidates else None


def load_averaged_row(csv_path: Path, max_rows: int = None):
    """
    Load rows from a summary CSV and return a single averaged row.

    Only rows where every numeric column is non-empty are included
    (string columns such as *-winner and *-origin are excluded from
    the completeness check and are carried over from the first valid row).

    max_rows: if given, consider at most this many complete rows.
              None means use all complete rows.

    Returns (averaged_row_dict, cols, n_used) or (None, [], 0) if no rows qualify.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        all_rows = list(reader)

    if not all_rows:
        return None, [], 0

    first = all_rows[0]
    numeric_cols = []
    string_cols  = []
    for c in cols:
        v = first.get(c, "")
        try:
            float(v)
            numeric_cols.append(c)
        except (ValueError, TypeError):
            string_cols.append(c)

    complete_rows = [
        r for r in all_rows
        if all(r.get(c, "") not in ("", None) for c in numeric_cols)
    ]

    if not complete_rows:
        return None, [], 0

    if max_rows is not None:
        complete_rows = complete_rows[:max_rows]

    n = len(complete_rows)

    averaged = {}
    for c in numeric_cols:
        averaged[c] = str(round(sum(float(r[c]) for r in complete_rows) / n, 6))
    for c in string_cols:
        averaged[c] = complete_rows[0].get(c, "")

    return averaged, cols, n


def make_label(prompt_len, batch_size, num_batches, recompute, is_sep=False):
    """
    Build a multi-line x-tick label showing all non-default dimensions.
      prompt length always shown as p{N}
      batch size shown as bs{B} only if > 1
      num batches shown as nbs{B} only if > 1
      recompute shown as rc{R} only if present
      sep shown as 'sep' only if True
    """
    parts = [f"p{prompt_len}"]
    if batch_size != 1:
        parts.append(f"bs{batch_size}")
    if num_batches != 1:
        parts.append(f"nbs{num_batches}")
    if recompute is not None:
        parts.append(f"rc{recompute}")
    if is_sep:
        parts.append("sep")
    return "\n".join(parts)


def scan_folder(folder: Path, avg_n: int = None):
    """
    Scan folder for prompt_* subdirs, return list of:
      (label, prompt_len, batch_size, recompute, is_sep, row, cols, mode_str)
    sorted by (prompt_len, batch_size, recompute, is_sep).

    avg_n: number of complete rows to average per CSV.
           None means average all complete rows.
           Pass 1 to reproduce the old single-row behaviour.
    """
    entries = []
    for subdir in sorted(folder.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = parse_subdir(subdir.name)
        if parsed is None:
            print(f"  Skipping '{subdir.name}' (no match)", file=sys.stderr)
            continue
        prompt_len, batch_size, num_batches, recompute, is_sep = parsed
        csv_path = find_summary_csv(subdir)
        if csv_path is None:
            print(f"  No *_summary.csv in '{subdir.name}', skipping", file=sys.stderr)
            continue
        row, cols, n_used = load_averaged_row(csv_path, max_rows=avg_n)
        if row is None:
            print(f"  No complete rows in '{subdir.name}', skipping", file=sys.stderr)
            continue
        is_batched = detect_batched(cols, [row])
        is_nosep   = detect_nosep(cols)
        mode_str   = "batched" if is_batched else ("nosep" if is_nosep else "sep")
        label = make_label(prompt_len, batch_size, num_batches, recompute, is_sep)
        entries.append((label, prompt_len, batch_size, num_batches, recompute, is_sep, row, cols, mode_str))
        rc_str  = f"  recompute={recompute}" if recompute is not None else ""
        nbs_str = f"  nbs={num_batches}" if num_batches != 1 else ""
        sep_str = "  sep" if is_sep else ""
        avg_str = f"avg over {n_used} rows" if avg_n is None else f"avg over first {n_used} rows"
        print(f"  '{subdir.name}' -> {mode_str}{rc_str}{nbs_str}{sep_str}  "
              f"{avg_str}  csv={csv_path.name}", file=sys.stderr)

    # Sort by (prompt_len, batch_size, num_batches, recompute, is_sep)
    entries.sort(key=lambda e: (e[1], e[2], e[3], e[4] if e[4] is not None else -1, e[5]))
    return entries


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(folder, out_path=None, dpi=150, show=False, avg_n=None, batched=False):
    folder = Path(folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    avg_desc = "all complete rows" if avg_n is None else f"first {avg_n} complete rows"
    print(f"Scanning '{folder}' (averaging over {avg_desc}) ...", file=sys.stderr)
    entries = scan_folder(folder, avg_n=avg_n)

    if not entries:
        print("No valid workload subdirectories found.", file=sys.stderr)
        sys.exit(1)

    # mode_str per entry is "sep", "nosep", or "batched"
    # --batched flag forces batched for all entries
    entry_modes = set(e[8] for e in entries)
    if len(entry_modes) > 1:
        print(f"Warning: mixed modes {entry_modes} — using per-entry mode.",
              file=sys.stderr)

    labels    = []
    all_segs  = []
    all_modes = []
    for label, _, _, _, _, _, row, cols, mode_str in entries:
        labels.append(label)
        m = "batched" if batched else mode_str
        all_modes.append(m)
        if m == "batched":
            all_segs.append(build_segments_batched(row))
        elif m == "nosep":
            all_segs.append(build_segments_nosep(row))
        else:
            all_segs.append(build_segments_sep(row))

    # Choose legend order based on dominant mode
    unique_modes = set(all_modes)
    if "batched" in unique_modes and len(unique_modes) == 1:
        legend_order = BATCHED_LEGEND_ORDER
    elif "nosep" in unique_modes and "sep" not in unique_modes and "batched" not in unique_modes:
        legend_order = NOSEP_LEGEND_ORDER
    elif "sep" in unique_modes and "nosep" not in unique_modes and "batched" not in unique_modes:
        legend_order = SEP_LEGEND_ORDER
    else:
        # Mixed: union of all legend orders, deduped
        seen = set()
        legend_order = []
        for l in BATCHED_LEGEND_ORDER + SEP_LEGEND_ORDER + NOSEP_LEGEND_ORDER:
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
    for (label, _, _, _, _, _, row, cols, mode_str), segs, m in zip(entries, all_segs, all_modes):
        seg_total = sum(segs.values())
        if m in ("batched", "nosep"):
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

    avg_group = parser.add_mutually_exclusive_group()
    avg_group.add_argument(
        "--avg", action="store_true",
        help="Average all complete rows in each summary CSV (default: use first row only).",
    )
    avg_group.add_argument(
        "--avg-n", type=int, default=None, metavar="N",
        help="Average the first N complete rows in each summary CSV.",
    )
    parser.add_argument(
        "--batched", action="store_true",
        help="Force batched mode for all entries (auto-detected if omitted).",
    )

    args = parser.parse_args()

    if args.avg:
        avg_n = None
    elif args.avg_n is not None:
        avg_n = args.avg_n
    else:
        avg_n = 1

    out = args.out or (Path(args.folder).name + "_multi.png")
    plot(args.folder, out_path=out, dpi=args.dpi, show=args.show,
         avg_n=avg_n, batched=args.batched)


if __name__ == "__main__":
    main()