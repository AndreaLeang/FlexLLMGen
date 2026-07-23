#!/usr/bin/env python3
"""
plot_gt_breakdown.py
=====================
Stacked bar plot of the *ground-truth* latency breakdown (no estimator bars),
with a twin right-hand y-axis showing GT decode throughput. Adapted from
plot_gt_vs_estimator.py, stripped down to GT-only, with a few extra options.

Usage
-----
  python plot_gt_breakdown.py sweep.csv \\
      [--x-axis batch_size|recompute_len|both|cpu_gpu_ratio] \\
      [--figsize W H] [--no-legend] [--hide-non-ok] \\
      [--normalize] [--normalize-throughput] \\
      [--hide-values] [--no-title] \\
      [--out figure.png] [--dpi 150] [--show]

Options (see plot_gt_breakdown() docstring for full detail)
-------------------------------------------------------------
  --x-axis                 What goes on the x-tick labels: batch_size,
                            recompute_len, both, or cpu_gpu_ratio (for
                            --cpu-computation-ratios sweeps). Default:
                            batch_size.
  --no-legend               Hide the legend (shown by default).
  --figsize W H             Explicit figure size in inches. Default (omitted):
                            width scales with the number of entries in the CSV.
  --hide-non-ok             Drop rows whose status != 'ok' entirely (they will
                            not appear on the x-axis at all). Default behavior
                            is to KEEP them as x-ticks but draw no bar for them
                            and shade that column instead.
  --normalize                Multiply each bar by num_batches so the y-axis
                            shows total latency across the full workload
                            (fair comparison across batch sizes). Same as the
                            --normalize flag in plot_gt_vs_estimator.py.
  --normalize-throughput    Scale the throughput axis so the max value is 1.0.
  --hide-values             Hide numeric text annotations on bars/totals/
                            throughput markers (axes/ticks/grid stay as-is).
  --no-title                Omit the figure title entirely.
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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Color palette / stacking order for GT segments (same as plot_gt_vs_estimator.py)
# ---------------------------------------------------------------------------

GT_COLORS = {
    "PinnedMemory CPU":   "#FFC107",   # amber
    "Recompute Load":     "#FF9800",   # orange
    "Recompute CUDA":     "#90CAF9",   # light blue
    "MHA CUDA":           "#2196F3",   # blue
    "KVCache Load":       "#9C27B0",   # purple
    "KVCache Store":      "#E91E63",   # pink
    "Misc. CPU":          "#9E9E9E",   # grey
}

GT_STACK_ORDER = [
    "Recompute Load",
    "Recompute CUDA",
    "MHA CUDA",
    "PinnedMemory CPU",
    "KVCache Load",
    "KVCache Store",
    "Misc. CPU",
]

# ---------------------------------------------------------------------------
# Color palette / stacking order for --cpu-computation mode GT segments (see
# GT_SEGMENT_NAMES_CPU_COMPUTE / build_gt_segments_cpu_computation in
# gt_vs_estimator.py, and trace_analyzer.py's --cpu-computation mode). This
# taxonomy is distinct from GT_COLORS/GT_STACK_ORDER above: "cpu_copy" is the
# 2 CPU<->CPU general_copy() calls in load_cache (outside of smart_copy) --
# NOT the pageable-to-pinned-memory concept behind "PinnedMemory CPU" below.
# "PinnedMemory CPU" here is the aten::pin_memory call nested inside each
# smart_copy call (pins the CPU-side buffer before its async H2D
# cudaMemcpyAsync) -- same concept and same color as --batched mode's own
# "PinnedMemory CPU" segment, just scoped per-smart_copy-call instead of
# whole-window. "other cpu copy" is each smart_copy call's own duration minus
# its nested pin_memory duration (the rest of smart_copy/general_copy/copy's
# CPU-side work). "cpu_compute" is the CPU-side portion of compute_layer not
# accounted for by the CUDA kernels it dispatches.
# ---------------------------------------------------------------------------

GT_COLORS_CPU_COMPUTE = {
    "load_weight":         "#8D6E63",   # brown
    "load_hidden_compute": "#FF9800",   # orange
    "cpu_copy":            "#FDD835",   # yellow (distinct from PinnedMemory's amber)
    "PinnedMemory CPU":    "#FFC107",   # amber (same color as --batched PinnedMemory CPU)
    "other cpu copy":      "#FFB300",   # dark amber (formerly "smart_copy"'s color)
    "KVCache Load":        "#9C27B0",   # purple (same concept as --batched KVCache Load)
    "load_hidden":         "#4CAF50",   # green
    "GPU Compute":         "#2196F3",   # blue (same concept as --batched MHA CUDA)
    "cpu_compute":         "#00BCD4",   # cyan
    "KVCache Store":       "#E91E63",   # pink (same as --batched KVCache Store)
    "store_hidden":        "#8BC34A",   # light green
    "sync":                "#607D8B",   # blue-grey
    "Misc. CPU":           "#9E9E9E",   # grey (same as --batched Misc. CPU)
}

GT_STACK_ORDER_CPU_COMPUTE = [
    "load_weight",
    "load_hidden_compute",
    "cpu_copy",
    "PinnedMemory CPU",
    "other cpu copy",
    "KVCache Load",
    "load_hidden",
    "cpu_compute",
    "GPU Compute",
    "KVCache Store",
    "store_hidden",
    "sync",
    "Misc. CPU",
]

BAR_ALPHA = 0.92
EDGE_COLOR = "white"
EDGE_WIDTH = 0.5

US_TO_MS = 1e-3   # CSV latency columns (gt_*_us) are in microseconds; display in ms

THROUGHPUT_COLOR = "#D32F2F"   # strong red
THROUGHPUT_MARKER = "D"        # diamond

NOT_OK_SHADE_COLOR = "#BDBDBD"
NOT_OK_SHADE_ALPHA = 0.35


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _fv(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (ValueError, TypeError):
        return default


def load_csv(csv_path: str) -> Tuple[List[Dict], List[str]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, fieldnames


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


def get_gt_segments_cpu_computation(row: Dict) -> Dict[str, float]:
    """
    Same idea as get_gt_segments(), but reads the columns produced for
    --cpu-computation sweeps (see GT_SEGMENT_NAMES_CPU_COMPUTE /
    build_gt_segments_cpu_computation in gt_vs_estimator.py).
    """
    mapping = {
        "load_weight":         "gt_load_weight_us",
        "load_hidden_compute": "gt_load_hidden_compute_us",
        "cpu_copy":            "gt_cpu_copy_us",
        "PinnedMemory CPU":    "gt_PinnedMemory_CPU_us",
        "other cpu copy":      "gt_other_cpu_copy_us",
        "KVCache Load":        "gt_KVCache_Load_us",
        "load_hidden":         "gt_load_hidden_us",
        "GPU Compute":         "gt_GPU_Compute_us",
        "cpu_compute":         "gt_cpu_compute_us",
        "KVCache Store":       "gt_KVCache_Store_us",
        "store_hidden":        "gt_store_hidden_us",
        "sync":                "gt_sync_us",
        "Misc. CPU":           "gt_Misc_CPU_us",
    }
    return {seg: _fv(row, col) for seg, col in mapping.items()}


def is_ok(row: Dict) -> bool:
    return (row.get("status") or "").strip().lower() == "ok"


def _fmt_ratio(val) -> str:
    """
    Format a cpu_gpu_ratio CSV value for an x-tick label. The column is
    written as a float string ("13.0", "0.0", ...) by build_csv_row(); '%g'
    strips the trailing ".0" for the common integer-percent case ("13.0" ->
    "13") while still showing a fractional ratio as-is ("12.5" -> "12.5").
    Non-numeric/missing values pass through unchanged so a malformed row
    still gets *some* label instead of raising.
    """
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    return f"{f:g}"


def make_x_label(row: Dict, x_axis: str) -> str:
    bs = row.get("batch_size", "?")
    rc = row.get("recompute_len", "?")
    if x_axis == "batch_size":
        return f"{bs}"
    elif x_axis == "recompute_len":
        return f"{rc}"
    elif x_axis == "both":
        return f"{bs}\n{rc}"
    elif x_axis == "cpu_gpu_ratio":
        return _fmt_ratio(row.get("cpu_gpu_ratio", "?"))
    else:
        raise ValueError(f"Unknown x_axis: {x_axis!r}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_gt_breakdown(
    csv_path: str,
    x_axis: str = "batch_size",
    out_path: Optional[str] = None,
    dpi: int = 150,
    show: bool = False,
    title: Optional[str] = None,
    show_legend: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    show_non_ok: bool = True,
    normalize: bool = False,
    normalize_throughput: bool = False,
    show_values: bool = True,
    show_title: bool = True,
    cpu_computation: bool = False,
) -> plt.Figure:
    """
    Create the ground-truth-only latency breakdown plot with a throughput
    overlay on a twin y-axis.

    Parameters
    ----------
    csv_path
        Path to the sweep CSV (e.g. gt_vs_estimator.py output).
    x_axis
        What to show on the x-tick labels: "batch_size", "recompute_len",
        "both", or "cpu_gpu_ratio" (the CPU/GPU attention-compute split
        percentage from a --cpu-computation sweep -- see
        gt_vs_estimator.py's --cpu-computation-ratios). Rows are plotted in
        the order they appear in the CSV; this does not sort by x_axis, so
        a --cpu-computation-ratios sweep already comes out in ratio order
        because that's the order gt_vs_estimator.py wrote the rows in
        (same as batch_size/recompute_len sweeps today).
    out_path
        Save the figure here (None = don't save).
    dpi
        DPI for the saved figure.
    show
        Call plt.show() at the end.
    title
        Figure title override (a sensible default is generated otherwise).
    show_legend
        If True (default), draw the segment legend. If False, omit it.
    figsize
        Explicit (width, height) in inches. If None (default), the width is
        scaled proportionally to the number of entries (rows) in the CSV so
        wide sweeps don't get squished, and the height is fixed at 7.
    show_non_ok
        If True (default), rows whose status != 'ok' are still shown as
        x-ticks, but no bar is drawn for them -- instead that column is
        shaded grey (annotated with the status / skip_reason if present).
        If False, such rows are dropped entirely and do not appear on the
        x-axis at all.
    normalize
        Matches the original plot_gt_vs_estimator.py --normalize flag: if
        True, every bar's segments are multiplied by that row's num_batches
        (= num_prompts / batch_size), so the y-axis shows total latency
        across all iterations needed to process the full num_prompts
        workload, rather than per-iteration latency. This puts different
        batch sizes on a fair total-work basis (e.g. bs=1 x 16 iterations
        vs. bs=8 x 2 iterations). Default: False (per-iteration latency,
        raw gt_*_us columns as-is).
    normalize_throughput
        If True, the twin-axis throughput values are divided by the max
        observed throughput so the tallest point reads as 1.0. If False
        (default), raw tok/s values are plotted.
    show_values
        If True (default), draw the numeric text annotations on top of/inside
        the bar segments and totals, and next to each throughput marker. If
        False, those text labels are omitted -- the y-axes, ticks, and grid
        are unaffected, only the in-plot number annotations are hidden.
    show_title
        If True (default), draw the figure title. If False, omit it entirely
        (the `title` argument is ignored in that case).
    cpu_computation
        If True, read the GT segment columns produced by --cpu-computation
        sweeps (gt_vs_estimator.py --cpu-computation) instead of the default
        --batched taxonomy: stacks load_weight/load_hidden_compute/cpu_copy/
        PinnedMemory CPU/other cpu copy/KVCache Load/load_hidden/cpu_compute/
        GPU Compute/KVCache Store/store_hidden/sync/Misc. CPU (see
        GT_STACK_ORDER_CPU_COMPUTE /
        GT_COLORS_CPU_COMPUTE / get_gt_segments_cpu_computation). Default:
        False (unchanged --batched behavior).
    """
    stack_order = GT_STACK_ORDER_CPU_COMPUTE if cpu_computation else GT_STACK_ORDER
    colors = GT_COLORS_CPU_COMPUTE if cpu_computation else GT_COLORS
    segment_fn = get_gt_segments_cpu_computation if cpu_computation else get_gt_segments

    rows, fieldnames = load_csv(csv_path)
    if not rows:
        print("No rows in CSV.", file=sys.stderr)
        sys.exit(1)

    if not show_non_ok:
        rows = [r for r in rows if is_ok(r)]
    if not rows:
        print("No rows left after filtering non-'ok' entries.", file=sys.stderr)
        sys.exit(1)

    n = len(rows)
    x = np.arange(n)
    bar_w = 0.6

    # ------------------------------------------------------------- figure --
    if figsize is not None:
        fig_w, fig_h = figsize
    else:
        fig_w = max(8.0, n * 1.0)
        fig_h = 7.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    legend_patches: Dict[str, mpatches.Patch] = {}

    def add_patch(label: str, color: str):
        if label not in legend_patches:
            legend_patches[label] = mpatches.Patch(
                facecolor=color, edgecolor="#cccccc", alpha=BAR_ALPHA, label=label
            )

    # --------------------------------------------------------------- bars --
    ok_mask = np.array([is_ok(r) for r in rows])
    # Per-row scale factor: 1.0 normally, num_batches when normalize=True
    # (mirrors the original plot_gt_vs_estimator.py --normalize flag).
    multipliers = np.array([
        _fv(r, "num_batches", 1.0) if normalize else 1.0
        for r in rows
    ])
    bottoms = np.zeros(n)
    all_totals = np.zeros(n)

    for seg in stack_order:
        vals = np.array([segment_fn(r).get(seg, 0.0) if ok else 0.0
                          for r, ok in zip(rows, ok_mask)]) * multipliers * US_TO_MS
        color = colors[seg]
        # Only draw bars for 'ok' rows -- pass width 0 (i.e. skip) for
        # not-ok rows by masking x/vals/bottoms down to the ok subset.
        if ok_mask.any():
            ax.bar(
                x[ok_mask] + 0, vals[ok_mask], bar_w,
                bottom=bottoms[ok_mask],
                color=color, alpha=BAR_ALPHA,
                edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH,
            )
            for xi, v, b in zip(x[ok_mask], vals[ok_mask], bottoms[ok_mask]):
                if show_values and v > 0.06:
                    ax.text(
                        xi, b + v / 2, f"{v:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold",
                    )
        bottoms += vals
        add_patch(seg, color)

    all_totals = bottoms.copy()

    if show_values:
        total_offset = (all_totals.max() if n else 1.0) * 0.012 or 0.01
        for xi, ok, total in zip(x, ok_mask, all_totals):
            if ok:
                ax.text(
                    xi, total + total_offset, f"{total:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333",
                )

    # ---------------------------------------------------- shade non-ok cols
    shade_handle = None
    for xi, r, ok in zip(x, rows, ok_mask):
        if ok:
            continue
        ax.axvspan(xi - bar_w / 2 - 0.05, xi + bar_w / 2 + 0.05,
                   color=NOT_OK_SHADE_COLOR, alpha=NOT_OK_SHADE_ALPHA, zorder=0)
        label = (r.get("status") or "?").strip()
        reason = (r.get("skip_reason") or "").strip()
        note = label if not reason else f"{label}\n({reason})"
        ax.text(
            xi, 0.5, note, transform=ax.get_xaxis_transform(),
            ha="center", va="center", fontsize=7.5, color="#616161",
            style="italic", rotation=90,
        )
        if shade_handle is None:
            shade_handle = mpatches.Patch(
                facecolor=NOT_OK_SHADE_COLOR, alpha=NOT_OK_SHADE_ALPHA,
                edgecolor="none", label="status != 'ok' (no data)",
            )

    # ---------------------------------------------------------------- axes --
    max_total = max(all_totals.max() if n else 1.0, 1.0)
    ax.set_ylim(0, max_total * 1.20)
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([make_x_label(r, x_axis) for r in rows], fontsize=9)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    _x_axis_labels = {
        "batch_size":    "Batch Size",
        "recompute_len": "Recompute Length",
        "both":          "Batch Size / Recompute Length",
        "cpu_gpu_ratio": "CPU/GPU Compute Ratio (% CPU)",
    }
    ax.set_xlabel(_x_axis_labels.get(x_axis, x_axis), fontsize=10)

    mode_suffix = " (CPU/GPU Compute Split)" if cpu_computation else ""
    default_title = (
        f"Ground Truth Latency Breakdown{mode_suffix} (total, all iterations)"
        if normalize else
        f"Ground Truth Latency Breakdown{mode_suffix}"
    )
    if show_title:
        ax.set_title(title or default_title, fontsize=13, fontweight="bold", pad=14)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------- throughput overlay --
    tp_vals: List[Optional[float]] = []
    for r, ok in zip(rows, ok_mask):
        if not ok:
            tp_vals.append(None)
            continue
        raw = r.get("gt_throughput_tok_per_s", "")
        try:
            v = float(raw)
            tp_vals.append(v if v > 0 else None)
        except (ValueError, TypeError):
            tp_vals.append(None)

    valid = [v for v in tp_vals if v is not None]
    ax2 = None
    tp_handle = None
    if not valid:
        print("[warn] no gt_throughput_tok_per_s values found -- skipping "
              "throughput overlay.", file=sys.stderr)
    else:
        ax2 = ax.twinx()

        tp_x = np.array([xi for xi, v in zip(x, tp_vals) if v is not None], dtype=float)
        tp_y = np.array([v for v in tp_vals if v is not None], dtype=float)

        if normalize_throughput:
            tp_max = tp_y.max()
            tp_y = tp_y / tp_max

        ax2.plot(
            tp_x, tp_y,
            color=THROUGHPUT_COLOR, linewidth=1.8,
            linestyle="--", alpha=0.9, zorder=5,
        )
        ax2.scatter(
            tp_x, tp_y,
            color=THROUGHPUT_COLOR, marker=THROUGHPUT_MARKER,
            s=55, zorder=6, alpha=0.95,
        )
        if show_values:
            fmt = "{:.2f}" if normalize_throughput else "{:.1f}"
            for xi_s, v in zip(tp_x, tp_y):
                ax2.text(
                    xi_s, v, "  " + fmt.format(v),
                    va="center", ha="left",
                    fontsize=7.5, color=THROUGHPUT_COLOR, fontweight="bold",
                )

        if normalize_throughput:
            ax2.set_ylim(0, 1.1)
        else:
            ax2.set_ylim(0, tp_y.max() * 1.30)
        ax2.set_ylabel(
            "Normalized Throughput" if normalize_throughput else "Throughput",
            fontsize=10, color=THROUGHPUT_COLOR,
        )
        ax2.tick_params(axis="y", colors=THROUGHPUT_COLOR)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(THROUGHPUT_COLOR)
        ax2.spines["top"].set_visible(False)

        tp_handle = mlines.Line2D(
            [], [],
            color=THROUGHPUT_COLOR, linewidth=1.8, linestyle="--",
            marker=THROUGHPUT_MARKER, markersize=6,
            label="GT Throughput" + (" (normalized)" if normalize_throughput else " (tok/s)"),
        )

    # -------------------------------------------------------------- legend --
    if show_legend:
        handles = list(legend_patches.values())
        if shade_handle is not None:
            handles = handles + [shade_handle]
        if tp_handle is not None:
            handles = handles + [tp_handle]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01 if ax2 is None else 1.12, 1.0),
            borderaxespad=0.0,
            fontsize=8,
            framealpha=0.92,
            edgecolor="#cccccc",
            ncol=1,
        )

    fig.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to: {out_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot GT-only latency breakdown from a sweep CSV."
    )
    parser.add_argument("csv", help="Path to the sweep CSV")
    parser.add_argument(
        "--x-axis",
        choices=["batch_size", "recompute_len", "both", "cpu_gpu_ratio"],
        default="batch_size",
        help="What to display as x-tick labels. 'cpu_gpu_ratio' reads the "
             "cpu_gpu_ratio column written by a gt_vs_estimator.py "
             "--cpu-computation --cpu-computation-ratios sweep. "
             "Default: batch_size.",
    )
    parser.add_argument("--out", default=None, metavar="OUTPUT.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--no-legend", action="store_true",
        help="Hide the legend (shown by default).",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=None, metavar=("W", "H"),
        help="Explicit figure size in inches. Default: width scales with the "
             "number of CSV entries.",
    )
    parser.add_argument(
        "--hide-non-ok", action="store_true",
        help="Drop rows whose status != 'ok' entirely, instead of keeping "
             "them as shaded, bar-less x-ticks (the default).",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help=(
            "Multiply each bar by num_batches (= num_prompts / batch_size) so "
            "the y-axis shows total latency across all iterations for the "
            "full workload. Enables fair comparison across different batch "
            "sizes. Matches the --normalize flag in plot_gt_vs_estimator.py."
        ),
    )
    parser.add_argument(
        "--normalize-throughput", action="store_true",
        help="Scale the twin-axis throughput values so the max is 1.0.",
    )
    parser.add_argument(
        "--hide-values", action="store_true",
        help="Hide the numeric text annotations on bars/totals/throughput "
             "markers (axes, ticks, and grid are unaffected). Shown by "
             "default.",
    )
    parser.add_argument(
        "--no-title", action="store_true",
        help="Omit the figure title entirely (shown by default).",
    )
    parser.add_argument(
        "--cpu-computation", action="store_true",
        help=(
            "Plot a sweep CSV produced by gt_vs_estimator.py --cpu-computation "
            "instead of the default --batched-mode CSV: stacks the "
            "cpu_computation GT segment taxonomy (load_weight/"
            "load_hidden_compute/cpu_copy/PinnedMemory CPU/other cpu copy/"
            "KVCache Load/load_hidden/cpu_compute/GPU Compute/KVCache Store/"
            "store_hidden/sync/Misc. CPU) "
            "instead of the --batched one (PinnedMemory CPU/Recompute Load/"
            "Recompute CUDA/MHA CUDA/KVCache Load/KVCache Store/Misc. CPU). "
            "Default: off (unchanged --batched behavior)."
        ),
    )
    args = parser.parse_args()

    out = args.out or str(Path(args.csv).with_suffix(".png"))
    plot_gt_breakdown(
        csv_path=args.csv,
        x_axis=args.x_axis,
        out_path=out,
        dpi=args.dpi,
        show=args.show,
        title=args.title,
        show_legend=not args.no_legend,
        figsize=tuple(args.figsize) if args.figsize else None,
        show_non_ok=not args.hide_non_ok,
        normalize=args.normalize,
        normalize_throughput=args.normalize_throughput,
        show_values=not args.hide_values,
        show_title=not args.no_title,
        cpu_computation=args.cpu_computation,
    )


if __name__ == "__main__":
    main()