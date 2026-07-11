#!/usr/bin/env python3
"""
plot_gt_vs_estimators.py
=========================
Grouped bar plot comparing the ground-truth latency breakdown against one or
more estimator modes (auto-detected from est_{mode}_total_us columns), plus
a printed analysis of how costly it would be to trust each estimator's
"fastest" configuration instead of the true (ground-truth) optimum.

This is the multi-bar sibling of plot_gt_breakdown.py -- same CLI surface
(--x-axis, --figsize, --no-legend, --hide-non-ok, --normalize,
--normalize-throughput, --hide-values, --no-title, default --out path) but
draws one bar per estimator mode alongside the GT bar at every x-tick,
grouped side by side.

Usage
-----
  python plot_gt_vs_estimators.py comparison.csv \\
      [--x-axis batch_size|recompute_len|both] [--modes MODE [MODE ...]] \\
      [--figsize W H] [--no-legend] [--hide-non-ok] \\
      [--normalize] [--normalize-throughput] \\
      [--hide-values] [--no-title] \\
      [--out figure.png] [--dpi 150] [--show]
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
# Color palette / stacking order (same as plot_gt_breakdown.py / the
# original plot_gt_vs_estimator.py)
# ---------------------------------------------------------------------------

GT_COLORS = {
    "PinnedMemory CPU":   "#FFC107",
    "Recompute Load":     "#FF9800",
    "Recompute CUDA":     "#90CAF9",
    "MHA CUDA":           "#2196F3",
    "KVCache Load":       "#9C27B0",
    "KVCache Store":      "#E91E63",
    "Misc. CPU":          "#9E9E9E",
}

EST_COLORS = {
    "PinnedMemory CPU (phase1)": "#FFE082",
    "PinnedMemory CPU (phase2)": "#FFB300",
    "Recompute Load":            "#FB8C00",
    "Recompute CUDA":            "#64B5F6",
    "MHA CUDA":                  "#1565C0",
    "KVCache Load":              "#9C27B0",
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

EST_STACK_ORDER = [
    "Recompute Load",
    "Recompute CUDA",
    "MHA CUDA",
    "PinnedMemory CPU (phase1)",
    "PinnedMemory CPU (phase2)",
    "KVCache Load",
]

# hatch, edge_color pairs to visually distinguish estimator modes
MODE_BAR_STYLES = [
    ("//",   "#1A237E"),
    ("\\\\", "#4A148C"),
    ("xx",   "#880E4F"),
    ("oo",   "#1B5E20"),
    ("++",   "#E65100"),
]

BAR_ALPHA = 0.92
EDGE_COLOR = "white"
EDGE_WIDTH = 0.5

US_TO_MS = 1e-3   # CSV latency columns (*_us) are in microseconds; display in ms

THROUGHPUT_COLOR = "#D32F2F"
THROUGHPUT_MARKER = "D"

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


def _detect_estimator_modes(fieldnames: List[str]) -> List[str]:
    """Infer estimator mode names from CSV columns named est_{mode}_total_us."""
    modes, seen = [], set()
    for col in fieldnames:
        if col.startswith("est_") and col.endswith("_total_us"):
            mode = col[len("est_"):-len("_total_us")]
            if mode not in seen:
                modes.append(mode)
                seen.add(mode)
    return modes


def load_comparison_csv(csv_path: str) -> Tuple[List[Dict], List[str], List[str]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    modes = _detect_estimator_modes(fieldnames)
    return rows, fieldnames, modes


def is_ok(row: Dict) -> bool:
    return (row.get("status") or "").strip().lower() == "ok"


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
    return {
        "PinnedMemory CPU (phase1)": _fv(row, f"{pfx}_PinnedMemory_CPU_phase1_us"),
        "PinnedMemory CPU (phase2)": _fv(row, f"{pfx}_PinnedMemory_CPU_phase2_us"),
        "Recompute Load":            _fv(row, f"{pfx}_Recompute_Load_us"),
        "Recompute CUDA":            _fv(row, f"{pfx}_Recompute_CUDA_us"),
        "MHA CUDA":                  _fv(row, f"{pfx}_MHA_CUDA_us"),
        "KVCache Load":              _fv(row, f"{pfx}_KVCache_Load_K_us") + _fv(row, f"{pfx}_KVCache_Load_V_us"),
    }


def get_est_total_us(row: Dict, mode: str) -> Optional[float]:
    raw = row.get(f"est_{mode}_total_us", "")
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def make_x_label(row: Dict, x_axis: str) -> str:
    bs = row.get("batch_size", "?")
    rc = row.get("recompute_len", "?")
    if x_axis == "batch_size":
        return f"{bs}"
    elif x_axis == "recompute_len":
        return f"{rc}"
    elif x_axis == "both":
        return f"{bs}\n{rc}"
    else:
        raise ValueError(f"Unknown x_axis: {x_axis!r}")


def config_label(row: Dict) -> str:
    """Human-readable configuration identifier for printed analysis output."""
    exp_id = row.get("experiment_id")
    if exp_id:
        return exp_id
    return f"bs={row.get('batch_size', '?')}, rc={row.get('recompute_len', '?')}"


# ---------------------------------------------------------------------------
# Estimator quality analysis
# ---------------------------------------------------------------------------

def analyze_estimator_choices(
    rows: List[Dict], modes: List[str], normalize: bool = False,
) -> Dict:
    """
    For each estimator mode, find the configuration it would pick as
    "fastest", then compare the *actual* (ground-truth) throughput at that
    configuration against the true best achievable ground-truth throughput.

    normalize
        If False (default), "fastest" = lowest predicted per-iteration
        latency (est_{mode}_total_us) -- i.e. what the estimator predicts
        for a single decode step, ignoring how many iterations different
        batch sizes need to clear the same workload.
        If True, "fastest" = lowest predicted *total* latency across the
        full workload (est_{mode}_total_us * num_batches), which is the
        fair comparison across configurations with different batch sizes
        (matches the plot's --normalize flag). This can change which
        config an estimator "picks", since per-iteration latency and
        total-workload latency don't always rank configs the same way
        once batch size varies num_batches.

    Returns a dict:
      {
        "normalize": bool,
        "gt_max_throughput": float,
        "gt_max_throughput_config": str,
        "per_mode": {
            mode: {
                "chosen_config": str,
                "chosen_est_latency_us": float,
                "tied_configs": [str, ...],   # other configs within 0.1% of the min
                "gt_throughput_at_chosen": float,
                "throughput_gap_pct": float,   # positive = estimator's pick is worse
            },
            ...
        },
      }
    """
    ok_rows = [r for r in rows if is_ok(r)]
    if not ok_rows:
        return {}

    def gt_tp(row: Dict) -> Optional[float]:
        raw = row.get("gt_throughput_tok_per_s", "")
        try:
            v = float(raw)
            return v if v > 0 else None
        except (ValueError, TypeError):
            return None

    tp_rows = [(r, gt_tp(r)) for r in ok_rows]
    tp_rows = [(r, v) for r, v in tp_rows if v is not None]
    if not tp_rows:
        return {}

    gt_best_row, gt_max_tp = max(tp_rows, key=lambda rv: rv[1])

    per_mode = {}
    for mode in modes:
        candidates = []
        for r in ok_rows:
            per_iter = get_est_total_us(r, mode)
            if per_iter is None:
                continue
            mult = _fv(r, "num_batches", 1.0) if normalize else 1.0
            candidates.append((r, per_iter * mult))
        if not candidates:
            continue
        chosen_row, chosen_latency = min(candidates, key=lambda rv: rv[1])
        # Flag any other configs within 0.1% of the chosen (effectively tied,
        # since a min() tie-break is otherwise silently arbitrary).
        tied = [
            config_label(r) for r, v in candidates
            if r is not chosen_row and chosen_latency > 0
            and abs(v - chosen_latency) / chosen_latency < 0.001
        ]
        chosen_tp = gt_tp(chosen_row)
        if chosen_tp is None:
            continue
        gap_pct = (gt_max_tp - chosen_tp) / gt_max_tp * 100.0
        per_mode[mode] = {
            "chosen_config": config_label(chosen_row),
            "chosen_est_latency_us": chosen_latency,
            "tied_configs": tied,
            "gt_throughput_at_chosen": chosen_tp,
            "throughput_gap_pct": gap_pct,
        }

    return {
        "normalize": normalize,
        "gt_max_throughput": gt_max_tp,
        "gt_max_throughput_config": config_label(gt_best_row),
        "per_mode": per_mode,
    }


def print_estimator_analysis(analysis: Dict) -> None:
    if not analysis:
        print("\n[analysis] No valid ground-truth throughput data available.",
              file=sys.stderr)
        return

    latency_kind = (
        "total latency across the full workload (per-iteration x num_batches)"
        if analysis.get("normalize") else
        "per-iteration latency"
    )

    print("\n" + "=" * 72)
    print("ESTIMATOR CONFIGURATION-SELECTION ANALYSIS")
    print(f"(estimators pick their 'fastest' config by {latency_kind})")
    print("=" * 72)
    print(f"1) Max ground-truth throughput : {analysis['gt_max_throughput']:.2f} tok/s"
          f"   (config: {analysis['gt_max_throughput_config']})")

    if not analysis["per_mode"]:
        print("\nNo estimator modes had usable predictions.")
        return

    print(f"\n{'Estimator':<14} {'Chosen config':<40} {'GT throughput':>14} {'Gap vs. best':>13}")
    print("-" * 72)
    for mode, info in analysis["per_mode"].items():
        print(
            f"{mode:<14} {info['chosen_config']:<40} "
            f"{info['gt_throughput_at_chosen']:>10.2f} tok/s "
            f"{info['throughput_gap_pct']:>11.1f}%"
        )
        if info["tied_configs"]:
            print(f"{'':<14} (tied within 0.1% with: {', '.join(info['tied_configs'])})")
    print()
    print("2) 'Chosen config' = the configuration each estimator predicts has")
    print(f"   the lowest {latency_kind}.")
    print("3) 'GT throughput' = the *actual* measured throughput at that chosen")
    print("   configuration (not the estimator's prediction).")
    print("4) 'Gap vs. best' = % below the true max GT throughput from (1);")
    print("   positive means trusting that estimator costs you throughput.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_gt_vs_estimators(
    csv_path: str,
    x_axis: str = "batch_size",
    out_path: Optional[str] = None,
    dpi: int = 150,
    show: bool = False,
    modes_to_plot: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    show_non_ok: bool = True,
    normalize: bool = False,
    normalize_throughput: bool = False,
    show_values: bool = True,
    show_title: bool = True,
) -> Tuple[plt.Figure, Dict]:
    """
    Grouped bar plot: ground truth + every estimator mode, side by side at
    each x-tick, stacked by latency segment. Mirrors plot_gt_breakdown.py's
    option set; see that module's docstring for the shared options
    (x_axis, show_legend, figsize, show_non_ok, normalize,
    normalize_throughput, show_values, show_title).

    Additional parameter
    ---------------------
    modes_to_plot
        Subset of estimator mode names to include (None = every est_{mode}_*
        group of columns found in the CSV).

    Returns
    -------
    (fig, analysis) where `analysis` is the dict produced by
    analyze_estimator_choices() (also printed to stdout as a side effect).
    """
    rows, fieldnames, all_modes = load_comparison_csv(csv_path)
    if not rows:
        print("No rows in CSV.", file=sys.stderr)
        sys.exit(1)

    modes = modes_to_plot if modes_to_plot is not None else all_modes
    if not modes:
        print("No estimator modes found in CSV (looked for est_{mode}_total_us "
              "columns).", file=sys.stderr)
        sys.exit(1)

    if not show_non_ok:
        rows = [r for r in rows if is_ok(r)]
    if not rows:
        print("No rows left after filtering non-'ok' entries.", file=sys.stderr)
        sys.exit(1)

    ok_mask = np.array([is_ok(r) for r in rows])
    n_ticks = len(rows)
    n_bars = 1 + len(modes)
    multipliers = np.array([
        _fv(r, "num_batches", 1.0) if normalize else 1.0
        for r in rows
    ])

    group_width = 0.8
    bar_w = group_width / n_bars
    offsets = np.linspace(-(group_width / 2) + bar_w / 2,
                          (group_width / 2) - bar_w / 2,
                          n_bars)
    x = np.arange(n_ticks)

    if figsize is not None:
        fig_w, fig_h = figsize
    else:
        fig_w = max(8.0, n_ticks * n_bars * 1.4)
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

    all_totals: List[float] = []

    # ------------------------------------------------------------------ GT --
    gt_bottoms = np.zeros(n_ticks)
    for seg in GT_STACK_ORDER:
        vals = np.array([get_gt_segments(r).get(seg, 0.0) if ok else 0.0
                          for r, ok in zip(rows, ok_mask)]) * multipliers * US_TO_MS
        color = GT_COLORS[seg]
        if ok_mask.any():
            ax.bar(
                x[ok_mask] + offsets[0], vals[ok_mask], bar_w,
                bottom=gt_bottoms[ok_mask],
                color=color, alpha=BAR_ALPHA,
                edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH,
            )
            if show_values:
                for xi, v, b in zip(x[ok_mask], vals[ok_mask], gt_bottoms[ok_mask]):
                    if v > 0.06:
                        ax.text(
                            xi + offsets[0], b + v / 2, f"{v:.2f}",
                            ha="center", va="center", fontsize=6.5,
                            color="white", fontweight="bold",
                        )
        gt_bottoms += vals
        add_patch(f"GT: {seg}", color)

    all_totals.extend(gt_bottoms.tolist())

    # --------------------------------------------------------- Estimators --
    mode_handles: List[mpatches.Patch] = []
    est_bottoms_by_mode: Dict[str, np.ndarray] = {}
    for mi, mode in enumerate(modes):
        bar_idx = mi + 1
        hatch, edge_c = MODE_BAR_STYLES[mi % len(MODE_BAR_STYLES)]
        est_bottoms = np.zeros(n_ticks)
        for seg in EST_STACK_ORDER:
            vals = np.array([get_est_segments(r, mode).get(seg, 0.0) if ok else 0.0
                              for r, ok in zip(rows, ok_mask)]) * multipliers * US_TO_MS
            color = EST_COLORS[seg]
            if ok_mask.any():
                ax.bar(
                    x[ok_mask] + offsets[bar_idx], vals[ok_mask], bar_w,
                    bottom=est_bottoms[ok_mask],
                    color=color, alpha=BAR_ALPHA,
                    edgecolor=edge_c, linewidth=EDGE_WIDTH,
                    hatch=hatch,
                )
                if show_values:
                    for xi, v, b in zip(x[ok_mask], vals[ok_mask], est_bottoms[ok_mask]):
                        if v > 0.06:
                            ax.text(
                                xi + offsets[bar_idx], b + v / 2, f"{v:.2f}",
                                ha="center", va="center", fontsize=6.5,
                                color="white", fontweight="bold",
                            )
            est_bottoms += vals
            add_patch(f"Est: {seg}", color)

        est_bottoms_by_mode[mode] = est_bottoms
        all_totals.extend(est_bottoms.tolist())

        mode_handles.append(mpatches.Patch(
            facecolor="#F5F5F5", edgecolor=edge_c, hatch=hatch,
            alpha=BAR_ALPHA, linewidth=EDGE_WIDTH, label=f"Mode: {mode}",
        ))

    # -------------------------------------------------- bar total labels --
    if show_values:
        totals_max = max(all_totals) if all_totals else 1.0
        total_offset = totals_max * 0.012 or 0.01
        for xi, ok, total in zip(x, ok_mask, gt_bottoms):
            if ok:
                ax.text(
                    xi + offsets[0], total + total_offset, f"{total:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333",
                )
        for mi, mode in enumerate(modes):
            bar_idx = mi + 1
            for xi, ok, total in zip(x, ok_mask, est_bottoms_by_mode[mode]):
                if ok:
                    ax.text(
                        xi + offsets[bar_idx], total + total_offset, f"{total:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color="#555555",
                        style="italic",
                    )

    # ---------------------------------------------------- shade non-ok cols
    shade_handle = None
    for xi, r, ok in zip(x, rows, ok_mask):
        if ok:
            continue
        ax.axvspan(xi - group_width / 2 - 0.05, xi + group_width / 2 + 0.05,
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
    max_total = max(all_totals) if all_totals else 1.0
    ax.set_ylim(0, max_total * 1.20)
    ax.set_xlim(-0.65, n_ticks - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([make_x_label(r, x_axis) for r in rows], fontsize=9)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    _x_axis_labels = {
        "batch_size":    "Batch Size",
        "recompute_len": "Recompute Length",
        "both":          "Batch Size / Recompute Length",
    }
    ax.set_xlabel(_x_axis_labels.get(x_axis, x_axis), fontsize=10)

    default_title = (
        "Ground Truth vs. Estimator Latency Breakdown (total, all iterations)"
        if normalize else
        "Ground Truth vs. Estimator Latency Breakdown"
    )
    if show_title:
        ax.set_title(title or default_title, fontsize=13, fontweight="bold", pad=14)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(False)

    gt_handles = [p for lbl, p in legend_patches.items() if lbl.startswith("GT:")]
    est_handles = [p for lbl, p in legend_patches.items() if lbl.startswith("Est:")]
    blank = mpatches.Patch(visible=False, label="")

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
        tp_x_shifted = tp_x + offsets[0]

        if normalize_throughput:
            tp_y = tp_y / tp_y.max()

        ax2.plot(
            tp_x_shifted, tp_y,
            color=THROUGHPUT_COLOR, linewidth=1.8,
            linestyle="--", alpha=0.9, zorder=5,
        )
        ax2.scatter(
            tp_x_shifted, tp_y,
            color=THROUGHPUT_COLOR, marker=THROUGHPUT_MARKER,
            s=55, zorder=6, alpha=0.95,
        )
        if show_values:
            fmt = "{:.2f}" if normalize_throughput else "{:.1f}"
            for xi_s, v in zip(tp_x_shifted, tp_y):
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
        ax2.spines["top"].set_visible(True)

        tp_handle = mlines.Line2D(
            [], [],
            color=THROUGHPUT_COLOR, linewidth=1.8, linestyle="--",
            marker=THROUGHPUT_MARKER, markersize=6,
            label="GT Throughput" + (" (normalized)" if normalize_throughput else " (tok/s)"),
        )

    # -------------------------------------------------------------- legend --
    if show_legend:
        handles = gt_handles + [blank] + est_handles + [blank] + mode_handles
        if shade_handle is not None:
            handles = handles + [blank, shade_handle]
        if tp_handle is not None:
            handles = handles + [blank, tp_handle]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.16 if ax2 is not None else 1.01, 1.0),
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

    analysis = analyze_estimator_choices(rows, modes, normalize=normalize)
    print_estimator_analysis(analysis)

    return fig, analysis


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot GT vs. estimator(s) grouped bar chart from a comparison CSV."
    )
    parser.add_argument("csv", help="Path to the comparison CSV")
    parser.add_argument(
        "--x-axis",
        choices=["batch_size", "recompute_len", "both"],
        default="batch_size",
        help="What to display as x-tick labels. Default: batch_size.",
    )
    parser.add_argument(
        "--modes", nargs="+", default=None, metavar="MODE",
        help="Subset of estimator mode names to include (default: all detected).",
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
             "number of CSV entries and estimator modes.",
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
            "sizes."
        ),
    )
    parser.add_argument(
        "--normalize-throughput", action="store_true",
        help="Scale the twin-axis throughput values so the max is 1.0 "
             "(axis capped at 1.1).",
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
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out = args.out or str(csv_path.with_name(csv_path.stem + "_vs_estimators.png"))
    plot_gt_vs_estimators(
        csv_path=args.csv,
        x_axis=args.x_axis,
        out_path=out,
        dpi=args.dpi,
        show=args.show,
        modes_to_plot=args.modes,
        title=args.title,
        show_legend=not args.no_legend,
        figsize=tuple(args.figsize) if args.figsize else None,
        show_non_ok=not args.hide_non_ok,
        normalize=args.normalize,
        normalize_throughput=args.normalize_throughput,
        show_values=not args.hide_values,
        show_title=not args.no_title,
    )


if __name__ == "__main__":
    main()