"""
plot_xfer_bandwidth.py

Reads the xfer CSV produced by:
    cpu_delegation_microbench.py --bench xfer --out results.csv
(i.e. results_xfer.csv) and plots payload size vs. achieved bandwidth for:

  - pin_memory(), contiguous source
  - pin_memory(), non-contiguous source
  - PCIe H2D transfer (copy_(..., non_blocking=True)), contiguous source
  - PCIe H2D transfer (copy_(..., non_blocking=True)), non-contiguous source

pin_memory() always materializes a new *contiguous* pinned buffer regardless
of the source's layout, so the two PCIe transfer curves are expected to
roughly track each other once pinned; the pin_memory curves are where the
non-contiguous penalty should actually show up. Both are plotted so you can
see whether that's true, rather than assuming it -- pass --no-pin-split or
--no-copy-split to drop a pair down to a single combined line if you'd
rather compare just 3 curves.

If the CSV has multiple rows for the same payload size (e.g. the sweep was
re-run and appended), each point plots the median across runs with a shaded
min-max band showing run-to-run variance -- itself a useful signal for
host-level jitter.

Usage:
    python plot_xfer_bandwidth.py --csv results_xfer.csv
    python plot_xfer_bandwidth.py --csv results_xfer.csv --out bw.png --log-y
"""

import argparse
import csv
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required = {"actual_mb", "layout", "pin_memory_gbps", "copy_gbps"}
    missing = required - set(fieldnames)
    if missing:
        raise SystemExit(
            f"{csv_path} is missing columns {sorted(missing)} -- this script expects "
            f"the xfer CSV produced by cpu_delegation_microbench.py --bench xfer --out ..."
        )
    if not rows:
        raise SystemExit(f"{csv_path} has a header but no data rows.")
    return rows


def group_series(rows, layout, value_key):
    """Group rows by payload size for one (layout, metric) series, returning
    sorted (size_mb, median, lo, hi) tuples. lo/hi are the min/max across
    repeated runs at that size (equal to median when there's only one)."""
    buckets = defaultdict(list)
    for r in rows:
        if r["layout"] != layout:
            continue
        size = round(float(r["actual_mb"]), 6)
        buckets[size].append(float(r[value_key]))

    return [(size, statistics.median(vals), min(vals), max(vals))
            for size, vals in sorted(buckets.items())]


def plot_series(ax, points, label, **kwargs):
    if not points:
        return
    sizes = [p[0] for p in points]
    meds = [p[1] for p in points]
    los = [p[2] for p in points]
    his = [p[3] for p in points]
    line, = ax.plot(sizes, meds, marker="o", markersize=4, label=label, **kwargs)
    if any(hi > lo for lo, hi in zip(los, his)):
        ax.fill_between(sizes, los, his, alpha=0.15, color=line.get_color())


def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results_xfer.csv",
                    help="xfer CSV from cpu_delegation_microbench.py --bench xfer --out ...")
    p.add_argument("--out", default="xfer_bandwidth.png", help="output image path")
    p.add_argument("--log-y", action="store_true",
                    help="use a log-scale y-axis (useful if pin_memory and PCIe "
                         "bandwidths differ by orders of magnitude)")
    p.add_argument("--no-pin-split", action="store_true",
                    help="combine contiguous/non-contiguous pin_memory into a single "
                         "line (median of both) instead of two")
    p.add_argument("--no-copy-split", action="store_true",
                    help="combine contiguous/non-contiguous PCIe transfer into a single "
                         "line (median of both) instead of two")
    p.add_argument("--title", default="CPU->GPU transfer bandwidth vs. payload size")
    return p


def main():
    args = build_argparser().parse_args()
    rows = load_rows(args.csv)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    if args.no_pin_split:
        plot_series(ax, group_series(rows, "contiguous", "pin_memory_gbps") +
                    group_series(rows, "non_contiguous", "pin_memory_gbps"),
                    "pin_memory", linestyle="--", color="tab:green")
    else:
        plot_series(ax, group_series(rows, "contiguous", "pin_memory_gbps"),
                    "pin_memory (contiguous)", linestyle="--", color="tab:blue")
        plot_series(ax, group_series(rows, "non_contiguous", "pin_memory_gbps"),
                    "pin_memory (non-contiguous)", linestyle="--", color="tab:orange")

    if args.no_copy_split:
        plot_series(ax, group_series(rows, "contiguous", "copy_gbps") +
                    group_series(rows, "non_contiguous", "copy_gbps"),
                    "PCIe H2D transfer", linestyle="-", color="tab:green")
    else:
        plot_series(ax, group_series(rows, "contiguous", "copy_gbps"),
                    "PCIe H2D transfer (contiguous)", linestyle="-", color="tab:blue")
        plot_series(ax, group_series(rows, "non_contiguous", "copy_gbps"),
                    "PCIe H2D transfer (non-contiguous)", linestyle="-", color="tab:orange")

    # ax.set_xscale("log", base=2)
    if args.log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Payload size (MB, log scale)")
    ax.set_ylabel("Bandwidth (GB/s)" + (", log scale" if args.log_y else ""))
    ax.set_title(args.title)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()