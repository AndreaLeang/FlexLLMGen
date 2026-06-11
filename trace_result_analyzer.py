#!/usr/bin/env python3
"""
Second-pass analysis of the supergroup CSV produced by analyze_trace.py.

For each supergroup row, computes:

mha-gen:
  mha-gen-sum            : sum of all 8 mha-gen op durations
  mha-gen-compute-cuda-sum : sum of ALL mha-gen-compute-cuda-N durations
  mha-gen-recompute-cuda : sum of mha-gen-compute-cuda-N where origin == fwd_pre_mha

mlp:
  mlp-sum                : sum of all 8 mlp op durations
  mlp-phase-1            : max(load-hidden-compute-cudamemcpy, pin-memory-1)
  mlp-phase-1-winner     : which operand was the max
  mlp-compute-cuda-sum   : sum of all mlp-compute-cuda-N durations  (renamed from mla)
  mlp-phase-2            : max of three groups:
                             A = mlp-compute-cuda-sum + pin-memory-2
                             B = load-cache-cudamemcpy-1 + load-cache-cudamemcpy-2
                             C = store-cache-cudamemcpy-1 + store-cache-cudamemcpy-2
  mlp-phase-2-winner     : which group (A/B/C) was the max

Usage:
    python analyze_csv.py <input.csv> [--out output.csv]
"""

import csv
import argparse
import sys
from pathlib import Path


def safe_float(val):
    """Return float or None for empty/missing values."""
    try:
        return float(val) if val not in (None, "", "None") else None
    except (ValueError, TypeError):
        return None


def sum_cols(row, col_names):
    """Sum the values of named columns, treating missing/empty as 0."""
    total = 0.0
    for c in col_names:
        v = safe_float(row.get(c))
        if v is not None:
            total += v
    return round(total, 3)


def analyze_row(row, all_cols):
    result = dict(row)  # carry through all original columns

    # ------------------------------------------------------------------ #
    # mha-gen
    # ------------------------------------------------------------------ #
    MHA_OPS = [
        "mha-gen_load_weight", "mha-gen_load_hidden_compute",
        "mha-gen_load_cache", "mha-gen_load_hidden",
        "mha-gen_compute_layer", "mha-gen_store_cache",
        "mha-gen_store_hidden", "mha-gen_sync",
    ]
    result["mha-gen-sum"] = sum_cols(row, MHA_OPS)

    # Collect all mha-gen-compute-cuda-N columns (not origin columns)
    cuda_cols = sorted(
        [c for c in all_cols if c.startswith("mha-gen-compute-cuda-")
         and not c.endswith("-origin")],
        key=lambda c: int(c.split("-")[-1])
    )
    result["mha-gen-compute-cuda-sum"] = sum_cols(row, cuda_cols)

    # Sum only fwd_pre_mha ops
    recompute_total = 0.0
    for c in cuda_cols:
        origin_col = c + "-origin"
        if row.get(origin_col) == "fwd_pre_mha":
            v = safe_float(row.get(c))
            if v is not None:
                recompute_total += v
    result["mha-gen-recompute-cuda"] = round(recompute_total, 3)

    # ------------------------------------------------------------------ #
    # mlp
    # ------------------------------------------------------------------ #
    MLP_OPS = [
        "mlp_load_weight", "mlp_load_hidden_compute",
        "mlp_load_cache", "mlp_load_hidden",
        "mlp_compute_layer", "mlp_store_cache",
        "mlp_store_hidden", "mlp_sync",
    ]
    result["mlp-sum"] = sum_cols(row, MLP_OPS)

    # Phase 1: max(load-hidden-compute-cudamemcpy, pin-memory-1)
    lhc_memcpy = safe_float(row.get("load-hidden-compute-cudamemcpy"))
    pm1        = safe_float(row.get("pin-memory-1"))
    if lhc_memcpy is not None and pm1 is not None:
        if lhc_memcpy >= pm1:
            result["mlp-phase-1"] = round(lhc_memcpy, 3)
            result["mlp-phase-1-winner"] = "load-hidden-compute-cudamemcpy"
        else:
            result["mlp-phase-1"] = round(pm1, 3)
            result["mlp-phase-1-winner"] = "pin-memory-1"
    elif lhc_memcpy is not None:
        result["mlp-phase-1"] = round(lhc_memcpy, 3)
        result["mlp-phase-1-winner"] = "load-hidden-compute-cudamemcpy"
    elif pm1 is not None:
        result["mlp-phase-1"] = round(pm1, 3)
        result["mlp-phase-1-winner"] = "pin-memory-1"
    else:
        result["mlp-phase-1"] = None
        result["mlp-phase-1-winner"] = None

    # mlp compute cuda sum
    mlp_cuda_cols = sorted(
        [c for c in all_cols if c.startswith("mlp-compute-cuda-")],
        key=lambda c: int(c.split("-")[-1])
    )
    mlp_cuda_sum = sum_cols(row, mlp_cuda_cols)
    result["mlp-compute-cuda-sum"] = mlp_cuda_sum

    # Phase 2: max of A, B, C
    pm2    = safe_float(row.get("pin-memory-2")) or 0.0
    lc_mc1 = safe_float(row.get("load-cache-cudamemcpy-1")) or 0.0
    lc_mc2 = safe_float(row.get("load-cache-cudamemcpy-2")) or 0.0
    sc_mc1 = safe_float(row.get("store-cache-cudamemcpy-1")) or 0.0
    sc_mc2 = safe_float(row.get("store-cache-cudamemcpy-2")) or 0.0

    A = round(mlp_cuda_sum + pm2, 3)
    B = round(lc_mc1 + lc_mc2, 3)
    C = round(sc_mc1 + sc_mc2, 3)

    result["mlp-phase-2-A"] = A  # mlp-compute-cuda-sum + pin-memory-2
    result["mlp-phase-2-B"] = B  # load-cache-cudamemcpy-1 + load-cache-cudamemcpy-2
    result["mlp-phase-2-C"] = C  # store-cache-cudamemcpy-1 + store-cache-cudamemcpy-2

    max_val = max(A, B, C)
    result["mlp-phase-2"] = max_val
    if max_val == A:
        result["mlp-phase-2-winner"] = "mlp-compute-cuda-sum+pin-memory-2"
    elif max_val == B:
        result["mlp-phase-2-winner"] = "load-cache-cudamemcpy-1+2"
    else:
        result["mlp-phase-2-winner"] = "store-cache-cudamemcpy-1+2"

    return result


def analyze_csv(input_path, output_path=None):
    input_path = Path(input_path)

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        all_cols = reader.fieldnames or []
        rows = list(reader)

    print(f"Loaded {len(rows)} rows, {len(all_cols)} columns.", file=sys.stderr)

    results = [analyze_row(row, all_cols) for row in rows]

    # Build output column order: original cols first, then derived cols appended
    derived_cols = [
        "mha-gen-sum", "mha-gen-compute-cuda-sum", "mha-gen-recompute-cuda",
        "mlp-sum",
        "mlp-phase-1", "mlp-phase-1-winner",
        "mlp-compute-cuda-sum",
        "mlp-phase-2-A", "mlp-phase-2-B", "mlp-phase-2-C",
        "mlp-phase-2", "mlp-phase-2-winner",
    ]
    out_cols = list(all_cols) + [c for c in derived_cols if c not in all_cols]

    if output_path is None:
        output_path = input_path.stem + "_summary.csv"
    output_path = Path(output_path)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({col: row.get(col, "") for col in out_cols})

    print(f"Written to {output_path}", file=sys.stderr)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Summarize supergroup CSV with derived latency metrics."
    )
    parser.add_argument("input", help="Input CSV from analyze_trace.py.")
    parser.add_argument("--out", default=None, metavar="OUTPUT.csv")
    args = parser.parse_args()
    analyze_csv(args.input, output_path=args.out)


if __name__ == "__main__":
    main()