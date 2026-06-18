#!/usr/bin/env python3
"""
Second-pass analysis of the trace CSV produced by analyze_trace.py.

Supports both sep and nosep CSVs (auto-detected by column names, or use --nosep).

SEP mode (original behavior, unchanged):
  mha-gen-sum            : sum of all 8 mha-gen op durations
  mha-gen-compute-cuda-sum : sum of ALL mha-gen-compute-cuda-N durations
  mha-gen-recompute-cuda : sum of mha-gen-compute-cuda-N where origin == fwd_pre_mha
  mlp-sum                : sum of all 8 mlp op durations
  mlp-phase-1            : max(load-hidden-compute-cudamemcpy, pin-memory-1)
  mlp-phase-1-winner     : which operand was the max
  mlp-compute-cuda-sum   : sum of all mlp-compute-cuda-N durations
  mlp-phase-2            : max of A = mlp-compute-cuda-sum + pin-memory-2,
                                   B = load-cache-cudamemcpy-1+2,
                                   C = store-cache-cudamemcpy-1+2
  mlp-phase-2-winner     : "mlp-compute-cuda-sum+pin-memory-2",
                            "load-cache-cudamemcpy-1+2", or
                            "store-cache-cudamemcpy-1+2"

NOSEP mode (--nosep flag or auto-detected):
  sum-all                : sum of all 8 op durations
  compute-cuda-sum       : sum of all compute-cuda-N durations
  phase-1                : max(pin-memory-1, load-hidden-compute-cudamemcpy)
  phase-1-winner         : which operand was the max
  phase-2                : max of A = compute-cuda-sum + pin-memory-2,
                                   B = load-cache-cudamemcpy-1+2,
                                   C = store-cache-cudamemcpy-1+2
  phase-2-winner         : "pin-memory-2+compute-cuda-sum",
                            "load-cache-cudamemcpy-1+2", or
                            "store-cache-cudamemcpy-1+2"
  misc-cpu               : sum-all - phase-1 - phase-2

Usage:
    python trace_result_analyzer.py <input.csv> [--out output.csv] [--nosep | --batched]
"""

import csv
import argparse
import sys
from pathlib import Path


def safe_float(val):
    try:
        return float(val) if val not in (None, "", "None") else None
    except (ValueError, TypeError):
        return None


def fv(row, key):
    """Return float value for a CSV cell, defaulting to 0.0."""
    v = safe_float(row.get(key))
    return v if v is not None else 0.0


def sum_cols(row, col_names):
    total = 0.0
    for c in col_names:
        v = safe_float(row.get(c))
        if v is not None:
            total += v
    return round(total, 3)


def detect_nosep(all_cols):
    return "load_weight" in all_cols and "mha-gen_load_weight" not in all_cols


def detect_batched(all_cols):
    """
    Batched CSVs have 'group' and 'load_weight' (like nosep) but their
    compute-cuda-N-origin tags use 'fwd_pre_mha' (not 'fwd_pre' as in nosep).
    We detect by checking for at least one fwd_pre_mha origin column.
    """
    return any(c.endswith("-origin") and "compute-cuda-" in c for c in all_cols)


# ---------------------------------------------------------------------------
# SEP row analysis (original logic, unchanged)
# ---------------------------------------------------------------------------

def analyze_row_sep(row, all_cols):
    result = dict(row)

    MHA_OPS = [
        "mha-gen_load_weight", "mha-gen_load_hidden_compute",
        "mha-gen_load_cache", "mha-gen_load_hidden",
        "mha-gen_compute_layer", "mha-gen_store_cache",
        "mha-gen_store_hidden", "mha-gen_sync",
    ]
    result["mha-gen-sum"] = sum_cols(row, MHA_OPS)

    cuda_cols = sorted(
        [c for c in all_cols if c.startswith("mha-gen-compute-cuda-")
         and not c.endswith("-origin")],
        key=lambda c: int(c.split("-")[-1])
    )
    result["mha-gen-compute-cuda-sum"] = sum_cols(row, cuda_cols)

    recompute_total = 0.0
    for c in cuda_cols:
        origin_col = c + "-origin"
        if row.get(origin_col) == "fwd_pre_mha":
            v = safe_float(row.get(c))
            if v is not None:
                recompute_total += v
    result["mha-gen-recompute-cuda"] = round(recompute_total, 3)

    MLP_OPS = [
        "mlp_load_weight", "mlp_load_hidden_compute",
        "mlp_load_cache", "mlp_load_hidden",
        "mlp_compute_layer", "mlp_store_cache",
        "mlp_store_hidden", "mlp_sync",
    ]
    result["mlp-sum"] = sum_cols(row, MLP_OPS)

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

    mlp_cuda_cols = sorted(
        [c for c in all_cols if c.startswith("mlp-compute-cuda-")],
        key=lambda c: int(c.split("-")[-1])
    )
    mlp_cuda_sum = sum_cols(row, mlp_cuda_cols)
    result["mlp-compute-cuda-sum"] = mlp_cuda_sum

    pm2    = safe_float(row.get("pin-memory-2")) or 0.0
    lc_mc1 = safe_float(row.get("load-cache-cudamemcpy-1")) or 0.0
    lc_mc2 = safe_float(row.get("load-cache-cudamemcpy-2")) or 0.0
    sc_mc1 = safe_float(row.get("store-cache-cudamemcpy-1")) or 0.0
    sc_mc2 = safe_float(row.get("store-cache-cudamemcpy-2")) or 0.0

    A = round(mlp_cuda_sum + pm2, 3)
    B = round(lc_mc1 + lc_mc2, 3)
    C = round(sc_mc1 + sc_mc2, 3)

    result["mlp-phase-2-A"] = A
    result["mlp-phase-2-B"] = B
    result["mlp-phase-2-C"] = C

    max_val = max(A, B, C)
    result["mlp-phase-2"] = max_val
    if max_val == A:
        result["mlp-phase-2-winner"] = "mlp-compute-cuda-sum+pin-memory-2"
    elif max_val == B:
        result["mlp-phase-2-winner"] = "load-cache-cudamemcpy-1+2"
    else:
        result["mlp-phase-2-winner"] = "store-cache-cudamemcpy-1+2"

    return result


# ---------------------------------------------------------------------------
# NOSEP row analysis
# ---------------------------------------------------------------------------

def analyze_row_nosep(row, all_cols):
    result = dict(row)

    OPS_8 = [
        "load_weight", "load_hidden_compute", "load_cache", "load_hidden",
        "compute_layer", "store_cache", "store_hidden", "sync",
    ]
    total = sum_cols(row, OPS_8)
    result["sum-all"] = total

    cuda_cols = sorted(
        [c for c in all_cols if c.startswith("compute-cuda-")
         and not c.endswith("-origin")],
        key=lambda c: int(c.split("-")[-1])
    )
    compute_cuda_sum = sum_cols(row, cuda_cols)
    result["compute-cuda-sum"] = compute_cuda_sum

    # recompute-cuda: sum of ops whose origin is fwd_pre (mirrors mha-gen-recompute-cuda in sep)
    recompute_total = 0.0
    for c in cuda_cols:
        origin_col = c + "-origin"
        if row.get(origin_col) == "fwd_pre":
            v = safe_float(row.get(c))
            if v is not None:
                recompute_total += v
    result["recompute-cuda"] = round(recompute_total, 3)

    lhc_memcpy = safe_float(row.get("load-hidden-compute-cudamemcpy")) or 0.0
    pm1  = safe_float(row.get("pin-memory-1")) or 0.0
    pm2  = safe_float(row.get("pin-memory-2")) or 0.0
    lc1  = safe_float(row.get("load-cache-cudamemcpy-1")) or 0.0
    lc2  = safe_float(row.get("load-cache-cudamemcpy-2")) or 0.0
    sc1  = safe_float(row.get("store-cache-cudamemcpy-1")) or 0.0
    sc2  = safe_float(row.get("store-cache-cudamemcpy-2")) or 0.0

    if pm1 >= lhc_memcpy:
        result["phase-1"] = round(pm1, 3)
        result["phase-1-winner"] = "pin-memory-1"
    else:
        result["phase-1"] = round(lhc_memcpy, 3)
        result["phase-1-winner"] = "load-hidden-compute-cudamemcpy"

    A = round(compute_cuda_sum + pm2, 3)
    B = round(lc1 + lc2, 3)
    C = round(sc1 + sc2, 3)

    result["phase-2-A"] = A
    result["phase-2-B"] = B
    result["phase-2-C"] = C

    max_val = max(A, B, C)
    result["phase-2"] = max_val
    if max_val == A:
        result["phase-2-winner"] = "pin-memory-2+compute-cuda-sum"
    elif max_val == B:
        result["phase-2-winner"] = "load-cache-cudamemcpy-1+2"
    else:
        result["phase-2-winner"] = "store-cache-cudamemcpy-1+2"

    result["misc-cpu"] = round(total - result["phase-1"] - result["phase-2"], 3)

    return result


# ---------------------------------------------------------------------------
# BATCHED row analysis
# ---------------------------------------------------------------------------

def analyze_row_batched(row, all_cols):
    """
    Batched mode analysis. Computes:

    sum-all:
      Sum of all 8 op durations.

    recompute-cuda:
      Sum of compute-cuda-N ops whose origin is 'fwd_pre_mha'.

    mha-gen-cuda:
      Sum of compute-cuda-N ops whose origin is 'mha_gen'.

    Critical path = max(path1, path2, path3):

      path1 = max(subpath1, subpath2, subpath3) + load-cache-cudamemcpy-2
        subpath1 = pin-memory-1 + pin-memory-2
        subpath2 = pin-memory-1 + load-cache-cudamemcpy-1
        subpath3 = load-hidden-compute-cudamemcpy + load-cache-cudamemcpy-1

      path2 = load-hidden-compute-cudamemcpy + recompute-cuda + mha-gen-cuda

      path3 = pin-memory-1 + pin-memory-2
              + store-cache-cudamemcpy-1 + store-cache-cudamemcpy-2
    """
    result = dict(row)

    OPS_8 = [
        "load_weight", "load_hidden_compute", "load_cache", "load_hidden",
        "compute_layer", "store_cache", "store_hidden", "sync",
    ]
    result["sum-all"] = sum_cols(row, OPS_8)

    # Split compute-cuda ops by origin
    cuda_cols = sorted(
        [c for c in all_cols if c.startswith("compute-cuda-")
         and not c.endswith("-origin")],
        key=lambda c: int(c.split("-")[-1])
    )
    recompute_total = 0.0
    mha_gen_total   = 0.0
    for c in cuda_cols:
        v = safe_float(row.get(c))
        if v is None:
            continue
        origin = row.get(c + "-origin", "")
        if origin == "fwd_pre_mha":
            recompute_total += v
        elif origin == "mha_gen":
            mha_gen_total += v
    result["recompute-cuda"] = round(recompute_total, 3)
    result["mha-gen-cuda"]   = round(mha_gen_total, 3)

    lhc = fv(row, "load-hidden-compute-cudamemcpy")
    pm1 = fv(row, "pin-memory-1")
    pm2 = fv(row, "pin-memory-2")
    lc1 = fv(row, "load-cache-cudamemcpy-1")
    lc2 = fv(row, "load-cache-cudamemcpy-2")
    sc1 = fv(row, "store-cache-cudamemcpy-1")
    sc2 = fv(row, "store-cache-cudamemcpy-2")

    # --- path1 ---
    sp1 = round(pm1 + pm2, 3)
    sp2 = round(pm1 + lc1, 3)
    sp3 = round(lhc + lc1, 3)
    result["path1-subpath1"] = sp1
    result["path1-subpath2"] = sp2
    result["path1-subpath3"] = sp3

    path1_inner     = max(sp1, sp2, sp3)
    path1_inner_idx = [sp1, sp2, sp3].index(path1_inner) + 1
    path1           = round(path1_inner + lc2, 3)
    result["path1-inner"]        = path1_inner
    result["path1-inner-winner"] = f"subpath{path1_inner_idx}"
    result["path1"]              = path1

    # --- path2 ---
    path2 = round(lhc + recompute_total + mha_gen_total, 3)
    result["path2"] = path2

    # --- path3 ---
    path3 = round(pm1 + pm2 + sc1 + sc2, 3)
    result["path3"] = path3

    # --- critical path ---
    crit = max(path1, path2, path3)
    crit_idx = [path1, path2, path3].index(crit) + 1
    result["critical-path"]        = crit
    result["critical-path-winner"] = f"path{crit_idx}"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_csv(input_path, output_path=None, nosep=None, batched=False):
    input_path = Path(input_path)

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        all_cols = reader.fieldnames or []
        rows = list(reader)

    print(f"Loaded {len(rows)} rows, {len(all_cols)} columns.", file=sys.stderr)

    # Mode resolution: explicit flags take priority; otherwise auto-detect
    if batched:
        mode = "batched"
    elif nosep is None:
        if detect_nosep(all_cols) and detect_batched(all_cols):
            # Both look like nosep structure — check origin tag values to distinguish
            # batched uses 'fwd_pre_mha'; nosep uses 'fwd_pre'
            origin_vals = {row.get(c, "") for row in rows[:5]
                           for c in all_cols if c.endswith("-origin")}
            mode = "batched" if "fwd_pre_mha" in origin_vals else "nosep"
        elif detect_nosep(all_cols):
            mode = "nosep"
        else:
            mode = "sep"
    else:
        mode = "nosep" if nosep else "sep"

    print(f"Mode: {mode}", file=sys.stderr)

    if mode == "batched":
        results = [analyze_row_batched(row, all_cols) for row in rows]
        derived_cols = [
            "sum-all",
            "recompute-cuda", "mha-gen-cuda",
            "path1-subpath1", "path1-subpath2", "path1-subpath3",
            "path1-inner", "path1-inner-winner", "path1",
            "path2",
            "path3",
            "critical-path", "critical-path-winner",
        ]
    elif mode == "nosep":
        results = [analyze_row_nosep(row, all_cols) for row in rows]
        derived_cols = [
            "sum-all", "compute-cuda-sum",
            "phase-1", "phase-1-winner",
            "phase-2-A", "phase-2-B", "phase-2-C",
            "phase-2", "phase-2-winner",
            "misc-cpu",
            "recompute-cuda",
        ]
    else:
        results = [analyze_row_sep(row, all_cols) for row in rows]
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
        description="Summarize trace CSV with derived latency phase metrics."
    )
    parser.add_argument("input", help="Input CSV from trace_analyzer.py.")
    parser.add_argument("--out", default=None, metavar="OUTPUT.csv")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--nosep", action="store_true",
        help="Force nosep mode (auto-detected if omitted).",
    )
    mode_group.add_argument(
        "--batched", action="store_true",
        help="Use batched mode: critical-path analysis for consecutive mha_gen groups.",
    )
    args = parser.parse_args()
    analyze_csv(args.input, output_path=args.out,
                nosep=args.nosep if args.nosep else None,
                batched=args.batched)


if __name__ == "__main__":
    main()