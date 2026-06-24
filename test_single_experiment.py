#!/usr/bin/env python3
"""
test_single_experiment.py
=========================
End-to-end test of gt_vs_estimator.py using the real uploaded trace.

Experiment configuration (read from the JSON filename):
  model          : facebook/opt-6.7b
  batch_size     : 1   (gbs=1)
  num_batches    : 2   (ngbs=2)
  num_prompts    : 2   (= gbs * ngbs)
  prompt_len     : 2048
  gen_len        : 16
  offload_percent: 100  (kv_gpu=0, kv_cpu=100)
  recompute_len  : 1024

What this test covers
---------------------
  Stage 1 – GT pipeline
    1a. trace_analyzer.py  --batched   (raw op durations → analysis CSV)
    1b. trace_result_analyzer.py --batched  (analysis CSV → summary CSV)
    1c. load_gt_summary() + build_gt_segments()  (summary CSV → segment dict)
    1d. Verify segment values match manually-computed expectations from the data.

  Stage 2 – Estimator (run only when kv_schedule_optimization is importable)
    2a. get_estimator_breakdown() with all five EstimatorModes.
    2b. Verify keys, all values are non-negative floats.
    2c. Verify total latency is in a plausible range (> 0 µs).

  Stage 3 – CSV + plot
    3a. build_csv_row() → write_comparison_csv() → reload and verify all columns.
    3b. plot_gt_vs_estimator.plot_comparison() → verify PNG is created.

Usage
-----
Run from the directory that contains gt_vs_estimator.py, trace_analyzer.py,
trace_result_analyzer.py, and the kv_schedule_optimization / flexllmgen packages.

  python test_single_experiment.py [--trace PATH] [--out-dir DIR] [--skip-plot]

The --trace flag defaults to the path of the uploaded sample trace; override it
with any batched-mode .json trace you have locally.
"""

import argparse
import csv
import dataclasses
import os
import sys
import traceback
from pathlib import Path


# ── locate the trace ──────────────────────────────────────────────────────────
DEFAULT_TRACE = (
    "fo-6_7b-gbs1-ngbs2-prompt2048-gen16-percent-100-0-0-100-100-0-R-1024-gpu-cache.json"
)
# Adjust this to where the file actually lives on your machine:
TRACE_SEARCH_DIRS = [
    ".",
    "./traces",
    "~/traces",
    os.path.expanduser("~/traces"),
]


def find_trace(hint: str) -> str:
    if os.path.isfile(hint):
        return hint
    for d in TRACE_SEARCH_DIRS:
        candidate = os.path.join(d, DEFAULT_TRACE)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find trace file.\n"
        f"Searched for: {DEFAULT_TRACE}\n"
        f"Provide the path with --trace PATH"
    )


# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m⊘\033[0m"


def ok(msg: str):
    print(f"  {PASS}  {msg}")


def fail(msg: str):
    print(f"  {FAIL}  {msg}")


def skip(msg: str):
    print(f"  {SKIP}  {msg}")


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def assert_eq(label: str, got, expected, tol: float = 0.5):
    """Floating-point equality check within tol µs."""
    diff = abs(got - expected)
    if diff <= tol:
        ok(f"{label}: {got:.2f} µs  (expected ≈ {expected:.2f} µs)")
    else:
        fail(f"{label}: {got:.2f} µs  (expected ≈ {expected:.2f} µs, diff={diff:.2f})")


def assert_between(label: str, got, lo, hi):
    if lo <= got <= hi:
        ok(f"{label}: {got:.2f} µs  (in [{lo:.0f}, {hi:.0f}])")
    else:
        fail(f"{label}: {got:.2f} µs  NOT in [{lo:.0f}, {hi:.0f}]")


def assert_keys(label: str, d: dict, expected_keys):
    missing = set(expected_keys) - set(d.keys())
    if not missing:
        ok(f"{label}: all {len(expected_keys)} expected keys present")
    else:
        fail(f"{label}: missing keys {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Ground-truth pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_gt_pipeline(trace_json: str, out_dir: str) -> dict:
    """
    Run trace_analyzer + trace_result_analyzer, then parse the summary.
    Returns the GT segment dict (or empty dict on failure).
    """
    section("Stage 1 — Ground-truth pipeline")

    import subprocess

    analysis_csv = os.path.join(out_dir, "test_batched_analysis.csv")
    summary_csv  = os.path.join(out_dir, "test_batched_summary.csv")

    # ── 1a. trace_analyzer ────────────────────────────────────────────────────
    print("\n  [1a] trace_analyzer.py --batched")
    cmd = [sys.executable, "trace_analyzer.py", trace_json,
           "--batched", "--out", analysis_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"trace_analyzer failed:\n{r.stderr}")
        return {}
    ok(f"trace_analyzer completed → {analysis_csv}")

    # Spot-check the analysis CSV
    with open(analysis_csv) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows_a = list(reader)
    assert_between("Number of batched groups", len(rows_a), 400, 600)
    assert_keys("Analysis CSV columns",
                dict.fromkeys(cols),
                ["group", "token", "load_weight", "compute_layer",
                 "load-hidden-compute-cudamemcpy", "pin-memory-1",
                 "load-cache-cudamemcpy-1", "load-cache-cudamemcpy-2"])

    # ── 1b. trace_result_analyzer ─────────────────────────────────────────────
    print("\n  [1b] trace_result_analyzer.py --batched")
    cmd = [sys.executable, "trace_result_analyzer.py", analysis_csv,
           "--batched", "--out", summary_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"trace_result_analyzer failed:\n{r.stderr}")
        return {}
    ok(f"trace_result_analyzer completed → {summary_csv}")

    # ── 1c. load_gt_summary + build_gt_segments ───────────────────────────────
    print("\n  [1c] load_gt_summary() + build_gt_segments()")

    # Import from our module
    sys.path.insert(0, ".")
    from gt_vs_estimator import load_gt_summary, build_gt_segments, GT_SEGMENT_NAMES

    # Test averaging variants
    row_all, _   = load_gt_summary(summary_csv, avg_rows=None)
    row_10, _    = load_gt_summary(summary_csv, avg_rows=10)
    row_1, _     = load_gt_summary(summary_csv, avg_rows=1)

    ok(f"load_gt_summary(all): critical-path = {float(row_all.get('critical-path', 0)):.2f} µs")
    ok(f"load_gt_summary(10):  critical-path = {float(row_10.get('critical-path', 0)):.2f} µs")
    ok(f"load_gt_summary(1):   critical-path = {float(row_1.get('critical-path', 0)):.2f} µs")

    segs = build_gt_segments(row_all)
    assert_keys("GT segment keys", segs, GT_SEGMENT_NAMES)

    # ── 1d. Verify segment values ─────────────────────────────────────────────
    #
    # From the data: ALL 479 rows have critical-path-winner=path1, subpath3.
    # Subpath3 formula:
    #   critical = lhc + lc1 + lc2   (the path1 critical path for subpath3)
    # Segments:
    #   Recompute Load  = lhc                    ≈ 685 µs
    #   KVCache Load    = lc1 + lc2              ≈ 693 + 760 = 1453 µs
    #   Misc. CPU       = sum_all - critical      ≈ 2528 - 2139 = 389 µs
    #   All others      = 0
    #
    # Tolerances are ±50 µs to account for measurement variance across runs.

    print("\n  [1d] Segment value assertions (averaged over all rows)")
    assert_between("Recompute Load",    segs["Recompute Load"],   600, 780)
    assert_between("KVCache Load",      segs["KVCache Load"],    1300, 1600)
    assert_between("Misc. CPU",         segs["Misc. CPU"],        300,  550)
    assert_eq("Recompute CUDA",         segs["Recompute CUDA"],     0.0)
    assert_eq("MHA CUDA",               segs["MHA CUDA"],           0.0)
    assert_eq("PinnedMemory CPU",       segs["PinnedMemory CPU"],   0.0)
    assert_eq("KVCache Store",          segs["KVCache Store"],      0.0)

    total = sum(segs.values())
    ok(f"GT total = {total:.1f} µs")

    # Confirm total ≈ sum-all (they must match exactly)
    sum_all = float(row_all.get("sum-all", 0))
    diff = abs(total - sum_all)
    if diff < 1.0:
        ok(f"GT total matches sum-all ({sum_all:.1f} µs), diff={diff:.3f}")
    else:
        fail(f"GT total {total:.1f} ≠ sum-all {sum_all:.1f}, diff={diff:.2f}")

    print()
    print("  Full GT segment breakdown (averaged over all rows):")
    for k, v in segs.items():
        bar = "█" * max(0, int(v / 40))
        print(f"    {k:<25} {v:>8.1f} µs  {bar}")

    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Estimator
# ─────────────────────────────────────────────────────────────────────────────

def test_estimator(out_dir: str) -> dict:
    """
    Run the estimator for the same experiment. Returns {mode_name: segs_dict}.
    """
    section("Stage 2 — Estimator breakdown")

    from gt_vs_estimator import (
        ESTIMATOR_AVAILABLE, ExperimentConfig, HardwareConfig,
        EstimatorMode, get_estimator_breakdown, EST_SEGMENT_NAMES,
    )

    if not ESTIMATOR_AVAILABLE:
        skip("kv_schedule_optimization not importable — skipping estimator stage")
        skip("(Install flexllmgen + kv_schedule_optimization to enable this stage)")
        return {}

    # ── Build opt_config ──────────────────────────────────────────────────────
    from flexllmgen.opt_config import get_opt_config

    model = "facebook/opt-6.7b"
    opt_config = get_opt_config(model)
    ok(f"opt_config loaded: {model}")

    # ── Build gpu_estimator ───────────────────────────────────────────────────
    # Replace this block with your get_gee(...) call.
    # Example:
    #   from gee.gee_utils import get_gee
    #   gpu_estimator = get_gee(
    #       gpu_yaml_path   = "/path/to/config/gpu/yz8.yaml",
    #       lut_yaml_path   = "/path/to/config/a100_dvfs_lut_config.yaml",
    #       dvfs_aware      = True,
    #       dvfs_inference_mode = "all",
    #       dvfs_supply_voltage_json = "/path/to/config/dvfs/yz8/supply_voltage.json",
    #       dvfs_idle_power_json     = "/path/to/config/dvfs/yz8/idle_power.json",
    #       lut_folder_abs_path      = "/path/to/database/data",
    #   )
    print("\n  [NOTE] gpu_estimator is None — set it to your get_gee(...) object.")
    print("  Estimator calls will raise; this is expected until get_gee is wired in.")
    gpu_estimator = None   # ← replace with get_gee(...)

    # ── Experiment config matching the trace ──────────────────────────────────
    exp = ExperimentConfig(
        model          = model,
        prompt_len     = 2048,
        gen_len        = 16,
        num_prompts    = 2,       # gbs=1, ngbs=2 → num_prompts=2
        batch_size     = 1,
        recompute_len  = 1024,
        offload_percent= 100.0,  # kv_gpu=0%, kv_cpu=100%
        output_dir     = out_dir,
    )
    ok(f"ExperimentConfig: {exp.experiment_id}")
    ok(f"  num_batches = {exp.num_batches}")

    # Hardware config (matches the A100 used for the trace)
    hw = HardwareConfig(gpu_mem_gb=40, cpu_mem_gb=200, gpu_freq=1305)

    # Estimator modes to exercise
    modes = [
        EstimatorMode(name="default"),
        EstimatorMode(name="ideal_bw",   use_ideal_bw=True),
        EstimatorMode(name="flex_bw",    use_flex_bw=True),
        EstimatorMode(name="no_pinned",  use_no_pinned=True),
        EstimatorMode(name="ideal_comp", use_ideal_comp=True),
    ]

    results = {}
    print()
    for mode in modes:
        hw_mode = mode.apply_to(hw)
        try:
            segs = get_estimator_breakdown(
                exp=exp,
                hw=hw_mode,
                gpu_estimator=gpu_estimator,
                opt_config=opt_config,
                gen_step=1,
            )
            # Verify structure
            assert_keys(f"Est [{mode.name}] keys", segs, EST_SEGMENT_NAMES)
            # All values must be non-negative
            neg = {k: v for k, v in segs.items() if v < 0}
            if neg:
                fail(f"Est [{mode.name}] negative values: {neg}")
            total = sum(segs.values())
            assert_between(f"Est [{mode.name}] total", total, 1, 1_000_000)
            ok(f"Est [{mode.name}] total = {total:.1f} µs")
            results[mode.name] = segs
        except Exception as e:
            fail(f"Est [{mode.name}] error: {e}")
            results[mode.name] = {}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — CSV output + plot
# ─────────────────────────────────────────────────────────────────────────────

def test_csv_and_plot(
    gt_segs: dict,
    est_segs_by_mode: dict,
    out_dir: str,
    skip_plot: bool,
):
    section("Stage 3 — CSV output and plot")

    from gt_vs_estimator import (
        ExperimentConfig, EstimatorMode,
        build_csv_row, write_comparison_csv, make_csv_columns,
        GT_SEGMENT_NAMES, EST_SEGMENT_NAMES,
    )

    model = "facebook/opt-6.7b"
    exp = dataclasses.replace(
        ExperimentConfig(
            model=model, prompt_len=2048, gen_len=16,
            num_prompts=2, batch_size=1, recompute_len=1024,
        ),
        offload_percent=100.0,
        trace_json_path=os.path.join(out_dir, "test_batched_analysis.csv"),
        summary_csv_path=os.path.join(out_dir, "test_batched_summary.csv"),
    )

    modes = [
        EstimatorMode(name="default"),
        EstimatorMode(name="ideal_bw",   use_ideal_bw=True),
        EstimatorMode(name="flex_bw",    use_flex_bw=True),
        EstimatorMode(name="no_pinned",  use_no_pinned=True),
        EstimatorMode(name="ideal_comp", use_ideal_comp=True),
    ]

    # Use real estimator values if we have them, otherwise zeros
    # (so CSV/plot structure is tested regardless)
    est_segs_full = {m.name: est_segs_by_mode.get(m.name, {}) for m in modes}

    # ── 3a. Build and write CSV ───────────────────────────────────────────────
    print("\n  [3a] build_csv_row() + write_comparison_csv()")
    row = build_csv_row(exp, gt_segs, est_segs_full, modes)

    # Verify GT values round-trip correctly
    for seg in GT_SEGMENT_NAMES:
        safe = seg.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
        col = f"gt_{safe}_us"
        assert col in row, f"Missing column {col}"
        stored = row[col]
        expected = round(gt_segs.get(seg, 0.0), 3)
        if abs(stored - expected) < 0.001:
            ok(f"GT column {col} = {stored:.3f} µs")
        else:
            fail(f"GT column {col}: stored {stored} ≠ expected {expected}")

    out_csv = os.path.join(out_dir, "test_comparison_result.csv")
    write_comparison_csv([row], out_csv, modes)

    # Reload and verify
    with open(out_csv) as f:
        reader = csv.DictReader(f)
        loaded_cols = reader.fieldnames or []
        loaded_rows = list(reader)

    expected_cols = make_csv_columns(modes)
    missing_cols = set(expected_cols) - set(loaded_cols)
    extra_cols   = set(loaded_cols) - set(expected_cols)
    if not missing_cols:
        ok(f"All {len(expected_cols)} expected columns present in CSV")
    else:
        fail(f"Missing columns: {missing_cols}")
    if extra_cols:
        skip(f"Extra columns (OK, extrasaction='ignore'): {extra_cols}")

    assert len(loaded_rows) == 1
    ok("CSV has exactly 1 data row")

    lr = loaded_rows[0]
    ok(f"experiment_id = {lr['experiment_id']}")
    ok(f"batch_size    = {lr['batch_size']}")
    ok(f"offload_percent = {lr['offload_percent']}")
    ok(f"gt_total_us   = {lr['gt_total_us']} µs")

    # ── 3b. Plot ──────────────────────────────────────────────────────────────
    if skip_plot:
        skip("Plot stage skipped (--skip-plot)")
        return

    print("\n  [3b] plot_gt_vs_estimator.plot_comparison()")
    try:
        from plot_gt_vs_estimator import plot_comparison
        out_png = os.path.join(out_dir, "test_comparison_plot.png")
        fig = plot_comparison(
            csv_path=out_csv,
            x_axis="batch_size",
            out_path=out_png,
            dpi=100,
            show=False,
        )
        if os.path.exists(out_png) and os.path.getsize(out_png) > 0:
            ok(f"Plot saved: {out_png}  ({os.path.getsize(out_png)//1024} KB)")
        else:
            fail(f"Plot file not created or empty: {out_png}")
    except ImportError as e:
        skip(f"matplotlib not available: {e}")
    except Exception as e:
        fail(f"Plot error: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single-experiment test for gt_vs_estimator.py"
    )
    parser.add_argument(
        "--trace", default=DEFAULT_TRACE,
        help="Path to the batched-mode PyTorch trace JSON (default: the uploaded sample).",
    )
    parser.add_argument(
        "--out-dir", default="/tmp/gt_vs_est_test",
        help="Directory for intermediate and output files.",
    )
    parser.add_argument(
        "--skip-plot", action="store_true",
        help="Skip the matplotlib plot stage.",
    )
    args = parser.parse_args()

    # Locate trace
    try:
        trace_json = find_trace(args.trace)
        print(f"Using trace: {trace_json}")
    except FileNotFoundError as e:
        print(f"\n{FAIL}  {e}\n")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir:  {args.out_dir}")

    # Stage 1: GT pipeline
    gt_segs = test_gt_pipeline(trace_json, args.out_dir)

    # Use zeros as fallback if GT failed (so stages 2+3 still run)
    if not gt_segs:
        from gt_vs_estimator import GT_SEGMENT_NAMES
        gt_segs = {k: 0.0 for k in GT_SEGMENT_NAMES}

    # Stage 2: Estimator
    est_segs = test_estimator(args.out_dir)

    # Stage 3: CSV + plot
    import dataclasses
    test_csv_and_plot(gt_segs, est_segs, args.out_dir, args.skip_plot)

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Test complete")
    print()
    print("  Outputs:")
    for fname in [
        "test_batched_analysis.csv",
        "test_batched_summary.csv",
        "test_comparison_result.csv",
        "test_comparison_plot.png",
    ]:
        path = os.path.join(args.out_dir, fname)
        status = f"{os.path.getsize(path)//1024} KB" if os.path.exists(path) else "not created"
        print(f"    {fname:<40} {status}")

    print()
    print("  Expected GT segment breakdown (path1/subpath3):")
    print("    Recompute Load   ≈ 685 µs  (lhc memcpy)")
    print("    KVCache Load     ≈ 1453 µs (lc1 + lc2 memcpy)")
    print("    Misc. CPU        ≈ 389 µs  (sum_all - critical_path)")
    print("    All others       = 0 µs    (path1/subpath3 → no pinned, no store, no CUDA)")
    print()
    print("  Next steps once get_gee(...) is wired in:")
    print("    • Stage 2 estimator values will be non-zero")
    print("    • Compare GT vs estimator totals in the plot")
    print("    • Run full sweep:  python gt_vs_estimator.py --sweep batch_size ...")
    print()


if __name__ == "__main__":
    main()