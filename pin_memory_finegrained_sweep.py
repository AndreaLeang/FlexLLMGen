#!/usr/bin/env python3
"""
pin_memory_finegrained_sweep.py

Fine-grained size sweep for Tensor.pin_memory() (i.e. `.pinned_memory()`).
Sweeps 50 log-spaced sizes between 64MiB and 8GiB. Deliberately narrower
in scope than the original benchmark suite:
  - never calls torch.set_num_threads() -- uses whatever the process's
    default thread count is, untouched
  - always uses a fresh, cold tensor per call (reuse_buffer=False,
    pre_touch=False, matching Experiment A's defaults) -- no toggles

Per-size iteration count is adaptive rather than fixed: for each size, a
few calibration calls run first, their steady-state latency (median of
the calibration calls, excluding the first -- which is a known cold
allocator-cache-miss outlier) is used to estimate how many more calls fit
into a TARGET_SECONDS_PER_SIZE budget, and that many additional calls run.
This means small/fast sizes get many more samples (cheap to collect,
better statistics) and large/slow sizes get fewer (expensive), while total
sweep time stays roughly predictable (~N_POINTS * TARGET_SECONDS_PER_SIZE)
regardless of how latency actually scales with size. All calibration calls
are kept as real data points, not discarded.

Same per-call diagnostics as before (wall/cpu time, page faults, cgroup
throttle deltas, vmstat compaction/THP deltas), in case you want to dig
into any part of the size-vs-variance curve later.

Swap `pin_tensor()` below for your actual `.pinned_memory()` call if it
differs from stock `Tensor.pin_memory()`.

Requires: torch (CUDA build, run on a machine with a visible GPU), pandas, numpy.
"""

import gc
import os
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

assert torch.cuda.is_available(), "Needs a CUDA-visible GPU (pin_memory requires an active CUDA context)."

MB = 1024 ** 2
GB = 1024 ** 3

N_POINTS = 50
MIN_SIZE = 64 * MB     # 0.0625 GB, matches the original sweep's smallest point
MAX_SIZE = 8 * GB

TARGET_SECONDS_PER_SIZE = 10.0  # aim to spend about this long benchmarking each size
N_CALIBRATION_ITERS = 4         # calls used to estimate per-call cost before planning the rest
MIN_ITERS_PER_SIZE = 5          # floor, even if calibration alone already blew the time budget
MAX_ITERS_PER_SIZE = 2000       # ceiling -- safety valve against a bad calibration estimate


def pin_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.pin_memory()


# --------------------------------------------------------------------------
# System-state probes (same as the original suite)
# --------------------------------------------------------------------------
def read_cgroup_cpu_stat():
    for p in (
        Path("/sys/fs/cgroup/cpu.stat"),
        Path("/sys/fs/cgroup/cpu/cpu.stat"),
        Path("/sys/fs/cgroup/cpu,cpuacct/cpu.stat"),
    ):
        if p.exists():
            stats = {}
            for line in p.read_text().splitlines():
                k, v = line.split()
                stats[k] = int(v)
            return stats
    return None


def read_rusage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {"utime": r.ru_utime, "stime": r.ru_stime,
            "minflt": r.ru_minflt, "majflt": r.ru_majflt}


VMSTAT_KEYS = ("pgfault", "pgmajfault", "compact_stall", "thp_fault_fallback")


def read_vmstat():
    stats = {}
    with open("/proc/vmstat") as f:
        for line in f:
            k, v = line.split()
            if k in VMSTAT_KEYS:
                stats[k] = int(v)
    return stats


def print_system_info():
    print("=" * 70)
    print("System info")
    print("=" * 70)
    print(f"CPU count (os.cpu_count): {os.cpu_count()}")
    print(f"torch.get_num_threads() [left untouched throughout]: {torch.get_num_threads()}")
    print(f"cgroup cpu.stat: {read_cgroup_cpu_stat()}")
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.exists() and period_path.exists():
        q, p = int(quota_path.read_text()), int(period_path.read_text())
        cores = "unlimited" if q < 0 else f"{q / p:.2f} cores"
        print(f"cfs_quota_us={q} cfs_period_us={p} -> {cores}")
    print()


# --------------------------------------------------------------------------
# Core benchmark, with adaptive per-size iteration count
# --------------------------------------------------------------------------
def _run_one_call(size_bytes, n_elements, dtype, idx, label):
    """Runs and times a single pin call. Returns (record_dict, full_iteration_ms).

    full_iteration_ms brackets the *entire* loop body (tensor alloc, probes,
    the pin call, cleanup) -- used for calibration/budgeting. record['wall_ms']
    is the tighter, probe-free timing of just the pin call itself -- used for
    the actual analysis, uncontaminated by bookkeeping overhead.
    """
    iter_t0 = time.perf_counter()

    t = torch.empty(n_elements, dtype=dtype)  # fresh, cold tensor every call

    cg0, ru0, vm0 = read_cgroup_cpu_stat(), read_rusage(), read_vmstat()
    t0 = time.perf_counter()
    pinned = pin_tensor(t)
    t1 = time.perf_counter()
    cg1, ru1, vm1 = read_cgroup_cpu_stat(), read_rusage(), read_vmstat()

    wall_ms = (t1 - t0) * 1000
    cpu_ms = ((ru1["utime"] + ru1["stime"]) - (ru0["utime"] + ru0["stime"])) * 1000

    rec = {
        "label": label, "iter": idx, "size_bytes": size_bytes,
        "num_threads": torch.get_num_threads(),
        "wall_ms": wall_ms, "cpu_ms": cpu_ms, "gap_ms": wall_ms - cpu_ms,
        "minflt_delta": ru1["minflt"] - ru0["minflt"],
        "majflt_delta": ru1["majflt"] - ru0["majflt"],
    }
    if cg0 and cg1:
        tkey = "throttled_usec" if "throttled_usec" in cg0 else "throttled_time"
        divisor = 1 if tkey == "throttled_usec" else 1000  # normalize to us
        rec["nr_throttled_delta"] = cg1.get("nr_throttled", 0) - cg0.get("nr_throttled", 0)
        rec["throttled_time_delta_us"] = (cg1.get(tkey, 0) - cg0.get(tkey, 0)) / divisor
    for k in VMSTAT_KEYS:
        rec[f"vmstat_{k}_delta"] = vm1.get(k, 0) - vm0.get(k, 0)

    del pinned, t
    gc.collect()  # ensure pinned memory is actually released before next iter

    full_iteration_ms = (time.perf_counter() - iter_t0) * 1000
    return rec, full_iteration_ms


def run_experiment(size_bytes, dtype=torch.float16, label="",
                    n_calibration=N_CALIBRATION_ITERS,
                    target_seconds=TARGET_SECONDS_PER_SIZE,
                    min_iters=MIN_ITERS_PER_SIZE,
                    max_iters=MAX_ITERS_PER_SIZE):
    elem_size = torch.empty((1,), dtype=dtype).element_size()
    n_elements = size_bytes // elem_size

    records = []
    iteration_times_ms = []

    # --- calibration phase: also real data, not thrown away ---
    for i in range(n_calibration):
        rec, full_ms = _run_one_call(size_bytes, n_elements, dtype, i, label)
        records.append(rec)
        iteration_times_ms.append(full_ms)

    calibration_elapsed_ms = sum(iteration_times_ms)

    # Estimate steady-state per-iteration cost. Drop the first calibration
    # call from the estimate when we have more than one -- it's a known
    # cold allocator-cache-miss outlier and would bias the estimate high.
    steady_state_samples = iteration_times_ms[1:] if len(iteration_times_ms) > 1 else iteration_times_ms
    est_ms_per_iter = statistics.median(steady_state_samples)

    remaining_budget_ms = target_seconds * 1000 - calibration_elapsed_ms
    n_extra = int(remaining_budget_ms // est_ms_per_iter) if est_ms_per_iter > 0 else 0
    n_extra = max(0, n_extra)

    if n_calibration + n_extra < min_iters:
        n_extra = min_iters - n_calibration
    n_extra = min(n_extra, max_iters - n_calibration)
    n_extra = max(n_extra, 0)

    # --- main phase: run however many more calls the budget allows ---
    for i in range(n_extra):
        rec, full_ms = _run_one_call(size_bytes, n_elements, dtype, n_calibration + i, label)
        records.append(rec)
        iteration_times_ms.append(full_ms)

    actual_elapsed_s = sum(iteration_times_ms) / 1000
    print(f"  [{label}] calibration est={est_ms_per_iter:.2f}ms/iter -> "
          f"{n_calibration}+{n_extra}={n_calibration + n_extra} iters, "
          f"actual elapsed {actual_elapsed_s:.2f}s (target {target_seconds:.1f}s)")

    return records


def main():
    print_system_info()

    sizes = np.logspace(np.log2(MIN_SIZE), np.log2(MAX_SIZE), num=N_POINTS, base=2).astype(np.int64)
    print(f"Sweeping {N_POINTS} log-spaced sizes from {MIN_SIZE / GB:.4f}GB to {MAX_SIZE / GB:.2f}GB, "
          f"target ~{TARGET_SECONDS_PER_SIZE:.0f}s/size after a {N_CALIBRATION_ITERS}-call calibration "
          f"(predicted total ~{N_POINTS * TARGET_SECONDS_PER_SIZE / 60:.1f} min, iteration count adapts per size)\n")

    sweep_t0 = time.perf_counter()
    all_records = []
    for idx, sz in enumerate(sizes):
        sz = int(sz)
        label = f"{sz / GB:.4f}GB"
        print(f"[{idx + 1}/{N_POINTS}] {label} ({sz} bytes)...")
        all_records.extend(run_experiment(sz, label=label))
    sweep_elapsed_min = (time.perf_counter() - sweep_t0) / 60

    raw_df = pd.DataFrame(all_records)
    raw_df.to_csv("pin_memory_finegrained_raw.csv", index=False)

    summary_df = raw_df.groupby(["label", "size_bytes"])["wall_ms"].agg(
        n_iters="count",
        mean="mean", std="std",
        p50=lambda s: s.quantile(0.50),
        p90=lambda s: s.quantile(0.90),
        p99=lambda s: s.quantile(0.99),
        max="max",
    ).reset_index()
    summary_df["cv"] = summary_df["std"] / summary_df["mean"]
    summary_df = summary_df.sort_values("size_bytes")
    summary_df.to_csv("pin_memory_finegrained_summary.csv", index=False)

    print(f"\nTotal sweep time: {sweep_elapsed_min:.1f} min "
          f"(iterations per size ranged {summary_df['n_iters'].min()}-{summary_df['n_iters'].max()})")
    print("Saved raw per-call data to pin_memory_finegrained_raw.csv")
    print("Saved summary stats to pin_memory_finegrained_summary.csv")


if __name__ == "__main__":
    main()