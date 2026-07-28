#!/usr/bin/env python3
"""
pin_memory_benchmark.py

Microbenchmark suite for isolating sources of latency variance in
`Tensor.pin_memory()` (i.e. `.pinned_memory()` in your FlexGen fork).

For every call, records:
  - wall_ms       : wall-clock latency of the pin call
  - cpu_ms        : actual CPU time consumed by the process (user+sys) during
                    the call, from getrusage()
  - gap_ms        : wall_ms - cpu_ms. A large gap means the process was
                    descheduled/blocked/throttled for that time rather than
                    actually computing -- the key signal for cgroup throttling
                    or driver/GPU blocking, as opposed to genuinely more work.
  - minflt/majflt : page-fault count deltas (cold first-touch cost)
  - nr_throttled / throttled_time deltas from the cgroup cpu.stat file
    (v1 or v2), if the process is running under a CPU quota
  - vmstat deltas for pgfault, pgmajfault, compact_stall, thp_fault_fallback
    (memory-fragmentation / THP-compaction indicators)

HOW TO ADAPT:
  - `pin_tensor()` below wraps the stock `Tensor.pin_memory()`. Swap this for
    your actual `.pinned_memory()` call if your FlexGen fork's TorchTensor
    wrapper does something different internally.
  - `experiment_size_sweep()`'s default `sizes` are generic byte counts.
    Replace with the actual per-batch-size KV cache byte counts from your
    workload (e.g. bytes for batch=1,2,4,8,...) so the sweep lines up with
    what you're actually observing.
  - The cgroup paths in `read_cgroup_cpu_stat()` assume a fairly flat cgroup
    layout. If your container's cgroup is nested (common on shared hosts),
    run `cat /proc/self/cgroup` to find your exact path and adjust.

Requires: torch (CUDA build, run on a machine with a visible GPU), pandas.
"""

import gc
import os
import resource
import time
from pathlib import Path

import pandas as pd
import torch

assert torch.cuda.is_available(), "Needs a CUDA-visible GPU (pin_memory requires an active CUDA context)."

MB = 1024 ** 2
GB = 1024 ** 3


# --------------------------------------------------------------------------
# Swap this out if your `.pinned_memory()` differs from stock pin_memory()
# --------------------------------------------------------------------------
def pin_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.pin_memory()


# --------------------------------------------------------------------------
# System-state probes
# --------------------------------------------------------------------------
def read_cgroup_cpu_stat():
    """Works for common cgroup v1 / v2 layouts; returns None if unreadable."""
    for p in (
        Path("/sys/fs/cgroup/cpu.stat"),                # cgroup v2 unified
        Path("/sys/fs/cgroup/cpu/cpu.stat"),             # cgroup v1
        Path("/sys/fs/cgroup/cpu,cpuacct/cpu.stat"),     # cgroup v1 combined controller
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
    print(f"cgroup cpu.stat: {read_cgroup_cpu_stat()}")
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.exists() and period_path.exists():
        q, p = int(quota_path.read_text()), int(period_path.read_text())
        cores = "unlimited" if q < 0 else f"{q / p:.2f} cores"
        print(f"cfs_quota_us={q} cfs_period_us={p} -> {cores}")
    thp_path = Path("/sys/kernel/mm/transparent_hugepage/enabled")
    if thp_path.exists():
        print(f"THP mode: {thp_path.read_text().strip()}")
    print(f"torch.get_num_threads(): {torch.get_num_threads()}")
    print()


# --------------------------------------------------------------------------
# Core benchmark
# --------------------------------------------------------------------------
def run_experiment(size_bytes, dtype=torch.float16, n_iters=20,
                    reuse_buffer=False, pre_touch=False, num_threads=None,
                    label=""):
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    elem_size = torch.empty((1,), dtype=dtype).element_size()
    n_elements = size_bytes // elem_size

    src = None
    if reuse_buffer:
        src = torch.empty(n_elements, dtype=dtype)
        if pre_touch:
            src.zero_()

    records = []
    for i in range(n_iters):
        if reuse_buffer:
            t = src
        else:
            t = torch.empty(n_elements, dtype=dtype)
            if pre_touch:
                t.zero_()

        cg0, ru0, vm0 = read_cgroup_cpu_stat(), read_rusage(), read_vmstat()
        t0 = time.perf_counter()
        pinned = pin_tensor(t)
        t1 = time.perf_counter()
        cg1, ru1, vm1 = read_cgroup_cpu_stat(), read_rusage(), read_vmstat()

        wall_ms = (t1 - t0) * 1000
        cpu_ms = ((ru1["utime"] + ru1["stime"]) - (ru0["utime"] + ru0["stime"])) * 1000

        rec = {
            "label": label, "iter": i, "size_bytes": size_bytes,
            "num_threads": torch.get_num_threads(),
            "reuse_buffer": reuse_buffer, "pre_touch": pre_touch,
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

        records.append(rec)

        del pinned
        if not reuse_buffer:
            del t
        gc.collect()  # ensure pinned memory is actually released before next iter

    return records


def summarize(records, group_cols):
    df = pd.DataFrame(records)
    agg = df.groupby(group_cols)["wall_ms"].agg(
        mean="mean", std="std",
        p50=lambda s: s.quantile(0.50),
        p90=lambda s: s.quantile(0.90),
        p99=lambda s: s.quantile(0.99),
        max="max",
    )
    agg["cv"] = agg["std"] / agg["mean"]  # coefficient of variation
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    extra_cols = [c for c in ("gap_ms", "nr_throttled_delta", "throttled_time_delta_us",
                               "majflt_delta", "vmstat_compact_stall_delta") if c in df.columns]
    if extra_cols:
        print("mean of diagnostic columns per group:")
        print(df.groupby(group_cols)[extra_cols].mean().to_string(float_format=lambda x: f"{x:.3f}"))
        print()
    return agg


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def experiment_size_sweep(sizes=(64 * MB, 256 * MB, 1 * GB, 2 * GB, 4 * GB, 8 * GB), n_iters=20):
    print("### Experiment A: latency & variance vs. tensor size ###")
    print("Tests whether variance grows (and where it kicks in) as size increases.\n")
    all_records = []
    for sz in sizes:
        all_records.extend(run_experiment(sz, n_iters=n_iters, label=f"{sz / GB:.3f}GB"))
    raw_df = pd.DataFrame(all_records)
    summary_df = summarize(all_records, ["label", "size_bytes"])
    return raw_df, summary_df


def experiment_thread_sweep(size_bytes=8 * GB, thread_counts=(1, 2, 4, 8, 13, 16), n_iters=20):
    print("### Experiment B: latency & variance vs. thread count (fixed large size) ###")
    print("Tests whether variance collapses under fewer threads -- the signature")
    print("of a multi-threaded copy colliding with a cgroup CPU quota.\n")
    all_records = []
    for nt in thread_counts:
        all_records.extend(run_experiment(size_bytes, n_iters=n_iters, num_threads=nt, label=f"threads={nt}"))
    raw_df = pd.DataFrame(all_records)
    summary_df = summarize(all_records, ["label", "num_threads"])
    return raw_df, summary_df


def experiment_cache_and_paging(size_bytes=8 * GB, n_iters=20):
    print("### Experiment C: allocator-cache / cold-page-fault isolation ###")
    print("reuse_buffer=True reuses the same source tensor across iters (steady-state")
    print("allocator cache); pre_touch=True forces pages resident before pinning,")
    print("separating first-touch page-fault cost from the pin operation itself.\n")
    all_records = []
    for reuse in (False, True):
        for pretouch in (False, True):
            label = f"reuse={reuse},pretouch={pretouch}"
            all_records.extend(run_experiment(size_bytes, n_iters=n_iters,
                                               reuse_buffer=reuse, pre_touch=pretouch, label=label))
    raw_df = pd.DataFrame(all_records)
    summary_df = summarize(all_records, ["label", "reuse_buffer", "pre_touch"])
    return raw_df, summary_df


if __name__ == "__main__":
    print_system_info()

    raw_a, sum_a = experiment_size_sweep()
    raw_b, sum_b = experiment_thread_sweep()
    raw_c, sum_c = experiment_cache_and_paging()

    raw_a["experiment"], raw_b["experiment"], raw_c["experiment"] = "size_sweep", "thread_sweep", "cache_paging"
    raw_all = pd.concat([raw_a, raw_b, raw_c], ignore_index=True, sort=False)
    raw_all.to_csv("pin_memory_benchmark_raw.csv", index=False)

    sum_a["experiment"], sum_b["experiment"], sum_c["experiment"] = "size_sweep", "thread_sweep", "cache_paging"
    summary_all = pd.concat(
        [sum_a.reset_index(), sum_b.reset_index(), sum_c.reset_index()], ignore_index=True, sort=False
    )
    summary_all.to_csv("pin_memory_benchmark_summary.csv", index=False)

    print("Saved raw per-call data to pin_memory_benchmark_raw.csv")
    print("Saved summary stats to pin_memory_benchmark_summary.csv")