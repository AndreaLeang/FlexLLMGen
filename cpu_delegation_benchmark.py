"""
cpu_delegation_microbench.py

Microbenchmarks for isolating CPU-side overhead in CPU-delegated attention /
KV cache offloading (see pytorch_backend.py: TorchDevice.mha_gen ->
_attention_value / _mixed_cpu_attention, and general_copy()'s pin_memory
relay path).

Instead of tying shapes to a specific model's (n_heads, head_dim, batch,
seq_len), the bmm and transfer benchmarks sweep directly over payload size
(the total bytes touched by the operation), which is more useful for seeing
where CPU compute crosses cache/memory-bandwidth boundaries and where H2D
transfer crosses into PCIe-bandwidth-saturated territory -- independent of
which model you're actually running.

Three independent benchmarks, selectable with --bench:

  bmm     CPU torch.bmm(A, B) cost for square (batch, dim, dim) operands,
          swept across --bmm-sizes-mb (payload size of one operand). Also
          reports which CPU backend torch is dispatching to (MKL / oneDNN /
          OpenMP) and how many threads it's using.

  xfer    CPU->GPU H2D transfer cost for contiguous vs. non-contiguous
          source tensors, swept across --xfer-sizes-mb (1 MB up to a size
          that saturates PCIe bandwidth by default). Splits timing into the
          pin_memory() step and the copy_(..., non_blocking=True) step --
          this mirrors the relay path in general_copy() (`src =
          src.pin_memory(); dst.copy_(src, non_blocking=True)`), the likely
          source of the batch-size-dependent pin_memory latency fluctuation
          observed earlier.

  stall   CPU-side latency inflation from vCPU oversubscription: runs a
          fixed-size bmm workload (--stall-size-mb / --stall-batch) while
          background processes contend for the same physical cores (or,
          with --no-bg, with zero background processes to isolate host-level
          / cloud-vCPU scheduling jitter from self-induced contention), and
          reports tail latency plus (if available) cgroup cpu.stat
          throttling counters sampled before and after.

  all     run all three in sequence (default).

Usage:
  python cpu_delegation_microbench.py --bench bmm \
      --bmm-sizes-mb 0.0625 0.25 1 4 16 64 256

  python cpu_delegation_microbench.py --bench xfer \
      --xfer-sizes-mb 1 2 4 8 16 32 64 128 256 512 1024

  python cpu_delegation_microbench.py --bench stall --oversub 0.5 1 2 4

  # isolate host-level (cloud vCPU) jitter with zero self-induced contention
  python cpu_delegation_microbench.py --bench stall --no-bg --iters 2000

  python cpu_delegation_microbench.py --bench all --out results.csv
"""

import argparse
import csv
import multiprocessing as mp
import os
import statistics
import time
from dataclasses import dataclass
from typing import List

import torch


# --------------------------------------------------------------------------
# Common utilities
# --------------------------------------------------------------------------

@dataclass
class Timing:
    n: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


def summarize(latencies_s: List[float]) -> Timing:
    ms = sorted(x * 1000.0 for x in latencies_s)
    n = len(ms)

    def pct(p):
        idx = min(n - 1, int(p * n))
        return ms[idx]

    return Timing(
        n=n,
        mean_ms=statistics.mean(ms),
        median_ms=statistics.median(ms),
        p95_ms=pct(0.95),
        p99_ms=pct(0.99),
        std_ms=statistics.pstdev(ms) if n > 1 else 0.0,
        min_ms=ms[0],
        max_ms=ms[-1],
    )


def dtype_size(dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def mb_to_bytes(mb: float) -> int:
    return int(mb * 1024 * 1024)


def square_bmm_dim(batch: int, target_bytes: int, dtype):
    """Pick a square matrix dim so a (batch, dim, dim) tensor is ~target_bytes.
    Returns (dim, actual_bytes) -- actual_bytes may differ slightly from
    target_bytes since dim must be an integer."""
    elem_bytes = dtype_size(dtype)
    n_elems_per_matrix = max(1, target_bytes // (batch * elem_bytes))
    dim = max(1, int(round(n_elems_per_matrix ** 0.5)))
    actual_bytes = batch * dim * dim * elem_bytes
    return dim, actual_bytes


def _parse_stat_file(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                key, val = parts
                try:
                    out[key] = int(val)
                except ValueError:
                    pass
    return out


def read_cgroup_cpu_info():
    """The scheduling budget a process is actually confined to can be lower
    than os.cpu_count()/vCPU count on shared-core cloud instances (this is
    what cpu.cfs_quota_us / cpu.cfs_period_us vs. os.cpu_count() surfaced
    previously). Handles both cgroup v1 and v2 layouts."""
    info = {
        "os_cpu_count": os.cpu_count(),
        "cgroup_version": None,
        "quota_cores": None,
        "throttled_time_ns": None,
        "nr_throttled": None,
        "nr_periods": None,
    }

    v2_max, v2_stat = "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.stat"
    v1_quota = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    v1_period = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    v1_stat = "/sys/fs/cgroup/cpu/cpu.stat"

    try:
        if os.path.exists(v2_max):
            info["cgroup_version"] = "v2"
            with open(v2_max) as f:
                quota, period = f.read().split()
            if quota != "max":
                info["quota_cores"] = int(quota) / int(period)
            if os.path.exists(v2_stat):
                stat = _parse_stat_file(v2_stat)
                if "throttled_usec" in stat:
                    info["throttled_time_ns"] = stat["throttled_usec"] * 1000
                info["nr_throttled"] = stat.get("nr_throttled")
                info["nr_periods"] = stat.get("nr_periods")
        elif os.path.exists(v1_quota):
            info["cgroup_version"] = "v1"
            with open(v1_quota) as f:
                quota = int(f.read().strip())
            with open(v1_period) as f:
                period = int(f.read().strip())
            if quota > 0:
                info["quota_cores"] = quota / period
            if os.path.exists(v1_stat):
                stat = _parse_stat_file(v1_stat)
                info["throttled_time_ns"] = stat.get("throttled_time")  # already ns in v1
                info["nr_throttled"] = stat.get("nr_throttled")
                info["nr_periods"] = stat.get("nr_periods")
    except (OSError, ValueError):
        pass

    return info


def report_cpu_backend():
    """torch.bmm on CPU dispatches through ATen's CPU GEMM path, which is
    backed by MKL if torch was built with it, else OpenBLAS/native. This
    prints what's actually active plus the effective (cgroup-aware) core
    budget, which os.cpu_count() alone can overstate on shared vCPU hosts."""
    print("=" * 78)
    print("CPU BACKEND INFO")
    print("=" * 78)
    print(f"torch version:             {torch.__version__}")
    print(f"MKL available:             {torch.backends.mkl.is_available()}")
    print(f"MKL-DNN (oneDNN) available:{torch.backends.mkldnn.is_available()}")
    print(f"OpenMP available:          {torch.backends.openmp.is_available()}")
    print(f"torch.get_num_threads():        {torch.get_num_threads()}")
    print(f"torch.get_num_interop_threads():{torch.get_num_interop_threads()}")

    cginfo = read_cgroup_cpu_info()
    print(f"os.cpu_count():            {cginfo['os_cpu_count']}")
    if cginfo["quota_cores"] is not None:
        print(f"cgroup ({cginfo['cgroup_version']}) quota:       {cginfo['quota_cores']:.2f} cores "
              f"<-- true scheduling budget; can be < os.cpu_count() on shared vCPU hosts")
    else:
        print(f"cgroup quota:              unconfined / not found "
              f"(detected version: {cginfo['cgroup_version']})")

    print("\ntorch.__config__.parallel_info():")
    print(torch.__config__.parallel_info())
    print("torch.__config__.show() (BLAS/LAPACK build info):")
    print(torch.__config__.show())
    print("=" * 78)
    return cginfo


# --------------------------------------------------------------------------
# 1) CPU bmm() payload-size sweep
# --------------------------------------------------------------------------

def bench_bmm(batch, dim, warmup=10, iters=50, dtype=torch.float32):
    """Generic CPU torch.bmm(A, B) benchmark, A/B: (batch, dim, dim).
    dtype defaults to float32 because the real CPU-delegated attention path
    upcasts via `.float()` before running on CPU (CPU fp16 matmul is
    unsupported/very slow) -- see pytorch_backend.py's `q.float().cpu()`.
    """
    a = torch.randn(batch, dim, dim, dtype=dtype)
    b = torch.randn(batch, dim, dim, dtype=dtype)

    for _ in range(warmup):
        _ = torch.bmm(a, b)

    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = torch.bmm(a, b)
        lat.append(time.perf_counter() - t0)

    flops = 2 * batch * (dim ** 3)  # batch * 2*M*K*N with M=K=N=dim
    mean_s = statistics.mean(lat)
    gflops = (flops / mean_s) / 1e9 if mean_s > 0 else float("nan")
    read_bytes = 2 * batch * dim * dim * dtype_size(dtype)  # A + B read per call
    gbps = (read_bytes / mean_s) / 1e9 if mean_s > 0 else float("nan")

    return summarize(lat), gflops, gbps


def run_bmm_benchmark(args):
    report_cpu_backend()
    rows = []
    print(f"\n{'target(MB)':>11} {'batch':>6} {'dim':>6} {'actual(MB)':>11} | "
          f"{'median(ms)':>12} {'p99(ms)':>10} {'GFLOP/s':>9} {'GB/s':>8}")
    for size_mb in args.bmm_sizes_mb:
        target_bytes = mb_to_bytes(size_mb)
        dim, actual_bytes = square_bmm_dim(args.bmm_batch, target_bytes, torch.float32)
        t, gflops, gbps = bench_bmm(args.bmm_batch, dim, warmup=args.warmup, iters=args.iters)
        actual_mb = actual_bytes / (1024 * 1024)
        print(f"{size_mb:>11.4g} {args.bmm_batch:>6} {dim:>6} {actual_mb:>11.4f} | "
              f"{t.median_ms:>12.4f} {t.p99_ms:>10.4f} {gflops:>9.2f} {gbps:>8.2f}")
        rows.append({
            "target_mb": size_mb, "batch": args.bmm_batch, "dim": dim,
            "actual_mb": actual_mb, "median_ms": t.median_ms, "p99_ms": t.p99_ms,
            "std_ms": t.std_ms, "gflops": gflops, "gbps": gbps,
        })
    return rows


# --------------------------------------------------------------------------
# 2) CPU -> GPU transfer payload-size sweep (contiguous vs non-contiguous)
# --------------------------------------------------------------------------

def make_noncontiguous(t: torch.Tensor) -> torch.Tensor:
    """Return a non-contiguous view with the same shape/values as `t`,
    standing in for the slices general_copy() actually sees: KV cache tiles
    arrive there as slices of a permuted cache tensor, which is
    non-contiguous relative to the compact layout pin_memory()/copy_ would
    prefer.
    """
    padded = torch.empty((t.shape[-1],) + t.shape[:-1], dtype=t.dtype)
    padded.copy_(t.movedim(-1, 0))
    view = padded.movedim(0, -1)
    assert view.shape == t.shape and not view.is_contiguous()
    return view


def bench_h2d_transfer(target_bytes, dtype=torch.float16, cols=4096, warmup=5, iters=20):
    if not torch.cuda.is_available():
        return None

    elem_bytes = dtype_size(dtype)
    total_elems = max(cols, int(target_bytes // elem_bytes))
    rows = max(1, total_elems // cols)
    shape = (rows, cols)
    actual_bytes = rows * cols * elem_bytes

    base = torch.randn(*shape, dtype=dtype)
    sources = {
        "contiguous": base.clone(),
        "non_contiguous": make_noncontiguous(base),
    }
    gpu_dst = torch.empty(shape, dtype=dtype, device="cuda")

    results = {"shape": shape, "actual_bytes": actual_bytes}
    for label, src in sources.items():
        assert src.is_contiguous() == (label == "contiguous")
        pin_lat, copy_lat, total_lat = [], [], []
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        for i in range(warmup + iters):
            t0 = time.perf_counter()
            pinned = src.pin_memory()  # mirrors general_copy(): `src = src.pin_memory()`
            t1 = time.perf_counter()

            torch.cuda.synchronize()
            start_evt.record()
            gpu_dst.copy_(pinned, non_blocking=True)  # mirrors `dst.copy_(src, non_blocking=True)`
            end_evt.record()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            if i >= warmup:
                pin_lat.append(t1 - t0)
                copy_lat.append(start_evt.elapsed_time(end_evt) / 1000.0)
                total_lat.append(t2 - t0)

        copy_median_s = statistics.median(copy_lat)
        total_median_s = statistics.median(total_lat)
        results[label] = {
            "pin_memory": summarize(pin_lat),
            "copy_": summarize(copy_lat),
            "total": summarize(total_lat),
            "copy_gbps": (actual_bytes / copy_median_s) / 1e9 if copy_median_s > 0 else float("nan"),
            "total_gbps": (actual_bytes / total_median_s) / 1e9 if total_median_s > 0 else float("nan"),
        }
    return results


def run_xfer_benchmark(args):
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False -- skipping the CPU->GPU transfer "
              "benchmark (run this on the GPU pod, not locally).")
        return []

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    if max(args.xfer_sizes_mb) >= 256:
        print("Note: large payload points will take a while, since pin_memory() "
              "re-pins the full buffer every iteration (matching general_copy()'s "
              "behavior) -- reduce --iters if this is too slow.")
    rows = []
    print(f"\n{'target(MB)':>11} {'actual(MB)':>11} {'layout':>15} | "
          f"{'pin_memory median(ms)':>22} {'copy_ median(ms)':>17} {'copy GB/s':>10} "
          f"{'total median(ms)':>17} {'total p99(ms)':>14}")
    for size_mb in args.xfer_sizes_mb:
        target_bytes = mb_to_bytes(size_mb)
        res = bench_h2d_transfer(target_bytes, dtype=torch.float16, cols=args.xfer_cols,
                                  warmup=args.warmup, iters=args.iters)
        actual_mb = res["actual_bytes"] / (1024 * 1024)
        for layout in ("contiguous", "non_contiguous"):
            r = res[layout]
            print(f"{size_mb:>11.4g} {actual_mb:>11.4f} {layout:>15} | "
                  f"{r['pin_memory'].median_ms:>22.4f} {r['copy_'].median_ms:>17.4f} "
                  f"{r['copy_gbps']:>10.2f} {r['total'].median_ms:>17.4f} {r['total'].p99_ms:>14.4f}")
            rows.append({
                "target_mb": size_mb, "actual_mb": actual_mb, "layout": layout,
                "pin_memory_median_ms": r["pin_memory"].median_ms,
                "pin_memory_p99_ms": r["pin_memory"].p99_ms,
                "copy_median_ms": r["copy_"].median_ms,
                "copy_p99_ms": r["copy_"].p99_ms,
                "copy_gbps": r["copy_gbps"],
                "total_median_ms": r["total"].median_ms,
                "total_p99_ms": r["total"].p99_ms,
                "total_std_ms": r["total"].std_ms,
                "total_gbps": r["total_gbps"],
            })
    return rows


# --------------------------------------------------------------------------
# 3) vCPU oversubscription stall benchmark
# --------------------------------------------------------------------------

def _bg_worker(stop_flag, batch, dim):
    """Background contender: same square-bmm workload as the foreground
    measurement, pinned to 1 thread so the total oversubscription ratio is
    controlled purely by process count vs. core count."""
    torch.set_num_threads(1)
    a = torch.randn(batch, dim, dim)
    b = torch.randn(batch, dim, dim)
    while not stop_flag.value:
        _ = torch.bmm(a, b)


def bench_oversubscription(n_bg_procs, batch, dim, warmup=10, iters=100):
    """Runs `n_bg_procs` background bmm workers concurrently with a
    foreground measurement loop doing the same workload, quantifying how
    much vCPU oversubscription / noisy-neighbor contention inflates and
    adds jitter to CPU-delegated attention latency. n_bg_procs=0 measures
    the foreground workload alone -- any spikes in that case come from the
    host's own scheduling (real vCPU oversubscription / co-tenant noise)
    rather than contention this script generates itself.

    Returns (Timing, n_spikes_gt_2x_median, n_spikes_gt_5x_median).
    """
    torch.set_num_threads(1)
    a = torch.randn(batch, dim, dim)
    b = torch.randn(batch, dim, dim)

    stop_flag = mp.Value("b", False)
    procs = [mp.Process(target=_bg_worker, args=(stop_flag, batch, dim))
             for _ in range(n_bg_procs)]
    for p in procs:
        p.start()
    if procs:
        time.sleep(0.5)  # let background workers ramp up

    try:
        for _ in range(warmup):
            _ = torch.bmm(a, b)

        lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = torch.bmm(a, b)
            lat.append(time.perf_counter() - t0)
    finally:
        stop_flag.value = True
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    median_s = statistics.median(lat)
    n_spikes_2x = sum(1 for x in lat if x > 2 * median_s)
    n_spikes_5x = sum(1 for x in lat if x > 5 * median_s)
    return summarize(lat), n_spikes_2x, n_spikes_5x


def run_stall_benchmark(args):
    cginfo = read_cgroup_cpu_info()
    cores = cginfo["quota_cores"] or cginfo["os_cpu_count"]
    print(f"\nUsing {cores:.1f} cores as the oversubscription baseline "
          f"({'cgroup quota' if cginfo['quota_cores'] else 'os.cpu_count()'})")

    target_bytes = mb_to_bytes(args.stall_size_mb)
    dim, actual_bytes = square_bmm_dim(args.stall_batch, target_bytes, torch.float32)
    print(f"Workload: batch={args.stall_batch}, dim={dim} "
          f"(~{actual_bytes / (1024 * 1024):.3f} MB per bmm operand)\n")

    if args.no_bg:
        print("--no-bg set: spawning zero background processes, --oversub is ignored. "
              "Any spikes/tail latency below reflect the host's own vCPU scheduling "
              "rather than contention this script is generating -- consider a larger "
              "--iters to widen the observation window, since host-level noisy-neighbor "
              "events can be intermittent.\n")
        runs = [("no_bg", 0)]
    else:
        runs = [(mult, max(0, int(round(mult * cores)) - 1)) for mult in args.oversub]

    print(f"{'mode':>10} {'bg procs':>9} {'median(ms)':>12} {'p95(ms)':>10} "
          f"{'p99(ms)':>10} {'max(ms)':>10} {'std(ms)':>10} "
          f"{'spikes>2x':>10} {'spikes>5x':>10} {'throttled_delta(ms)':>20}")
    rows = []
    for mode, n_bg in runs:
        before = read_cgroup_cpu_info()
        t, n_spikes_2x, n_spikes_5x = bench_oversubscription(
            n_bg, args.stall_batch, dim, warmup=args.warmup, iters=args.iters)
        after = read_cgroup_cpu_info()

        throttled_delta = None
        if before["throttled_time_ns"] is not None and after["throttled_time_ns"] is not None:
            throttled_delta = (after["throttled_time_ns"] - before["throttled_time_ns"]) / 1e6

        delta_str = f"{throttled_delta:.2f}" if throttled_delta is not None else "n/a"
        print(f"{str(mode):>10} {n_bg:>9} {t.median_ms:>12.4f} {t.p95_ms:>10.4f} "
              f"{t.p99_ms:>10.4f} {t.max_ms:>10.4f} {t.std_ms:>10.4f} "
              f"{n_spikes_2x:>10} {n_spikes_5x:>10} {delta_str:>20}")
        rows.append({
            "mode": mode, "bg_procs": n_bg, "batch": args.stall_batch, "dim": dim,
            "median_ms": t.median_ms, "p95_ms": t.p95_ms,
            "p99_ms": t.p99_ms, "max_ms": t.max_ms, "std_ms": t.std_ms,
            "n_spikes_gt_2x_median": n_spikes_2x, "n_spikes_gt_5x_median": n_spikes_5x,
            "throttled_delta_ms": throttled_delta,
        })
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", choices=["bmm", "xfer", "stall", "all"], default="all")

    p.add_argument("--bmm-sizes-mb", type=float, nargs="+",
                    default=[0.0625, 0.25, 1, 4, 16, 64, 256],
                    help="payload size (MB) of one bmm operand to sweep")
    p.add_argument("--bmm-batch", type=int, default=32,
                    help="batch dim for the square (batch, dim, dim) bmm")

    p.add_argument("--xfer-sizes-mb", type=float, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                    help="payload size (MB) to sweep for the H2D transfer benchmark, "
                         "from 1 MB up to a size that should saturate PCIe bandwidth")
    p.add_argument("--xfer-cols", type=int, default=4096,
                    help="inner (contiguous) dimension used to shape transfer tensors; "
                         "row count is derived to hit the target payload size")

    p.add_argument("--stall-size-mb", type=float, default=1.0,
                    help="payload size (MB) of one bmm operand used for the stall workload")
    p.add_argument("--stall-batch", type=int, default=32,
                    help="batch dim for the stall benchmark's bmm workload")
    p.add_argument("--oversub", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0],
                    help="oversubscription multipliers relative to the cgroup core quota "
                         "(1.0 = fully subscribed, 2.0 = 2x oversubscribed); ignored if "
                         "--no-bg is set")
    p.add_argument("--no-bg", action="store_true",
                    help="stall bench: spawn zero background processes and just measure "
                         "the foreground workload alone (ignores --oversub). Use this to "
                         "check whether latency stalls persist purely from the cloud "
                         "host's own vCPU scheduling, independent of any contention this "
                         "script generates itself")

    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", type=str, default=None,
                    help="CSV path stem to append results to -- since bmm/xfer/stall have "
                         "different columns, each is written to its own file "
                         "(e.g. --out results.csv -> results_bmm.csv, results_xfer.csv, "
                         "results_stall.csv); each is appended to across runs")
    return p


def derive_csv_path(base_path, bench_name):
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".csv"
    return f"{root}_{bench_name}{ext}"


def write_csv(rows, path):
    if not rows:
        return
    file_exists = os.path.exists(path)
    fieldnames = list(rows[0].keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    args = build_argparser().parse_args()
    all_rows = {}

    if args.bench in ("bmm", "all"):
        print("\n" + "#" * 78)
        print("# 1) CPU-side bmm() payload-size sweep")
        print("#" * 78)
        all_rows["bmm"] = run_bmm_benchmark(args)

    if args.bench in ("xfer", "all"):
        print("\n" + "#" * 78)
        print("# 2) CPU->GPU transfer payload-size sweep (contiguous vs non-contiguous, pin_memory)")
        print("#" * 78)
        all_rows["xfer"] = run_xfer_benchmark(args)

    if args.bench in ("stall", "all"):
        print("\n" + "#" * 78)
        print("# 3) vCPU oversubscription stall benchmark")
        print("#" * 78)
        all_rows["stall"] = run_stall_benchmark(args)

    if args.out:
        written = []
        for bench_name, rows in all_rows.items():
            if not rows:
                continue
            path = derive_csv_path(args.out, bench_name)
            write_csv(rows, path)
            written.append(path)
        if written:
            print(f"\nWrote results to: {', '.join(written)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safe if a CUDA context is already live in xfer
    main()