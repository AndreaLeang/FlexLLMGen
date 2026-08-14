#!/usr/bin/env python3
"""
cpu_matmul_bench.py
====================

Two things in one script, per the discussion this is built for:

  1) Batched matmul throughput across a sweep of shapes and dtypes
     (float32 / bfloat16 / float16), on CPU.

  2) A diagnostic trace that shows which backend (oneDNN vs MKL) and which
     ISA-level kernel (avx2 / avx512_core / avx512_core_vnni /
     avx512_core_amx / ...) actually executed a given op -- so you don't
     have to infer it from timing alone.

USAGE
-----
Basic throughput sweep, default dtypes/shapes, default thread count:
    python cpu_matmul_bench.py

Also print a oneDNN dispatch trace for one representative shape:
    python cpu_matmul_bench.py --diagnose

Get the MKL-level trace too (MKL is what plain fp32 matmul/bmm usually
goes through, and is the library with the historical vendor-check
behavior -- this env var has to be set before the process starts,
a Python-level os.environ set after import torch is too late):
    MKL_VERBOSE=1 python cpu_matmul_bench.py --diagnose 2>&1 | tee mkl_verbose.log

Restrict to bfloat16 only, fixed thread count, dump raw results to CSV:
    python cpu_matmul_bench.py --dtypes bfloat16 --threads 56 --csv out.csv

NOTES FOR A FAIR CROSS-MACHINE COMPARISON
------------------------------------------
- Pin threads to physical (non-hyperthread) core count with --threads;
  comparing default thread counts across CPUs with different core counts
  conflates "more cores" with "faster per-core".
- On 2-socket boxes, bind to one socket to avoid cross-socket NUMA noise:
      numactl --cpunodebind=0 --membind=0 python cpu_matmul_bench.py ...
- Pin the frequency governor to "performance" if you want repeatable
  numbers instead of turbo-boost variance:
      sudo cpupower frequency-set -g performance
- If running inside a container/VM, /proc/cpuinfo may not reflect the
  host's real ISA support (AMX in particular is sometimes masked) --
  cross-check against the host or a bare-metal run if numbers look off.
"""

import argparse
import csv
import os
import platform
import re
import statistics
import sys
import time

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Backend / ISA introspection
# --------------------------------------------------------------------------

ISA_FLAGS_OF_INTEREST = [
    "avx", "avx2", "avx512f", "avx512bw", "avx512vnni", "avx512_vnni",
    "avx512_bf16", "avx512_fp16", "amx_tile", "amx_bf16", "amx_int8", "amx_fp16",
]


def get_cpu_isa_flags():
    """Best-effort read of relevant ISA flags from /proc/cpuinfo (Linux only)."""
    if platform.system() != "Linux":
        return {"note": f"flag check only implemented for Linux (system={platform.system()})"}
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
    except OSError as e:
        return {"error": str(e)}
    m = re.search(r"^flags\s*:\s*(.*)$", text, re.MULTILINE)
    if not m:
        return {"error": "could not find a 'flags' line in /proc/cpuinfo"}
    present = set(m.group(1).split())
    return {flag: (flag in present) for flag in ISA_FLAGS_OF_INTEREST}


def print_backend_info():
    print("=" * 78)
    print("Environment / backend summary")
    print("=" * 78)
    print(f"torch version          : {torch.__version__}")
    print(f"platform                : {platform.platform()}")
    try:
        print(f"ATen cpu_capability     : {torch.backends.cpu.get_cpu_capability()}")
    except AttributeError:
        print("ATen cpu_capability     : (not available in this torch version)")
    print(f"num threads              : {torch.get_num_threads()}")
    print(f"MKL available            : {torch.backends.mkl.is_available()}")
    print(f"oneDNN (mkldnn) available: {torch.backends.mkldnn.is_available()}")
    try:
        print(f"OpenMP available         : {torch.backends.openmp.is_available()}")
    except AttributeError:
        pass
    print()
    print("torch.__config__.show():")
    print(torch.__config__.show())
    print("Relevant /proc/cpuinfo ISA flags:")
    for k, v in get_cpu_isa_flags().items():
        print(f"  {k:14s}: {v}")
    print("=" * 78)
    print()


def diagnose_dispatch(batch, m, k, n, dtype, use_linear=True):
    """
    Run a single representative op wrapped in oneDNN's verbose mode, so the
    JIT-selected primitive/ISA tag (e.g. avx512_core_amx, avx512_core_vnni,
    avx512_core_fp16, avx2) prints directly to stdout from the C++ library.

    This only covers ops that go through oneDNN. Plain fp32 torch.matmul/
    torch.bmm typically go through MKL's sgemm instead, which has its own
    verbose mode (MKL_VERBOSE=1) that must be set as a shell env var before
    the process starts -- see the module docstring.
    """
    op_name = "nn.Linear" if use_linear else "torch.bmm"
    print(f"--- oneDNN dispatch trace: batch={batch} m={m} k={k} n={n} "
          f"dtype={dtype} op={op_name} ---")

    x = torch.randn(batch, m, k, dtype=dtype)
    if use_linear:
        layer = nn.Linear(k, n, bias=False).to(dtype)
    else:
        w = torch.randn(batch, k, n, dtype=dtype)

    try:
        verbose_ctx = torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON)
    except AttributeError:
        print("torch.backends.mkldnn.verbose() not available in this torch version.")
        print("Fall back to: DNNL_VERBOSE=1 python cpu_matmul_bench.py --diagnose")
        verbose_ctx = None

    with torch.no_grad():
        if verbose_ctx is not None:
            with verbose_ctx:
                _ = layer(x) if use_linear else torch.bmm(x, w)
        else:
            _ = layer(x) if use_linear else torch.bmm(x, w)

    if dtype == torch.float16:
        print("(dtype=float16: if no oneDNN lines appeared above, this op likely "
              "upcast to float32 internally rather than dispatching a native "
              "fp16 kernel -- see the fp16-vs-bf16 discussion this script is for.)")
    print(f"--- end trace ---\n")


# --------------------------------------------------------------------------
# Throughput benchmark
# --------------------------------------------------------------------------

DEFAULT_SHAPES = [
    # (batch, m, k, n) -- batch/m roughly stand in for (batch*seq, tokens),
    # k/n for hidden/intermediate dims. Edit freely for your own workload.
    (1,    128,  4096,  4096),
    (1,    128,  4096, 11008),
    (8,    128,  4096,  4096),
    (32,   128,  4096,  4096),
    (1,   2048,  4096,  4096),
    (1,    128, 11008,  4096),
]

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def flops_batched_matmul(batch, m, k, n):
    return 2.0 * batch * m * k * n  # multiply + add per MAC


def time_callable(fn, n_warmup, n_iters):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_shape(batch, m, k, n, dtype, use_linear, n_warmup, n_iters):
    x = torch.randn(batch, m, k, dtype=dtype)
    if use_linear:
        layer = nn.Linear(k, n, bias=False).to(dtype)
        fn = lambda: layer(x)
    else:
        w = torch.randn(batch, k, n, dtype=dtype)
        fn = lambda: torch.bmm(x, w)

    with torch.no_grad():
        times = time_callable(fn, n_warmup, n_iters)

    median_t = statistics.median(times)
    min_t = min(times)
    flops = flops_batched_matmul(batch, m, k, n)
    return {
        "op": "linear" if use_linear else "bmm",
        "batch": batch, "m": m, "k": k, "n": n, "dtype": str(dtype).replace("torch.", ""),
        "median_ms": median_t * 1e3,
        "min_ms": min_t * 1e3,
        "median_GFLOPs": flops / median_t / 1e9,
        "best_GFLOPs": flops / min_t / 1e9,
    }


def print_results_table(results):
    headers = ["op", "batch", "m", "k", "n", "dtype", "median_ms", "min_ms",
               "median_GFLOPs", "best_GFLOPs"]

    def cell(r, h):
        v = r[h]
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    col_w = {h: max(len(h), max((len(cell(r, h)) for r in results), default=0))
             for h in headers}

    def fmt_row(vals):
        return "  ".join(str(v).rjust(col_w[h]) for h, v in zip(headers, vals))

    print(fmt_row(headers))
    print(fmt_row(["-" * col_w[h] for h in headers]))
    for r in results:
        print(fmt_row([cell(r, h) for h in headers]))
    print()


def write_csv(results, path):
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows to {path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtypes", type=str, default="float32,bfloat16,float16",
                    help="comma-separated subset of: float32,bfloat16,float16")
    p.add_argument("--op", choices=["linear", "bmm", "both"], default="both",
                    help="which op to benchmark (default: both)")
    p.add_argument("--threads", type=int, default=None,
                    help="override torch.set_num_threads(); default leaves torch's own default")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--diagnose", action="store_true",
                    help="also print a oneDNN dispatch trace for one representative shape per dtype")
    p.add_argument("--csv", type=str, default=None, help="optional path to dump raw results as CSV")
    return p.parse_args()


def main():
    args = parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    dtypes = [DTYPE_MAP[d.strip()] for d in args.dtypes.split(",") if d.strip()]
    use_linear_flags = {"linear": [True], "bmm": [False], "both": [True, False]}[args.op]

    print_backend_info()

    results = []
    for dtype in dtypes:
        for use_linear in use_linear_flags:
            for (batch, m, k, n) in DEFAULT_SHAPES:
                r = bench_shape(batch, m, k, n, dtype, use_linear, args.warmup, args.iters)
                results.append(r)

    print_results_table(results)

    if args.csv:
        write_csv(results, args.csv)

    if args.diagnose:
        print("Diagnostic dispatch traces (one representative shape per dtype/op):")
        print("(Run with MKL_VERBOSE=1 set in the shell to also see MKL's own trace")
        print(" for plain fp32 matmul/bmm, which usually bypasses oneDNN entirely.)\n")
        batch, m, k, n = DEFAULT_SHAPES[0]
        for dtype in dtypes:
            for use_linear in use_linear_flags:
                diagnose_dispatch(batch, m, k, n, dtype, use_linear)


if __name__ == "__main__":
    main()