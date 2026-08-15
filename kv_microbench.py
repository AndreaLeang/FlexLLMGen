#!/usr/bin/env python3
"""
kv_microbench.py
=================
Generates the three microbenchmark inputs kv_bench_predictor.py's measured-
mode knobs consume, all under one --out-prefix:

  1. CPU pinning      <prefix>_pinning.csv       (bytes,latency_s table)
  2. PCIe H2D transfer <prefix>_pcie.csv          (bytes,latency_s table)
  3. GPU compute       <prefix>_gpu_compute.csv   (detailed per-shape TFLOP/s)
                        + <prefix>_gpu_compute_summary.txt with the
                        recommended --gemm-tflops/--attn-tflops values,
                        also printed to stdout, ready to paste into
                        kv_bench_predictor.py

Consolidates cpu_delegation_benchmark.py's pin_memory()/copy_() timing code
(same warmup+iters+percentile style, same CUDA-event-based copy timing) into
a single-purpose script and adds the GPU-compute benchmark that script never
had. Drops that script's CPU-only bmm bench and the vCPU-oversubscription
("stall") bench -- both were for a different question (is CPU-delegated
attention/host noisy-neighbor jitter a problem), not inputs this prediction
model's knobs take.

Swept byte sizes / GEMM & attention shapes are DERIVED from a workload
envelope (opt-2.7b..opt-30b, seq_len 1k..8k tokens, batch_size 1..32,
offload_percent 0..100, recompute fraction 0..0.5 of seq_len -- see
WORKLOAD_RANGE / --models / --seq-lens / --batch-sizes below to change it),
not hand-picked. The byte-size derivation mirrors
kv_schedule_optimization.get_bytes_to_load()/get_bytes_to_store() exactly
(bytes_per_token = hidden_size * 2), so swept sizes match what a real run
actually transfers/pins rather than an arbitrary power-of-two ladder. The
GEMM/attention shapes mirror layer_calc_pred()'s "Actual Model" query
construction: QKVO projections + FFN matmuls are M=batch_size (or
M=batch_size*recompute_len for KV recomputation), K/N=hidden_size or
ffn_embed_dim -- real torch.mm calls, not batched -- while QK^T / Attn.V are
genuinely batched (batch_size*n_head, 1, head_dim) x (.., head_dim, seq_len)
torch.bmm calls. These two op shapes have very different achieved-throughput
profiles in practice (dense square-ish GEMMs vs. small-M memory-bound bmms),
which is exactly why kv_bench_predictor.py's GPU-compute knob keeps them
separate (--gemm-tflops / --attn-tflops).

Every benchmark uses adaptive iteration counts, not a fixed count: each
size/shape runs a fixed warmup, measures those warmup latencies, then picks
enough timed iterations to cover >= --target-seconds (default 1s) of total
measured time (see _adaptive_iters()). A fixed iteration count applied
equally to a 5us pin call and a 300ms one means the fast one ends up timed
for a handful of microseconds total, dominated by whatever jitter the
measurement itself introduces -- adaptive sizing keeps small payloads from
being under-sampled without wasting time re-running already-slow ones.

Requires a CUDA GPU (pin_memory()/H2D copy/matmul throughput are all
meaningless to benchmark without one) -- exits early with a clear message
if none is visible.

Usage
-----
  # everything, using the default workload envelope:
  python kv_microbench.py --bench all --out-prefix opt2.7b-30b_1k-8k

  # just one component, smaller/faster sweep:
  python kv_microbench.py --bench pinning --out-prefix quick \\
      --batch-sizes 1 8 32 --seq-lens 1024 8192

Then feed the outputs straight into kv_bench_predictor.py:
  python kv_bench_predictor.py --bench-csv opt-30b_p1024_g32_bench.csv \\
      --pcie-bw 16 --gpu-tflops 312 \\
      --pinning-bench-csv opt2.7b-30b_1k-8k_pinning.csv \\
      --pcie-bench-csv opt2.7b-30b_1k-8k_pcie.csv \\
      --gemm-tflops <printed recommendation> --attn-tflops <printed recommendation>
"""

import argparse
import csv
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


# ===========================================================================
# Common utilities (same style as cpu_delegation_benchmark.py)
# ===========================================================================

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


def print_env_info():
    print("=" * 78)
    print(f"torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(f"CUDA version:  {torch.version.cuda}")
    else:
        print("GPU:           none visible")
    print("=" * 78)


def _adaptive_iters(warmup_latencies_s: List[float], target_seconds: float = 1.0,
                     min_iters: int = 10, max_iters: int = 2000) -> int:
    """
    Given per-call latencies measured during warmup, pick how many timed
    iterations to run so the timed phase covers >= target_seconds of total
    wall time. Uses the warmup MEDIAN (matching this script's
    median-everywhere convention -- see run() docstrings) as the
    per-iteration cost estimate. Clamped to [min_iters, max_iters] so a
    pathologically fast op doesn't demand tens of thousands of iterations,
    and a pathologically slow one still gets at least min_iters for a
    usable median/percentile.
    """
    if not warmup_latencies_s:
        return min_iters
    median_s = statistics.median(warmup_latencies_s)
    if median_s <= 0:
        return max_iters
    iters = math.ceil(target_seconds / median_s)
    return max(min_iters, min(max_iters, iters))


# ===========================================================================
# Workload envelope -> byte-size / GEMM / attention-bmm shape ranges
#
# hidden_size/ffn_embed_dim/n_head per model: tries flexllmgen's own
# get_opt_config() first (exact match for whatever's on PYTHONPATH in your
# real environment); falls back to this table of public OPT configs if
# flexllmgen isn't importable, so this script stays runnable standalone
# (e.g. on a bare GPU pod that doesn't have your flexllmgen checkout).
# ===========================================================================

_OPT_CONFIG_FALLBACK = {
    # name: (hidden_size, ffn_embed_dim, n_head)
    "opt-2.7b": (2560, 10240, 32),
    "opt-6.7b": (4096, 16384, 32),
    "opt-13b":  (5120, 20480, 40),
    "opt-30b":  (7168, 28672, 56),
}


def _opt_config(model_name: str) -> Tuple[int, int, int]:
    try:
        from flexllmgen.opt_config import get_opt_config
        c = get_opt_config(f"facebook/{model_name}")
        return c.hidden_size, c.ffn_embed_dim, c.n_head
    except ImportError:
        pass
    if model_name not in _OPT_CONFIG_FALLBACK:
        raise ValueError(
            f"No config for '{model_name}' -- flexllmgen isn't importable, "
            f"and it's not in the fallback table ({sorted(_OPT_CONFIG_FALLBACK)}). "
            f"Add it to _OPT_CONFIG_FALLBACK or run where flexllmgen is on PYTHONPATH."
        )
    return _OPT_CONFIG_FALLBACK[model_name]


@dataclass
class WorkloadRange:
    """The 'typical' envelope everything below is swept over."""
    models: List[str] = field(default_factory=lambda: ["opt-2.7b", "opt-6.7b", "opt-13b", "opt-30b"])
    seq_lens: List[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192])   # prompt_len + gen_len
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    offload_percents: List[float] = field(default_factory=lambda: [0, 25, 50, 75, 100])
    recompute_fractions: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5])  # of seq_len


def kv_transfer_byte_bounds(wr: WorkloadRange) -> Tuple[int, int]:
    """
    Min/max bytes a single get_bytes_to_load()/get_bytes_to_store() call
    would produce anywhere across the workload envelope -- i.e. the actual
    range a PCIe transfer or CPU-pinning call sees in this pipeline, not an
    arbitrary MB ladder. Mirrors those two functions' formulas exactly
    (bytes_per_token = hidden_size * 2, fp16/bf16).
    """
    sizes = []
    for model in wr.models:
        h1, _, _ = _opt_config(model)
        bytes_per_token = h1 * 2
        for seq_len in wr.seq_lens:
            for batch_size in wr.batch_sizes:
                for offload_pct in wr.offload_percents:
                    n_offloaded = batch_size - (batch_size * (100 - offload_pct)) // 100
                    for recomp_frac in wr.recompute_fractions:
                        recomp_len = int(recomp_frac * seq_len)
                        kv_load = (seq_len - recomp_len) * bytes_per_token * n_offloaded
                        recomp_load = recomp_len * bytes_per_token * batch_size
                        kv_store = batch_size * bytes_per_token  # 1 token/batch, steady-state decode
                        sizes.extend([kv_load, recomp_load, kv_store])
    sizes = [s for s in sizes if s > 0]
    return min(sizes), max(sizes)


def logspace_sizes(lo_bytes: int, hi_bytes: int, n: int = 14) -> List[int]:
    """n log-spaced byte sizes spanning [lo_bytes, hi_bytes], de-duplicated."""
    lo_bytes = max(1, lo_bytes)
    lo, hi = math.log(lo_bytes), math.log(hi_bytes)
    return sorted({int(round(math.exp(lo + (hi - lo) * i / (n - 1)))) for i in range(n)})


def gemm_shapes(wr: WorkloadRange, max_shapes: int = 60) -> List[Tuple[int, int, int]]:
    """
    (M, K, N) triples for the dense-GEMM bucket: QKVO projections
    (M=batch_size, K=N=hidden_size), FFN's two matmuls (K/N=hidden_size/
    ffn_embed_dim), and KV-recomputation's projection GEMMs
    (M=batch_size*recompute_len, K=N=hidden_size). Evenly subsampled to
    max_shapes if the full grid exceeds it, so growing WORKLOAD_RANGE later
    doesn't silently blow up runtime.
    """
    shapes = set()
    for model in wr.models:
        h1, h2, _ = _opt_config(model)
        for batch_size in wr.batch_sizes:
            shapes.add((batch_size, h1, h1))
            shapes.add((batch_size, h1, h2))
            shapes.add((batch_size, h2, h1))
            for seq_len in wr.seq_lens:
                for recomp_frac in wr.recompute_fractions:
                    if recomp_frac <= 0:
                        continue
                    recomp_len = max(1, int(recomp_frac * seq_len))
                    shapes.add((batch_size * recomp_len, h1, h1))
    return _subsample(sorted(shapes), max_shapes)


def attn_bmm_shapes(wr: WorkloadRange, max_shapes: int = 60) -> List[Tuple[int, int, int]]:
    """
    (batch, K, N) triples for the attention-bmm bucket: QK^T / Attn.V,
    batch=batch_size*n_head, K=head_dim, N=seq_len (FLOP count is the same
    either way round, so one sweep covers both bmms).
    """
    shapes = set()
    for model in wr.models:
        h1, _, nh = _opt_config(model)
        head_dim = h1 // nh
        for batch_size in wr.batch_sizes:
            for seq_len in wr.seq_lens:
                shapes.add((batch_size * nh, head_dim, seq_len))
    return _subsample(sorted(shapes), max_shapes)


def _subsample(items: List, max_n: int) -> List:
    if len(items) <= max_n:
        return items
    stride = len(items) / max_n
    return [items[int(i * stride)] for i in range(max_n)]


# ===========================================================================
# 1) CPU pinning + 2) PCIe H2D transfer
#    (measured together per size -- pin_memory() then copy_(), same as
#    general_copy()'s relay path: `src = src.pin_memory(); dst.copy_(src,
#    non_blocking=True)`)
# ===========================================================================

def make_noncontiguous(t: torch.Tensor) -> torch.Tensor:
    """Non-contiguous view with the same shape/values as `t`, standing in for
    the slices general_copy() actually sees: KV cache tiles arrive there as
    slices of a permuted cache tensor.

    Built as a stride-2 slice along the last dimension of a double-width
    buffer, rather than a transpose -- a transpose-based construction
    (padded.movedim(...)) degenerates back to a CONTIGUOUS tensor whenever
    one of the two dimensions is 1, since PyTorch leaves a size-1
    dimension's stride unconstrained for the purposes of is_contiguous().
    That's not a rare shape: bench_pin_and_transfer() computes
    rows = max(1, total_elems // cols), so every swept byte size below
    roughly 2*cols*elem_bytes (the smallest several points in the sweep,
    not an edge case) hits rows==1 and used to raise here. The stride-2
    slice stays non-contiguous regardless of row count, since the
    non-contiguity comes from the column stride, not from a row/column
    transpose that a size-1 row can quietly undo.
    """
    last = t.shape[-1]
    padded = torch.empty(t.shape[:-1] + (last * 2,), dtype=t.dtype)
    padded[..., ::2] = t
    view = padded[..., ::2]
    assert view.shape == t.shape and not view.is_contiguous()
    return view


def bench_pin_and_transfer(target_bytes: int, dtype=torch.float16, cols: int = 4096,
                            warmup: int = 5, target_seconds: float = 1.0,
                            min_iters: int = 10, max_iters: int = 2000) -> Dict:
    elem_bytes = dtype_size(dtype)
    total_elems = max(cols, int(target_bytes // elem_bytes))
    rows = max(1, total_elems // cols)
    shape = (rows, cols)
    actual_bytes = rows * cols * elem_bytes

    base = torch.randn(*shape, dtype=dtype)
    sources = {"contiguous": base.clone(), "non_contiguous": make_noncontiguous(base)}
    gpu_dst = torch.empty(shape, dtype=dtype, device="cuda")

    out = {"actual_bytes": actual_bytes}
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    def _one_iter(src):
        t0 = time.perf_counter()
        pinned = src.pin_memory()
        t1 = time.perf_counter()

        torch.cuda.synchronize()
        start_evt.record()
        gpu_dst.copy_(pinned, non_blocking=True)
        end_evt.record()
        torch.cuda.synchronize()
        return (t1 - t0), (start_evt.elapsed_time(end_evt) / 1000.0)

    for label, src in sources.items():
        warmup_total_s = []
        for _ in range(warmup):
            pin_s, copy_s = _one_iter(src)
            warmup_total_s.append(pin_s + copy_s)
        iters = _adaptive_iters(warmup_total_s, target_seconds, min_iters, max_iters)

        pin_lat, copy_lat = [], []
        for _ in range(iters):
            pin_s, copy_s = _one_iter(src)
            pin_lat.append(pin_s)
            copy_lat.append(copy_s)
        out[label] = {"pin_s": pin_lat, "copy_s": copy_lat, "n_iters": iters}
    return out


def run_pinning_and_pcie(wr: WorkloadRange, sizes: List[int], layout: str,
                          warmup: int, target_seconds: float, min_iters: int, max_iters: int,
                          cols: int = 4096):
    """Returns (pinning_points, pcie_points, detailed_rows). *_points are
    [(bytes, latency_s), ...] for the primary output CSVs, using `layout`;
    detailed_rows has both layouts' full percentile breakdown for inspection."""
    pinning_points, pcie_points, detailed = [], [], []
    print(f"\n{'bytes':>12} {'layout':>15} {'n_iters':>8} | {'pin median(ms)':>15} "
          f"{'copy median(ms)':>16} {'copy GB/s':>10} {'copy p99(ms)':>13}")
    for target_bytes in sizes:
        res = bench_pin_and_transfer(target_bytes, dtype=torch.float16, cols=cols,
                                      warmup=warmup, target_seconds=target_seconds,
                                      min_iters=min_iters, max_iters=max_iters)
        actual_bytes = res["actual_bytes"]
        for lb in ("contiguous", "non_contiguous"):
            n_iters = res[lb]["n_iters"]
            pin_t = summarize(res[lb]["pin_s"])
            copy_t = summarize(res[lb]["copy_s"])
            copy_median_s = copy_t.median_ms / 1000.0
            copy_gbps = (actual_bytes / copy_median_s) / 1e9 if copy_median_s > 0 else float("nan")
            print(f"{actual_bytes:>12} {lb:>15} {n_iters:>8} | {pin_t.median_ms:>15.5f} "
                  f"{copy_t.median_ms:>16.5f} {copy_gbps:>10.2f} {copy_t.p99_ms:>13.5f}")
            detailed.append({
                "bytes": actual_bytes, "layout": lb, "n_iters": n_iters,
                "pin_median_ms": pin_t.median_ms, "pin_p99_ms": pin_t.p99_ms,
                "copy_median_ms": copy_t.median_ms, "copy_p99_ms": copy_t.p99_ms,
                "copy_gbps": copy_gbps,
            })
            if lb == layout:
                pinning_points.append((float(actual_bytes), pin_t.median_ms / 1000.0))
                pcie_points.append((float(actual_bytes), copy_median_s))
    return pinning_points, pcie_points, detailed


# ===========================================================================
# 3) GPU compute: dense GEMM vs. attention bmm, achieved TFLOP/s
# ===========================================================================

def bench_gemm(M: int, K: int, N: int, dtype=torch.bfloat16, warmup: int = 10,
                target_seconds: float = 1.0, min_iters: int = 10, max_iters: int = 2000) -> Tuple[float, int]:
    """torch.mm(X[M,K], W[K,N]) -- the QKVO/FFN/recompute projection shape.
    Returns (achieved TFLOP/s [median over iters], n_iters used)."""
    X = torch.randn(M, K, dtype=dtype, device="cuda")
    W = torch.randn(K, N, dtype=dtype, device="cuda")
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    def _one_iter():
        start_evt.record()
        _ = torch.mm(X, W)
        end_evt.record()
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt) / 1000.0

    warmup_lat = [_one_iter() for _ in range(warmup)]
    iters = _adaptive_iters(warmup_lat, target_seconds, min_iters, max_iters)
    lat_s = [_one_iter() for _ in range(iters)]

    median_s = statistics.median(lat_s)
    tflops = (2 * M * K * N / median_s) / 1e12 if median_s > 0 else float("nan")
    return tflops, iters


def bench_attn_bmm(batch: int, K: int, N: int, dtype=torch.bfloat16, warmup: int = 10,
                    target_seconds: float = 1.0, min_iters: int = 10, max_iters: int = 2000) -> Tuple[float, int]:
    """torch.bmm(A[batch,1,K], B[batch,K,N]) -- the QK^T / Attn.V shape
    (batch = batch_size*n_head, M=1 since decode processes one token).
    Returns (achieved TFLOP/s [median over iters], n_iters used)."""
    A = torch.randn(batch, 1, K, dtype=dtype, device="cuda")
    B = torch.randn(batch, K, N, dtype=dtype, device="cuda")
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    def _one_iter():
        start_evt.record()
        _ = torch.bmm(A, B)
        end_evt.record()
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt) / 1000.0

    warmup_lat = [_one_iter() for _ in range(warmup)]
    iters = _adaptive_iters(warmup_lat, target_seconds, min_iters, max_iters)
    lat_s = [_one_iter() for _ in range(iters)]

    median_s = statistics.median(lat_s)
    tflops = (2 * batch * K * N / median_s) / 1e12 if median_s > 0 else float("nan")
    return tflops, iters


def run_gpu_compute(wr: WorkloadRange, max_shapes: int, warmup: int,
                     target_seconds: float, min_iters: int, max_iters: int):
    rows = []
    print(f"\n-- GEMM shapes (QKVO/FFN/recompute projections) --")
    print(f"{'M':>7} {'K':>7} {'N':>7} {'n_iters':>8} | {'TFLOP/s':>9}")
    gemm_tflops_list = []
    for (M, K, N) in gemm_shapes(wr, max_shapes):
        t, n_iters = bench_gemm(M, K, N, warmup=warmup, target_seconds=target_seconds,
                                 min_iters=min_iters, max_iters=max_iters)
        print(f"{M:>7} {K:>7} {N:>7} {n_iters:>8} | {t:>9.2f}")
        gemm_tflops_list.append(t)
        rows.append({"op": "gemm", "M": M, "K": K, "N": N, "n_iters": n_iters, "tflops": t})

    print(f"\n-- Attention bmm shapes (QK^T / Attn.V) --")
    print(f"{'batch':>7} {'head_dim':>9} {'seq_len':>8} {'n_iters':>8} | {'TFLOP/s':>9}")
    attn_tflops_list = []
    for (batch, K, N) in attn_bmm_shapes(wr, max_shapes):
        t, n_iters = bench_attn_bmm(batch, K, N, warmup=warmup, target_seconds=target_seconds,
                                     min_iters=min_iters, max_iters=max_iters)
        print(f"{batch:>7} {K:>9} {N:>8} {n_iters:>8} | {t:>9.2f}")
        attn_tflops_list.append(t)
        rows.append({"op": "attn_bmm", "M": batch, "K": K, "N": N, "n_iters": n_iters, "tflops": t})

    gemm_med = statistics.median(gemm_tflops_list)
    attn_med = statistics.median(attn_tflops_list)
    print(f"\n{'='*60}")
    print(f"GEMM achieved TFLOP/s : median={gemm_med:.2f}  "
          f"(min={min(gemm_tflops_list):.2f}, max={max(gemm_tflops_list):.2f}, n={len(gemm_tflops_list)})")
    print(f"Attn achieved TFLOP/s : median={attn_med:.2f}  "
          f"(min={min(attn_tflops_list):.2f}, max={max(attn_tflops_list):.2f}, n={len(attn_tflops_list)})")
    print(f"If GEMM/attn TFLOP/s vary a lot across shapes above, the flat "
          f"median may not be representative -- eyeball the detailed CSV "
          f"before trusting it.")
    print(f"\nSuggested kv_bench_predictor.py flags:")
    print(f"  --gemm-tflops {gemm_med:.2f} --attn-tflops {attn_med:.2f}")
    print(f"{'='*60}")
    return rows, gemm_med, attn_med


# ===========================================================================
# CSV output
# ===========================================================================

def write_latency_table(points: List[Tuple[float, float]], path: str):
    """bytes,latency_s -- exactly the format kv_bench_predictor.load_latency_table() expects."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bytes", "latency_s"])
        for b, s in sorted(points):
            writer.writerow([b, s])
    print(f"Wrote {len(points)} point(s) to {path}")


def write_detailed_csv(rows: List[Dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} row(s) to {path}")


# ===========================================================================
# CLI
# ===========================================================================

def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", choices=["pinning", "pcie", "gpu-compute", "all"], default="all")
    p.add_argument("--out-prefix", type=str, default="kv_microbench",
                    help="output files: <prefix>_pinning.csv, <prefix>_pcie.csv, "
                         "<prefix>_gpu_compute.csv")

    p.add_argument("--models", nargs="+", default=WorkloadRange().models,
                    help="model names as used in _OPT_CONFIG_FALLBACK / "
                         "get_opt_config('facebook/<name>')")
    p.add_argument("--seq-lens", type=int, nargs="+", default=WorkloadRange().seq_lens,
                    help="prompt_len + gen_len values to sweep, tokens")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=WorkloadRange().batch_sizes)
    p.add_argument("--offload-percents", type=float, nargs="+", default=WorkloadRange().offload_percents)
    p.add_argument("--recompute-fractions", type=float, nargs="+", default=WorkloadRange().recompute_fractions,
                    help="recompute_len as a fraction of seq_len")

    p.add_argument("--n-sizes", type=int, default=14,
                    help="number of log-spaced byte sizes to sweep for pinning/pcie")
    p.add_argument("--layout", choices=["contiguous", "non_contiguous"], default="non_contiguous",
                    help="which layout's numbers become the primary pinning/pcie output CSV "
                         "(both are always measured and kept in the detailed printout) -- "
                         "non_contiguous matches how KV cache tiles actually arrive at "
                         "general_copy() in production")
    p.add_argument("--xfer-cols", type=int, default=4096)
    p.add_argument("--xfer-warmup", type=int, default=5,
                    help="fixed warmup iterations before timing starts (per size/layout)")
    p.add_argument("--xfer-target-seconds", type=float, default=1.0,
                    help="run enough timed iterations to cover at least this much total "
                         "wall time, estimated from the warmup median -- see _adaptive_iters()")
    p.add_argument("--xfer-min-iters", type=int, default=10)
    p.add_argument("--xfer-max-iters", type=int, default=2000)

    p.add_argument("--max-gemm-shapes", type=int, default=60)
    p.add_argument("--max-attn-shapes", type=int, default=60)
    p.add_argument("--compute-warmup", type=int, default=10)
    p.add_argument("--compute-target-seconds", type=float, default=1.0)
    p.add_argument("--compute-min-iters", type=int, default=10)
    p.add_argument("--compute-max-iters", type=int, default=2000)
    return p


def main():
    args = build_argparser().parse_args()
    print_env_info()
    if not torch.cuda.is_available():
        print("\nERROR: no CUDA GPU visible. Every benchmark here (pin_memory(), "
              "H2D copy, matmul throughput) requires one -- run this on the GPU "
              "pod, not locally.")
        raise SystemExit(1)

    wr = WorkloadRange(
        models=args.models, seq_lens=args.seq_lens, batch_sizes=args.batch_sizes,
        offload_percents=args.offload_percents, recompute_fractions=args.recompute_fractions,
    )

    if args.bench in ("pinning", "pcie", "all"):
        lo, hi = kv_transfer_byte_bounds(wr)
        sizes = logspace_sizes(lo, hi, n=args.n_sizes)
        print(f"\nWorkload-derived transfer size range: {lo:,} - {hi:,} bytes "
              f"({lo/1e6:.3f} MB - {hi/1e9:.3f} GB), swept at {len(sizes)} log-spaced points")
        pinning_points, pcie_points, detailed = run_pinning_and_pcie(
            wr, sizes, args.layout, args.xfer_warmup,
            args.xfer_target_seconds, args.xfer_min_iters, args.xfer_max_iters,
            cols=args.xfer_cols,
        )
        if args.bench in ("pinning", "all"):
            write_latency_table(pinning_points, f"{args.out_prefix}_pinning.csv")
        if args.bench in ("pcie", "all"):
            write_latency_table(pcie_points, f"{args.out_prefix}_pcie.csv")
        write_detailed_csv(detailed, f"{args.out_prefix}_pinning_pcie_detail.csv")

    if args.bench in ("gpu-compute", "all"):
        rows, gemm_med, attn_med = run_gpu_compute(
            wr, max(args.max_gemm_shapes, args.max_attn_shapes), args.compute_warmup,
            args.compute_target_seconds, args.compute_min_iters, args.compute_max_iters,
        )
        write_detailed_csv(rows, f"{args.out_prefix}_gpu_compute.csv")
        with open(f"{args.out_prefix}_gpu_compute_summary.txt", "w") as f:
            f.write(f"--gemm-tflops {gemm_med:.2f} --attn-tflops {attn_med:.2f}\n")
        print(f"Wrote summary to {args.out_prefix}_gpu_compute_summary.txt")


if __name__ == "__main__":
    main()