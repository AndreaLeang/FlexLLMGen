"""
Combined cost model: KV-cache offload % + recomputation sweep, restricted to
policies where weights and activations always stay on GPU.

Merges two things from cost_model.py:
  1. solve_lp's placement search (here fixed to wg=hg=100%, only cache moves)
  2. get_optimal_split_point's recompute-vs-transfer tradeoff

Hardware constants are configurable via HardwareConfig, not hardcoded.
"""

import dataclasses
import math
from typing import List, Optional, Tuple, Dict

GB = 1024 ** 3
T = 1e12


@dataclasses.dataclass
class HardwareConfig:
    """GPU hardware constants. Rates are in bytes/s or FLOP/s."""
    gpu_matmul_flops: float = 312 * T    # peak GPU matmul throughput (e.g. A100 bf16)
    cpu_gpu_bandwidth: float = 16 * GB   # PCIe/NVLink achievable bandwidth, bytes/s
    dtype_bytes: int = 2                 # bytes/element (fp16/bf16=2, fp32=4)
    gpu_mem_bytes: float = 24 * GB       # usable GPU memory budget


@dataclasses.dataclass
class ModelConfig:
    """Model architecture + workload shape."""
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_dim: int = 16384
    num_heads: int = 32
    vocab_size: int = 50272  # OPT default; only galactica-30b overrides this
    prompt_len: int = 512
    gen_len: int = 32


# ---------------------------------------------------------------------------
# Pre-defined hardware configurations
# ---------------------------------------------------------------------------
# FP16 tensor-core throughput is dense (no structured-sparsity) unless noted.
# Sources: NVIDIA A100/H100/L40 datasheets; RTX PRO 6000 Blackwell figure is
# estimated (see note below) -- verify against your own card if precision
# matters, e.g. via the same profiling approach as your int8_bench work.
GPU_PRESETS = {
    # (gpu_matmul_flops FP16 dense, gpu_mem_bytes)
    "a100-80gb": (312 * T, 80 * GB),
    "h100-80gb": (989.5 * T, 80 * GB),   # 1979 TFLOPS w/ sparsity, halved for dense
    "l40": (181.05 * T, 48 * GB),
    # RTX PRO 6000 Blackwell (96GB): NVIDIA/OEM datasheets only publish a
    # single "peak" FP16 figure (1 PFLOPS) with no explicit dense/sparse
    # split. Blackwell marketing conventionally quotes the sparse number as
    # "peak" (matches the FP4/FP8/FP16 2x doubling pattern in the spec
    # sheet), so this halves it to estimate dense throughput. Least certain
    # of the four -- worth verifying empirically on your actual card.
    "rtx-pro-6000": (1000 * T, 96 * GB),
}

PCIE_PRESETS = {
    "16gb/s": 16 * GB,
    "32gb/s": 32 * GB,
    "64gb/s": 64 * GB,
}

# From FlexLLMGen's opt_config.py get_opt_config (num_layers, hidden_size,
# ffn_dim, num_heads). max_seq_len and vocab_size are not modeled here.
MODEL_PRESETS = {
    "opt-1.3b": dict(num_layers=24, hidden_size=2048, ffn_dim=2048 * 4, num_heads=32),
    "opt-2.7b": dict(num_layers=32, hidden_size=2560, ffn_dim=2560 * 4, num_heads=32),
    "opt-6.7b": dict(num_layers=32, hidden_size=4096, ffn_dim=4096 * 4, num_heads=32),
    "opt-13b": dict(num_layers=40, hidden_size=5120, ffn_dim=5120 * 4, num_heads=40),
    "opt-30b": dict(num_layers=48, hidden_size=7168, ffn_dim=7168 * 4, num_heads=56),
    "galactica-30b": dict(num_layers=48, hidden_size=7168, ffn_dim=7168 * 4, num_heads=56, vocab_size=50000),
    "opt-66b": dict(num_layers=64, hidden_size=9216, ffn_dim=9216 * 4, num_heads=72),
    "opt-175b": dict(num_layers=96, hidden_size=12288, ffn_dim=12288 * 4, num_heads=96),
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def kv_cache_bytes(gpu_batch_size, num_tokens, hidden_size, dtype_bytes):
    """Bytes to store K and V for num_tokens tokens, one layer."""
    return 2 * gpu_batch_size * num_tokens * hidden_size * dtype_bytes


def weight_bytes(hidden_size, ffn_dim, num_layers, vocab_size, dtype_bytes):
    """
    Total model weight size: transformer blocks (QKVO + 2-layer FFN per
    layer) plus the token embedding table. Assumes input/output embeddings
    are weight-tied (OPT's default) -- if your model doesn't tie them,
    double the embedding term.
    """
    layer_elems = 4 * hidden_size ** 2 + 2 * hidden_size * ffn_dim
    embed_elems = vocab_size * hidden_size
    return (layer_elems * num_layers + embed_elems) * dtype_bytes


def decode_activation_bytes(gpu_batch_size, hidden_size, ffn_dim, num_heads,
                             prompt_len, gen_len, gpu_cache_frac, dtype_bytes):
    """
    Per-layer transient activation/working-buffer footprint during one decode
    step: QKV projections, attention-score buffer (scales with num_heads and
    sequence length), attention output, and MLP intermediates.

    Mirrors solve_lp's `interg` term (element counts assume fp16 there;
    rescaled by dtype_bytes/2 here to stay configurable).
    """
    gbs, h1, h2, nh = gpu_batch_size, hidden_size, ffn_dim, num_heads
    seq = prompt_len + gen_len
    cg = gpu_cache_frac

    elems_fp16_equiv = (
        8 * gbs * h1                                          # QKV projections
        + gbs * (2 * h1 + 2 * seq * h1 + 2 * nh * seq) * cg    # attention score buffer
        + gbs * (2 * seq * h1 + 2 * h1) * cg                   # attention output buffer
        + 4 * gbs * h1                                         # output projection
        + 2 * gbs * h2                                         # MLP up-projection
        + 2 * gbs * h1                                         # MLP down-projection
    )
    return elems_fp16_equiv * (dtype_bytes / 2)


def optimal_recompute_len(hw: HardwareConfig, gpu_batch_size, hidden_size,
                           num_cpu_tokens, step=1) -> Tuple[int, float]:
    """
    Reproduces the prior work's get_optimal_split_point() exactly: for a
    given count of CPU-resident cache tokens, find how many to recompute
    on GPU vs. transfer from CPU, by minimizing

        t_activation_fetch + max(t_recompute_compute, t_transfer)

    This search deliberately does NOT know about t_mha (GPU-resident
    attention compute) -- matching the prior work's function signature,
    which has no such concept either. The interaction between this
    result and t_mha is handled separately, at the point where the total
    per-layer latency is assembled (see layer_decode_breakdown()), not
    here in the split-point search itself.

    Uses HardwareConfig instead of hardcoded v_com / v_gpu.
    """
    v_com = hw.cpu_gpu_bandwidth
    v_gpu = hw.gpu_matmul_flops
    dtype_bytes = hw.dtype_bytes

    best_t, best_len = float("inf"), 0
    for recompute_len in range(0, num_cpu_tokens + 1, step):
        t_activation_fetch = _recompute_activation_fetch_time(hw, gpu_batch_size, hidden_size, recompute_len)
        t_recompute_compute = _recompute_compute_time(hw, gpu_batch_size, hidden_size, recompute_len)

        remaining_tokens = num_cpu_tokens - recompute_len
        transfer_bytes = kv_cache_bytes(gpu_batch_size, remaining_tokens, hidden_size, dtype_bytes)
        t_transfer = transfer_bytes / v_com

        t_total = t_activation_fetch + max(t_recompute_compute, t_transfer)

        if t_total < best_t:
            best_t, best_len = t_total, recompute_len

    return best_len, best_t


def _recompute_compute_time(hw, gpu_batch_size, hidden_size, recompute_len):
    """
    GPU compute to regenerate K,V for `recompute_len` tokens: the two
    projection GEMMs, matching the prior work's get_optimal_split_point
    exactly (N_recompute_flops = 4 * gpu_batch_size * recompute_len *
    input_dim ** 2). Deliberately does NOT add an attention-op term for
    those tokens -- an earlier version of this function did, but that
    diverges from the prior work being reproduced here.
    """
    proj_flops = 4 * gpu_batch_size * recompute_len * hidden_size * hidden_size
    return proj_flops / hw.gpu_matmul_flops


def _recompute_activation_fetch_time(hw, gpu_batch_size, hidden_size, recompute_len):
    """
    Time to fetch, from CPU, the stored per-token hidden states needed to
    recompute K,V for `recompute_len` past tokens. This is memory_recompute_
    activations / v_com in the original snippet -- one hidden-state vector
    per recomputed token (not doubled like K,V), and it must complete before
    the recompute GEMM can start, so it's a serial prefix, not something
    that overlaps with the recompute-vs-transfer race.
    """
    activation_bytes = gpu_batch_size * recompute_len * hidden_size * hw.dtype_bytes
    return activation_bytes / hw.cpu_gpu_bandwidth


def _cache_time_for_len(hw, gpu_batch_size, hidden_size, num_cpu_tokens, recompute_len):
    recompute_len = min(recompute_len, num_cpu_tokens)
    remaining = num_cpu_tokens - recompute_len
    t_activation_fetch = _recompute_activation_fetch_time(hw, gpu_batch_size, hidden_size, recompute_len)
    t_recompute_compute = _recompute_compute_time(hw, gpu_batch_size, hidden_size, recompute_len)
    transfer_bytes = kv_cache_bytes(gpu_batch_size, remaining, hidden_size, hw.dtype_bytes)
    t_transfer = transfer_bytes / hw.cpu_gpu_bandwidth
    return t_activation_fetch + max(t_recompute_compute, t_transfer)


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(model: ModelConfig, hw: HardwareConfig, gpu_batch_size,
                     offload_frac, total_prompts, recompute_len: Optional[int] = None) -> Dict:
    """
    Evaluate one (gpu_batch_size, offload_frac[, recompute_len]) policy.
    Weights and activations are always GPU-resident; only the KV cache is
    split GPU/CPU, with an optional recompute strategy on the CPU share.

    num_gpu_batches (nb) is fixed at 1: bls = gpu_batch_size, matching the
    reference's structure exactly (its batch_size is used as-is, repeated
    num_batches times sequentially -- no round-expansion concept). An
    earlier version auto-derived nb = ceil(total_prompts / gbs) to cover the
    whole job in one round, reasoned as harmless since nothing in the model
    then depended on round size. That stopped being true once
    free_mem_required (below) was added: it scales with bls as a per-round
    staging buffer, so larger bls now has a real, unbounded memory cost with
    no offsetting benefit -- nb=1 is the model-consistent choice. The
    remainder-handling fix (point 3 below) is kept even though bls can no
    longer exceed total_prompts on its own, since a user could still pass a
    gpu_batch_size larger than total_prompts directly.

    Cross-checked against a reference get_available_offloadings()
    implementation. Corrections from that check, on top of the earlier
    bls/peak-token fixes:

    1. offload_frac is PER-SEQUENCE, not per-token: it selects what
       fraction of the bls sequences in a round are CPU-resident (their
       entire cache), matching the reference's whole-sequence offloading
       granularity -- not a fraction of every sequence's own tokens.
    2. GPU-resident cache is CUMULATIVE across the whole job, not freed
       after each round -- matches the reference's actual_kv_cache_bytes =
       (on_gpu_fraction) * num_batches * cache_bytes.
    3. The reference's num_batches = total_prompts // bls (floor) silently
       drops the remainder when bls doesn't evenly divide total_prompts.
       Fixed below by explicitly accounting for the remainder rather than
       dropping it (matters whenever gpu_batch_size > total_prompts).
    4. free_mem_required, matching the reference's exact formula: an extra
       staging-buffer term based on the FULL (un-split) per-round cache
       size at bls, separate from the actual on-GPU cache after offloading.

    recompute_len: tokens/layer to recompute rather than transfer, for the
                   CPU-resident (offloaded) sequences specifically, during
                   THEIR OWN autoregressive decoding. None -> auto-optimized.
    total_prompts: total prompts in the job. Drives both the auto-derived nb
                   and the cumulative memory check.
    """
    h1, h2, l, nh = model.hidden_size, model.ffn_dim, model.num_layers, model.num_heads
    s, n = model.prompt_len, model.gen_len

    # num_gpu_batches (nb) is fixed at 1: bls = gpu_batch_size exactly, matching
    # the reference's structure (no round-expansion concept at all -- its
    # batch_size is used as-is, many times sequentially via num_batches).
    # An earlier version auto-derived nb = ceil(total_prompts / gbs) to cover
    # the whole job in one round, reasoned as harmless since nothing in this
    # model rewarded or penalized round size. That's no longer true: adding
    # free_mem_required below (which scales with bls, as a per-round staging
    # buffer) means larger bls now has a real, unbounded cost with no
    # offsetting benefit -- so the model-consistent choice is always nb=1.
    num_gpu_batches = 1
    bls = gpu_batch_size * num_gpu_batches

    # avg_tokens: average cache length across the decode process -> feeds
    # the TIME formulas, matching solve_lp's dtocg/compg use of (s + n/2).
    # peak_tokens: the FULL final cache length -> feeds the MEMORY formulas,
    # matching solve_lp's gpu_home_p/gpu_home_g use of (s + n).
    avg_tokens = s + n / 2
    peak_tokens = s + n

    num_gpu_seqs = bls * (1 - offload_frac)   # GPU-resident sequences, this round
    num_cpu_seqs = bls * offload_frac         # CPU-resident (offloaded) sequences, this round

    # ---- GPU memory feasibility (cumulative across the whole job) ----
    w_bytes = weight_bytes(h1, h2, l, model.vocab_size, hw.dtype_bytes)
    # Working buffers: transient, current round only -> gbs-scaled (matches
    # solve_lp's interp/interg; only one micro-batch's activations are live
    # on the compute engines at a time).
    act_bytes = decode_activation_bytes(gpu_batch_size, h1, h2, nh, s, n,
                                         1 - offload_frac, hw.dtype_bytes) * l
    # KV cache: cumulative across every prompt actually processed so far,
    # including a partial final round (fixed: no longer silently dropped).
    num_complete_rounds = total_prompts // bls
    remainder_prompts = total_prompts % bls
    cumulative_on_gpu_prompts = num_gpu_seqs * num_complete_rounds + (1 - offload_frac) * remainder_prompts
    gpu_cache_bytes = kv_cache_bytes(cumulative_on_gpu_prompts, peak_tokens, h1, hw.dtype_bytes) * l

    # ---- Extra headroom, matching the reference's free_mem_required exactly ----
    # Based on the FULL (un-split) per-round cache size at peak_tokens -- not
    # the actual on-GPU cache after offloading, which is a separate quantity.
    full_round_cache_bytes = kv_cache_bytes(bls, peak_tokens, h1, hw.dtype_bytes) * l
    free_mem_required = (full_round_cache_bytes / 2) * (s / peak_tokens) * (s / h1) * (nh + 1) / nh
    if offload_frac != 0:
        free_mem_required += full_round_cache_bytes

    # ---- Hidden state buffer, matching the reference's total_hidden_bytes ----
    # One hidden-state vector per token per layer, single copy (not doubled
    # like K,V), for the current round's bls sequences at peak_tokens.
    hidden_bytes = bls * peak_tokens * h1 * l * hw.dtype_bytes

    total_gpu_bytes = w_bytes + act_bytes + gpu_cache_bytes + free_mem_required + hidden_bytes
    feasible = total_gpu_bytes <= hw.gpu_mem_bytes

    # ---- Base per-layer decode compute, for ONE round ----
    # QKVO + FFN projections happen for every sequence regardless of cache
    # location (they don't depend on where the cache lives).
    base_flops = bls * (8 * h1 ** 2 + 4 * h1 * h2)
    # Attention compute for GPU-resident sequences, over their FULL average
    # cache (no longer reduced by offload_frac -- that fraction now selects
    # WHICH sequences, not a per-sequence token split).
    base_flops += 4 * num_gpu_seqs * avg_tokens * h1
    t_compute = base_flops / hw.gpu_matmul_flops

    # ---- CPU-resident sequences: recompute-vs-transfer tradeoff, over their
    # OWN full average cache length (avg_tokens), not a fraction of it ----
    cpu_seqs_int = int(round(num_cpu_seqs))
    avg_tokens_int = int(round(avg_tokens))
    if cpu_seqs_int > 0:
        if recompute_len is None:
            recompute_len, t_cpu_seqs = optimal_recompute_len(
                hw, cpu_seqs_int, h1, avg_tokens_int)
        else:
            t_cpu_seqs = _cache_time_for_len(
                hw, cpu_seqs_int, h1, avg_tokens_int, recompute_len)
    else:
        recompute_len, t_cpu_seqs = 0, 0.0

    # GPU-resident sequences' attention compute and CPU-resident sequences'
    # transfer-or-recompute handling run on separate engines/sequences and
    # can overlap; only additive where they'd share the same tensor cores
    # (t_compute already folds in GPU-resident attention; t_cpu_seqs' own
    # recompute share was accounted for with base_compute_time=0.0 above
    # since it's a disjoint set of sequences, not sharing t_compute's work).
    t_layer_decode = max(t_compute, t_cpu_seqs)
    t_decode_total = t_layer_decode * l * (n - 1)

    # ---- Prefill: compute-bound, cache freshly generated, no transfer/recompute
    # needed. bls-scaled (whole round). ----
    prefill_flops_per_layer = (
        bls * (8 * s * h1 ** 2 + 4 * s * h1 * h2)
        + 4 * bls * s ** 2 * h1
    )
    t_prefill_total = (prefill_flops_per_layer / hw.gpu_matmul_flops) * l

    total_time = t_prefill_total + t_decode_total  # time for ONE round (bls sequences)
    throughput = (bls * n) / total_time if total_time > 0 else 0.0

    num_rounds = math.ceil(total_prompts / bls)
    wall_time_s = num_rounds * total_time
    real_tokens = total_prompts * n
    utilization = total_prompts / (num_rounds * bls)

    return {
        "gpu_batch_size": gpu_batch_size,
        "num_gpu_batches": num_gpu_batches,
        "bls": bls,
        "offload_frac": offload_frac,
        "recompute_len": recompute_len,
        "feasible": feasible,
        "gpu_mem_used_gb": total_gpu_bytes / GB,
        "gpu_cache_gb": gpu_cache_bytes / GB,
        "free_mem_required_gb": free_mem_required / GB,
        "hidden_bytes_gb": hidden_bytes / GB,
        "num_complete_rounds": num_complete_rounds,
        "t_prefill_s": t_prefill_total,
        "t_decode_s": t_decode_total,
        "total_time_s": total_time,                # one round (bls sequences)
        "throughput_tok_s": throughput,             # one round, steady-state rate
        "total_prompts": total_prompts,
        "num_rounds": num_rounds,
        "batch_utilization": utilization,
        "wall_time_s": wall_time_s,
        "effective_throughput_tok_s": real_tokens / wall_time_s if wall_time_s > 0 else 0.0,
    }


def layer_decode_breakdown(
    model: ModelConfig,
    hw: HardwareConfig,
    gpu_batch_size: int,
    offload_frac: float,
    recompute_len: Optional[int] = None,
) -> Tuple[Dict[str, float], int]:
    """
    Per-layer, per-decode-step latency breakdown for ONE representative
    middle layer -- the baseline_model analog of
    kv_schedule_optimization.layer_prediction()'s component_breakdown.

    Segment keys match gt_vs_estimator.py's EST_SEGMENT_NAMES exactly, so
    the two estimators can be compared side by side in the same CSV columns:
        "PinnedMemory CPU (phase1)", "PinnedMemory CPU (phase2)",
        "Recompute Load", "Recompute CUDA", "MHA CUDA",
        "KVCache Load K", "KVCache Load V"

    Two-resource model (IMPORTANT)
    -------------------------------
    This is the "perfect overlap" model: two independent HARDWARE RESOURCES
    each process their own queue of work back-to-back, and the two
    resources run concurrently with each other:

      GPU tensor cores : t_mha (attention for the GPU-resident share --
                         no dependency, can start immediately), THEN
                         t_recompute_cuda (regenerating K/V for the
                         CPU-resident share's recomputed tokens -- can only
                         START once the activation fetch below is done).
      PCIe / DMA       : t_recompute_load (fetching the activations needed
                         to recompute), THEN t_transfer (moving the
                         CPU-resident share's remaining, non-recomputed
                         tokens' K/V). Both are PCIe traffic, so they share
                         the one link and run sequentially on it.

    GPU's own finish time is max(t_mha, t_recompute_load) + t_recompute_cuda:
    t_mha runs immediately, but t_recompute_cuda has to wait for whichever
    is later -- "GPU free" or "fetch done" -- before it can start. PCIe's
    own finish time is simply t_recompute_load + t_transfer (strictly
    sequential on one link). The layer's actual latency is whichever
    resource finishes last; the other resource had slack.

    This replaces an earlier version of this function that grouped by
    SEQUENCE (GPU-resident vs. CPU-resident) rather than by RESOURCE, and
    treated t_mha and t_recompute_cuda as independent even though they
    both need the same tensor cores. Reproduces the prior work's
    get_optimal_split_point() for choosing recompute_len itself (see
    optimal_recompute_len()) -- that search intentionally has no notion of
    t_mha, matching the prior work exactly; this function is where t_mha
    gets folded back in, at the point of computing the actual total.

    Segment reporting: whichever resource is NOT the bottleneck contributes
    0.0 to every one of its segments -- it finished with slack, so none of
    its legs extend the critical path. This means, unlike an earlier
    version of this function, two segments can legitimately be non-zero
    together now: if GPU is the bottleneck, "Recompute CUDA" is always
    reported (it's unconditionally part of GPU's own chain) alongside
    whichever of "MHA CUDA" / "Recompute Load" was the longer of the two
    things it had to wait on; if PCIe is the bottleneck, "Recompute Load"
    and "KVCache Load K/V" are reported together. sum(segments.values())
    still always equals the true max()-based total latency -- see the
    inline comments below for why each branch preserves that.

    No separate pinned-memory staging phase is modeled, so both
    PinnedMemory segments are always 0.0 -- any host-side copy cost is
    folded directly into the PCIe transfer bandwidth. K and V are an equal
    split of kv_cache_bytes() (which counts both together).

    Parameters
    ----------
    gpu_batch_size : sequences in this layer's micro-batch (gbs).
    offload_frac   : fraction (0-1) of those sequences that are CPU-resident.
                     Use achievable_offload_fracs(gbs) for valid values.
    recompute_len  : tokens/layer to recompute for the CPU-resident share.
                     None -> auto-optimized via optimal_recompute_len(),
                     which reproduces the prior work's get_optimal_split_point.

    Returns
    -------
    (segments, recompute_len_used)
      segments           : Dict[str, float] segment_name -> seconds.
      recompute_len_used : the recompute length actually used (echoes the
                            input if given, else the auto-optimized value).
    """
    h1 = model.hidden_size
    s, n = model.prompt_len, model.gen_len
    avg_tokens = s + n / 2

    num_gpu_seqs = gpu_batch_size * (1 - offload_frac)
    num_cpu_seqs = gpu_batch_size * offload_frac

    # MHA CUDA: attention compute for the GPU-resident share only, over
    # their average cache length. Excludes QKVO/FFN projections, matching
    # kv_schedule_optimization's break_MHA=True / layer_type="MHA" convention.
    mha_flops = 4 * num_gpu_seqs * avg_tokens * h1
    t_mha = mha_flops / hw.gpu_matmul_flops

    segments = {
        "PinnedMemory CPU (phase1)": 0.0,
        "PinnedMemory CPU (phase2)": 0.0,
        "Recompute Load": 0.0,
        "Recompute CUDA": 0.0,
        "MHA CUDA": 0.0,
        "KVCache Load K": 0.0,
        "KVCache Load V": 0.0,
    }

    cpu_seqs_int = int(round(num_cpu_seqs))
    if cpu_seqs_int == 0:
        # No offloaded sequences this layer -- MHA compute is the only
        # thing running, unconditionally.
        segments["MHA CUDA"] = t_mha
        return segments, 0

    avg_tokens_int = int(round(avg_tokens))
    if recompute_len is None:
        recompute_len, _ = optimal_recompute_len(hw, cpu_seqs_int, h1, avg_tokens_int)
    recompute_len = max(0, min(recompute_len, avg_tokens_int))
    remaining = avg_tokens_int - recompute_len

    t_recompute_load = _recompute_activation_fetch_time(hw, cpu_seqs_int, h1, recompute_len)
    t_recompute_cuda = _recompute_compute_time(hw, cpu_seqs_int, h1, recompute_len)
    transfer_bytes = kv_cache_bytes(cpu_seqs_int, remaining, h1, hw.dtype_bytes)
    t_transfer = transfer_bytes / hw.cpu_gpu_bandwidth

    # GPU's own finish time: t_mha runs immediately (no dependency);
    # t_recompute_cuda waits for max(t_mha, t_recompute_load) -- whichever
    # of "GPU is free" or "the fetch is done" comes later -- then takes
    # t_recompute_cuda more.
    gpu_wait = max(t_mha, t_recompute_load)
    gpu_side = gpu_wait + t_recompute_cuda
    # PCIe's own finish time: fetch, then transfer, sequential on one link.
    pcie_side = t_recompute_load + t_transfer

    if gpu_side >= pcie_side:
        # GPU is the bottleneck. Report whichever of t_mha / t_recompute_load
        # was the longer of the two things t_recompute_cuda had to wait on
        # (matches gpu_wait exactly), plus t_recompute_cuda itself, which is
        # unconditionally part of GPU's chain. These two sum to gpu_side.
        if t_mha >= t_recompute_load:
            segments["MHA CUDA"] = t_mha
        else:
            segments["Recompute Load"] = t_recompute_load
        segments["Recompute CUDA"] = t_recompute_cuda
        # KVCache Load K/V stay 0.0 -- PCIe finished with slack.
    else:
        # PCIe is the bottleneck. Both of its legs are unconditionally part
        # of its chain (fetch then transfer), summing to pcie_side exactly.
        segments["Recompute Load"] = t_recompute_load
        segments["KVCache Load K"] = t_transfer / 2
        segments["KVCache Load V"] = t_transfer / 2
        # MHA CUDA / Recompute CUDA stay 0.0 -- GPU finished with slack.

    return segments, recompute_len


def achievable_offload_fracs(gpu_batch_size: int) -> List[float]:
    """
    One prompt is the smallest unit of offload within a batch -- you can't
    offload a fraction of a single sequence's cache. So for a given
    gpu_batch_size (gbs), the only achievable offload fractions are k/gbs
    for integer k in [0, gbs] (matches the reference's
    batch_size_to_distinct_offloadings dict, generalized to any gbs instead
    of a fixed lookup table). gbs=1 -> [0.0, 1.0] only; gbs=4 -> [0.0, 0.25,
    0.5, 0.75, 1.0]; etc.
    """
    return [k / gpu_batch_size for k in range(gpu_batch_size + 1)]


def sweep_gpu_resident_policies(
    model: ModelConfig,
    hw: HardwareConfig,
    gpu_batch_sizes: List[int],
    total_prompts: int,
    recompute_lens: Optional[List[int]] = None,
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Grid search over gpu_batch_size x offload_frac x recompute_len, restricted
    to policies with weights + activations fully GPU-resident. num_gpu_batches
    is auto-derived inside evaluate_policy from total_prompts and
    gpu_batch_size (see its docstring) -- no longer swept explicitly.

    offload_frac is no longer a free-form list: for each gbs, only the
    fractions achievable at whole-prompt granularity (k/gbs) are swept (see
    achievable_offload_fracs).

    If recompute_lens is None, the optimal recompute_len is auto-found for
    each (gbs, offload_frac) pair instead of being swept explicitly.

    Returns (best_feasible_result, all_results).
    """
    all_results = []
    best = None

    for gbs in gpu_batch_sizes:
        for offload_frac in achievable_offload_fracs(gbs):
            candidates = [None] if recompute_lens is None else recompute_lens
            for rlen in candidates:
                result = evaluate_policy(model, hw, gbs, offload_frac, total_prompts,
                                          recompute_len=rlen)
                all_results.append(result)
                if result["feasible"] and (
                    best is None or result["effective_throughput_tok_s"] > best["effective_throughput_tok_s"]
                ):
                    best = result

    return best, all_results


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Sweep GPU batch size, KV-cache offload %%, and recompute "
                     "length for policies with weights + activations always on GPU.")

    parser.add_argument("--gpu", choices=sorted(GPU_PRESETS), default="a100-80gb",
                         help="Sets gpu_matmul_flops and gpu_mem_bytes.")
    parser.add_argument("--pcie", choices=sorted(PCIE_PRESETS), default="16gb/s",
                         help="CPU<->GPU achievable bandwidth.")
    parser.add_argument("--dtype-bytes", type=int, default=2,
                         help="Bytes per element (fp16/bf16=2, fp32=4).")

    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="opt-6.7b",
                         help="Sets num_layers, hidden_size, ffn_dim, num_heads.")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--total-prompts", type=int, required=True,
                         help="Total prompts in the job. Required: GPU-resident "
                              "KV cache is treated as cumulative across the "
                              "whole job (never freed between batches), so "
                              "this drives the OOM check, not just wall-clock time.")

    parser.add_argument("--gpu-batch-sizes", type=int, nargs="+",
                         default=[1, 4, 8, 16, 32, 64],
                         help="Micro-batch size (gbs). num_gpu_batches is auto-derived "
                              "per gbs as ceil(total_prompts / gbs), so bls always "
                              "covers the whole job in one round. offload_frac is "
                              "also auto-generated per gbs as k/gbs (one prompt is "
                              "the smallest offload granularity within a batch).")
    parser.add_argument("--recompute-len", type=int, default=None,
                         help="Fix a single recompute length instead of "
                              "auto-optimizing per (gbs, offload_frac).")
    parser.add_argument("--top-n", type=int, default=10,
                         help="How many top results to print.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    gpu_matmul_flops, gpu_mem_bytes = GPU_PRESETS[args.gpu]
    hw = HardwareConfig(
        gpu_matmul_flops=gpu_matmul_flops,
        cpu_gpu_bandwidth=PCIE_PRESETS[args.pcie],
        dtype_bytes=args.dtype_bytes,
        gpu_mem_bytes=gpu_mem_bytes,
    )

    model = ModelConfig(
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        **MODEL_PRESETS[args.model],
    )

    recompute_lens = [args.recompute_len] if args.recompute_len is not None else None

    print(f"GPU: {args.gpu}  (gpu_matmul_flops={hw.gpu_matmul_flops/T:.1f} TFLOPS, "
          f"gpu_mem={hw.gpu_mem_bytes/GB:.0f} GB)")
    print(f"PCIe: {args.pcie}  ({hw.cpu_gpu_bandwidth/GB:.0f} GB/s)")
    print(f"Model: {args.model}  (layers={model.num_layers}, hidden={model.hidden_size}, "
          f"ffn={model.ffn_dim}, heads={model.num_heads})")
    print(f"Workload: prompt_len={model.prompt_len}, gen_len={model.gen_len}\n")

    best, all_results = sweep_gpu_resident_policies(
        model, hw, args.gpu_batch_sizes,
        total_prompts=args.total_prompts, recompute_lens=recompute_lens
    )

    print("Best feasible policy:")
    print(best)

    feasible_results = [r for r in all_results if r["feasible"]]
    print(f"\nTop {args.top_n} feasible results ({len(feasible_results)} of "
          f"{len(all_results)} total configs were feasible):")
    for r in sorted(feasible_results, key=lambda x: -x["effective_throughput_tok_s"])[:args.top_n]:
        print(f"[OK] gbs={r['gpu_batch_size']:>3} nb={r['num_gpu_batches']:>3} bls={r['bls']:>4} "
              f"offload={r['offload_frac']:.2f} recompute_len={r['recompute_len']:>4} "
              f"cache={r['gpu_cache_gb']:.2f}GB mem={r['gpu_mem_used_gb']:.2f}GB "
              f"rounds={r['num_rounds']} util={r['batch_utilization']*100:.1f}% "
              f"wall_time={r['wall_time_s']:.2f}s "
              f"effective_throughput={r['effective_throughput_tok_s']:.2f} tok/s")


if __name__ == "__main__":
    main()