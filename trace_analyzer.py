#!/usr/bin/env python3
"""
Torch profile trace analyzer for FlexLLMGen / flex_opt_kvpr traces.

Three modes:

--sep (default):
  Each output row is a supergroup = consecutive mha_gen -> mlp pair from decode.
  Supergroup 0 (warm-up) and all prefill groups (token 0) are skipped.
  mha_gen compute CUDA ops tagged 'mha_gen' or 'fwd_pre_mha' by cpu_op attribution.

--nosep:
  Each output row is a single nosep group from decode (token >= 1).
  Group 0 (warm-up) is skipped.
  compute_layer contains both mha_gen and mlp in one forward pass.
  CUDA ops tagged: 'fwd_pre' (before mha_gen in forward), 'mha_gen', or 'mlp'.

--batched:
  Each output row is the FIRST group of a consecutive mha_gen->mha_gen pair
  from decode (token >= 1). The first such pair (warm-up) is skipped.
  Reports the same 8 ops as sep mha_gen groups, plus load_cache pin_memory
  sub-ops and store_cache memcpy sub-ops.
  compute_layer CUDA ops tagged 'mha_gen' or 'fwd_pre_mha'.

Usage:
    python trace_analyzer.py <trace.json> [--max-groups N] [--out output.csv]
                                          [--nosep | --batched]

All durations in microseconds (us).
"""

import bisect
import json
import csv
import argparse
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_indices(events):
    parent_to_children = {}
    py_id_map = {}
    corr_to_gpu = {}
    ext_id_to_cpu_ops = {}
    corrs_with_cpu_rt = set()

    # Pre-filtered lists for fast time-window queries (populated below, sorted after)
    # cuda_rt_by_tid holds BOTH cuda_runtime (cudaLaunchKernel etc.) AND
    # cuda_driver (cuLaunchKernel etc.) dispatch events. cuBLASLt picks a
    # cutlass tensor-op kernel for some GEMM shapes and dispatches it via the
    # driver API instead of the runtime API -- those dispatches only show up
    # as cat="cuda_driver" in the trace, never "cuda_runtime". Excluding them
    # here silently drops every cutlass-dispatched kernel from every window
    # search below (get_gpu_events_in_window, get_gpu_events_via_cpu_chain,
    # _has_gpu_activity_in_window, and the cuda_rt_in_fwd primary loop in
    # get_gpu_events_in_forward_sep/nosep) -- verified on the sample trace:
    # 462/3088 GPU-side events (~15%), all cutlass::Kernel2 GEMMs, were
    # dispatched exclusively via cuda_driver and were being missed.
    cuda_rt_by_tid = {}   # tid -> [event, ...]  (cuda_runtime + cuda_driver dispatch events)
    pin_mem_list   = []   # aten::pin_memory cpu_op events
    gpu_event_list = []   # kernel/gpu_memcpy/gpu_memset events (for cutlass fallback)

    for e in events:
        args = e.get("args") or {}
        cat  = e.get("cat")
        py_id     = args.get("Python id")
        parent_id = args.get("Python parent id")
        corr      = args.get("correlation")
        ext_id    = args.get("External id")

        if py_id is not None:
            py_id_map[py_id] = e
        if parent_id is not None:
            parent_to_children.setdefault(parent_id, []).append(e)
        if corr is not None and cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            corr_to_gpu[corr] = e
            gpu_event_list.append(e)
        if ext_id is not None and cat == "cpu_op":
            ext_id_to_cpu_ops.setdefault(ext_id, []).append(e)
            if e.get("name") == "aten::pin_memory":
                pin_mem_list.append(e)
        if corr is not None and cat in ("cuda_runtime", "cuda_driver"):
            corrs_with_cpu_rt.add(corr)
            cuda_rt_by_tid.setdefault(e["tid"], []).append(e)

    # Sort each pre-filtered list once by timestamp
    for lst in cuda_rt_by_tid.values():
        lst.sort(key=lambda e: e["ts"])
    pin_mem_list.sort(key=lambda e: e["ts"])
    gpu_event_list.sort(key=lambda e: e["ts"])

    # Pre-compute timestamp key arrays for bisect
    cuda_rt_ts_by_tid = {tid: [e["ts"] for e in lst]
                         for tid, lst in cuda_rt_by_tid.items()}
    pin_mem_ts   = [e["ts"] for e in pin_mem_list]
    gpu_event_ts = [e["ts"] for e in gpu_event_list]

    return (parent_to_children, py_id_map, corr_to_gpu, ext_id_to_cpu_ops,
            corrs_with_cpu_rt,
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            pin_mem_list, pin_mem_ts,
            gpu_event_list, gpu_event_ts)


def detect_main_tid(events):
    timer_tids = [e["tid"] for e in events if e.get("name") == "timer.py(20): start"]
    if timer_tids:
        return timer_tids[0]
    tid_counts = Counter(e["tid"] for e in events if e.get("cat") == "python_function")
    return tid_counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

OP8_NAMES = {
    "load_weight", "load_hidden_compute", "load_cache",
    "store_hidden", "load_hidden", "compute_layer",
    "store_cache", "sync",
}


def is_op8(event):
    return any(op in event.get("name", "") for op in OP8_NAMES)


# ---------------------------------------------------------------------------
# Interval overlap helpers
# ---------------------------------------------------------------------------
#
# Used to decompose PCIe (cudaMemcpyAsync) and CPU-side (aten::pin_memory,
# general_copy, _attention_value) op latencies into the portion that is
# actually hidden behind other device activity vs. the portion that is not
# (and therefore sits on the true critical path), by comparing each op's
# real [ts, ts+dur) window against the windows of the ops it is meant to be
# overlapped with. All three helpers operate on plain (start, end) tuples in
# the trace's own timestamp units (us).

def _merge_intervals(intervals):
    """Sort + coalesce overlapping/touching (start, end) tuples."""
    ivs = sorted((s, e) for s, e in intervals if e > s)
    merged = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def interval_union_duration(intervals):
    """Total duration covered by the union of (start, end) tuples (no double-counting)."""
    return sum(e - s for s, e in _merge_intervals(intervals))


def interval_overlap_duration(intervals_a, intervals_b):
    """Total duration where union(intervals_a) overlaps union(intervals_b)."""
    ma = _merge_intervals(intervals_a)
    mb = _merge_intervals(intervals_b)
    i, j = 0, 0
    total = 0.0
    while i < len(ma) and j < len(mb):
        s = max(ma[i][0], mb[j][0])
        e = min(ma[i][1], mb[j][1])
        if s < e:
            total += e - s
        if ma[i][1] < mb[j][1]:
            i += 1
        else:
            j += 1
    return total


def overlap_breakdown(intervals, ref_intervals):
    """
    Decompose `intervals` (e.g. all PCIe memcpys of one category) against
    `ref_intervals` (e.g. all GPU-compute windows) into
    (total, overlapped, non_overlapped), each in us, 3-decimal rounded.
    non_overlapped is the portion of `intervals` NOT hidden behind
    `ref_intervals` -- i.e. the portion that still sits on the wall-clock
    critical path.
    """
    total = interval_union_duration(intervals)
    overlap = interval_overlap_duration(intervals, ref_intervals)
    nonoverlap = max(total - overlap, 0.0)
    return round(total, 3), round(overlap, 3), round(nonoverlap, 3)


def events_to_intervals(events):
    """[event, ...] with 'ts'/'dur' keys -> [(start, end), ...], skipping None."""
    return [(e["ts"], e["ts"] + e["dur"]) for e in events if e is not None]


def get_cuda_rt_in_window(cuda_rt_by_tid, cuda_rt_ts_by_tid, ts_start, ts_end, main_tid):
    """Return cuda_runtime/cuda_driver dispatch events for main_tid in [ts_start, ts_end] using bisect."""
    lst = cuda_rt_by_tid.get(main_tid)
    if not lst:
        return []
    ts_keys = cuda_rt_ts_by_tid[main_tid]
    lo = bisect.bisect_left(ts_keys, ts_start)
    hi = bisect.bisect_right(ts_keys, ts_end)
    return lst[lo:hi]


def get_gpu_events_in_window(cuda_rt_by_tid, cuda_rt_ts_by_tid, ts_start, ts_end,
                              main_tid, corr_to_gpu, filter_names=None):
    results = []
    for cr in get_cuda_rt_in_window(cuda_rt_by_tid, cuda_rt_ts_by_tid,
                                     ts_start, ts_end, main_tid):
        if filter_names and cr.get("name") not in filter_names:
            continue
        corr = (cr.get("args") or {}).get("correlation")
        if corr is None:
            continue
        gpu_ev = corr_to_gpu.get(corr)
        if gpu_ev is not None:
            results.append(gpu_ev)
    return results


def get_gpu_events_via_cpu_chain(cuda_rt_by_tid, cuda_rt_ts_by_tid, pyfn_event,
                                  cl_ts_s, cl_ts_e, main_tid, corr_to_gpu,
                                  ext_id_to_cpu_ops, exclude_cats=None):
    """GPU events whose originating cpu_op falls within pyfn_event's window."""
    exclude_cats = exclude_cats or set()
    pyfn_ts_s = pyfn_event["ts"]
    pyfn_ts_e = pyfn_ts_s + pyfn_event["dur"]
    results = []
    for cr in get_cuda_rt_in_window(cuda_rt_by_tid, cuda_rt_ts_by_tid,
                                     cl_ts_s, cl_ts_e, main_tid):
        args   = cr.get("args") or {}
        ext_id = args.get("External id")
        corr   = args.get("correlation")
        gpu_ev = corr_to_gpu.get(corr)
        if gpu_ev is None or gpu_ev.get("cat") in exclude_cats:
            continue
        cpu_ops = ext_id_to_cpu_ops.get(ext_id, [])
        if not cpu_ops:
            continue
        cpu_op = min(cpu_ops, key=lambda e: abs(e["ts"] - cr["ts"]))
        if pyfn_ts_s <= cpu_op["ts"] <= pyfn_ts_e:
            results.append(gpu_ev)
    return results


def get_pin_memory_events_in_window(pin_mem_list, pin_mem_ts, ts_start, ts_end):
    """Return aten::pin_memory events in [ts_start, ts_end] using bisect."""
    lo = bisect.bisect_left(pin_mem_ts, ts_start)
    hi = bisect.bisect_right(pin_mem_ts, ts_end)
    return pin_mem_list[lo:hi]


def get_token_boundaries(events):
    timer_starts = sorted(
        [e for e in events if e.get("name") == "timer.py(20): start"],
        key=lambda e: e["ts"],
    )
    timer_stops = sorted(
        [e for e in events if e.get("name") == "timer.py(31): stop"],
        key=lambda e: e["ts"],
    )
    return list(zip([t["ts"] for t in timer_starts], [t["ts"] for t in timer_stops]))


def get_gen_children(events, parent_to_children):
    gen_loop = next(
        (e for e in events
         if ("generation_loop_overlap_single_batch" in e.get("name", "")
         and e.get("cat") == "python_function") or 
         ("generation_loop_overlap_multi_batch" in e.get("name", "") and e.get("cat") == "python_function")),
        None
    )
    if gen_loop is None:
        raise RuntimeError("Could not find generation_loop_overlap_single_batch or generation_loop_overlap_multi_batch.")
    gen_loop_id = gen_loop["args"]["Python id"]
    return sorted(parent_to_children.get(gen_loop_id, []), key=lambda e: e["ts"])


# ---------------------------------------------------------------------------
# SEP mode helpers
# ---------------------------------------------------------------------------

def get_compute_layer_type_sep(cl_event, parent_to_children):
    cl_id = cl_event["args"]["Python id"]
    for fc in parent_to_children.get(cl_id, []):
        if "forward" not in fc.get("name", ""):
            continue
        fwd_id = fc["args"]["Python id"]
        fwd_sub = parent_to_children.get(fwd_id, [])
        names = [s.get("name", "") for s in fwd_sub]
        if any("mha_gen" in n for n in names):
            return "mha_gen"
        if any("mlp" in n and "pytorch_backend" in n for n in names):
            return "mlp"
    return "other"


def get_forward_pyfn(cl_event, parent_to_children):
    cl_id = cl_event["args"]["Python id"]
    for fc in parent_to_children.get(cl_id, []):
        if "forward" in fc.get("name", ""):
            return fc
    return None


def get_inner_pyfn(cl_event, parent_to_children, name_check):
    cl_id = cl_event["args"]["Python id"]
    for fc in parent_to_children.get(cl_id, []):
        if "forward" not in fc.get("name", ""):
            continue
        fwd_id = fc["args"]["Python id"]
        for sub in parent_to_children.get(fwd_id, []):
            if name_check(sub.get("name", "")):
                return sub
    return None


def get_gpu_events_in_forward_sep(cuda_rt_by_tid, cuda_rt_ts_by_tid,
                                   gpu_event_list, gpu_event_ts,
                                   cl_event, parent_to_children,
                                   main_tid, corr_to_gpu, ext_id_to_cpu_ops,
                                   corrs_with_cpu_rt):
    """
    SEP mode: all GPU events for compute_layer -> forward(), tagged by origin.
    Returns list of (gpu_event, origin_tag) sorted by GPU timestamp.
    Origins: 'mha_gen', 'fwd_pre_mha'
    """
    fwd_fn = get_forward_pyfn(cl_event, parent_to_children)
    if fwd_fn is None:
        return []

    mhag_fn = get_inner_pyfn(
        cl_event, parent_to_children,
        lambda n: "mha_gen" in n and "pytorch_backend" in n
    )
    mhag_ts_s = mhag_fn["ts"] if mhag_fn else float("inf")
    mhag_ts_e = (mhag_fn["ts"] + mhag_fn["dur"]) if mhag_fn else float("inf")

    cuda_rt_in_fwd = get_cuda_rt_in_window(
        cuda_rt_by_tid, cuda_rt_ts_by_tid,
        fwd_fn["ts"], fwd_fn["ts"] + fwd_fn["dur"], main_tid
    )
    results = []
    found_corrs = set()
    for cr in cuda_rt_in_fwd:
        args   = cr.get("args") or {}
        corr   = args.get("correlation")
        ext_id = args.get("External id")
        if corr is None:
            continue
        gpu_ev = corr_to_gpu.get(corr)
        if gpu_ev is None:
            continue
        cpu_ops = ext_id_to_cpu_ops.get(ext_id, [])
        if cpu_ops:
            cpu_op = min(cpu_ops, key=lambda e: abs(e["ts"] - cr["ts"]))
            in_mha = mhag_ts_s <= cpu_op["ts"] <= mhag_ts_e
            origin = "mha_gen" if in_mha else "fwd_pre_mha"
        else:
            origin = "fwd_pre_mha"
        results.append((gpu_ev, origin))
        found_corrs.add(corr)

    # Cutlass kernels with no cuda_runtime record — use gpu_event_list + bisect
    if results:
        gpu_ts_min = min(r[0]["ts"] for r in results)
        gpu_ts_max = max(r[0]["ts"] + r[0]["dur"] for r in results)
    else:
        gpu_ts_min = fwd_fn["ts"]
        gpu_ts_max = fwd_fn["ts"] + fwd_fn["dur"]

    lo = bisect.bisect_left(gpu_event_ts, gpu_ts_min)
    hi = bisect.bisect_right(gpu_event_ts, gpu_ts_max + 100)
    for e in gpu_event_list[lo:hi]:
        corr = (e.get("args") or {}).get("correlation")
        if corr is None or corr in found_corrs or corr in corrs_with_cpu_rt:
            continue
        origin = "fwd_pre_mha"
        results.append((e, origin))
        found_corrs.add(corr)

    results.sort(key=lambda x: x[0]["ts"])
    return results


def identify_supergroups_sep(events, parent_to_children, token_boundaries):
    """SEP mode: find consecutive mha_gen->mlp pairs, skip warm-up."""
    gen_children = get_gen_children(events, parent_to_children)
    all_groups = []
    for ti, (ts_start, ts_end) in enumerate(token_boundaries):
        token_8ops = [e for e in gen_children
                      if is_op8(e) and ts_start <= e["ts"] <= ts_end]
        i = 0
        while i + 7 < len(token_8ops):
            grp = token_8ops[i:i + 8]
            cl = next((e for e in grp if "compute_layer" in e.get("name", "")), None)
            if cl is not None:
                ftype = get_compute_layer_type_sep(cl, parent_to_children)
                all_groups.append({"token": ti, "type": ftype, "events": grp, "cl": cl})
            i += 8

    decode_groups = [g for g in all_groups if g["token"] >= 1]
    supergroups = []
    i = 0
    while i < len(decode_groups) - 1:
        a, b = decode_groups[i], decode_groups[i + 1]
        if a["type"] == "mha_gen" and b["type"] == "mlp":
            supergroups.append({"token": a["token"], "mha_gen": a, "mlp": b})
            i += 2
        else:
            i += 1
    return supergroups[1:]  # skip warm-up


# ---------------------------------------------------------------------------
# NOSEP mode helpers
# ---------------------------------------------------------------------------

def get_nosep_pyfns(cl_event, parent_to_children):
    """
    NOSEP: compute_layer -> forward -> [forward(mha), forward(mlp)]
    Returns (mha_fwd, mhag_fn, mlp_fwd, mlp_fn) or all None if not nosep.
    """
    cl_id = cl_event["args"]["Python id"]
    for fc in parent_to_children.get(cl_id, []):
        if "forward" not in fc.get("name", ""):
            continue
        fwd_id = fc["args"]["Python id"]
        fwd_ch = sorted(parent_to_children.get(fwd_id, []), key=lambda e: e["ts"])
        sub_fwds = [c for c in fwd_ch if "forward" in c.get("name", "")]
        if len(sub_fwds) == 2:
            mha_fwd, mlp_fwd = sub_fwds[0], sub_fwds[1]
            mhag_fn = next(
                (c for c in parent_to_children.get(mha_fwd["args"]["Python id"], [])
                 if "mha_gen" in c.get("name", "")),
                None
            )
            mlp_fn = next(
                (c for c in parent_to_children.get(mlp_fwd["args"]["Python id"], [])
                 if "mlp" in c.get("name", "") and "pytorch_backend" in c.get("name", "")),
                None
            )
            return mha_fwd, mhag_fn, mlp_fwd, mlp_fn
    return None, None, None, None


def is_nosep_group(cl_event, parent_to_children):
    mha_fwd, _, _, _ = get_nosep_pyfns(cl_event, parent_to_children)
    return mha_fwd is not None


def get_gpu_events_in_forward_nosep(cuda_rt_by_tid, cuda_rt_ts_by_tid,
                                     gpu_event_list, gpu_event_ts,
                                     cl_event, parent_to_children,
                                     main_tid, corr_to_gpu, ext_id_to_cpu_ops,
                                     corrs_with_cpu_rt):
    """
    NOSEP mode: GPU events for compute_layer with 3-way origin tagging.
    Origins: 'fwd_pre' (before mha_gen), 'mha_gen', 'mlp'
    Uses cpu_op attribution where available, GPU timestamp for cutlass kernels.
    """
    mha_fwd, mhag_fn, mlp_fwd, mlp_fn = get_nosep_pyfns(cl_event, parent_to_children)
    if mha_fwd is None:
        return []

    cl_ts_s = cl_event["ts"]
    cl_ts_e = cl_ts_s + cl_event["dur"]

    mhag_ts_s = mhag_fn["ts"] if mhag_fn else float("inf")
    mhag_ts_e = (mhag_fn["ts"] + mhag_fn["dur"]) if mhag_fn else float("inf")
    mlp_ts_s  = mlp_fn["ts"] if mlp_fn else float("inf")
    mlp_ts_e  = (mlp_fn["ts"] + mlp_fn["dur"]) if mlp_fn else float("inf")

    cuda_rt_in_cl = get_cuda_rt_in_window(
        cuda_rt_by_tid, cuda_rt_ts_by_tid, cl_ts_s, cl_ts_e, main_tid
    )
    results = []
    found_corrs = set()

    for cr in sorted(cuda_rt_in_cl, key=lambda e: e["ts"]):
        args   = cr.get("args") or {}
        ext_id = args.get("External id")
        corr   = args.get("correlation")
        gpu_ev = corr_to_gpu.get(corr)
        if gpu_ev is None:
            continue
        cpu_ops = ext_id_to_cpu_ops.get(ext_id, [])
        if cpu_ops:
            cpu_op = min(cpu_ops, key=lambda e: abs(e["ts"] - cr["ts"]))
            cpu_ts = cpu_op["ts"]
            if mhag_ts_s <= cpu_ts <= mhag_ts_e:
                origin = "mha_gen"
            elif mlp_ts_s <= cpu_ts <= mlp_ts_e:
                origin = "mlp"
            else:
                origin = "fwd_pre"
        else:
            origin = "fwd_pre"
        results.append((gpu_ev, origin))
        found_corrs.add(corr)

    # Cutlass kernels with no cuda_runtime record — use gpu_event_list + bisect
    if results:
        gpu_ts_min = min(r[0]["ts"] for r in results)
        gpu_ts_max = max(r[0]["ts"] + r[0]["dur"] for r in results)
    else:
        gpu_ts_min = cl_ts_s
        gpu_ts_max = cl_ts_e

    lo = bisect.bisect_left(gpu_event_ts, gpu_ts_min)
    hi = bisect.bisect_right(gpu_event_ts, gpu_ts_max + 100)
    for e in gpu_event_list[lo:hi]:
        corr = (e.get("args") or {}).get("correlation")
        if corr is None or corr in found_corrs or corr in corrs_with_cpu_rt:
            continue
        gts = e["ts"]
        if mhag_ts_s <= gts <= mhag_ts_e + 500:
            origin = "mha_gen"
        elif mlp_ts_s <= gts <= mlp_ts_e + 500:
            origin = "mlp"
        else:
            origin = "fwd_pre"
        results.append((e, origin))
        found_corrs.add(corr)

    results.sort(key=lambda x: x[0]["ts"])
    return results


def identify_nosep_groups(events, parent_to_children, token_boundaries):
    """NOSEP mode: find all nosep groups from decode tokens, skip group 0."""
    gen_children = get_gen_children(events, parent_to_children)
    nosep_groups = []
    for ti, (ts_start, ts_end) in enumerate(token_boundaries):
        if ti == 0:
            continue  # skip prefill
        token_8ops = [e for e in gen_children
                      if is_op8(e) and ts_start <= e["ts"] <= ts_end]
        i = 0
        while i + 7 < len(token_8ops):
            grp = token_8ops[i:i + 8]
            cl = next((e for e in grp if "compute_layer" in e.get("name", "")), None)
            if cl is not None and is_nosep_group(cl, parent_to_children):
                nosep_groups.append({"token": ti, "events": grp, "cl": cl})
            i += 8
    return nosep_groups[1:]  # skip warm-up (first nosep group)


# ---------------------------------------------------------------------------
# BATCHED mode helpers
# ---------------------------------------------------------------------------

def identify_batched_groups(events, parent_to_children, token_boundaries,
                             classify_fn=get_compute_layer_type_sep,
                             pair_type="mha_gen"):
    """
    BATCHED mode: find all 8-op groups that are mha_gen AND whose immediately
    following 8-op group is also mha_gen (i.e., the first of a consecutive pair).
    All prefill (token 0) groups are skipped, and the first qualifying pair
    across all decode tokens is skipped as warm-up.

    Each returned group dict includes 'async_sc_siblings': a sorted list of
    sibling store_cache python_function events (under the same generation_loop
    parent) that may be searched with bisect to find the async store_cache
    dispatcher after sync ends.  Building this once here avoids re-sorting
    the full sibling list (16 000+ events) on every call to extract_batched_metrics.

    classify_fn / pair_type : parameterize the "what counts as a pair" test so
        other modes (e.g. --cpu-computation) can reuse this exact grouping /
        warm-up-skip / async-sibling-caching logic with a different
        compute_layer classifier. Defaults reproduce --batched's original
        behavior exactly (classify_fn=get_compute_layer_type_sep,
        pair_type="mha_gen") -- existing callers are unaffected.
    """
    gen_children = get_gen_children(events, parent_to_children)
    all_groups = []
    # Cache: gen_loop_parent_id -> sorted list of async store_cache siblings
    _sc_sibling_cache = {}

    for ti, (ts_start, ts_end) in enumerate(token_boundaries):
        if ti == 0:
            continue  # skip prefill
        token_8ops = [e for e in gen_children
                      if is_op8(e) and ts_start <= e["ts"] <= ts_end]
        grps = []
        i = 0
        while i + 7 < len(token_8ops):
            grp = token_8ops[i:i + 8]
            cl = next((e for e in grp if "compute_layer" in e.get("name", "")), None)
            if cl is not None:
                ftype = classify_fn(cl, parent_to_children)
                grps.append({"token": ti, "type": ftype, "events": grp, "cl": cl})
            i += 8
        # Collect the first group of each consecutive pair-type->pair-type pair
        for j in range(len(grps) - 1):
            if grps[j]["type"] == pair_type and grps[j + 1]["type"] == pair_type:
                g = grps[j]
                # Build async store_cache sibling list once per gen_loop parent
                outer_sc = next(
                    (e for e in g["events"] if "store_cache" in e.get("name", "")),
                    None
                )
                async_sc_siblings = []
                if outer_sc is not None:
                    pid = outer_sc["args"].get("Python parent id")
                    if pid is not None:
                        if pid not in _sc_sibling_cache:
                            siblings = parent_to_children.get(pid, [])
                            tid_val = outer_sc.get("tid")
                            sc_sib = sorted(
                                [s for s in siblings
                                 if "store_cache" in s.get("name", "")
                                 and s.get("tid") == tid_val],
                                key=lambda e: e["ts"]
                            )
                            _sc_sibling_cache[pid] = sc_sib
                        async_sc_siblings = _sc_sibling_cache[pid]
                g["async_sc_siblings"] = async_sc_siblings
                all_groups.append(g)

    return all_groups[1:]  # skip warm-up (first batched group)


def extract_batched_metrics(grp, cuda_rt_by_tid, cuda_rt_ts_by_tid,
                             gpu_event_list, gpu_event_ts,
                             corr_to_gpu, ext_id_to_cpu_ops,
                             corrs_with_cpu_rt,
                             pin_mem_list, pin_mem_ts,
                             parent_to_children, main_tid):
    """
    BATCHED mode metric extraction for a single mha_gen group.
    Reports the 8 ops plus sub-ops for load_hidden_compute, load_cache,
    store_cache (same as nosep), and compute_layer CUDA ops tagged
    'mha_gen' or 'fwd_pre_mha' (same as sep mha_gen).
    """
    evts = {e["name"].split(": ")[-1]: e for e in grp["events"]}
    def dur(key):
        e = evts.get(key)
        return round(e["dur"], 3) if e else None

    cl = grp["cl"]

    # compute_layer CUDA ops: same tagging as sep mha_gen (fwd_pre_mha / mha_gen)
    tagged_gpu_ops = get_gpu_events_in_forward_sep(
        cuda_rt_by_tid, cuda_rt_ts_by_tid,
        gpu_event_list, gpu_event_ts,
        cl, parent_to_children,
        main_tid, corr_to_gpu, ext_id_to_cpu_ops, corrs_with_cpu_rt
    )

    # load_hidden_compute cudaMemcpyAsync (recompute-related PCIe transfer)
    lhc = evts.get("load_hidden_compute")
    lhc_gpu = []
    lhc_memcpy_dur = None
    if lhc:
        lhc_gpu = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            lhc["ts"], lhc["ts"] + lhc["dur"],
            main_tid, corr_to_gpu, filter_names={"cudaMemcpyAsync"}
        )
        if lhc_gpu:
            lhc_memcpy_dur = round(lhc_gpu[0]["dur"], 3)

    # load_cache: pin_memory x2 (CPU pageable->pinned), cudaMemcpyAsync x2 (KV cache load, PCIe)
    lc = evts.get("load_cache")
    pins, lc_gpu = [], []
    pm1, pm2, lc_mc1, lc_mc2 = None, None, None, None
    if lc:
        lc_ts_s, lc_ts_e = lc["ts"], lc["ts"] + lc["dur"]
        pins = get_pin_memory_events_in_window(pin_mem_list, pin_mem_ts,
                                               lc_ts_s, lc_ts_e)
        if len(pins) > 0: pm1 = round(pins[0]["dur"], 3)
        if len(pins) > 1: pm2 = round(pins[1]["dur"], 3)
        lc_gpu = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            lc_ts_s, lc_ts_e, main_tid, corr_to_gpu,
            filter_names={"cudaMemcpyAsync"}
        )
        if len(lc_gpu) > 0: lc_mc1 = round(lc_gpu[0]["dur"], 3)
        if len(lc_gpu) > 1: lc_mc2 = round(lc_gpu[1]["dur"], 3)

    # store_cache: the 8-op group's store_cache only acquires a CUDA stream and
    # dispatches no memcpy. The actual DtoH transfers are dispatched by a sibling
    # store_cache call (flex_opt_kvpr.py(932)) that runs after sync ends, under the
    # same generation_loop parent. identify_batched_groups pre-builds a sorted list
    # of these siblings; we bisect to find the first one after sync ends.
    sync_e = evts.get("sync")
    sc_gpu = []
    sc_mc1, sc_mc2 = None, None
    if sync_e is not None:
        sync_end = sync_e["ts"] + sync_e["dur"]
        async_sc_siblings = grp.get("async_sc_siblings", [])
        if async_sc_siblings:
            sc_ts_list = [s["ts"] for s in async_sc_siblings]
            lo = bisect.bisect_left(sc_ts_list, sync_end)
            if lo < len(async_sc_siblings):
                async_sc = async_sc_siblings[lo]
                sc_gpu = get_gpu_events_in_window(
                    cuda_rt_by_tid, cuda_rt_ts_by_tid,
                    async_sc["ts"], async_sc["ts"] + async_sc["dur"],
                    main_tid, corr_to_gpu, filter_names={"cudaMemcpyAsync"}
                )
                if len(sc_gpu) > 0: sc_mc1 = round(sc_gpu[0]["dur"], 3)
                if len(sc_gpu) > 1: sc_mc2 = round(sc_gpu[1]["dur"], 3)

    # -----------------------------------------------------------------
    # Overlap-aware breakdown.
    #
    # GPU busy window = union of all compute_layer CUDA kernels (both
    # mha_gen- and fwd_pre_mha/recompute-origin -- anything running on the
    # GPU compute stream counts as "GPU busy" for the purpose of deciding
    # whether a PCIe transfer was hidden).
    #
    # PCIe categories (each is its own cudaMemcpyAsync pair/singleton,
    # dispatched on the copy engine, so they use their OWN [ts, ts+dur)
    # windows -- not the CPU-thread dispatcher's window):
    #   - KV cache load  : load-cache-cudamemcpy-1/2  (H2D)
    #   - KV cache store  : store-cache-cudamemcpy-1/2 (D2H)
    #   - recompute memcpy: load-hidden-compute-cudamemcpy
    # For each, we report total / overlapped-with-GPU / non-overlapped.
    #
    # CPU category (pageable->pinned only in --batched mode; --cpu-computation
    # mode adds internal-copy and cpu-mha-compute, see extract_cpu_computation_metrics):
    #   - pin-memory-1/2, checked against GPU-busy UNION all-PCIe-busy,
    #     since a pin_memory call fully hidden behind either a GPU kernel or
    #     a PCIe transfer is not on the critical path.
    # -----------------------------------------------------------------
    gpu_intervals = events_to_intervals([gpu_ev for gpu_ev, _origin in tagged_gpu_ops])

    kv_load_intervals  = events_to_intervals(lc_gpu[:2])
    kv_store_intervals = events_to_intervals(sc_gpu[:2])
    recompute_mc_intervals = events_to_intervals(lhc_gpu[:1])
    pcie_all_intervals = kv_load_intervals + kv_store_intervals + recompute_mc_intervals

    kv_load_total, kv_load_ov, kv_load_nonov = overlap_breakdown(kv_load_intervals, gpu_intervals)
    kv_store_total, kv_store_ov, kv_store_nonov = overlap_breakdown(kv_store_intervals, gpu_intervals)
    recompute_mc_total, recompute_mc_ov, recompute_mc_nonov = overlap_breakdown(recompute_mc_intervals, gpu_intervals)

    device_intervals = gpu_intervals + pcie_all_intervals  # GPU compute UNION all PCIe
    pin_intervals = events_to_intervals(pins[:2])
    pin_total, pin_ov, pin_nonov = overlap_breakdown(pin_intervals, device_intervals)

    metrics = {
        "load_weight":                    dur("load_weight"),
        "load_hidden_compute":             dur("load_hidden_compute"),
        "load-hidden-compute-cudamemcpy":  lhc_memcpy_dur,
        "load_cache":                      dur("load_cache"),
        "pin-memory-1":                    pm1,
        "pin-memory-2":                    pm2,
        "load-cache-cudamemcpy-1":         lc_mc1,
        "load-cache-cudamemcpy-2":         lc_mc2,
        "load_hidden":                     dur("load_hidden"),
        "compute_layer":                   dur("compute_layer"),
        "store_cache":                     dur("store_cache"),
        "store-cache-cudamemcpy-1":        sc_mc1,
        "store-cache-cudamemcpy-2":        sc_mc2,
        "store_hidden":                    dur("store_hidden"),
        "sync":                            dur("sync"),
        # --- overlap-aware PCIe breakdown (all in us) ---
        "pcie-kv-load-total-us":              kv_load_total,
        "pcie-kv-load-overlapped-us":         kv_load_ov,
        "pcie-kv-load-nonoverlapped-us":      kv_load_nonov,
        "pcie-kv-store-total-us":             kv_store_total,
        "pcie-kv-store-overlapped-us":        kv_store_ov,
        "pcie-kv-store-nonoverlapped-us":     kv_store_nonov,
        "pcie-recompute-memcpy-total-us":         recompute_mc_total,
        "pcie-recompute-memcpy-overlapped-us":    recompute_mc_ov,
        "pcie-recompute-memcpy-nonoverlapped-us": recompute_mc_nonov,
        # --- overlap-aware CPU (misc) breakdown (all in us) ---
        "cpu-pageable-pinned-total-us":         pin_total,
        "cpu-pageable-pinned-overlapped-us":    pin_ov,
        "cpu-pageable-pinned-nonoverlapped-us": pin_nonov,
    }
    for idx, (gpu_ev, origin) in enumerate(tagged_gpu_ops, 1):
        metrics[f"compute-cuda-{idx}"] = round(gpu_ev["dur"], 3)
        metrics[f"compute-cuda-{idx}-origin"] = origin

    return metrics


# ---------------------------------------------------------------------------
# CPU-COMPUTATION mode helpers (--cpu-gpu-compute traces)
# ---------------------------------------------------------------------------

def get_compute_layer_type_cpu_mixed(cl_event, parent_to_children):
    """
    Classifier for --cpu-computation mode, parallel to get_compute_layer_type_sep:
    a compute_layer group is "cpu_mixed" iff its forward()'s direct children
    include a call to _mixed_cpu_attention (only present when the run used
    --cpu-gpu-compute, gated purely on that flag -- see SelfAttention.forward()
    in flex_opt_kvpr.py). Every other group (MLP, or SelfAttention groups from
    a non-cpu_gpu_compute run) classifies as "other".
    """
    cl_id = cl_event["args"]["Python id"]
    for fc in parent_to_children.get(cl_id, []):
        if "forward" not in fc.get("name", ""):
            continue
        fwd_id = fc["args"]["Python id"]
        fwd_sub = parent_to_children.get(fwd_id, [])
        names = [s.get("name", "") for s in fwd_sub]
        if any("_mixed_cpu_attention" in n for n in names):
            return "cpu_mixed"
    return "other"


def identify_cpu_computation_groups(events, parent_to_children, token_boundaries):
    """
    CPU-COMPUTATION mode: find all 8-op groups whose own compute_layer AND the
    immediately following group's compute_layer both hit _mixed_cpu_attention
    (i.e. the first of a consecutive cpu_mixed->cpu_mixed pair -- this is the
    same "first of a same-typed consecutive pair" shape as --batched's
    mha_gen->mha_gen search, just with a different classifier, so it's
    implemented as a thin wrapper over identify_batched_groups). All prefill
    (token 0) groups are skipped, and the first qualifying pair across all
    decode tokens is skipped as warm-up -- identical conventions to --batched.
    """
    return identify_batched_groups(
        events, parent_to_children, token_boundaries,
        classify_fn=get_compute_layer_type_cpu_mixed,
        pair_type="cpu_mixed",
    )


def find_descendants_by_name(parent_to_children, root_python_id, name_substr):
    """
    BFS/DFS over the full descendant subtree of a python_function event
    (identified by its "Python id"), collecting every descendant whose name
    contains name_substr, sorted by timestamp.

    Needed (rather than a shallow 1-level check like
    get_compute_layer_type_sep's) because calls like general_copy/smart_copy/
    _attention_value are nested at varying depths below the op8 event that
    contains them (e.g. load_cache -> SelfAttention.load_cache -> general_copy),
    unlike the fixed compute_layer->forward->{mha_gen,mlp} depth used by the
    classifiers above.
    """
    result = []
    stack = list(parent_to_children.get(root_python_id, []))
    while stack:
        e = stack.pop()
        if name_substr in e.get("name", ""):
            result.append(e)
        child_id = e["args"].get("Python id")
        if child_id is not None:
            stack.extend(parent_to_children.get(child_id, []))
    result.sort(key=lambda e: e["ts"])
    return result


def _has_gpu_activity_in_window(cuda_rt_by_tid, cuda_rt_ts_by_tid, ts_start, ts_end, main_tid):
    """True if any cuda_runtime call was dispatched from main_tid in [ts_start, ts_end]."""
    return len(get_cuda_rt_in_window(cuda_rt_by_tid, cuda_rt_ts_by_tid, ts_start, ts_end, main_tid)) > 0


def extract_cpu_computation_metrics(grp, cuda_rt_by_tid, cuda_rt_ts_by_tid,
                                     gpu_event_list, gpu_event_ts,
                                     corr_to_gpu, ext_id_to_cpu_ops,
                                     corrs_with_cpu_rt,
                                     pin_mem_list, pin_mem_ts,
                                     parent_to_children, main_tid):
    """
    CPU-COMPUTATION mode metric extraction for a single cpu_mixed group.

    load_cache breakdown (SelfAttention.load_cache's path==3, the
    --cpu-gpu-compute branch):
      general_copy(k_buf, ..., k_home, ...); general_copy(v_buf, ..., v_home, ...)
      k_home.smart_copy(gpu, gpu_indices); v_home.smart_copy(gpu, gpu_indices)
    smart_copy() internally calls .copy() -> general_copy() itself, so a naive
    substring search for "general_copy" under load_cache finds 4 matches (2
    explicit CPU<->CPU copies + 2 nested inside the 2 smart_copy calls). Only
    the 2 explicit ones -- "outside of smart_copy" -- are recorded as
    cpu-copy-1/2; the smart_copy calls' own end-to-end latency is recorded
    separately as smart-copy-1/2, and the GPU-stream cudaMemcpyAsync ops they
    dispatch (H2D transfer of the GPU-resident cache slice) as
    load-cache-cudamemcpy-1/2, same mechanism as --batched's pin-memory/
    cudamemcpy extraction. Each smart_copy call also contains exactly one
    aten::pin_memory cpu_op nested inside it (verified on the sample trace:
    35/35 groups, both smart_copy calls each contain exactly 1) -- recorded
    per-call as pin-memory-1/2, mirroring --batched's own pin-memory
    extraction but scoped to each smart_copy's own window individually
    rather than the whole load_cache window (there are two distinct
    smart_copy calls here, so a whole-window search would conflate them).

    compute_layer breakdown:
      all CUDA-stream ops dispatched anywhere in compute_layer's window are
      collected as compute-cuda-N (no mha_gen/fwd_pre_mha origin tagging --
      that distinction doesn't cleanly apply once cpu_gpu_compute changes
      the call graph). _attention_value (pytorch_backend.py) is called twice
      per group: once from _mixed_cpu_attention (on tensors already moved to
      CPU via .float().cpu() -- no correlated GPU dispatch) and once from
      _mixed_gpu_attention (on GPU tensors -- dispatches CUDA kernels). The
      one with NO correlated GPU activity in its own window is the CPU-stream
      one; its own duration is recorded as attention-value-cpu.

    store_cache: unchanged from --batched (general_copy's actual DtoH copy is
    dispatched asynchronously after sync ends, via a sibling store_cache call
    under the same generation_loop parent -- see extract_batched_metrics).
    """
    evts = {e["name"].split(": ")[-1]: e for e in grp["events"]}

    def dur(key):
        e = evts.get(key)
        return round(e["dur"], 3) if e else None

    cl = grp["cl"]

    # --- load_hidden_compute cudaMemcpyAsync (only relevant when recompute_len > 0) ---
    lhc = evts.get("load_hidden_compute")
    lhc_gpu = []
    lhc_memcpy_dur = None
    if lhc:
        lhc_gpu = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            lhc["ts"], lhc["ts"] + lhc["dur"],
            main_tid, corr_to_gpu, filter_names={"cudaMemcpyAsync"}
        )
        if lhc_gpu:
            lhc_memcpy_dur = round(lhc_gpu[0]["dur"], 3)

    # --- load_cache breakdown ---
    lc = evts.get("load_cache")
    cpu_copy1 = cpu_copy2 = None
    smart_copy1 = smart_copy2 = None
    pin_mem1 = pin_mem2 = None
    lc_mc1 = lc_mc2 = None
    outer_general_copies, lc_gpu = [], []
    pm1, pm2 = [], []
    if lc:
        lc_id = lc["args"]["Python id"]
        smart_copies = find_descendants_by_name(parent_to_children, lc_id, "smart_copy")
        general_copies = find_descendants_by_name(parent_to_children, lc_id, "general_copy")
        sc_windows = [(s["ts"], s["ts"] + s["dur"]) for s in smart_copies]

        def _inside_any(e, windows):
            return any(w0 <= e["ts"] <= w1 for w0, w1 in windows)

        # "outside of smart_copy" -- excludes the general_copy calls nested
        # inside each smart_copy's own .copy() -> general_copy() chain.
        outer_general_copies = sorted(
            (g for g in general_copies if not _inside_any(g, sc_windows)),
            key=lambda e: e["ts"],
        )
        if len(outer_general_copies) > 0: cpu_copy1 = round(outer_general_copies[0]["dur"], 3)
        if len(outer_general_copies) > 1: cpu_copy2 = round(outer_general_copies[1]["dur"], 3)
        if len(smart_copies) > 0: smart_copy1 = round(smart_copies[0]["dur"], 3)
        if len(smart_copies) > 1: smart_copy2 = round(smart_copies[1]["dur"], 3)

        # aten::pin_memory nested inside each smart_copy call, individually.
        if len(smart_copies) > 0:
            pm1 = get_pin_memory_events_in_window(
                pin_mem_list, pin_mem_ts,
                smart_copies[0]["ts"], smart_copies[0]["ts"] + smart_copies[0]["dur"]
            )
            if pm1: pin_mem1 = round(pm1[0]["dur"], 3)
        if len(smart_copies) > 1:
            pm2 = get_pin_memory_events_in_window(
                pin_mem_list, pin_mem_ts,
                smart_copies[1]["ts"], smart_copies[1]["ts"] + smart_copies[1]["dur"]
            )
            if pm2: pin_mem2 = round(pm2[0]["dur"], 3)

        lc_gpu = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            lc["ts"], lc["ts"] + lc["dur"], main_tid, corr_to_gpu,
            filter_names={"cudaMemcpyAsync"}
        )
        if len(lc_gpu) > 0: lc_mc1 = round(lc_gpu[0]["dur"], 3)
        if len(lc_gpu) > 1: lc_mc2 = round(lc_gpu[1]["dur"], 3)

    # --- compute_layer breakdown ---
    cl_gpu_events = get_gpu_events_in_window(
        cuda_rt_by_tid, cuda_rt_ts_by_tid,
        cl["ts"], cl["ts"] + cl["dur"], main_tid, corr_to_gpu
    )
    attn_value_cpu_dur = None
    cl_id = cl["args"]["Python id"]
    attn_value_events = find_descendants_by_name(parent_to_children, cl_id, "_attention_value")
    cpu_side = [
        e for e in attn_value_events
        if not _has_gpu_activity_in_window(cuda_rt_by_tid, cuda_rt_ts_by_tid,
                                            e["ts"], e["ts"] + e["dur"], main_tid)
    ]
    if cpu_side:
        attn_value_cpu_dur = round(cpu_side[0]["dur"], 3)

    # --- store_cache: identical mechanism to --batched (see extract_batched_metrics) ---
    sync_e = evts.get("sync")
    sc_gpu = []
    sc_mc1 = sc_mc2 = None
    if sync_e is not None:
        sync_end = sync_e["ts"] + sync_e["dur"]
        async_sc_siblings = grp.get("async_sc_siblings", [])
        if async_sc_siblings:
            sc_ts_list = [s["ts"] for s in async_sc_siblings]
            lo = bisect.bisect_left(sc_ts_list, sync_end)
            if lo < len(async_sc_siblings):
                async_sc = async_sc_siblings[lo]
                sc_gpu = get_gpu_events_in_window(
                    cuda_rt_by_tid, cuda_rt_ts_by_tid,
                    async_sc["ts"], async_sc["ts"] + async_sc["dur"],
                    main_tid, corr_to_gpu, filter_names={"cudaMemcpyAsync"}
                )
                if len(sc_gpu) > 0: sc_mc1 = round(sc_gpu[0]["dur"], 3)
                if len(sc_gpu) > 1: sc_mc2 = round(sc_gpu[1]["dur"], 3)

    # -----------------------------------------------------------------
    # Overlap-aware breakdown (see extract_batched_metrics for the general
    # approach). In --cpu-computation mode, "GPU busy" is every CUDA-stream
    # op dispatched anywhere in compute_layer's window (cl_gpu_events --
    # no mha_gen/fwd_pre_mha origin split is available here). CPU
    # sub-categories add "internal cpu copy" (the 2 general_copy calls
    # outside of smart_copy) and "cpu mha compute" (the CPU-thread
    # _attention_value call) on top of --batched's pageable->pinned.
    # -----------------------------------------------------------------
    gpu_intervals = events_to_intervals(cl_gpu_events)

    kv_load_intervals  = events_to_intervals(lc_gpu[:2])
    kv_store_intervals = events_to_intervals(sc_gpu[:2])
    recompute_mc_intervals = events_to_intervals(lhc_gpu[:1])
    pcie_all_intervals = kv_load_intervals + kv_store_intervals + recompute_mc_intervals

    kv_load_total, kv_load_ov, kv_load_nonov = overlap_breakdown(kv_load_intervals, gpu_intervals)
    kv_store_total, kv_store_ov, kv_store_nonov = overlap_breakdown(kv_store_intervals, gpu_intervals)
    recompute_mc_total, recompute_mc_ov, recompute_mc_nonov = overlap_breakdown(recompute_mc_intervals, gpu_intervals)

    device_intervals = gpu_intervals + pcie_all_intervals  # GPU compute UNION all PCIe

    pin_events = ([pm1[0]] if pm1 else []) + ([pm2[0]] if pm2 else [])
    pin_intervals = events_to_intervals(pin_events)
    pin_total, pin_ov, pin_nonov = overlap_breakdown(pin_intervals, device_intervals)

    copy_intervals = events_to_intervals(outer_general_copies[:2])
    copy_total, copy_ov, copy_nonov = overlap_breakdown(copy_intervals, device_intervals)

    mha_cpu_intervals = events_to_intervals(cpu_side[:1])
    mha_cpu_total, mha_cpu_ov, mha_cpu_nonov = overlap_breakdown(mha_cpu_intervals, device_intervals)

    metrics = {
        "load_weight":                    dur("load_weight"),
        "load_hidden_compute":             dur("load_hidden_compute"),
        "load-hidden-compute-cudamemcpy":  lhc_memcpy_dur,
        "load_cache":                      dur("load_cache"),
        "cpu-copy-1":                      cpu_copy1,
        "cpu-copy-2":                      cpu_copy2,
        "smart-copy-1":                    smart_copy1,
        "smart-copy-2":                    smart_copy2,
        "pin-memory-1":                    pin_mem1,
        "pin-memory-2":                    pin_mem2,
        "load-cache-cudamemcpy-1":         lc_mc1,
        "load-cache-cudamemcpy-2":         lc_mc2,
        "load_hidden":                     dur("load_hidden"),
        "compute_layer":                   dur("compute_layer"),
        "attention-value-cpu":             attn_value_cpu_dur,
        "store_cache":                     dur("store_cache"),
        "store-cache-cudamemcpy-1":        sc_mc1,
        "store-cache-cudamemcpy-2":        sc_mc2,
        "store_hidden":                    dur("store_hidden"),
        "sync":                            dur("sync"),
        # --- overlap-aware PCIe breakdown (all in us) ---
        "pcie-kv-load-total-us":              kv_load_total,
        "pcie-kv-load-overlapped-us":         kv_load_ov,
        "pcie-kv-load-nonoverlapped-us":      kv_load_nonov,
        "pcie-kv-store-total-us":             kv_store_total,
        "pcie-kv-store-overlapped-us":        kv_store_ov,
        "pcie-kv-store-nonoverlapped-us":     kv_store_nonov,
        "pcie-recompute-memcpy-total-us":         recompute_mc_total,
        "pcie-recompute-memcpy-overlapped-us":    recompute_mc_ov,
        "pcie-recompute-memcpy-nonoverlapped-us": recompute_mc_nonov,
        # --- overlap-aware CPU (misc) breakdown (all in us) ---
        "cpu-pageable-pinned-total-us":         pin_total,
        "cpu-pageable-pinned-overlapped-us":    pin_ov,
        "cpu-pageable-pinned-nonoverlapped-us": pin_nonov,
        "cpu-internal-copy-total-us":           copy_total,
        "cpu-internal-copy-overlapped-us":      copy_ov,
        "cpu-internal-copy-nonoverlapped-us":   copy_nonov,
        "cpu-mha-compute-total-us":             mha_cpu_total,
        "cpu-mha-compute-overlapped-us":        mha_cpu_ov,
        "cpu-mha-compute-nonoverlapped-us":     mha_cpu_nonov,
    }
    for idx, gpu_ev in enumerate(cl_gpu_events, 1):
        metrics[f"compute-cuda-{idx}"] = round(gpu_ev["dur"], 3)

    return metrics


# ---------------------------------------------------------------------------
# SEP metric extraction
# ---------------------------------------------------------------------------

def extract_mha_gen_metrics_sep(grp, cuda_rt_by_tid, cuda_rt_ts_by_tid,
                                 gpu_event_list, gpu_event_ts,
                                 corr_to_gpu, ext_id_to_cpu_ops,
                                 corrs_with_cpu_rt, parent_to_children, main_tid):
    evts = {e["name"].split(": ")[-1]: e for e in grp["events"]}
    def dur(key):
        e = evts.get(key)
        return round(e["dur"], 3) if e else None

    cl = grp["cl"]
    tagged_gpu_ops = get_gpu_events_in_forward_sep(
        cuda_rt_by_tid, cuda_rt_ts_by_tid,
        gpu_event_list, gpu_event_ts,
        cl, parent_to_children,
        main_tid, corr_to_gpu, ext_id_to_cpu_ops, corrs_with_cpu_rt
    )
    metrics = {
        "mha-gen_load_weight":         dur("load_weight"),
        "mha-gen_load_hidden_compute":  dur("load_hidden_compute"),
        "mha-gen_load_cache":           dur("load_cache"),
        "mha-gen_load_hidden":          dur("load_hidden"),
        "mha-gen_compute_layer":        dur("compute_layer"),
        "mha-gen_store_cache":          dur("store_cache"),
        "mha-gen_store_hidden":         dur("store_hidden"),
        "mha-gen_sync":                 dur("sync"),
    }
    for idx, (gpu_ev, origin) in enumerate(tagged_gpu_ops, 1):
        metrics[f"mha-gen-compute-cuda-{idx}"] = round(gpu_ev["dur"], 3)
        metrics[f"mha-gen-compute-cuda-{idx}-origin"] = origin
    return metrics


def extract_mlp_metrics_sep(grp, cuda_rt_by_tid, cuda_rt_ts_by_tid,
                              corr_to_gpu, ext_id_to_cpu_ops,
                              pin_mem_list, pin_mem_ts,
                              parent_to_children, main_tid):
    evts = {e["name"].split(": ")[-1]: e for e in grp["events"]}
    def dur(key):
        e = evts.get(key)
        return round(e["dur"], 3) if e else None

    cl = grp["cl"]
    cl_ts_s, cl_ts_e = cl["ts"], cl["ts"] + cl["dur"]
    mlp_fn = get_inner_pyfn(
        cl, parent_to_children,
        lambda n: "mlp" in n and "pytorch_backend" in n
    )

    lhc = evts.get("load_hidden_compute")
    lhc_memcpy_dur = None
    if lhc:
        lhc_gpu = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            lhc["ts"], lhc["ts"] + lhc["dur"],
            main_tid, corr_to_gpu, filter_names={"cudaMemcpyAsync"}
        )
        if lhc_gpu:
            lhc_memcpy_dur = round(lhc_gpu[0]["dur"], 3)

    lc = evts.get("load_cache")
    pm1, pm2, lc_mc1, lc_mc2 = None, None, None, None
    if lc:
        lc_ts_s, lc_ts_e = lc["ts"], lc["ts"] + lc["dur"]
        pins = get_pin_memory_events_in_window(pin_mem_list, pin_mem_ts,
                                               lc_ts_s, lc_ts_e)
        if len(pins) > 0: pm1 = round(pins[0]["dur"], 3)
        if len(pins) > 1: pm2 = round(pins[1]["dur"], 3)
        lc_gpu = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            lc_ts_s, lc_ts_e, main_tid, corr_to_gpu,
            filter_names={"cudaMemcpyAsync"}
        )
        if len(lc_gpu) > 0: lc_mc1 = round(lc_gpu[0]["dur"], 3)
        if len(lc_gpu) > 1: lc_mc2 = round(lc_gpu[1]["dur"], 3)

    if mlp_fn is not None:
        cl_gpu_events = get_gpu_events_via_cpu_chain(
            cuda_rt_by_tid, cuda_rt_ts_by_tid, mlp_fn,
            cl_ts_s, cl_ts_e, main_tid, corr_to_gpu, ext_id_to_cpu_ops,
            exclude_cats={"gpu_memset"}
        )
    else:
        cl_gpu_events = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            cl_ts_s, cl_ts_e, main_tid, corr_to_gpu
        )

    sc = evts.get("store_cache")
    sc_mc1, sc_mc2 = None, None
    if sc:
        sc_gpu = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            sc["ts"], sc["ts"] + sc["dur"],
            main_tid, corr_to_gpu, filter_names={"cudaMemcpyAsync"}
        )
        if len(sc_gpu) > 0: sc_mc1 = round(sc_gpu[0]["dur"], 3)
        if len(sc_gpu) > 1: sc_mc2 = round(sc_gpu[1]["dur"], 3)

    metrics = {
        "mlp_load_weight":                dur("load_weight"),
        "mlp_load_hidden_compute":         dur("load_hidden_compute"),
        "load-hidden-compute-cudamemcpy":  lhc_memcpy_dur,
        "mlp_load_cache":                  dur("load_cache"),
        "pin-memory-1":                    pm1,
        "pin-memory-2":                    pm2,
        "load-cache-cudamemcpy-1":         lc_mc1,
        "load-cache-cudamemcpy-2":         lc_mc2,
        "mlp_load_hidden":                 dur("load_hidden"),
        "mlp_compute_layer":               dur("compute_layer"),
        "mlp_store_cache":                 dur("store_cache"),
        "store-cache-cudamemcpy-1":        sc_mc1,
        "store-cache-cudamemcpy-2":        sc_mc2,
        "mlp_store_hidden":                dur("store_hidden"),
        "mlp_sync":                        dur("sync"),
    }
    for idx, gpu_ev in enumerate(cl_gpu_events, 1):
        metrics[f"mlp-compute-cuda-{idx}"] = round(gpu_ev["dur"], 3)
    return metrics


# ---------------------------------------------------------------------------
# NOSEP metric extraction
# ---------------------------------------------------------------------------

def extract_nosep_metrics(grp, cuda_rt_by_tid, cuda_rt_ts_by_tid,
                           gpu_event_list, gpu_event_ts,
                           corr_to_gpu, ext_id_to_cpu_ops,
                           corrs_with_cpu_rt,
                           pin_mem_list, pin_mem_ts,
                           parent_to_children, main_tid):
    evts = {e["name"].split(": ")[-1]: e for e in grp["events"]}
    def dur(key):
        e = evts.get(key)
        return round(e["dur"], 3) if e else None

    cl = grp["cl"]
    cl_ts_s, cl_ts_e = cl["ts"], cl["ts"] + cl["dur"]

    # --- load_hidden_compute cuda memcpy ---
    lhc = evts.get("load_hidden_compute")
    lhc_memcpy_dur = None
    if lhc:
        lhc_gpu = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            lhc["ts"], lhc["ts"] + lhc["dur"],
            main_tid, corr_to_gpu, filter_names={"cudaMemcpyAsync"}
        )
        if lhc_gpu:
            lhc_memcpy_dur = round(lhc_gpu[0]["dur"], 3)

    # --- load_cache: pin_memory x2, cudaMemcpyAsync x2 ---
    lc = evts.get("load_cache")
    pm1, pm2, lc_mc1, lc_mc2 = None, None, None, None
    if lc:
        lc_ts_s, lc_ts_e = lc["ts"], lc["ts"] + lc["dur"]
        pins = get_pin_memory_events_in_window(pin_mem_list, pin_mem_ts,
                                               lc_ts_s, lc_ts_e)
        if len(pins) > 0: pm1 = round(pins[0]["dur"], 3)
        if len(pins) > 1: pm2 = round(pins[1]["dur"], 3)
        lc_gpu = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            lc_ts_s, lc_ts_e, main_tid, corr_to_gpu,
            filter_names={"cudaMemcpyAsync"}
        )
        if len(lc_gpu) > 0: lc_mc1 = round(lc_gpu[0]["dur"], 3)
        if len(lc_gpu) > 1: lc_mc2 = round(lc_gpu[1]["dur"], 3)

    # --- compute_layer: 3-way tagged GPU ops ---
    tagged_gpu_ops = get_gpu_events_in_forward_nosep(
        cuda_rt_by_tid, cuda_rt_ts_by_tid,
        gpu_event_list, gpu_event_ts,
        cl, parent_to_children,
        main_tid, corr_to_gpu, ext_id_to_cpu_ops, corrs_with_cpu_rt
    )

    # --- store_cache: cudaMemcpyAsync x2 ---
    sc = evts.get("store_cache")
    sc_mc1, sc_mc2 = None, None
    if sc:
        sc_gpu = get_gpu_events_in_window(
            cuda_rt_by_tid, cuda_rt_ts_by_tid,
            sc["ts"], sc["ts"] + sc["dur"],
            main_tid, corr_to_gpu, filter_names={"cudaMemcpyAsync"}
        )
        if len(sc_gpu) > 0: sc_mc1 = round(sc_gpu[0]["dur"], 3)
        if len(sc_gpu) > 1: sc_mc2 = round(sc_gpu[1]["dur"], 3)

    metrics = {
        "load_weight":                    dur("load_weight"),
        "load_hidden_compute":             dur("load_hidden_compute"),
        "load-hidden-compute-cudamemcpy":  lhc_memcpy_dur,
        "load_cache":                      dur("load_cache"),
        "pin-memory-1":                    pm1,
        "pin-memory-2":                    pm2,
        "load-cache-cudamemcpy-1":         lc_mc1,
        "load-cache-cudamemcpy-2":         lc_mc2,
        "load_hidden":                     dur("load_hidden"),
        "compute_layer":                   dur("compute_layer"),
        "store_cache":                     dur("store_cache"),
        "store-cache-cudamemcpy-1":        sc_mc1,
        "store-cache-cudamemcpy-2":        sc_mc2,
        "store_hidden":                    dur("store_hidden"),
        "sync":                            dur("sync"),
    }
    for idx, (gpu_ev, origin) in enumerate(tagged_gpu_ops, 1):
        metrics[f"compute-cuda-{idx}"] = round(gpu_ev["dur"], 3)
        metrics[f"compute-cuda-{idx}-origin"] = origin

    return metrics


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_trace(trace_path, max_groups=None, output_path=None, nosep=False, batched=False,
                   cpu_computation=False):
    print(f"Loading trace: {trace_path}", file=sys.stderr)
    with open(trace_path) as f:
        data = json.load(f)
    events = data["traceEvents"]
    print(f"  {len(events)} events loaded.", file=sys.stderr)

    print("Building indices...", file=sys.stderr)
    (parent_to_children, py_id_map, corr_to_gpu, ext_id_to_cpu_ops,
     corrs_with_cpu_rt,
     cuda_rt_by_tid, cuda_rt_ts_by_tid,
     pin_mem_list, pin_mem_ts,
     gpu_event_list, gpu_event_ts) = build_indices(events)
    main_tid = detect_main_tid(events)
    print(f"  Main thread TID: {main_tid}", file=sys.stderr)

    print("Detecting token boundaries...", file=sys.stderr)
    token_boundaries = get_token_boundaries(events)
    print(f"  {len(token_boundaries)} tokens found.", file=sys.stderr)

    rows = []

    if cpu_computation:
        print("Mode: CPU-COMPUTATION — identifying first-of-consecutive-cpu_mixed groups...",
              file=sys.stderr)
        groups = identify_cpu_computation_groups(events, parent_to_children, token_boundaries)
        print(f"  {len(groups)} cpu-computation groups (warm-up skipped).", file=sys.stderr)
        if max_groups is not None:
            groups = groups[:max_groups]
            print(f"  Limiting to first {max_groups} groups.", file=sys.stderr)

        for group_num, grp in enumerate(groups, 1):
            metrics = extract_cpu_computation_metrics(
                grp, cuda_rt_by_tid, cuda_rt_ts_by_tid,
                gpu_event_list, gpu_event_ts,
                corr_to_gpu, ext_id_to_cpu_ops,
                corrs_with_cpu_rt,
                pin_mem_list, pin_mem_ts,
                parent_to_children, main_tid
            )
            rows.append({"group": group_num, "token": grp["token"], **metrics})

    elif batched:
        print("Mode: BATCHED — identifying first-of-consecutive-mha_gen groups...",
              file=sys.stderr)
        groups = identify_batched_groups(events, parent_to_children, token_boundaries)
        print(f"  {len(groups)} batched groups (warm-up skipped).", file=sys.stderr)
        if max_groups is not None:
            groups = groups[:max_groups]
            print(f"  Limiting to first {max_groups} groups.", file=sys.stderr)

        for group_num, grp in enumerate(groups, 1):
            metrics = extract_batched_metrics(
                grp, cuda_rt_by_tid, cuda_rt_ts_by_tid,
                gpu_event_list, gpu_event_ts,
                corr_to_gpu, ext_id_to_cpu_ops,
                corrs_with_cpu_rt,
                pin_mem_list, pin_mem_ts,
                parent_to_children, main_tid
            )
            rows.append({"group": group_num, "token": grp["token"], **metrics})

    elif nosep:
        print("Mode: NOSEP — identifying merged mha+mlp groups...", file=sys.stderr)
        groups = identify_nosep_groups(events, parent_to_children, token_boundaries)
        print(f"  {len(groups)} nosep groups (warm-up skipped).", file=sys.stderr)
        if max_groups is not None:
            groups = groups[:max_groups]
            print(f"  Limiting to first {max_groups} groups.", file=sys.stderr)

        for group_num, grp in enumerate(groups, 1):
            metrics = extract_nosep_metrics(
                grp, cuda_rt_by_tid, cuda_rt_ts_by_tid,
                gpu_event_list, gpu_event_ts,
                corr_to_gpu, ext_id_to_cpu_ops,
                corrs_with_cpu_rt,
                pin_mem_list, pin_mem_ts,
                parent_to_children, main_tid
            )
            rows.append({"group": group_num, "token": grp["token"], **metrics})

    else:
        print("Mode: SEP — identifying mha_gen->mlp supergroups...", file=sys.stderr)
        supergroups = identify_supergroups_sep(events, parent_to_children, token_boundaries)
        print(f"  {len(supergroups)} supergroups (warm-up skipped).", file=sys.stderr)
        if max_groups is not None:
            supergroups = supergroups[:max_groups]
            print(f"  Limiting to first {max_groups} supergroups.", file=sys.stderr)

        for sg_num, sg in enumerate(supergroups, 1):
            mha_metrics = extract_mha_gen_metrics_sep(
                sg["mha_gen"], cuda_rt_by_tid, cuda_rt_ts_by_tid,
                gpu_event_list, gpu_event_ts,
                corr_to_gpu, ext_id_to_cpu_ops,
                corrs_with_cpu_rt, parent_to_children, main_tid
            )
            mlp_metrics = extract_mlp_metrics_sep(
                sg["mlp"], cuda_rt_by_tid, cuda_rt_ts_by_tid,
                corr_to_gpu, ext_id_to_cpu_ops,
                pin_mem_list, pin_mem_ts,
                parent_to_children, main_tid
            )
            rows.append({
                "supergroup": sg_num, "token": sg["token"],
                **mha_metrics, **mlp_metrics,
            })

    if not rows:
        print("No rows to write.", file=sys.stderr)
        return rows

    fixed_cols = list(rows[0].keys())[:2]  # group/supergroup + token
    seen = set(fixed_cols)
    dynamic_cols = []
    for row in rows:
        for k in row:
            if k not in seen:
                dynamic_cols.append(k)
                seen.add(k)
    all_cols = fixed_cols + dynamic_cols

    if output_path is None:
        if cpu_computation:
            suffix = "_cpu_computation_analysis.csv"
        elif batched:
            suffix = "_batched_analysis.csv"
        elif nosep:
            suffix = "_nosep_analysis.csv"
        else:
            suffix = "_analysis.csv"
        output_path = Path(trace_path).stem + suffix
    output_path = Path(output_path)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in all_cols})

    print(f"Written {len(rows)} rows to {output_path}", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze torch profile trace — mha_gen/mlp latencies."
    )
    parser.add_argument("trace", help="Path to the torch profile JSON trace file.")
    parser.add_argument(
        "--max-groups", type=int, default=None, metavar="N",
        help="Limit to first N groups/supergroups (default: all).",
    )
    parser.add_argument(
        "--out", default=None, metavar="OUTPUT.csv",
        help="Output CSV path.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--nosep", action="store_true",
        help="Use nosep mode: merged mha+mlp compute_layer.",
    )
    mode_group.add_argument(
        "--batched", action="store_true",
        help="Use batched mode: first group of each consecutive mha_gen->mha_gen pair.",
    )
    mode_group.add_argument(
        "--cpu-computation", action="store_true",
        help="Use cpu-computation mode: first group of each consecutive "
             "cpu_mixed->cpu_mixed pair (traces collected with "
             "flex_opt_kvpr.py's --cpu-gpu-compute).",
    )
    args = parser.parse_args()
    analyze_trace(args.trace, max_groups=args.max_groups,
                  output_path=args.out, nosep=args.nosep, batched=args.batched,
                  cpu_computation=args.cpu_computation)


if __name__ == "__main__":
    main()