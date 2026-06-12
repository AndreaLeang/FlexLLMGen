#!/usr/bin/env python3
"""
Torch profile trace analyzer for FlexLLMGen / flex_opt_kvpr traces.

Two modes:

--sep (default):
  Each output row is a supergroup = consecutive mha_gen -> mlp pair from decode.
  Supergroup 0 (warm-up) and all prefill groups (token 0) are skipped.
  mha_gen compute CUDA ops tagged 'mha_gen' or 'fwd_pre_mha' by cpu_op attribution.

--nosep:
  Each output row is a single nosep group from decode (token >= 1).
  Group 0 (warm-up) is skipped.
  compute_layer contains both mha_gen and mlp in one forward pass.
  CUDA ops tagged: 'fwd_pre' (before mha_gen in forward), 'mha_gen', or 'mlp'.

Usage:
    python analyze_trace.py <trace.json> [--max-groups N] [--out output.csv] [--nosep]

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
    cuda_rt_by_tid = {}   # tid -> [event, ...]  (cuda_runtime events only)
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
        if corr is not None and cat == "cuda_runtime":
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


def get_cuda_rt_in_window(cuda_rt_by_tid, cuda_rt_ts_by_tid, ts_start, ts_end, main_tid):
    """Return cuda_runtime events for main_tid in [ts_start, ts_end] using bisect."""
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

def analyze_trace(trace_path, max_groups=None, output_path=None, nosep=False):
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

    if nosep:
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
        suffix = "_nosep_analysis.csv" if nosep else "_analysis.csv"
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
    parser.add_argument(
        "--nosep", action="store_true",
        help="Use nosep mode: merged mha+mlp compute_layer.",
    )
    args = parser.parse_args()
    analyze_trace(args.trace, max_groups=args.max_groups,
                  output_path=args.out, nosep=args.nosep)


if __name__ == "__main__":
    main()