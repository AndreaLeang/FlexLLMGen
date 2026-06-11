#!/usr/bin/env python3
"""
Torch profile trace analyzer for FlexLLMGen / flex_opt_kvpr traces.

Each output row is a supergroup = consecutive mha_gen -> mlp pair from decode.
Supergroup 0 (warm-up) and all prefill groups (token 0) are skipped.

mha_gen compute CUDA ops are tagged by origin via cpu_op attribution:
  - "mha_gen"     : cpu_op dispatching the kernel falls within mha_gen's Python window
  - "fwd_pre_mha" : cpu_op falls within forward() but before mha_gen
  - "no_cpu_op"   : no cuda_runtime record (cutlass cudaLaunchKernelEx path);
                    these are classified fwd_pre_mha based on GPU execution time
                    (they execute before mha_gen starts)
Both are included in the CSV as mha-gen-compute-cuda-N, with a separate
mha-gen-compute-cuda-N-origin column recording the tag.

Usage:
    python analyze_trace.py <trace.json> [--max-groups N] [--out output.csv]

All durations in microseconds (us).
"""

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

    for e in events:
        args = e.get("args", {})
        py_id = args.get("Python id")
        parent_id = args.get("Python parent id")
        corr = args.get("correlation")
        ext_id = args.get("External id")

        if py_id is not None:
            py_id_map[py_id] = e
        if parent_id is not None:
            parent_to_children.setdefault(parent_id, []).append(e)
        if corr is not None and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            corr_to_gpu[corr] = e
        if ext_id is not None and e.get("cat") == "cpu_op":
            ext_id_to_cpu_ops.setdefault(ext_id, []).append(e)
        if corr is not None and e.get("cat") == "cuda_runtime":
            corrs_with_cpu_rt.add(corr)

    return parent_to_children, py_id_map, corr_to_gpu, ext_id_to_cpu_ops, corrs_with_cpu_rt


def detect_main_tid(events):
    timer_tids = [e["tid"] for e in events if e.get("name") == "timer.py(20): start"]
    if timer_tids:
        return timer_tids[0]
    tid_counts = Counter(e["tid"] for e in events if e.get("cat") == "python_function")
    return tid_counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OP8_NAMES = {
    "load_weight", "load_hidden_compute", "load_cache",
    "store_hidden", "load_hidden", "compute_layer",
    "store_cache", "sync",
}


def is_op8(event):
    return any(op in event.get("name", "") for op in OP8_NAMES)


def get_compute_layer_type(cl_event, parent_to_children):
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


def get_cuda_rt_in_window(events, ts_start, ts_end, main_tid):
    return [
        e for e in events
        if e.get("cat") == "cuda_runtime"
        and e.get("tid") == main_tid
        and ts_start <= e["ts"] <= ts_end
    ]


def get_gpu_events_in_forward(events, cl_event, parent_to_children,
                               main_tid, corr_to_gpu, ext_id_to_cpu_ops,
                               corrs_with_cpu_rt):
    """
    All GPU events for mha_gen compute_layer -> forward(), tagged by origin.

    Returns list of (gpu_event, origin_tag) sorted by GPU timestamp.

    Origin tagging uses cpu_op attribution (authoritative):
      - For ops with a cuda_runtime record: check whether the originating
        cpu_op's timestamp falls within mha_gen's Python window.
      - For ops with no cuda_runtime record (cutlass cudaLaunchKernelEx):
        tagged fwd_pre_mha since they execute before mha_gen starts on GPU.

    Origin values:
      'mha_gen'     - dispatched from within mha_gen Python function
      'fwd_pre_mha' - dispatched from forward() before mha_gen
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

    # Method 1: cpu_op -> cuda_rt -> GPU within forward CPU window
    cuda_rt_in_fwd = get_cuda_rt_in_window(
        events, fwd_fn["ts"], fwd_fn["ts"] + fwd_fn["dur"], main_tid
    )
    results = []
    found_corrs = set()
    for cr in cuda_rt_in_fwd:
        corr = cr.get("args", {}).get("correlation")
        ext_id = cr.get("args", {}).get("External id")
        if corr is None:
            continue
        gpu_ev = corr_to_gpu.get(corr)
        if gpu_ev is None:
            continue

        # Determine origin via cpu_op timestamp
        cpu_ops = ext_id_to_cpu_ops.get(ext_id, [])
        if cpu_ops:
            cpu_op = min(cpu_ops, key=lambda e: abs(e["ts"] - cr["ts"]))
            in_mha = mhag_ts_s <= cpu_op["ts"] <= mhag_ts_e
            origin = "mha_gen" if in_mha else "fwd_pre_mha"
        else:
            origin = "fwd_pre_mha"  # no cpu_op = fwd_pre_mha by convention

        results.append((gpu_ev, origin))
        found_corrs.add(corr)

    # Method 2: GPU-side window fallback for cutlass kernels (no cuda_runtime record)
    if results:
        gpu_ts_min = min(r[0]["ts"] for r in results)
        gpu_ts_max = max(r[0]["ts"] + r[0]["dur"] for r in results)
    else:
        gpu_ts_min = fwd_fn["ts"]
        gpu_ts_max = fwd_fn["ts"] + fwd_fn["dur"]

    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        corr = e.get("args", {}).get("correlation")
        if corr is None or corr in found_corrs:
            continue
        if corr in corrs_with_cpu_rt:
            continue
        if gpu_ts_min <= e["ts"] <= gpu_ts_max + 100:
            # These have no cuda_runtime record; they execute before mha_gen starts
            origin = "fwd_pre_mha"
            results.append((e, origin))
            found_corrs.add(corr)

    results.sort(key=lambda x: x[0]["ts"])
    return results


def get_gpu_events_via_cpu_chain(events, pyfn_event, cl_ts_s, cl_ts_e,
                                  main_tid, corr_to_gpu, ext_id_to_cpu_ops,
                                  exclude_cats=None):
    exclude_cats = exclude_cats or set()
    pyfn_ts_s = pyfn_event["ts"]
    pyfn_ts_e = pyfn_ts_s + pyfn_event["dur"]

    results = []
    for cr in get_cuda_rt_in_window(events, cl_ts_s, cl_ts_e, main_tid):
        ext_id = cr.get("args", {}).get("External id")
        corr = cr.get("args", {}).get("correlation")
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


def get_gpu_events_in_window(events, ts_start, ts_end, main_tid, corr_to_gpu,
                              filter_names=None):
    results = []
    for cr in get_cuda_rt_in_window(events, ts_start, ts_end, main_tid):
        if filter_names and cr.get("name") not in filter_names:
            continue
        corr = cr.get("args", {}).get("correlation")
        if corr is None:
            continue
        gpu_ev = corr_to_gpu.get(corr)
        if gpu_ev is not None:
            results.append(gpu_ev)
    return results


def get_pin_memory_events_in_window(events, ts_start, ts_end):
    return sorted(
        [e for e in events
         if e.get("name") == "aten::pin_memory"
         and e.get("cat") == "cpu_op"
         and ts_start <= e["ts"] <= ts_end],
        key=lambda e: e["ts"],
    )


# ---------------------------------------------------------------------------
# Token boundaries and supergroup identification
# ---------------------------------------------------------------------------

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


def identify_supergroups(events, parent_to_children, token_boundaries):
    """
    Identify consecutive mha_gen -> mlp pairs (supergroups) from decode tokens.
    Skips all prefill (token 0) groups and the first decode pair (warm-up).
    """
    gen_loop_candidates = [
        e for e in events
        if "generation_loop_overlap_single_batch" in e.get("name", "")
        and e.get("cat") == "python_function"
    ]
    if not gen_loop_candidates:
        raise RuntimeError("Could not find generation_loop_overlap_single_batch event.")
    gen_loop_id = gen_loop_candidates[0]["args"]["Python id"]

    gen_children = sorted(
        parent_to_children.get(gen_loop_id, []), key=lambda e: e["ts"]
    )

    all_groups = []
    for ti, (ts_start, ts_end) in enumerate(token_boundaries):
        token_8ops = [
            e for e in gen_children
            if is_op8(e) and ts_start <= e["ts"] <= ts_end
        ]
        i = 0
        while i + 7 < len(token_8ops):
            grp = token_8ops[i:i + 8]
            cl = next((e for e in grp if "compute_layer" in e.get("name", "")), None)
            if cl is not None:
                ftype = get_compute_layer_type(cl, parent_to_children)
                all_groups.append({
                    "token": ti, "type": ftype,
                    "events": grp, "cl": cl
                })
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

    return supergroups[1:]  # skip warm-up pair


# ---------------------------------------------------------------------------
# Per-group metric extraction
# ---------------------------------------------------------------------------

def extract_mha_gen_metrics(grp, events, corr_to_gpu, ext_id_to_cpu_ops,
                             corrs_with_cpu_rt, parent_to_children, main_tid):
    evts = {e["name"].split(": ")[-1]: e for e in grp["events"]}

    def dur(key):
        e = evts.get(key)
        return round(e["dur"], 3) if e else None

    cl = grp["cl"]
    tagged_gpu_ops = get_gpu_events_in_forward(
        events, cl, parent_to_children,
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


def extract_mlp_metrics(grp, events, corr_to_gpu, ext_id_to_cpu_ops,
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
            events, lhc["ts"], lhc["ts"] + lhc["dur"],
            main_tid, corr_to_gpu, filter_names={"cudaMemcpyAsync"}
        )
        if lhc_gpu:
            lhc_memcpy_dur = round(lhc_gpu[0]["dur"], 3)

    lc = evts.get("load_cache")
    pm1, pm2, lc_mc1, lc_mc2 = None, None, None, None
    if lc:
        lc_ts_s, lc_ts_e = lc["ts"], lc["ts"] + lc["dur"]
        pins = get_pin_memory_events_in_window(events, lc_ts_s, lc_ts_e)
        if len(pins) > 0: pm1 = round(pins[0]["dur"], 3)
        if len(pins) > 1: pm2 = round(pins[1]["dur"], 3)
        lc_gpu = get_gpu_events_in_window(
            events, lc_ts_s, lc_ts_e, main_tid, corr_to_gpu,
            filter_names={"cudaMemcpyAsync"}
        )
        if len(lc_gpu) > 0: lc_mc1 = round(lc_gpu[0]["dur"], 3)
        if len(lc_gpu) > 1: lc_mc2 = round(lc_gpu[1]["dur"], 3)

    if mlp_fn is not None:
        cl_gpu_events = get_gpu_events_via_cpu_chain(
            events, mlp_fn, cl_ts_s, cl_ts_e,
            main_tid, corr_to_gpu, ext_id_to_cpu_ops,
            exclude_cats={"gpu_memset"}
        )
    else:
        cl_gpu_events = get_gpu_events_in_window(
            events, cl_ts_s, cl_ts_e, main_tid, corr_to_gpu
        )

    sc = evts.get("store_cache")
    sc_mc1, sc_mc2 = None, None
    if sc:
        sc_gpu = get_gpu_events_in_window(
            events, sc["ts"], sc["ts"] + sc["dur"],
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
# Main analysis
# ---------------------------------------------------------------------------

def analyze_trace(trace_path, max_groups=None, output_path=None):
    print(f"Loading trace: {trace_path}", file=sys.stderr)
    with open(trace_path) as f:
        data = json.load(f)
    events = data["traceEvents"]
    print(f"  {len(events)} events loaded.", file=sys.stderr)

    print("Building indices...", file=sys.stderr)
    parent_to_children, py_id_map, corr_to_gpu, ext_id_to_cpu_ops, corrs_with_cpu_rt = \
        build_indices(events)
    main_tid = detect_main_tid(events)
    print(f"  Main thread TID: {main_tid}", file=sys.stderr)

    print("Detecting token boundaries...", file=sys.stderr)
    token_boundaries = get_token_boundaries(events)
    print(f"  {len(token_boundaries)} tokens found.", file=sys.stderr)

    print("Identifying mha_gen->mlp supergroups...", file=sys.stderr)
    supergroups = identify_supergroups(events, parent_to_children, token_boundaries)
    print(f"  {len(supergroups)} supergroups (warm-up skipped).", file=sys.stderr)

    if max_groups is not None:
        supergroups = supergroups[:max_groups]
        print(f"  Limiting to first {max_groups} supergroups.", file=sys.stderr)

    rows = []
    for sg_num, sg in enumerate(supergroups, 1):
        mha_metrics = extract_mha_gen_metrics(
            sg["mha_gen"], events, corr_to_gpu, ext_id_to_cpu_ops,
            corrs_with_cpu_rt, parent_to_children, main_tid
        )
        mlp_metrics = extract_mlp_metrics(
            sg["mlp"], events, corr_to_gpu, ext_id_to_cpu_ops,
            parent_to_children, main_tid
        )
        rows.append({
            "supergroup": sg_num,
            "token": sg["token"],
            **mha_metrics,
            **mlp_metrics,
        })

    if not rows:
        print("No rows to write.", file=sys.stderr)
        return rows

    fixed_cols = ["supergroup", "token"]
    seen = set(fixed_cols)
    dynamic_cols = []
    for row in rows:
        for k in row:
            if k not in seen:
                dynamic_cols.append(k)
                seen.add(k)
    all_cols = fixed_cols + dynamic_cols

    if output_path is None:
        output_path = Path(trace_path).stem + "_analysis.csv"
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
        description="Analyze torch profile trace — mha_gen/mlp supergroup latencies."
    )
    parser.add_argument("trace", help="Path to the torch profile JSON trace file.")
    parser.add_argument(
        "--max-groups", type=int, default=None, metavar="N",
        help="Limit to first N supergroups (default: all).",
    )
    parser.add_argument(
        "--out", default=None, metavar="OUTPUT.csv",
        help="Output CSV path (default: <trace_stem>_analysis.csv).",
    )
    args = parser.parse_args()
    analyze_trace(args.trace, max_groups=args.max_groups, output_path=args.out)


if __name__ == "__main__":
    main()