#!/usr/bin/env python3
"""
gt_vs_estimator.py
==================
Compare ground-truth latency breakdowns (from PyTorch profiler traces) against
estimator predictions from kv_schedule_optimization.py.

Pipeline per experiment
-----------------------
  1. Call get_available_offloadings() to find the minimum feasible offload % for
     each batch size.
  2. Invoke flexllmgen/flex_opt_kvpr.py (via subprocess) with --profile to record
     a PyTorch trace JSON.
  3. Run trace_analyzer.py (--batched mode) then trace_result_analyzer.py on the
     JSON to produce a summary CSV.
  4. Read the summary CSV and build the ground-truth latency segment dict.
  5. Call layer_prediction() directly on the "representative" MHA layer to get raw
     estimator component latencies (in seconds, converted to µs).
  6. Write one row per experiment to a comparison CSV.

Segment name mapping
--------------------
  Estimator component_breakdown indices → canonical segment names:
    [0]  PtP_1           → "PinnedMemory CPU (phase1)"
    [1]  PtP_2           → "PinnedMemory CPU (phase2)"
    [2]  Recomp Transfer → "Recompute Load"
    [3]  Recomp Calc     → "Recompute CUDA"
    [4]  Layer Calc      → "MHA CUDA"
    [5]  KV Load K       → "KVCache Load K"
    [6]  KV Load V       → "KVCache Load V"

  Ground-truth (batched mode) → canonical segment names (from build_segments_batched):
    "PinnedMemory CPU"  → split here as "PinnedMemory CPU (phase1+2)" (combined)
    "Recompute Load"    → "Recompute Load"
    "Recompute CUDA"    → "Recompute CUDA"
    "MHA CUDA"          → "MHA CUDA"
    "KVCache Load"      → "KVCache Load K+V" (combined k+v)
    "KVCache Store"     → "KVCache Store"
    "Misc. CPU"         → "Misc. CPU"

  Note: The estimator models only the critical path; it has no Store or Misc CPU
  component. For fair comparison the total latency (sum of all segments) is also
  recorded for both GT and estimator.

Usage
-----
  python gt_vs_estimator.py --config sweep_config.yaml

  Or import and use the Python API directly (see __main__ at the bottom).

Dependencies (must be importable)
----------------------------------
  kv_schedule_optimization   (your estimator — must be on PYTHONPATH)
  trace_analyzer             (must be on PYTHONPATH or same directory)
  trace_result_analyzer      (same)
  flexllmgen                 (for get_opt_config, Policy, etc.)
  gee / energaizer artifact  (for get_gee — user supplies gpu_estimator object)
"""

import argparse
import csv
import dataclasses
import glob
import gzip
import math
import shutil
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional import guard — estimator and trace tools may not be installed in
# this sandbox; the script will still parse / plan correctly.
# ---------------------------------------------------------------------------
try:
    from kv_schedule_optimization import (
        CostModelConfig,
        get_available_offloadings,
        layer_prediction,
        pinned_pred,
        transfer_pred,
        recomp_calc_pred,
        layer_calc_pred,
    )
    from flexllmgen.opt_config import get_opt_config
    from flexllmgen.utils import GB
    ESTIMATOR_AVAILABLE = True
except ImportError:
    ESTIMATOR_AVAILABLE = False
    GB = 1024 ** 3  # fallback so dataclasses can initialise

try:
    from trace_result_analyzer import analyze_csv   # second-pass analysis
    from trace_analyzer import analyze_trace         # first-pass raw extraction
    TRACE_TOOLS_AVAILABLE = True
except ImportError:
    TRACE_TOOLS_AVAILABLE = False

sys.path.append( '../energaizer-ispass26-artifact/') # to be able to find energaizer-ispass26-artifact
from gee.gee_utils import get_gee


# ---------------------------------------------------------------------------
# OOM handling
# ---------------------------------------------------------------------------

class OOMError(RuntimeError):
    """
    Raised when an experiment cannot run because it exceeds GPU/CPU memory,
    either analytically (no feasible offloading scheme exists) or at runtime
    (CUDA out-of-memory during profiling).
    """


_OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "outofmemoryerror",
    "cudaerroroutofmemory",
    "oom killed",
)


def _is_oom_output(text: str) -> bool:
    """Return True if subprocess stderr contains a recognisable OOM message."""
    lower = text.lower()
    return any(m in lower for m in _OOM_MARKERS)


# Status values written to the CSV "status" column.
STATUS_OK    = "ok"
STATUS_OOM   = "oom"      # no feasible offload scheme, or CUDA OOM at runtime
STATUS_ERROR = "error"    # any other failure


# ---------------------------------------------------------------------------
# Trace cleanup constants
# ---------------------------------------------------------------------------

TRACE_CLEANUP_NONE     = "none"       # keep traces as-is (default)
TRACE_CLEANUP_COMPRESS = "compress"   # gzip each .json in-place -> .json.gz
TRACE_CLEANUP_DELETE   = "delete"     # permanently remove each .json


# ===========================================================================
# Section 1 – Configuration dataclasses
# ===========================================================================

@dataclasses.dataclass
class HardwareConfig:
    """Maps to CostModelConfig / CLI flags of kv_schedule_optimization."""
    gpu_mem_gb: int = 40            # --gpu-mem
    cpu_mem_gb: int = 200           # --cpu-mem
    gpu_freq: int = 1305            # --gpu-freq

    # Bandwidth / compute model flags
    use_ideal_bw: bool = False      # --i-BW
    use_flex_bw: bool = False       # --f-BW
    use_no_pinned: bool = False     # --nP
    use_ideal_comp: bool = False    # --i-C

    # GPU/CPU memory usage fractions (alpha_g / alpha_c in the estimator)
    alpha_g: float = 0.95
    alpha_c: float = 1.0

    # numactl binding for flexllmgen subprocess
    cpu_bind: Optional[str] = None  # e.g. "0"
    gpu_bind: Optional[str] = None  # e.g. "0"  (CUDA_VISIBLE_DEVICES)
    sudo_password: Optional[str] = None

    def to_cost_model_config(self) -> "CostModelConfig":
        cfg = CostModelConfig()
        cfg.gmem = self.alpha_g * self.gpu_mem_gb * GB
        cfg.cmem = self.alpha_c * self.cpu_mem_gb * GB
        cfg.gpu_freq = self.gpu_freq
        cfg.use_ideal_bw = self.use_ideal_bw
        cfg.use_flex_bw = self.use_flex_bw
        cfg.use_no_pinned = self.use_no_pinned
        cfg.use_ideal_comp = self.use_ideal_comp
        return cfg


@dataclasses.dataclass
class EstimatorMode:
    """
    One estimator configuration (set of bandwidth / compute flags).

    name        Short label used in CSV column headers and plot legend.
    use_ideal_bw   --i-BW flag
    use_flex_bw    --f-BW flag
    use_no_pinned  --nP flag
    use_ideal_comp --i-C flag
    """
    name: str = "default"
    use_ideal_bw: bool = False
    use_flex_bw: bool = False
    use_no_pinned: bool = False
    use_ideal_comp: bool = False

    def apply_to(self, hw: HardwareConfig) -> HardwareConfig:
        """Return a copy of hw with these mode flags applied."""
        import copy
        hw2 = copy.copy(hw)
        hw2.use_ideal_bw = self.use_ideal_bw
        hw2.use_flex_bw = self.use_flex_bw
        hw2.use_no_pinned = self.use_no_pinned
        hw2.use_ideal_comp = self.use_ideal_comp
        return hw2


@dataclasses.dataclass
class ExperimentConfig:
    """
    One (model, workload, batch, recompute) experiment point.

    offload_percent   percentage of KV cache offloaded to CPU (0-100).
                      None means "use minimum feasible" (computed automatically).
    trace_json_path   if the trace already exists, skip profiling.
    summary_csv_path  if the summary CSV already exists, skip analysis.
    """
    # ---- workload ----
    model: str = "facebook/opt-6.7b"
    prompt_len: int = 2048
    gen_len: int = 16
    num_prompts: int = 16           # total = batch_size * num_batches

    # ---- strategy ----
    batch_size: int = 1             # gpu_batch_size (gbs)
    recompute_len: int = 0

    offload_percent: Optional[float] = None   # None → auto minimum

    # ---- paths ----
    flexllmgen_script: str = "flexllmgen/flex_opt_kvpr.py"
    trace_analyzer_script: str = "trace_analyzer.py"
    result_analyzer_script: str = "trace_result_analyzer.py"
    output_dir: str = "./gt_vs_est_runs"

    # ---- pre-computed paths (populated by the runner) ----
    trace_json_path: Optional[str] = None
    summary_csv_path: Optional[str] = None

    # ---- derived (computed from num_prompts / batch_size) ----
    @property
    def num_batches(self) -> int:
        return self.num_prompts // self.batch_size

    @property
    def experiment_id(self) -> str:
        rc = f"_rc{self.recompute_len}" if self.recompute_len > 0 else ""
        off = f"_off{int(self.offload_percent)}" if self.offload_percent is not None else ""
        return (
            f"{self.model.split('/')[-1]}"
            f"_p{self.prompt_len}_g{self.gen_len}"
            f"_np{self.num_prompts}_bs{self.batch_size}"
            f"{rc}{off}"
        )


# ===========================================================================
# Section 2 – Estimator interface
# ===========================================================================

# Canonical segment names shared between GT and estimator outputs.
EST_SEGMENT_NAMES = [
    "PinnedMemory CPU (phase1)",  # component_breakdown[0]
    "PinnedMemory CPU (phase2)",  # component_breakdown[1]
    "Recompute Load",             # component_breakdown[2]
    "Recompute CUDA",             # component_breakdown[3]
    "MHA CUDA",                   # component_breakdown[4]
    "KVCache Load K",             # component_breakdown[5]
    "KVCache Load V",             # component_breakdown[6]
]

GT_SEGMENT_NAMES = [
    "PinnedMemory CPU",
    "Recompute Load",
    "Recompute CUDA",
    "MHA CUDA",
    "KVCache Load",
    "KVCache Store",
    "Misc. CPU",
]


def _determine_is_load_store(
    batch_idx: int,
    num_batches: int,
    offload_percent: float,
    last_token: bool = False,
) -> int:
    """
    Determine the is_load_store flag for a given batch index, mirroring
    multi_batch_forward_pass / single_batch_forward_pass in the estimator.

    In multi-batch (num_batches > 1):
      - first batch (idx 0):      load only  → 1
      - middle batches:           load+store  → 3  (or 1 if last_token)
      - last batch (idx N-1):     store only  → 2  (or 0 if last_token)

    In single-batch (num_batches == 1):
      - always no load/store → 0  (all KV stays on GPU when offload=0)
      - but if offload_percent > 0 with single batch, it's load only → 1
        (the estimator uses will_load=3 for non-last MLP, but MHA is 0)

    For the "representative middle MHA layer" used in the comparison, we use the
    bidirectional (load+store) case for multi-batch, or no-load for single-batch
    with 0% offload.
    """
    if num_batches == 1:
        if offload_percent == 0:
            return 0   # everything on GPU, no transfer
        else:
            return 1   # partial offload: load only (single batch)
    else:
        if batch_idx == 0:
            return 1       # first batch: load KV
        elif batch_idx == num_batches - 1:
            return 0 if last_token else 2   # last batch: store KV
        else:
            return 0 if last_token else 3   # middle: bidirectional


def get_estimator_breakdown(
    exp: ExperimentConfig,
    hw: HardwareConfig,
    gpu_estimator: Any,
    opt_config: Any,
    gen_step: int = 1,          # which decode step to evaluate (1 = first decode token)
) -> Dict[str, float]:
    """
    Call layer_prediction() for a "representative" MHA decode layer and return
    the raw component breakdown as a dict of {segment_name: latency_µs}.

    The "representative" layer is the bidirectional (load+store) middle layer for
    multi-batch, or the no-load layer for single-batch without offloading.

    component_breakdown values are in **seconds** from layer_prediction;
    we convert to µs here (× 1e6).

    Parameters
    ----------
    gen_step : int
        Which token index in the decode phase (0-indexed within decode).
        Default=1 (second decode token, after warm-up), matching what
        the GT analysis averages over.
    """
    if not ESTIMATOR_AVAILABLE:
        raise RuntimeError("kv_schedule_optimization is not importable.")

    hw_cfg = hw.to_cost_model_config()
    offload_pct = exp.offload_percent if exp.offload_percent is not None else 0.0
    num_batches = exp.num_batches

    # Select is_load_store for representative middle MHA layer
    if num_batches == 1:
        is_load_store = 0 if offload_pct == 0 else 1
    else:
        # Middle batch → bidirectional (load + store)
        is_load_store = 3

    _, _, _, _, _, component_breakdown = layer_prediction(
        opt_config=opt_config,
        is_load_store=is_load_store,
        batch_size=exp.batch_size,
        num_of_batches=num_batches,
        offload_percent=offload_pct,
        recomp_len=exp.recompute_len,
        prompt_len=exp.prompt_len,
        gen_len=gen_step,
        hardware_config=hw_cfg,
        gpu_estimator=gpu_estimator,
        break_MHA=True,
        layer_type="MHA",
        first_token=False,
    )

    # component_breakdown is in seconds; multiply by 1e6 for µs
    S2US = 1e6
    return {
        "PinnedMemory CPU (phase1)": component_breakdown[0] * S2US,
        "PinnedMemory CPU (phase2)": component_breakdown[1] * S2US,
        "Recompute Load":            component_breakdown[2] * S2US,
        "Recompute CUDA":            component_breakdown[3] * S2US,
        "MHA CUDA":                  component_breakdown[4] * S2US,
        "KVCache Load K":            component_breakdown[5] * S2US,
        "KVCache Load V":            component_breakdown[6] * S2US,
    }


def get_min_offload_percent(
    exp: ExperimentConfig,
    hw: HardwareConfig,
    opt_config: Any,
) -> float:
    """
    Call get_available_offloadings() to find the minimum feasible offload
    percentage for exp.batch_size, given the hardware config.
    Returns 0.0 if no offloading is needed (all KV fits on GPU).
    """
    hw_cfg = hw.to_cost_model_config()
    seq_len = exp.prompt_len + exp.gen_len

    feasible = get_available_offloadings(
        opt_config=opt_config,
        hardware_config=hw_cfg,
        batch_sizes=[exp.batch_size],
        num_of_prompts=exp.num_prompts,
        prompt_len=exp.prompt_len,
        gen_len=exp.gen_len,
        seq_len=seq_len,
        min_offloading=True,      # ← stop at first (minimum) feasible entry
    )
    if exp.batch_size not in feasible or not feasible[exp.batch_size]:
        raise OOMError(
            f"No feasible offloading strategy for batch_size={exp.batch_size} "
            f"— model+KV cache exceeds available GPU/CPU memory."
        )
    return float(feasible[exp.batch_size][0])


# ===========================================================================
# Section 3 – Ground-truth pipeline (trace → CSV → segments)
# ===========================================================================

def build_flexllm_command(
    exp: ExperimentConfig,
    hw: HardwareConfig,
    trace_dir: str,
) -> List[str]:
    """
    Build the subprocess command for flex_opt_kvpr.py, mirroring
    run-all-offload-sweep-batch.sh.

    percent[0] = weight on GPU (100 for opt-6.7b)
    percent[1] = weight on CPU (0)
    percent[2] = KV cache on GPU  = 100 - offload_percent
    percent[3] = KV cache on CPU  = offload_percent
    percent[4] = activations on GPU (100)
    percent[5] = activations on CPU (0)
    """
    offload = int(exp.offload_percent) if exp.offload_percent is not None else 0
    kv_gpu = 100 - offload
    kv_cpu = offload

    # Weight placement: use 100% GPU for models that fit (opt-6.7b / 13b);
    # a future enhancement can look this up from opt_config.model_bytes().
    w_gpu = 100
    w_cpu = 0

    percent_args = [str(w_gpu), str(w_cpu), str(kv_gpu), str(kv_cpu), "100", "0"]

    python_exe = sys.executable
    cmd = [
        python_exe, exp.flexllmgen_script,
        "--model", exp.model,
        "--prompt-len", str(exp.prompt_len),
        "--gen-len", str(exp.gen_len),
        "--gpu-batch-size", str(exp.batch_size),
        "--num-gpu-batches", str(exp.num_batches),
        "--percent", *percent_args,
        "--recompute-len", str(exp.recompute_len),
        "--sep-layer", "true",
        "--profile",
        "--save-to", trace_dir,
    ]

    if hw.cpu_bind is not None and hw.gpu_bind is not None:
        numactl = [
            "numactl",
            f"--cpunodebind={hw.cpu_bind}",
            f"--membind={hw.cpu_bind}",
        ]
        env_prefix = [f"CUDA_VISIBLE_DEVICES={hw.gpu_bind}"]
        if hw.sudo_password:
            cmd = (
                ["bash", "-c",
                 f"echo '{hw.sudo_password}' | sudo -S "
                 + " ".join(numactl)
                 + " env " + " ".join(env_prefix)
                 + " " + " ".join(cmd)]
            )
        else:
            cmd = numactl + ["env"] + env_prefix + cmd

    return cmd


def run_flexllm_profile(
    exp: ExperimentConfig,
    hw: HardwareConfig,
    trace_dir: str,
    dry_run: bool = False,
    _generated_traces: Optional[List[str]] = None,
) -> str:
    """
    Run flex_opt_kvpr.py and return the path to the produced JSON trace.

    The filename is constructed by get_filename() inside flex_opt_kvpr.py:
        fo-{model_size}-gbs{gbs}-ngbs{ngbs}-prompt{p}-gen{g}-percent-{pcts}-[R-{rc}-]gpu-cache.json

    We re-derive the expected filename here to locate it after the run.

    _generated_traces : optional list; if provided, the absolute path of
        the .json is appended when a trace is freshly profiled OR when a
        compressed .json.gz is decompressed back to .json.  Pre-existing
        .json files (the [skip] branch) are never recorded here, so the
        cleanup step will not touch them.
    """
    os.makedirs(trace_dir, exist_ok=True)

    offload = int(exp.offload_percent) if exp.offload_percent is not None else 0
    kv_gpu = 100 - offload
    kv_cpu = offload
    w_gpu, w_cpu = 100, 0
    pcts = f"{w_gpu}-{w_cpu}-{kv_gpu}-{kv_cpu}-100-0-"
    model_size = exp.model.split("-")[-1]
    rc_part = f"R-{exp.recompute_len}-" if exp.recompute_len > 0 else ""
    expected_stem = (
        f"fo-{model_size}"
        f"-gbs{exp.batch_size}"
        f"-ngbs{exp.num_batches}"
        f"-prompt{exp.prompt_len}"
        f"-gen{exp.gen_len}"
        f"-percent-{pcts}"
        f"{rc_part}gpu-cache"
    )
    expected_json = os.path.join(trace_dir, expected_stem + ".json")

    if os.path.exists(expected_json):
        print(f"  [skip] trace already exists: {expected_json}")
        return expected_json

    # Check for a previously compressed version (.json.gz) and decompress
    # it rather than re-running the expensive profiling step.
    gz_json = expected_json + ".gz"
    if os.path.exists(gz_json):
        print(f"  [decompress] found compressed trace: {gz_json}")
        if not dry_run:
            with gzip.open(gz_json, "rb") as f_in, \
                 open(expected_json, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"  [decompress] restored: {expected_json}")
            # Decompressed .json is recorded so cleanup can re-compress
            # or delete it at the end of this run.
            if _generated_traces is not None:
                _generated_traces.append(os.path.abspath(expected_json))
        else:
            print(f"  [dry_run] would decompress {gz_json} -> {expected_json}")
        return expected_json

    cmd = build_flexllm_command(exp, hw, trace_dir)
    print(f"  [run] flex_opt_kvpr: {' '.join(cmd)}")
    if dry_run:
        print("  [dry_run] would execute the above command.")
        return expected_json

    # Capture stderr to detect CUDA OOM; stdout streams live for progress.
    result = subprocess.run(cmd, capture_output=False, text=True,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_text = result.stderr or ""
        print(stderr_text, file=sys.stderr, end="")   # surface to user
        if _is_oom_output(stderr_text):
            raise OOMError(
                f"CUDA out-of-memory during profiling for {exp.experiment_id}"
            )
        raise RuntimeError(
            f"flex_opt_kvpr.py failed (exit {result.returncode}) for {exp.experiment_id}"
        )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")  # warnings on success

    # Verify the file appeared
    if not os.path.exists(expected_json):
        # Try glob fallback in case the name differs slightly
        candidates = sorted(glob.glob(os.path.join(trace_dir, "fo-*.json")))
        if candidates:
            expected_json = candidates[-1]
            print(f"  [warn] expected path not found; using {expected_json}")
        else:
            raise FileNotFoundError(
                f"Trace JSON not found after run. Expected: {expected_json}"
            )

    # Record newly-profiled trace for optional cleanup at run end.
    if _generated_traces is not None:
        _generated_traces.append(os.path.abspath(expected_json))

    return expected_json


def run_trace_analysis(
    trace_json: str,
    exp: ExperimentConfig,
    dry_run: bool = False,
) -> str:
    """
    Run trace_analyzer.py (--batched) then trace_result_analyzer.py (--batched)
    and return the path to the final summary CSV.

    Output CSVs are written to exp.output_dir (not next to the trace), so that
    read-only trace locations (e.g. an uploads folder) are not a problem.
    """
    stem = Path(trace_json).stem
    out_dir = exp.output_dir
    os.makedirs(out_dir, exist_ok=True)

    analysis_csv = os.path.join(out_dir, stem + "_batched_analysis.csv")
    summary_csv  = os.path.join(out_dir, stem + "_batched_analysis_summary.csv")

    if os.path.exists(summary_csv):
        print(f"  [skip] summary CSV already exists: {summary_csv}")
        return summary_csv

    # Step 1: trace_analyzer.py
    cmd1 = [
        sys.executable, exp.trace_analyzer_script,
        trace_json, "--batched", "--out", analysis_csv,
    ]
    print(f"  [run] trace_analyzer: {' '.join(cmd1)}")
    if not dry_run:
        r = subprocess.run(cmd1, capture_output=False, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"trace_analyzer.py failed for {trace_json}")

    # Step 2: trace_result_analyzer.py
    cmd2 = [
        sys.executable, exp.result_analyzer_script,
        analysis_csv, "--batched", "--out", summary_csv,
    ]
    print(f"  [run] trace_result_analyzer: {' '.join(cmd2)}")
    if not dry_run:
        r = subprocess.run(cmd2, capture_output=False, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"trace_result_analyzer.py failed for {analysis_csv}")

    return summary_csv


def load_gt_summary(summary_csv: str, avg_rows: Optional[int] = None) -> Tuple[Dict, List[str]]:
    """
    Load a batched-mode summary CSV and return (averaged_row, col_names).

    avg_rows=None  → average all rows with complete numeric columns.
    avg_rows=N     → average first N complete rows.
    """
    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        all_rows = list(reader)

    if not all_rows:
        raise ValueError(f"Empty summary CSV: {summary_csv}")

    # Identify numeric columns
    numeric_cols, string_cols = [], []
    for c in cols:
        try:
            float(all_rows[0].get(c, ""))
            numeric_cols.append(c)
        except (ValueError, TypeError):
            string_cols.append(c)

    complete = [r for r in all_rows if all(r.get(c, "") not in ("", None) for c in numeric_cols)]
    if not complete:
        raise ValueError(f"No complete rows in {summary_csv}")

    if avg_rows is not None:
        complete = complete[:avg_rows]

    n = len(complete)
    averaged = {}
    for c in numeric_cols:
        averaged[c] = round(sum(float(r[c]) for r in complete) / n, 6)
    for c in string_cols:
        averaged[c] = complete[0].get(c, "")

    return averaged, cols


def _fv(row: Dict, key: str) -> float:
    v = row.get(key, 0)
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def build_gt_segments(row: Dict) -> Dict[str, float]:
    """
    Extract latency breakdown from a batched summary row.
    Mirrors build_segments_batched() in plot_latency_single.py, but returns
    ALL segments (including those not on the critical path, where they are 0).
    The critical-path segments come from the row; KVCache Store and Misc. CPU
    are always present regardless of which path won.
    """
    crit_winner = row.get("critical-path-winner", "")
    inner_winner = row.get("path1-inner-winner", "")

    lhc = _fv(row, "load-hidden-compute-cudamemcpy")
    pm1 = _fv(row, "pin-memory-1")
    pm2 = _fv(row, "pin-memory-2")
    lc1 = _fv(row, "load-cache-cudamemcpy-1")
    lc2 = _fv(row, "load-cache-cudamemcpy-2")
    sc1 = _fv(row, "store-cache-cudamemcpy-1")
    sc2 = _fv(row, "store-cache-cudamemcpy-2")
    recompute = _fv(row, "recompute-cuda")
    mha_cuda  = _fv(row, "mha-gen-cuda")
    sum_all   = _fv(row, "sum-all")
    crit      = _fv(row, "critical-path")

    segs: Dict[str, float] = {k: 0.0 for k in GT_SEGMENT_NAMES}
    segs["Misc. CPU"] = sum_all - crit

    if crit_winner == "path1":
        if inner_winner == "subpath1":
            segs["PinnedMemory CPU"] = pm1 + pm2
            segs["KVCache Load"]     = lc2
        elif inner_winner == "subpath2":
            segs["PinnedMemory CPU"] = pm1
            segs["KVCache Load"]     = lc1 + lc2
        else:  # subpath3
            segs["Recompute Load"] = lhc
            segs["KVCache Load"]   = lc1 + lc2
    elif crit_winner == "path2":
        segs["Recompute Load"] = lhc
        segs["Recompute CUDA"] = recompute
        segs["MHA CUDA"]       = mha_cuda
    else:  # path3
        segs["PinnedMemory CPU"] = pm1 + pm2
        segs["KVCache Store"]    = sc1 + sc2

    return segs


# ===========================================================================
# Section 4 – Sweep helpers
# ===========================================================================

def sweep_batch_size(
    base: ExperimentConfig,
    batch_sizes: List[int],
    hw: HardwareConfig,
    opt_config: Any,
) -> List[ExperimentConfig]:
    """
    Vary batch_size (with num_batches = num_prompts // batch_size), fixing all
    other params. offload_percent is set to the minimum feasible for each batch.
    Recompute length is fixed at 0 (as specified in the requirement).
    """
    experiments = []
    for bs in batch_sizes:
        exp = dataclasses.replace(base, batch_size=bs, recompute_len=0)
        try:
            min_off = get_min_offload_percent(exp, hw, opt_config)
        except OOMError as e:
            print(f"  [OOM] sweep_batch_size: bs={bs} skipped — {e}")
            continue
        exp = dataclasses.replace(exp, offload_percent=min_off)
        print(
            f"  sweep_batch_size: bs={bs}, num_batches={exp.num_batches}, "
            f"min_offload={min_off:.1f}%"
        )
        experiments.append(exp)
    return experiments


def sweep_recompute(
    base: ExperimentConfig,
    recompute_lens: List[int],
    hw: HardwareConfig,
    opt_config: Any,
) -> List[ExperimentConfig]:
    """
    Vary recompute_len, fixing batch_size and other params.
    offload_percent is set to the minimum feasible (re-evaluated since recompute
    can affect the memory requirement — it typically doesn't change offload, but
    we keep this consistent).
    """
    experiments = []
    for rc in recompute_lens:
        exp = dataclasses.replace(base, recompute_len=rc)
        try:
            min_off = get_min_offload_percent(exp, hw, opt_config)
        except OOMError as e:
            print(f"  [OOM] sweep_recompute: rc={rc} skipped — {e}")
            continue
        exp = dataclasses.replace(exp, offload_percent=min_off)
        print(
            f"  sweep_recompute: rc={rc}, bs={exp.batch_size}, "
            f"min_offload={min_off:.1f}%"
        )
        experiments.append(exp)
    return experiments


def sweep_batch_and_recompute(
    base: ExperimentConfig,
    batch_sizes: List[int],
    recompute_lens: List[int],
    hw: HardwareConfig,
    opt_config: Any,
) -> List[ExperimentConfig]:
    """
    Full cross-product of batch sizes and recompute lengths.
    offload_percent is always the minimum feasible for each (bs, rc) pair.
    """
    experiments = []
    for bs in batch_sizes:
        for rc in recompute_lens:
            exp = dataclasses.replace(base, batch_size=bs, recompute_len=rc)
            try:
                min_off = get_min_offload_percent(exp, hw, opt_config)
            except OOMError as e:
                print(f"  [OOM] sweep_batch_and_recompute: bs={bs}, rc={rc} skipped — {e}")
                continue
            exp = dataclasses.replace(exp, offload_percent=min_off)
            print(
                f"  sweep_batch_and_recompute: bs={bs}, rc={rc}, "
                f"min_offload={min_off:.1f}%"
            )
            experiments.append(exp)
    return experiments


# ===========================================================================
# Section 5 – CSV output
# ===========================================================================

# Fixed experiment metadata columns
META_COLS = [
    "experiment_id",
    "model",
    "prompt_len",
    "gen_len",
    "num_prompts",
    "batch_size",
    "num_batches",
    "recompute_len",
    "offload_percent",   # minimum feasible, as used
    "status",            # "ok" | "oom" | "error"
    "skip_reason",       # human-readable explanation when status != "ok"
    "trace_json",
    "summary_csv",
]


def make_csv_columns(estimator_modes: List[EstimatorMode]) -> List[str]:
    """Return the full ordered list of CSV column names."""
    cols = list(META_COLS)
    # Ground-truth columns
    for seg in GT_SEGMENT_NAMES:
        safe = seg.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
        cols.append(f"gt_{safe}_us")
    cols.append("gt_total_us")
    # Estimator columns (one group per mode)
    for mode in estimator_modes:
        pfx = f"est_{mode.name}"
        for seg in EST_SEGMENT_NAMES:
            safe = seg.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
            cols.append(f"{pfx}_{safe}_us")
        cols.append(f"{pfx}_total_us")
    return cols


def build_csv_row(
    exp: ExperimentConfig,
    gt_segs: Dict[str, float],
    est_segs_by_mode: Dict[str, Dict[str, float]],  # mode_name → segment dict
    estimator_modes: List[EstimatorMode],
    status: str = STATUS_OK,
    skip_reason: str = "",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "experiment_id":  exp.experiment_id,
        "model":          exp.model,
        "prompt_len":     exp.prompt_len,
        "gen_len":        exp.gen_len,
        "num_prompts":    exp.num_prompts,
        "batch_size":     exp.batch_size,
        "num_batches":    exp.num_batches,
        "recompute_len":  exp.recompute_len,
        "offload_percent": round(exp.offload_percent, 2) if exp.offload_percent is not None else "",
        "status":         status,
        "skip_reason":    skip_reason,
        "trace_json":     exp.trace_json_path or "",
        "summary_csv":    exp.summary_csv_path or "",
    }

    # GT segments
    for seg in GT_SEGMENT_NAMES:
        safe = seg.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
        row[f"gt_{safe}_us"] = round(gt_segs.get(seg, 0.0), 3)
    row["gt_total_us"] = round(sum(gt_segs.values()), 3)

    # Estimator segments per mode
    for mode in estimator_modes:
        pfx = f"est_{mode.name}"
        segs = est_segs_by_mode.get(mode.name, {})
        for seg in EST_SEGMENT_NAMES:
            safe = seg.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
            row[f"{pfx}_{safe}_us"] = round(segs.get(seg, 0.0), 3)
        row[f"{pfx}_total_us"] = round(sum(segs.values()), 3)

    return row


def write_comparison_csv(
    rows: List[Dict],
    output_path: str,
    estimator_modes: List[EstimatorMode],
) -> None:
    cols = make_csv_columns(estimator_modes)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})
    print(f"\nComparison CSV written to: {output_path}")


# ===========================================================================
# Section 6 – Main runner
# ===========================================================================

def _skipped_row(
    exp: ExperimentConfig,
    estimator_modes: List[EstimatorMode],
    status: str,
    reason: str,
) -> Dict:
    """Return a CSV row with all segment values blank and status/reason set."""
    print(f"  [{status.upper()}] {reason}")
    return build_csv_row(
        exp,
        gt_segs={},
        est_segs_by_mode={m.name: {} for m in estimator_modes},
        estimator_modes=estimator_modes,
        status=status,
        skip_reason=reason,
    )


def run_experiment(
    exp: ExperimentConfig,
    hw: HardwareConfig,
    estimator_modes: List[EstimatorMode],
    gpu_estimator: Any,
    opt_config: Any,
    dry_run: bool = False,
    avg_gt_rows: Optional[int] = None,
    _generated_traces: Optional[List[str]] = None,
) -> Dict:
    """
    Execute the full pipeline for one experiment and return a CSV row dict.

    Always returns a dict (never None). OOM and other failures produce a row
    with status="oom" or status="error" and blank segment values, so the CSV
    remains complete and the sweep continues uninterrupted.

    _generated_traces : optional list passed through to run_flexllm_profile.
        Paths of .json files that are newly profiled OR decompressed from .gz
        are appended here, making them eligible for --trace-cleanup at run end.
        Pre-existing .json files (the [skip] branch) are never appended.
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {exp.experiment_id}")
    print(f"{'='*60}")

    # Step 1: resolve offload_percent
    if exp.offload_percent is None:
        if ESTIMATOR_AVAILABLE:
            print("  Step 1: computing minimum offload percent...")
            try:
                exp = dataclasses.replace(
                    exp,
                    offload_percent=get_min_offload_percent(exp, hw, opt_config),
                )
            except OOMError as e:
                return _skipped_row(exp, estimator_modes, STATUS_OOM, str(e))
            except Exception as e:
                return _skipped_row(exp, estimator_modes, STATUS_ERROR,
                                    f"Step 1 failed: {e}")
        else:
            print("  Step 1: [skip] estimator not available, defaulting to 0% offload")
            exp = dataclasses.replace(exp, offload_percent=0.0)
    print(f"  offload_percent = {exp.offload_percent:.1f}%")

    # Step 2: run profiling trace (skip if trace_json_path already provided)
    if exp.trace_json_path and os.path.isfile(exp.trace_json_path):
        print(f"  Step 2: [skip] using existing trace: {exp.trace_json_path}")
        trace_json = exp.trace_json_path
    else:
        trace_dir = os.path.join(
            exp.output_dir,
            f"prompt_{exp.prompt_len}_bs{exp.batch_size}"
            + (f"_rc{exp.recompute_len}" if exp.recompute_len > 0 else ""),
        )
        print("  Step 2: running flex_opt_kvpr profiling...")
        try:
            trace_json = run_flexllm_profile(
                exp, hw, trace_dir,
                dry_run=dry_run,
                _generated_traces=_generated_traces,
            )
            exp = dataclasses.replace(exp, trace_json_path=trace_json)
        except OOMError as e:
            return _skipped_row(exp, estimator_modes, STATUS_OOM, str(e))
        except Exception as e:
            return _skipped_row(exp, estimator_modes, STATUS_ERROR,
                                f"Step 2 profiling failed: {e}")

    # Step 3: trace analysis (skip if summary_csv_path already provided)
    if exp.summary_csv_path and os.path.isfile(exp.summary_csv_path):
        print(f"  Step 3: [skip] using existing summary CSV: {exp.summary_csv_path}")
        summary_csv = exp.summary_csv_path
    else:
        print("  Step 3: running trace analysis pipeline...")
        try:
            summary_csv = run_trace_analysis(trace_json, exp, dry_run=dry_run)
            exp = dataclasses.replace(exp, summary_csv_path=summary_csv)
        except Exception as e:
            return _skipped_row(exp, estimator_modes, STATUS_ERROR,
                                f"Step 3 trace analysis failed: {e}")

    # Step 4: parse GT
    print("  Step 4: loading GT breakdown...")
    gt_segs: Dict[str, float] = {}
    if not dry_run:
        try:
            gt_row, _ = load_gt_summary(summary_csv, avg_rows=avg_gt_rows)
            gt_segs = build_gt_segments(gt_row)
            print("  GT segments (µs):")
            for k, v in gt_segs.items():
                print(f"    {k}: {v:.1f}")
        except Exception as e:
            return _skipped_row(exp, estimator_modes, STATUS_ERROR,
                                f"Step 4 GT parsing failed: {e}")

    # Step 5: estimator breakdown per mode (failures here are non-fatal —
    # GT data is kept; only that mode's columns are blank)
    print("  Step 5: running estimator for each mode...")
    est_segs_by_mode: Dict[str, Dict[str, float]] = {}
    if ESTIMATOR_AVAILABLE and not dry_run:
        for mode in estimator_modes:
            hw_mode = mode.apply_to(hw)
            try:
                segs = get_estimator_breakdown(
                    exp=exp,
                    hw=hw_mode,
                    gpu_estimator=gpu_estimator,
                    opt_config=opt_config,
                )
                est_segs_by_mode[mode.name] = segs
                total = sum(segs.values())
                print(f"  Estimator [{mode.name}] total = {total:.1f} µs")
            except Exception as e:
                print(f"  ERROR in estimator [{mode.name}]: {e}")
                est_segs_by_mode[mode.name] = {}
    else:
        for mode in estimator_modes:
            est_segs_by_mode[mode.name] = {}

    # Step 6: build row
    return build_csv_row(exp, gt_segs, est_segs_by_mode, estimator_modes,
                         status=STATUS_OK)


def cleanup_traces(
    generated_trace_paths: List[str],
    mode: str,
    dry_run: bool = False,
) -> None:
    """
    Compress or delete raw .json trace files generated (or decompressed) in
    this run.  Only paths recorded in generated_trace_paths are touched;
    pre-existing .json files that were simply skipped are never included.

    Parameters
    ----------
    generated_trace_paths  Absolute paths of .json traces written this run.
    mode                   TRACE_CLEANUP_COMPRESS | TRACE_CLEANUP_DELETE |
                           TRACE_CLEANUP_NONE
    dry_run                If True, print what would happen but do nothing.
    """
    if mode == TRACE_CLEANUP_NONE or not generated_trace_paths:
        return

    print(f"\n\u2500\u2500 Trace cleanup ({mode}) {chr(45)*38}")
    total_freed = 0.0

    for path in generated_trace_paths:
        if not os.path.isfile(path):
            print(f"  [skip] not found: {path}")
            continue
        size_mb = os.path.getsize(path) / (1024 ** 2)

        if mode == TRACE_CLEANUP_DELETE:
            print(f"  [delete] {path}  ({size_mb:.1f} MB)")
            if not dry_run:
                os.remove(path)
            total_freed += size_mb

        elif mode == TRACE_CLEANUP_COMPRESS:
            gz_path = path + ".gz"
            print(f"  [compress] {path}  ({size_mb:.1f} MB)  \u2192  {gz_path}")
            if not dry_run:
                with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                gz_size_mb = os.path.getsize(gz_path) / (1024 ** 2)
                os.remove(path)
                saved_mb = size_mb - gz_size_mb
                print(f"    compressed: {gz_size_mb:.1f} MB  (saved {saved_mb:.1f} MB)")
                total_freed += saved_mb
            else:
                total_freed += size_mb

    verb = "would free" if dry_run else "freed"
    suffix = "MB saved" if (mode == TRACE_CLEANUP_COMPRESS and not dry_run) else "MB"
    print(f"  Total {verb}: {total_freed:.1f} {suffix}")
    print(f"{chr(45)*54}")


def run_comparison(
    experiments: List[ExperimentConfig],
    hw: HardwareConfig,
    estimator_modes: List[EstimatorMode],
    gpu_estimator: Any,
    opt_config: Any,
    output_csv: str = "comparison_results.csv",
    dry_run: bool = False,
    avg_gt_rows: Optional[int] = None,
    trace_cleanup: str = TRACE_CLEANUP_NONE,
) -> List[Dict]:
    """
    Run all experiments and write the comparison CSV.

    Parameters
    ----------
    experiments      List of ExperimentConfig objects (from sweep helpers or manual).
    hw               Hardware configuration (shared across experiments).
    estimator_modes  List of EstimatorMode to compare (at least one).
    gpu_estimator    Initialised GEE estimator object (pass None to skip estimator).
    opt_config       FlexLLM OPT config object (from get_opt_config(model)).
    output_csv       Path for the output comparison CSV.
    dry_run          If True, print commands but don't execute them.
    avg_gt_rows      Number of GT rows to average (None = all).
    trace_cleanup    What to do with .json traces after the CSV is written:
                       "none"     - keep as-is (default)
                       "compress" - gzip in-place (.json -> .json.gz)
                       "delete"   - permanently remove
                     Applies to traces that were newly profiled OR decompressed
                     from .gz in this run.  Pre-existing .json files are never
                     touched.
    """
    all_rows = []
    generated_traces: List[str] = []   # paths written/restored this run
    n_ok, n_oom, n_err = 0, 0, 0

    for exp in experiments:
        row = run_experiment(
            exp=exp,
            hw=hw,
            estimator_modes=estimator_modes,
            gpu_estimator=gpu_estimator,
            opt_config=opt_config,
            dry_run=dry_run,
            avg_gt_rows=avg_gt_rows,
            _generated_traces=generated_traces,
        )
        all_rows.append(row)
        s = row.get("status", STATUS_OK)
        if s == STATUS_OOM:    n_oom += 1
        elif s == STATUS_OK:   n_ok  += 1
        else:                  n_err += 1

    print(f"\nSweep complete: {n_ok} ok, {n_oom} oom, {n_err} error "
          f"(out of {len(all_rows)} total)")

    if all_rows:
        write_comparison_csv(all_rows, output_csv, estimator_modes)
        if n_oom or n_err:
            print(f"  Note: skipped rows (oom/error) are included in the CSV "
                  f"with blank segment values \u2014 see the 'status' column.")
    else:
        print("No experiments to write.")

    # Cleanup runs after the CSV is safely on disk.
    cleanup_traces(generated_traces, mode=trace_cleanup, dry_run=dry_run)

    return all_rows


# ===========================================================================
# Section 7 – Entry point / example usage
# ===========================================================================

def parse_flexllm_filename(path: str) -> dict:
    """
    Extract experiment parameters from a flex_opt_kvpr JSON filename.
    Pattern: fo-{size}-gbs{bs}-ngbs{nb}-prompt{p}-gen{g}-percent-{p0}-{p1}-{p2}-{p3}-{p4}-{p5}-[R-{rc}-]gpu-cache.json

    Returns a dict with keys: batch_size, num_batches, prompt_len, gen_len,
    offload_percent, recompute_len (may be partial if the filename doesn't match).
    """
    import re
    stem = Path(path).stem
    result = {}

    m = re.search(r'-gbs(\d+)-', stem)
    if m: result['batch_size'] = int(m.group(1))

    m = re.search(r'-ngbs(\d+)-', stem)
    if m: result['num_batches'] = int(m.group(1))

    m = re.search(r'-prompt(\d+)-', stem)
    if m: result['prompt_len'] = int(m.group(1))

    m = re.search(r'-gen(\d+)-', stem)
    if m: result['gen_len'] = int(m.group(1))

    # percent-p0-p1-p2-p3-p4-p5-  (p3 = kv_cpu_percent = offload_percent)
    m = re.search(r'-percent-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-', stem)
    if m: result['offload_percent'] = float(m.group(4))   # p3 = kv_cpu

    m = re.search(r'-R-(\d+)-', stem)
    if m: result['recompute_len'] = int(m.group(1))
    else: result['recompute_len'] = 0

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare ground-truth traces vs estimator latency breakdown."
    )
    parser.add_argument("--model", default="facebook/opt-6.7b")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--gen-len", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument(
        "--sweep", choices=["batch_size", "recompute", "batch_and_recompute"],
        default="batch_size",
        help="Which sweep to run.",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
        help="Batch sizes to sweep (for batch_size and batch_and_recompute sweeps).",
    )
    parser.add_argument(
        "--recompute-lens", type=int, nargs="+", default=[0, 256, 512, 1024],
        help="Recompute lengths to sweep (for recompute and batch_and_recompute sweeps).",
    )
    parser.add_argument("--fixed-batch-size", type=int, default=2,
        help="Batch size for the recompute-only sweep.")
    parser.add_argument("--gpu-mem", type=int, default=40)
    parser.add_argument("--cpu-mem", type=int, default=200)
    parser.add_argument("--gpu-freq", type=int, default=1305)
    parser.add_argument("--cpu-bind", default=None)
    parser.add_argument("--gpu-bind", default=None)
    parser.add_argument("--sudo-password", default=None)
    parser.add_argument("--output-dir", default="./gt_vs_est_runs")
    parser.add_argument("--output-csv", default="comparison_results.csv")
    parser.add_argument("--flexllm-script", default="flexllmgen/flex_opt_kvpr.py")
    parser.add_argument("--trace-analyzer-script", default="trace_analyzer.py")
    parser.add_argument("--result-analyzer-script", default="trace_result_analyzer.py")
    parser.add_argument("--dry-run", action="store_true",
        help="Print commands without executing them.")
    parser.add_argument("--avg-gt-rows", type=int, default=None,
        help="Number of GT summary rows to average (default: all).")
    parser.add_argument(
        "--trace-cleanup",
        choices=[TRACE_CLEANUP_NONE, TRACE_CLEANUP_COMPRESS, TRACE_CLEANUP_DELETE],
        default=TRACE_CLEANUP_NONE,
        metavar="{none,compress,delete}",
        help=(
            "What to do with raw .json trace files after the CSV is written. "
            "  none     - keep as-is (default). "
            "  compress - gzip each trace in-place (.json -> .json.gz). "
            "  delete   - permanently remove each trace. "
            "Applies to traces newly profiled OR decompressed from .gz in "
            "this run. Pre-existing .json files are never touched."
        ),
    )
    
    # ── Short-circuit flags for debugging with an existing trace ──────────────
    parser.add_argument(
        "--trace-json", default=None, metavar="PATH",
        help=(
            "Path to an existing PyTorch trace JSON. Skips flex_opt_kvpr profiling "
            "(step 2) and runs only trace_analyzer + trace_result_analyzer on this file. "
            "Also sets --num-prompts, --batch-size, --recompute-len, and --offload-percent "
            "automatically from the filename if they can be parsed from it."
        ),
    )
    parser.add_argument(
        "--summary-csv", default=None, metavar="PATH",
        help=(
            "Path to an existing batched summary CSV (output of trace_result_analyzer). "
            "Skips both profiling (step 2) and trace analysis (step 3) entirely. "
            "Requires --trace-json to also be set (used for metadata only)."
        ),
    )
    args = parser.parse_args()
    
    # --- Hardware config ---
    hw = HardwareConfig(
        gpu_mem_gb=args.gpu_mem,
        cpu_mem_gb=args.cpu_mem,
        gpu_freq=args.gpu_freq,
        cpu_bind=args.cpu_bind,
        gpu_bind=args.gpu_bind,
        sudo_password=args.sudo_password,
    )
    
    # --- Estimator modes to compare ---
    # Add / remove modes here; each will produce its own column group in the CSV.
    estimator_modes = [
        EstimatorMode(name="ideal", use_ideal_bw=True, use_ideal_comp=True, use_no_pinned=True)
        # EstimatorMode(name="default"),
        # EstimatorMode(name="ideal_bw",    use_ideal_bw=True),
        # EstimatorMode(name="flex_bw",     use_flex_bw=True),
        # EstimatorMode(name="no_pinned",   use_no_pinned=True),
        # EstimatorMode(name="ideal_comp",  use_ideal_comp=True),
    ]
    
    # --- Load opt_config and gpu_estimator ---
    # gpu_estimator is intentionally left as None here for the user to replace.
    # Replace the block below with your get_gee(...) call:
    #
    #   from gee.gee_utils import get_gee
    #   gpu_estimator = get_gee(
    #       gpu_yaml_path="...",
    #       lut_yaml_path="...",
    #       dvfs_aware=True, ...
    #   )
    gpu_estimator = None
    # print("currently getting gpu estimator")
    # gpu_estimator = get_gee(gpu_yaml_path="../energaizer-ispass26-artifact/config/gpu/yz8.yaml", 
    #                     lut_yaml_path="../energaizer-ispass26-artifact/experiments_endtoend/exp_config/a100_dvfs_lut_config.yaml", 
    #                     dvfs_aware=True, dvfs_inference_mode='all', 
    #                     dvfs_supply_voltage_json="../energaizer-ispass26-artifact/config/dvfs/yz8/supply_voltage.json",
    #                     dvfs_idle_power_json="../energaizer-ispass26-artifact/config/dvfs/yz8/idle_power.json", 
    #                     lut_folder_abs_path="../energaizer-ispass26-artifact/database/data")
    
    opt_config = None
    if ESTIMATOR_AVAILABLE:
        opt_config = get_opt_config(args.model)
    
    # --- Base experiment config ---
    base = ExperimentConfig(
        model=args.model,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        num_prompts=args.num_prompts,
        output_dir=args.output_dir,
        flexllmgen_script=args.flexllm_script,
        trace_analyzer_script=args.trace_analyzer_script,
        result_analyzer_script=args.result_analyzer_script,
    )
    
    # --- Handle --trace-json shortcut (single experiment from existing trace) ---
    if args.trace_json is not None:
        trace_path = os.path.abspath(args.trace_json)
        if not os.path.isfile(trace_path):
            print(f"ERROR: --trace-json file not found: {trace_path}")
            sys.exit(1)
    
        # Auto-parse parameters from the filename, then let explicit CLI args override
        parsed = parse_flexllm_filename(trace_path)
        print(f"Parsed from filename: {parsed}")
    
        bs      = parsed.get('batch_size',      1)
        nb      = parsed.get('num_batches',     1)
        p       = parsed.get('prompt_len',      args.prompt_len)
        g       = parsed.get('gen_len',         args.gen_len)
        off     = parsed.get('offload_percent', 0.0)
        rc      = parsed.get('recompute_len',   0)
    
        exp = ExperimentConfig(
            model          = args.model,
            prompt_len     = p,
            gen_len        = g,
            num_prompts    = bs * nb,
            batch_size     = bs,
            recompute_len  = rc,
            offload_percent= off,
            output_dir     = args.output_dir,
            flexllmgen_script       = args.flexllm_script,
            trace_analyzer_script   = args.trace_analyzer_script,
            result_analyzer_script  = args.result_analyzer_script,
            trace_json_path  = trace_path,
            summary_csv_path = os.path.abspath(args.summary_csv) if args.summary_csv else None,
        )
        experiments = [exp]
    
    elif args.sweep == "batch_size":
        experiments = sweep_batch_size(
            base, args.batch_sizes, hw, opt_config
        ) if ESTIMATOR_AVAILABLE else [
            dataclasses.replace(base, batch_size=bs, recompute_len=0, offload_percent=0.0)
            for bs in args.batch_sizes
        ]
    elif args.sweep == "recompute":
        base_rc = dataclasses.replace(base, batch_size=args.fixed_batch_size)
        experiments = sweep_recompute(
            base_rc, args.recompute_lens, hw, opt_config
        ) if ESTIMATOR_AVAILABLE else [
            dataclasses.replace(base_rc, recompute_len=rc, offload_percent=0.0)
            for rc in args.recompute_lens
        ]
    else:  # batch_and_recompute
        experiments = sweep_batch_and_recompute(
            base, args.batch_sizes, args.recompute_lens, hw, opt_config
        ) if ESTIMATOR_AVAILABLE else [
            dataclasses.replace(base, batch_size=bs, recompute_len=rc, offload_percent=0.0)
            for bs in args.batch_sizes for rc in args.recompute_lens
        ]
    
    # --- Run ---
    run_comparison(
        experiments=experiments,
        hw=hw,
        estimator_modes=estimator_modes,
        gpu_estimator=gpu_estimator,
        opt_config=opt_config,
        output_csv=args.output_csv,
        dry_run=args.dry_run,
        avg_gt_rows=args.avg_gt_rows,
        trace_cleanup=args.trace_cleanup,
    )


if __name__ == "__main__":
    main()