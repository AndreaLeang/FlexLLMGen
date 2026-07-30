#!/usr/bin/env python3
"""
flex_bench_runner.py
=====================
Thin wrapper around flexllmgen.flex_opt_kvpr.run_flexllmgen() for direct
(non-profiling) throughput/latency benchmarking, driven by gt_vs_estimator.py's
run_flexllm_bench().

Reuses flex_opt_kvpr.py's own CLI surface (add_parser_arguments()) so every
--model/--percent/--gpu-batch-size/... flag gt_vs_estimator.py already knows
how to build works unchanged here -- --profile/--save-to are simply unused
(this script never profiles). One new flag is added:

  --bench-num-runs N   Total run_flexllmgen() calls (default 6). Run 0 is
                        always reported with is_warmup=true; it is the
                        caller's job to drop it before averaging (see
                        gt_vs_estimator.py's aggregate_bench_runs()) --
                        this script reports every run's raw numbers as-is.

Why call run_flexllmgen() directly instead of parsing flex_opt_kvpr.py's own
stdout: those prints are for a human (formatting/units/--verbose-gated), not
a stable machine interface -- they can and do change independent of this
script. Calling the function directly and emitting our OWN single-line JSON
result means the only thing we depend on is run_flexllmgen()'s return value
(total_throughput, decode_latency), which is already exactly the
prefill-excluded decode-phase pair gt_vs_estimator.py's optimal-config
comparison needs (decode_latency = sum(costs[1:]), i.e. costs[0]/prefill is
already dropped). flex_opt_kvpr.py's own prints still happen and still go to
stdout as before (useful context for a human re-reading the log) -- the
result line is simply the last line, with a sentinel prefix the caller greps
for.

flex_opt_kvpr.py's own __main__ already calls run_flexllmgen() repeatedly in
a single process (see its --sweep-average option) and relies on
OptLM.__del__ -> delete_all_weights() to release the previous iteration's
weights before the next allocates -- this script uses the exact same
pattern, once per --bench-num-runs.

Output (last line of stdout)
-----------------------------
  FLEXBENCH_RESULT <json>

  <json> is one of:
    {"ok": true, "runs": [
        {"run_idx": 0, "is_warmup": true,
         "total_throughput": <tok/s>, "decode_latency_s": <s>,
         "decode_throughput_tok_per_s": <tok/s>},
        ... one entry per --bench-num-runs ...
    ]}

    {"ok": false, "run_idx": <int>, "error": "<message>"}
      -- run_flexllmgen() returned None (e.g. model-init failure). Note:
      flex_opt_kvpr.py's own init path swallows the real exception (a bare
      `except:` around `OptLM(...)`), so "error" here is necessarily
      generic -- there is no traceback to forward. A genuine CUDA OOM
      during generate() is NOT caught anywhere in flex_opt_kvpr.py and
      instead propagates as a real exception -> nonzero exit code and
      "out of memory" text in stderr, same as the existing --profile path
      (see gt_vs_estimator.py's _is_oom_output()). Exits with code 1 in
      this case, same as the propagating-exception case, so both look
      like "some kind of failure" at the exit-code level to the caller;
      only the OOM-text-in-stderr case can be told apart from a generic
      error.

Usage
-----
  python flex_bench_runner.py --model facebook/opt-6.7b \\
      --gpu-batch-size 4 --num-gpu-batches 4 --prompt-len 2048 --gen-len 16 \\
      --percent 100 0 80 20 100 0 --recompute-len 0 --bench-num-runs 6
"""

import argparse
import json
import sys

from flexllmgen.flex_opt_kvpr import add_parser_arguments, run_flexllmgen

RESULT_PREFIX = "FLEXBENCH_RESULT "


def main():
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    parser.add_argument(
        "--bench-num-runs", type=int, default=6, metavar="N",
        help=(
            "Total run_flexllmgen() calls (default 6). Run 0 is always "
            "reported with is_warmup=true; the caller decides whether/how "
            "to drop it before averaging."
        ),
    )
    args = parser.parse_args()

    assert len(args.percent) == 6, "need 6 arguments in percent"
    # This script never profiles -- force it off regardless of how the
    # caller assembled the command line, so a stray --profile can't
    # silently switch run_flexllmgen() into the trace-recording branch
    # (which needs --save-to and writes a .json this script has no use
    # for and never sets up a directory for).
    args.profile = False

    runs = []
    for i in range(args.bench_num_runs):
        kind = "warmup" if i == 0 else "measured"
        print(f"[flex_bench_runner] run {i}/{args.bench_num_runs - 1} ({kind})...",
              file=sys.stderr)

        result = run_flexllmgen(args)
        if result is None:
            payload = {
                "ok": False,
                "run_idx": i,
                "error": (
                    "run_flexllmgen() returned None (model-init failed -- "
                    "flex_opt_kvpr.py's own except-block around OptLM(...) "
                    "swallows the real exception, so no further detail is "
                    "available here)."
                ),
            }
            print(RESULT_PREFIX + json.dumps(payload))
            sys.exit(1)

        total_throughput, decode_latency = result
        num_prompts = args.num_gpu_batches * args.gpu_batch_size
        # Same formula run_flexllmgen() itself uses internally for
        # decode_throughput (it just doesn't return it) -- gen_len - 1
        # because costs[0]/the first token (prefill) is already excluded
        # from decode_latency.
        decode_throughput = num_prompts * (args.gen_len - 1) / max(decode_latency, 1e-10)

        runs.append({
            "run_idx": i,
            "is_warmup": i == 0,
            "total_throughput": total_throughput,
            "decode_latency_s": decode_latency,
            "decode_throughput_tok_per_s": decode_throughput,
        })

    print(RESULT_PREFIX + json.dumps({"ok": True, "runs": runs}))


if __name__ == "__main__":
    main()