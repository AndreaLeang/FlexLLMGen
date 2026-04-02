"""
run_experiment.py
Sweeps over prompt lengths and output lengths.
Each cell runs for at least n_iters iterations AND min_duration_s seconds.
Results are saved per-cell as JSON and aggregated into a single CSV.

Usage:
    python run_experiment.py --model meta-llama/Llama-3.2-1B --gpu 0
    python run_experiment.py --model meta-llama/Llama-3.1-8B --gpu 0 1 \
                             --n_iters 10 --min_duration 30
"""

import argparse, json, time, csv, os
from dataclasses import asdict
from pathlib import Path
from llm_bench import LLMPowerBench


# ── Sweep configuration ───────────────────────────────────────────────

PROMPTS = {
    "short": "What is PCIe?",
    "medium": (
        "Explain the differences between PCIe Gen 4 and Gen 5, "
        "including bandwidth, power, and use cases for AI accelerators."
    ),
    "long": (
        "You are an expert in computer architecture. Provide a detailed "
        "technical analysis of how KV cache offloading from GPU to CPU DRAM "
        "affects inference latency and power consumption in large language "
        "models. Discuss the role of PCIe bandwidth, CPU DRAM throughput, "
        "and GPU utilization. Include specific considerations for A100 GPUs."
    ),
}

OUTPUT_LENGTHS = [32, 128, 256]

# cooldown between cells to let GPU thermals settle
COOLDOWN_S = 5


# ── Helpers ───────────────────────────────────────────────────────────

def cell_tag(model_name:str, prompt_len: int, gen_len: int, num_prompts: int, batch_size: int) -> str:
    model_name_spec = model_name.split('/')[1]
    return f"{model_name_spec}_pLen_{prompt_len}_gLen_{gen_len}_numP_{num_prompts}_bSize_{batch_size}"


def flat_row(tag: str, result) -> dict:
    """Flatten one InferenceResult into a single CSV row."""
    row = {
        "cell":             tag,
        "prompt_tokens":    result.prompt_tokens,
        "output_tokens":    result.output_tokens,
        "n_iters":          result.n_iters,
        "total_duration_s": result.total_duration_s,
    }
    for phase in result.phases:
        p = phase.name
        row[f"{p}_avg_dur_s"]        = phase.avg_duration_s
        row[f"{p}_tok_per_s"]        = phase.throughput_tok_s
        row[f"{p}_avg_cpu_pkg_w"]    = phase.avg_cpu_pkg_w
        row[f"{p}_avg_cpu_dram_w"]   = phase.avg_cpu_dram_w
        row[f"{p}_energy_cpu_pkg_j"]  = phase.energy_cpu_pkg_j
        row[f"{p}_energy_cpu_dram_j"] = phase.energy_cpu_dram_j
        row[f"{p}_energy_per_tok_j"]  = phase.energy_per_token_j
        for i, gw in enumerate(phase.avg_gpu_w):
            row[f"{p}_avg_gpu{i}_w"]      = gw
            row[f"{p}_energy_gpu{i}_j"]   = phase.energy_gpu_j[i]
        for i, sw in enumerate(phase.avg_socket_pkg_w):
            row[f"{p}_avg_s{i}_pkg_w"]    = sw
            row[f"{p}_avg_s{i}_dram_w"]   = phase.avg_socket_dram_w[i]
    return row


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--gpu",          nargs="+", type=int, default=[0])
    parser.add_argument("--sockets",      nargs="+", type=int, default=None,
                        help="RAPL socket ids (default: auto-detect)")
    parser.add_argument("--interval_ms",  type=int,   default=50)
    parser.add_argument("--n_iters",      type=int,   default=5,
                        help="minimum iterations per cell")
    parser.add_argument("--min_duration", type=float, default=10.0,
                        help="minimum seconds per cell")
    parser.add_argument("--out_dir",      default="rapl-nvml-power-monitor-main/power_results")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--gen-len", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--off-per", type=int, default=0)
    parser.add_argument("--recomp-per", type=int, default=0)



    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True)

    bench = LLMPowerBench(
        model_id=args.model,
        gpu_indices=args.gpu,
        socket_ids=args.sockets,
        monitor_interval_ms=args.interval_ms,
        block_size= args.block_size,
        num_of_blocks= args.num_blocks,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        recomp_percent=args.recomp_per,
        offload_percent=args.off_per,
    ).load()

    # ── warm-up (not recorded) ────────────────────────────────────────

    all_rows   = []
    n_cells    = len(PROMPTS) * len(OUTPUT_LENGTHS)
    cell_count = 0

    # FIX BELOW

    tag = cell_tag(args.model, args.prompt_len, args.gen_len, args.block_size*args.num_blocks, args.block_size)

    print(f"{'='*64}")
    print(f"  [{cell_count}/{n_cells}]  {tag}")
    print(f"{'='*64}")

    result = bench.run(
        n_iters=args.n_iters,
        min_duration_s=args.min_duration,
    )
    bench.print_report(result)

    # save per-cell JSON (full detail including raw phase data)
    json_path = out_dir / f"{tag}.json"
    bench.save_json(result, str(json_path))

    # save per-cell power trace CSV
    csv_path = out_dir / f"{tag}_trace.csv"
    mon_stub = type("M", (), {
        "samples":     result.all_samples,
        "gpu_indices": args.gpu,
    })()
    # reuse PowerMonitor.save_csv logic inline
    n_sockets = (len(result.all_samples[0].socket_pkg_w)
                    if result.all_samples else 0)
    skt_cols  = ([f"s{i}_pkg_w"  for i in range(n_sockets)] +
                    [f"s{i}_dram_w" for i in range(n_sockets)])
    gpu_cols  = [f"gpu{i}_w" for i in args.gpu]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "cpu_pkg_w", "cpu_dram_w"]
                    + skt_cols + gpu_cols)
        for x in result.all_samples:
            w.writerow([
                f"{x.timestamp:.4f}",
                f"{x.cpu_pkg_w:.2f}", f"{x.cpu_dram_w:.2f}",
                *[f"{v:.2f}" for v in x.socket_pkg_w],
                *[f"{v:.2f}" for v in x.socket_dram_w],
                *[f"{g:.2f}" for g in x.gpu_w],
            ])

    all_rows.append(flat_row(tag, result))


    # ── write aggregated summary CSV ──────────────────────────────────
    summary_path = out_dir / "power_summary.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

    print(f"\n{'='*64}")
    print(f"  Done.  {cell_count} cells completed.")
    print(f"  Summary CSV  → {summary_path}")
    print(f"  Per-cell JSON/trace → {out_dir}/")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
