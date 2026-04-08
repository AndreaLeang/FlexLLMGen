"""
llm_bench.py  (iterated run update)

Key change: run() now accepts n_iters and min_duration_s.
The prefill+decode loop repeats until BOTH conditions are met:
  - at least n_iters iterations completed
  - at least min_duration_s seconds of wall time elapsed

Per-phase stats are averaged across all iterations.
The monitor runs continuously across the entire loop, so even
fast phases (tokenize, detokenize) accumulate enough samples.
"""

import time, json
import sys
sys.path.append( '/home/akleang/akleang/FlexLLMGen/') # to be able to find felxllmgen
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import torch
import numpy as np
from transformers import AutoTokenizer
from power_monitor import PowerMonitor, PowerSample

from flexllmgen.compression import CompressionConfig
from flexllmgen.opt_config import get_opt_config
from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk,
    TorchMixedDevice)
from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool)
from flexllmgen.flex_opt_kvpr_power import OptLM, Policy, get_test_inputs


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class PhaseStats:
    name:              str
    n_iters:           int           # how many iterations contributed
    total_duration_s:  float         # wall time across all iterations
    avg_duration_s:    float         # mean per-iteration duration
    n_tokens_per_iter: int
    avg_cpu_pkg_w:     float
    avg_cpu_dram_w:    float
    avg_gpu_w:         List[float]
    energy_cpu_pkg_j:  float         # total energy (all iters)
    energy_cpu_dram_j: float
    energy_gpu_j:      List[float]

    # layer specific
    layer_type:    str
    batch_num:     int         # the batch this layer is for
    token_gen:     int         # what number token is being generated [0, gen_len-1]

    avg_socket_pkg_w:  List[float] = field(default_factory=list)
    avg_socket_dram_w: List[float] = field(default_factory=list)

    @property
    def throughput_tok_s(self):
        return (self.n_tokens_per_iter / self.avg_duration_s
                if self.avg_duration_s > 0 else 0.0)

    @property
    def energy_per_token_j(self):
        total_e = (self.energy_cpu_pkg_j + self.energy_cpu_dram_j
                   + sum(self.energy_gpu_j))
        total_t = self.n_tokens_per_iter * self.n_iters
        return total_e / total_t if total_t > 0 else 0.0


@dataclass
class InferenceResult:
    prompt:        List[List[int]]
    output:        str           # output from the last iteration
    prompt_tokens: int
    output_tokens: int
    n_iters:       int
    n_iters_layer: int
    total_duration_s: float
    phases:        List[PhaseStats]
    layers:        List[List[List[PhaseStats]]]
    all_samples:   List[PowerSample] = field(repr=False)


# ── Phase accumulator ────────────────────────────────────────────────

class _PhaseAccum:
    """Accumulates power samples across multiple iterations for one phase."""

    def __init__(self, name: str, n_tokens: int, layer_type="None", batch_num=-1, token_gen=-1):
        self.name     = name
        self.n_tokens = n_tokens
        
        # for layer specific accumulation
        self.layer_type = layer_type
        self.batch_num  = batch_num
        self.token_gen  = token_gen
        
        self.slices:  List[List[PowerSample]] = []   # one list per iter
        self.durations: List[float] = []

    def add(self, samples: List[PowerSample], i0: int, i1: int,
             t_start: float, t_end: float):
        self.slices.append(samples[i0:i1])
        self.durations.append(t_end - t_start)

    def to_stats(self) -> PhaseStats:
        # flatten all samples across iterations
        all_s   = [x for sl in self.slices for x in sl]
        n       = len(all_s) or 1
        n_iters = len(self.durations)
        total_dur = sum(self.durations)
        avg_dur   = total_dur / n_iters if n_iters else 0.0

        # infer topology from any available sample
        ref       = all_s[0] if all_s else None
        n_gpus    = len(ref.gpu_w)         if ref else 0
        n_sockets = len(ref.socket_pkg_w)  if ref else 0

        avg_pkg  = sum(x.cpu_pkg_w  for x in all_s) / n
        avg_dram = sum(x.cpu_dram_w for x in all_s) / n
        avg_gpu  = [sum(x.gpu_w[i]  for x in all_s) / n
                    for i in range(n_gpus)]
        skt_pkg  = ([sum(x.socket_pkg_w[i]  for x in all_s) / n
                     for i in range(n_sockets)]
                    if all_s else [0.0] * n_sockets)
        skt_dram = ([sum(x.socket_dram_w[i] for x in all_s) / n
                     for i in range(n_sockets)]
                    if all_s else [0.0] * n_sockets)

        return PhaseStats(
            name=self.name,
            n_iters=n_iters,
            total_duration_s=total_dur,
            avg_duration_s=avg_dur,
            n_tokens_per_iter=self.n_tokens,
            avg_cpu_pkg_w=avg_pkg,   energy_cpu_pkg_j=avg_pkg  * total_dur,
            avg_cpu_dram_w=avg_dram, energy_cpu_dram_j=avg_dram * total_dur,
            avg_gpu_w=avg_gpu,       energy_gpu_j=[g * total_dur for g in avg_gpu],
            avg_socket_pkg_w=skt_pkg,
            avg_socket_dram_w=skt_dram,
            layer_type=self.layer_type,
            batch_num=self.batch_num,
            token_gen=self.token_gen,
        )


# ── Main benchmark class ──────────────────────────────────────────────

class LLMPowerBench:

    def __init__(
        self,
        model_id:            str,
        gpu_indices:         List[int]           = None,
        socket_ids:          Optional[List[int]] = None,
        monitor_interval_ms: int                 = 50,
        block_size:          int                 = 1,
        num_of_blocks:       int                 = 1,
        prompt_len:          int                 = 2048,
        gen_len:             int                 = 16,
        recomp_percent:      int                 = 0,
        recomp_len:          int                 = 0,
        offload_percent:     int                 = 0,
        dtype:               torch.dtype         = torch.float16,
    ):
        self.model_id        = model_id
        self.gpu_indices     = gpu_indices or [0]
        self.socket_ids      = socket_ids
        self.interval_ms     = monitor_interval_ms
        self.block_size      = block_size
        self.num_of_blocks   = num_of_blocks
        self.num_prompts     = block_size * num_of_blocks
        self.prompt_len      = prompt_len
        self.gen_len         = gen_len
        self.recomp_percent  = recomp_percent
        self.offload_percent = offload_percent
        self.dtype          = dtype
        self.tokenizer      = None
        self.model          = None
        self.env            = None
        self.policy         = None


    def load(self):
        ### Redirect imports for flex_opt_kvpr
        print(f"Loading {self.model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        gpu = TorchDevice("cuda:0")
        cpu = TorchDevice("cpu")
        disk = TorchDisk("")
        self.env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
        percent = [100, 0, 100-self.offload_percent, self.offload_percent, 100, 0]
        self.policy = Policy(self.block_size, self.num_of_blocks,
                        percent[0], percent[1],
                        percent[2], percent[3],
                        percent[4], percent[5],
                        True, True, True,
                        False, 1.0,
                        False,
                        CompressionConfig(num_bits=4, group_size=64,
                                        group_dim=0, symmetric=False),
                        False,
                        CompressionConfig(num_bits=4, group_size=64,
                                        group_dim=2, symmetric=False))

        self.opt_config = get_opt_config(self.model_id)
        self.recomp_len = (self.recomp_percent * self.prompt_len) // 100
        
        self.model      = OptLM(self.opt_config, self.env, "~/opt_weights", self.policy, self.recomp_len, False)
        return self


    def run(
        self,
        n_iters:                 int   = 5,         # minimum iterations
        n_iters_layer:           int   = 200,
        min_duration_s:          float = 10.0,      # keep going until this elapsed
        do_sample:               bool  = False,
        temperature:             float = 1.0,
    ) -> InferenceResult:
        """
        Runs prefill+decode repeatedly until both n_iters AND
        min_duration_s are satisfied, then reports averaged stats.
        The power monitor runs continuously across all iterations.
        """

        mon = PowerMonitor(
            interval_ms=self.interval_ms,
            gpu_indices=self.gpu_indices,
            socket_ids=self.socket_ids,
        )

        # tokenize inputs & warmup
        warmup_inputs = get_test_inputs(32, self.num_prompts, self.tokenizer)
        inputs = get_test_inputs(self.prompt_len, self.num_prompts, self.tokenizer)
        num_layers = self.model.num_layers
        num_gpu_batches = self.num_of_blocks
        gpu_batch_size = self.block_size
        overlap = self.policy.overlap
        prompt_len, gen_len = self.prompt_len, self.gen_len

        # Warm up
        print("\n[warm-up] 3 iters, no min duration ...")
        for i in range(3):
            output_ids = self.model.generate(warmup_inputs, max_new_tokens=1, verbose=0)
        print("[warm-up] done\n")
        self.model.execute_gen_len = self.gen_len

        # Setting up 
        task = Task(
            inputs=inputs,
            prompt_len= prompt_len,
            gen_len=gen_len,
            cut_gen_len=None,
            do_sample=False,
            temperature=1.0,
            stop=None,
        )
        self.model.set_task(task)
      
        # accumulators — one per phase
        acc_prefill = _PhaseAccum("prefill",    prompt_len)
        acc_decode  = _PhaseAccum("decode",     gen_len)
              
        tot_refresh_cache_time = 0
        iteration = 0
        mon.start()
        loop_start = time.perf_counter()

        while True:
            iteration += 1
            # ── Refresh Cache ───────────────────────────────────────────────
            # Intermediate tensors
            # The following buffers store values used
            # for the i-th token, j-th layer, k-th gpu batch.
            t0 = time.perf_counter()
            # Output token ids
            self.model.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
                self.model.config.pad_token_id, dtype=np.int32)
            self.model.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
            self.model.output_ids[:, :prompt_len] = np.asarray(task.inputs)
            #Intermediate Tensors
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.model.cache_home[j][k].clear()
                    self.model.cache_read_buf[j][k].clear()
                    self.model.cache_write_buf[j][k].clear()
                    self.model.cpu_cache_read_buf[j][k].clear()
                    self.model.hidden_compute_home[j][k].clear()
                    self.model.hidden_compute_read_buf[j][k].clear()
            for j in range(num_layers):
                self.model.weight_read_buf[j].clear()
            for k in range(num_gpu_batches):
                self.model.attention_mask[k].clear()
            self.model.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)
            t1 = time.perf_counter()
            tot_refresh_cache_time += t1-t0

            # ── prefill ───────────────────────────────────────────────
            t0 = time.perf_counter()
            i0 = len(mon.samples)
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.model.init_cache(j, k)
                    if self.recomp_len > 0:
                        self.model.init_hidden(j, k)
            torch.cuda.synchronize()
            i1 = len(mon.samples)
            t1 = time.perf_counter()
            acc_prefill.add(mon.samples, i0, i1, t0, t1)

            # ── decode ────────────────────────────────────────────────
            t0 = time.perf_counter()
            i0 = len(mon.samples)
          
            # Power Caputure for entire inference
            if num_gpu_batches == 1:
                self.model.generation_loop_overlap_single_batch()
            else:
                self.model.generation_loop_overlap_multi_batch()
          
            out_ids = self.model.output_ids

            torch.cuda.synchronize()
            i1 = len(mon.samples)
            t1 = time.perf_counter()
            acc_decode.add(mon.samples, i0, i1, t0, t1)

            elapsed    = time.perf_counter() - loop_start

            print(f"iterations {iteration}  "
                  f"prefill {acc_prefill.durations[-1]*1000:.0f}ms  "
                  f"decode {acc_decode.durations[-1]*1000:.0f}ms  "
                  f"elapsed {elapsed:.1f}s")

            # stop when BOTH conditions met
            if iteration >= n_iters and elapsed >= min_duration_s:
                break

        mon.stop()
        total_dur = time.perf_counter() - loop_start - tot_refresh_cache_time

        # decode output from last iteration
        new_ids    = out_ids[0][prompt_len:]
        output     = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        output_len = len(new_ids)

        phases = [acc_prefill.to_stats(), acc_decode.to_stats()]
        layers = []
        
        return InferenceResult(
            prompt=task.inputs, output=output,
            prompt_tokens=prompt_len, output_tokens=output_len,
            n_iters=iteration, n_iters_layer = n_iters_layer, total_duration_s=total_dur,
            phases=phases, layers=layers, all_samples=mon.samples,
        )
      

    def run_layers(
        self,
        n_iters:                 int   = 5,         # minimum iterations
        n_iters_layer:           int   = 200,
        min_duration_s:          float = 10.0,      # keep going until this elapsed
        do_sample:               bool  = False,
        temperature:             float = 1.0,
    ) -> InferenceResult:
        """
        Runs prefill+decode repeatedly until both n_iters AND
        min_duration_s are satisfied, then reports averaged stats.
        The power monitor runs continuously across all iterations.
        """

        mon = PowerMonitor(
            interval_ms=self.interval_ms,
            gpu_indices=self.gpu_indices,
            socket_ids=self.socket_ids,
        )

        # tokenize inputs & warmup
        warmup_inputs = get_test_inputs(32, self.num_prompts, self.tokenizer)
        inputs = get_test_inputs(self.prompt_len, self.num_prompts, self.tokenizer)
        num_layers = self.model.num_layers
        num_gpu_batches = self.num_of_blocks
        gpu_batch_size = self.block_size
        overlap = self.policy.overlap
        prompt_len, gen_len = self.prompt_len, self.gen_len

        # Warm up
        print("\n[warm-up] 3 iters, no min duration ...")
        for i in range(3):
            output_ids = self.model.generate(warmup_inputs, max_new_tokens=1, verbose=0)
        print("[warm-up] done\n")
        self.model.execute_gen_len = self.gen_len

        # Setting up 
        task = Task(
            inputs=inputs,
            prompt_len= prompt_len,
            gen_len=gen_len,
            cut_gen_len=None,
            do_sample=False,
            temperature=1.0,
            stop=None,
        )
        self.model.set_task(task)
      
        # accumulators — one per phase
        acc_prefill = _PhaseAccum("prefill",    prompt_len)
        acc_decode  = _PhaseAccum("decode",     gen_len)
        all_acc_layers = []
        for i in range(gen_len):
            cur_gen = []
            for j in range(num_layers):
                cur_layer = []
                for k in range(num_gpu_batches):
                    cur_layer.append( _PhaseAccum("layer_"+str(j)+"_B_"+str(k) + "_G_"+str(i), 1, layer_type="None", batch_num=k, token_gen=i))
                cur_gen.append(cur_layer)
            all_acc_layers.append(cur_gen)
        # L_0_B_0_G_0,  L_0_B_1_G_0, L_0_B_2_G_0, L_0_B_3_G_0, L_1_B_0_G_0, L_1_B_1_G_0, ...
      
        tot_refresh_cache_time = 0
        iteration = 0
        mon.start()
        loop_start = time.perf_counter()

        while True:
            iteration += 1
            # ── Refresh Cache ───────────────────────────────────────────────
            # Intermediate tensors
            # The following buffers store values used
            # for the i-th token, j-th layer, k-th gpu batch.
            t0 = time.perf_counter()
            # Output token ids
            self.model.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
                self.model.config.pad_token_id, dtype=np.int32)
            self.model.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
            self.model.output_ids[:, :prompt_len] = np.asarray(task.inputs)
            #Intermediate Tensors
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.model.cache_home[j][k].clear()
                    self.model.cache_read_buf[j][k].clear()
                    self.model.cache_write_buf[j][k].clear()
                    self.model.cpu_cache_read_buf[j][k].clear()
                    self.model.hidden_compute_home[j][k].clear()
                    self.model.hidden_compute_read_buf[j][k].clear()
            for j in range(num_layers):
                self.model.weight_read_buf[j].clear()
            for k in range(num_gpu_batches):
                self.model.attention_mask[k].clear()
            self.model.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)
            t1 = time.perf_counter()
            tot_refresh_cache_time += t1-t0

            # ── prefill ───────────────────────────────────────────────
            t0 = time.perf_counter()
            i0 = len(mon.samples)
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.model.init_cache(j, k)
                    if self.recomp_len > 0:
                        self.model.init_hidden(j, k)
            torch.cuda.synchronize()
            i1 = len(mon.samples)
            t1 = time.perf_counter()
            acc_prefill.add(mon.samples, i0, i1, t0, t1)

            # ── decode ────────────────────────────────────────────────
            t0 = time.perf_counter()
            i0 = len(mon.samples)
          
            # Power Caputure for entire inference
            # if num_gpu_batches == 1:
            #     self.model.generation_loop_overlap_single_batch()
            # else:
            #     self.model.generation_loop_overlap_multi_batch()
          
            # Power Capture layer by layer
            if num_gpu_batches == 1:
                # Prologue
                for k in range(self.model.num_gpu_batches):
                    self.model.load_weight(0, 0, k)
                self.model.sync()
        
                # Generate
                for i in range(self.model.execute_gen_len):
                    # print(f"i:{i}, self.model.execute_gen_len: {self.model.execute_gen_len}")
                    self.model.update_attention_mask(i, 0)
                    for j in range(num_layers):
                        # print(f"i:{i}, j: {j}")
                        for each_iter in range(n_iters_layer):
                            # print(f"each_iter: {each_iter}, i: {i}, j: {j}")
                            repeating = each_iter!=(n_iters_layer-1)
                            # if i > 0:
                            #     print(f"each_iter: {each_iter}, i: {i}, j: {j}")
                            lt0 = time.perf_counter()
                            li0 = len(mon.samples)
                            self.model.load_weight(i, j+1, 0)
                            self.model.load_hidden_compute(i,j+1, 0)
                            self.model.load_cache(i, j+1, 0)
                            self.model.load_hidden(i, j, 0, repeating=repeating)
                            self.model.compute_layer(i, j, 0, repeating=repeating)
                            self.model.store_cache(i, j-1, 0, repeating=repeating)
                            self.model.store_hidden(i, j, 0, repeating=repeating)
                            self.model.sync()
                            li1 = len(mon.samples)
                            lt1 = time.perf_counter()
                            all_acc_layers[i][j][0].add(mon.samples, li0, li1, lt0, lt1)
        
                    
            else:
                # Prologue
                for k in range(num_gpu_batches):
                    self.model.load_weight(0, 0, k)
                self.model.load_hidden(0, 0, 0)
                self.model.sync()
        
                # Generate
                for i in range(self.model.execute_gen_len):
                    for k in range(num_gpu_batches):
                        self.update_attention_mask(i, k)
                    for j in range(num_layers):
                        for k in range(num_gpu_batches):
                            for each_iter in range(n_iters_layer):
                              repeating = each_iter!=(n_iters_layer-1)
                              lt0 = time.perf_counter()
                              li0 = len(mon.samples)
                              self.model.load_weight(i, j+1, k)
                              self.model.load_hidden_compute(i,j, k+1)
                              self.model.load_cache(i, j, k+1)
                              self.model.store_hidden(i, j, k-1, repeating=repeating)
                              self.model.load_hidden(i, j, k+1, repeating=repeating)
                              self.model.compute_layer(i, j, k, repeating=repeating)
                              self.model.store_cache(i, j, k-1, repeating=repeating)
                              self.model.sync()
                              li1 = len(mon.samples)
                              lt1 = time.perf_counter()
                              all_acc_layers[i][j][k].add(mon.samples, li0, li1, lt0, lt1)
                          
        
                # Epilogue
                self.model.store_hidden(gen_len-1, num_layers-1, num_gpu_batches-1)
          
            out_ids = self.model.output_ids

            torch.cuda.synchronize()
            i1 = len(mon.samples)
            t1 = time.perf_counter()
            acc_decode.add(mon.samples, i0, i1, t0, t1)

            elapsed    = time.perf_counter() - loop_start

            print(f"iterations {iteration}  "
                  f"prefill {acc_prefill.durations[-1]*1000:.0f}ms  "
                  f"decode {acc_decode.durations[-1]*1000:.0f}ms  "
                  f"elapsed {elapsed:.1f}s")

            # stop when BOTH conditions met
            if iteration >= n_iters and elapsed >= min_duration_s:
                break

        mon.stop()
        total_dur = time.perf_counter() - loop_start - tot_refresh_cache_time

        # decode output from last iteration
        new_ids    = out_ids[0][prompt_len:]
        output     = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        output_len = len(new_ids)

        phases = [acc_prefill.to_stats(), acc_decode.to_stats()]
        layers = []
        for i in range(gen_len):
            cur_gen = []
            for j in range(num_layers):
                cur_layer = []
                for k in range(num_gpu_batches):
                    cur_layer.append(all_acc_layers[i][j][k].to_stats())
                cur_gen.append(cur_layer)
            layers.append(cur_gen)      

        return InferenceResult(
            prompt=task.inputs, output=output,
            prompt_tokens=prompt_len, output_tokens=output_len,
            n_iters=iteration, n_iters_layer = n_iters_layer, total_duration_s=total_dur,
            phases=phases, layers=layers, all_samples=mon.samples,
        )

    def print_report(self, r: InferenceResult):
        n_sockets = len(r.phases[0].avg_socket_pkg_w)
        sep = "─" * (82 + n_sockets * 18)
        print(sep)
        print(f"  Prompt tokens: {r.prompt_tokens}   "
              f"Output tokens: {r.output_tokens}   "
              f"Iterations: {r.n_iters}   "
              f"Total: {r.total_duration_s:.1f}s")
        print(sep)
        skt_hdr = "".join(f"  S{i}-pkg  S{i}-drm" for i in range(n_sockets))
        print(f"  {'Phase':<10} {'Iters':>5} {'Avg dur':>8} {'Tok/s':>7} "
              f"{'TOT-pkg':>8} {'TOT-drm':>8} {'GPU-0':>7}{skt_hdr}  {'J/tok':>7}")
        print(sep)
        for p in r.phases:
            gpu0    = p.avg_gpu_w[0] if p.avg_gpu_w else 0.0
            skt_str = "".join(
                f"  {p.avg_socket_pkg_w[i]:>6.1f}W {p.avg_socket_dram_w[i]:>6.1f}W"
                for i in range(n_sockets)
            )
            print(f"  {p.name:<10} {p.n_iters:>5} {p.avg_duration_s:>7.3f}s "
                  f"{p.throughput_tok_s:>7.1f} "
                  f"{p.avg_cpu_pkg_w:>7.1f}W {p.avg_cpu_dram_w:>7.1f}W "
                  f"{gpu0:>6.1f}W{skt_str}  {p.energy_per_token_j:>7.4f}")
        for each_gen_layer_list in r.layers:
            for each_layer_batch_list in each_gen_layer_list:
                for p in each_layer_batch_list:
                    print(f"p: {p}")
                    gpu0    = p.avg_gpu_w[0] if p.avg_gpu_w else 0.0
                    skt_str = "".join(
                        f"  {p.avg_socket_pkg_w[i]:>6.1f}W {p.avg_socket_dram_w[i]:>6.1f}W"
                        for i in range(n_sockets)
                    )
                    print(f"  {p.name:<10} {p.n_iters:>5} {p.avg_duration_s:>7.3f}s "
                          f"{p.throughput_tok_s:>7.1f} "
                          f"{p.avg_cpu_pkg_w:>7.1f}W {p.avg_cpu_dram_w:>7.1f}W "
                          f"{gpu0:>6.1f}W{skt_str}  {p.energy_per_token_j:>7.4f}")
        print(sep)

    def save_json(self, r: InferenceResult, path: str):
        layers_dict = []
        for each_gen_list in r.layers:
            cur_gen = []
            for each_layer_list in each_gen_list:
                cur_layer = []
                for each_layer in each_layer_list:
                    cur_layer.append(asdict(each_layer))
                cur_gen.append(cur_layer)
            layers_dict.append(cur_gen)   
        with open(path, "w") as f:
            json.dump({
                "prompt": r.prompt, "output": r.output,
                "prompt_tokens": r.prompt_tokens,
                "output_tokens": r.output_tokens,
                "n_iters": r.n_iters,
                "total_duration_s": r.total_duration_s,
                "phases": [asdict(p) for p in r.phases],
                "layers": layers_dict,
            }, f, indent=2)
        print(f"Saved → {path}")
