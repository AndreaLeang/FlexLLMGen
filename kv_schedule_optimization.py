"""
Cost Model for OPT in FlexLLMGen.

Dependencies:
pip install pulp

Example Usages:
1. Find a policy:
python cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 32 \
                     --gpu-mem 16 --cpu-mem 200 --nvme-mem 1500
2. Estimate the throughput for a given policy:
python cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 32 \
                     --gpu-mem 16 --cpu-mem 200 --nvme-mem 1500 \
                     --gpu-batch-size 48 --num-gpu-batches 3 \
                     --percent 20 80 0 100 0 100 \
                     --alpha-g 1.2 --alpha-c 1.2 --alpha-n 1.2

Note:
1. You need to fit the hardware constants for your device,
   see class CostModelConfig.
   (We fit them using gradient descent by collecting real run data points.
    Profiling for primitive modules or take numbers from the internet will not be accurate.)
2. Adjust relaxation ratio alpha_g, alpha_c, and alpha_n carefully,
   a smaller ratio cause a conservative policy,
   and a larger ratio cause an aggresive policy.
3. The cost model does not consider CPU compute delegation,
   and the support for quantization is incomplete.
4. In the second use case of estimating throughput for a fixed policy,
   relax the constraints alpha_g, alpha_c, and alpha_n to allow imprecise peak memory estimation.
"""

import argparse
import dataclasses
import math
import numpy as np
import pulp

from flexllmgen.compression import CompressionConfig, Policy
from flexllmgen.opt_config import get_opt_config
# from flexllmgen.flex_opt import Policy
from flexllmgen.utils import GB, T

alpha_g = 0.8
alpha_c = 0.8
alpha_n = 0.8


@dataclasses.dataclass
class CostModelConfig:
    s: int = 512
    n: int = 32

    l: int = 96
    h1: int = 12288
    h2: int = 12288 * 4
    nh: int = 96

    gmem: int = 15 * GB
    cmem: int = 204 * GB
    nmem: int = 1500 * GB

    # hardware constants
    # default value aligned on google cloud T4
    ctog_bdw: float = 12.89 * GB
    gtoc_bdw_cache: float = 0.97 * GB
    gtoc_bdw_hidden: float = 4.82 * GB

    dtoc_bdw: float = 0.473 * GB
    ctod_bdw_cache_p: float = 0.746 * GB
    ctod_bdw_hidden_p: float = 2.015 * GB
    ctod_bdw_g: float = 2.015 * GB

    mm_flops_p: float = 21.24 * T
    mm_flops_g: float = 4.3 * T
    bmm_flops_p: float = 9.97 * T
    bmm_flops_g: float = 0.079 * T
    cpu_flops: float = 0.0123 * T

    c1: float = 0.0168
    c2: float = 0.0328
    c3: float = 0.0621



def get_available_offloadings(model, hardware_config, batch_sizes):
    total_available_gpu = hardware_config.gpu_mem # Bytes
    total_weight_bytes = model.model_bytes() # Bytes
    num_heads = model.n_head

    batch_size_to_distinct_offloadings = {
        1:[0, 100],
        2:[0, 50, 100],
        4:[0, 20, 50, 70, 100],
        8:[0, 10, 20, 30, 50, 60, 70, 80, 100],
        16:[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }

    feasible_strategies = {}
    for each_batch_size in batch_sizes:
        for each_possible_offloading in batch_size_to_distinct_offloadings[each_batch_size]:
            total_kv_cache_bytes = model.cache_bytes(batch_size, seq_len) # Bytes (Toal KV Cache per forward pass) (seq_len = prompt_len+gen_len)
            total_hidden_bytes = model.hidden_bytes(batch_size, seq_len) # Bytes (Total Hidden State per forward pass) (seq_len = prompt_len+gen_len)
            num_prompts_on_gpu = int(each_batch_size* num_heads * (100-each_possible_offloading) / 100) // num_heads
            actual_kv_cache_bytes = (num_prompts_on_gpu / each_batch_size) * total_kv_cache_bytes

            if total_weight_bytes + actual_kv_cache_bytes + total_hidden_bytes <= total_available_gpu:
                if feasible_strategies[each_batch_size] is None:
                    feasible_strategies[each_batch_size] = []
                feasible_strategies[each_batch_size].append(each_possible_offloading)
    return feasible_strategies
    

def get_batch_sizes(model, num_of_prompts, prompt_len, gen_len, hardware_config):
    num_of_prompts = model.num_of_prompts
    possible_batch_sizes = [1]
    cur_num_batches = num_of_prompts
    # test if %2 and repeat until none
    while cur_num_batches % 2 == 0 and cur_num_batches > 0:
        possible_batch_sizes.append(num_of_prompts // cur_num_batches)
        cur_num_batches //= 2
    return possible_batch_sizes


def strategy_prediction(model, num_of_prompts, prompt_len, gen_len, hardware_config, recomp_len, offload_percent, batch_size, num_batches):
    #offloading percent is amount offloaded to the cpu
    
    tot_energy = 0
    tot_latency = 0
    num_hidden_layers = model.num_hidden_layers

    # Forward Pass Prediction
    #is_load_store: 0: none, 1: load only, 2: store only, 3: load and store
    input_output_latency = layer_prediction(model, 1, "input", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len) + (num_batches-1)*layer_prediction(model, 0, "input", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len)
                          +(num_batches)*layer_prediction(model, 1, "output", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len) + layer_prediction(model, 0, "output", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len)
    
    for cur_gen_len in range(1, gen_len+1):
        if num_batches == 1:
            tot_MHA_latency = num_hidden_layers*(layer_prediction(model, 1, "MHA", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len))
            tot_MLP_latency = num_hidden_layers*(layer_prediction(model, 2, "MLP", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len))
        else:
            tot_MHA_latency = num_hidden_layers*(layer_prediction(model, 1, "MHA", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len) + layer_prediction(model, 2, "MHA", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len) + (num_batches-2)*layer_prediction(model, 3, "MHA", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len))
            tot_MLP_latency = (num_hidden_layers-1)*(layer_prediction(model, 2, "MLP", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len) + layer_prediction(model, 1, "MLP", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len) + (num_batches-2)*layer_prediction(model, 0, "MLP", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len))
            tot_MLP_latency += layer_prediction(model, 2, "MLP", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len) + (num_batches-1)*layer_prediction(model, 0, "MLP", batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len)

        middle_layer_latency = tot_MHA_latency + tot_MLP_latency

        one_forward_latency = input_output_latency + middle_layer_latency
        tot_latency += one_forward_latency

    # get total energy and latency
    return tot_energy, tot_latency

def get_bytes_to_load(model, batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len):
    recomp_load_bytes = recomp_len * 8192 * batch_size # 8192 bytes/token
    kv_load_bytes = (prompt_len + gen_len-recomp_len) * 8192 * (batch_size-((batch_size*(100-offload_percent))//100))
    return recomp_load_bytes, kv_load_bytes

def get_bytes_to_store(model, batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len):
    kv_store_bytes = batch_size * 8192
    return kv_store_bytes

def layer_prediction(model, is_load_store, layer_type="MHA", batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len):
    #layer type determines the actual recomputation time + compute layer time
    if is_load_store == 0:
        #no load or store, just the layer computations
        #TODO: have teh model carry layer latency
        return model.layer_latency(layer_type)
    elif is_load_store == 2:
        # store only --> single directional
        return max(model.layer_latency(layer_type), transfer_pred(get_bytes_to_store(model, batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len), hardware_config))
    else:
        recomp_bytes, kv_load_bytes = get_bytes_to_load(model, batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len)
        #use is_load_store==1 as single directional
        #recomp transfer & first kv load are always single directional. the second kv load uses single_directional
        pinned_latency = pinned_pred(kv_load_bytes, hardware_config)
        first_half_latency = max(pinned_time, recomp_prep_pred(prompt_len, recomp_len, hardware_config)+transfer_pred(recomp_bytes, hardware_config))
        recomp_latency = 0
        if layer_type == "MHA":
            recomp_latency = recomp_pred(recomp_len, hardware_config)
        second_single_dir = is_load_store == 1
        second_half_latency = max(pinned_latency + recomp_latency + model.layer_latency(layer_type), transfer_pred(kv_load_bytes, hardware_config)+ transfer_pred(kv_load_bytes, hardware_config, single_directional = second_single_dir))

        return first_half_latency + second_half_latency


def pinned_pred(bytes, hardware_config):
    return 5.168e-14 * bytes^2 + 3.317e-05 * bytes - 104.7


def recomp_prep_pred(prompt_len, recomp_len, hardware_config):
    #TODO: collect data
    return None

def transfer_pred(bytes, hardware_config, single_directional=True):
    if single_directional:
        return 1.415 * math.log10(bytes) + 13.38
    else:
        return 3.626 * math.log10(bytes) + 26.13 

def recomp_pred(recomp_len, hardware_config):
    #TODO: collect data
    return None


def disect_input(model, opt_config, num_of_prompts, prompt_len, gen_len, hardware_config, var_to_min="latency"):
    # break model into layers
  
    ### UNDERSTAND WHAT STRATEGIES ARE AVVAILABLE
    # understand what unique batch size is available 
    batch_sizes = get_batch_sizes(model, hardware_config)
    print(f'got batch sizes: {batch_sizes}')
    # understand what % offloadings are available 
    all_feasible_strategies_dict = get_available_offloadings(model, hardware_config, batch_sizes)
    print(f'got all feasible strats: {all_feasible_strategies_dict}')
    
    ### ITERATE AND COMPARE STRATEGIES

    #iterate through 0-100% recomputing, % offloading, and batch size
    min_objective_val = float('inf')
    min_strategy = None

    for each_batch_size in batch_sizes:
        for each_feasible_offloading in all_feasible_strategies_dict[each_batch_size]:
            for each_recomp_percent in range(0, 100, 10):
                each_recomp_len = prompt_len * each_recomp_percent // 100 # recomp is only for prompt len
                print(f'cur strat: {each_batch_size}, {each_recomp_percent}, {each_recomp_len}')
                #Model Prediction 
                cur_energy, cur_latency = strategy_prediction(model, num_of_prompts, prompt_len, gen_len, hardware_config, each_recomp_len, each_feasible_offloading)
                
                cur_objective_val = cur_latency
                if var_to_min == "energy":
                    cur_objective_val = cur_energy
                # compare to optimal policy seen so far
                if cur_objective_val < min_objective_val:
                    min_objective_val = cur_objective_val
                    min_strategy = (each_batch_size, each_feasible_offloading, each_recomp_len, cur_energy, cur_latency)

    # return best policy and optimal latency, energy, etc.
    return min_objective_val, min_strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-175b")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--gpu-mem", type=int, default=15)
    parser.add_argument("--cpu-mem", type=int, default=200)
    parser.add_argument("--nvme-mem", type=int, default=1500)

    parser.add_argument("--np", "--num-prompts", type=int)


    args = parser.parse_args()
    config = CostModelConfig()

    opt_config = get_opt_config(args.model)
    config.l = opt_config.num_hidden_layers
    config.h1 = opt_config.hidden_size
    config.h2 = opt_config.ffn_embed_dim
    config.nh = opt_config.n_head

    config.s = args.prompt_len
    config.n = args.gen_len

    config.gmem = args.gpu_mem * GB
    config.cmem = args.cpu_mem * GB
    config.nmem = args.nvme_mem * GB

    #TODO: specify hardware config
    disect_input(args.model, opt_config, args.np, args.prompt_len, args.gen_len, config)

