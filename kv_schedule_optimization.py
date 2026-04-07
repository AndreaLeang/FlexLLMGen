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
import os
import sys
import csv

from flexllmgen.compression import CompressionConfig, Policy
from flexllmgen.opt_config import get_opt_config
# from flexllmgen.flex_opt import Policy
from flexllmgen.utils import GB, T

sys.path.append( '/home/akleang/akleang/energaizer-ispass26-artifact/') # to be able to find energaizer-ispass26-artifact
from gee.gee_utils import get_gee

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



def get_available_offloadings(opt_config, hardware_config, batch_sizes, seq_len):
    total_available_gpu = hardware_config.gmem # Bytes
    total_weight_bytes = opt_config.model_bytes() # Bytes
    num_heads = opt_config.n_head

    batch_size_to_distinct_offloadings = {
        1:[0, 100],
        2:[0, 50, 100],
        4:[0, 20, 50, 70, 100],
        8:[0, 10, 20, 30, 50, 60, 70, 80, 100],
        16:[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }

    feasible_strategies = {}
    for each_batch_size in batch_sizes:
        total_kv_cache_bytes = opt_config.cache_bytes(each_batch_size, seq_len) # Bytes (Toal KV Cache per forward pass) (seq_len = prompt_len+gen_len)
        total_hidden_bytes = opt_config.hidden_bytes(each_batch_size, seq_len) # Bytes (Total Hidden State per forward pass) (seq_len = prompt_len+gen_len)
      
        for each_possible_offloading in batch_size_to_distinct_offloadings[each_batch_size]:
            num_prompts_on_gpu = int(each_batch_size* num_heads * (100-each_possible_offloading) / 100) // num_heads
            actual_kv_cache_bytes = (num_prompts_on_gpu / each_batch_size) * total_kv_cache_bytes

            if total_weight_bytes + actual_kv_cache_bytes + total_hidden_bytes <= total_available_gpu:
                if each_batch_size not in feasible_strategies:
                    feasible_strategies[each_batch_size] = []
                feasible_strategies[each_batch_size].append(each_possible_offloading)
    return feasible_strategies
    

def get_batch_sizes(num_of_prompts):
    possible_batch_sizes = []
    cur_num_batches = num_of_prompts
    # test if %2 and repeat until none
    while (cur_num_batches % 2 == 0 and cur_num_batches > 0) or cur_num_batches == 1:
        possible_batch_sizes.append(num_of_prompts // cur_num_batches)
        cur_num_batches //= 2
    return possible_batch_sizes


def strategy_prediction(model, num_of_prompts, prompt_len, gen_len, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator):
    #offloading percent is amount offloaded to the cpu
    
    tot_energy = 0
    tot_latency = 0
    num_hidden_layers = model.num_hidden_layers

    # Forward Pass Prediction
    #is_load_store: 0: none, 1: load only, 2: store only, 3: load and store
    input_energy, input_latency = layer_prediction(model, 1, batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len, hardware_config, gpu_estimator, "input")
    no_load_input_energy, no_load_input_latency = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len, hardware_config, gpu_estimator, "input")
    input_energy += (num_batches-1)*no_load_input_energy
    input_latency += (num_batches-1)*no_load_input_latency
  
    output_energy, output_latency =  layer_prediction(model, 1, batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len, hardware_config, gpu_estimator, "output")
    output_energy *= (num_batches)
    output_latency *= (num_batches)
    no_store_output_energy, no_store_output_latency = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, gen_len, hardware_config, gpu_estimator, "output")
    output_energy += no_store_output_energy
    output_latency += no_store_output_latency
  
    for cur_gen_len in range(1, gen_len+1):
        if num_batches == 1:
            tot_MHA_energy, tot_MHA_latency = num_hidden_layers*(layer_prediction(model, 1, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MHA"))
            tot_MLP_energy, tot_MLP_latency = num_hidden_layers*(layer_prediction(model, 2, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP"))
        else:
            tot_MHA_energy, tot_MHA_latency = layer_prediction(model, 1, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MHA") 
            single_store_MHA_energy, single_store_MHA_latency = layer_prediction(model, 2, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MHA") 
            bi_dir_MHA_energy, bi_dir_MHA_latency = layer_prediction(model, 3, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MHA")
            
            tot_MHA_energy += single_store_MHA_energy + (num_batches-2)*bi_dir_MHA_energy
            tot_MHA_latency += single_store_MHA_latency + (num_batches-2)*bi_dir_MHA_latency
            tot_MHA_energy *= num_hidden_layers
            tot_MHA_latency *= num_hidden_layers
          
            tot_MLP_energy, tot_MLP_latency = layer_prediction(model, 2, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP") 
            single_load_tot_MLP_energy, single_load_MLP_latency = layer_prediction(model, 1, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP") 
            bi_dir_MLP_energy, bi_dir_MLP_latency = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP")
            tot_MLP_energy += single_load_tot_MLP_energy + (num_batches-2)*bi_dir_MLP_energy
            tot_MLP_latency += single_load_MLP_latency + (num_batches-2)*bi_dir_MLP_latency
            tot_MLP_energy *= num_hidden_layers
            tot_MLP_latency *= num_hidden_layers
            tot_MLP_energy *= (num_hidden_layers-1)
            tot_MLP_latency *= (num_hidden_layers-1)

            # Last MLP layer: pattern changes 
            bi_dir_MLP_energy, bi_dir_MLP_latency = layer_prediction(model, 2, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP") 
            single_dir_MLP_energy, single_dir_MLP_latency = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP")
            tot_MLP_energy += bi_dir_MLP_energy + (num_batches-1)*single_dir_MLP_energy
            tot_MLP_latency += bi_dir_MLP_latency + (num_batches-1)*single_dir_MLP_latency
      
        middle_layer_latency = tot_MHA_latency + tot_MLP_latency
        middle_layer_energy = tot_MHA_energy + tot_MLP_energy

        one_forward_latency = input_latency + middle_layer_latency + output_latency
        one_forward_energy = input_energy + middle_layer_energy + output_energy
        tot_latency += one_forward_latency
        tot_energy  += one_forward_energy

    # get total energy and latency
    return tot_energy, tot_latency

def get_bytes_to_load(model, batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len):
    recomp_load_bytes = recomp_len * 8192 * batch_size # 8192 bytes/token
    kv_load_bytes = (prompt_len + gen_len-recomp_len) * 8192 * (batch_size-((batch_size*(100-offload_percent))//100))
    return recomp_load_bytes, kv_load_bytes

def get_bytes_to_store(batch_size):
    kv_store_bytes = batch_size * 8192 # 1 token per batch
    return kv_store_bytes

def layer_prediction(opt_config, is_load_store, batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len, hardware_config, gpu_estimator, layer_type="MHA"):
    #layer type determines the actual recomputation time + compute layer time
    layer_calc_time, layer_calc_energy = layer_calc_pred(opt_config, batch_size, hardware_config, gpu_estimator, layer_type)
  
    if is_load_store == 0:
        #no load or store, just the layer computations
        return layer_calc_energy, layer_calc_time
    elif is_load_store == 2:
        # store only --> single directional
        transfer_energy, transfer_lat = transfer_pred(get_bytes_to_store(batch_size), hardware_config)
        return layer_calc_energy+transfer_energy, max(layer_calc_time, transfer_lat)
    else:
        recomp_bytes, kv_load_bytes = get_bytes_to_load(opt_config, batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len)
        #use is_load_store==1 as single directional
        #recomp transfer & first kv load are always single directional. the second kv load uses single_directional
        pinned_energy, pinned_latency = pinned_pred(kv_load_bytes, hardware_config)
        # first_half_latency = max(pinned_latency, recomp_prep_pred(prompt_len, recomp_len, hardware_config)+transfer_pred(recomp_bytes, hardware_config))
        transfer_energy, transfer_lat = transfer_pred(get_bytes_to_store(batch_size), hardware_config)
        first_half_latency = max(pinned_latency, transfer_lat)
        recomp_energy, recomp_latency = (0,0)
        if layer_type == "MHA":
            recomp_energy, recomp_latency = recomp_calc_pred(opt_config, batch_size, prompt_len, gen_len, recomp_len, gpu_estimator, hardware_config)
        second_single_dir = is_load_store == 1
        k_transfer_energy, k_transfer_latency = transfer_pred(kv_load_bytes, hardware_config)
        v_transfer_energy, v_transfer_latency = transfer_pred(kv_load_bytes, hardware_config, single_directional = second_single_dir)
        second_half_latency = max(pinned_latency + recomp_latency + layer_calc_time, k_transfer_latency + v_transfer_latency)
        tot_energy = pinned_energy + transfer_energy + recomp_energy + k_transfer_energy + v_transfer_energy
      
        return tot_energy, first_half_latency + second_half_latency


def pinned_pred(bytes, hardware_config):
    #TODO: energy
    return 0, 5.168e-14 * bytes**2 + 3.317e-05 * bytes - 104.7


# def recomp_prep_pred(prompt_len, recomp_len, hardware_config):
#     #TODO: collect data
#     # removed for now
#     return 0

def transfer_pred(bytes, hardware_config, single_directional=True):
    #TODO: energy
    if bytes == 0:
        return 0,0
    if single_directional:
        return 0, 1.415 * math.log10(bytes) + 13.38
    
    return 0, 3.626 * math.log10(bytes) + 26.13 

def recomp_calc_pred(opt_config, batch_size, prompt_len, cur_gen_len, recomp_len, gpu_estimator, hardware_config):
    # estimator: op
    # FlashAttention: ['flashattention_v2']
    # elementWise   : ['pointwise_mul', 'pointwise_add', 'scalar_mul', 'scalar_add', 'typecast_to_fp32', 'typecast_to_bf16', 'relu', 'gelu', 'silu', 'tanh', 'sigmoid', 'unspecified_activation', 'unspecified_tensor', 'unspecified_scalar']
    # gemmLike      : ['gemm', 'fmha-approximate']
    # NonLinear     : ['softmax', 'layernorm', 'softmax_fusion', 'layernorm_fusion']
    # energaizer-ispass26-artifact/experiments_single/gee_estimator.py --> query specs 
    all_queries = []
    all_query_types = []
  
    layer_norm_query = {'batch': batch_size,
                         'dim': recomp_len, 
                         'prec': 'bf16'}
    layer_norm_query_type = ('layernorm')
    all_queries.append(layer_norm_query)
    all_query_types.append(layer_norm_query_type)
  
    linear_query = {
    	'batch': batch_size,
    	'dimM' : recomp_len,
    	'dimN' : opt_config.hidden_size,
    	'dimK' : opt_config.hidden_size,
    	'precM': 'bf16',
    	'precA': 'bf16',
    	'useTensorCore':  True
    }
    linear_query_type = ('gemm', 'tc', 'bf16')
    for i in range(2):
        all_queries.append(linear_query)
        all_query_types.append(linear_query_type)
    
    reshape_query = {
        'dim ': batch_size*opt_config.n_head,
        'op': 'unspecified_tensor',
        'prec': 'bf16',
    }
    reshape_query_type = ('typecast_to_bf16')
    for i in range(4):
        all_queries.append(reshape_query)
        all_query_types.append(reshape_query_type)
  
    copy_1_query = {
        'dim ': recomp_len,
        'op': 'unspecified_tensor',
        'prec': 'bf16',
    }
    copy_1_query_type = ('typecast_to_bf16')
    for i in range(2):
        all_queries.append(copy_1_query)
        all_query_types.append(copy_1_query_type)
    
    copy_2_query = {
        'dim ': prompt_len + cur_gen_len - recomp_len,
        'op': 'unspecified_tensor',
        'prec': 'bf16',
    }
    copy_2_query_type = ('typecast_to_bf16')
    for i in range(2):
        all_queries.append(copy_2_query)
        all_query_types.append(copy_2_query_type)

    #TODO: verify target frequency
    target_freq = 201
    tot_lat = 0
    tot_energy = 0
    print(gpu_estimator)
    for each_ind in range(len(all_queries)):
        latency, _, energy = gpu_estimator.lookup(all_queries[each_ind], all_query_types[each_ind], target_freq=target_freq, lookup_target='all')
        tot_lat += latency
        tot_energy += energy
    
    return tot_energy, tot_lat

def layer_calc_pred(opt_config, batch_size, hardware_config, gpu_estimator, layer_type="MHA"):\
    #input: 
    #output: 
    #MLP: 
  
    #MHA: 
    # W_k, w_q, w_v, w_out = [opt_config.hidden_size, opt_config.hidden_size]
    # 1x F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
    # inputs.data: [batch_size, 1, opt_config.hidden_size]
    # 3x F.linear(hidden, w_q.data, bias=b_q.data)
    # xAT+b → [batch_size, 1, opt_config.hidden_size]*[4096, 4096]T + [opt_config.hidden_size]
    # 3x REshape: (batch_size, n_head, 1, head_dim) → (b*n_head, 1, head_dim) 
    
    # n_head*head_dim = opt_config.hidden_size
    # Cur_seq_len = prompt_len + cur_gen_len
    # _attention_value:
    # torch.bmm(q, k): [batch_size * num_head, 1, head_dim] * [batch_size * num_head, head_dim, cur_seq_len]
    # torch.where() → elementwise [batch_size, 1, 1, cur_seq_len] skim over [batch_size, num_head, 1, cur_seq_len]
    # (no gpu op) view atten_weights → [batch_size*num_head, 1, cur_seq_len]
    # Torch.softmax on dim=2 for  [batch_size*num_head, 1, cur_seq_len]
    
    
    # torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)
    # [batch_size*n_head, 1, cur_seq_len] * [batch_size * n_head, cur_seq_len, head_dim]
    
    # Back to MHA: 
    # (b, 1, h)
    # (no gpu op) Value (rtn from MHA) = value.transpose(1, 2).view(b, tgt_s, h)
    # (b, n_head, 1, head_dim) → (b, 1,  n_head, head_dim) → (b, 1, opt_config.hidden_size)
    # value = F.linear(value, w_out.data, bias=b_out.data)
    # xAT+b → [b, 1, opt_config.hidden_size]*[opt_config.hidden_size, opt_config.hidden_size]T + [opt_config.hidden_size]
    # Add: value.add_(inputs.data)
    # [batch_size, 1, opt_config.hidden_size] + [batch_size, 1, opt_config.hidden_size]

    
    
  
    #TODO
    return 0, 0


def disect_input(model, opt_config, num_of_prompts, prompt_len, gen_len, hardware_config, save_results, gpu_estimator, var_to_min="latency"):
    # break model into layers
  
    ### UNDERSTAND WHAT STRATEGIES ARE AVVAILABLE
                                                                                                                                                                                       
    # understand what unique batch size is available 
    batch_sizes = get_batch_sizes(num_of_prompts)
    # understand what % offloadings are available 
    all_feasible_strategies_dict = get_available_offloadings(opt_config, hardware_config, batch_sizes, prompt_len+gen_len)
    
    ### ITERATE AND COMPARE STRATEGIES

    #iterate through 0-100% recomputing, % offloading, and batch size
    min_objective_val = float('inf')
    min_strategy = None

    if save_results: 
        all_results = {}

    for each_batch_size in batch_sizes:
        for each_feasible_offloading in all_feasible_strategies_dict[each_batch_size]:
            for each_recomp_percent in range(0, 100, 10):
                each_recomp_len = prompt_len * each_recomp_percent // 100 # recomp is only for prompt len
                print(f'cur strat: {each_batch_size}, {each_feasible_offloading}, {each_recomp_len}')
                #Model Prediction 
                cur_energy, cur_latency = strategy_prediction(opt_config, num_of_prompts, prompt_len, gen_len, hardware_config, each_recomp_len, each_feasible_offloading, each_batch_size, num_of_prompts // each_batch_size, gpu_estimator)
                if save_results: 
                    cur_strat = (each_batch_size, each_feasible_offloading, each_recomp_len)
                    all_results[cur_strat] = (cur_energy, cur_latency)
              
                cur_objective_val = cur_latency
                if var_to_min == "energy":
                    cur_objective_val = cur_energy
                # compare to optimal policy seen so far
                if cur_objective_val < min_objective_val:
                    min_objective_val = cur_objective_val
                    min_strategy = (each_batch_size, each_feasible_offloading, each_recomp_len, cur_energy, cur_latency)

    if save_results:
        csv_filename = "all_pred_totP_" + str(num_of_prompts) +"prompt_len_" + str(prompt_len) + "gen_len" + str(gen_len) + ".csv"
        print(csv_filename)
        fieldnames = ["Batch Size", "Offloading Percent to CPU", "Recompute Length", "Energy (J)", "Latency (s)"]
        write_header = not os.path.exists(csv_filename)
      
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for each_strat in all_results:
              writer.writerow({'Batch Size': each_strat[0], 
                      'Offloading Percent to CPU': each_strat[1],
                      'Recompute Length': each_strat[2], 
                      'Energy (J)': all_results[each_strat][0], 
                      'Latency (s)': all_results[each_strat][1], 
                      })

    # return best policy and optimal latency, energy, etc.
    print(f'best policy: batch_size = {min_strategy[0]}, offloading_percent = {min_strategy[1]}, recomp_len = {min_strategy[2]}, energy = {min_strategy[3]}, latency = {min_strategy[4]}')
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
    parser.add_argument("--save", "--save-all-strats", action="store_true")


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
    gpu_estimator = get_gee(gpu_yaml_path="/home/akleang/akleang/energaizer-ispass26-artifact/config/gpu/yz8.yaml", 
                        lut_yaml_path="/home/akleang/akleang/energaizer-ispass26-artifact/experiments_endtoend/exp_config/a100_dvfs_lut_config.yaml", 
                        dvfs_aware=True, dvfs_inference_mode='all', 
                        dvfs_supply_voltage_json="/home/akleang/akleang/energaizer-ispass26-artifact/config/dvfs/yz8/supply_voltage.json",
                        dvfs_idle_power_json="/home/akleang/akleang/energaizer-ispass26-artifact/config/dvfs/yz8/idle_power.json", 
                        lut_folder_abs_path="/home/akleang/akleang/energaizer-ispass26-artifact/database/data")
    print("got gpu_est")
    disect_input(args.model, opt_config, args.np, args.prompt_len, args.gen_len, config, gpu_estimator, args.save)

