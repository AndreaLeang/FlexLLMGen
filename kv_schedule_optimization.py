"""

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


@dataclasses.dataclass
class CostModelConfig:
    s: int = 512
    n: int = 32

    l: int = 96
    h1: int = 12288
    h2: int = 12288 * 4
    nh: int = 96

    gmem: int = 40 * GB
    cmem: int = 200 * GB

    gpu_freq: int = 1305

    

def get_available_offloadings(opt_config, hardware_config, batch_sizes, num_of_prompts, seq_len, min_offloading=True):
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
        num_batches = num_of_prompts // each_batch_size

        for each_possible_offloading in batch_size_to_distinct_offloadings[each_batch_size]:
            num_prompts_on_gpu = int(each_batch_size* num_heads * (100-each_possible_offloading) / 100) // num_heads
            actual_kv_cache_bytes = (num_prompts_on_gpu / each_batch_size) * num_batches * total_kv_cache_bytes
            if total_weight_bytes + actual_kv_cache_bytes + total_hidden_bytes <= total_available_gpu:
                if each_batch_size not in feasible_strategies:
                    feasible_strategies[each_batch_size] = []
                feasible_strategies[each_batch_size].append(each_possible_offloading)
                if min_offloading: 
                    break
    return feasible_strategies
    

def get_batch_sizes(num_of_prompts):
    possible_batch_sizes = []
    cur_num_batches = num_of_prompts
    # test if %2 and repeat until none
    while (cur_num_batches % 2 == 0 and cur_num_batches > 0) or cur_num_batches == 1:
        possible_batch_sizes.append(num_of_prompts // cur_num_batches)
        cur_num_batches //= 2
    return possible_batch_sizes

def fast_strat_prediction(model, num_of_prompts, prompt_len, gen_len, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator):
    tot_energy = 0
    tot_latency = 0
    time_to_first_token = 0
    avg_energy_per_layer = {"input": (0.0, 0), "output": (0.0, 0), "MHA": (0.0, 0), "MLP": (0.0, 0)} # (tot, num of occurances)
    avg_latency_per_layer = {"input": (0.0, 0), "output": (0.0, 0), "MHA": (0.0, 0), "MLP": (0.0, 0)}
    
    num_hidden_layers = model.num_hidden_layers
    
    # First Token Prediction
    fir_input_energy, fir_input_latency, fir_output_energy, fir_output_latency, fir_tot_MHA_energy, fir_tot_MHA_latency, fir_tot_MLP_energy, fir_tot_MLP_latency, fir_tot_transfer_energy, fir_tot_active_energy, fir_tot_transfer_latency = first_token_forward_pass(model, num_of_prompts, prompt_len, 0, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator, num_hidden_layers)
    
    first_token_energy = fir_input_energy + fir_tot_MHA_energy + fir_tot_MLP_energy + fir_output_energy
    first_token_latency = fir_input_latency + fir_tot_MHA_latency + fir_tot_MLP_latency + fir_output_latency

    

    # rest of Tokens Prediction --> simplified to (gen_len - 1)*forward_pass_latency
    if num_batches == 1: 
        input_energy, input_latency, output_energy, output_latency, tot_MHA_energy, tot_MHA_latency, tot_MLP_energy, tot_MLP_latency, cur_transfer_energy, cur_active_energy, cur_transfer_latency = single_batch_forward_pass(model, num_of_prompts, prompt_len, gen_len-1, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator, num_hidden_layers)
    else:
        input_energy, input_latency, output_energy, output_latency, tot_MHA_energy, tot_MHA_latency, tot_MLP_energy, tot_MLP_latency, cur_transfer_energy, cur_active_energy, cur_transfer_latency  = multi_batch_forward_pass(model, num_of_prompts, prompt_len, gen_len-1, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator, num_hidden_layers)
    other_token_energy = (gen_len - 1) * (input_energy + tot_MHA_energy + tot_MLP_energy + output_energy)
    other_token_latency = (gen_len - 1) * (input_latency + tot_MHA_latency + tot_MLP_latency + output_latency)
    other_token_transfer_energy = (gen_len - 1) * cur_transfer_energy
    other_token_active_energy = (gen_len - 1) * cur_active_energy
    other_token_transfer_latency = (gen_len - 1) * cur_transfer_latency

    tot_energy = first_token_energy + other_token_energy
    tot_latency = first_token_latency + other_token_latency
    time_to_first_token = first_token_latency
    tot_transfer_energy = fir_tot_transfer_energy + other_token_transfer_energy
    tot_active_energy = fir_tot_active_energy + other_token_active_energy
    tot_transfer_latency = fir_tot_transfer_latency + other_token_transfer_latency
  
    avg_energy_per_layer["input"] = (fir_input_energy + (gen_len - 1) *input_energy, gen_len*num_batches)
    avg_energy_per_layer["output"] = (fir_output_energy + (gen_len - 1) *output_energy, gen_len*num_batches)
    avg_energy_per_layer["MHA"] = (fir_tot_MHA_energy+(gen_len - 1) *tot_MHA_energy, gen_len*num_batches*num_hidden_layers)
    avg_energy_per_layer["MLP"] = (fir_tot_MLP_energy+(gen_len - 1) *tot_MLP_energy, gen_len*num_batches*num_hidden_layers)

    avg_latency_per_layer["input"] = (fir_input_latency+(gen_len - 1) *input_latency,gen_len*num_batches)
    avg_latency_per_layer["output"] = (fir_output_latency+(gen_len - 1) *output_latency, gen_len*num_batches)
    avg_latency_per_layer["MHA"] = (fir_tot_MHA_latency + (gen_len - 1) *tot_MHA_latency, gen_len*num_batches*num_hidden_layers)
    avg_latency_per_layer["MLP"] = (fir_tot_MLP_latency+(gen_len - 1) *tot_MLP_latency, gen_len*num_batches*num_hidden_layers)

    percent_energy_offloading = tot_transfer_energy / tot_energy * 100
    percent_energy_active = tot_active_energy / tot_energy * 100
    percent_latency_transfer = tot_transfer_latency / tot_latency * 100
  
    # get total energy and latency and time_to_first_token
    return tot_energy, tot_latency, time_to_first_token, avg_energy_per_layer, avg_latency_per_layer, percent_energy_offloading, percent_energy_active, percent_latency_transfer


def strategy_prediction(model, num_of_prompts, prompt_len, gen_len, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator):
    #offloading percent is amount offloaded to the cpu

    tot_energy = 0
    tot_latency = 0
    time_to_first_token = 0
    tot_transfer_energy = 0.0
    tot_active_energy = 0.0
    tot_transfer_latency = 0.0
    avg_energy_per_layer = {"input": (0.0, 0), "output": (0.0, 0), "MHA": (0.0, 0), "MLP": (0.0, 0)} # (tot, num of occurances)
    avg_latency_per_layer = {"input": (0.0, 0), "output": (0.0, 0), "MHA": (0.0, 0), "MLP": (0.0, 0)}
    
    num_hidden_layers = model.num_hidden_layers

    # Forward Pass Prediction
    # layer prediction: is_load_store: 0: none, 1: load only, 2: store only, 3: load and store
  
    for cur_gen_len in range(gen_len):
        # input, output, MHA and MLP
        if cur_gen_len == 0:
            input_energy, input_latency, output_energy, output_latency, tot_MHA_energy, tot_MHA_latency, tot_MLP_energy, tot_MLP_latency, cur_transfer_energy, cur_active_energy, cur_transfer_latency = first_token_forward_pass(model, num_of_prompts, prompt_len, cur_gen_len, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator, num_hidden_layers)
        elif num_batches == 1:
            input_energy, input_latency, output_energy, output_latency, tot_MHA_energy, tot_MHA_latency, tot_MLP_energy, tot_MLP_latency, cur_transfer_energy, cur_active_energy, cur_transfer_latency = single_batch_forward_pass(model, num_of_prompts, prompt_len, cur_gen_len, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator, num_hidden_layers, last_token=cur_gen_len==1)
        else:
            input_energy, input_latency, output_energy, output_latency, tot_MHA_energy, tot_MHA_latency, tot_MLP_energy, tot_MLP_latency, cur_transfer_energy, cur_active_energy, cur_transfer_latency  = multi_batch_forward_pass(model, num_of_prompts, prompt_len, cur_gen_len, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator, num_hidden_layers, last_token=cur_gen_len==1)
      
        middle_layer_latency = tot_MHA_latency + tot_MLP_latency
        middle_layer_energy = tot_MHA_energy + tot_MLP_energy
        avg_energy_per_layer["input"] = (avg_energy_per_layer["input"][0]+input_energy, avg_energy_per_layer["input"][1] + num_batches)
        avg_energy_per_layer["output"] = (avg_energy_per_layer["output"][0]+output_energy, avg_energy_per_layer["output"][1] + num_batches)
        avg_energy_per_layer["MHA"] = (avg_energy_per_layer["MHA"][0]+tot_MHA_energy, avg_energy_per_layer["MHA"][1] + num_batches*num_hidden_layers)
        avg_energy_per_layer["MLP"] = (avg_energy_per_layer["MLP"][0]+tot_MLP_energy, avg_energy_per_layer["MLP"][1] + num_batches*num_hidden_layers)

        avg_latency_per_layer["input"] = (avg_latency_per_layer["input"][0]+input_latency, avg_latency_per_layer["input"][1] + num_batches)
        avg_latency_per_layer["output"] = (avg_latency_per_layer["output"][0]+output_latency, avg_latency_per_layer["output"][1] + num_batches)
        avg_latency_per_layer["MHA"] = (avg_latency_per_layer["MHA"][0]+tot_MHA_latency, avg_latency_per_layer["MHA"][1] + num_batches*num_hidden_layers)
        avg_latency_per_layer["MLP"] = (avg_latency_per_layer["MLP"][0]+tot_MLP_latency, avg_latency_per_layer["MLP"][1] + num_batches*num_hidden_layers)

        # print(f"layer avg input latency: {input_latency/ num_batches}")
        # print(f"layer avg MHA latency: {tot_MHA_latency/ (num_batches*num_hidden_layers)}")
        # print(f"layer avg MLP latency: {tot_MLP_latency/ (num_batches*num_hidden_layers)}")
        # print(f"layer avg output latency: {output_latency / num_batches}")
      
        # print(f"each total input latency: {input_latency}")
        # print(f"total forward pass MHA latency: {tot_MHA_latency}")
        # print(f"total forward pass MLP latency: {tot_MLP_latency}")
        # print(f"each total output latency: {output_latency}")
        # print(f"total forward pass middle layers latency: {middle_layer_latency}")
      
        one_forward_latency = input_latency + middle_layer_latency + output_latency
        one_forward_energy = input_energy + middle_layer_energy + output_energy
        tot_latency += one_forward_latency
        tot_energy  += one_forward_energy
        tot_transfer_energy += cur_transfer_energy
        tot_active_energy += cur_active_energy
        tot_transfer_latency += cur_transfer_latency
      
        # print(f"cur pass transfer energy: {cur_transfer_energy}, tot_transfer_energy: {tot_transfer_energy}")
        # print(f"cur pass active energy: {tot_active_energy}, tot_transfer_energy: {cur_active_energy}")
        # print(f"cur pass tot energy: {one_forward_energy}, tot seen energy: {tot_energy}")

        if cur_gen_len == 0:
            time_to_first_token = one_forward_latency
        print(f"total energy  of this forward pass: {one_forward_energy}")
        print(f"total energy seen so far: {tot_energy}")

    # get total energy and latency and time_to_first_token
    percent_energy_offloading = tot_transfer_energy / tot_energy *100
    percent_energy_active = tot_active_energy / tot_energy * 100
    percent_latency_transfer = tot_transfer_latency / tot_latency * 100
    return tot_energy, tot_latency, time_to_first_token, avg_energy_per_layer, avg_latency_per_layer, percent_energy_offloading, percent_energy_active, percent_latency_transfer


def first_token_forward_pass(model, num_of_prompts, prompt_len, cur_gen_len, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator, num_hidden_layers):
    
    input_energy, input_latency, input_transfer_energy, input_active_energy, input_transfer_latency = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "input", True)
    output_energy, output_latency, output_transfer_energy, output_active_energy, output_transfer_latency  = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "output", True)
  
    # MHA: compute, MLP: load + store unless last layer. then just store
    tot_MHA_energy, tot_MHA_latency, MHA_transfer_energy, MHA_active_energy, MHA_transfer_latency  =layer_prediction(model, 2, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MHA", True)
    # print(f"Single MHA latency: {tot_MHA_latency}")
    tot_MHA_energy *= num_hidden_layers
    tot_MHA_latency *= num_hidden_layers
    MHA_transfer_energy *= num_hidden_layers
    MHA_active_energy *= num_hidden_layers

    tot_MLP_energy, tot_MLP_latency, MLP_transfer_energy, MLP_active_energy, MLP_transfer_latency = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP", True)
    # print(f"Single MLP latency: {tot_MLP_latency}")
    tot_MLP_energy *= num_hidden_layers
    tot_MLP_latency *= num_hidden_layers
    MLP_transfer_energy *= num_hidden_layers
    MLP_active_energy *= num_hidden_layers


    total_transfer_energy = num_batches*(input_transfer_energy + MHA_transfer_energy + MLP_transfer_energy + output_transfer_energy)
    total_active_energy = num_batches*(input_active_energy + MHA_active_energy + MLP_active_energy + output_active_energy)
    total_transfer_latency = num_batches*(input_transfer_latency + MHA_transfer_latency + MLP_transfer_latency + output_transfer_latency) 
  
    return num_batches*input_energy, num_batches*input_latency, num_batches*output_energy, num_batches*output_latency, num_batches*tot_MHA_energy, num_batches*tot_MHA_latency, num_batches*tot_MLP_energy, num_batches*tot_MLP_latency, total_transfer_energy, total_active_energy, total_transfer_latency


def single_batch_forward_pass(model, num_of_prompts, prompt_len, cur_gen_len, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator, num_hidden_layers, last_token=False):
    # input: nothing*(num_batches -1) + load
    input_energy, input_latency, input_transfer_energy, input_active_energy, input_transfer_latency = layer_prediction(model, 1, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "input")

    #output
    if last_token:
        output_energy, output_latency, output_transfer_energy, output_active_energy, output_transfer_latency  = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "output")
    else: 
        output_energy, output_latency, output_transfer_energy, output_active_energy, output_transfer_latency  = layer_prediction(model, 2, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "output")
  
    # MHA: compute, MLP: load + store unless last layer. then just store
    tot_MHA_energy, tot_MHA_latency, MHA_transfer_energy, MHA_active_energy, MHA_transfer_latency  =layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MHA")
    tot_MHA_energy *= num_hidden_layers
    tot_MHA_latency *= num_hidden_layers
    MHA_transfer_energy *= num_hidden_layers
    MHA_active_energy *= num_hidden_layers
    MHA_transfer_latency *= num_hidden_layers
  
    will_load = 3
    if last_token:
        will_load = 1 # no storing for last pass
    tot_MLP_energy, tot_MLP_latency, MLP_transfer_energy, MLP_active_energy, MLP_transfer_latency = layer_prediction(model, will_load, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP")
    tot_MLP_energy *= (num_hidden_layers-1)
    tot_MLP_latency *= (num_hidden_layers-1)
    MLP_transfer_energy *= (num_hidden_layers-1)
    MLP_active_energy *= (num_hidden_layers-1)
    MLP_transfer_latency *= (num_hidden_layers-1)
    
    last_MLP_energy, last_MLP_latency, last_MLP_transfer_energy, last_MLP_active_energy, last_MLP_transfer_latency = layer_prediction(model, 2, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP")
    tot_MLP_energy += last_MLP_energy
    tot_MLP_latency += last_MLP_latency
    MLP_transfer_energy += last_MLP_transfer_energy
    MLP_active_energy += last_MLP_active_energy
    MLP_transfer_latency += last_MLP_transfer_latency
  
    total_transfer_energy = input_transfer_energy + MHA_transfer_energy + MLP_transfer_energy + output_transfer_energy
    total_active_energy = input_active_energy + MHA_active_energy + MLP_active_energy + output_active_energy
    total_transfer_latency = input_transfer_latency + MHA_transfer_latency + MLP_transfer_latency + output_transfer_latency 
  
    return input_energy, input_latency, output_energy, output_latency, tot_MHA_energy, tot_MHA_latency, tot_MLP_energy, tot_MLP_latency, total_transfer_energy, total_active_energy, total_transfer_latency

def multi_batch_forward_pass(model, num_of_prompts, prompt_len, cur_gen_len, hardware_config, recomp_len, offload_percent, batch_size, num_batches, gpu_estimator, num_hidden_layers, last_token=False):
    # input: nothing*(num_batches -1) + load
    load_input_energy, load_input_latency, load_input_transfer_energy, load_input_active_energy, load_input_transfer_latency = layer_prediction(model, 1, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "input")
    no_load_input_energy, no_load_input_latency, no_load_input_transfer_energy, no_load_input_active_energy, no_load_input_transfer_latency = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "input")
    # print(f"load input: {load_input_latency}")
    # print(f"no load input: {no_load_input_latency}")
    input_energy = (num_batches-1)*no_load_input_energy + load_input_energy 
    input_latency = (num_batches-1)*no_load_input_latency + load_input_latency 
    input_transfer_energy = (num_batches-1)*load_input_transfer_energy + no_load_input_transfer_energy 
    input_active_energy = (num_batches-1)*load_input_active_energy + no_load_input_active_energy 
    input_transfer_latency = (num_batches-1)*load_input_transfer_latency + no_load_input_transfer_latency

    # output: store + nothing*(num_batches -1) 
    no_store_output_energy, no_store_output_latency, no_store_output_transfer_energy, no_store_output_active_energy, no_store_output_transfer_latency = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "output")
    default_output_energy = (num_batches-1)*no_store_output_energy
    default_output_latency = (num_batches-1)*no_store_output_latency
    default_output_transfer_energy = (num_batches-1)*no_store_output_transfer_energy
    default_output_active_energy = (num_batches-1)*no_store_output_active_energy
    default_output_transfer_latency = (num_batches-1)*no_store_output_transfer_latency
  
    if last_token:
        output_energy = default_output_energy + no_store_output_energy
        output_latency = default_output_latency + no_store_output_latency
        output_transfer_energy = default_output_transfer_energy + no_store_output_transfer_energy
        output_active_energy = default_output_active_energy + no_store_output_active_energy
        output_transfer_latency = default_output_transfer_latency + no_store_output_transfer_latency
    else: 
        store_output_energy, store_output_latency, store_output_transfer_energy, store_output_active_energy, store_output_transfer_latency =  layer_prediction(model, 2, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "output")
        output_energy = default_output_energy + store_output_energy
        output_latency = default_output_latency + store_output_latency
        output_transfer_energy = default_output_transfer_energy + store_output_transfer_energy
        output_active_energy = default_output_active_energy + store_output_active_energy
        output_transfer_latency = default_output_transfer_latency + store_output_transfer_latency
  
    # MHA: load, load_store*(num_batches-2), store   MLP: store, nothing*(num_batches-2), load
    will_store = 2
    bi_load = 3
    if last_token:
        will_store = 0 # no storing for last forward pass
        bi_load = 1
    single_load_MHA_energy, single_load_MHA_latency, single_load_MHA_transfer_energy, single_load_MHA_active_energy, single_load_MHA_transfer_latency = layer_prediction(model, 1, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MHA") 
    single_store_MHA_energy, single_store_MHA_latency, single_store_MHA_transfer_energy, single_store_MHA_active_energy, single_store_MHA_transfer_latency = layer_prediction(model, will_store, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MHA") 
    bi_dir_MHA_energy, bi_dir_MHA_latency, bi_dir_MHA_transfer_energy, bi_dir_MHA_active_energy, bi_dir_MHA_transfer_latency = layer_prediction(model, bi_load, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MHA")

    tot_MHA_energy = single_load_MHA_energy + (num_batches-2)*bi_dir_MHA_energy + single_store_MHA_energy
    tot_MHA_latency = single_load_MHA_latency + (num_batches-2)*bi_dir_MHA_latency + single_store_MHA_latency
    MHA_transfer_energy = single_load_MHA_transfer_energy + (num_batches-2)*bi_dir_MHA_transfer_energy + single_store_MHA_transfer_energy
    MHA_active_energy = single_load_MHA_active_energy + (num_batches-2)*bi_dir_MHA_active_energy + single_store_MHA_active_energy
    MHA_transfer_latency = single_load_MHA_transfer_latency + (num_batches-2)*bi_dir_MHA_transfer_latency + single_store_MHA_transfer_latency
    
    tot_MHA_energy *= num_hidden_layers
    tot_MHA_latency *= num_hidden_layers  
    MHA_transfer_energy *= num_hidden_layers  
    MHA_active_energy *= num_hidden_layers  
    MHA_transfer_latency *= num_hidden_layers  
  

    single_store_energy, single_store_MLP_latency, single_store_MLP_transfer_energy, single_store_MLP_active_energy, single_store_MLP_transfer_latency = layer_prediction(model, will_store, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP") 
    single_load_MLP_energy, single_load_MLP_latency, single_load_MLP_transfer_energy, single_load_MLP_active_energy, single_load_MLP_transfer_latency = layer_prediction(model, 1, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP") 
    nothing_MLP_energy, nothing_MLP_latency, nothing_MLP_transfer_energy, nothing_MLP_active_energy, nothing_MLP_transfer_latency = layer_prediction(model, 0, batch_size, num_batches, offload_percent, recomp_len, prompt_len, cur_gen_len, hardware_config, gpu_estimator, "MLP")

    tot_MLP_energy = single_store_energy + (num_batches-2)*nothing_MLP_energy + single_load_MLP_energy
    tot_MLP_latency = single_store_MLP_latency + (num_batches-2)*nothing_MLP_latency + single_load_MLP_latency
    MLP_transfer_energy = single_store_MLP_transfer_energy + (num_batches-2)*nothing_MLP_transfer_energy + single_load_MLP_transfer_energy
    MLP_active_energy = single_store_MLP_active_energy + (num_batches-2)*nothing_MLP_active_energy + single_load_MLP_active_energy
    MLP_transfer_latency = single_store_MLP_transfer_latency + (num_batches-2)*nothing_MLP_transfer_latency + single_load_MLP_transfer_latency
    tot_MLP_energy *= (num_hidden_layers-1)
    tot_MLP_latency *= (num_hidden_layers-1)
    MLP_transfer_energy *= (num_hidden_layers-1)
    MLP_active_energy *= (num_hidden_layers-1)
    MLP_transfer_latency *= (num_hidden_layers-1)

    # Last MLP layer: pattern changes (no load in end)
    tot_MLP_energy += single_store_energy + (num_batches-1)*nothing_MLP_energy
    tot_MLP_latency += single_store_MLP_latency + (num_batches-1)*nothing_MLP_latency
    MLP_transfer_energy += single_store_MLP_transfer_energy + (num_batches-1)*nothing_MLP_transfer_energy
    MLP_active_energy += single_store_MLP_active_energy + (num_batches-1)*nothing_MLP_active_energy
    MLP_transfer_latency += single_store_MLP_transfer_latency + (num_batches-1)*nothing_MLP_transfer_latency

    total_transfer_energy = input_transfer_energy + MHA_transfer_energy + MLP_transfer_energy + output_transfer_energy
    total_active_energy = input_active_energy + MHA_active_energy + MLP_active_energy + output_active_energy
    total_transfer_latency = input_transfer_latency + MHA_transfer_latency + MLP_transfer_latency + output_transfer_latency 

    return input_energy, input_latency, output_energy, output_latency, tot_MHA_energy, tot_MHA_latency, tot_MLP_energy, tot_MLP_latency, total_transfer_energy, total_active_energy, total_transfer_latency




def get_bytes_to_load(model, batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len, first_token = False):
    if first_token: # nothing is loaded
        recomp_load_bytes = 0
        kv_load_bytes = 0
    else:
        recomp_load_bytes = recomp_len * 8192 * batch_size # 8192 bytes/token
        kv_load_bytes = (prompt_len + gen_len-recomp_len) * 8192 * (batch_size-((batch_size*(100-offload_percent))//100)) 
    return recomp_load_bytes, kv_load_bytes

def get_bytes_to_store(batch_size, prompt_len=0, first_token=False):
    if first_token:
        kv_store_bytes = batch_size * prompt_len * 8192 # prompt_len tokens per batch
    else:
        kv_store_bytes = batch_size * 8192 # 1 token per batch
    return kv_store_bytes


def layer_prediction(opt_config, is_load_store, batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len, hardware_config, gpu_estimator, layer_type="MHA", first_token=False):
    # Return: Tot Energy, Tot Latency, Energy for Data Transfer, Energy for Active GPU , Latency for Data Transfer
    #layer type determines the actual recomputation time + compute layer time
    # print(f"layer info: layer_type: {layer_type}, gen_len: {gen_len}, load or store: {is_load_store}, recomp_len: {recomp_len}")
    layer_calc_energy, layer_calc_latency, fir_token_after_KV_energy, fir_token_after_KV_latency = layer_calc_pred(opt_config, prompt_len, gen_len, batch_size, hardware_config, gpu_estimator, layer_type, first_token)
    if first_token:
        store_KV_energy = 0.0
        store_KV_latency = 0.0
        if layer_type == "MHA":
            kv_bytes_to_store = get_bytes_to_store(batch_size, prompt_len, first_token=first_token)
            store_KV_energy, store_KV_latency = transfer_pred(kv_bytes_to_store, hardware_config, gpu_estimator) # using HtoD est. for this DtoH
        tot_energy = layer_calc_energy+store_KV_energy
        tot_latency = (layer_calc_latency-fir_token_after_KV_latency) + max(fir_token_after_KV_latency, store_KV_latency)
        # print(f"first token: layer: {layer_type}, layer calc lat: {layer_calc_latency}, store lat: {store_KV_latency}, tot layer lat: {tot_latency}")
        return tot_energy, tot_latency, store_KV_energy, layer_calc_energy, store_KV_latency


    if layer_type == "MHA" and recomp_len > 0:
        recomp_energy, recomp_latency = recomp_calc_pred(opt_config, batch_size, prompt_len, gen_len, recomp_len, gpu_estimator, hardware_config)
        layer_calc_energy += recomp_energy
        layer_calc_latency += recomp_latency
  
    if is_load_store == 0:
        #no load or store, just the layer computations
        energy_transfer = 0.0
        energy_active = layer_calc_energy
        latency_transfer = 0.0
        return layer_calc_energy, layer_calc_latency, energy_transfer, energy_active, latency_transfer
    elif is_load_store == 2:
        # store only --> single directional
        transfer_energy, transfer_latency = transfer_pred(get_bytes_to_store(batch_size), hardware_config, gpu_estimator)
        
        return layer_calc_energy+transfer_energy, max(layer_calc_latency, transfer_latency), transfer_energy, layer_calc_energy, transfer_latency
    else:
        recomp_bytes, kv_load_bytes = get_bytes_to_load(opt_config, batch_size, num_of_batches, offload_percent, recomp_len, prompt_len, gen_len)
        # print(f"recomp_bytes: {recomp_bytes}, kv_load_bytes: {kv_load_bytes}")
        
        #recomp transfer & first kv load are always single directional. the second kv load uses single_directional indicator
        pinned_energy, pinned_latency = pinned_pred(kv_load_bytes, hardware_config, gpu_estimator)
        # first_half_latency = max(pinned_latency, recomp_prep_pred(prompt_len, recomp_len, hardware_config)+transfer_pred(recomp_bytes, hardware_config, gpu_estimator))
        recomp_transfer_energy, recomp_transfer_latency = transfer_pred(recomp_bytes, hardware_config, gpu_estimator)
        # print(f"load and store first half: latency is max of pinned: {pinned_latency} and transfer: {recomp_transfer_latency}")
        first_half_latency = max(pinned_latency, recomp_transfer_latency)
        
        k_transfer_energy, k_transfer_latency = transfer_pred(kv_load_bytes, hardware_config, gpu_estimator)
        if is_load_store == 1: #use is_load_store==1 as single directional
            v_transfer_energy = k_transfer_energy
            v_transfer_latency = k_transfer_latency
        else: 
            v_transfer_energy, v_transfer_latency = transfer_pred(kv_load_bytes, hardware_config, gpu_estimator, single_directional = False)
        # print(f"load and store second half: latency is max of pinned + layer calc: {pinned_latency  + layer_calc_latency} and transfer: {k_transfer_latency + v_transfer_latency}")
        second_half_latency = max(pinned_latency + layer_calc_latency, k_transfer_latency + v_transfer_latency)
        tot_energy =  pinned_energy + recomp_transfer_energy + layer_calc_energy + k_transfer_energy + v_transfer_energy
        tot_transfer_energy = recomp_transfer_energy + k_transfer_energy + v_transfer_energy
        tot_active_energy = layer_calc_energy
        tot_transfer_lat = recomp_transfer_latency + k_transfer_latency + v_transfer_latency
      
        return tot_energy, first_half_latency + second_half_latency, tot_transfer_energy, tot_active_energy, tot_transfer_lat


def pinned_pred(bytes, hardware_config, gpu_estimator):
    if bytes == 0:
        return 0, 0
    latency_us = max(0, 5.168e-14 * bytes**2 + 3.317e-05 * bytes - 104.7)
    latency = latency_us / 1000000.0
    # print(f"pinned: bytes (B): {bytes}, latency (s): {latency}")

    # Pageable to Pinned does not contribute to GPU energy --> idle 
    gpu_energy = gpu_estimator.dvfs_idle_power[str(hardware_config.gpu_freq)] * latency 
    return gpu_energy, latency


def transfer_pred(bytes, hardware_config, gpu_estimator, single_directional=True):
    if bytes == 0:
        return 0,0
    bytes = bytes / 1000000000.0 # bytes --> GB
    if single_directional:
        bandwidth = 1.415 * math.log10(bytes) + 13.38
        bandwidth = 12.04 - 5.561*math.exp(-259.6*bytes) 
    else: 
        bandwidth = 3.626 * math.log10(bytes) + 26.13 
        bandwidth = 11.58 - 6.802*math.exp(-249.4*bytes)
      
    # bandwidth = max(min(bandwidth, 15.0), 0)
    bandwidth = max(bandwidth, 0)
    latency = max(bytes/bandwidth, 0)
    
    gpu_energy = gpu_estimator.dvfs_idle_power[str(hardware_config.gpu_freq)] * latency
    return gpu_energy, latency

def recomp_calc_pred(opt_config, batch_size, prompt_len, cur_gen_len, recomp_len, gpu_estimator, hardware_config, first_token=False):
    # estimator: op
    # FlashAttention: ['flashattention_v2']
    # elementWise   : ['pointwise_mul', 'pointwise_add', 'scalar_mul', 'scalar_add', 'typecast_to_fp32', 'typecast_to_bf16', 'relu', 'gelu', 'silu', 'tanh', 'sigmoid', 'unspecified_activation', 'unspecified_tensor', 'unspecified_scalar']
    # gemmLike      : ['gemm', 'fmha-approximate']
    # NonLinear     : ['softmax', 'layernorm', 'softmax_fusion', 'layernorm_fusion']
    # energaizer-ispass26-artifact/experiments_single/gee_estimator.py --> query specs 

    if first_token:
        return 0, 0 # no recomputation, layer calculation will already take care of it

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
  
    copy_1_query = {
        'dim': recomp_len,
        'op': 'unspecified_tensor',
        'prec': 'bf16',
    }
    copy_1_query_type = ('elementwise')
    for i in range(2):
        all_queries.append(copy_1_query)
        all_query_types.append(copy_1_query_type)

    if (prompt_len + cur_gen_len - recomp_len) > 0:
        copy_2_query = {
            'dim': prompt_len + cur_gen_len - recomp_len,
            'op': 'unspecified_tensor',
            'prec': 'bf16',
        }
        copy_2_query_type = ('elementwise')
        for i in range(2):
            all_queries.append(copy_2_query)
            all_query_types.append(copy_2_query_type)

    #TODO: verify target frequency
    tot_lat = 0
    tot_energy = 0
    for each_ind in range(len(all_queries)):
        latency, _, energy = gpu_estimator.lookup(all_queries[each_ind], all_query_types[each_ind], target_freq=hardware_config.gpu_freq, lookup_target='all')
        # print(f"recomp gpu op: query: {all_queries[each_ind]}, latency: {latency}")
        tot_lat += latency
        tot_energy += energy
    tot_lat /= 1000.0 # ms --> s
    return tot_energy, tot_lat

def layer_calc_pred(opt_config, prompt_len, gen_len, batch_size, hardware_config, gpu_estimator, layer_type="MHA", first_token=False):
    all_queries = []
    all_query_types = []

    hidden_size = opt_config.hidden_size
    cur_seq_len = prompt_len + gen_len
    prev_not_seen = 1
    if first_token:
        prev_not_seen = prompt_len
  
    #input: 
    if layer_type == "input":
        # print(f"layer calc: input")
        embed_query = {
            'dim': batch_size*prev_not_seen,
            'op': 'unspecified_tensor',
            'prec': 'bf16',
        }
        embed_query_type = ('elementwise')
        all_queries.append(embed_query)
        all_query_types.append(embed_query_type)
      
        cumsum_query = {
            'dim': batch_size*cur_seq_len,
            'op': 'unspecified_tensor',
            'prec': 'bf16',
        }
        cumsum_query_type = ('elementwise')
        all_queries.append(cumsum_query)
        all_query_types.append(cumsum_query_type)
        int_query = {
            'dim': batch_size*cur_seq_len,
            'op': 'pointwise_add',
            'prec': 'bf16',
        }
        int_query_type = ('elementwise')
        all_queries.append(int_query)
        all_query_types.append(int_query_type)
        matMul_element_query = {
            'dim': batch_size*cur_seq_len,
            'op': 'pointwise_mul',
            'prec': 'bf16',
        }
        matMul_element_query_type = ('elementwise')
        all_queries.append(matMul_element_query)
        all_query_types.append(matMul_element_query_type)
        add_scal_query = {
            'dim': batch_size*cur_seq_len,
            'op': 'scalar_add',
            'prec': 'bf16',
        }
        add_scal_query_type = ('elementwise')
        all_queries.append(add_scal_query)
        all_query_types.append(add_scal_query_type)

        embed_query = {
            'dim': batch_size*prev_not_seen,
            'op': 'unspecified_tensor',
            'prec': 'bf16',
        }
        embed_query_type = ('elementwise')
        all_queries.append(embed_query)
        all_query_types.append(embed_query_type)
      
        add_mat_query = {
            'dim': batch_size*prev_not_seen*hidden_size,
            'op': 'pointwise_add',
            'prec': 'bf16',
        }
        add_mat_query_type = ('elementwise')
        all_queries.append(add_mat_query)
        all_query_types.append(add_mat_query_type)

    elif layer_type == "output":
        # print(f"layer calc: output")
        #output:  
        layer_norm_query = {'batch': batch_size,
                             'dim': hidden_size, 
                             'prec': 'bf16'}
        layer_norm_query_type = ('layernorm')
        all_queries.append(layer_norm_query)
        all_query_types.append(layer_norm_query_type)
      
        linear_query = {
        	'batch': batch_size,
        	'dimM' : prev_not_seen,
        	'dimN' : hidden_size,
        	'dimK' : opt_config.vocab_size,
        	'precM': 'bf16',
        	'precA': 'bf16',
        	'useTensorCore':  True
        }
        linear_query_type = ('gemm', 'tc', 'bf16')
        all_queries.append(linear_query)
        all_query_types.append(linear_query_type)

        argmax_query = {
            'dim': batch_size*opt_config.vocab_size,
            'op': 'scalar_add',
            'prec': 'bf16',
        }
        argmax_query_type = ('elementwise')
        all_queries.append(argmax_query)
        all_query_types.append(argmax_query_type)

    elif layer_type == "MLP":
        # print(f"layer calc: MLP")
        #MLP: 
        ffn_embed_dim = opt_config.ffn_embed_dim
      
        layer_norm_query = {'batch': batch_size,
                             'dim': hidden_size, 
                             'prec': 'bf16'}
        layer_norm_query_type = ('layernorm')
        all_queries.append(layer_norm_query)
        all_query_types.append(layer_norm_query_type)

        linear_query = {
        	'batch': batch_size,
        	'dimM' : prev_not_seen,
        	'dimN' : hidden_size,
        	'dimK' : ffn_embed_dim,
        	'precM': 'bf16',
        	'precA': 'bf16',
        	'useTensorCore':  True
        }
        linear_query_type = ('gemm', 'tc', 'bf16')
        all_queries.append(linear_query)
        all_query_types.append(linear_query_type)
    
        relu_query = {
            'dim': batch_size*prev_not_seen*ffn_embed_dim,
            'op': 'relu',
            'prec': 'bf16',
        }
        relu_query_type = ('elementwise')
        all_queries.append(relu_query)
        all_query_types.append(relu_query_type)
      
        linear_query = {
        	'batch': batch_size,
        	'dimM' : prev_not_seen,
        	'dimN' : ffn_embed_dim,
        	'dimK' : hidden_size,
        	'precM': 'bf16',
        	'precA': 'bf16',
        	'useTensorCore':  True
        }
        linear_query_type = ('gemm', 'tc', 'bf16')
        all_queries.append(linear_query)
        all_query_types.append(linear_query_type)
      
        add_query = {
            'dim': batch_size*prev_not_seen*ffn_embed_dim,
            'op': 'pointwise_add',
            'prec': 'bf16',
        }
        add_query_type = ('elementwise')
        all_queries.append(add_query)
        all_query_types.append(add_query_type)
  
    
    elif layer_type == "MHA":
        # print(f"layer calc: MHA")
        num_head = opt_config.n_head
        head_dim = hidden_size // num_head
      
        layer_norm_query = {'batch': batch_size,
                             'dim': hidden_size, 
                             'prec': 'bf16'}
        layer_norm_query_type = ('layernorm')
        all_queries.append(layer_norm_query)
        all_query_types.append(layer_norm_query_type)
      
        linear_query = {
        	'batch': batch_size,
        	'dimM' : prev_not_seen,
        	'dimN' : hidden_size,
        	'dimK' : hidden_size,
        	'precM': 'bf16',
        	'precA': 'bf16',
        	'useTensorCore':  True
        }
        linear_query_type = ('gemm', 'tc', 'bf16')
        for i in range(3):
            all_queries.append(linear_query)
            all_query_types.append(linear_query_type)
      
        # 3x REshape: (batch_size, n_head, 1, head_dim) → (b*n_head, 1, head_dim) 
        reshape_query = {
            'dim': batch_size*num_head*prev_not_seen*head_dim,
            'op': 'unspecified_tensor',
            'prec': 'bf16',
        }
        reshape_query_type = ('elementwise')
        for i in range(3):
            all_queries.append(reshape_query)
            all_query_types.append(reshape_query_type)
          
        bmm_query = {
        	'batch': batch_size*num_head,
        	'dimM' : prev_not_seen,
        	'dimN' : cur_seq_len,
        	'dimK' : head_dim,
        	'precM': 'bf16',
        	'precA': 'bf16',
        	'useTensorCore':  True
        }
        bmm_query_type = ('gemm', 'tc', 'bf16')
        all_queries.append(bmm_query)
        all_query_types.append(bmm_query_type)

        # torch.where() → elementwise [batch_size, 1, 1, cur_seq_len] skim over [batch_size, num_head, 1, cur_seq_len]
        where_query = {
            'dim': batch_size*prev_not_seen*cur_seq_len,
            'op': 'unspecified_tensor',
            'prec': 'bf16',
        }
        where_query_type = ('elementwise')
        all_queries.append(where_query)
        all_query_types.append(where_query_type)
      
        softmax_query = {'batch': batch_size*num_head*prev_not_seen,
                         'dim': cur_seq_len, 
                         'prec': 'bf16'}
        softmax_query_type = ('softmax')
        all_queries.append(softmax_query)
        all_query_types.append(softmax_query_type)
        
        bmm_query = {
        	'batch': batch_size*num_head,
        	'dimM' : prev_not_seen,
        	'dimN' : head_dim,
        	'dimK' : cur_seq_len,
        	'precM': 'bf16',
        	'precA': 'bf16',
        	'useTensorCore':  True
        }
        bmm_query_type = ('gemm', 'tc', 'bf16')
        all_queries.append(bmm_query)
        all_query_types.append(bmm_query_type)
        
        linear_query = {
        	'batch': batch_size,
        	'dimM' : prev_not_seen,
        	'dimN' : hidden_size,
        	'dimK' : hidden_size,
        	'precM': 'bf16',
        	'precA': 'bf16',
        	'useTensorCore':  True
        }
        linear_query_type = ('gemm', 'tc', 'bf16')
        all_queries.append(linear_query)
        all_query_types.append(linear_query_type)
          
        # [batch_size, prev_not_seen, opt_config.hidden_size] + [batch_size, prev_not_seen, opt_config.hidden_size]
        add_query = {
            'dim': batch_size*prev_not_seen*hidden_size,
            'op': 'pointwise_add',
            'prec': 'bf16',
        }
        add_query_type = ('elementwise')
        all_queries.append(add_query)
        all_query_types.append(add_query_type)
    else: 
        print(f"Layer type {layer_type} not supported")
        return 0, 0
  
    #TODO: verify target frequency
    tot_lat = 0
    tot_energy = 0
    fir_token_after_KV_latency = 0
    fir_token_after_KV_energy = 0
    
    for each_ind in range(len(all_queries)):
        latency, _, energy = gpu_estimator.lookup(all_queries[each_ind], all_query_types[each_ind], target_freq=hardware_config.gpu_freq, lookup_target='all')
        tot_lat += latency
        tot_energy += energy
        if layer_type == "MHA" and first_token and each_ind > 2:
            fir_token_after_KV_latency += latency
            fir_token_after_KV_energy += energy
    tot_lat /= 1000.0 # ms --> s
    fir_token_after_KV_latency /= 1000.0
    # print(f"layer_calc: type: {layer_type}, energy (J): {tot_energy}, latency (s) : {tot_lat}")
    return tot_energy, tot_lat, fir_token_after_KV_energy, fir_token_after_KV_latency





def disect_input(model, opt_config, num_of_prompts, prompt_len, gen_len, hardware_config, save_results, testing, fast, gpu_estimator, var_to_min="latency"):
    # break model into layers
  
    ### UNDERSTAND WHAT STRATEGIES ARE AVVAILABLE
                                                                                                                                                                                       
    # understand what unique batch size is available 
    batch_sizes = get_batch_sizes(num_of_prompts)
    # understand what % offloadings are available 
    all_feasible_strategies_dict = get_available_offloadings(opt_config, hardware_config, batch_sizes, num_of_prompts, prompt_len+gen_len)
    
    ### ITERATE AND COMPARE STRATEGIES

    #iterate through 0-100% recomputing, % offloading, and batch size
    min_objective_val = float('inf')
    min_strategy = None

    if save_results: 
        all_results = {}

    if testing: 
        
        test_batch_size = 1
        test_offloading_per = 10
        test_recomp_len = 0

        batch_sizes = [1, 2, 4, 8]
        all_feasible_strategies_dict = {1: [10], 2:[60], 4:[60], 8:[90]}

        # batch_sizes = [1, 2]
        # all_feasible_strategies_dict = {1: [10], 2:[60]}

        # batch_sizes = [4]
        # all_feasible_strategies_dict = {4:[60]}

    # start searching 
    for each_batch_size in batch_sizes:
        for each_feasible_offloading in all_feasible_strategies_dict[each_batch_size]:
            for each_recomp_percent in range(0, 110, 10):
                each_recomp_len = prompt_len * each_recomp_percent // 100 # recomp is only for prompt len
                print(" ")
                print(f'cur strat: {each_batch_size}, {each_feasible_offloading}, {each_recomp_len}')
                #Model Prediction 
                if fast:
                    cur_energy, cur_latency, cur_TTFT, avg_energy_per_layer, avg_latency_per_layer, per_transfer_energy, per_active_energy, per_transfer_latency = fast_strat_prediction(opt_config, num_of_prompts, prompt_len, gen_len, hardware_config, each_recomp_len, each_feasible_offloading, each_batch_size, num_of_prompts // each_batch_size, gpu_estimator)
                else: 
                    cur_energy, cur_latency, cur_TTFT, avg_energy_per_layer, avg_latency_per_layer, per_transfer_energy, per_active_energy, per_transfer_latency = strategy_prediction(opt_config, num_of_prompts, prompt_len, gen_len, hardware_config, each_recomp_len, each_feasible_offloading, each_batch_size, num_of_prompts // each_batch_size, gpu_estimator)
                if save_results: 
                    cur_strat = (each_batch_size, each_feasible_offloading, each_recomp_len)
                    all_results[cur_strat] = (cur_energy, cur_latency, cur_TTFT, avg_energy_per_layer, avg_latency_per_layer, per_transfer_energy, per_active_energy, per_transfer_latency)
              
                cur_objective_val = cur_latency
                if var_to_min == "energy":
                    cur_objective_val = cur_energy
                # compare to optimal policy seen so far
                if cur_objective_val < min_objective_val:
                    min_objective_val = cur_objective_val
                    min_strategy = (each_batch_size, each_feasible_offloading, each_recomp_len, cur_energy, cur_latency, per_transfer_energy, per_active_energy, per_transfer_latency)

    if save_results:
        csv_filename = "all_pred_totP_" + str(num_of_prompts) +"prompt_len_" + str(prompt_len) + "gen_len" + str(gen_len) + ".csv"
        print(csv_filename)
        fieldnames = ["Batch Size", "Offloading Percent to CPU", "Recompute Length", "Energy (J)", "Latency (s)", "Time to First Token (s)", "Avg Input Layer Energy (J)", "Avg Input Layer Latency (s)", "Avg Output Layer Energy (J)", "Avg Output Layer Latency (s)", "Avg MHA Layer Energy (J)", "Avg MHA Layer Latency (s)", "Avg MLP Layer Energy (J)", "Avg MLP Layer Latency (s)", "Percent Transfer Energy (%)", "Percent Active Energy (%)", "Percent Transfer Latency (%)"]
        write_header = not os.path.exists(csv_filename)
      
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            print(f"# of results: {len(all_results)}")
            for each_strat in all_results:
              cur_energy, cur_latency, cur_TTFT, avg_energy_per_layer, avg_latency_per_layer, per_transfer_energy, per_active_energy, per_transfer_latency = all_results[each_strat]
              writer.writerow({'Batch Size': each_strat[0], 
                      'Offloading Percent to CPU': each_strat[1],
                      'Recompute Length': each_strat[2], 
                      'Energy (J)': cur_energy, 
                      'Latency (s)': cur_latency, 
                      'Time to First Token (s)': cur_TTFT,
                      'Avg Input Layer Energy (J)': avg_energy_per_layer["input"][0] / avg_energy_per_layer["input"][1], 
                      'Avg Input Layer Latency (s)': avg_latency_per_layer["input"][0] / avg_latency_per_layer["input"][1], 
                      'Avg Output Layer Energy (J)': avg_energy_per_layer["output"][0] / avg_energy_per_layer["output"][1], 
                      'Avg Output Layer Latency (s)': avg_latency_per_layer["output"][0] / avg_latency_per_layer["output"][1], 
                      'Avg MHA Layer Energy (J)': avg_energy_per_layer["MHA"][0] / avg_energy_per_layer["MHA"][1], 
                      'Avg MHA Layer Latency (s)': avg_latency_per_layer["MHA"][0] / avg_latency_per_layer["MHA"][1],
                      'Avg MLP Layer Energy (J)': avg_energy_per_layer["MLP"][0] / avg_energy_per_layer["MLP"][1], 
                      'Avg MLP Layer Latency (s)': avg_latency_per_layer["MLP"][0] / avg_latency_per_layer["MLP"][1],
                      'Percent Transfer Energy (%)': per_transfer_energy,
                      'Percent Active Energy (%)': per_active_energy,
                      'Percent Transfer Latency (%)': per_transfer_latency,
                      })

    # return best policy and optimal latency, energy, etc.
    print(f'best policy: batch_size = {min_strategy[0]}, offloading_percent = {min_strategy[1]}, recomp_len = {min_strategy[2]}, energy = {min_strategy[3]}, latency = {min_strategy[4]}, % Energy for Transfer = {min_strategy[5]}, % Energy for Active = {min_strategy[6]}, % Latency for Transfer = {min_strategy[7]}')
    return min_objective_val, min_strategy

def single_strat_pred(model, opt_config, num_of_prompts, prompt_len, gen_len, hardware_config, save_results, batch_size, offloading_per, recomp_len, fast, gpu_estimator, var_to_min="latency"):
    min_objective_val = float('inf')
    min_strategy = None

    if save_results: 
        all_results = {}

    # single run
    if fast: 
        cur_energy, cur_latency, cur_TTFT, avg_energy_per_layer, avg_latency_per_layer, per_transfer_energy, per_active_energy, per_transfer_latency  = fast_strat_prediction(opt_config, num_of_prompts, prompt_len, gen_len, hardware_config, recomp_len, offloading_per, batch_size, num_of_prompts // batch_size, gpu_estimator)
    else:
        cur_energy, cur_latency, cur_TTFT, avg_energy_per_layer, avg_latency_per_layer, per_transfer_energy, per_active_energy, per_transfer_latency  = strategy_prediction(opt_config, num_of_prompts, prompt_len, gen_len, hardware_config, recomp_len, offloading_per, batch_size, num_of_prompts // batch_size, gpu_estimator)
    if save_results: 
        cur_strat = (batch_size, offloading_per, recomp_len)
        all_results[cur_strat] = (cur_energy, cur_latency, cur_TTFT, avg_energy_per_layer, avg_latency_per_layer, per_transfer_energy, per_active_energy, per_transfer_latency )
  
    min_objective_val = cur_latency
    min_strategy = (batch_size, offloading_per, recomp_len, cur_energy, cur_latency, per_transfer_energy, per_active_energy, per_transfer_latency )
    
    if save_results:
        csv_filename = "all_pred_totP_" + str(num_of_prompts) +"prompt_len_" + str(prompt_len) + "gen_len" + str(gen_len) + ".csv"
        print(csv_filename)
        fieldnames = ["Batch Size", "Offloading Percent to CPU", "Recompute Length", "Energy (J)", "Latency (s)", "Time to First Token (s)", "Avg Input Layer Energy (J)", "Avg Input Layer Latency (s)", "Avg Output Layer Energy (J)", "Avg Output Layer Latency (s)", "Avg MHA Layer Energy (J)", "Avg MHA Layer Latency (s)", "Avg MLP Layer Energy (J)", "Avg MLP Layer Latency (s)", "Percent Transfer Energy (%)", "Percent Active Energy (%)", "Percent Transfer Latency (%)"]
        write_header = not os.path.exists(csv_filename)
      
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            print(f"# of results: {len(all_results)}")
            for each_strat in all_results:
              cur_energy, cur_latency, cur_TTFT, avg_energy_per_layer, avg_latency_per_layer, per_transfer_energy, per_active_energy, per_transfer_latency = all_results[each_strat]
              writer.writerow({'Batch Size': each_strat[0], 
                      'Offloading Percent to CPU': each_strat[1],
                      'Recompute Length': each_strat[2], 
                      'Energy (J)': cur_energy, 
                      'Latency (s)': cur_latency, 
                      'Time to First Token (s)': cur_TTFT,
                      'Avg Input Layer Energy (J)': avg_energy_per_layer["input"][0] / avg_energy_per_layer["input"][1], 
                      'Avg Input Layer Latency (s)': avg_latency_per_layer["input"][0] / avg_latency_per_layer["input"][1], 
                      'Avg Output Layer Energy (J)': avg_energy_per_layer["output"][0] / avg_energy_per_layer["output"][1], 
                      'Avg Output Layer Latency (s)': avg_latency_per_layer["output"][0] / avg_latency_per_layer["output"][1], 
                      'Avg MHA Layer Energy (J)': avg_energy_per_layer["MHA"][0] / avg_energy_per_layer["MHA"][1], 
                      'Avg MHA Layer Latency (s)': avg_latency_per_layer["MHA"][0] / avg_latency_per_layer["MHA"][1],
                      'Avg MLP Layer Energy (J)': avg_energy_per_layer["MLP"][0] / avg_energy_per_layer["MLP"][1], 
                      'Avg MLP Layer Latency (s)': avg_latency_per_layer["MLP"][0] / avg_latency_per_layer["MLP"][1],
                      'Percent Transfer Energy (%)': per_transfer_energy,
                      'Percent Active Energy (%)': per_active_energy,
                      'Percent Transfer Latency (%)': per_transfer_latency,
                      })

    # return best policy and optimal latency, energy, etc.
    print(f'best policy: batch_size = {min_strategy[0]}, offloading_percent = {min_strategy[1]}, recomp_len = {min_strategy[2]}, energy = {min_strategy[3]}, latency = {min_strategy[4]}, % Energy for Transfer = {min_strategy[5]}, % Energy for Active = {min_strategy[6]}, % Latency for Transfer = {min_strategy[7]}')
    return min_objective_val, min_strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-175b")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
  
    parser.add_argument("--gpu-mem", type=int, default=40)
    parser.add_argument("--cpu-mem", type=int, default=200)
    parser.add_argument("--cpu-usage", "--per-cpu-mem", type=int, default = 100)
    parser.add_argument("--gpu-usage", "--per-gpu-mem",type=int, default = 65)
    parser.add_argument("--gpu-freq", "--gpu-frequency",type=int, default = 1305)

    parser.add_argument("--s", "--specific-est", action="store_true")
    parser.add_argument("--gbs", "--batch-size", type=int, default = 1)
    parser.add_argument("--off-per", "--offloading-percent", type=int, default = 0)
    parser.add_argument("--recomp-len", type=int, default = 0)

    parser.add_argument("--np", "--num-prompts", type=int)
    parser.add_argument("--test", "--testing", action="store_true")
    parser.add_argument("--fast", action="store_true")
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

    alpha_c = args.cpu_usage / 100
    alpha_g = args.gpu_usage / 100

    config.cmem = alpha_c * args.cpu_mem * GB
    config.gmem = alpha_g * args.gpu_mem * GB
    config.gpu_freq = args.gpu_freq
    

    #TODO: specify hardware config
    print("currently getting gpu estimator")
    gpu_estimator = get_gee(gpu_yaml_path="/home/akleang/akleang/energaizer-ispass26-artifact/config/gpu/yz8.yaml", 
                        lut_yaml_path="/home/akleang/akleang/energaizer-ispass26-artifact/experiments_endtoend/exp_config/a100_dvfs_lut_config.yaml", 
                        dvfs_aware=True, dvfs_inference_mode='all', 
                        dvfs_supply_voltage_json="/home/akleang/akleang/energaizer-ispass26-artifact/config/dvfs/yz8/supply_voltage.json",
                        dvfs_idle_power_json="/home/akleang/akleang/energaizer-ispass26-artifact/config/dvfs/yz8/idle_power.json", 
                        lut_folder_abs_path="/home/akleang/akleang/energaizer-ispass26-artifact/database/data")
    if args.s:
        single_strat_pred(args.model, opt_config, args.np, args.prompt_len, args.gen_len, config, args.save, args.gbs, args.off_per, args.recomp_len, args.fast, gpu_estimator)
    else: 
        disect_input(args.model, opt_config, args.np, args.prompt_len, args.gen_len, config, args.save, args.test, args.fast, gpu_estimator)

