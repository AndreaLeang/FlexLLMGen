For running experiments + prediction model:

## Conda Environment:
dependencies
  - python=3.10
  - pip
  - pip:
      - numpy
      - pandas
      - pynvml
      - pyyaml
      - scipy
      - torch==2.7.1
      - torchaudio
      - torchvision
      - torchlens
      - pulp
      - nvidia-ml-py
      - psutil
      - transformers

cd into FlexLLMGen folder and run
pip install -e .

## Collecting data for Modeling:
```
python3 flexllmgen/flex_opt_kvpr.py --gen-len 16 --profile --model facebook/opt-6.7b --percent 100 0 0 100 100 0 --prompt-len 2048 --sweep-re --sweep-b-size
```

With an option to bind CPU and GPU:
```
sudo numactl --cpunodebind=0 --membind=0 env CUDA_VISIBLE_DEVICES=0 python3 ... 
```

Middle two Percent numbers: % KV Cache on GPU, % KV Cache on CPU
--sweep-b-size sweeps through the block sizes to use 16 total prompts, but if you want to use a certain block size + number of blocks, use --gpu-batch-size 2 --num-gpu-batches 8
Prompt Len: 1024, 2048, 3072, 4096
sweep-re automatically sweeps through all possible recomputation lengths
python3 kv_timer_verifier.py --event-dist --split-dir --recomp
Automatically detects json files 
optional: --flops to include FLOPS recording

*one*.csv - One-directional Loads and Stores
*bi*.csv - Bi-directional Loads and Stores
*pinned.csv - Pageable to Pinned Components

*mha_flops.csv - FLOPS during Layer Computations
*recomp_flops.csv - FLOPS during Recomputation


## Modeling (Matlab scripts in Modeling folder in FlexLLMGen):
### PtP:
put all the *pinned.csv files in a folder and run Pinned_Modeling.m
Set the folder/filenames in matlab script: 
folder_path : folder where *pinned.csv files are
csv_folder: folder relative to script where *pinned.csv files are
cur_file_beg: beginning of filename for png
Change last saveas line to match folder where graph should be saved
Resulting curve/equation will be output in matlab
### Bandwidth:
put all files *one*.csv in a folder (best folder where the script is) and run Overall_Transfer_BWTrend.m 
Set the folder/filenames in matlab script: 
folder_path : folder where only *pinned.csv files are
csv_folder: folder relative to script where *pinned.csv files are
cur_file_beg: beginning of filename for png
Change last saveas line to match folder where graph should be saved
Resulting curve/equation will be output in matlab
put all *bi*.csv files in a different folder and run Overall_Transfer_BWTrend.m 
Set one_test to False
Set the folder/filenames in matlab script: 
folder_path : folder where *pinned.csv files are
csv_folder: folder relative to script where *pinned.csv files are
cur_file_beg: beginning of filename for png
Resulting curve/equation will be output in matlab

## Changing Prediction Model: kv_schedule_optimization.py
### PtP:
function pinned_pred: change latency_us = equation 
### Bandwidth:
function transfer_pred: change if single_directional:equation  for single directional bandwidth 
change else equation for bi-directional bandwidth 
Change line sys.path.append( '/home/akleang/akleang/energaizer-ispass26-artifact/') to relevant EnergAIzer path

### Running Prediction Model: kv_schedule_optimization.py
```
python3 kv_schedule_optimization.py --save --gen-len 16 --d --fast --model facebook/opt-6.7b --gpu-mem 40 gpu-freq 1305 --prompt-len 4096 --np 16 
```
change model, gpu-mem, gpu-freq, prompt-len, np to fit current workload & gpu memory size & frequency

## Getting True Latency and Energy for Workload on Strategy:
### Latency:
```
python3 flexllmgen/flex_opt_kvpr.py --gen-len 16 --profile --model facebook/opt-6.7b --gpu-batch-size 2 --num-gpu-batches 8 --percent 100 0 0 100 100 0 --prompt-len 2048 --recomp-len 0
```
Decode latency is stored in experienced_data.csv
### Energy:
```
sudo CUDA_VISIBLE_DEVICES=0 python3 rapl-nvml-power-monitor-main/run_experiment.py  --gpu 0  --sockets 0 1 --n_iters 5 --min_duration 1.0 --gen-len 16 --model facebook/opt-13b --prompt-len 1024 --block-size 8 --num-blocks 2 --off-per 90 --recomp-len 0
```
off-per = % of KV Cache on CPU
change gpu and CUDA_VISIBLE_DEVICES to relevant gpu

## Trace Structure

### w/ recompute
1. Timer start/stop in python; the total number of start/stop pair equals to the number of tokens generated. 
2. Within each start-stop pair, there is a single python operator 'update_attention_mask'. Afterwards, there are multiple loops with the following 8 operators: load_weight, load_hidden_compute, load_cache, store_hidden, load_hidden, compute_layer, store_cache, sync
3. For each group of these 8 operators, we can identify which layer is corresponds to from looking at what function 'compute_layer' calls. We are particularly interested in a function 'mha_gen' (compute_layer -> forward -> mha_gen) and 'mlp' (compute_layer -> forward -> mlp). Group 2 of these 8-operator group with the compute_layer calling mha_gen and mlp. 
4. For these mha_gen-mlp group, we are interested in the duration (latency as recorded in this trace) of these sub-function calls:
  - mha_gen's 'load_weight' duration
  - mha_gen's 'load_hidden_compute' duration
  - mha_gen's 'load_cache' duration
  - mha_gen's 'load_hidden' duration
  - mha_gen's 'compute_layer' duration
  - within mha_gen's 'compute_layer', identify any cuda calls originating from functions within 'compute_layer' (there can be multiple, all mapped to the same cuda stream). for all those cuda calls, record their durations (annotate as mha-gen-compute-cuda)
  - mha_gen's 'store_cache' duration
  - mha_gen's 'store_hidden' duration
  - mha_gen's 'sync' duration
  - mlp's 'load_weight' duration
  - mlp's 'load_hidden_compute' duration
  - mlp's 'load_hidden_compute' calls cudaMemcpyAsync internally; record the corresponding cuda stream's duration (annotate as load-hidden-compute-cudamemcpy)
  - mlp's 'load_cache' duration
  - mlp's 'load_cache' 'aten::pin_memory''s duration; there are two of 'aten::pin_memory'; record for each of them and annotate as pin-memory-1 and pin-memory-2
  - mlp's 'load_cache' also invokes two calls of cudaMemcpyAsync. record the corresponding cuda stream operation's duration for each (annotate as load-cache-cudamemcpy-1, load-cache-cudamemcpy-2)
  - mlp's 'load_hidden' duration
  - mlp's 'compute_layer' duration
  - within mlp's 'compute_layer', there are multiple cuda calls invoked. for all those calls, record their cuda stream's duration (annotate as mlp-compute-cuda). 
  - mlp's 'store_cache' duration
  - mlp's 'store_cache' calls two cudaMemcpyAsync operations. for each of them, record their cuda stream's duration (annotate as store-cache-cudamemcpy-1, store-cache-cudamemcpy-2).
  - mlp's 'store_hidden' duration
  - mlp's 'sync' duration
5. Return the results as a csv file. 