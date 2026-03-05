import json
import csv
import argparse
import os

from pathlib import Path
# import matplotlib.pyplot as plt

def get_all_gpu_memcpy_correlations(json_filename, get_cpu_time=False, est_bandwidth=False, record_ind_events=False, split_dir=False, re_dist=False):
    # open json file
    with open(json_filename, 'r') as file:
        # load json file
        data = json.load(file)
    
    # initialize the dictionary to hold the ac2g
    ac2g_dict = {}
    total_loading_cache_time_gpu = 0
    total_storing_cache_time_gpu = 0

    total_loading_cache_time_cpu = 0
    total_storing_cache_time_cpu = 0

    total_loading_bytes = 0
    total_storing_bytes = 0

    est_loading_bandwidth = 0 # = total loading bytes / total time 
    est_storing_bandwidth = 0 # = total storing bytes / total time 

    load_intervals = []
    store_intervals = []
    recomp_data_transfer_intervals = []
    recomp_data_calc_intervals = []

    # for each item in the result of the "traceEvents" key,
    num_of_events = len(data['traceEvents'])

    # get load and store intervals
    for event_idx in range(num_of_events):
        event = data['traceEvents'][event_idx]

        if event['name'] == 'flex_opt_kvpr.py(890): load_cache': 
            loading_cache_start_time = event['ts']
            loading_cache_end_time = event['ts'] + event['dur']
            load_intervals.append((loading_cache_start_time, loading_cache_end_time))
        elif event['name'] == 'flex_opt_kvpr.py(932): store_cache': 
            storing_cache_start_time = event['ts']
            storing_cache_end_time = event['ts'] + event['dur']
            store_intervals.append((storing_cache_start_time, storing_cache_end_time))
        elif event['name'] == 'flex_opt_kvpr.py(911): load_hidden_compute':
            load_for_recompute_start_time = event['ts']
            load_for_recompute_end_time = event['ts'] + event['dur']
            recomp_data_transfer_intervals.append((load_for_recompute_start_time, load_for_recompute_end_time))
        elif event['name'] == 'flex_opt_kvpr.py(1025): compute_layer':
            recompute_start_time = event['ts']
            recompute_end_time = event['ts'] + event['dur']
            recomp_data_calc_intervals.append((recompute_start_time, recompute_end_time))

    # print(f"load_intervals: {load_intervals}")
    # print(f"store_intervals: {store_intervals}")

    load_memcpy_correlations = []
    store_memcpy_correlations = []
    recomp_load_memcpy_correlations = []

    recomp_compute_times = {}
    recomp_compute_idx = 0
    tot_recomp_time = 0

    # get corrleations from valid cudaMemcpyAsync events
    for event_idx in range(num_of_events):
        event = data['traceEvents'][event_idx]
        in_load = False
        in_store = False
        if event['name'] == 'cudaMemcpyAsync':
            # print(f"found cudaMemcpyAsync")
            for interval in load_intervals:
                # print(f"checking interval {interval}")
                if event['ts'] >= interval[0] and event['ts'] < interval[1]:
                    load_memcpy_correlations.append(event['args']['correlation'])
                    if get_cpu_time:
                        total_loading_cache_time_cpu += event['dur']
                    in_load = True
                    break
                elif event['ts'] < interval[0]:
                    break

            if not in_load:
                for interval in store_intervals:
                    if event['ts'] >= interval[0] and event['ts'] < interval[1]:
                        store_memcpy_correlations.append(event['args']['correlation'])
                        if get_cpu_time:
                            total_storing_cache_time_cpu += event['dur']
                        in_store = True
                        break
                    elif event['ts'] < interval[0]:
                        break
            if not in_store and re_dist:
                for interval in recomp_data_transfer_intervals:
                    if event['ts'] >= interval[0] and event['ts'] < interval[1]:
                        recomp_load_memcpy_correlations.append(event['args']['correlation'])
                        break
                    elif event['ts'] < interval[0]:
                        break
        elif re_dist and event['name'] == 'pytorch_backend.py(425): mha_gen':
            # difference between of start of compute_layer and start of mha
            for interval in recomp_data_calc_intervals:
                if event['ts'] >= interval[0] and event['ts'] < interval[1]:
                    calc_time = event['ts'] - interval[0]
                    tot_recomp_time += calc_time
                    recomp_compute_times[recomp_compute_idx] = calc_time
                    recomp_compute_idx += 1
                elif event['ts'] < interval[0]:
                    break

    # get the actual GPU memcpy durations

    if record_ind_events:
        loading_events = {}
        storing_events = {}
        cur_loading_idx = 0
        cur_storing_idx = 0

    if split_dir: 
        load_intervals = []
        store_intervals = []
        bi_dir_loading_events = {}
        bi_load = {}
        bi_dir_load_ind = 0
        bi_dir_storing_events = {}
        bi_dir_store_ind = 0
        one_dir_loading_events = {}
        one_dir_load_ind = 0
        one_dir_storing_events = {}
        one_dir_store_ind = 0

    if re_dist:
        re_load_intervals = []
        all_re_load = {}
        bi_re_load = {}
        bi_re_load_events = {}
        bi_dir_re_load_ind = 0
        one_re_load = {}
        one_re_load_events = {}
        one_dir_re_load_ind = 0
        cur_re_loading_idx = 0

    for event_idx in range(num_of_events):
        # find the cudaMemcpyAsync events under load_cache and store_cache
        # then find the ac2g connecting that to 
        #   Load_Cache: MemcpyHtoD (Pinned -> Device)
        #   Store_cache: MemcpyDtoH (Device -> Pageable)

        # (each item should be a dictionary in of it itself)
        event = data['traceEvents'][event_idx]
        if event['name'] == 'Memcpy HtoD (Pinned -> Device)':
            if event['args']['correlation'] in load_memcpy_correlations:
                total_loading_cache_time_gpu += event['dur']
                if est_bandwidth: 
                    total_loading_bytes += event['args']['bytes']
                if split_dir: 
                    loading_cache_start_time = event['ts']
                    loading_cache_end_time = event['ts'] + event['dur']
                    load_intervals.append((loading_cache_start_time, loading_cache_end_time, cur_loading_idx))
                if record_ind_events:
                    loading_events[cur_loading_idx] = (event['args']['bytes'], event['args']['memory bandwidth (GB/s)'])
                    cur_loading_idx += 1
            elif re_dist and event['args']['correlation'] in recomp_load_memcpy_correlations: 
                re_loading_start_time = event['ts']
                re_loading_end_time = event['ts'] + event['dur']
                re_load_intervals.append((re_loading_start_time, re_loading_end_time, cur_re_loading_idx))
                
                all_re_load[cur_re_loading_idx] = (event['args']['bytes'], event['args']['memory bandwidth (GB/s)'])
                cur_re_loading_idx += 1
                
                
        elif event['name'] == 'Memcpy DtoH (Device -> Pageable)':
            if event['args']['correlation'] in store_memcpy_correlations:
                total_storing_cache_time_gpu += event['dur']
                if est_bandwidth: 
                    total_storing_bytes += event['args']['bytes']
                if split_dir: 
                    storing_cache_start_time = event['ts']
                    storing_cache_end_time = event['ts'] + event['dur']
                    store_intervals.append((storing_cache_start_time, storing_cache_end_time, cur_storing_idx))
                if record_ind_events:
                    storing_events[cur_storing_idx] = (event['args']['bytes'], event['args']['memory bandwidth (GB/s)'])
                    cur_storing_idx += 1
                
    if split_dir:
        for each_store_interval in store_intervals:
            store_start = each_store_interval[0]
            store_end = each_store_interval[1]
            store_id = each_store_interval[2]
            found_pair = False
            for each_load_interval in load_intervals:
                load_start = each_load_interval[0]
                load_end = each_load_interval[1]
                load_id = each_load_interval[2]
                if store_start >= load_start and store_end < load_end:
                    # found pair of bi directional
                    if load_id not in bi_load:
                        bi_dir_loading_events[bi_dir_load_ind] = (load_id, loading_events[load_id][0], loading_events[load_id][1])
                        bi_dir_load_ind += 1
                        bi_load[load_id] = True
                    bi_dir_storing_events[bi_dir_store_ind] = (store_id, storing_events[store_id][0], storing_events[store_id][1])
                    bi_dir_store_ind += 1
                    found_pair = True
                    break
                elif store_end < load_start:
                    break
            for each_re_load_interval in re_load_intervals:
                load_start = each_re_load_interval[0]
                load_end = each_re_load_interval[1]
                load_id = each_re_load_interval[2]
                if store_start >= load_start and store_end < load_end:
                    # found pair of bi directional
                    if load_id not in bi_re_load:
                        bi_re_load_events[bi_dir_re_load_ind] = (load_id, all_re_load[load_id][0], all_re_load[load_id][1])
                        bi_dir_re_load_ind += 1
                        bi_re_load[load_id] = True
                    bi_dir_storing_events[bi_dir_store_ind] = (store_id, storing_events[store_id][0], storing_events[store_id][1])
                    bi_dir_store_ind += 1
                    found_pair = True
                    break
                elif store_end < load_start:
                    break
            if not found_pair: 
                # add to 1 direction
                one_dir_storing_events[one_dir_store_ind] = (store_id, storing_events[store_id][0], storing_events[store_id][1])
                one_dir_store_ind += 1
        # one direction load
        for each_load_interval in load_intervals:
            load_id = each_load_interval[2]
            if load_id not in bi_load:
                one_dir_loading_events[one_dir_load_ind] = (load_id, loading_events[load_id][0], loading_events[load_id][1])
                one_dir_load_ind += 1
        # one direction re load
        for each_re_load_interval in re_load_intervals:
            load_id = each_re_load_interval[2]
            if load_id not in bi_re_load:
                one_re_load_events[one_dir_re_load_ind] = (load_id, all_re_load[load_id][0], all_re_load[load_id][1])
                one_dir_re_load_ind += 1
                    
        

    if record_ind_events:
        cur_filename = json_filename[:-5] 
        print(f"cur_filename: {cur_filename}")
        with open(cur_filename + '_all_loading.csv', 'w', newline='') as csvfile:
            fieldnames = ['idx', 'data (B)', 'bandwidth (GB/s)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for idx in range(cur_loading_idx):
                writer.writerow({'idx': idx, 'data (B)': loading_events[idx][0] , 'bandwidth (GB/s)': loading_events[idx][1]})
        with open(cur_filename + '_all_storing.csv', 'w', newline='') as csvfile:
            fieldnames = ['idx', 'data (B)', 'bandwidth (GB/s)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(cur_storing_idx):
                writer.writerow({'idx': idx, 'data (B)': storing_events[idx][0] , 'bandwidth (GB/s)': storing_events[idx][1]})
                
        if split_dir: 
            # bidirectional Load
            with open(cur_filename + '_bi_loading.csv', 'w', newline='') as csvfile:
                fieldnames = ['idx', 'og Index', 'data (B)', 'bandwidth (GB/s)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
    
                for idx in range(bi_dir_load_ind):
                    writer.writerow({'idx': idx, 'og Index': bi_dir_loading_events[idx][0], 'data (B)': bi_dir_loading_events[idx][1] , 'bandwidth (GB/s)': bi_dir_loading_events[idx][2]})

            # Bidirectional Store
            with open(cur_filename + '_bi_storing.csv', 'w', newline='') as csvfile:
                fieldnames = ['idx', 'og Index', 'data (B)', 'bandwidth (GB/s)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
    
                for idx in range(bi_dir_store_ind):
                    writer.writerow({'idx': idx, 'og Index': bi_dir_storing_events[idx][0], 'data (B)': bi_dir_storing_events[idx][1] , 'bandwidth (GB/s)': bi_dir_storing_events[idx][2]})

            # One Direction Load
            with open(cur_filename + '_one_loading.csv', 'w', newline='') as csvfile:
                fieldnames = ['idx', 'og Index', 'data (B)', 'bandwidth (GB/s)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
    
                for idx in range(one_dir_load_ind):
                    writer.writerow({'idx': idx, 'og Index': one_dir_loading_events[idx][0], 'data (B)': one_dir_loading_events[idx][1] , 'bandwidth (GB/s)': one_dir_loading_events[idx][2]})
            
            # One Direction Store
            with open(cur_filename + '_one_storing.csv', 'w', newline='') as csvfile:
                fieldnames = ['idx', 'og Index', 'data (B)', 'bandwidth (GB/s)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
    
                for idx in range(one_dir_store_ind):
                    writer.writerow({'idx': idx, 'og Index': one_dir_storing_events[idx][0], 'data (B)': one_dir_storing_events[idx][1] , 'bandwidth (GB/s)': one_dir_storing_events[idx][2]})
        if re_dist: 
            # Recomputation's Compute
            with open(cur_filename + '_all_load_r.csv', 'w', newline='') as csvfile:
                fieldnames = ['idx', 'Compute Time (s)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
    
                for idx in range(recomp_compute_idx):
                    writer.writerow({'idx': idx, 'Compute Time (s)': recomp_compute_times[idx]})
            # Recomputation's Load
            with open(cur_filename + '_all_load_r.csv', 'w', newline='') as csvfile:
                fieldnames = ['idx', 'data (B)', 'bandwidth (GB/s)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
    
                for idx in range(cur_re_loading_idx):
                    writer.writerow({'idx': idx, 'data (B)': all_re_load[idx][0] , 'bandwidth (GB/s)': all_re_load[idx][1]})
            # Recomputation's Bidirectional Load
            with open(cur_filename + '_bi_load_r.csv', 'w', newline='') as csvfile:
                fieldnames = ['idx', 'og Index', 'data (B)', 'bandwidth (GB/s)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
    
                for idx in range(bi_dir_re_load_ind):
                    writer.writerow({'idx': idx, 'og Index': bi_re_load_events[idx][0], 'data (B)': bi_re_load_events[idx][1] , 'bandwidth (GB/s)': bi_re_load_events[idx][2]})
            # Recomputation's One Direction Load
            with open(cur_filename + '_one_load_r.csv', 'w', newline='') as csvfile:
                fieldnames = ['idx', 'og Index', 'data (B)', 'bandwidth (GB/s)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
    
                for idx in range(one_dir_re_load_ind):
                    writer.writerow({'idx': idx, 'og Index': one_re_load_events[idx][0], 'data (B)': one_re_load_events[idx][1] , 'bandwidth (GB/s)': one_re_load_events[idx][2]})
            

    
    total_loading_cache_time_gpu /= 1000000.0 # originally in microseconds (10^-6)
    total_storing_cache_time_gpu /= 1000000.0
    if get_cpu_time:
        total_loading_cache_time_cpu /= 1000000.0
        total_storing_cache_time_cpu /= 1000000.0
    else: 
        total_loading_cache_time_cpu = None
        total_storing_cache_time_cpu = None

    if est_bandwidth:
        total_loading_bytes /= 1000000000.0 # B --> GB
        total_storing_bytes /= 1000000000.0 # B --> GB
    #     est_loading_bandwidth = total_loading_bytes / total_loading_cache_time_gpu # GB / s
    #     est_storing_bandwidth = total_storing_bytes / total_storing_cache_time_gpu # GB / s

    # Check if file exists to write header only once
    all_file_var = json_filename.split('-')
    cur_gbs = all_file_var[9]
    kv_gpu_percent = int(all_file_var[9])
    cur_recompute_len = all_file_var[14]
    csv_filename = json_filename.split('-percent')[0] + '-' + all_file_var[9] + '-' + all_file_var[10] + '_trace_stats.csv' # added header for recomp
    fieldnames = ['kv_gpu_percent', 'recompute_len', 'tot_loading_time_gpu (s)', 'tot_storing_time_gpu (s)', 'tot_loading_time_cpu (s)','tot_storing_time_cpu (s)',  'total_loading_bytes (GB)', 'total_storing_bytes (GB)', 'total_recompute_time (s)']

    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    # Open the file in append mode ('a')
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'kv_gpu_percent': kv_gpu_percent, 
                'recompute_len': cur_recompute_len,
                'tot_loading_time_gpu (s)': total_loading_cache_time_gpu, 
                'tot_storing_time_gpu (s)': total_storing_cache_time_gpu, 
                'tot_loading_time_cpu (s)': total_loading_cache_time_cpu, 
                'tot_storing_time_cpu (s)': total_storing_cache_time_cpu, 
                'total_loading_bytes (GB)': total_loading_bytes, 
                'total_storing_bytes (GB)': total_storing_bytes,
                'total_recompute_time (s)': tot_recomp_time})


    return total_loading_cache_time_gpu, total_storing_cache_time_gpu, total_loading_cache_time_cpu, total_storing_cache_time_cpu, total_loading_bytes, total_storing_bytes, tot_recomp_time # in s

def get_all_cpu_memcpy_correlations(json_filename):
    # open json file
    with open(json_filename, 'r') as file:
        # load json file
        data = json.load(file)
    
    # initialize the dictionary to hold the ac2g
    ac2g_dict = {}
    total_loading_cache_time = 0
    total_storing_cache_time = 0


def add_parser_arguments(parser):
    parser.add_argument('--files', nargs='+', help='List of files (space-separated)')
    parser.add_argument('--cpu-time',action="store_true", help="Measure the CPU data transfer time")
    parser.add_argument('--est-bandwidth', action="store_true", help="Measure the estimated bandwidth of the data transfer")
    parser.add_argument('--event-dist', action="store_true", help="Measure the distribution of the data transfer bytes")
    parser.add_argument('--split-dir', action="store_true", help="Split into bi direction and single direction data transfers")
    parser.add_argument('--recomp', action="store_true", help="Measure the distribution of data transfer for recomputation")

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent.absolute()
    # traces_dir = SCRIPT_DIR / "traces"
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)

    args = parser.parse_args()

    batch_filenames = args.files 
    
    all_kv_times = {}
    # batch_tklqt = [12014.2443359375, 18471.943251953126, 49995.56291894531, 104777.457796875, 232505.366203125, 461461.98833007814]
    for batch_filename in batch_filenames:
        print(f"Processing {batch_filename}")
        all_kv_times[batch_filename] = get_all_gpu_memcpy_correlations(str(SCRIPT_DIR / batch_filename), args.cpu_time, args.est_bandwidth, args.event_dist, args.split_dir, args.recomp)
        print(f"Total GPU Loading Cache Time for {batch_filename}: {all_kv_times[batch_filename][0]} s")
        print(f"Total GPU Storing Cache Time for {batch_filename}: {all_kv_times[batch_filename][1]} s")
        if args.cpu_time:
            print(f"Total CPU Loading Cache Time for {batch_filename}: {all_kv_times[batch_filename][2]} s")
            print(f"Total CPU Storing Cache Time for {batch_filename}: {all_kv_times[batch_filename][3]} s")
        if args.est_bandwidth:
            print(f"Total Loading Bytes for {batch_filename}: {all_kv_times[batch_filename][4]} GB")
            print(f"Total Storing Bytes (GB) for {batch_filename}: {all_kv_times[batch_filename][5]} GB")
        if args.recomp:
            print(f"Total Recomputation Loading Time for {batch_filename}: {all_kv_times[batch_filename][6]} s")
    print()
    print("GPU Load, GPU Store, CPU Load, CPU Store, Loading Bytes, Storing Bytes")
    for batch_filename in batch_filenames:
        print(f"for file {batch_filename}: {all_kv_times[batch_filename]}")
    

    
