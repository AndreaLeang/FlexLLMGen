import json
import csv
import argparse
import os

from pathlib import Path
# import matplotlib.pyplot as plt

def get_all_gpu_memcpy_correlations(json_filename, get_cpu_time=False, est_bandwidth=False, record_ind_events=False, split_dir=False, re_dist=False, re_no_load=False):
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

    #TODO
    # Get Smart copy head for pinned and memcpyasync connection
    pinned_smart_copy_intervals = {} # start of load_cache = [start of smart copy, end of smart copy, start of second smart copy, end of second smart copy]
    for event_idx in range(num_of_events):
        event = data['traceEvents'][event_idx]

        if event['name'] == 'pytorch_backend.py(142): smart_copy': 
            for interval in load_intervals:
                if event['ts'] >= interval[0] and event['ts'] < interval[1]:
                    if interval[0] not in pinned_smart_copy_intervals:
                        pinned_smart_copy_intervals[interval[0]] = []
                    pinned_smart_copy_intervals[interval[0]].append((event['ts'], event['ts'] + event['dur']))
                    break
    # Connect Memcpyasync bytes to each smart copy

    # Connect pinned to smart copy & record both the time for pinned & bytes


    # Connect Memcpyasync to KV Cache load & Store + 
    load_memcpy_correlations = []
    store_memcpy_correlations = []
    recomp_load_memcpy_correlations = []
    

    recomp_calc_times = {} # [idx] = time from start of compute layer to start of mha (rercompute)
    recomp_prep_transfer_times = {} # [idx] = time from start of load_hidden_compute to start of cudaMemcpyAsync, corrlelation (recompute prep)
    smart_load_cache_big_blocks = {} #[start of smart copy] = (end of smart copy, correlation of memcpyasync)
    kv_memcpy_to_bytes = {} # 
    recomp_memcpy_to_bytes = {}
    cache_pinned_times = {} # [idx] =  time of aten::pin_memory, bytes transferred (pinned) --> need to link to amount of bytes --> link to the memcpyasync 
    
    recomp_prep_idx = 0
    recomp_calc_idx = 0
    cache_pinned_idx = 0
    
    tot_recomp_prep_transfer_time = 0
    tot_recomp_transfer_time = 0
    tot_recomp_calc_time = 0
    tot_pinned_cache_time = 0
    

    # get corrleations from valid cudaMemcpyAsync events
    for event_idx in range(num_of_events):
        event = data['traceEvents'][event_idx]
        in_load = False
        in_store = False
        if event['name'] == 'cudaMemcpyAsync':
            for interval in load_intervals:
                if event['ts'] >= interval[0] and event['ts'] < interval[1]:
                    load_memcpy_correlations.append(event['args']['correlation'])
                    #which smart copy is this memcpyasync connecting to?
                    smart_copy_intervals = pinned_smart_copy_intervals[interval[0]]
                    if event['ts'] >= smart_copy_intervals[0][0] and event['ts'] < smart_copy_intervals[0][1]:
                        smart_load_cache_big_blocks[smart_copy_intervals[0][0]] = (smart_copy_intervals[0][1], event['args']['correlation'])
                    else:
                        smart_load_cache_big_blocks[smart_copy_intervals[1][0]] = (smart_copy_intervals[1][1], event['args']['correlation'])

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
                        recomp_prep_transfer_times[recomp_prep_idx] = (event['ts'] - interval[0], event['args']['correlation'])
                        recomp_prep_idx += 1
                        tot_recomp_prep_transfer_time += event['ts'] - interval[0]
                        break
                    elif event['ts'] < interval[0]:
                        break
        elif re_dist and event['name'] == 'pytorch_backend.py(425): mha_gen':
            # difference between of start of compute_layer and start of mha
            for interval in recomp_data_calc_intervals:
                if event['ts'] >= interval[0] and event['ts'] < interval[1]:
                    calc_time = event['ts'] - interval[0]
                    tot_recomp_calc_time += calc_time
                    recomp_calc_times[recomp_calc_idx] = calc_time
                    recomp_calc_idx += 1
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
                kv_memcpy_to_bytes[event['args']['correlation']] = event['args']['bytes']
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
                recomp_memcpy_to_bytes[event['args']['correlation']] = event['args']['bytes']
                re_loading_start_time = event['ts']
                re_loading_end_time = event['ts'] + event['dur']
                re_load_intervals.append((re_loading_start_time, re_loading_end_time, cur_re_loading_idx))
                tot_recomp_transfer_time += event['dur']
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

    used_smart_copy = {}
    for event_idx in range(num_of_events):
        # pair pinned with bytes 
        event = data['traceEvents'][event_idx]
        if event['name'] == 'aten::pin_memory':
            #check if pinned is for load_cache or load_hidden
            for interval_start in smart_load_cache_big_blocks.keys():
                (interval_end, memcpy_correlation) = smart_load_cache_big_blocks[interval_start]
                if event['ts'] >= interval_start and event['ts'] < interval_end:
                    # found which load_cache pinned is in
                    #find memcpyasync to get amount of data transferred
                    bytes_transferred = kv_memcpy_to_bytes[memcpy_correlation]
                    cache_pinned_times[cache_pinned_idx] = (event['dur'], bytes_transferred) # pinned duration, bytes transferred
                    cache_pinned_idx += 1

                    tot_pinned_cache_time += event['dur']
                    
                    used_smart_copy[interval_start] = True
                    break
    

                
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
                    
        
    cur_filename = json_filename[:-5] 
    print(f"cur_filename: {cur_filename}")

    if record_ind_events:    
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
        if not re_no_load:
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
        
        # Recomp prep time
        with open(cur_filename + '_recomp_prep.csv', 'w', newline='') as csvfile:
            fieldnames = ['idx', 'transfer prep time (s)', 'bytes transferred (B)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(recomp_prep_idx):
                cur_time_cor = recomp_prep_transfer_times[idx]
                recomp_prep_transfer_times[idx] = (cur_time_cor[0], recomp_memcpy_to_bytes[cur_time_cor[1]])
                writer.writerow({'idx': idx, 'transfer prep time (s)': recomp_prep_transfer_times[idx][0], 'bytes transferred (B)': recomp_prep_transfer_times[idx][1]})

        # Recomp calc time
        with open(cur_filename + '_recomp_calc.csv', 'w', newline='') as csvfile:
            fieldnames = ['idx', 'compute time (s)', 'bytes transferred (B)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(recomp_calc_idx):
                writer.writerow({'idx': idx, 'compute time (s)': recomp_calc_times[idx], 'bytes transferred (B)': recomp_prep_transfer_times[idx][1]})
    
    # Record Pinned Cache Times
    with open(cur_filename + '_pinned.csv', 'w', newline='') as csvfile:
        fieldnames = ['idx', 'pinned time (s)', 'bytes transferred (B)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(cache_pinned_idx):
            writer.writerow({'idx': idx, 'pinned time (s)': cache_pinned_times[idx][0], 'bytes transferred (B)': cache_pinned_times[idx][1]})
    
    
    total_loading_cache_time_gpu /= 1000000.0 # originally in microseconds (10^-6)
    total_storing_cache_time_gpu /= 1000000.0

    tot_pinned_cache_time  /= 1000000.0

    if re_dist:
        tot_recomp_prep_transfer_time /= 1000000.0 # originally in microseconds (10^-6)
        tot_recomp_transfer_time /= 1000000.0
        tot_recomp_calc_time  /= 1000000.0
    else:
        tot_recomp_prep_transfer_time = None
        tot_recomp_transfer_time = None
        tot_recomp_calc_time = None
    
    if get_cpu_time:
        total_loading_cache_time_cpu /= 1000000.0
        total_storing_cache_time_cpu /= 1000000.0
    else: 
        total_loading_cache_time_cpu = None
        total_storing_cache_time_cpu = None

    if est_bandwidth:
        total_loading_bytes /= 1000000000.0 # B --> GB
        total_storing_bytes /= 1000000000.0 # B --> GB
    else: 
        total_loading_bytes = None
        total_storing_bytes = None
    #     est_loading_bandwidth = total_loading_bytes / total_loading_cache_time_gpu # GB / s
    #     est_storing_bandwidth = total_storing_bytes / total_storing_cache_time_gpu # GB / s

    # Check if file exists to write header only once
    all_file_var = json_filename.split('-')
    cur_gbs = all_file_var[9]
    kv_gpu_percent = int(all_file_var[9])
    cur_recompute_len = all_file_var[14]
    csv_filename = json_filename.split('-percent')[0] + '-' + all_file_var[9] + '-' + all_file_var[10] + '_trace_stats.csv' # added header for recomp
    fieldnames = ['kv_gpu_percent', 'recompute_len', 'tot_loading_time_gpu (s)', 'tot_storing_time_gpu (s)', 
        'tot_loading_time_cpu (s)','tot_storing_time_cpu (s)',  'total_loading_bytes (GB)', 'total_storing_bytes (GB)',
        'tot_pinned_cache_time (s)', 'tot_recomp_prep_transfer_time (s)', 'tot_recomp_transfer_time (s)', 
        'tot_recomp_calc_time (s)']


    write_header = not os.path.exists(csv_filename)
    
    # Open the file in append mode ('a')
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({'kv_gpu_percent': kv_gpu_percent, 
                'recompute_len': cur_recompute_len,
                'tot_loading_time_gpu (s)': total_loading_cache_time_gpu, 
                'tot_storing_time_gpu (s)': total_storing_cache_time_gpu, 
                'tot_loading_time_cpu (s)': total_loading_cache_time_cpu, 
                'tot_storing_time_cpu (s)': total_storing_cache_time_cpu, 
                'total_loading_bytes (GB)': total_loading_bytes, 
                'total_storing_bytes (GB)': total_storing_bytes,
                'tot_pinned_cache_time (s)': tot_pinned_cache_time,
                'tot_recomp_prep_transfer_time (s)': tot_recomp_prep_transfer_time,
                'tot_recomp_transfer_time (s)': tot_recomp_transfer_time,
                'tot_recomp_calc_time (s)': tot_recomp_calc_time,
                })


    return total_loading_cache_time_gpu, total_storing_cache_time_gpu, total_loading_cache_time_cpu, total_storing_cache_time_cpu, total_loading_bytes, total_storing_bytes, tot_pinned_cache_time, tot_recomp_prep_transfer_time, tot_recomp_transfer_time, tot_recomp_calc_time

def save_as_csv(csv_filename, headers, data):
    # data is formatted: [idx] = {header:value}
    write_header = not os.path.exists(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for idx in range(len(data)):
            writer.writerow(data[idx])
    



def add_parser_arguments(parser):
    parser.add_argument('--files', nargs='+', help='List of files (space-separated)')
    parser.add_argument('--cpu-time',action="store_true", help="Measure the CPU data transfer time")
    parser.add_argument('--est-bandwidth', action="store_true", help="Measure the estimated bandwidth of the data transfer")
    parser.add_argument('--event-dist', action="store_true", help="Measure the distribution of the data transfer bytes")
    parser.add_argument('--split-dir', action="store_true", help="Split into bi direction and single direction data transfers")
    parser.add_argument('--recomp', action="store_true", help="Measure the distribution of data transfer for recomputation")
    parser.add_argument('--recomp-no-load', action="store_true", help="do not save load csvs")

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
        all_kv_times[batch_filename] = get_all_gpu_memcpy_correlations(str(SCRIPT_DIR / batch_filename), args.cpu_time, args.est_bandwidth, args.event_dist, args.split_dir, args.recomp, args.recomp_no_load)
        print(f"Total GPU Loading Cache Time for {batch_filename}: {all_kv_times[batch_filename][0]} s")
        print(f"Total GPU Storing Cache Time for {batch_filename}: {all_kv_times[batch_filename][1]} s")
        print(f"Total Pinned Time for {batch_filename}: {all_kv_times[batch_filename][6]} s")
        if args.cpu_time:
            print(f"Total CPU Loading Cache Time for {batch_filename}: {all_kv_times[batch_filename][2]} s")
            print(f"Total CPU Storing Cache Time for {batch_filename}: {all_kv_times[batch_filename][3]} s")
        if args.est_bandwidth:
            print(f"Total Loading Bytes for {batch_filename}: {all_kv_times[batch_filename][4]} GB")
            print(f"Total Storing Bytes (GB) for {batch_filename}: {all_kv_times[batch_filename][5]} GB")
        if args.recomp:
            print(f"Total Recomputation Loading Time for {batch_filename}: {all_kv_times[batch_filename][8]} s")
    print()
    print("GPU Load, GPU Store, CPU Load, CPU Store, Loading Bytes, Storing Bytes")
    for batch_filename in batch_filenames:
        print(f"for file {batch_filename}: {all_kv_times[batch_filename]}")
    

    
