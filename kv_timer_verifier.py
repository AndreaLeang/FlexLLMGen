import json
import argparse

from pathlib import Path
# import matplotlib.pyplot as plt

def get_all_gpu_memcpy_correlations(json_filename, get_cpu_time=False, est_bandwidth=False, record_ind_events=False):
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

    # for each item in the result of the "traceEvents" key,
    num_of_events = len(data['traceEvents'])

    # get load and store intervals
    for event_idx in range(num_of_events):
        event = data['traceEvents'][event_idx]

        if event['name'] == 'flex_opt.py(647): load_cache':
            loading_cache_start_time = event['ts']
            loading_cache_end_time = event['ts'] + event['dur']
            load_intervals.append((loading_cache_start_time, loading_cache_end_time))
        elif event['name'] == 'flex_opt.py(669): store_cache':
            storing_cache_start_time = event['ts']
            storing_cache_end_time = event['ts'] + event['dur']
            store_intervals.append((storing_cache_start_time, storing_cache_end_time))

    # print(f"load_intervals: {load_intervals}")
    # print(f"store_intervals: {store_intervals}")

    load_memcpy_correlations = []
    store_memcpy_correlations = []

    # get corrleations from valid cudaMemcpyAsync events
    for event_idx in range(num_of_events):
        event = data['traceEvents'][event_idx]
        in_load = False
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
                        break
                    elif event['ts'] < interval[0]:
                        break

    # get the actual GPU memcpy durations

    if record_ind_events:
        loading_events = {}
        storing_events = {}
        cur_loading_idx = 0
        cur_storing_idx = 0
        

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
                if record_ind_events:
                    loading_events[cur_loading_idx] = (event['args']['bytes'], event['args']['memory bandwidth (GB/s)'])
                    cur_loading_idx += 1
        elif event['name'] == 'Memcpy DtoH (Device -> Pageable)':
            if event['args']['correlation'] in store_memcpy_correlations:
                total_storing_cache_time_gpu += event['dur']
                if est_bandwidth: 
                    total_storing_bytes += event['args']['bytes']
                if record_ind_events:
                    storing_events[cur_storing_idx] = (event['args']['bytes'], event['args']['memory bandwidth (GB/s)'])
                    cur_storing_idx += 1

    if record_ind_events:
        cur_filename = json_filename.split('.')[0] + '.csv'
        print(f"cur_filename: {cur_filename}")
        with open(cur_filename, 'w', newline='') as csvfile:
            fieldnames = ['idx', 'data (GB)', 'bandwidth (GB/s)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for idx in range(cur_loading_idx):
                writer.writerow({'idx': idx, 'data': loading_events[idx][0] / 1000000000.0, 'bandwidth': loading_events[idx][1]})
            for idx in range(cur_storing_idx):
                writer.writerow({'idx': idx, 'data': storing_events[idx][0] / 1000000000.0, 'bandwidth': storing_events[idx][1]})

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

    return total_loading_cache_time_gpu, total_storing_cache_time_gpu, total_loading_cache_time_cpu, total_storing_cache_time_cpu, total_loading_bytes, total_storing_bytes # in s

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
        all_kv_times[batch_filename] = get_all_gpu_memcpy_correlations(str(SCRIPT_DIR / batch_filename), args.cpu_time, args.est_bandwidth, args.event_dist)
        print(f"Total GPU Loading Cache Time for {batch_filename}: {all_kv_times[batch_filename][0]} s")
        print(f"Total GPU Storing Cache Time for {batch_filename}: {all_kv_times[batch_filename][1]} s")
        if args.cpu_time:
            print(f"Total CPU Loading Cache Time for {batch_filename}: {all_kv_times[batch_filename][2]} s")
            print(f"Total CPU Storing Cache Time for {batch_filename}: {all_kv_times[batch_filename][3]} s")
        if args.est_bandwidth:
            print(f"Total Loading Bytes for {batch_filename}: {all_kv_times[batch_filename][4]} GB")
            print(f"Total Storing Bytes (GB) for {batch_filename}: {all_kv_times[batch_filename][5]} GB")
    print()
    print("GPU Load, GPU Store, CPU Load, CPU Store, Loading Bytes, Storing Bytes")
    for batch_filename in batch_filenames:
        print(f"for file {batch_filename}: {all_kv_times[batch_filename]}")
    

    
