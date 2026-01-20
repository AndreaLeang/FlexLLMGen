import json
import argparse

from pathlib import Path
# import matplotlib.pyplot as plt

def get_all_ac2g(json_filename):
    # open json file
    with open(json_filename, 'r') as file:
        # load json file
        data = json.load(file)
    
    # initialize the dictionary to hold the ac2g
    ac2g_dict = {}
    total_loading_cache_time = 0
    total_storing_cache_time = 0

    load_correlations = {} # id: duration
    store_correlations = {} # id:duration

    loading_cache_start_time = float('inf')
    loading_cache_end_time = float('-inf')
    storing_cache_start_time = float('inf')
    storing_cache_end_time = float('-inf')

    # for each item in the result of the "traceEvents" key,
    num_of_events = len(data['traceEvents'])

    for event_idx in range(num_of_events):
        # find the cudaMemcpyAsync events under load_cache and store_cache
        # then find the ac2g connecting that to 
        #   Load_Cache: MemcpyHtoD (Pinned -> Device)
        #   Store_cache: MemcpyDtoH (Device -> Pageable)

        # (each item should be a dictionary in of it itself)
        event = data['traceEvents'][event_idx]

        # verify if we are currently loading or storing cache
        if event['name'] == 'flex_opt.py(640): load_cache':
            print("Found load_cache")
            # event name may change with edits to flex_opt.py (line number)
            loading_cache_start_time = event['ts']
            loading_cache_end_time = event['ts'] + event['dur']
            load_correlations = {}

        elif event['name'] == 'flex_opt.py(662): store_cache':
            # event name may change with edits to flex_opt.py (line number)
            storing_cache_start_time = event['ts']
            storing_cache_end_time = event['ts'] + event['dur']
            store_correlations = {}
        else:
            # start looking
            if event['ts'] >= loading_cache_start_time and event['ts'] <= loading_cache_end_time:
                # inside load_cache
                # look for the cudamemcpy --> find the HtoD connected to it (can be multiple)
                # remember to look for matching correlation --> HtoD can appear before the cudamemcpyAsync***
                print(f"found event {event['name']} in load_cache")
                if event['name'] == 'cudaMemcpyAsync':
                    print("Found cudaMemcpyAsync")
                    cur_correlation_id = event['args']['correlation_id']
                    if cur_correlation_id in load_correlations:
                        total_loading_cache_time += event['dur']
                    else:
                        load_correlations[cur_correlation_id] = 0.0

                elif event['name'] == 'Memcpy HtoD (Pinned -> Device)':
                    print("Found Memcpy HtoD (Pinned -> Device)")
                    cur_correlation_id = event['args']['correlation_id']
                    if cur_correlation_id in load_correlations:
                        total_loading_cache_time += event['dur']
                    else:
                        load_correlations[cur_correlation_id] = event['dur']
                    

            elif event['ts'] >= storing_cache_start_time and event['ts'] < storing_cache_end_time:
                #inside store_cache
                # look for the cudamemcpy --> find the DtoH connected to it (can be multiple)
                if event['name'] == 'cudaMemcpyAsync':
                    cur_correlation_id = event['args']['correlation_id']
                    if cur_correlation_id in store_correlations:
                        total_storing_cache_time += event['dur']
                    else:
                        store_correlations[cur_correlation_id] = 0.0
                elif event['name'] == 'MemcpyDtoH (Device -> Pageable)':
                    cur_correlation_id = event['args']['correlation_id']
                    if cur_correlation_id in store_correlations:
                        total_storing_cache_time += event['dur']
                    else:
                        store_correlations[cur_correlation_id] = event['dur']
        

    # total_loading_cache_time /= 1000.0
    # total_storing_cache_time /= 1000.0

    return total_loading_cache_time, total_storing_cache_time # in microseconds


def get_all_memcpy_correlations(json_filename):
    # open json file
    with open(json_filename, 'r') as file:
        # load json file
        data = json.load(file)
    
    # initialize the dictionary to hold the ac2g
    ac2g_dict = {}
    total_loading_cache_time = 0
    total_storing_cache_time = 0

    load_intervals = []
    store_intervals = []

    # for each item in the result of the "traceEvents" key,
    num_of_events = len(data['traceEvents'])

    # get load and store intervals
    for event_idx in range(num_of_events):
        event = data['traceEvents'][event_idx]

        if event['name'] == 'flex_opt.py(640): load_cache':
            loading_cache_start_time = event['ts']
            loading_cache_end_time = event['ts'] + event['dur']
            load_intervals.append((loading_cache_start_time, loading_cache_end_time))
        elif event['name'] == 'flex_opt.py(662): store_cache':
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
        if event['name'] == 'cudaMemcpyAsync':
            # print(f"found cudaMemcpyAsync")
            for interval in load_intervals:
                # print(f"checking interval {interval}")
                if event['ts'] >= interval[0] and event['ts'] < interval[1]:
                    load_memcpy_correlations.append(event['args']['correlation'])
                    break
                elif event['ts'] < interval[0]:
                    break
            
            for interval in store_intervals:
                if event['ts'] >= interval[0] and event['ts'] < interval[1]:
                    store_memcpy_correlations.append(event['args']['correlation'])
                    break
                elif event['ts'] < interval[0]:
                    break

    # print(f"load_memcpy_correlations: {load_memcpy_correlations}")
    # print(f"store_memcpy_correlations: {store_memcpy_correlations}")

    # get the actual GPU memcpy durations?
    for event_idx in range(num_of_events):
        # find the cudaMemcpyAsync events under load_cache and store_cache
        # then find the ac2g connecting that to 
        #   Load_Cache: MemcpyHtoD (Pinned -> Device)
        #   Store_cache: MemcpyDtoH (Device -> Pageable)

        # (each item should be a dictionary in of it itself)
        event = data['traceEvents'][event_idx]
        if event['name'] == 'Memcpy HtoD (Pinned -> Device)':
            if event['args']['correlation'] in load_memcpy_correlations:
                total_loading_cache_time += event['dur']
        elif event['name'] == 'Memcpy DtoH (Device -> Pageable)':
            if event['args']['correlation'] in store_memcpy_correlations:
                total_storing_cache_time += event['dur']

    total_loading_cache_time /= 1000000.0 # originally in microseconds (10^-6)
    total_storing_cache_time /= 1000000.0

    return total_loading_cache_time, total_storing_cache_time # in s


def add_parser_arguments(parser):
    parser.add_argument('--files', nargs='+', help='List of files (space-separated)')


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent.absolute()
    # traces_dir = SCRIPT_DIR / "traces"
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)

    args = parser.parse_args()

    batch_filenames = args.files
    
    batch_tklqt = []
    # batch_tklqt = [12014.2443359375, 18471.943251953126, 49995.56291894531, 104777.457796875, 232505.366203125, 461461.98833007814]
    for batch_filename in batch_filenames:
        print(f"Processing {batch_filename}")
        batch_tklqt.append(get_all_memcpy_correlations(str(SCRIPT_DIR / batch_filename)))
        print(f"Total Loading Cache Time for {batch_filename}: {batch_tklqt[-1][0]} s")
        print(f"Total Storing Cache Time for {batch_filename}: {batch_tklqt[-1][1]} s")
    

    # total_tkqlt = get_all_ac2g(str(BERT_DIR / json_filename))
    # print(f"Total TKQLT: {total_tkqlt} ms")
    
