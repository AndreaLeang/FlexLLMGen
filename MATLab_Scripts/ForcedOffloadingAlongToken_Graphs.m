close all % just as caution
clf

% % Aggregate Plot
% % For each file, plot bandwdith over data amount
folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\ForcedOffloadingAlongToken'; % Replace with your path
file_pattern = fullfile(folder_path, '*.csv');
files = dir(file_pattern);
tbl = struct2table(files);

for i=1:height(tbl)
    cur_file = string(tbl{i, "name"});
    % Read csv 
    T = readtable("ForcedOffloadingAlongToken\"+cur_file);
    all_latency = T.Latency_s_;
    all_throughput = T.Throughput_tkn_s_;
    all_GPU_energy = T.GPUEnergy_J_;
    all_token_len = T.SeqLen;

    % Clean OOM 
    nanIdx = isnan(all_latency);
    
    % Remove the identified rows 
    T_clean = T(~nanIdx, :);
    height(T)
    height(T_clean)

    % Plot 
    sgtitle('Latency, Throughput, Energy Over Sequence Length');
    hold on 
    subplot(1, 3, 1);
    bar(T_clean.SeqLen, T_clean.Latency_s_)
    hold on
    xline(2047, ':', {'No', 'Offloading'})
    xline(2576, ':', {'Offload', '1 Batch'})
    xline(3056, ':', {'Offload', '2 Batches'})
    xline(3728, ':', {'Offload', '3 Batches'})
    ylim([0, 65])
    ylabel('Latency (s)');
    xlabel('Sequence Length (Tokens)');

    subplot(1, 3, 2);
    bar(T_clean.SeqLen, T_clean.Throughput_tkn_s_)
    hold on
    xline(2047, ':', {'No', 'Offloading'})
    xline(2576, ':', {'Offload', '1 Batch'})
    xline(3056, ':', {'Offload', '2 Batches'})
    xline(3728, ':', {'Offload', '3 Batches'})
    ylabel('Throughput (Tokens/s)');
    xlabel('Sequence Length (Tokens)');

    subplot(1, 3, 3);
    bar(T_clean.SeqLen, T_clean.GPUEnergy_J_ + T_clean.CPUEnergyPKG_J_)
    hold on
    xline(2047, ':', {'No', 'Offloading'})
    xline(2576, ':', {'Offload', '1 Batch'})
    xline(3056, ':', {'Offload', '2 Batches'})
    xline(3728, ':', {'Offload', '3 Batches'})
    ylim([0, 17500])
    ylabel('Total Energy (J)');
    xlabel('Sequence Length (Tokens)');

    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "-Motivation-ForcedOffloadingAlongToken"
    saveas(gcf,"ForcedOffloadingAlongToken\" + cur_file_beg + ".png" );
    clf


    % Amount of KV Cache on GPU 
    % title('Used GPU Memory Over Sequence Length');
    T_clean.Model_Size = 12.386 * ones(height(T_clean), 1);
    T_clean.Hidden_Size = 4096 * ones(height(T_clean), 1);
    T_clean.LowerBoundFreeMemory = T_clean.CacheOnGPU_GB_; 
    % need 3 matrices, K, V, and KQ^T  KV -/2-> K (seq_len*h*num_batches_gpu) 
    % -*prompt_len/seq_len -> (prompt_len*h*num_batches_gpu)
    % -*prompt_len/hidden_size --> (prompt_len * prompt_len*num_batches_gpu)
    T_clean_fir = T_clean.SeqLen < 2590;
    T_clean_sec = T_clean.SeqLen > 2590 & T_clean.SeqLen < 3070;
    T_clean_thr = T_clean.SeqLen > 3070 & T_clean.SeqLen < 3740;
    T_clean_lst = T_clean.SeqLen > 3740;

    T_clean.LowerBoundFreeMemory(T_clean_fir, :) = T_clean.CacheOnGPU_GB_(T_clean_fir, :) ./ 4 + T_clean.CacheOnGPU_GB_(T_clean_fir, :)./4 ./2 .* (T_clean.PromptLen(T_clean_fir, :) ./ T_clean.SeqLen(T_clean_fir, :)) .* (T_clean.PromptLen(T_clean_fir, :) ./ T_clean.Hidden_Size(T_clean_fir, :)).*1.03125;
    T_clean.LowerBoundFreeMemory(T_clean_sec, :) = T_clean.CacheOnGPU_GB_(T_clean_sec, :) ./ 3  + T_clean.CacheOnGPU_GB_(T_clean_sec, :)./3 ./2 .* (T_clean.PromptLen(T_clean_sec, :) ./ T_clean.SeqLen(T_clean_sec, :)) .* (T_clean.PromptLen(T_clean_sec, :) ./ T_clean.Hidden_Size(T_clean_sec, :)).*1.03125;
    T_clean.LowerBoundFreeMemory(T_clean_thr, :) = T_clean.CacheOnGPU_GB_(T_clean_thr, :) ./ 2 + T_clean.CacheOnGPU_GB_(T_clean_thr, :)./2 ./2 .* (T_clean.PromptLen(T_clean_thr, :) ./ T_clean.SeqLen(T_clean_thr, :)) .* (T_clean.PromptLen(T_clean_thr, :) ./ T_clean.Hidden_Size(T_clean_thr, :)).*1.03125;
    T_clean.LowerBoundFreeMemory(T_clean_lst, :) = T_clean.CacheOnGPU_GB_(T_clean_lst, :) + T_clean.CacheOnGPU_GB_(T_clean_lst, :) ./2 .* (T_clean.PromptLen(T_clean_lst, :) ./ T_clean.SeqLen(T_clean_lst, :)) .* (T_clean.PromptLen(T_clean_lst, :) ./ T_clean.Hidden_Size(T_clean_lst, :)) .*1.03125;

    T_clean.CPU_KVCache = T_clean.TotalKVCache_GB_ - T_clean.CacheOnGPU_GB_;
    % bar(T_clean.SeqLen, T_clean.OccupiedGPUMemory_GB_)
    bar(T_clean.SeqLen, [T_clean.Model_Size, T_clean.HiddenSize_GB_,  T_clean.LowerBoundFreeMemory, T_clean.CacheOnGPU_GB_, T_clean.CPU_KVCache], 'stacked')
    legend('Model Weights','Hidden Layer Results','Est. Free Memory Required', 'KV Cache on GPU', 'Offloaded KV Cache')
    legend('Location', 'northwest')
    yline(39.4, "Color",'r', 'LineWidth',3, "DisplayName","Total GPU Memory")

    ylabel('Used GPU Memory (GB)', 'FontSize',14);
    xlabel('Sequence Length (Tokens)', 'FontSize',14);
    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "-Motivation-MaxKVCacheAlongToken"
    saveas(gcf,"ForcedOffloadingAlongToken\" + cur_file_beg + ".png" );
    clf


    % CAL: Combined Graph
    tiledlayout(2,1);

    nexttile(1, [1,1])
    bar(T_clean.SeqLen, [T_clean.Model_Size, T_clean.HiddenSize_GB_,  T_clean.LowerBoundFreeMemory, T_clean.CacheOnGPU_GB_, T_clean.CPU_KVCache], 'stacked')
    legend('off')
    yline(39.4, "Color",'r', 'LineWidth',3, "DisplayName","Total GPU Memory")

    ylabel('Used GPU Memory (GB)');
    xlabel('Sequence Length (Tokens)');

    nexttile(2)
    bar(T_clean.SeqLen, T_clean.Throughput_tkn_s_)
    hold on
    xline(2047, ':', {'No', 'Offload'})
    xline(2576, ':', {'Offload', '1 Block'})
    xline(3056, ':', {'Offload', '2 Blocks'})
    xline(3728, ':', {'Offload', '3 Blocks'})
    ylabel('Throughput (Tokens/s)');
    xlabel('Sequence Length (Tokens)');

    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "-Motivation-CAL-Combined"
    saveas(gcf,"ForcedOffloadingAlongToken\" + cur_file_beg + ".png" );
    clf

    % CAL - Ind decode throughput
    bar(T_clean.SeqLen, T_clean.Throughput_tkn_s_)
    hold on
    xline(2047, ':', {'No', 'Offload'})
    xline(2576, ':', {'Offload', '1 Block'})
    xline(3056, ':', {'Offload', '2 Blocks'})
    xline(3728, ':', {'Offload', '3 Blocks'})
    ylabel('Throughput (Tokens/s)');
    xlabel('Sequence Length (Tokens)');


    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "-Motivation-CAL-DecodeThroughput"
    saveas(gcf,"ForcedOffloadingAlongToken\" + cur_file_beg + ".png" );
    

end 

