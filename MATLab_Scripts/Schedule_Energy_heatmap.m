close all % just as caution
clf

% % Aggregate Plot
% % For each file, plot bandwdith over data amount
folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\EnergyPattern'; % Replace with your path
file_pattern = fullfile(folder_path, '*.csv');
files = dir(file_pattern);
tbl = struct2table(files);

for i=1:height(tbl)
    cur_file = string(tbl{i, "name"})
    % Read csv 
    T = readtable("EnergyPattern\" + cur_file);
    all_energy = T.EstimatedGPUEnergy_J_;
    all_block_size = T.Batch_size;
    all_recomp_len = T.RecomputeLen;

    % Clean OOM 
    nanIdx = isnan(all_energy);
    
    % Remove the identified rows 
    T_clean = T(~nanIdx, :);
    height(T)
    height(T_clean)

    % Plot 
    % GPU Energy
    h = heatmap(T_clean, "Batch_size", "RecomputeLen", "ColorVariable","TotalGPUEnergy_J_")
    % h.ColorLimits = [3000 6000]; 
    colormap(nebula(7))
    title('')
    % title('GPU Energy Across Recomputation and Block Size Combinations');
    ylabel('\fontsize{14}Recompute Len (tokens)');
    xlabel('\fontsize{14}Block Size');
    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_GPU_Energy_Heatmap"
    saveas(gcf,"EnergyPattern\" + cur_file_beg + ".png" );
    clf

    % CPU Energy
    h = heatmap(T_clean, "Batch_size", "RecomputeLen", "ColorVariable","TotalCPUEnergy_J_", 'CellLabelFormat','%5.0f')
    % h.ColorLimits = [30 70];g
    colormap(nebula(7))
    title('');
    % title('CPU Energy Across Recomputation and Block Size Combinations');
    ylabel('\fontsize{14}Recompute Len (tokens)');
    xlabel('\fontsize{14}Block Size');
    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_CPU_Energy_Heatmap"
    saveas(gcf,"EnergyPattern\" + cur_file_beg + ".png" );
    clf

    % Plot 
    % GPU+CPU Energy
    h = heatmap(T_clean, "Batch_size", "RecomputeLen", "ColorVariable","TotalEnergy_J_",'CellLabelFormat','%5.0f')
    % h.ColorLimits = [30 70]; 
    colormap(nebula(7))
    title('')
    % title('Total Energy Across Recomputation and Block Size Combinations');
    ylabel('\fontsize{14}Recompute Len (tokens)');
    xlabel('\fontsize{14}Block Size');
    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_Tot_Energy_Heatmap"
    saveas(gcf,"EnergyPattern\" + cur_file_beg + ".png" );
    clf

     % Plot 
    % Est GPU Energy
    h = heatmap(T_clean, "Batch_size", "RecomputeLen", "ColorVariable","EstimatedGPUEnergy_J_")
    % h.ColorLimits = [3000 6000]; 
    colormap(nebula(7))
    title('')
    % title('GPU Energy Across Recomputation and Block Size Combinations');
    ylabel('\fontsize{14}Recompute Len (tokens)');
    xlabel('\fontsize{14}Block Size');
    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_est_GPU_Energy_Heatmap"
    saveas(gcf,"EnergyPattern\" + cur_file_beg + ".png" );
    clf
end 
