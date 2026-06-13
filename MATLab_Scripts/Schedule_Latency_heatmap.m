close all % just as caution
clf

% % Aggregate Plot
% % For each file, plot bandwdith over data amount
folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\LatencyHeatmap\'; % Replace with your path
file_pattern = fullfile(folder_path, '*.csv');
files = dir(file_pattern);
tbl = struct2table(files);

for i=1:height(tbl)
    cur_file = string(tbl{i, "name"})
    % Read csv 
    T = readtable("LatencyHeatmap\" + cur_file);
    all_latency = T.EstimatedLatency_s_;
    all_block_size = T.Batch_size;
    all_recomp_len = T.RecomputeLen;

    % Clean OOM 
    nanIdx = isnan(all_latency);
    
    % Remove the identified rows 
    T_clean = T(~nanIdx, :);
    height(T)
    height(T_clean)

    % Plot Experiment
    h = heatmap(T_clean, "Batch_size", "RecomputeLen", "ColorVariable","Latency_s_", 'CellLabelFormat','%0.2f')
    h.ColorLimits = [25 65]; 
    colormap(nebula(7))
    title('')
    % title('Latency Across Recomputation and Block Size Combinations');
    ylabel('\fontsize{14}Recompute Len (tokens)');
    xlabel('\fontsize{14}Block Size');

    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_Act_Latency_Heatmap"
    saveas(gcf,"LatencyHeatmap\" + cur_file_beg + ".png" );
    clf

    % Plot Estimate
    h = heatmap(T_clean, "Batch_size", "RecomputeLen", "ColorVariable","EstimatedLatency_s_", 'CellLabelFormat','%0.2f')
    h.ColorLimits = [25 65]; 
    colormap(nebula(7))
    title('')
    % title('Latency Across Recomputation and Block Size Combinations');
    ylabel('\fontsize{14}Recompute Len (tokens)');
    xlabel('\fontsize{14}Block Size');

    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_Est_Latency_Heatmap"
    saveas(gcf,"LatencyHeatmap\" + cur_file_beg + ".png" );
    clf

    % Plot Lat Diff
    % h = heatmap(T_clean, "Batch_size", "RecomputeLen", "ColorVariable","DifferenceLatency___", 'CellLabelFormat','%0.2f') %exp and log
    h = heatmap(T_clean, "Batch_size", "RecomputeLen", "ColorVariable","DifferenceLatencyWTotal___", 'CellLabelFormat','%0.2f')
    h.ColorLimits = [0 100]; 
    colors = [9/255, 122/255, 14/255;   % Green
              250/255, 245/255, 155/255;   % Yellow
              186/255, 19/255, 0/255];  % Red
    
    % Interpolate to create a 256-color map
    cMap = interp1(linspace(0, 1, size(colors, 1)), colors, linspace(0, 1, 256));
    colormap(cMap);
    title('')
    % title('Model Latency Difference (%) Across Recomputation and Block Size Combinations');
    ylabel('\fontsize{14}Recompute Len (tokens)');
    xlabel('\fontsize{14}Block Size');

    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_Latency_PerDiff_Heatmap"
    saveas(gcf,"LatencyHeatmap\" + cur_file_beg + ".png" );
    clf

    % Plot Percent of Estimated Latency due to Data Transfer
    h = heatmap(T_clean, "Batch_size", "RecomputeLen", "ColorVariable","EstLatTransferPercent___", 'CellLabelFormat','%3.0f')
    h.ColorLimits = [0 100]; 
    colors = [9/255, 122/255, 14/255;   % Green
              250/255, 245/255, 155/255;   % Yellow
              186/255, 19/255, 0/255];  % Red
    % Interpolate to create a 256-color map
    cMap = interp1(linspace(0, 1, size(colors, 1)), colors, linspace(0, 1, 256));
    colormap(cMap);
    % title('Percent of Estimated Latency due to Data Transfer')
    title('');
    ylabel('\fontsize{14}Recompute Len (tokens)');
    xlabel('\fontsize{14}Block Size');

    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_Est_Latency_DataTransferPer_Heatmap"
    saveas(gcf,"LatencyHeatmap\" + cur_file_beg + ".png" );
    clf

    % Plot Latency and Energy Together
    T_clean.LatRound = round(T_clean.Latency_s_, 2);
    T_clean.EnergyRound = round(T_clean.TotalEnergy_J_, 0);
    T_clean.DisplayLabel = num2str(T_clean.LatRound) + "/" + num2str(T_clean.EnergyRound);
    
    unique_B = unique(T_clean.Batch_size);
    unique_R = unique(T_clean.RecomputeLen);
    
    numRows = numel(unique_R);
    numCols = numel(unique_B);

    colorGrid = NaN(numRows, numCols);
    labelGrid = strings(numRows, numCols);

    % 4. Populate grids by directly matching rows from Table T
    for j = 1:height(T_clean)
        % Find the grid coordinates (row/col indices) for the current B and R values
        r_idx = find(unique_R == T_clean.RecomputeLen(j));
        c_idx = find(unique_B == T_clean.Batch_size(j));
        
        if ~isempty(r_idx) && ~isempty(c_idx)
            colorGrid(r_idx, c_idx) = T_clean.Latency_s_(j);
            labelGrid(r_idx, c_idx) = T_clean.DisplayLabel(j); % Use T.E(i) if it's already a string array
        end
    end

    
    % 4. Plot using imagesc
    figure;
    imagesc(colorGrid); 
    colorbar;
    colormap(nebula(7));
    clim([25, 65]);
    ax = gca;
         
    % 5. Format axes ticks and labels 
    ax.XTick = 1:numel(unique_B);
    ax.XTickLabel = string(unique_B);
    ax.YTick = 1:numel(unique_R);
    ax.YTickLabel = string(unique_R);
    ax.TickLength = [0, 0];
    
    ylabel('\fontsize{14}Recompute Len (tokens)');
    xlabel('\fontsize{14}Block Size');

    for col = 0.5 : 1 : (numCols + 0.5)
        line([col, col], [0.5, numRows + 0.5], 'Color', 'black', 'LineWidth', 0.5);
    end
    % Draw horizontal line dividers
    for row = 0.5 : 1 : (numRows + 0.5)
        line([0.5, numCols + 0.5], [row, row], 'Color', 'black', 'LineWidth', 0.5);
    end
    
    % 6. Overlay the custom E string labels inside the cells
    [numRows, numCols] = size(colorGrid);
    for row = 1:numRows
        for col = 1:numCols
            labelText = labelGrid(row, col);
            
            % Only print if there is data in the cell
            if ~ismissing(labelText) && labelText ~= ""
                text(col, row, labelText, ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'Color', 'white');
            end
        end
    end

    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_CAL_Lat_Energy_Heatmap";
    saveas(gcf,"LatencyHeatmap\" + cur_file_beg + ".png" );
    clf


end 
