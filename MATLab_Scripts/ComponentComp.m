close all % just as caution
clf

% % Aggregate Plot
% % For each file, plot bandwdith over data amount
folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\ComponentComp'; % Replace with your path
file_pattern = fullfile(folder_path, '*.csv');
files = dir(file_pattern);
tbl = struct2table(files);

for i=1:height(tbl)
    cur_file = string(tbl{i, "name"})
    % Read csv 
    T = readtable("ComponentComp\" + cur_file);
    all_latency = T.ActualLatency_s_;

    % Clean OOM 
    nanIdx = isnan(all_latency);
    
    % Remove the identified rows 
    T_clean = T(~nanIdx, :);
    height(T)
    height(T_clean)

    names = T_clean.Properties.VariableNames;
    lat_names = contains(names, "Latency");
    names = names(lat_names)

    for cur_name_ind = 2:numel(names)
        col_name = names{cur_name_ind}
        file_addition = erase(col_name, "Latency_s_")

        % Plot Experiment
        h = heatmap(T_clean, "blockSize", "RecomputeLength", "ColorVariable",col_name, 'CellLabelFormat','%0.3f')
        h.ColorLimits = [25 65]; 
        colormap(nebula(7))
        title('')
        % title('Latency Across Recomputation and Block Size Combinations');
        ylabel('\fontsize{14}Recompute Len (tokens)');
        xlabel('\fontsize{14}Block Size');
    
        % save
        cur_file_beg = file_addition +"_Lat_Heatmap"
        saveas(gcf,"ComponentComp\" + cur_file_beg + ".png" );
        clf
    
        T_clean.Diff = (abs(T_clean.ActualLatency_s_ - T_clean.(col_name)) ./ T_clean.ActualLatency_s_) * 100
        % Plot Percent of Estimated Latency due to Data Transfer
        h = heatmap(T_clean, "blockSize", "RecomputeLength", "ColorVariable","Diff", 'CellLabelFormat','%2.2f')
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
        cur_file_beg =  file_addition + "_Err_Heatmap"
        saveas(gcf,"ComponentComp\" + cur_file_beg + ".png" );
        clf
    end
end 
