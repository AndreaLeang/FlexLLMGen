close all % just as caution
clf

% % Aggregate Plot
% % For each file, plot bandwdith over data amount
folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\OptimalCompComponent'; % Replace with your path
file_pattern = fullfile(folder_path, '*.csv');
files = dir(file_pattern);
tbl = struct2table(files);

for i=1:height(tbl)
    cur_file = string(tbl{i, "name"})
    % Read csv 
    T = readtable("OptimalCompComponent\" + cur_file);
    all_PromptLen = T.PromptLen;

    % Clean OOM 
    nanIdx = isnan(all_PromptLen);
    
    % Remove the identified rows 
    T_clean = T(~nanIdx, :);
    height(T)
    height(T_clean)

    % GPU and CPU Trends 
    T_clean.Workload = num2str(T_clean.PromptLen) + "\_" + num2str(T_clean.NumOfPrompts)
    cur_partition_var = T_clean.Model;
    [G,ID] = findgroups(T_clean{:, "Model"})
    ID(1) = {'opt-2.7b'}
    ID(2) = {'opt-6.7b'}
    ID(3) = {'opt-13b'}
    tcl = tiledlayout(2,3);

    for i=1:height(ID) 
        cur_data_idx = strcmp(cur_partition_var, ID(i));
        T_par = T_clean(cur_data_idx, :);
        nexttile(tcl)

        
        comb_data = [T_par.FREnergy, T_par.all_iEnergy, T_par.i_CEnergy, T_par.LEnergy, T_par.REnergy];
        bar(T_par.PromptLen, comb_data)
        hold on
        cur_title = "Model: " + ID(i);
        title(cur_title)
        if i==3
            lg = legend({'FR', 'All Ideal', 'I_{C}', 'L', '10% R'}); 
            lg.Location = 'northeastoutside';
        end
        % ax = gca; % axes handle
        % ax.YAxis.Exponent = 0;
        ylim([0 25000])
        xtickangle(45)
        if i==1
            ylabel('\fontsize{12}Total Energy Usage (J)');
        end
        xlabel('\fontsize{10}Prompt Length (tokens)');
    end
        
    

    for i=1:height(ID) 
        cur_data_idx = strcmp(cur_partition_var, ID(i));
        T_par = T_clean(cur_data_idx, :);
        nexttile(tcl)

        
        comb_data = [T_par.FRLatency, T_par.all_iLatency, T_par.i_CLatency, T_par.LLatency, T_par.RLatency];
        bar(T_par.PromptLen, comb_data)
        hold on
        cur_title = "Model: " + ID(i);
        title(cur_title)
        ylim([0 80])
        xtickangle(45)
        if i == 1
            ylabel('\fontsize{12}Latency (s)');
        end
        xlabel('\fontsize{10}Prompt Length (tokens)');
    end
        
    
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_all_solution_comp_latency";
    saveas(gcf,"OptimalCompComponent\" + cur_file_beg + ".png" );
    clf

end 
