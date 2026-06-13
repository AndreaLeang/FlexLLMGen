close all % just as caution
clf

% % Aggregate Plot
% % For each file, plot bandwdith over data amount
folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\OptimalComp'; % Replace with your path
file_pattern = fullfile(folder_path, '*.csv');
files = dir(file_pattern);
tbl = struct2table(files);

for i=1:height(tbl)
    cur_file = string(tbl{i, "name"})
    % Read csv 
    T = readtable("OptimalComp\" + cur_file);
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
    tcl = tiledlayout(1,3);

    for i=1:height(ID) 
        cur_data_idx = strcmp(cur_partition_var, ID(i));
        T_par = T_clean(cur_data_idx, :);
        nexttile(tcl)

        
        comb_data = [T_par.FREnergy, T_par.LEnergy, T_par.GEEnergy];
        bar(T_par.PromptLen, comb_data)
        hold on
        cur_title = "Model: " + ID(i);
        title(cur_title)
        lg = legend({'FR', 'L', 'GE'}); 
        lg.Location = 'northeast';
        % ax = gca; % axes handle
        ylim([0 45000])
        xtickangle(45)
        ylabel('\fontsize{14}Total Energy Usage (J)');
        xlabel('\fontsize{14}Prompt Length (tokens)');
    end
        
    
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_all_solution_comp";
    saveas(gcf,"OptimalComp\" + cur_file_beg + ".png" );
    clf

end 
