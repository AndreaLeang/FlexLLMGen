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
    all_energy = T.TotalGPUEnergy_J_;
    all_block_size = T.Batch_size;
    all_recomp_len = T.RecomputeLen;

    % Clean OOM 
    nanIdx = isnan(all_energy);
    
    % Remove the identified rows 
    T_clean = T(~nanIdx, :);
    height(T)
    height(T_clean)

    % GPU and CPU Trends 
    cur_partition_var = T_clean.Batch_size;
    [G,ID] = findgroups(T_clean{:, "Batch_size"})
    tcl = tiledlayout(2,2);
    for i=1:height(ID) 
        cur_data_idx = cur_partition_var == ID(i);
        T_par = T_clean(cur_data_idx, :);
        nexttile(tcl)
        comb_data = [T_par.TotalGPUEnergy_J_, T_par.TotalCPUEnergy_J_]
        single_bar = bar(T_par.RecomputeLen, comb_data);
        ylabel('Energy (J)');
        
        cur_title = "Block Size " + num2str(ID(i))

        title(cur_title)
        if i == 2
            lg = legend({'GPU Energy', 'CPU Energy'}); 
            lg.Location = 'northeastoutside';
        end
        hold on
        xlabel('Recomputation Length (tokens)');
        set(gca,'YColor','k');
    end
    


    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_Energy_over_Recomp_All";
    saveas(gcf,"EnergyPattern\" + cur_file_beg + ".png" );
    clf

    % CPU and Latency Trends 
    cur_partition_var = T_clean.Batch_size;
    [G,ID] = findgroups(T_clean{:, "Batch_size"})
    tcl = tiledlayout(2,2);
    for i=1:height(ID) 
        cur_data_idx = cur_partition_var == ID(i);
        T_par = T_clean(cur_data_idx, :);
        nexttile(tcl)
        nil = 0 * T_par.TotalGPUEnergy_J_;
        comb_data = [T_par.TotalCPUEnergy_J_, nil]
        single_bar = bar(T_par.RecomputeLen, comb_data, 'grouped');
        ylabel('Energy (J)');
        hold on 

        yyaxis right
        
        b = bar(T_par.RecomputeLen, [nil,  T_par.Latency_s_],'grouped')
        b(2).FaceColor = 'flat'; % Required to allow individual CData mapping
        b(2).CData = [221/255 84/255 0]; 
        ylabel('Latency (s)');
        
        cur_title = "Block Size " + num2str(ID(i))

        title(cur_title)
        if i == 2
            lg = legend([single_bar(1), b(2)], {'CPU Energy', 'Latency'});
            lg.Location = 'northeastoutside';
        end
        hold on
        xlabel('Recomputation Length (tokens)');
        set(gca,'YColor','k');
    end
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_CPU_Energy_and_Latency";
    saveas(gcf,"EnergyPattern\" + cur_file_beg + ".png" );
    clf

    % GPU and Latency Trends 
    cur_partition_var = T_clean.Batch_size;
    [G,ID] = findgroups(T_clean{:, "Batch_size"})
    tcl = tiledlayout(2,2);
    for i=1:height(ID) 
        cur_data_idx = cur_partition_var == ID(i);
        T_par = T_clean(cur_data_idx, :);
        nexttile(tcl)
        nil = 0 * T_par.TotalGPUEnergy_J_;
        comb_data = [T_par.TotalGPUEnergy_J_, nil]
        single_bar = bar(T_par.RecomputeLen, comb_data, 'grouped');
        ylabel('Energy (J)');
        hold on 

        yyaxis right
        
        b = bar(T_par.RecomputeLen, [nil,  T_par.Latency_s_],'grouped')
        b(2).FaceColor = 'flat'; % Required to allow individual CData mapping
        b(2).CData = [221/255 84/255 0]; 
        ylabel('Latency (s)');
        
        cur_title = "Block Size " + num2str(ID(i))

        title(cur_title)
        if i == 2
            lg = legend([single_bar(1), b(2)], {'GPU Energy', 'Latency'});
            lg.Location = 'northeastoutside';
        end
        hold on
        xlabel('Recomputation Length (tokens)');
        set(gca,'YColor','k');
    end
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_GPU_Energy_and_Latency";
    saveas(gcf,"EnergyPattern\" + cur_file_beg + ".png" );
    clf

    % Transfer and Active Energy Percentages
    tcl = tiledlayout(2,2);
    for i=1:height(ID) 
        cur_data_idx = cur_partition_var == ID(i);
        T_par = T_clean(cur_data_idx, :);
        nexttile(tcl)
        T_par.TrueGPUTransferPer = T_par.EstGPUTransferPercent___ .* T_par.EstimatedGPUEnergy_J_ ./ T_par.TotalGPUEnergy_J_;
        T_par.TrueGPUActivePer = T_par.EstGPUActivePercent___ .* T_par.EstimatedGPUEnergy_J_ ./ T_par.TotalGPUEnergy_J_;

        % Percentages Difference
        comb_data = [T_par.TrueGPUTransferPer, T_par.TrueGPUActivePer]
        bar(T_par.RecomputeLen, comb_data)
        hold on
        cur_title = "Block Size " + num2str(ID(i))
        title(cur_title)
        if i == 2
            lg = legend({'Transfer', 'Active'}); 
            lg.Location = 'northeastoutside';
        end
        ylabel('% of Experienced GPU Energy ');
        ylim([0, 100])
        xlabel('Recomputation Length (tokens)');
        
    end
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_Percent_Breakdown_All";
    saveas(gcf,"EnergyPattern\" + cur_file_beg + ".png" );
    clf

    % CPU and Latency + GPU + Total Energy Trends 
    cur_partition_var = T_clean.Batch_size;
    
    [G,ID] = findgroups(T_clean{:, "Batch_size"})
    tcl = tiledlayout(2,2);
    for i=1:height(ID) 
        cur_data_idx = cur_partition_var == ID(i);
        T_par = T_clean(cur_data_idx, :);
        nexttile(tcl)
        nil = 0 * T_par.TotalGPUEnergy_J_;
        comb_data = [T_par.TotalCPUEnergy_J_, T_par.TotalGPUEnergy_J_, T_par.TotalEnergy_J_, nil]
        single_bar = bar(T_par.RecomputeLen, comb_data, 'grouped');
        ylabel('Energy (J)');
        hold on 

        yyaxis right
        
        b = bar(T_par.RecomputeLen, [nil, nil, nil, T_par.Latency_s_],'grouped')
        b(2).FaceColor = 'flat'; % Required to allow individual CData mapping
        b(2).CData = [57/255 167/255 48/255]; 
        ylabel('Latency (s)');
        
        cur_title = "Block Size " + num2str(ID(i))

        title(cur_title)
        if i == 2
            lg = legend([single_bar(1), single_bar(2), single_bar(3), b(2)], {'CPU Energy','GPU Energy','Total Energy', 'Latency'});
            lg.Location = 'northoutside';
        end
        hold on
        xlabel('Recomputation Length (tokens)');
        set(gca,'YColor','k');
    end
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_CPU_Energy_and_Latency_and_GPU_and_Total";
    saveas(gcf,"EnergyPattern\" + cur_file_beg + ".png" );
    clf

    % Single: CPU and Latency + GPU + Total Energy Trends 
    cur_partition_var = T_clean.Batch_size;
    
    [G,ID] = findgroups(T_clean{:, "Batch_size"})
    cur_data_idx = cur_partition_var == ID(3);
    T_par = T_clean(cur_data_idx, :);
    nil = 0 * T_par.TotalGPUEnergy_J_;
    comb_data = [T_par.TotalCPUEnergy_J_, T_par.TotalGPUEnergy_J_, T_par.TotalEnergy_J_, nil]
    single_bar = bar(T_par.RecomputeLen, comb_data, 'grouped');
    ylabel('Energy (J)');
    hold on 

    yyaxis right
    
    b = bar(T_par.RecomputeLen, [nil, nil, nil, T_par.Latency_s_],'grouped')
    b(2).FaceColor = 'flat'; % Required to allow individual CData mapping
    b(2).CData = [57/255 167/255 48/255]; 
    ylabel('Latency (s)');
    
    lg = legend([single_bar(1), single_bar(2), single_bar(3), b(2)], {'CPU Energy','GPU Energy','Total Energy', 'Latency'});
    lg.Location = 'north';
    hold on
    xlabel('Recomputation Length (tokens)');
    set(gca,'YColor','k');
    
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "_CPU_Energy_and_Latency_and_GPU_and_Total_Single";
    saveas(gcf,"EnergyPattern\" + cur_file_beg + ".png" );
    clf

end 
