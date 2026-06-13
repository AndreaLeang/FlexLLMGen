close all % just as caution
clf

% % Aggregate Plot
% % For each file, plot bandwdith over data amount
folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\OffloadingPercent'; % Replace with your path
file_pattern = fullfile(folder_path, '*.csv');
files = dir(file_pattern);
tbl = struct2table(files);

for i=1:height(tbl)
    cur_file = string(tbl{i, "name"});
    % Read csv 
    T = readtable("OffloadingPercent\"+cur_file);
    all_extra_time = T.x_ExtraTimeDueToOffloading;
    all_total_time = T.Sweep__OfTotalTimeTransferringData;

    % Clean OOM 
    nanIdx = isnan(all_extra_time);
    
    
    % Remove the identified rows 
    T_clean = T(~nanIdx, :);
    height(T)
    height(T_clean)

    all_extra_time = T_clean.Sweep__OfTotalTimeTransferringData
    nanIdx50 = all_extra_time < 50;
    T_test = T_clean(~nanIdx50, :);
    height(T_clean)
    height(T_test) % Extra: 175/179, total: 149/179

    % Plot 
    % sgtitle('Percentage of Latency Due to Offloading KV Cache');
    hold on 
    % subplot(1, 2, 1);
    % histogram(T_clean.x_ExtraTimeDueToOffloading,'BinLimits',[0,100]) 
    % ylabel('Frequency');
    % xlabel('Percent of Extra Latency Loading KV Cache (%)');
    
    
    % subplot(1, 2, 2);
    histogram(T_clean.Sweep__OfTotalTimeTransferringData,'BinLimits',[0,100]) 
    ylabel('Frequency', "FontSize",14);
    xlabel('Percent of Total Latency From Loading KV Cache (%)',  "FontSize",14);


    % save
    cur_file_beg = extractBefore(cur_file,".csv");
    cur_file_beg = cur_file_beg + "Motivation-OffloadingPercent"
    saveas(gcf,"OffloadingPercent\" + cur_file_beg + ".png" );
    clf
end 