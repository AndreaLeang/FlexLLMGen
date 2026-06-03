close all % just as caution
clf


% Combine all (bytes, bandwidth) data and graphs them together
% bi_storing_netsres\
folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\PageableToPinned\remote\'; % Replace with your path
csv_folder = "PageableToPinned\remote\";
cur_file_beg = "remote_pinned_test";
file_pattern = fullfile(folder_path, '*.csv');
files = dir(file_pattern);
tbl = struct2table(files);

all_data_table = [];

for i=1:height(tbl)
% for i=1:2
    cur_file = string(tbl{i, "name"});
    cur_file_name_comp = split(cur_file, '-');
    cur_file_gbs = str2double(extractAfter(cur_file_name_comp(3), 3));
    cur_file_ngbs = str2double(extractAfter(cur_file_name_comp(4), 4));
    cur_prompt_len = str2double(extractAfter(cur_file_name_comp(5), 6));
    cur_gpu_percent =str2double(cur_file_name_comp(10));
    % Read csv
    if (i == 1)
        all_data_table = readtable(csv_folder + cur_file);
        all_data_table.gbs = cur_file_gbs * ones(height(all_data_table),1);
        all_data_table.ngbs = cur_file_ngbs * ones(height(all_data_table),1);
        all_data_table.tot_num_prompts = (cur_file_gbs * cur_file_ngbs) * ones(height(all_data_table),1);
        all_data_table.prompt_len = cur_prompt_len  * ones(height(all_data_table),1);
        all_data_table.gpu_per = cur_gpu_percent  * ones(height(all_data_table),1);
        all_data_table.tot_data_moved = ((0.008 * cur_file_gbs * cur_file_ngbs)) + (1.0* (cur_file_gbs * cur_file_ngbs*cur_prompt_len/2048)) * ones(height(all_data_table),1);
        % Remove Outliers
        cur_pinned_times = all_data_table.pinnedTime_s_;
        all_outliers = isoutlier(cur_pinned_times);
        all_data_table = all_data_table(~all_outliers, :);
    else
        T = readtable(csv_folder + cur_file);

        T.gbs = cur_file_gbs * ones(height(T),1);
        T.ngbs = cur_file_ngbs * ones(height(T),1);
        T.tot_num_prompts = (cur_file_gbs * cur_file_ngbs) * ones(height(T),1);
        T.prompt_len = cur_prompt_len  * ones(height(T),1);
        T.gpu_per = cur_gpu_percent  * ones(height(T),1);
        T.tot_data_moved = ((0.008 * cur_file_gbs * cur_file_ngbs)) + (1.0* (cur_file_gbs * cur_file_ngbs*cur_prompt_len/2048)) * ones(height(T),1);

        % Remove Outliers
        cur_pinned_times = T.pinnedTime_s_;
        all_outliers = isoutlier(cur_pinned_times);
        T = T(~all_outliers, :);

        % Concatenate current table to the accumulated data
        all_data_table = [all_data_table; T]; 
    end

end 
all_data_table.bytesTransferred_B_ = all_data_table.bytesTransferred_B_ ./ 1000000000;  % bytes --> GB
all_data_table.pinnedTime_s_ = all_data_table.pinnedTime_s_ ./ 1000000; % us --> s

all_data = all_data_table.bytesTransferred_B_;
all_pinned_time = all_data_table.pinnedTime_s_;

all_idx = all_data_table.idx;
all_gbs = all_data_table.gbs;
all_ngbs = all_data_table.ngbs;
all_gpu_per = all_data_table.gpu_per;
all_tot_prompts = all_data_table.tot_num_prompts;
all_tot_data_moved = all_data_table.tot_data_moved;
all_prompt_len = all_data_table.prompt_len;

% Quadratic Regression
f=fit(all_data,all_pinned_time,'poly2')

% Plot
scatter(all_data_table, "bytesTransferred_B_" , "pinnedTime_s_")
hold on
plot(f);
hold on 
% title('Time to Move Data from Pageable to Pinned Over Amount of Data Transferred');
ylabel('Pinned Time (s)');
xlabel('Amount of Data (GB)');
hold on 

% save
saveas(gcf,"PageableToPinned\pngs\" + cur_file_beg + ".png" );
clf



% % PtP Contribution to whole latency
% folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\PageableToPinned\remote_traceStats\'; % Replace with your path
% csv_folder = "PageableToPinned\remote_traceStats\";
% cur_file_beg = "PtPContribution";
% file_pattern = fullfile(folder_path, '*.csv');
% files = dir(file_pattern);
% tbl = struct2table(files);
% 
% all_data_table = [];
% 
% for i=1:height(tbl)
% % for i=1:2
%     cur_file = string(tbl{i, "name"});
%     all_data_table = readtable(csv_folder + cur_file);
%     all_latency = all_data_table.Latency_s_;
% 
%     % Clean OOM 
%     nanIdx = isnan(all_latency);
% 
%     % Remove the identified rows 
%     T_clean = all_data_table(~nanIdx, :);
% 
%     % Plot
%     bar(T_clean.SeqLen, T_clean.SinglePtPLatency___)
%     % histogram(T_clean.SinglePtPLatency___,'BinLimits',[0,100]) 
%     hold on
% 
%     % title('Time to Move Data from Pageable to Pinned Over Amount of Data Transferred');
%     ylabel('\fontsize{14}Pageable to Pinned Latency Contribution(%)');
%     xlabel('\fontsize{14}Sequence Length');
%     hold on 
% 
%     % save
%     saveas(gcf,"PageableToPinned\pngs\" + cur_file_beg + ".png" );
%     clf
% end
