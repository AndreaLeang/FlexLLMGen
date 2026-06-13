close all % just as caution
clf

% Combine all (bytes, bandwidth) data and graphs them together
% bi_storing_netsres\
one_test = true
if one_test
    folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\Transfer_bw\one_netsres'; % Replace with your path
    csv_folder = "Transfer_bw\one_netsres\";
    cur_file_beg = "one_netsres_test";
else
    folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\Transfer_bw\bi_netsres'; % Replace with your path
    csv_folder = "Transfer_bw\bi_netsres\";
    cur_file_beg = "bi_netsres_test";
end

file_pattern = fullfile(folder_path, '*.csv');
files = dir(file_pattern);
tbl = struct2table(files);
missing_tbl = ["temp"];

all_data_table = [];

for i=1:height(tbl)
% for i=1:2
    cur_file = string(tbl{i, "name"});
    cur_file_name_comp = split(cur_file, '-');
    cur_file_gbs = str2double(extractAfter(cur_file_name_comp(3), 3));
    cur_file_ngbs = str2double(extractAfter(cur_file_name_comp(4), 4));
    cur_prompt_len = str2double(extractAfter(cur_file_name_comp(5), 6));
    % cur_prompt_len = str2double(extractAfter(cur_file_name_comp(3), 3) + extractAfter(cur_file_name_comp(4), 4));
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
        % % Remove Outliers
        % cur_bandwidths = all_data_table.bandwidth_GB_s_;
        % all_outliers = isoutlier(cur_bandwidths);
        % all_data_table = all_data_table(~all_outliers, :);
    else
        T = readtable(csv_folder + cur_file);
        % if height(T) == 0 && (cur_gpu_percent ~= 100)
        %     missing_tbl = [missing_tbl; cur_file];
        % end
        T.gbs = cur_file_gbs * ones(height(T),1);
        T.ngbs = cur_file_ngbs * ones(height(T),1);
        T.tot_num_prompts = (cur_file_gbs * cur_file_ngbs) * ones(height(T),1);
        T.prompt_len = cur_prompt_len  * ones(height(T),1);
        T.gpu_per = cur_gpu_percent  * ones(height(T),1);
        T.tot_data_moved = ((0.008 * cur_file_gbs * cur_file_ngbs)) + (1.0* (cur_file_gbs * cur_file_ngbs*cur_prompt_len/2048)) * ones(height(T),1);

        % % Remove Outliers
        % cur_bandwidths = T.bandwidth_GB_s_;
        % all_outliers = isoutlier(cur_bandwidths);
        % T = T(~all_outliers, :);

        % Concatenate current table to the accumulated data
        all_data_table = [all_data_table; T]; 
    end

end 
all_data_table.data_B_ = all_data_table.data_B_ ./1000000000;  % bytes --> GB

all_data = all_data_table.data_B_;
all_idx = all_data_table.idx;
all_og_idx = all_data_table.ogIndex;
all_bandwidth = all_data_table.bandwidth_GB_s_;
all_gbs = all_data_table.gbs;
all_ngbs = all_data_table.ngbs;
all_gpu_per = all_data_table.gpu_per;
all_tot_prompts = all_data_table.tot_num_prompts;
all_tot_data_moved = all_data_table.tot_data_moved;
all_prompt_len = all_data_table.prompt_len;

%Clean up data
nanIdx = isnan(all_data) | isnan(all_bandwidth) | all_og_idx ./ 2 == 1;

% Remove the identified rows from both X and y
all_data_clean = all_data(~nanIdx);
all_bandwidth_clean = all_bandwidth(~nanIdx);

all_idx_clean = all_idx(~nanIdx);
T_clean = all_data_table(~nanIdx, :);
all_data_clean = T_clean.data_B_;



% split data by total data loaded
cur_partition_var = T_clean.data_B_;
[G,ID] = findgroups(T_clean{:, "data_B_"});

% T_clean.data_moved_group = G;

% Plot
scatter(all_data_clean, all_bandwidth_clean, 'o', 'MarkerEdgeColor',[92/255 170/255 242/255])
hold on


% log regression for each 
means = []

for i=1:height(ID) 
% for i=1:2
    cur_data_idx = cur_partition_var == ID(i);
    cur_all_data_clean = all_data_clean(cur_data_idx);
    cur_all_bandwidth_clean = all_bandwidth_clean(cur_data_idx);

    %Distribution
    curve_fit = fitdist(cur_all_bandwidth_clean,'Normal');
    cur_mean = curve_fit.mu;
    means = [means; cur_mean];

end

% % Log Regression over all 
% [f, gof] = fit(all_data_clean,all_bandwidth_clean, "log10")
% plot(f)

% % Log regression over means
% [avg_f, avg_gof] = fit(ID, means, "log10")
% plot(avg_f)
% hold on

% asymptoptic over all
% g = fittype('a-b*exp(-c*x)');
% [fitresult, gof] = fit(all_data_clean, all_bandwidth_clean, g, 'StartPoint', [16, 16, 1])
% with understanding that there's a fixed latency cost
g = fittype('b / ( ((a*b)/x) + 1)');
[fitresult, gof] = fit(all_data_clean, all_bandwidth_clean, g, 'StartPoint', [0.001, 16], 'Lower', [0, 0])

plot(fitresult, 'r')
% hold on 
% plot(all_data_clean, all_bandwidth_clean, 'o')
% hold on 

% plot(ID, means, 'o', 'MarkerSize',4,'MarkerEdgeColor',[1/255 61/255 117/255])
% hold on 

% asymptotic over means
% [avg_f, avg_gof] = fit(ID, means, g, 'StartPoint', [16, 16, 1])
% [avg_f, avg_gof] = fit(ID, means, g, 'StartPoint', [0.001, 16], 'Lower', [0, 0])
% plot(avg_f, 'r')

% title('BiDirectional Bandwidth Distribution Across Amount of Data Transferred');
ylabel('Bandwidth (GB/s)', "FontSize",14);
xlabel('Amount of Data (GB)', "FontSize",14);
xlim([0 max(all_data_clean)+0.05])
% set(gca, 'XScale', 'log')
% set(gca, 'YScale', 'log')
hold on 

% save
saveas(gcf,"Transfer_bw\pngs\" + cur_file_beg + ".png" );