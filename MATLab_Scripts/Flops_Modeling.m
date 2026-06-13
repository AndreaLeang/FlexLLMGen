close all % just as caution
clf


% Combine all (bytes, bandwidth) data and graphs them together
% bi_storing_netsres\
folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\Flops\remote\'; % Replace with your path
csv_folder = "Flops\remote\";
cur_file_beg = "flops_mha";
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
        
    else
        T = readtable(csv_folder + cur_file);

        T.gbs = cur_file_gbs * ones(height(T),1);
        T.ngbs = cur_file_ngbs * ones(height(T),1);
        T.tot_num_prompts = (cur_file_gbs * cur_file_ngbs) * ones(height(T),1);
        T.prompt_len = cur_prompt_len  * ones(height(T),1);
        T.gpu_per = cur_gpu_percent  * ones(height(T),1);

        % Concatenate current table to the accumulated data
        all_data_table = [all_data_table; T]; 
    end

end 
all_data_table.FLOPS = all_data_table.FLOPS ./ 1000000000000; % FLOPS --> TFLOPS

all_flops = all_data_table.FLOPS;
all_num_of_ops = all_data_table.Num_of_Operations;

g = fittype('b / ( ((a*b)/x) + 1)');
[fitresult, gof] = fit(all_num_of_ops, all_flops, g, 'StartPoint', [0.001, 312], 'Lower', [0, 0])



% Plot
scatter(all_data_table, "Num_of_Operations" , "FLOPS")
hold on
 
% yline(312, 'r', "Ideal FLOPS");
% hold on 
ylabel('TFLOPS');
xlabel('Number of Operations');
hold on 

% save
saveas(gcf,"Flops\" + cur_file_beg + ".png" );
% clf
% 
% 
% % remote-recompute
% folder_path = 'C:\Users\AndreaLeang\akleang\Heatmap\Flops\remote-recompute\'; % Replace with your path
% csv_folder = "Flops\remote-recompute\";
% cur_file_beg = "flops_recomp";
% file_pattern = fullfile(folder_path, '*.csv');
% files = dir(file_pattern);
% tbl = struct2table(files);
% 
% all_data_table = [];
% 
% for i=1:height(tbl)
% % for i=1:2
%     cur_file = string(tbl{i, "name"});
%     cur_file_name_comp = split(cur_file, '-');
%     cur_file_gbs = str2double(extractAfter(cur_file_name_comp(3), 3));
%     cur_file_ngbs = str2double(extractAfter(cur_file_name_comp(4), 4));
%     cur_prompt_len = str2double(extractAfter(cur_file_name_comp(5), 6));
%     cur_gpu_percent =str2double(cur_file_name_comp(10));
%     % Read csv
%     if (i == 1)
%         all_data_table = readtable(csv_folder + cur_file);
%         all_data_table.gbs = cur_file_gbs * ones(height(all_data_table),1);
%         all_data_table.ngbs = cur_file_ngbs * ones(height(all_data_table),1);
%         all_data_table.tot_num_prompts = (cur_file_gbs * cur_file_ngbs) * ones(height(all_data_table),1);
%         all_data_table.prompt_len = cur_prompt_len  * ones(height(all_data_table),1);
%         all_data_table.gpu_per = cur_gpu_percent  * ones(height(all_data_table),1);
% 
%     else
%         T = readtable(csv_folder + cur_file);
% 
%         T.gbs = cur_file_gbs * ones(height(T),1);
%         T.ngbs = cur_file_ngbs * ones(height(T),1);
%         T.tot_num_prompts = (cur_file_gbs * cur_file_ngbs) * ones(height(T),1);
%         T.prompt_len = cur_prompt_len  * ones(height(T),1);
%         T.gpu_per = cur_gpu_percent  * ones(height(T),1);
% 
%         % Concatenate current table to the accumulated data
%         all_data_table = [all_data_table; T]; 
%     end
% 
% end 
% all_data_table.FLOPS = all_data_table.FLOPS ./ 1000000000000; % FLOPS --> TFLOPS
% 
% all_flops = all_data_table.FLOPS;
% all_num_of_ops = all_data_table.Num_of_Operations;
% 
% g = fittype('b / ( ((a*b)/x) + 1)');
% [fitresult, gof] = fit(all_num_of_ops, all_flops, g, 'StartPoint', [0.001, 312], 'Lower', [0, 0])
% 
% 
% 
% % Plot
% scatter(all_data_table, "Num_of_Operations" , "FLOPS")
% hold on
% % plot(f);
% % hold on 
% yline(312, 'r', "Ideal FLOPS");
% hold on 
% % plot(fitresult)
% % legend('off')
% hold on
% % title('Time to Move Data from Pageable to Pinned Over Amount of Data Transferred');
% ylabel('TFLOPS');
% xlabel('Number of Operations');
% hold on 
% 
% % save
% saveas(gcf,"Flops\" + cur_file_beg + ".png" );


