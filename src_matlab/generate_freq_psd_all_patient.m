% This is for generate Freq and PSD for Raw data

clear;
windowSize =  2033;
sampleFreq = 1000.0;
video_code_list = dir('/tudelft.net/staff-umbrella2/tremor data/Tremor_data/');
% video_task_list = dir('/tudelft.net/staff-umbrella2/tremor data/Tremor_data/T008_Rechts/');
results_save_path = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/jianzheng/baseline_win61/';
if isdir(results_save_path) ==0
    mkdir(results_save_path);
end

noverlap = floor(windowSize/2);
f = [0:windowSize/2] .* (sampleFreq / windowSize);
freq_series = f(f<15);

% for all patients

for i = 3:length(video_code_list)
    patient_folder_name = video_code_list(i).name;
    patient_folder_name
    code_result_save_path = strcat(results_save_path,patient_folder_name);
    if isdir(code_result_save_path) == 0
        mkdir(code_result_save_path);
    end

    task_path = strcat(video_code_list(i).folder,'/',video_code_list(i).name);
    task_list = dir(task_path);
    for j = 3 : length(task_list)
        task_folder_path = task_list(j).folder;
        task_folder_name = task_list(j).name
        accFile = strcat(task_folder_path,'/',task_folder_name,'/kinect_accelerometer.tsv');
        % f = fopen(accFile);
        % acc = textscan(fopen(accFile), '%f %f %f %f %f %f %f %f %f');
        acc = tdfread(accFile,'\t');
        acc = struct2cell(acc);
        if isempty(acc)
            'empty kinect_accelerometer.tsv, skip'
            continue
        end

        freq = []; % sqrt(/3)
        isPeak = [];
        psd = [];
        % isPeak_overall = [];
        for k = 1:3
            accSignal = acc{k};
            [~, freq_,isPeak_] = AccToFreq(accSignal,sampleFreq,windowSize);
            [~,psd_,isPeak_overall_] = AccToFreqPSD(accSignal,sampleFreq,windowSize);
            if isempty(freq)
                freq = freq_.^2;
            else
                freq = freq+freq_.^2;
            end

            if isPeak_overall_ > 0 % x,y,z peak 
                if isempty(psd)
                    psd = psd_.^2;
                else
                    psd = psd+psd_.^2;
                end
            end 

            % if isempty(isPeak_overall)
            %     isPeak_overall = isPeak_overall_;
            % else
            %     isPeak_overall = isPeak_overall + peak_overall_;
            % end
        end

        freq = sqrt(freq/3);
        psd = sqrt(psd/3);
        % psd = psd(1:length(freq_series));
        % psd(2:length(freq_series)) = psd(2:length(freq_series)).*2;

        freq_overall = freq_series(find(psd==max(psd)));
        if max(psd) > (mean(psd) + 3*std(psd))
            isPeak_overall = 1;
        else
            isPeak_overall = 0;
        end
        count = length(freq);

        result_path = strcat(code_result_save_path,'/',task_folder_name,'/');
        if isdir(result_path)== 0
            mkdir(result_path);
            'path maked'
        end
        freq_txt_path = strcat(result_path,'freq.txt');

        % write results into a txt file

        freq_cell = cell(count,1);
        for m = 1:count
            freq_cell{m} = [m,freq(m)];
        end

        freqfid=fopen(freq_txt_path,'w');
        fprintf(freqfid,'%2d %2.11f\n',freq_cell{:});

        % if isPeak_overall ~= 0
        %     isPeak_overall = 1;
        % end
        row = [isPeak_overall,freq_overall];

        fprintf(freqfid,'%1d %2.11f\n', row);
        fclose(freqfid);
        
        series_len = length(freq_series);

        psd_txt_path = strcat(result_path,'psd.txt');
        psd = psd/max(psd);

        psd_cell = cell(series_len,1);
        for n = 1:series_len
            psd_cell{n} = [freq_series(n),psd(n)];
        end

        psdfid=fopen(psd_txt_path,'w');
        fprintf(psdfid,'%2.11f %2.11f\n',psd_cell{:});
        
        fclose(psdfid);

    end
end

% for one patient

% task_list = video_task_list;
% for j = 3 : length(task_list)
%     task_folder_path = task_list(j).folder;
%     task_folder_name = task_list(j).name;
%     accFile = strcat(task_folder_path,'/',task_folder_name,'/kinect_accelerometer.tsv');
%     % f = fopen(accFile);
%     % acc = textscan(fopen(accFile), '%f %f %f %f %f %f %f %f %f');
%     acc = tdfread(accFile,'\t');
%     acc = struct2cell(acc);
    
%     freq = []; % sqrt(/3)
%     freq_before = []; % (/3)
%     isPeak = [];
%     psd = [];
%     psd_before = [];
%     for k = 1:3 
%         accSignal = acc{k};
%         [~, freq_,isPeak_] = AccToFreq(accSignal,sampleFreq);
%         [~, psd_] = AccToFreqPSD(accSignal,sampleFreq);
%         if isempty(freq)
%             freq_before = freq_;
%             freq = freq_.^2;
%         else
%             freq_before = freq_before + freq_;
%             freq = freq+freq_.^2;
%         end

%         if isempty(psd)
%             psd_before = psd_;
%             psd = psd_.^2;
%         else
%             psd_before = psd_before + psd_;
%             psd = psd+psd_.^2;
%         end
%     end


%     freq_before = freq_before/3;
%     % freq_before = ['freq_before';freq_before];
%     freq = sqrt(freq/3);
%     % freq = ['freq';freq];
%     freq_diff = freq - freq_before;
%     % freq_diff = ['freq_diff';freq_diff];

%     psd_before = psd_before/3;
%     psd = sqrt(psd/3);
%     psd_diff = psd - psd_before;

%     freq_overall = freq_series(find(psd==max(psd)));

%     len = length(freq);

%     result_path = strcat(results_save_path,'/',task_folder_name,'/');
%     if isdir(result_path)==0
%         mkdir(result_path);
%         'path maked'
%     end
%     txt_path = strcat(result_path,'freq.txt');

%     % write results into a txt file

%     freq_cell = cell(len,1);
%     for k = 1:len
%         freq_cell{k} = [k,freq(k),freq_before(k),freq_diff(k)];
%     end
%     fid=fopen(txt_path,'w');
%     psd_head = ['count','freq','freq_before','diff'];
%     fprintf(fid,'%5s %10s %11s %10s\n', 'count','freq','freq_before','diff');

%     fprintf(fid,'%5d %2.8f %2.8f %3.8f\n',freq_cell{:});
    
%     fprintf(fid,'%s %2.8f\n','freq_overall: ',freq_overall);
%     fclose(fid);
    
% end

'done!'