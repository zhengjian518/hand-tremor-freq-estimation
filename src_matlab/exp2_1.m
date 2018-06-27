% For exp 2.1 Lagrangian freq estimation: pose estimation at all frames + FFT over joints (x,y) ( NO SMOOTH).

path = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/jianzheng/jointData/pathological-tremor-detection-from-video/results/';
video_code_list = dir(path);
results_save_path = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/jianzheng/exp_2_1_update/';
if isdir(results_save_path) ==0
    mkdir(results_save_path);
end

% Constant

windowSize =  61;
noverlap = floor(windowSize/2);
sampleFreq = 30.0;
f = [0:windowSize/2] .* (sampleFreq / windowSize);
freq_series = f(f<15);

% for all patients
for i = 3:length(video_code_list)
    patient_folder_name = video_code_list(i).name
    code_result_save_path = strcat(results_save_path,patient_folder_name);
    if isdir(code_result_save_path) == 0
        mkdir(code_result_save_path);
    end
    if strfind(patient_folder_name,'Links')
    	joint_number = 8;
    else
    	joint_number = 5;
    end

    task_path = strcat(video_code_list(i).folder,'/',video_code_list(i).name);
    task_list = dir(task_path);
    for j = 3 : length(task_list)
        task_folder_path = task_list(j).folder;
        task_folder_name = task_list(j).name
        cpmFile_path = strcat(task_folder_path,'/',task_folder_name,'/prediction_arr/');
        signal_cell = PosToSig(cpmFile_path,joint_number);
        if isempty(signal_cell)
            'something wrong, skip'
            continue
        end

        freq = []; % sqrt(/3)
        isPeak = [];
        psd = [];
		for k = 1:2
		    accSignal = signal_cell{k};
		    [~, freq_,~] = AccToFreq(accSignal,sampleFreq,windowSize);
		    [~, psd_,isPeak_overall_] = AccToFreqPSD(accSignal,sampleFreq,windowSize);
		    if isempty(freq)
		        freq = freq_.^2;
		    else
		        freq = freq+freq_.^2;
		    end
		    
            if isempty(psd)
                psd = psd_.^2;
            else
                psd = psd+psd_.^2;
            end
            
		end

		freq = sqrt(freq/2);
        psd = sqrt(psd/2);

        max_num = find(psd==max(psd));
        if length(max_num) == 1
        	freq_overall = freq_series(max_num);
        else
        	freq_overall = freq_series(max_num(1));
        end

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