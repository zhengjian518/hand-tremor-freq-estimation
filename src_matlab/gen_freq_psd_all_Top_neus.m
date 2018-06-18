% This is for generate Freq and PSD for Top_neus tasks

clear;
windowSize =  1023; % 31 frames window
% windowSize = 4033;
sampleFreq = 1000.0;
video_code_list = dir('/tudelft.net/staff-umbrella2/tremor data/Tremor_data/');
results_save_path = '../results/All_Top_neus_links_win31_results/';
if isdir(results_save_path) ==0
    mkdir(results_save_path);
end

noverlap = floor(windowSize/2);
f = [0:windowSize/2] .* (sampleFreq / windowSize);
freq_series = f(f<15);

for i = 3:length(video_code_list)
    patient_folder_name = video_code_list(i).name;
    fprintf('Processing on video_Code: %s.\n', patient_folder_name);

    if contains(patient_folder_name,"Rechts")
        accFile = strcat(video_code_list(i).folder,'/',patient_folder_name,'/Top_neus_links','/kinect_accelerometer.tsv');
    else
        accFile = strcat(video_code_list(i).folder,'/',patient_folder_name,'/Top_neus_rechts','/kinect_accelerometer.tsv');
    end

    f = fopen(accFile);
    try
        acc = textscan(f, '%f %f %f %f %f %f %f %f %f');
        freq = []; % sqrt(/3)
        freq_before = []; % (/3)
        isPeak = [];
        psd = [];
        psd_before = [];
        coordinate_count = 0;
        for j = 1:3
            accSignal = acc{j};
            [time, freq_,isPeak_] = AccToFreq(accSignal,sampleFreq,windowSize);
            [~, psd_,peak_overall] = AccToFreqPSD(accSignal,sampleFreq,windowSize);
            if isempty(freq)
                freq = freq_.^2;
            else
                freq = freq+freq_.^2;
            end

            if isempty(isPeak)
                isPeak = isPeak_;
            else
                isPeak = isPeak+isPeak_;
            end
            % if peak_overall == 1
            %     if isempty(psd)
            %         psd = psd_.^2;
            %     else
            %         psd = psd+psd_.^2;
            %     end
            % else
            %     fprintf('No peak found in the whole wideo in %s coordinate', j);
            % end
            coordinate_count = coordinate_count + peak_overall;
            if isempty(psd)
                psd = psd_.^2;
            else
                psd = psd+psd_.^2;
            end
        end

        freq = sqrt(freq/3);
        psd = sqrt(psd/3);
        [peak_psd,~] = max(psd);
        isPeak_overall = peak_psd > (mean(psd) + 3*std(psd)); % 1 * N
        freq_overall = freq_series(find(psd==max(psd)));

        len = length(freq);
        freq_cell = cell(len,1);
        for k = 1:len
            freq_cell{k} = [k,freq(k),isPeak(k)];
        end

        psd_cell = cell(length(freq_series),1);
        for m = 1:length(freq_series)
            psd_cell{m} = [freq_series(m),psd(m)];
        end

        % write results into a txt file
        patient_folder_path = strcat(results_save_path,patient_folder_name);
        if isdir(patient_folder_path) == 0
            mkdir(patient_folder_path);
        end

        freq_result_path = strcat(patient_folder_path,'/','freq_result.txt');
        fid_freq = fopen(freq_result_path,'w');
        fprintf(fid_freq,'%d %f %d \n',freq_cell{:});
        fprintf(fid_freq,'%d %f \n',isPeak_overall,freq_overall);
        fclose(fid_freq);

        psd_result_path = strcat(patient_folder_path,'/','psd.txt');

        fid_psd=fopen(psd_result_path,'w');
        fprintf(fid_psd,'%f %f \n',psd_cell{:});
        fclose(fid_psd);

    catch
        fprintf('No Top_neus task in video_code %s or missing data in tsv file, skipped.\n', patient_folder_name);
    end
end

'done!'