%%This program is for generating the PSD from acc and video.

clear;

videoCodeList = dir('../data/Top_neus/');
sampleFreq = 1000.0;
c = {};
for i = 3 :  length(videoCodeList)
    if mod(i-2,12)==1
        figure;
%         st = suptitle('PSD for Real Video');
%         set(st,'Position', [0.85 0.83 0.2 0.1],'Units', 'normalized')
        axes( 'Position', [0, 0.95, 1, 0.05] ) ;
        set( gca, 'Color', 'None', 'XColor', 'White', 'YColor', 'White' ) ;
        text( 0.5, 0, 'PSD for Real Videos', 'FontSize', 14', 'FontWeight', 'Bold', ...
              'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
        subplot(4,3,1);
    elseif mod(i-2,12)==0
        subplot(4,3,12);
    else
        subplot(4,3,mod(i-2,12));
    end
    
%     estFile = strcat(videoCodeList(1).folder,'/',videoCodeList(i).name,'/Rust/accumulated_psd/psd_avg.csv');
%     estmat = csvread(estFile,1,0); % batch_acc
%     est = estmat(:,1); % batch_acc
%     freq_est = estmat(:,2);
    folder = dir( strcat(videoCodeList(1).folder,'/',videoCodeList(i).name) );
    accFile = strcat(folder(3).folder,'/',folder(3).name,'/kinect_accelerometer.txt');
    acc = textscan(fopen(accFile), '%f %f %f %f %f %f %f %f %f');
    psd = [];
    for j = 1:3
        accSignal = acc{j};
        [freq, psd_] = AccToFreqPSD(accSignal,sampleFreq);
        if isempty(psd)
            psd = psd_;
        else
            psd = psd+psd_;
        end
    end
    psd = psd/3;
    color = 'r';
    if max(psd)>mean(psd)+3*std(psd)
        color = 'g';
    end
    
    plot(freq,psd,color,'LineWidth',3);
%     hold on;
%     plot(freq_est,est,'LineWidth',3);
    if mod(i-2,12)==0 || i==length(videoCodeList)
            hl = legend( 'Accelerometer','Video');
            set(hl,'Position', [0.85 0.83 0.2 0.1],'Units', 'normalized');
    end
    ylabel('PSD');
    xlabel('Frequency (Hz)');
    axis([0 15 0 1.1])
    title( strrep(videoCodeList(i).name,'_',' ') ); % strcat('Frequency along Time')
end
