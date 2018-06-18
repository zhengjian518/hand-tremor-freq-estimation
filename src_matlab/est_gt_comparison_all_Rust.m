%% This program is for generating the frequency curve from acc and video.

clear;

periodicVideo = {'T002','T003','T005','T006', ...
                        'T008','T009','T010','T011', ...
                        'T013','T014','T015','T016', ...
                        'T018','T025','T027','T029', ...
                        'T034','T035','T036','T037', ...
                        'T041'};
publishableVideo = {'T002','T004','T005','T006', ...
                        'T007','T008','T009','T011', ...
                        'T013','T014','T016','T021', ...
                        'T022','T025','T026','T027' ...
                        'T034','T035','T037','T039', ...
                        'T040','T042'};
videoCodeList = dir('../data/evaluate_FFT_threshold/');
sampleFreq = 1000.0;
c = {};
error = zeros(length(publishableVideo),1);
stddevi = zeros(length(publishableVideo),1);
k=0;
for i = 3 : length(videoCodeList)
% if contains(videoCodeList(i).name,publishableVideo)
    k=k+1;    
    if mod(k,12)==1
        figure;
        axes( 'Position', [0, 0.95, 1, 0.05] ) ;
        set( gca, 'Color', 'None', 'XColor', 'White', 'YColor', 'White' ) ;
        text( 0.5, 0, 'Frequency Estimations for Real Videos', 'FontSize', 14', 'FontWeight', 'Bold', ...
              'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
        subplot(4,3,1);
    elseif mod(k,12)==0
        subplot(4,3,12);
    else
        subplot(4,3,mod(k,12));
    end
    
    
    estFile = strcat(videoCodeList(1).folder,'/',videoCodeList(i).name,'/Rust/Rust_tfd_freq.csv');
    est = csvread(estFile,1,0); % batch_acc
    est = est(:,1); % batch_acc
    
    accFile = strcat('../data/evaluate_FFT/','/',videoCodeList(i).name,'/Rust/kinect_accelerometer.txt');
    acc = textscan(fopen(accFile), '%f %f %f %f %f %f %f %f %f');
    freq = [];
    isPeak = [];
    for j = 1:3
        accSignal = acc{j};
        [time, freq_,isPeak_] = AccToFreq(accSignal,sampleFreq);
%         isPeak
        if isempty(freq)
            freq = freq_;
        else
            freq = freq+freq_;
        end
%         if isempty(isPeak)
%             isPeak = isPeak_;
%         else
%             isPeak =isPeak+isPeak_;
%         end
    end
%     isPeak = isPeak >1;
%     if sum(isPeak)>length(isPeak)/2
%         color='g';
%     else
%         color='r';
%     end
    freq = freq/3;

    plot(time(1:length(est)), freq(1:length(est)),'b',time(1:length(est)), est,'r','LineWidth',3);hold on;
    if mod(k,12)==0 || k==length(videoCodeList)-2
            hl = legend( 'Accelerometer','Video');
            set(hl,'Position', [0.853 0.83 0.2 0.1],'Units', 'normalized');
    end
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    axis([0 max(time(1:length(est))) 0 max(max(freq(1:length(est)),est'))+2]);
    title(strrep(videoCodeList(i).name,'_',' ') ); % strcat('Frequency along Time')
    if contains(videoCodeList(i).name,periodicVideo)
        set(gca,'color',[220 220 220]/255);
    end
    error(k) =  mean( (est' - freq(1:length(est))).^2  );
    stddevi(k) = std(est);
    
% end
end
% error
% stddevi

% b = barwitherr(stddevi, errorBar);
figure
b = bar(error);
b(1).FaceColor = 'b';
set(gca,'XTick',1:length(periodicVideo),'xticklabel',periodicVideo);
xtickangle(-45);
title('MSE for Real Videos');
xlabel('Videos');
ylabel('Mean Squared Error');