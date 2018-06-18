% function [freq,avgPSD] = AccToFreqPSD(accSignal, sampleFreq,windowSize)
%     % Extract Offset
%     accSignal = accSignal - mean(accSignal);

%     % Butterworth bandpass filter 
%     fc = [2 14.5]; % 1.5 15
%     fs = sampleFreq;
%     [b,a] = butter(4,fc/(fs/2));
%     accSignal = filter(b,a,accSignal);

%     % STFT
%     switch nargin
%     case 2
%         windowSize =  4033;
% %         windowSize =  4096;
%     end
%     noverlap = floor(windowSize/2);
%     f = [0:windowSize/2] .* (sampleFreq / windowSize);
%     f = f(f<15);
%     r = (sampleFreq / 2) * 2 / (length(accSignal) - 1);
%     winTuk = tukeywin(windowSize,r);
%     sg = spectrogram(accSignal,winTuk,noverlap,f,sampleFreq,'yaxis');
%     sg = abs(sg);
%     avgPSD = mean(sg,2);
%     avgPSD = avgPSD/max(avgPSD);
%     freq = f;
% %     [M,I] = max(sg);
% %     time = [0:size(sg,2)-1] .* (windowSize-noverlap)/sampleFreq;
% %     freq = f(I);
% end


function [freq,avgPSD,peak_overall] = AccToFreqPSD(accSignal, sampleFreq,windowSize)
    % Extract Offset
    accSignal = accSignal - mean(accSignal);

    % Butterworth bandpass filter 
    fc = [2 14.5]; % 1.5 15
    fs = sampleFreq;
    [b,a] = butter(4,fc/(fs/2));
    accSignal = filter(b,a,accSignal);

    % STFT
    switch nargin
    case 2
        windowSize =  4033;
%         windowSize =  4096;
    end
    noverlap = floor(windowSize/2);
    f = [0:windowSize/2] .* (sampleFreq / windowSize);
    f = f(f<15);
    r = (sampleFreq / 2) * 2 / (length(accSignal) - 1);
    winTuk = tukeywin(windowSize,r);

    stride = windowSize - noverlap;
    avgPSD = zeros(length(f),1);
    peak_count = 0;
    for i = 1:(length(accSignal)/stride)-1
        accSignal_window = accSignal(1+(i-1)*stride:1+(i-1)*stride + windowSize-1);
        sg = spectrogram(accSignal_window,winTuk,0,f,sampleFreq,'yaxis');
        sg = abs(sg);
        [M,~] = max(sg);
        isPeak = M > (mean(sg) + 3*std(sg)); % 1 * N

        % only take psd with a peak into account
        if not (isPeak == 0)
            peak_count =peak_count + 1;
            PSD = mean(sg,2);
        else
            PSD = zeros(length(f),1);
        end
        avgPSD = avgPSD+PSD;
    end

    if peak_count>0
        avgPSD = avgPSD/max(avgPSD);
        peak_overall = 1;
    else
        avgPSD = zeros(length(f),1);
        peak_overall = 0;
    end
    freq = f;
%     [M,I] = max(sg);
%     time = [0:size(sg,2)-1] .* (windowSize-noverlap)/sampleFreq;
%     freq = f(I);
end
