function [time,freq,isPeak] = AccToFreq(accSignal, sampleFreq,windowSize)
    % Extract Offset
    accSignal = accSignal - mean(accSignal);

    % Butterworth bandpass filter 
    % fc = [1.5 20];
    fc = [2 14.5]; % 1.5 15
    fs = sampleFreq;
    [b,a] = butter(4,fc/(fs/2)); % 8th-oeder
    accSignal = filter(b,a,accSignal);

    % STFT
    switch nargin
    case 2
        windowSize =  4033;
%         windowSize =  4096;
    end
    noverlap = floor(windowSize/2);
    f = [0:windowSize/2] .* (sampleFreq / windowSize);
    f = f(f<20);
    r = (sampleFreq / 2) * 2 / (length(accSignal) - 1);
    winTuk = tukeywin(windowSize,r);
    sg = spectrogram(accSignal,winTuk,noverlap,f,sampleFreq,'yaxis');
%     spectrogram(sg,'yaxis')
    sg = abs(sg);
    [M,I] = max(sg);
    time = [0:size(sg,2)-1] .* (windowSize-noverlap)/sampleFreq;
    freq = f(I);
    isPeak = M > (mean(sg) + 3*std(sg));
    % isPeak = M > (mean(sg) + 4*std(sg));
%     for i =isPeak
%             figure
%             plot(sg(:,i))
%     end
end
