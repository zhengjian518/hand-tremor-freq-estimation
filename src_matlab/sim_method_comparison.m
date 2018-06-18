% This program compare three different cropping methods, taking 1st frame
% PE for cropping, sliding PE and tracking PE. The results are shown in
% curve and bar chart.

%% T008
% 
% accDir = '../data/T008_Links_acc/'; 
% estDir = '../data/compare_sim_method/T008/';
% estTypes = {'_sliding_PE','sim'};
% estLegends = {'Pixel-wise', 'Similarity'};
% estColor = {'r','g'};
% estLineWidth = [3 2];
% estFileType = '.mat';
% actionList = {'Rust','Handen_in_pronatie' , '100-7','Maanden_terug' , ...
%                          'Duimen_omhoog', 'Top-top', 'Top_neus_links', 'Top_neus_rechts'};
% actionLabel = {'Rust','Handen' , '100-7','Maanden' , ...
%                          'Duimen', 'Top', 'Links', 'Rechts'};
% sampleFreq = 1000.0;
% 
% errorBar = zeros(length(actionList)+1,length(estTypes));
% stddevi = zeros(length(actionList)+1,length(estTypes));
% 
% figure;
% axes( 'Position', [0, 0.95, 1, 0.05] ) ;
% set( gca, 'Color', 'None', 'XColor', 'White', 'YColor', 'White' ) ;
% text( 0.5, 0, 'Frequency Estimation for T008', 'FontSize', 14', 'FontWeight', 'Bold', ...
%       'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
% 
% for i = 1:length(actionList)
%     subplot(4, 2, i)
%     % Acceleration
%     accFile = strcat(accDir,actionList{i},'.txt');
%     acc = textscan(fopen(accFile), '%f %f %f %f %f %f %f %f %f');
%     freq = [];
%     for j = 1:3
%         accSignal = acc{j};
%         [time, freq_,isPeak_] = AccToFreq(accSignal,sampleFreq);
%         if isempty(freq)
%             freq = freq_;
%         else
%             freq = freq+freq_;
%         end
%     end
%     freq = freq/3;
%     plot(time,freq,'LineWidth',2);
%     hold on;
%     % Estimation from Video
%     for j = 1:2
%         estFile = strcat(estDir, actionList{i}, estTypes{j}, estFileType);
%         if strcmp(estFileType,'.csv')
%             est = csvread(estFile,1,0);% batch_acc
%             est = est(:,8); % batch_acc        
%         elseif  strcmp(estFileType,'.mat')
%             est = load(estFile); % batch_sim_mat
%             est = getfield(est,'freq'); % batch_sim_mat        
%         end
%         if j==1
%             sq_error =  (est - freq(1:length(est))).^2  ;
%         else
%             sq_error =  (est' - freq(1:length(est))).^2  ;
%         end
%         errorBar( i, j ) = mean( sq_error );
%         stddevi( i, j) = std(sq_error);
%         plot(time(1:length(est)), est,'LineWidth',estLineWidth(j),'Color',estColor{j}); % video sequence may be shorter than acc sequence
%         hold on;
%     end
%     
%     axis([0 time(length(est)) 0 9]);
%     xlabel('Time (s)');
%     ylabel('F (Hz)');
%     title(strrep(actionList{i},'_',' '));
% end
% hl = legend('Accelerometer',estLegends{1},estLegends{2});
% set(hl,'Position', [0.853 0.83 0.2 0.1],'Units', 'normalized');
% 
% errorBar(length(actionList)+1,:) = mean(errorBar);
% stddevi(length(actionList)+1,:) = mean(stddevi);
% 
% figure;
% % b = barwitherr(stddevi, errorBar);
% b = bar(errorBar);
% b(1).FaceColor = estColor{1};
% b(2).FaceColor = estColor{2}; % b(3).FaceColor = 'y';
% actionLabel(length(actionList)+1) = {'Average'};
% set(gca,'xticklabel',actionLabel);
% xtickangle(-45);
% % axis([0.5 9.5 -3 18])
% title('MSE for approaches on T008');
% xlabel('Videos');
% ylabel('Mean Squared Error');
% legend(estLegends{1},estLegends{2},'Location','northwest'); % ,estLegends{3}
%% T011
% set(0,'DefaulttextInterpreter','none')

accDir = '../data/T011_acc/'; 
estDir = '../data/compare_sim_method/T011/'; 
estTypes = {'_sliding_PE','sim'};
estLegends = {'Pixel-wise', 'Similarity'};
estColor = {'r','g'};
estLineWidth = [3 2];
estFileType = '.mat';
actionList = {'Rust','Handen_in_pronatie' , '100-7','Maanden_terug' , ...
                         'Duimen_omhoog', 'Top-top', 'Top_neus_links', 'Top_neus_rechts'};
actionLabel = {'Rust','Handen' , '100-7','Maanden' , ...
                         'Duimen', 'Top', 'Links', 'Rechts'};
sampleFreq = 1000.0;

errorBar = zeros(length(actionList)+1,length(estTypes));
stddevi = zeros(length(actionList)+1,length(estTypes));

figure;
axes( 'Position', [0, 0.95, 1, 0.05] ) ;
set( gca, 'Color', 'None', 'XColor', 'White', 'YColor', 'White' ) ;
text( 0.5, 0, 'Frequency Estimation for T011', 'FontSize', 14', 'FontWeight', 'Bold', ...
      'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;

for i = 1:length(actionList)
    subplot(4, 2, i)
    % Acceleration
    accFile = strcat(accDir,actionList{i},'.txt');
    acc = textscan(fopen(accFile), '%f %f %f %f %f %f %f %f %f');
    freq = [];
    for j = 1:3
        accSignal = acc{j};
        [time, freq_,isPeak_] = AccToFreq(accSignal,sampleFreq);
        if isempty(freq)
            freq = freq_;
        else
            freq = freq+freq_;
        end
    end
    freq = freq/3;
    plot(time,freq,'LineWidth',2);
    hold on;
    % Estimation from Video
    for j = 1:2
        estFile = strcat(estDir, actionList{i}, estTypes{j}, estFileType);
        if strcmp(estFileType,'.csv')
            est = csvread(estFile,1,0);% batch_acc
            est = est(:,8); % batch_acc        
        elseif  strcmp(estFileType,'.mat')
            est = load(estFile); % batch_sim_mat
            est = getfield(est,'freq'); % batch_sim_mat        
        end
        if j==1
            sq_error =  (est - freq(1:length(est))).^2  ;
        else
            sq_error =  (est' - freq(1:length(est))).^2  ;
        end
        errorBar( i, j ) = mean( sq_error );
        stddevi( i, j) = std(sq_error);
        plot(time(1:length(est)), est,'LineWidth',estLineWidth(j),'Color',estColor{j}); % video sequence may be shorter than acc sequence
        hold on;
    end
    
    axis([0 time(length(est)) 0 9]);
    xlabel('Time (s)');
    ylabel('F (Hz)');
    title(strrep(actionList{i},'_',' '));
end
hl = legend('Accelerometer',estLegends{1},estLegends{2});
set(hl,'Position', [0.853 0.83 0.2 0.1],'Units', 'normalized');

errorBar(length(actionList)+1,:) = mean(errorBar);
stddevi(length(actionList)+1,:) = mean(stddevi);

figure;
% b = barwitherr(stddevi, errorBar);
b = bar(errorBar);
b(1).FaceColor = estColor{1};
b(2).FaceColor = estColor{2}; % b(3).FaceColor = 'y';
actionLabel(length(actionList)+1) = {'Average'};
set(gca,'xticklabel',actionLabel);
xtickangle(-45);
% axis([0.5 9.5 -3 18])
title('MSE for approaches on T011');
xlabel('Videos');
ylabel('Mean Squared Error');
legend(estLegends{1},estLegends{2},'Location','northwest'); 