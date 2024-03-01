clc 
clear all; 
close all; 

%% Folders 
fprintf('script: Folder .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic

% Define data path
addpath("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/FunctionFiles")
folderpath_individuel_data = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/Indiv_data/";

names = ["andreas", "andrew", "benedikte", "gritt", "maria", "thomas", "trine", "trine2"]; % "mia" excluded

% Preallocation
total_data_vertical = cell(1,1,numel(names));
total_type_vertical = cell(1,1,numel(names)); 

% Load vertical data 
% - total_data_vertical: preprocessed recorded data 
% - total_type_vertical: control or perturbation trials 
for i = 1:numel(names)
    load(folderpath_individuel_data + names(i) + ".mat" );     total_data_vertical{:,:,i} = data_indiv; 
    load(folderpath_individuel_data + names(i) + "_type.mat"); total_type_vertical{:,:,i} = type;
end

clear data_indiv type 
fprintf('done [ %4.2f sec ] \n', toc);

%% Abbreviation and Acquisition Set-Up
fprintf('script: Abbreviation and Acquisition Set-Up  .  .  .  .  .  .  .  '); tic

% Protocol abbreviation types
CTL = 1; VER = 2; HOR = 3; CTL2 = 4; 

% Sensor abbreviation type
SOL = 1; TA = 2; ANG = 3; FSR = 4; time = 5; VEL = 6; ACC = 7;

% Plotting labels 
labels = ["Soleus"; "Tibialis"; "Position"; "";  ""; "Velocity"; "Acceleration"];
labels_ms = ["Soleus"+newline+"[\muV]";"Tibialis"+newline+"[\muV]"; "Position"+newline+"[Deg]";  "";  "Time"+newline+"[ms]"; "Velocity"+newline+"[Deg/ms]";"Acceleration"+newline+"[Deg/ms^2]"];
labels_sec = ["Soleus"+newline+"[\muV]";"Tibialis"+newline+"[\muV]"; "Position"+newline+"[Deg]"; "";  "Time"+newline+"[sec]";"Velocity"+newline+"[Deg/s]";"Acceleration"+newline+"[Deg/s^2]"];

% Define the options to align with
align_with_obtions = ["second_begin", "four_begin", "six_begin"];

% Define the step values that were tested
steps_tested = [2,4,6];

% Define conversion functions
ms2sec = @(x) x*10^-3;         % Ms to sec 
sec2ms = @(x) x*10^3;          % Sec to ms 

% Acquisition setup 
sweep_length = 10;             % Signal length in second
Fs = 2000;                     % Samples per second
dt = 1/Fs;                     % Seconds per sample
pre_trig = 4;                  % Pre-trigger 
N = Fs*sweep_length;           % Total number of samples per signal

% Define window parameters
screen_size = get(0, 'ScreenSize');
screen_width = screen_size(3);
screen_height = screen_size(4);
clear screen_size % clear variable to free memory

fprintf('done [ %4.2f sec ] \n', toc);

%% Normalize
fprintf('script: Normalize EMG   .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic; 

normalize = true;      % enable or disable
norm_array_ms = [-800, -200];  % normalize array, denoted in ms                          


% Normalize vertical
if normalize  
    if exist('normalized_done', 'var')
        fprintf(2, 'Re-entry prevented \n')
    else 
        for sub = 1:numel(names)
            data = total_data_vertical{1,1,sub};  
            type = total_type_vertical{1,1,sub}; no = type{2}; yes = type{1};
    
            x_axis = data{VER,time};
            temp1 = find(floor(norm_array_ms(1)) == x_axis);
            temp2 = find(floor(norm_array_ms(2)) == x_axis); 
            norm_sample = [temp1, temp2(1)]; 
        
            sol_max = max(mean(data{VER,SOL}(no,norm_sample(1):norm_sample(2)),1)); 
            ta_max  = max(mean(data{VER,TA }(no,norm_sample(1):norm_sample(2)),1)); 

            data{VER,SOL} = data{VER,SOL}/sol_max;
            data{VER,TA } = data{VER,TA }/ta_max; 

            total_data_vertical{1,1,sub} = data; % save normalize
        end
       normalized_done = true; 
       fprintf('done [ %4.2f sec ] \n', toc);
    end 
else 
    fprintf('disable \n');
end 




%% Speed and aceleration
fprintf('script: Speed and aceleration .  .  .  .  .  .  .  .  .  .  .  .  '); tic;

span_position = 10; %5;          % inc. sample in guassian filter span
span_velocity = 6; %5;          % inc. sample in guassian filter span
span_acceleration = 1;% 10;     % inc. sample in gaussian filter span
plot_data = true; 

% Normalize vertical
if true  
    if exist('normalized_done', 'var')
        fprintf(2, 'Re-entry prevented \n')
    else 
          end 
else 
    fprintf('disable \n');
end 

for sub = 1:numel(names)
    data = total_data_vertical{1,1,sub}; 
    dim = 2;    % dimension, compute the columns
    order = 1;  % compute the first-order derivate,
    
    % Position [d] 
    data{VER,ANG} = rescale(data{VER,ANG})*60-10;              % rescale the signal;
    
    % Velocity [d/s] 
    diffs1 = diff(data{VER,ANG}, order, dim)./(dt*10^3);    % [deg/sample] 
    diffs1 = padarray(diffs1, [0 1], 'post');               % zeropadding 
    data{VER,VEL} = smoothdata(diffs1, dim, 'gaussian', span_velocity);  % gaussian smoothing

    % Acceleration [d/s^2]
    diffs2 = diff(diffs1, order, dim)./(dt*10^3);     % [deg/sample^2]
    diffs2 = padarray(diffs2, [0 1], 'post');         % zeropadding
    data{VER,ACC} = smoothdata(diffs2, dim, 'gaussian', span_velocity);  % gaussian smoothing

    total_data_vertical{1,1,sub} = data;
end

% figure; hold on
% for sub =1:numel(names)
% data = total_data_vertical{1,1,sub}; 
% type = total_type_vertical{1,1,sub}; no = type{2}; yes = type{1};
% plot(data{VER,time},mean(data{VER,ANG}(no,:),1))  
% end

fprintf('done [ %4.2f sec ] \n', toc);

%% Offset

% % Generelt 
% subject = 6; 
% offset_ms = 20; 
% offset_sample = ms2sec(offset_ms)*Fs; 
% data = total_data_vertical{1,1,subject};  
% Nsweep = size(data{VER,SOL},1); 
% for type = [SOL, TA, ANG, VEL]
%     data{VER,type}(:,:) = [zeros(Nsweep, offset_sample), data{VER,type}(:,1:end-offset_sample)];
% end
% total_data_vertical{1,1,subject} = data; 

%% Cross correlation 
fprintf('script: Cross correlation  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic

croos_bool = true; 
templ_array = [500:1500];     % template array, denoted in samples
show_cross = true; 
subject = 2; 
protocol = SOL; 

if croos_bool
    if exist('croos_done', 'var')
        fprintf(2, 'Re-entry prevented \n')
    else
        for sub = 1:numel(names)
            % Load data
            data = total_data_vertical{1,1,sub}; 
            type = total_type_vertical{1,1,sub}; no = type{2}; yes = type{1};
            x_axis = data{VER,time};
        
            % Re-align - cross correlation 
            y_yes = mean(data{VER,protocol}(yes,:),1); 
            y_no  = mean(data{VER,protocol}(no,:),1); 
            template = mean(data{VER,protocol}(no,templ_array),1);
            [rx, lags] = xcorr(y_yes, template);  % Cross-correlation
            
            % Re-align - find Peaks
            [pks, locs] = findpeaks(rescale(rx), 'MinPeakDistance', 500);
            tmp = find(lags(locs) > 0);
            pks = pks(tmp);
            locs = locs(tmp);
            delay = templ_array(1)-lags(locs(1)); % denoted in samples                 
            
            switch sub 
                case 2 
                    delay = delay - 20;
                case 4
                    delay = delay - 20;
                case 6
                    delay = delay + 25; 
            end 
            
            %Re-align - actual
            for type = [SOL, TA, ANG, VEL]
                data{VER,type}(no(:),:) = [data{VER,type}(no(:),delay+1:end), zeros(numel(no), delay)];
            end
            
            % Show cross correlation
            if and(show_cross, sub == subject)   
                figure('name', 'Cross correlation');
                subplot(311); hold on
                    plot(templ_array, template, 'color', [0.7, 0.7, 0.7], 'LineWidth', 5 )
                    plot(y_no, 'color', "blue", 'LineWidth', 2)
                    plot(y_yes, 'color', 'black')
                    title("before")
                    legend(["template", "No", "Yes"])
                subplot(312); hold on 
                    title("Cross correlation")
                    N = numel(y_yes); 
                    plot(lags(locs(1)):lags(locs(1))+numel(template)-1, template, 'color', [0.7, 0.7, 0.7], 'LineWidth', 5);
                    plot(0:N-1, y_yes, 'color', 'black')
                    plot(lags, rescale(rx), 'color', 'blue');
                    plot(lags(locs), pks, 'rx', 'linewidth', 1);
                    plot(lags(locs(1)), pks(1), 'o');
                    plot([lags(locs(1)), lags(locs(1))], [0,pks(1) ])
                    legend(["template","Yes","","","",""])
                subplot(313); hold on
                    title("After")
                    plot(mean(data{VER,protocol}(no, :),1), 'color', "blue", 'LineWidth', 2)
                    plot(mean(data{VER,protocol}(yes,:),1), 'color', 'black', 'LineWidth', 1)
                    legend(["No","Yes"])
            end
    
            % Save data
            total_data_vertical{1,1,sub} = data; 
        end 
        croos_done = true; % prevent re-entry
        fprintf('done [ %4.2f sec ] \n', toc);
    end
else 
    fprintf('disable \n');
end

%% Task 4.1 Vertical perturbation
fprintf('script: TASK 4.1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic

show_plt = true; 
show_boxplt = true; 
subject = 1; 
x_range = [-300 300]; 
plt_show = [ANG, VEL, SOL, TA]; 
box_dim = 1:11; 
savepgn = false; 

% Defined size for window-analysis
SLR = 1; MLR = 2;
pre_search = [0 , 20]; % denoted in ms 
SLR_search = [39, 59]; % denoted in ms  
MLR_search = [60, 80]; % denoted in ms

%   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
% Load data
data = total_data_vertical{1,1,subject}; 
type = total_type_vertical{1,1,subject}; no = type{2}; yes = type{1};
x_axis = data{VER,time};


if show_plt
    % Patch properties 
    patchcolor = "yellow"; %[251 244 199]/255; 
    patchcolor_slr = "blue";
    patchcolor_mlr = "red"; 
    FaceAlpha = 0.1; 
    patX_predict = [pre_search(1) pre_search(2) pre_search(2) pre_search(1)];    % ms 
    patX_slr     = [SLR_search(1) SLR_search(2) SLR_search(2) SLR_search(1)];    % ms
    patX_mlr     = [MLR_search(1) MLR_search(2) MLR_search(2) MLR_search(1)];    % ms
    patY = [-1000 -1000 1000 1000];
    EdgeColor = "none"; %[37 137 70]/255;
    lineWidth_patch = 0.5;

    % Plot properties 
    color_no = [0.75, 0.75, 0.75]; 
    color_yes = "black"; 
    linewidth_no = 3; 
    linewidth_yes = 1; 
    
    % Check if a figure with the name 'TASK3' is open
    fig = findobj('Name', 'Vertical Perturbation');
    if ~isempty(fig), close(fig); end
    
    % Begin plot 
    figure('Name', 'Vertical Perturbation'); hold on; 
    sgtitle("Subject: " + subject)
    for i = 1:numel(plt_show)
        subplot(numel(plt_show), 1, i); hold on
        ylabel(labels_ms(plt_show(i)))
        xlim(x_range) 

        plot(x_axis, mean(data{VER,plt_show(i)}(no(box_dim),:),1),  'color', color_no,  "linewidth", linewidth_no)
        plot(x_axis, mean(data{VER,plt_show(i)}(yes(box_dim),:),1), 'color', color_yes, "linewidth", linewidth_yes)
        YL = get(gca, 'YLim'); ylim([YL(1) YL(2)]);
        if or(plt_show(i) == ANG, plt_show(i) == VEL)  
            patch(patX_predict, patY, patchcolor,'FaceAlpha',FaceAlpha, 'EdgeColor', EdgeColor, 'LineWidth', lineWidth_patch) % predictor
        else
            patch(patX_slr, patY, patchcolor_slr,'FaceAlpha',FaceAlpha, 'EdgeColor', EdgeColor, 'LineWidth', lineWidth_patch)     % depended 1
            patch(patX_mlr, patY, patchcolor_mlr,'FaceAlpha',FaceAlpha, 'EdgeColor', EdgeColor, 'LineWidth', lineWidth_patch)     % depended 2
        end 
        set(gca, 'Layer', 'top') 
        plot(x_axis, mean(data{VER,plt_show(i)}(no(box_dim),:),1), 'color', color_no, "linewidth", linewidth_no)
        plot(x_axis, mean(data{VER,plt_show(i)}(yes(box_dim),:),1), 'color', color_yes, "linewidth", linewidth_yes)
    end
    filename = "Subject"+subject+".png";
    filepath = 'C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/png files/vertical/';
    fullpath = fullfile(filepath, filename);
    
    if savepgn
        saveas(gcf, fullpath, 'png'); 
    end


    if show_boxplt
        % Boxplot
        v_pert = struct; 
        for sub = 1:numel(names)
            data = total_data_vertical{1,1,sub}; 
            type = total_type_vertical{1,1,sub}; no = type{2}; yes = type{1};
            x_axis = data{VER,time};
    
            pre_search_sample = [find(pre_search(1) == x_axis), find(pre_search(2) == x_axis)];
            SLR_search_sample = [find(SLR_search(1) == x_axis), find(SLR_search(2) == x_axis)];
            MLR_search_sample = [find(MLR_search(1) == x_axis), find(MLR_search(2) == x_axis)];
           
 
            for sensor = [ANG, VEL, ACC]
                v_pert.PRE_no(sub,sensor,:) = mean(data{VER,sensor}(no(box_dim) , [pre_search_sample(1):pre_search_sample(2)]), 2);
                v_pert.PRE_yes(sub,sensor,:) = mean(data{VER,sensor}(yes(box_dim), [pre_search_sample(1):pre_search_sample(2)]), 2);
            end

            for muscle = [SOL, TA]
                % Short lantency reflex 
                v_pert.CTL(sub,muscle,SLR,:) = mean(data{VER,muscle}(no(box_dim) , [SLR_search_sample(1):SLR_search_sample(2)]), 2);
                v_pert.VER(sub,muscle,SLR,:) = mean(data{VER,muscle}(yes(box_dim), [SLR_search_sample(1):SLR_search_sample(2)]), 2);
        
                % Medium lantency reflex
                v_pert.CTL(sub,muscle,MLR,:) = mean(data{VER,muscle}(no(box_dim),  [MLR_search_sample(1):MLR_search_sample(2)]), 2);
                v_pert.VER(sub,muscle,MLR,:) = mean(data{VER,muscle}(yes(box_dim), [MLR_search_sample(1):MLR_search_sample(2)]), 2);
            end
        end
    
        % Group labels for boxplot
        sublabels = {'CTL', 'VER'};
        grpLabels = {'1', '2','3','4','5','6','7','8','9','10','11','12'}; 
        grpLabels_conditions = {'SLR', 'MLR'};
    
        % Ny boxplot 
        v_box_soleus_slr   = {squeeze(v_pert.CTL(:,SOL,SLR,:))',squeeze(v_pert.VER(:,SOL,SLR,:))'};  
        v_box_soleus_mlr   = {squeeze(v_pert.CTL(:,SOL,MLR,:))',squeeze(v_pert.VER(:,SOL,MLR,:))'};  
        v_box_sol_slr_avg  = {squeeze(mean(v_pert.CTL(:,SOL,SLR,:),4)), squeeze(mean(v_pert.VER(:,SOL,SLR,:),4))};
        v_box_sol_mlr_avg  = {squeeze(mean(v_pert.CTL(:,SOL,MLR,:),4)), squeeze(mean(v_pert.VER(:,SOL,MLR,:),4))};
    
        % Ny boxplot 
        v_box_tibialis_slr   = {squeeze(v_pert.CTL(:,TA,SLR,:))',squeeze(v_pert.VER(:,TA,SLR,:))'};  
        v_box_tibialis_mlr   = {squeeze(v_pert.CTL(:,TA,MLR,:))',squeeze(v_pert.VER(:,TA,MLR,:))'};  
        v_box_ta_slr_avg  = {squeeze(mean(v_pert.CTL(:,TA,SLR,:),4)), squeeze(mean(v_pert.VER(:,TA,SLR,:),4))};
        v_box_ta_mlr_avg  = {squeeze(mean(v_pert.CTL(:,TA,MLR,:),4)), squeeze(mean(v_pert.VER(:,TA,MLR,:),4))};

    
        % Check if a figure with the name 'TASK3' is open
        fig = findobj('Name', 'Vertical box plot');
        if ~isempty(fig), close(fig); end
        
        % Begin plot 
        figure('Name', 'Vertical box plot'); hold on; 
        subplot(2,4,1:3)
            boxplotGroup(v_box_soleus_slr,'primaryLabels',sublabels,'SecondaryLabels',grpLabels(1:numel(names)), 'interGroupSpace',2,'GroupLines',true,'GroupType','betweenGroups')
            title("Short lantency reflex")
            xlabel("Subject")
            ylabel("Normalized"+newline+"Soleus")
        subplot(2,4,4)
            boxplotGroup(v_box_sol_slr_avg, 'primaryLabels',sublabels)
            title("Grouped all subjects")
        subplot(2,4,5:7)
            boxplotGroup(v_box_soleus_mlr,'primaryLabels',sublabels,'SecondaryLabels',grpLabels(1:numel(names)), 'interGroupSpace',2,'GroupLines',true,'GroupType','betweenGroups')
            ylabel(labels(TA))
            title("Medium lantency reflex")
            xlabel("Subject")
            ylabel("Normalized"+newline+"Soleus")
        subplot(2,4,8)
            boxplotGroup(v_box_sol_mlr_avg, 'primaryLabels',sublabels)
            title("Grouped all subjects")
   

        % Check if a figure with the name 'TASK3' is open
        fig = findobj('Name', 'Vertical box plot2');
        if ~isempty(fig), close(fig); end
        
        % Begin plot 
        figure('Name', 'Vertical box plot2'); hold on; 
        subplot(2,4,1:3)
            boxplotGroup(v_box_tibialis_slr,'primaryLabels',sublabels,'SecondaryLabels',grpLabels(1:numel(names)), 'interGroupSpace',2,'GroupLines',true,'GroupType','betweenGroups')
            title("Short lantency reflex")
            xlabel("Subject")
            ylabel("Normalized"+newline+"Tibialis")
        subplot(2,4,4)
            boxplotGroup(v_box_ta_slr_avg, 'primaryLabels',sublabels)
            title("Grouped all subjects")
        subplot(2,4,5:7)
            boxplotGroup(v_box_tibialis_mlr,'primaryLabels',sublabels,'SecondaryLabels',grpLabels(1:numel(names)), 'interGroupSpace',2,'GroupLines',true,'GroupType','betweenGroups')
            ylabel(labels(TA))
            title("Medium lantency reflex")
            xlabel("Subject")
            ylabel("Normalized"+newline+"Tibialis")
        subplot(2,4,8)
            boxplotGroup(v_box_ta_mlr_avg, 'primaryLabels',sublabels)
            title("Grouped all subjects")
    end

    fprintf('done [ %4.2f sec ] \n', toc);
else 
    fprintf('disable \n');
end

%% Task 5.0 Pre-baseline vs Post-baseline 
% Does the spinal influence chance due to the experienced protocols. 

color_idv = [0.6 0.6 0.6]; 
color_mean = "black"; 
linewidth_mean = 1; 
%patch 
FaceAlpha = 0.1; 
EdgeColor = "none";
lineWidth_patch = 1; 
patchcolor_SLR = "blue";
patchcolor_MLR = "red";
x_pat_slr =  [0.7, 2.3, 2.3, 0.7]; 
x_pat_mlr =  [2.7, 4.3, 4.3, 2.7]; 
y_pat = [-1000 -1000 2000 2000];

fig = findobj('Name', 'HOR new plot');
    if ~isempty(fig), close(fig); end
    figure('Name','HOR new plot') ; 
    subplot(211); hold on; title("Short-latency Window                    Medium-latency Window"); xlim([0.5 4.5])
        ylabel("Normalize" +newline+ "Soleus activity")
           
        %Individuel SLR
        plot(ones(size(names)),   v_box_sol_slr_avg{1}, ".", "color", color_idv)
        plot(ones(size(names))+1, v_box_sol_slr_avg{2}, ".", "color", color_idv)
        for sub = 1:numel(names)
            plot([1,2],[v_box_sol_slr_avg{1}(sub),v_box_sol_slr_avg{2}(sub)], "color", color_idv)
        end

        %Individuel MLR
        plot(ones(size(names))+2,   v_box_sol_mlr_avg{1}, ".", "color", color_idv)
        plot(ones(size(names))+3, v_box_sol_mlr_avg{2}  , ".", "color", color_idv)
        for sub = 1:numel(names)
            plot([3,4],[v_box_sol_mlr_avg{1}(sub),v_box_sol_mlr_avg{2}(sub)], "color", color_idv)
        end

        %Mean SLR
        plot(1, mean(v_box_sol_slr_avg{1}),"o", 'linewidth', 1,'color',color_mean )
        plot(2, mean(v_box_sol_slr_avg{2}),"o", 'linewidth', 1,'color',color_mean)
        plot([1,2], [mean(v_box_sol_slr_avg{1}),mean(v_box_sol_slr_avg{2})], 'linewidth',linewidth_mean, "color", color_mean) 

        %Mean MLR
        plot(3, mean(v_box_sol_mlr_avg{1}),"o", 'linewidth', 1,'color',color_mean )
        plot(4, mean(v_box_sol_mlr_avg{2}),"o", 'linewidth', 1,'color',color_mean)
        plot([3, 4], [mean(v_box_sol_mlr_avg{1}),mean(v_box_sol_mlr_avg{2})], 'linewidth',linewidth_mean, "color", color_mean) 

        plot([1.2,1.8], [2.5, 2.5],'-|', 'linewidth',1.5, "color", color_mean) 
        plot([1.5], [2.5],'x', 'linewidth',1.5, "color", "Black") 

        YL = get(gca, 'YLim'); ylim([YL(1) YL(2)])        
        patch(x_pat_slr,y_pat,patchcolor_SLR,'FaceAlpha',FaceAlpha, 'EdgeColor', EdgeColor, 'LineWidth', lineWidth_patch)
        patch(x_pat_mlr,y_pat,patchcolor_MLR,'FaceAlpha',FaceAlpha, 'EdgeColor', EdgeColor, 'LineWidth', lineWidth_patch)

        xticks([1 2 3 4]);
        xticklabels({'Control', 'Perturbation', 'Control', 'Perturbation'});
        grid on;

    subplot(212); hold on; title("Short-latency Window                    Medium-latency Window"); xlim([0.5 4.5])
        ylabel("Normalize" +newline+ "Tibialis activity")
            
%Individuel SLR
        plot(ones(size(names)),   v_box_ta_slr_avg{1}, ".", "color", color_idv)
        plot(ones(size(names))+1, v_box_ta_slr_avg{2}, ".", "color", color_idv)
        for sub = 1:numel(names)
            plot([1,2],[v_box_ta_slr_avg{1}(sub),v_box_ta_slr_avg{2}(sub)], "color", color_idv)
        end

        %Individuel MLR
        plot(ones(size(names))+2,   v_box_ta_mlr_avg{1}, ".", "color", color_idv)
        plot(ones(size(names))+3, v_box_ta_mlr_avg{2}  , ".", "color", color_idv)
        for sub = 1:numel(names)
            plot([3,4],[v_box_ta_mlr_avg{1}(sub),v_box_ta_mlr_avg{2}(sub)], "color", color_idv)
        end

        %Mean SLR
        plot(1, mean(v_box_ta_slr_avg{1}),"o", 'linewidth', 1,'color',color_mean )
        plot(2, mean(v_box_ta_slr_avg{2}),"o", 'linewidth', 1,'color',color_mean)
        plot([1,2], [mean(v_box_ta_slr_avg{1}),mean(v_box_ta_slr_avg{2})], 'linewidth',linewidth_mean, "color", color_mean) 

        %Mean MLR
        plot(3, mean(v_box_ta_mlr_avg{1}),"o", 'linewidth', 1,'color',color_mean )
        plot(4, mean(v_box_ta_mlr_avg{2}),"o", 'linewidth', 1,'color',color_mean)
        plot([3, 4], [mean(v_box_ta_mlr_avg{1}),mean(v_box_ta_mlr_avg{2})], 'linewidth',linewidth_mean, "color", color_mean) 



        YL = get(gca, 'YLim'); ylim([YL(1) YL(2)])        
        patch(x_pat_slr,y_pat,patchcolor_SLR,'FaceAlpha',FaceAlpha, 'EdgeColor', EdgeColor, 'LineWidth', lineWidth_patch)
        patch(x_pat_mlr,y_pat,patchcolor_MLR,'FaceAlpha',FaceAlpha, 'EdgeColor', EdgeColor, 'LineWidth', lineWidth_patch)

        xticks([1 2 3 4]);
        xticklabels({'Control', 'Perturbation', 'Control', 'Perturbation'});
        grid on;

    % Ny boxplot 
    v_box_soleus_slr   = {squeeze(v_pert.CTL(:,SOL,SLR,:))',squeeze(v_pert.VER(:,SOL,SLR,:))'};  
    v_box_soleus_mlr   = {squeeze(v_pert.CTL(:,SOL,MLR,:))',squeeze(v_pert.VER(:,SOL,MLR,:))'};  
    v_box_sol_slr_avg  = {squeeze(mean(v_pert.CTL(:,SOL,SLR,:),4)), squeeze(mean(v_pert.VER(:,SOL,SLR,:),4))};
    v_box_sol_mlr_avg  = {squeeze(mean(v_pert.CTL(:,SOL,MLR,:),4)), squeeze(mean(v_pert.VER(:,SOL,MLR,:),4))};

    % Ny boxplot 
    v_box_tibialis_slr   = {squeeze(v_pert.CTL(:,TA,SLR,:))',squeeze(v_pert.VER(:,TA,SLR,:))'};  
    v_box_tibialis_mlr   = {squeeze(v_pert.CTL(:,TA,MLR,:))',squeeze(v_pert.VER(:,TA,MLR,:))'};  
    v_box_ta_slr_avg  = {squeeze(mean(v_pert.CTL(:,TA,SLR,:),4)), squeeze(mean(v_pert.VER(:,TA,SLR,:),4))};
    v_box_ta_mlr_avg  = {squeeze(mean(v_pert.CTL(:,TA,MLR,:),4)), squeeze(mean(v_pert.VER(:,TA,MLR,:),4))};



[~,h] = signrank(v_box_sol_slr_avg{1}, v_box_sol_slr_avg{2});
disp("SOL, SLR " + h)
[~,h] = signrank(v_box_sol_mlr_avg{1}, v_box_sol_mlr_avg{2});
disp("SOL, MLR " + h)
[~,h] = signrank(v_box_ta_slr_avg{1}, v_box_ta_slr_avg{2});
disp("TA, SLR " + h)
[~,h] = signrank(v_box_ta_mlr_avg{1},v_box_ta_mlr_avg{2});
disp("TA, MLR " + h)



p = 0.33; q = 0.67; 

rate = 1/p + 1/(p*g); 
for i = 2:18
    rate = rate + 1/(p*g^i)
end 



%% Task 4.2 Vertical FC deprresion 
fprintf('script: TASK 4.2  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic

show_plt    = false; 
show_boxplt = true; 
savepgn     = false; 

subject = 3; 
x_range = [-500 200]; 
plt_show = [ANG, VEL, SOL, TA]; 
box_dim = 1:11; 



% Defined size for window-analysis
SLR = 1; MLR = 2;
pre_search = [0 , 1]; % denoted in ms
SLR_search(1,:) = [44, 72]; % denoted in ms  
SLR_search(2,:) = [50, 94]; % denoted in ms  
SLR_search(3,:) = [33, 49]; % denoted in ms  
SLR_search(4,:) = [1, 1]; % denoted in ms  
SLR_search(5,:) = [44, 57]; % denoted in ms  
SLR_search(6,:) = [1, 1]; % denoted in ms  
SLR_search(7,:) = [3, 55]; % denoted in ms  
SLR_search(8,:) = [-13, 27]; % denoted in ms  





%   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
offset_ms([1,2,3,4]) = [53,88,38,33];  
offset_ms([5,6,7,8]) = [39,34,22,50];

offset_sample = ms2sec(offset_ms)*Fs; 
for sub = 1:numel(names)
    data = total_data_vertical{1,1,sub};  
    Nsweep = size(data{VER,SOL},1); 
    for type = [SOL, TA, ANG, VEL]
        data{VER,type}(:,:) = [zeros(Nsweep, offset_sample(sub)), data{VER,type}(:,1:end-offset_sample(sub))];
    end
    total_data_vertical{1,1,sub} = data; 
end

% Load data
data = total_data_vertical{1,1,subject}; 
type = total_type_vertical{1,1,subject}; no = type{2}; yes = type{1};
x_axis = data{VER,time};



if show_plt
    % Patch properties 
    patchcolor = [251 244 199]/255; 
    FaceAlpha = 0.4; 
    patX_predict = [pre_search(1) pre_search(2) pre_search(2) pre_search(1)];    % ms 
    patX_slr     = [SLR_search(subject,1) SLR_search(subject,2) SLR_search(subject,2) SLR_search(subject,1)];    % ms
    patY = [-1000 -1000 1000 1000];
    EdgeColor = [37 137 70]/255;
    lineWidth_patch = 0.5;

    % Plot properties 
    color_no = [0.75, 0.75, 0.75]; 
    color_yes = "black"; 
    linewidth_no = 3; 
    linewidth_yes = 1; 
    
    % Check if a figure with the name 'TASK3' is open
    fig = findobj('Name', 'Vertical Perturbation2');
    if ~isempty(fig), close(fig); end
    
    % Begin plot 
    figure('Name', 'Vertical Perturbation2'); hold on; 
    sgtitle("Subject: " + subject)
    for i = 1:numel(plt_show)
        subplot(numel(plt_show), 1, i); hold on
        ylabel(labels_ms(plt_show(i)))
        xlim(x_range) 

        plot(x_axis, mean(data{VER,plt_show(i)}(no(box_dim),:),1),  'color', color_no,  "linewidth", linewidth_no)
        plot(x_axis, mean(data{VER,plt_show(i)}(yes(box_dim),:),1), 'color', color_yes, "linewidth", linewidth_yes)
        YL = get(gca, 'YLim'); ylim([YL(1) YL(2)]);
        if or(plt_show(i) == ANG, plt_show(i) == VEL)  
            patch(patX_predict, patY, patchcolor,'FaceAlpha',FaceAlpha, 'EdgeColor', EdgeColor, 'LineWidth', lineWidth_patch) % predictor
        else
            patch(patX_slr, patY, patchcolor,'FaceAlpha',FaceAlpha, 'EdgeColor', EdgeColor, 'LineWidth', lineWidth_patch)     % depended 1
        end 
        set(gca, 'Layer', 'top') 
        plot(x_axis, mean(data{VER,plt_show(i)}(no(box_dim),:),1), 'color', color_no, "linewidth", linewidth_no)
        plot(x_axis, mean(data{VER,plt_show(i)}(yes(box_dim),:),1), 'color', color_yes, "linewidth", linewidth_yes)
    end
    filename = "Subject"+subject+".png";
    filepath = 'C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/png files/vertical/';
    fullpath = fullfile(filepath, filename);
    
    if savepgn
        saveas(gcf, fullpath, 'png'); 
    end

    fprintf('done [ %4.2f sec ] \n', toc);
else 
    fprintf('disable \n');
end





