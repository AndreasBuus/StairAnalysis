clc 
clear; 
close all; 

%% Load data and define abbreviation
fprintf('script: Folder .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic

names = ["Pegah", "Nikolas"]; 

% find the current folder path 
folderpath = strrep(fileparts(matlab.desktop.editor.getActiveFilename),'\','/'); 

% Define data path
addpath(folderpath +"/FunctionFiles")
folderpath_preprocessed_data = folderpath + "/data_preprocessed/"; 
folderpath_stepIndex_data = folderpath + "/step_indexs/"; 

% Preallocation
total_data = cell(1,1,numel(names));
total_type = cell(1,1,numel(names)); 
total_step = cell(1,1,numel(names)); 

% Protocol abbreviation types
CTL = 1;     % Control perturbation trials
HOR = 2;     % Horizontal perturbation trials
VER = 3;     % Vertical perturbation trials
protocol_all = [CTL];

% Sensor abbreviation type
SOL = 1;    % Soleus []
TA = 2;     % Tibialis []
ANG = 3;    % Ankel position [deg]
FSR = 4;    % Foot force switch 
time = 5;   % Time [ms]
VEL = 6;    % Velocity [deg/ms]
ACC = 7;    % Acceleration [deg/ms^2]
step = 8;   % Step index


% Preallocation
total_data = cell(1,1,numel(names));
total_type = cell(1,1,numel(names)); 
total_step = cell(1,1,numel(names)); 

% Load preproccessed data  
for sub = 1:length(names)
    load(folderpath_preprocessed_data + names(sub) + "_data.mat");   % example: data{protocol, sensor}(sweep, data number)
    total_data{:,:,sub} = data;

    load(folderpath_stepIndex_data + names(sub) + "_offset.mat");       % manipulated data
    %load(folderpath_stepIndex_data + names(sub) + "_step_index.mat");  % raw and properly errored data
    total_step{:,:,sub} = offset;
end 


% Plotting labels 
labels = ["Soleus"; "Tibialis"; "Position"; "FSR";  "Time"; "Velocity"; "Acceleration"];
labels_ms = ["Soleus"+newline+"[\muV]";"Tibialis"+newline+"[\muV]"; "Position"+newline+"[Deg]";  "";  "Time"+newline+"[ms]"; "Velocity"+newline+"[Deg/ms]";"Acceleration"+newline+"[Deg/ms^2]"];
labels_sec = ["Soleus"+newline+"[\muV]";"Tibialis"+newline+"[\muV]"; "Position"+newline+"[Deg]"; "";  "Time"+newline+"[sec]";"Velocity"+newline+"[Deg/s]";"Acceleration"+newline+"[Deg/s^2]"];

% Global arrays
align_with_obtions = ["second_begin", "four_begin", "six_begin"];
steps_tested = [2,4,6];

% Global function
ms2sec = @(x) x*10^-3;         % ms to sec 
sec2ms = @(x) x*10^3;          % Sec to ms 

% Global window definition
screensize = get(0,'ScreenSize');
width = screensize(3);
height = screensize(4);

Fs = 2000;                      % Samples per second
dt = 1/Fs;                      % Seconds per sample
pre_trig = 4;                   % Pre-trigger 

fprintf('done [ %4.2f sec ] \n', toc);


%% Normalize EMG (make as a function instead) 
fprintf('script: Normalize EMG   .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic
normalize = true;      % enable or disable
span = 20;             % how big is the smooth span.
normalizing_step = 2;  % which step is the data being normalized to [0,2,4]?
%  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . 
if normalize 
    if (exist('normalize_done', 'var') == 1)
        fprintf(2, 're-entry prevented \n');
    else
        % Normalize prebaseline control and horizontal 
        for sub = 1:length(names) % loop through subjects  
            % Load data 
            data = total_data{1,1,sub};
            step_index = total_step{1,1,sub};
            [data] = func_normalize_EMG(step_index, data, 'protocols', CTL,  'normalize_to_step', normalizing_step,'span', span);
            total_data{1,1,sub} = data; % update data
        end
        normalize_done = true; 
        fprintf('done [ %4.2f sec ] \n', toc);
    end
else 
    fprintf('disable \n');
end 

clear data factor

%% Remove saturated data 
fprintf('script: Remove saturated data .  .  .  .  .  .  .  .  .  .  .  .  '); tic
remove_saturated = true;    % enable or disable 
threshold = -10;             % remove ANG data if lower than threshold
span = [0, 20];              % ms 
%  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . 
if remove_saturated  
    % From ms to samples 
    span = ms2sec(span)*Fs;        % from ms to sample
    exc_ctl = cell(size(names));    % exclude Control
    for sub = 1:numel(names) % loop through subjects
        % Load data 
        data = total_data{1,1,sub}; 
        step_index = total_step{1,1,sub};      
        % Find saturated idxs
        for sweep = 1:size(data{CTL,FSR},1) % loop through sweeps 
            for step = 1:3 % loop through steps 
                [rise] = func_find_edge(steps_tested(step)); 
                edge = step_index{CTL}(sweep,rise);
                y = data{CTL,ANG}(sweep, span(1)+edge:span(2)+edge); 
                if any(find(y<threshold))
                    exc_ctl{sub} = unique([exc_ctl{sub}, sweep]);  
                end
            end 
        end
        % Remove saturated idxs
        step_index{CTL}(exc_ctl{sub},:) = []; 
        for i = [SOL, TA, FSR, ANG]
            data{CTL,i}(exc_ctl{sub},:) = []; 
        end 
        % Display which step to remove
        if ~isempty( exc_ctl{sub} )
            msg = "\n     Saturated data. Subject: " + sub + ". Sweep: " + num2str(exc_ctl{sub}) + " "; 
            fprintf(2,msg);  
        end 
        % Save data
        total_data{1,1,sub} = data; 
        total_step{1,1,sub} = step_index;
    end
    fprintf('\n     done [ %4.2f sec ] \n', toc);
else 
    fprintf('disable \n');
end 

clear exc_ctl data step_index y


%% Speed and aceleration (make as a function instead) 
fprintf('script: Speed and aceleration .  .  .  .  .  .  .  .  .  .  .  .  '); tic
plot_data = false;          % enable or disable plot
span_position = 10;          % inc. sample in guassian filter span 
span_velocity = 30;          % inc. sample in guassian filter span 
span_acceleration = 10;     % inc. sample in gaussian filter span 
fc = 50;                           % Cutoff frequency for LowPass filter
order = 1;                          % Filter order 
[b,a] = butter(order,fc/(Fs/2));    % Filter coefficient
%  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . 
if (exist('velocity_done', 'var') == 1)
    fprintf(2, 're-entry prevented \n');
else
    % Control and Horizontal trials
    for sub = 1:length(names) % subjects
        data = total_data{1,1,sub};  % load data 
        for proto = protocol_all % protocols 
            order = 1; row  = 2; 
    
            % Position
            data{proto,ANG} = data{proto,ANG}; % rescale the signal; 
            posi = data{proto,ANG};
            data{proto,ANG} = smoothdata(data{proto,ANG}, row, 'gaussian', span_position);    % gaussian smoothing
            
            % Velocity
            diffs1 = diff(data{proto,ANG}, order, row)./(dt*10^3);            % [deg/sample]
            diffs1 = padarray(diffs1, [0 1], 'post');                   % zeropadding            
            lowpass1 = filter(b, a, diffs1, [], row);
            smooth1 = smoothdata(diffs1, row, 'gaussian', span_velocity);    % gaussian smoothing
            data{proto,VEL} = smooth1;
    
            % acceleration
            diffs2_raw = diff(diffs1, 1, 2)./(dt*10^3);                % [deg/sample^2]
            diffs2_raw = padarray(diffs2_raw, [0 1], 'post');          % zeropadding
            diffs2_smooth = diff(data{proto,VEL}, 1, 2)./(dt*10^3);    % [deg/sample^2]
            diffs2_smooth = padarray(diffs2_smooth, [0 1], 'post');    % zeropadding       
            data{proto,ACC} = diffs2_raw; 
    
            % Need to plot before and after 
            if plot_data == 1 && sub == 1 && proto == CTL 
                step_index = total_step{1,1,sub};
                
                sweep = 10; dur = 2000; before = 200;
                [rise, ~] = func_find_edge(4);
                rise_index = step_index{CTL}(sweep,rise);
                display_array = rise_index-before : rise_index+dur;
                L = length(display_array); 
    
                P2_raw = abs(fft(diffs1(sweep,display_array))/L);
                P1_raw = P2_raw(1:L/2+1);
                P1_raw(2:end-1) = 2*P1_raw(2:end-1);
    
                P2_lp = abs(fft(lowpass1(sweep,display_array))/L);
                P1_lp = P2_lp(1:L/2+1);
                P1_lp(2:end-1) = 2*P1_lp(2:end-1);
    
                P2_smo = abs(fft(smooth1(sweep,display_array))/L);
                P1_smo = P2_smo(1:L/2+1);
                P1_smo(2:end-1) = 2*P1_smo(2:end-1);
                f = Fs*(0:(L/2))/L;
    
                raw_color = [0.7 0.7 0.7]; 
                smo_color = 'blue'; 
                lp_color  = 'red'; 
                
                figure; hold on
                subplot(411); hold on; 
                plot(display_array, posi(sweep,display_array),'blue', display_array, data{proto,ANG}(sweep,display_array),'red')
                YL = get(gca, 'YLim'); ylim([YL(1) YL(2)]);
                plot([rise_index, rise_index],[YL(1) YL(2)])
                legend(["raw", "filtered", ""])
                subtitle("position")
    
                subplot(412); hold on; xlim([0 200])
                plot(f,P1_raw, 'linewidth', 4,'color', raw_color); 
                plot(f,P1_lp, lp_color);
                plot(f,P1_smo, smo_color);
                legend(["raw", "LP", "Smooth"])
                subtitle("frequency")
    
                subplot(413); hold on
                plot(display_array,zeros(size(display_array)), 'linewidth', 1, 'color', [0.9 0.9 0.9])          % foot contact
                plot(display_array, diffs1(sweep,display_array), 'linewidth', 2,'color', raw_color) % raw
                plot(display_array, lowpass1(sweep,display_array), lp_color)      % low passed
                plot(display_array, smooth1(sweep,display_array), smo_color)      % smoothed
                legend(["","raw", "LP", "Smooth"])
                subtitle("velocity")

    
                subplot(414); hold on
                plot(display_array, diffs2_raw(sweep,display_array), 'linewidth', 2,'color', raw_color) % raw            data{proto,ACC} = smoothdata(diffs2, 2, 'gaussian', span_acceleration);    % gaussian smoothing
                plot(display_array, diffs2_smooth(sweep,display_array), smo_color)      % smoothed
                legend(["raw","Smooth"])
                subtitle("acceleration")
            end
        end 
        velocity_done = true;
        total_data{1,1,sub} = data; 
    end
end
fprintf('done [ %4.2f sec ] \n', toc);


%% Weighted data 
fprintf('script: weighting_data  .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic
% Adjust the sample size to be similar 
weighting_data = true; 
inc_sub = 1:length(names);  % this include all subjects
%  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  
if weighting_data
    % Find the subject with smallest subject size
    smallestSampleSize = 50; 
    for sub = 1:numel(inc_sub)
        data = total_data{1,1,sub}; 
        sub_size = size(data{CTL,ANG},1); % sweep size 
        if smallestSampleSize > sub_size; smallestSampleSize = sub_size; end 
    end 
    disp("Smallest sample size was " + smallestSampleSize)
    % Remove sweeps larger than smallest subject size
    for sub = 1:numel(inc_sub)
        data = total_data{1,1,sub};         % load data
        step_index = total_step{1,1,sub};   % load data 
        sub_size = size(data{CTL,ANG},1);   % sweep size 
        if sub_size > smallestSampleSize 
            step_index{CTL}(smallestSampleSize+1:end,:) = []; 
            for i = [SOL, TA, FSR, ANG]
                data{CTL,i}(smallestSampleSize+1:end,:) = []; 
            end 
        end
        total_data{1,1,sub} = data;         % save data
        total_step{1,1,sub} = step_index;   % save data
    end 
    fprintf('done [ %4.2f sec ] \n', toc);
else 
    fprintf('disable \n');
end

%% Task 0.1: Show average sweep for single subject
fprintf('script: TASK 0.2  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic

show_plot = true;       % Disable or enable plot
subject =  2;            % Obtions: 1:8
proto = CTL;            % Obtions: CTL, VER, HOR 
str_sen = ["Position", "Soleus", "Tibialis"];    % Obtions: "Soleus", "Tibialis","Position", "Velocity", "Acceleration"; 
show_FSR = true; 
align_bool = true;      % Should the data be aligned with step specified in "Align with specific Stair step"
    alignWithStep = "four_begin"; % Obtions: "second_begin", "four_begin", "six_begin"
    before = 500; 
    after = 200; 

% .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
if show_plot
    fprintf(' plot subject: >> ' + names(subject) + ' << \n' );

if align_bool
    % Load data and align with defined step 
    data = total_data{1,1,subject}; 
    step_index = total_step{1,1,subject};
    temp = cell(3,7); 
    [temp{proto,:}] = func_align(step_index{proto}, data{proto,[1:4,6:7]}, 'sec_before', ms2sec(before), 'sec_after', ms2sec(after), 'alignStep', alignWithStep);
    clear data
    data = temp;
    x_axis = data{proto,time};
    x_axis = sec2ms(x_axis);
    str_xlabel = "Time [ms]" + newline + "Data normalized to step four"; 
else 
    % Load data 
    data = total_data{1,1,subject}; 
    N = size(data{1,1},2); 
    x_axis = linspace(-4, (N*dt)-4-dt, N); 
    x_axis = sec2ms(x_axis); 
    str_xlabel = "Time [ms]" + newline + "Data normalized to Force-Platform";
end 

switch proto
    case HOR
        type = total_type{:,:,subject};
        yes = type{3}; no = type{4}; str_title = "Horizontal perturbation"; 
    case VER
        type = total_type{:,:,subject};
        yes = type{1}; no = type{2}; str_title = "Vertical perturbation"; 
    case {CTL}
        str_title = "Pre-baseline Control";
end

figure; 
   % sgtitle(str_title + " - subject " + names(subject)); 
    for i = 1:size(str_sen,2) % check and plot data included in str_sen
        switch str_sen(i) 
            case "Soleus"
                sensor_modality = SOL; 
                str_ylabel = "Soleus" + newline + "[\muV]"; 
            case "Tibialis"
                sensor_modality = TA; 
                str_ylabel = "Tibialis"+newline+"[\muV]"; 
            case "Position"
                sensor_modality = ANG;
                str_ylabel = "Position" + newline + "[Deg]" + newline + "[Dorsal] <  > [Plantar]"; 
            case "Velocity"
                sensor_modality = VEL; 
                str_ylabel = "Velocity" + newline + "[Deg/s]"; 
            case "Acceleration"
                sensor_modality = ACC; 
                str_ylabel = "Acceleration " + newline + "[Deg/s^2]"; 
             otherwise
                disp("ERROR" + newline + "String: >>" + str_sen(i) + "<< is not registered.")
        end
        
        switch proto
            case {VER, HOR} 
                subplot(size(str_sen,2)*100 + 10 + i); hold on; ylabel(str_ylabel); %xlim([0, inf])
                plot(x_axis, mean(data{proto,sensor_modality}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
                plot(x_axis, mean(data{proto,sensor_modality}(yes,:),1), "LineWidth",1, "color","black")

                if show_FSR
                    y_fsr = rescale(mean(data{proto,FSR},1)); 
                    yyaxis right; ylabel("Phase"); ylim([-0.1 1.1])
                    plot(x_axis, y_fsr, "color",	"yellow");
                end
                
            case {CTL, CTL2}
                subplot(size(str_sen,2)*100 + 10 + i); hold on; ylabel(str_ylabel); % xlim([0, inf])
                % plt std around mean 
                y = mean(data{proto,sensor_modality},1); 
                std_dev = std(data{proto,sensor_modality});
                curve1 = y + std_dev;
                curve2 = y - std_dev;
                x2 = [x_axis, fliplr(x_axis)];
                inBetween = [curve1, fliplr(curve2)];
                fill(x2, inBetween, [0.75, 0.75, 0.75], 'LineStyle','none'); 
                plot(x_axis, y, 'LineWidth', 1, "color","black"); 

                if show_FSR % show FSR if enabled 
                    y_fsr = rescale(mean(data{proto,FSR},1)); 
                    yyaxis right; ylabel("Phase"); ylim([-0.1 1.1])
                    plot(x_axis, y_fsr, "color",	"yellow");
                end
        end
        if i == size(str_sen,2)
            xlabel(str_xlabel)
        end 
    end

    clear y_fsr
else 
    fprintf('disable \n');
end 


%% Task 0.2: Show individual sweep for single subject
% undersøg for metodisk fejl.
fprintf('script: TASK 0.3  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic

show_plot = false;       % Disable or enable plot
subject = 1;            % Obtions: 1:9
proto = CTL;            % Obtions: CTL
sensor_modality = SOL;  % Obtions: SOL, TA, ANG, VEL, ACC
before = 50;            % before foot strike included in ms
after = 0;              % after foot lift-off included in ms

% .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
if show_plot
fprintf('plot subject: >> ' + names(subject) + ' << \n' );
data = total_data{1,1,subject}; 
step_index = total_step{1,1,subject};
N = size(data{1,1},2); 
x_axis_total = linspace(-4, (N*dt)-4-dt, N); % time axis 
sweepNum = size(data{proto,ANG},1);   % total sweep size


figure('name','control sweep'); % Begin plot
    loop = true; sweep = 1; 
    while loop == true
        clc % clear cmd promt 
        sgtitle("Sweep: " + sweep) % display current sweep in promt

        subplot(211);  
        % plot data 
        yyaxis left;
        plot(x_axis_total, mean(data{proto,sensor_modality},1), '-', "LineWidth",3, "color",[0.75, 0.75, 0.75]) % mean plt 
        hold on 
        plot(x_axis_total, data{proto,sensor_modality}(sweep,:), '-', "LineWidth",1, "color","black") % sweep plt 
        hold off
      
        % plt formalia 
        xlabel("Time"+newline+"[sec]")
        ylabel(labels_ms(sensor_modality))
        title(['Black graph, Sweep data. {\color{gray} Gray graph, Mean data [n=' num2str(sweepNum) '].}'])

        if show_FSR
            y_fsr = rescale(data{proto,FSR}(sweep,:)); 
            yyaxis right; ylabel("Phase"); ylim([-0.1 1.1]); 
            plot(x_axis_total,y_fsr, "color",	"red"); 
        end
    
        % Define the data for each step 
        clear y
        for k = 1:3
            clear data_align
            data_align = cell(3,7); 
            [data_align{proto,:}] = func_align(step_index{proto}, data{proto,[1:4,6:7]}, 'sec_before', ms2sec(before), 'sec_after', ms2sec(after), 'alignStep', align_with_obtions(k));
            x_axis = data_align{proto,time};
            y{k,2} = sec2ms(x_axis);
            y{k,1} = data_align{proto,sensor_modality}(sweep,:); 
            y{k,3} = mean(data_align{proto,sensor_modality},1);
        end 
    
        % plot data for each step 
        for k = 1:3 % loop through steps
            subplot(233+k); hold on; 
            if sensor_modality == or(SOL,TA)
                ylim([0 ceil(max([max(y{1,1}),  max(y{2,1}), max(y{3,1})])/100)*100])
            end
            plot(y{k,2}, y{k,3}, "LineWidth",2, "color", [0.75, 0.75, 0.75]) % Mean
            plot(y{k,2}, y{k,1}, "LineWidth",1, "color","black") % Sweep
            ylim auto
            YL = get(gca, 'YLim'); ylim([YL(1) YL(2)]);
            plot([0, 0],[YL(1) YL(2)], "--","LineWidth",1, "Color", "red") % plt fsr 
    
            % plt formalia 
            xlabel("Time"+newline+"[ms]")
            title("Step: " + steps_tested(k))
            ylabel(labels_ms(sensor_modality))
        end

        % Wait for user input
        correctInput = false; 
        prompt = "Continue, press >c<" + newline + "Quite, press >q<"+ newline + "Change sweep number, press >t<"+ newline;    
        while correctInput == false     % Wait for correct user input
            str = input(prompt, 's');   % Save user input
            if strcmp(str,"q")          % If pressed - Quite the loop 
                disp("Loop stopped")
                loop = false; correctInput = true; 
                fig = findobj('Name', 'control sweep');
                if ~isempty(fig), close(fig); end
            elseif strcmp(str,"t")      % If pressed - Change sweep number
                sweep = input("New sweep number: ")-1; 
                correctInput = true; 
            elseif strcmp(str,"c")      % If pressed - Continue to next sweep
                correctInput = true; 
            end 
    
            if correctInput == false
                warning("Input not accepted")
            end
        end
        sweep = sweep + 1; % Continue to next sweep
        if sweep > size(data{proto,sensor_modality},1) % stop loop if max sweep reached
            loop = false; 
        end 
    end
else 
    fprintf('disable');
end 


%% Define windows
fprintf('script: TASK 0.3  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic

inc_sub = 1:length(names);
dep_off = 39; % depended search window beginning after foot contact
dep_len = 20; % depended search window length 

% .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
for step = [2,4,6]
    predict_search(step,:) = [0, 20]; 
    depend_search(step,:) = [predict_search(step,1)+dep_off, predict_search(step,1)+dep_off+dep_len];
end
% Preallocation
predictor_value = cell(size(names)); 
depended_value = cell(size(names)); 



for sub = inc_sub   % loop through subjects 
    % Load data from defined subject defined by loop
    data = total_data{1,1,sub};  
    step_index = total_step{1,1,sub};
        
    % Find predictor and depended 
    for k = 1:3                      
        step = steps_tested(k);
       
        [rise, fall] = func_find_edge(step);                    % rise and fall value for the given step
        falling = []; falling = step_index{CTL}(:,fall);   % fall indexes for all sweeps
        rising  = []; rising  = step_index{CTL}(:,rise);   % rise indexes for all sweeps  
        step_sec{sub}(k,:) = (falling(:) - rising(:))*dt;       % duration of each step phase, unit [sec]

        % Re-define window ms to sample
        predict_search_index = floor(ms2sec(predict_search(step,:))*Fs);  % unit [sample]
        depend_search_index  = floor(ms2sec(depend_search(step,:))*Fs);   % unit [sample]

        for sweep = 1:size(data{CTL,1},1)     
            % Find rise index for the given step and define window 
            rise_index = step_index{CTL}(sweep,rise);
           
            % clear values from previous subject
            predict_search_array = []; depend_search_array  = [];

            predict_search_array = predict_search_index(1)+rise_index : predict_search_index(2)+rise_index; 
            depend_search_array = depend_search_index(1)+rise_index : depend_search_index(2)+rise_index; 
          

            % PREDICTOR VALUES:
            pos = (data{CTL,ANG}((sweep),:)); 

            % The mean of the window span 
            % predictor_value{sub}(step,sweep) = mean(pos{predict_search_array),2); 

            % The max of the window span
            % predictor_value{sub}(step,sweep) =  max(ang_data(predict_search_array)); 
            
            % The difference from start to end window in velocity
            predictor_value{sub}(step,sweep) =  (pos(predict_search_array(1)) - pos(predict_search_array(end))) / sec2ms(diff(predict_search(step,:))*dt);
    
            % The difference from start window to smallest value in window in velocity
            % x = find(min(ang_data(predict_search_array)) == ang_data(predict_search_array)); 
            % predictor_value{sub}(step,sweep) =  (ang_data(predict_search_array(1)) - min(ang_data(predict_search_array))) / sec2ms(x*dt);
            
            % DEPENDED VALUES 
            for EMG = [SOL, TA]
                EMG_data = (data{CTL,EMG}((sweep),:)); 
            
                % EMG activity for the whole standphase 
                step_EMG{sub}(EMG,step,sweep) = mean(EMG_data(rising(sweep) : falling(sweep)));  

                % Mean EMG activty during the window span
                depended_value{sub}(EMG, step, sweep) =  mean(EMG_data(depend_search_array));
            end
        end %sweep
    end %step 
end % sub


%% Task 1: FC correlation with EMG (Seperate steps, Single subject) 

show_plot = true;           % plot the following figures for this task
subject = 2;                % subject to analyse
 % Xlim in plotting
before = 500;
after = 500;
xlimits = [-before after];

if show_plot
    % Load data for plot, defined by >> subject <<
    data = total_data{1,1,subject};  
    step_index = total_step{1,1,subject};
    
    % Plot figure
    fig = findobj('Name', 'Pre-baseline');
    if ~isempty(fig), close(fig); end
    figSize = [50 50 floor(width/2) floor(height/2)]; % where to plt and size
    figure('Name', 'Pre-baseline','Position', figSize); % begin plot 
    sgtitle("Subject: " + subject + ". [n = " + size(data{CTL,1},1) + "]."); 

    % Patch properties 
    y_pat = [-1000 -1000 1000 1000];
    patchcolor_pre = [251,244,199]/255; 
    patchcolor_dep = "blue"; %[251,244,199]/255;
    patchcolor_dep_MLR = "red"; %[251,244,199]/255; 

    FaceAlpha = 1; 
    FaceAlpha_dep = 0.1;
    EdgeColor_pre = "none";%[37,137,70]/255;
    EdgeColor_dep = "none";% [37,137,70]/255;
    
    for k = 1:3 % loop through steps       
        x_pat_pre = [predict_search(steps_tested(k),1) predict_search(steps_tested(k),2) predict_search(steps_tested(k),2) predict_search(steps_tested(k),1)];
        x_pat_dep = [depend_search(steps_tested(k),1) depend_search(steps_tested(k),2) depend_search(steps_tested(k),2) depend_search(steps_tested(k),1)];
        x_pat_dep_MLR = [depend_search(steps_tested(k),1)+21 depend_search(steps_tested(k),2)+21 depend_search(steps_tested(k),2)+21 depend_search(steps_tested(k),1)+21];

        data_plot = cell(3,7); 
        [data_plot{CTL,:}] = func_align(step_index{CTL}, data{CTL,[1:4,6:7]}, 'sec_before', ms2sec(before), 'sec_after', ms2sec(after), 'alignStep', align_with_obtions(k));
        
    % Predictor 
        sensor_type = [ANG, VEL]; % Ankle and velocity
        for i = 1:numel(sensor_type)
            subplot(4,3,0+k + (i-1)*3); hold on; % Ankel 
            % Formalia setup
            ylabel(labels_ms(sensor_type(i)));
            if sensor_type(i) == ANG; title("Step " + k*2); end
            %subtitle("["+ predict_search(steps_tested(k),1) + " : "+predict_search(steps_tested(k),2)+" ms]")
            xlim(xlimits)

            % Plt data 
            y = mean(data_plot{CTL,sensor_type(i)},1); 
            std_dev = std(data_plot{CTL,sensor_type(i)});
            curve1 = y + std_dev;
            curve2 = y - std_dev;
            x_axis = data_plot{CTL,time};
            x_axis = sec2ms(x_axis);
            x2 = [x_axis, fliplr(x_axis)];
            inBetween = [curve1, fliplr(curve2)];   
            fill(x2, inBetween, [0.75, 0.75, 0.75], 'LineStyle','none'); 
            plot(x_axis, y, 'LineWidth', 1, "color","black"); 
            YL = get(gca, 'YLim'); ylim([YL(1) YL(2)]);
            patch(x_pat_pre,y_pat,patchcolor_pre,'FaceAlpha',FaceAlpha, 'EdgeColor', EdgeColor_pre)
            set(gca, 'Layer', 'top')
            fill(x2, inBetween, [0.75, 0.75, 0.75], 'LineStyle','none'); 
            plot(x_axis, y, 'LineWidth', 1, "color","black"); 
        end 
    % Depended
        sensor_type = [SOL, TA]; % Soleus and Tibialis
        for i = 1:numel(sensor_type)
            subplot(4,3, 6+k+(i-1)*3); hold on; 
            % Formalia setup
            ylabel(labels_ms(sensor_type(i))); 
            %subtitle(labels(sensor_type(i)) +" "+ depend_search(steps_tested(k),1) + " : "+depend_search(steps_tested(k),2)+"ms")            
            xlim(xlimits)
    
            % Plt data 
            y = mean(data_plot{CTL,sensor_type(i)},1); 
            std_dev = std(data_plot{CTL,sensor_type(i)});
            curve1 = y + std_dev;
            curve2 = y - std_dev;
            x_axis = data_plot{CTL,time};
            x_axis = sec2ms(x_axis);
            x2 = [x_axis, fliplr(x_axis)];
            inBetween = [curve1, fliplr(curve2)];   
            fill(x2, inBetween, [0.75, 0.75, 0.75], 'LineStyle','none'); 
            plot(x_axis, y, 'LineWidth', 1, "color","black"); 
            YL = get(gca, 'YLim'); ylim([YL(1) YL(2)]);
            patch(x_pat_dep,y_pat,patchcolor_dep,'FaceAlpha',FaceAlpha_dep, 'EdgeColor', EdgeColor_dep)
            patch(x_pat_dep_MLR,y_pat,patchcolor_dep_MLR,'FaceAlpha',FaceAlpha_dep, 'EdgeColor', EdgeColor_dep)

            set(gca, 'Layer', 'top')
            fill(x2, inBetween, [0.75, 0.75, 0.75], 'LineStyle','none'); 
            plot(x_axis, y, 'LineWidth', 1, "color","black"); 
        end   
    end 


    % begin plot
    fig = findobj('Name', 'Pre-baseline2');
    if ~isempty(fig), close(fig); end
    screensize = get(0,'ScreenSize');
    figSize = [100 100 floor(width/1.5) floor(height/2)]; % where to plt and size
    figure('Position',figSize,'Name', 'Pre-baseline2'); 

    marksColor = "blue";
    marker = ["*",".","x"];    
    sgtitle("Single subject [sub="+ subject +"]. Weighted: "+weighting_data)

    sensory_type = [SOL, TA]; 
    for i = 1:numel(sensory_type)
        depended = []; predictor = [];  
        for k = 1:3   
            subplot(2,5,k+(i-1)*5 ); hold on
                if k == 1; ylabel(labels(sensory_type(i))); end
                de = []; de = (squeeze(depended_value{subject}(sensory_type(i), steps_tested(k),:))); 
                pr = []; pr = (squeeze(predictor_value{subject}(steps_tested(k),:)));
                

                mdl = fitlm(pr, de);
                b = table2array(mdl.Coefficients(1,1)); 
                a = table2array(mdl.Coefficients(2,1)); 
                p_value = round(table2array(mdl.Coefficients(2,4)), 3);
                r2 = round(mdl.Rsquared.Adjusted, 3);
                linearReg = @(x) x*a + b;     
                plot(pr, de, marker(k), "color", marksColor)          
                plot(pr, linearReg(pr), "color", "black")
                   subtitle(['P-value: {\color{gray}' num2str(p_value) '}. R^2: {\color{gray}' num2str(r2) '}'])
                if p_value < 0.05
                    subtitle(['P-value: {\color{black} ' num2str(p_value) '}. R^2' num2str(r2)])
                else 
                    subtitle(['P-value: {\color{red} ' num2str(p_value) '}. R^2' num2str(r2)])
                end
                title("Step " + steps_tested(k))
                xlabel("Avg. deg/ms ")

            subplot(2,5,4+5*(i-1):5+5*(i-1)); hold on
                depended(k,:) = (squeeze(depended_value{subject}(sensory_type(i), steps_tested(k),:))); 
                predictor(k,:) = (squeeze(predictor_value{subject}(steps_tested(k),:)));
                plot(predictor(k,:), depended(k,:), marker(k), "color", marksColor)
        end
        
        % Linear regression 
        depended_all_step = [depended(1,:) depended(2,:) depended(3,:)];
        predictor_all_step = [predictor(1,:) predictor(2,:) predictor(3,:)];
        mdl = fitlm(predictor_all_step, depended_all_step);  
        b = table2array(mdl.Coefficients(1,1)); 
        a = table2array(mdl.Coefficients(2,1));
        p_value =  table2array(mdl.Coefficients(2,4)); 
        r2 = round(mdl.Rsquared.Adjusted, 3); 
        linearReg = @(x) x*a + b; 
        plot(predictor_all_step, linearReg(predictor_all_step), "color", "black")


        % Levene's test and anova
        depended_all_step = [depended(1,:)', depended(2,:)', depended(3,:)'];
        predictor_all_step = [predictor(1,:)', predictor(2,:)', predictor(3,:)'];
    
        [p_levene_de] = vartestn(depended_all_step,'TestType','LeveneAbsolute','Display', 'off'); 
        [p_anova_de] = anova1(depended_all_step, [], 'off');
        [p_levene_pr] = vartestn(predictor_all_step,'TestType','LeveneAbsolute','Display', 'off'); 
        [p_anova_pr] = anova1(predictor_all_step, [], 'off');
            title([' Depend: Levene: {\color{gray}' num2str(round(p_levene_de,3)) '.} Anova: {\color{gray}' num2str(round(p_anova_de,3)) '.}' newline 'Predict:  Levene: {\color{gray}' num2str(round(p_levene_pr,3)) '.} Anova: {\color{gray}' num2str(round(p_anova_pr,3)) '.}'])
       
        
        % Plt formalia 
        legend(["Data: Step 2", "Data: Step 4", "Data: Step 6","Fit"])
        subtitle(['P-value: {\color{gray} ' num2str(p_value) '}. R^2: {\color{gray}' num2str(r2) ' }'])

        if p_value < 0.05
            subtitle(['P-value: {\color{black} ' num2str(p_value) '}. R^2' num2str(r2)])
        else 
            subtitle(['P-value: {\color{red} ' num2str(p_value) '}. R^2' num2str(r2)])
        end
    end 

    % Set y-limits and x-limits on all subplots 
    ax = findobj(gcf, 'type', 'axes');
    ylims = get(ax, 'YLim'); 
    xlims = get(ax, 'XLim'); 
    [~, idx_y_ta]  = max(cellfun(@(x) diff(x), ylims(1:4)));
    [~, idx_y_sol] = max(cellfun(@(x) diff(x), ylims(5:8)));
    [~, idx_x_ta]  = max(cellfun(@(x) diff(x), xlims(1:4)));
    [~, idx_x_sol] = max(cellfun(@(x) diff(x), xlims(5:8)));
    idx_y_sol = idx_y_sol+4; 
    idx_x_sol = idx_x_sol+4; 
    for i = 1:numel(ax)
        if any(i == 1:4)
            set(ax(i), 'YLim', ylims{idx_y_ta})
            set(ax(i), 'XLim', xlims{idx_x_ta})
        elseif any(i == 5:8)  
            set(ax(i), 'YLim', ylims{idx_y_sol})
            set(ax(i), 'XLim', xlims{idx_x_sol})
        end
    end
end 



%% Task 2: FC correlation with EMG (All steps, all subject) 
fprintf('script: TASK 1.2  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic
show_plot = true; 

%  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . 
if show_plot
    reg_data = struct; 
    marker = ["*",".","x"];    

    % Re-arrange data from cell to struct. 
    data_reg = struct; 
    data_reg.dep_steps = cell(2,3);     % depended sortet each step,  [sol,ta]
    data_reg.pre_steps = cell(1,3);     % predictor sortet each step, [vel]
    data_reg.pre = [];                  % depended sortet all step,   [sol,ta]
    data_reg.dep = cell(2,1);           % predictor sortet all step,  [vel]

    % Re-arrange data from cell to array and save in struct
    for sub = inc_sub
        for step = 1:3 
            for EMG = [SOL,TA]
                data_reg.dep_steps{EMG,step} = [data_reg.dep_steps{EMG,step}, nonzeros(squeeze(depended_value{sub}(EMG, steps_tested(step),:)))' ]; 
                data_reg.dep{EMG} = [data_reg.dep{EMG}, nonzeros(squeeze(depended_value{sub}(EMG, steps_tested(step),:)))' ];
            end
            data_reg.pre = [data_reg.pre,  nonzeros(squeeze(predictor_value{sub}(steps_tested(step),:)))'];
            data_reg.pre_steps{step} = [data_reg.pre_steps{step}, nonzeros(squeeze(predictor_value{sub}(steps_tested(step),:)))'];
        end 
    end 

    % Defining plt window size
    figSize = [100 100 width-200 height-200]; % where to plt and size
    fig = findobj('Name', 'Pre-baseline all subject');
    if ~isempty(fig), close(fig); end
    figure('Name','Pre-baseline all subject','Position', figSize); % begin plot 
    sgtitle("Correlation. [Subject size="+numel(inc_sub)+"]. Weighted: "+weighting_data)
    
    
    % Plot subject data in different colors 
    for sub = inc_sub % subject
        for sensory_type = [SOL,TA] % muscle type
            depended = []; predictor = [];  
            for k = 1:3 % loop steps 
                depended(k,:) = nonzeros(squeeze(depended_value{sub}(sensory_type, steps_tested(k),:))); 
                predictor(k,:) = nonzeros(squeeze(predictor_value{sub}(steps_tested(k),:)));
                
                % Remember p-value and slope for later plot
                mdl = fitlm(predictor(k,:), depended(k,:));  
                reg_data.slopes(sub, sensory_type, k) = table2array(mdl.Coefficients(2,1));   % slopes
                reg_data.p_value(sub, sensory_type, k) = table2array(mdl.Coefficients(2,4));  % p_values

                % Plt the individuel steps
                subplot(2, 5, 5*(sensory_type-1)+k); hold on; % 5*(s-1)+k={1,2,3,6,7,8}, k={1,2,3}, s={1,2}            
                plot(predictor(k,:), depended(k,:),'x');
                title("Step " + steps_tested(k))
                xlabel("Pos(s-e)/w")
                ylabel(labels(sensory_type))
            end

            % Plt the combined steps 
            subplot(2,5, 4+5*(sensory_type-1):5+5*(sensory_type-1)); hold on % 4+5*(s-1)={4,9}, 5+5*(s-1)={5,10} s={1,2}
            plot(predictor(:)', depended(:)','x')
            title("All steps")            
            ylabel(labels(sensory_type))
            xlabel("Pos(s-e)/w")

            % Remember p-value and slope for later plot
            mdl = fitlm(predictor(:), depended(:));   
            reg_data.slopes(sub, sensory_type, 4)  = table2array(mdl.Coefficients(2,1));  % slopes
            reg_data.p_value(sub, sensory_type, 4) = table2array(mdl.Coefficients(2,4));  % p_values
        end
    end 

    % Set y-limits and x-limits on all subplots 
    ax = findobj(gcf, 'type', 'axes');
    ylims = get(ax, 'YLim'); 
    xlims = get(ax, 'XLim'); 
    [~, idx_y_ta]  = max(cellfun(@(x) diff(x), ylims(1:4)));
    [~, idx_y_sol] = max(cellfun(@(x) diff(x), ylims(5:8)));
    [~, idx_x_ta]  = max(cellfun(@(x) diff(x), xlims(1:4)));
    [~, idx_x_sol] = max(cellfun(@(x) diff(x), xlims(5:8)));
    idx_y_sol = idx_y_sol+4; 
    idx_x_sol = idx_x_sol+4; 
    for i = 1:numel(ax)
        if any(i == 1:4)
            set(ax(i), 'YLim', ylims{idx_y_ta})
            set(ax(i), 'XLim', xlims{idx_x_ta})
        elseif any(i == 5:8)  
            set(ax(i), 'YLim', ylims{idx_y_sol})
            set(ax(i), 'XLim', xlims{idx_x_sol})
        end
    end
    
    % Plot regression 
    for sensory_type = [SOL,TA] % loop 
        for step = 1:3 % loop steps  
            mdl = fitlm(data_reg.pre_steps{step}, data_reg.dep_steps{sensory_type, step}); 
            b = table2array(mdl.Coefficients(1,1)); 
            a = table2array(mdl.Coefficients(2,1)); 
            p_value = table2array(mdl.Coefficients(2,4)); 
            r2 = round(mdl.Rsquared.Adjusted,3); 
            linearReg = @(x) x*a + b; 
            subplot(2, 5, 5*(sensory_type-1)+step); hold on; 
            plot(data_reg.pre_steps{step}, linearReg(data_reg.pre_steps{step}), "color", "red")
            if p_value < 0.05
                subtitle(['P-value: {\color{black} ' num2str(p_value) '}. R^2' num2str(r2)])
            else 
                subtitle(['P-value: {\color{red} ' num2str(p_value) '}. R^2' num2str(r2)])
            end           
        end
        mdl = fitlm(data_reg.pre, data_reg.dep{sensory_type}); 
        b = table2array(mdl.Coefficients(1,1)); 
        a = table2array(mdl.Coefficients(2,1)); 
        p_value = table2array(mdl.Coefficients(2,4)); 
        r2 = round(mdl.Rsquared.Adjusted,3); 
        linearReg = @(x) x*a + b; 
        kage = [min(data_reg.pre(:)), max(data_reg.pre(:))]; 

        subplot(2,5, 4+5*(sensory_type-1):5+5*(sensory_type-1)); hold on
        plot([kage(1),kage(2)], linearReg([kage(1),kage(2)]), "color", "red")
        if p_value < 0.05
            subtitle(['P-value: {\color{black} ' num2str(p_value) '}. R^2' num2str(r2)])
        else 
            subtitle(['P-value: {\color{red} ' num2str(p_value) '}. R^2' num2str(r2)])
        end   
    end
    
    fprintf('done [ %4.2f sec ] \n', toc);
else
    fprintf('disable \n');
end


%% Task 1.3 FC correlation with EMG (slopes)
fprintf('script: TASK 1.3  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  '); tic

% Find the best parameters for one subject and apply them to the other
%    subjects. 
% one factor anova

pltShow = true; 

if pltShow 
    if exist("reg_data")
        figSize = [300 250 width/1.4 200]; % where to plt and size

        % Check if a figure with the name 'TASK3' is open
        fig = findobj('Name', 'slopes');
        % If a figure is found, close it
        if ~isempty(fig), close(fig); end

        figure('name','slopes','Position', figSize); % begin plot 
        hold on 
        sgtitle("Best fit slopes [sub="+numel(inc_sub)+"]. Weighted: "+weighting_data)
        
        for sensory_type = [SOL, TA]
            for step = 1:4
                if step == 4
                    subplot(1,5,4:5); hold on 
                    title("All steps")
                else
                    subplot(1,5,step); hold on
                    title("Step " + step)
                end
                slopes  = reg_data.slopes(inc_sub, sensory_type, step); 
                p_value = reg_data.p_value(inc_sub,sensory_type, step);
                x_value = ones(1,size(slopes,1)); 
                if sensory_type == TA; x_value=x_value+1; end 
                
                plot([0 3],[0 0], 'color', 'black')
                plot(x_value(1), mean(slopes), "_", 'Color', 'black', 'linewidth', 4)
                plot(x_value, slopes, '.', 'color', 'blue') % soleus data indiv
                for i = 1:length(p_value)
                    if p_value(i) < 0.05 
                        plot(x_value, slopes(i), 'o', 'linewidth',2, 'color', [0.75,0.75,0.75]) % soleus data indiv
                    end 
                end 
                xticks([1 2]);
                xticklabels({'SOL', 'TA'});
                grid on;
    
                if step == 1
                    ylabel("Regression Slopes"+newline+" [ \alpha ]")
                elseif step == 4
                    legend(["", "Group mean", "Slope (subject)", "< 0.05"])
                end
    
            end 
        end
    
        % Set y-limits on all subplots 
        ax = findobj(gcf, 'type', 'axes');
        ylims = get(ax, 'YLim'); 
        max_value = max(cellfun(@max, ylims));
        min_value = min(cellfun(@min, ylims));
        for i = 1:numel(ax)
            set(ax(i), 'YLim', ([-2 6]))%([min_value max_value]))
        end 
    else 
        msg = "\n     Need to run 'Task 1.2' before to enable this section to plot \n"; 
        fprintf(2,msg); 
    end

    slope_data = squeeze(reg_data.slopes(inc_sub, SOL, 1:4)); 
    %slope_data = squeeze(reg_data.slopes(inc_sub, TA, 1:4)); 

    vartestn(slope_data(:,1:3),'TestType','LeveneAbsolute');
    anova1(slope_data(:,1:3));
    [H, p_Shapiro_Wilk, SWstatistic] = swtest(slope_data(:,4));
    disp("")
    disp("Shapiro-Wilk normality test, p-value: " + p_Shapiro_Wilk + " (normality if p<0.05)")

    disp("One-sample right tail t-test,  p_value: ")
    [H_ttest, p_ttest] = ttest(slope_data(:,1), 0, 'Tail', 'right')
    [H_ttest, p_ttest] = ttest(slope_data(:,2), 0, 'Tail', 'right')
    [H_ttest, p_ttest] = ttest(slope_data(:,3), 0, 'Tail', 'right')
    [H_ttest, p_ttest] = ttest(slope_data(:,4), 0, 'Tail', 'right')

    fprintf('done [ %4.2f sec ] \n', toc);
else
    fprintf('disable \n');
end