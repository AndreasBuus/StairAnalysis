clc; 
clear all; 
close all; 

SubjectName = "Pegah"; 

% Folders path 
folderpath_10sem = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/data/pegah_stairs/";

% File Path 
filepath_CTL_part1 = folderpath_10sem + "pegah_1to50_13feb.mat";
filepath_CTL_part2 = folderpath_10sem + "pegah_1to50_test2_13feb001.mat"; 


%% Abbreviation
fprintf('script: Abbreviation'); 

% protocol abbreviation types
CTL = 1;  % Control nr 1
VER = 2;  % Vertical 
HOR = 3;  % Horizontal
CTL2 = 4; % Control nr 2  

%ProtoAll = [CTL, VER, HOR, CTL2];
ProtoAll = [CTL];

% sensor abbreviation type
SOL = 1; % Soleus 
TA = 2;  % Tibialis
ANG = 3; % Ankle position 
FSR = 4; % Force sensitiv resistor 


data    = cell(3,4);
% example: data{protocol, sensor}(sweep, data number)

% include function folder
addpath("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/FunctionFiles")


%% Load data and Acquisition Set-Up from Mr Kick
fprintf('script: Load data and Acquisition Set-Up from Mr Kick'); 
% Functions used: [load_EMG_v2()]

% load pre-control
[SOL_CTL1, TA_CTL1, angle_CTL1, FSR_CTL1] = load_EMG_v2(filepath_CTL_part1); clear filepath_CTL_part1
[SOL_CTL2, TA_CTL2, angle_CTL2, FSR_CTL2] = load_EMG_v2(filepath_CTL_part2); clear filepath_CTL_part2

data{CTL,SOL} = [SOL_CTL1; SOL_CTL2];       clear SOL_CTL1 SOL_CTL2; 
data{CTL,TA}  = [TA_CTL1; TA_CTL2];         clear TA_CTL1 TA_CTL2; 
data{CTL,ANG}  = [angle_CTL1; angle_CTL2];  clear angle_CTL1 angle_CTL2; 
data{CTL,FSR}  = [FSR_CTL1; FSR_CTL2];      clear FSR_CTL1 FSR_CTL2; 

% Acquisition Set-Up
sweep_length = 10;              % Signal length in second
Fs = 2000;                      % Samples per second
dt = 1/Fs;                      % Seconds per sample
pre_trig = 4;                   % Pre-trigger 
N = Fs*sweep_length;            % Total number of samples per signal

% Exclude data 
exclude_CTL = [21, 22, 64, 73, 77, 93];               % excluded control sweeps
exclude_CTL2= [];               % excluded control sweeps
exclude_VER = [];               % excluded horizontal sweeps
exclude_HOR = [];               % excluded horizontal sweeps

clc; clear; 
Blue  = 1; BlueP = 2; 
Red   = 3; RedP  = 4;
Green = 5; GreenP= 6; 
White = 7; WhiteP= 8; 
Black = 9; BlackP= 10; 
fail = NaN; 

FootPos(1:10) = [fail, Black, Black, Black, Black, Black, Black, Black, Black, Black]; 
FootPos(11:20) = [Black, Black, Black, Black, Black, Black, Black, Black, Black, Black];
FootPos(21:30) = [Black, Black, Black, Black, Black, Black, Black, Black, Black, Black];
FootPos(31:40) = [Black, Black, Black, Black, Black, Black, Black, Black, Black, Black];
FootPos(41:50) = [Black, Black, Black, Black, Black, Black, Black, Black, Black, Black];
FootPos(51:60) = [Black, Black, Black, Black, Black, Black, Black, Black, Black, Black];
FootPos(61:70) = [Black, Black, Black, Black, Black, Black, Black, Black, Black, Black];
FootPos(71:80) = [Black, Black, Black, Black, Black, Black, Black, Black, Black, Black];
FootPos(81:90) = [Black, Black, Black, Black, Black, Black, Black, Black, Black, Black];
FootPos(91:100) = [Black, Black, Black, Black, Black, Black, Black, Black, Black, Black];

for i = [SOL, TA, FSR, ANG]
    data{CTL,i}(exclude_CTL,:) = []; 
end 


%% Filtrering and detrend (similar to MR. kick)
fprintf('script: Filtrering and detrend (similar to MR. kick)'); 

% Functions used: [rectify_filter()], [filt_FSR()]

fc = 40;                            % Cutoff frequency for LowPass filter
order = 1;                          % Filter order 
[b,a] = butter(order,fc/(Fs/2));    % Filter coefficient

% rectify and filter EMG. Remove noise in FSR 
for proto = ProtoAll
    [data{proto,SOL}, data{proto,TA}] = rectify_filter(data{proto,SOL}, data{proto,TA}, b, a);  
    [data{proto,FSR}] = func_filt_FSR(data{proto,FSR}, "test", false , "limit_pct", 0.95, "gap_size2", 150, 'gap_size3',800); 
end

%% Find correct position for Stand - and Swingphase
fprintf('script: Find correct position for Stand - and Swingphase'); 

step_index = cell(3,1);  % Index for position 
error_index = cell(3,1);

for proto = ProtoAll
    [step_index{proto}, error_index{proto}] = func_step_index(data{proto,FSR}, 'offset', 8500);
end

%% Readjust data to Local Peak instead of FSR 
fprintf('script: Readjust data to Local Peak instead of FSR '); 

readjust = true; 
show_gui = true; 
gui_subject = 1; 

folderpath = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/Subject_offsets/"; 
filename = "offset.mat";

error = zeros(numel(proto_all), numel(names),100); % 3x10x100
if readjust 
    for sub = 1:numel(names) % loop through subjects 

        % Some subject only completed one protocol. 
        if any(strcmp(names(sub), ["Christian", "Soeren"])) 
            protocols = [CTL]; 
        else 
            protocols = proto_all; 
        end
        
        % Check if an already defined offset file exist for subject 
        filename = sub+"offset.mat";
        filepath = fullfile(folderpath, filename);
        if exist(filepath, 'file') == 2
            load(filepath) % return 4x1 cell array called 'offset'
            total_step{1,1,sub} = offset; 
        
        % Create offset automatically if no file exist 
        else 
            data = total_data{1,1,sub};         % load data
            step_index = total_step{1,1,sub};   % load foot placement
            for proto = protocols 
                for sweep = 1:size(data{proto,ANG},1) % loop through sweeps        
                    for step = 1:3  % loop through steps   
                        
                        % Find template around foot-strike
                        [rise_num, ~] = func_find_edge(steps_tested(step));   
                        rise_index = step_index{proto}(sweep, rise_num); 
                        array = rise_index-400:rise_index+400;
                        template = data{proto,ANG}(sweep, array); 
                        signal = data{proto,ANG}(sweep, :); 
                    
                        % Peak inside template
                        [pks, locs] = findpeaks(template, 'MinPeakDistance', 200);
                        locs = locs + array(1);
                        
                        % Find the peak that follow the condition 
                        the_pks = 0; the_loc = 0;  % peaks and locations
                        for i = 1:numel(locs) % loop through locations
                            if pks(i) > signal(locs(i) - 200) && pks(i) > signal(locs(i)+200)
                                the_pks = pks(i);
                                the_loc = locs(i);
                            end 
                        end 
        
                        % Update step_index or throw error
                        if the_pks == 0     % non found: error
                            error(proto,sub,sweep) = 1;                     
                        else                % no error 
                            step_index{proto}(sweep, rise_num) = the_loc;
                        end 
                    end % step
                end % sweep
        
                % Display no-peak idxs 
                temp = find(error(proto,sub,:) == 1);
                if ~isempty(temp)
                    singleStr = string;
                    for i = 1:numel(temp)
                        singleStr = singleStr + num2str(temp(i)) + " "; 
                    end 
                    msg = "\n     No peak found. Subject: " + sub + ". Sweep: " + singleStr +  ". Protocol: " + proto + " "; 
                    fprintf(2,msg); 
                end 
            end % proto

        % Save realigned data 
        total_data{1,1,sub} = data; 
        total_step{1,1,sub} = step_index; % update step_index
        end % exist 
    end % sub
    fprintf('done [ %4.2f sec ] \n', toc);
else 
    fprintf('disable \n');
end  % readjust 
 
clear error offset

if show_gui
    fprintf('script: Re-adjust gui - [Waiting for user input]')
    data = total_data{1,1,gui_subject}; 
    type = total_type{1,1,gui_subject}; 

    filename = gui_subject+"offset.mat";
    filepath = fullfile(folderpath, filename);
    if exist(filepath, 'file') == 2
        load(filepath) % return 4x1 cell array called 'offset'
        step_index = offset; 
    else
        step_index = total_step{1,1,gui_subject}; % update step_index
    end

    offset = [];    % preparer for new input    
    %readjustFSR     % open gui
    Copy_of_readjustFSR
    pause           % wait for user input 
    
    if ~isempty(offset)
        if exist(filepath, 'file') == 2 
            prompt = newline + "Want to over save. YES: press >y<. NO, press >n<"+ newline;
            correctInput = false; 
            while correctInput == false     % Wait for correct user input
                switch input(prompt, 's')   % Save user input
                    case "y"
                        correctInput = true; 
                        oversave = true; 
                    case "n"
                        correctInput = true; 
                        oversave = false;
                    otherwise
                        correctInput = false;
                        warning("Input not accepted")
                end 
            end 
        else 
            oversave = true; 
        end 

        if oversave
            total_step{1,1,gui_subject} = offset; 
            save(folderpath + names(gui_subject)+"_offset",'offset')
            disp("Data saved")
        end         
    end 
    fprintf('gui done \n')
end 




% (sweep,1) -  Seventh  step, rising 
% (sweep,2) -  Sixth step, falling
% (sweep,3) -  Sixth step, rising
% (sweep,4) -  Fouth step, falling  <--  
% (sweep,5) -  Fouth step, rising  <-- 
% (sweep,6) -  Second step, falling  
% (sweep,7) -  Second step, rising  
% (sweep,8) -  Zero step, falling  
% (sweep,9) -  Zero step, rising  


% [step_index{CTL}(10,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',1, 'move_num', 1, 'sweep',10, 'offset',8500);
% [step_index{CTL}(27,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',3, 'move_num', 1, 'sweep',27, 'offset',8500);
% [step_index{CTL}(35,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',1, 'move_num', 1, 'sweep',35, 'offset',8500);
% [step_index{CTL}(40,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',1, 'move_num', 1, 'sweep',40, 'offset',8500);
% [step_index{CTL}(44,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',1, 'move_num', 1, 'sweep',44);
% [step_index{CTL}(50,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',1, 'move_num', 0, 'sweep',50, 'offset',8000);
% [step_index{CTL}(85,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',2, 'move_num', 1, 'sweep',85, 'offset',8500);
% [step_index{CTL}(87,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',7, 'move_num', 1, 'sweep',87, 'offset',8500);





if true 
    % >>>> TEST CODE <<<<
    proto = CTL; % CTL[], HOR[x], VER[], CTL2
    loop = true; sweep = 1; 
    prompt = "Continue, press >c<" + newline + "Quite, press >q<"+ newline + "Change sweep number, press >t<"+ newline;
    figure(2); 
    
    while loop == true
        clc
        sgtitle("Sweep: " + sweep + ". Protocol: " + proto)
        hold off;
        plot(data{proto, ANG}(sweep,:))
        hold on;
        plot(data{proto, FSR}(sweep,:))

        plot([8000,8000],[-1 6], "LineWidth",3, "Color", "red")

        [rise, fall] = func_find_edge(0); 
        plot([step_index{proto}(sweep, rise), step_index{proto}(sweep,rise)],[1 4], "LineWidth",2, "Color", "black")
        plot([step_index{proto}(sweep,rise), step_index{proto}(sweep,fall)],[2.5 2.5], "LineWidth",2, "Color", "black")
        plot([step_index{proto}(sweep,fall), step_index{proto}(sweep,fall)],[1 4], "LineWidth",2, "Color", "red")

        [rise, fall] = func_find_edge(2); 
        plot([step_index{proto}(sweep,rise), step_index{proto}(sweep,rise)],[1 4], "LineWidth",2, "Color", "black")
        plot([step_index{proto}(sweep,rise), step_index{proto}(sweep,fall)],[2.5 2.5], "LineWidth",2, "Color", "black")
        plot([step_index{proto}(sweep,fall), step_index{proto}(sweep,fall)],[1 4], "LineWidth",2, "Color", "red")

        [rise, fall] = func_find_edge(4); 
        plot([step_index{proto}(sweep, rise), step_index{proto}(sweep,rise)],[1 4], "LineWidth",2, "Color", "green")
        plot([step_index{proto}(sweep,rise), step_index{proto}(sweep,fall)],[2.5 2.5], "LineWidth",2, "Color", "black")
        plot([step_index{proto}(sweep,fall), step_index{proto}(sweep,fall)],[1 4], "LineWidth",2, "Color", "blue")
    
        [rise, fall] = func_find_edge(6); 
        plot([step_index{proto}(sweep, rise), step_index{proto}(sweep,rise)],[1 4], "LineWidth",2, "Color", "black")
        plot([step_index{proto}(sweep,rise), step_index{proto}(sweep,fall)],[2.5 2.5], "LineWidth",2, "Color", "black")
        plot([step_index{proto}(sweep,fall), step_index{proto}(sweep,fall)],[1 4], "LineWidth",2, "Color", "red")

        [rise, fall] = func_find_edge(7); 
        plot([step_index{proto}(sweep, rise), step_index{proto}(sweep,rise)],[1 4], "LineWidth",2, "Color", "yellow")

        correctInput = false; 
        while correctInput == false
            str = input(prompt, 's');
            if strcmp(str,"q")
                disp("Loop stopped")
                loop = false; correctInput = true; 
            elseif strcmp(str,"t")
                sweep = input("New sweep number: ")-1; 
                correctInput = true; 
            elseif strcmp(str,"c") %, sweep == size(data{proto, SOL}))
                correctInput = true; 
            end 
            if correctInput == false
                warning("Input not accepted")
            end
        end
        sweep = sweep + 1;
        if sweep > size(data{proto,SOL},1)
            loop = false; 
        end 
    end
    close 2
end


%% Save Data 

save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName + "_data", 'data')
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName+"_step",'step_index')

disp("Processed: " + SubjectName)