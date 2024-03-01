clc; 
clear all; 
close all; 

% REMEMBER !
% 1. ensure the correct subject name/alias is typed
% 2. ensure the "folderpath_data" point to your data folderpath 

SubjectName = "Pegah"

% Folders path to data 
folderpath_data = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/data/pegah_stairs/";

% File path to data
filepath_CTL_part1 = folderpath_data + "pegah_1to50_13feb.mat";
filepath_CTL_part2 = folderpath_data + "pegah_1to50_test2_13feb001.mat"; 

% Folderpath to main folder 
main_folderpath_unedited = strrep(fileparts(matlab.desktop.editor.getActiveFilename),'\','/');
x = strfind(main_folderpath_unedited,'/Stair_Matlab_code'); 
main_folderpath = main_folderpath_unedited(1:x(end)+17); 

%% Abbreviation
fprintf('script: Abbreviation'); 

% protocol abbreviation types
CTL  = 1;  % Control nr 1
VER  = 2;  % Vertical 
HOR  = 3;  % Horizontal
CTL2 = 4; % Control nr 2  

Protocol_All = [CTL];

% Sensor abbreviation
SOL = 1; % Soleus 
TA  = 2; % Tibialis
ANG = 3; % Ankle position 
FSR = 4; % Force sensitiv resistor 

data    = cell(3,4);
% example:
%   data{protocol, sensor}(sweep, data number)


% include function folder
addpath(main_folderpath +"/FunctionFiles")

fprintf(' ... done \n'); 
%% Load data and Acquisition Set-Up from Mr Kick
fprintf('script: Load data and Acquisition Set-Up from Mr Kick'); 
% Functions used: [load_EMG_v2()]

% load pre-control
[SOL_CTL1, TA_CTL1, angle_CTL1, FSR_CTL1] = load_EMG_v2(filepath_CTL_part1); 
[SOL_CTL2, TA_CTL2, angle_CTL2, FSR_CTL2] = load_EMG_v2(filepath_CTL_part2); 

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

% Exclude sweep 
exclude_CTL = [21, 22, 64, 73, 77, 93];               % excluded control sweeps
exclude_CTL2= [];               % excluded control sweeps
exclude_VER = [];               % excluded horizontal sweeps
exclude_HOR = [];               % excluded horizontal sweeps

for proto = Protocol_All
    for i = [SOL, TA, FSR, ANG]
        data{CTL,i}(exclude_CTL,:) = []; 
    end 
end
fprintf(' ... done \n'); 

%% Filtrering and detrend (similar to MR. kick)
fprintf('script: Filtrering and detrend (similar to MR. kick)'); 

% Functions used: [rectify_filter()], [filt_FSR()]

fc = 40;                            % Cutoff frequency for LowPass filter
order = 1;                          % Filter order 
[b,a] = butter(order,fc/(Fs/2));    % Filter coefficient

% rectify and filter EMG. Remove noise in FSR 
for proto = Protocol_All
    [data{proto,SOL}, data{proto,TA}] = rectify_filter(data{proto,SOL}, data{proto,TA}, b, a);  
    [data{proto,FSR}] = func_filt_FSR(data{proto,FSR}, "test", false , "limit_pct", 0.95, "gap_size2", 150, 'gap_size3',800); 
end

fprintf(' ... done \n'); 

%% Find correct position for Stand - and Swingphase
fprintf('script: Find correct position for Stand - and Swingphase'); 

step_index = cell(3,1);  % Index for position 
error_index = cell(3,1);

for proto = Protocol_All
    [step_index{proto}, error_index{proto}] = func_step_index(data{proto,FSR}, 'offset', 8500);
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


[step_index{CTL}(10,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',1, 'move_num', 1, 'sweep',10, 'offset',8500);
[step_index{CTL}(27,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',3, 'move_num', 1, 'sweep',27, 'offset',8500);
[step_index{CTL}(35,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',1, 'move_num', 1, 'sweep',35, 'offset',8500);
[step_index{CTL}(40,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',1, 'move_num', 1, 'sweep',40, 'offset',8500);
[step_index{CTL}(44,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',1, 'move_num', 1, 'sweep',44);
[step_index{CTL}(50,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',1, 'move_num', 0, 'sweep',50, 'offset',8000);
[step_index{CTL}(85,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',2, 'move_num', 1, 'sweep',85, 'offset',8500);
[step_index{CTL}(87,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',7, 'move_num', 1, 'sweep',87, 'offset',8500);

% >>>> TEST CODE <<<<
if true 
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

fprintf(' ... done \n'); 

%% Save Data 
fprintf('script: Save data'); 


if ~exist(main_folderpath, 'dir') == 7
    newfolder = fullfile(main_folderpath, "data_preprocessed")
    mkdir(newfolder); 
end

save(main_folderpath + "/data_preprocessed/" + SubjectName + "_data", 'data')
save(main_folderpath + "/data_preprocessed/" + SubjectName+"_step",'step_index')

fprintf(' ... done \n'); 