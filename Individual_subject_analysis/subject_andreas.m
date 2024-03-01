clc;
clear all; 
close all; 

SubjectName = "Andreas"; 

folderpath = "C:/Users/BuusA/OneDrive - Aalborg Universitet/9. semester\data\Andreas/";
folderpath_10sem = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/data/Andreas 25.04.2022/";

% Load control trials


filepath_CTL_part1 = folderpath + "andreas_2022_17_1_control_1-40.mat"; 
filepath_CTL_part2 = folderpath + "andreas_2022_17_1_control_41-82.mat";
filepath_CTL = folderpath_10sem + "control1.mat"; 

% load horizontal trials
filepath_HOR_part1 = folderpath + "andreas_2022_17_1_horisontal_1-60.mat";
filepath_HOR_part2 = folderpath + "andreas_2022_17_1_Horisontal_061-100.mat";

% load vertical trials
filepath_VER = folderpath_10sem + "vertical1.mat"; 

% include function folder
addpath("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/FunctionFiles")

% No records of CTL2


%% Abbreviation

% protocol abbreviation types
CTL = 1; VER = 2; HOR = 3; CTL2 = 4; 
ProtoAll = [CTL, VER, HOR];

% sensor abbreviation type
SOL = 1; TA = 2; ANG = 3; FSR = 4; time = 5; 

% sensor abbreviation subtypes
VEL = 1; ACC = 2; 

data    = cell(3,4);
% example: data{protocol, sensor}(sweep, data number)


%% Load data and Acquisition Set-Up from Mr Kick
% Functions used: [load_EMG()]

% load pre-control
[SOL_CTL1, TA_CTL1, angle_CTL1, FSR_CTL1] = load_EMG(filepath_CTL_part1); clear filepath_CTL_part1
[SOL_CTL2, TA_CTL2, angle_CTL2, FSR_CTL2] = load_EMG(filepath_CTL_part2); clear filepath_CTL_part2

data{CTL,SOL} = [SOL_CTL1; SOL_CTL2];       clear SOL_CTL1 SOL_CTL2; 
data{CTL,TA}  = [TA_CTL1; TA_CTL2];         clear TA_CTL1 TA_CTL2; 
data{CTL,ANG}  = [angle_CTL1; angle_CTL2];  clear angle_CTL1 angle_CTL2; 
data{CTL,FSR}  = [FSR_CTL1; FSR_CTL2];      clear FSR_CTL1 FSR_CTL2; 

% load horizontal 
[SOL_HOR1, TA_HOR1, angle_HOR1, FSR_HOR1] = load_EMG(filepath_HOR_part1); clear filepath_HOR_part1
[SOL_HOR2, TA_HOR2, angle_HOR2, FSR_HOR2] = load_EMG(filepath_HOR_part2); clear filepath_HOR_part2

data{HOR,SOL} = [SOL_HOR1; SOL_HOR2];       clear SOL_HOR1 SOL_HOR2; 
data{HOR,TA}  = [TA_HOR1; TA_HOR2];         clear TA_HOR1 TA_HOR2; 
data{HOR,ANG} = [angle_HOR1; angle_HOR2];   clear angle_HOR1 angle_HOR2; 
data{HOR,FSR} = [FSR_HOR1; FSR_HOR2];       clear FSR_HOR1 FSR_HOR2; 

% load vertical
[data{VER,1:4}] = load_EMG(filepath_VER); clear filepath_VER

% acquisition Set-Up
sweep_length = 10;              % Signal length in second
Fs = 2000;                      % Samples per second
dt = 1/Fs;                      % Seconds per sample
pre_trig = 4;                   % Pre-trigger 
N = Fs*sweep_length;            % Total number of samples per signal


%% Perturbation and exclude trails
% Functions used: [sort_sweeps()]

% perturbation trials
HOR_perturbation = [1,10,12,13,15,21,23,27,29,35,36,49,50,54,56,59,60,64,65,66,71,72,75,78,79,91,94,95,97]; 
VER_perturbation = [2,12,15,17,21,23,26,34,36,39,43,45,48,52,60,63,64,65,67]; 

% Identified sweep needed to be excluded 
exclude_CTL = [];               % excluded control sweeps
exclude_HOR = [19,85,33,38];    % excluded horizontal sweeps
exclude_VER = [40,68];          % excluded horizontal sweeps 

% Remove excluded sweeps and return perturbation array
[HOR_yes, HOR_no] = sort_sweeps(size(data{HOR,SOL},1), HOR_perturbation,  exclude_HOR); 
[VER_yes, VER_no] = sort_sweeps(size(data{VER,SOL},1), VER_perturbation,  exclude_VER); 


for i = [SOL, TA, FSR, ANG]
    data{CTL,i}(exclude_CTL,:) = []; 
    data{VER,i}(exclude_VER,:) = []; 
    data{HOR,i}(exclude_HOR,:) = []; 
end 

%% Load offset 

raw = data;  % save raw data 
load('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_andreas_ver'); 
offset_ver = offset; 
load('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_andreas_hor'); 
offset_hor = offset; 


%% Filtrering and detrend (similar to MR. kick)
% Functions used: [rectify_filter()], [filt_FSR()]

fc = 40;                            % Cutoff frequency for LowPass filter
order = 1;                          % Filter order 
[b,a] = butter(order,fc/(Fs/2));    % Filter coefficient

% rectify and filter EMG. Remove noise in FSR 
for proto = ProtoAll
    [data{proto,SOL}, data{proto,TA}] = rectify_filter(data{proto,SOL}, data{proto,TA}, b, a);  
end

for proto = ProtoAll
    [data{proto,FSR}] = func_filt_FSR(data{proto,FSR}, "test", 0 , "limit_pct", 0.95); 
end

% [data{VER,FSR}] = func_filt_FSR(data{VER,FSR}, "test", 1 , "limit_pct", 0.95); 


%% Begining of Stand- and Swingphase
% Functions used: [step_index_pos()], [step_index_corr()], [step_corr()]

% find change in FSR signal
step_index = cell(3,1);
error_index = cell(3,1);

for proto = ProtoAll
    [step_index{proto}, error_index{proto}] = func_step_index(data{proto,FSR});
end

sweep = 3; step_index{CTL}(sweep,[2]) = [17108]; 

[step_index{VER}(12,:)] = func_step_index_corr('FSR', data{VER,FSR}, 'direction', 'right', 'edge_num',4, 'move_num', 2, 'sweep',12);


if true 
    % >>>> TEST CODE <<<<
    proto = VER; % Checked: CTL[x], HOR[x], VER[x]
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

type{1} = VER_yes;   type{2} = VER_no; 
type{3} = HOR_yes;   type{4} = HOR_no; 
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName + "_data", 'data')
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName+"_type",'type')
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName+"_step",'step_index')

disp("Processed: " + SubjectName)
