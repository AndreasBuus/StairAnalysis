clc; 
clear all; 
close all; 

SubjectName = "Mia"; 

% Folders path 
folderpath_10sem = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/data/Mia 20.4.2022/";

% File Path 
filepath_CTL = folderpath_10sem + "mia_control1_(20_4_2022).mat";
filepath_CTL_2 = folderpath_10sem + "mia_control2_(20_4_2022).mat"; 
filepath_HOR = folderpath_10sem + "mia_horisontal1_(20_4_2022).mat"; 
filepath_VER = folderpath_10sem + "mia_vertical1_(20_4_2022).mat"; 

%% Abbreviation

% protocol abbreviation types
CTL = 1; VER = 2; HOR = 3; CTL2 = 4; 
ProtoAll = [CTL, VER, HOR, CTL2];

% sensor abbreviation type
SOL = 1; TA = 2; ANG = 3; FSR = 4; time = 5; 

% sensor abbreviation subtypes
VEL = 1; ACC = 2; 

data    = cell(3,4);
% example: data{protocol, sensor}(sweep, data number)

% include function folder
addpath("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/FunctionFiles")

%% Load data and Acquisition Set-Up from Mr Kick
% Functions used: [load_EMG()]


[data{CTL,1:4}] = load_EMG(filepath_CTL); clear filepath_CTL
[data{CTL2,1:4}] = load_EMG(filepath_CTL_2); clear filepath_CTL_2
[data{VER,1:4}] = load_EMG(filepath_VER); clear filepath_VER
[data{HOR,1:4}] = load_EMG(filepath_HOR); clear filepath_HOR

% Acquisition Set-Up
sweep_length = 10;              % Signal length in second
Fs = 2000;                      % Samples per second
dt = 1/Fs;                      % Seconds per sample
pre_trig = 4;                   % Pre-trigger 
N = Fs*sweep_length;            % Total number of samples per signal

% Exclude data 
exclude_CTL = [15,16];               % excluded control sweeps
exclude_CTL2= [];               % excluded control sweeps
exclude_VER = [];               % excluded horizontal sweeps
exclude_HOR = [];     % excluded horizontal sweeps


%HOR_perturbation = [7,8,16,18,30,33,35,38,39,41,49,52,54,56,59,60];
HOR_perturbation = [7,8,11,13,16,18,21,25,27,28,30,33,35,38,39,41,49,52,54,56,59,60];
VER_perturbation = [5,8,10,11,15,17,20,24,28,30,32,35,37,40,42,44,52,54,56,58];


[VER_yes, VER_no] = sort_sweeps(size(data{VER,SOL},1), VER_perturbation,  exclude_VER); 
[HOR_yes, HOR_no] = sort_sweeps(size(data{HOR,SOL},1), HOR_perturbation,  exclude_HOR); 


for i = [SOL, TA, FSR, ANG]
    data{CTL,i}(exclude_CTL,:) = []; 
    data{CTL2,i}(exclude_CTL2,:) = []; 

    data{VER,i}(exclude_VER,:) = []; 
    data{HOR,i}(exclude_HOR,:) = []; 
end 

%% Load offset 

raw = data;                     % save raw data 
load('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_mia_ver'); 
offset_ver = offset; 
load('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_mia_hor'); 
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

%gap_size2
%gap_size3

[data{CTL,FSR}] = func_filt_FSR(data{CTL,FSR}, "test", 0 , "limit_pct", 0.95, "gap_size2", 20, 'gap_size3',20); 
[data{VER,FSR}] = func_filt_FSR(data{VER,FSR}, "test", 0 , "limit_pct", 0.95); 
[data{HOR,FSR}] = func_filt_FSR(data{HOR,FSR}, "test", 0 , "limit_pct", 0.95); 
[data{CTL2,FSR}] = func_filt_FSR(data{CTL2,FSR}, "test", 0 , "limit_pct", 0.95); 



%% 𝔹𝕖𝕘𝕚𝕟𝕚𝕟𝕘 𝕠𝕗 𝕊𝕥𝕒𝕟𝕕- 𝕒𝕟𝕕 𝕊𝕨𝕚𝕟𝕘𝕡𝕙𝕒𝕤𝕖
% Functions used: [step_index_pos()], [step_index_corr()], [step_corr()]
% find change in FSR signal
step_index = cell(3,1);
error_index = cell(3,1);

for proto = ProtoAll
    for sweep = 1: size(data{proto,SOL},1)
        step_index{proto}(sweep,:) = [17280, 16362, 14673, 13888, 12381, 11654, 10201, 9427, 8194]; 
    end
end


if true 
    % >>>> TEST CODE <<<<
    proto = VER; % CTL[], HOR[x], VER[], CTL2
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

type{1} = VER_yes;   type{2} = VER_no; 
type{3} = HOR_yes;   type{4} = HOR_no; 
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName + "_data", 'data')
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName+"_type",'type')
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName+"_step",'step_index')

disp("Processed: " + SubjectName)