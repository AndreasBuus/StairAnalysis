clc;
clear; 
close all; 

SubjectName = "Benedikte"; 

folderpath = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/data/Benedikte 15.02.2022/"; 

filepath_CTL = folderpath + "mr.kick/benedikte_control_(15_2_2022)_forsoeg_2.mat"; 
filepath_HOR = folderpath + "mr.kick/benedikte_horizontal_(15_2_2022).mat";
filepath_VER = folderpath + "mr.kick/benedikte_vertical5cm_(15_2_2022).mat"; 

filepath_CTL_foot = folderpath + "control_test2/benedikte_control_foot.mat"; 
filepath_HOR_foot = folderpath + "Horisontal_test1/benedikte_horisontal_foot.mat"; 
filepath_VER_foot = folderpath + "vertical_test1/benedikte_vertical_foot.mat";

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

%%  𝕃𝕠𝕒𝕕 𝕕𝕒𝕥𝕒 𝕒𝕟𝕕 𝔸𝕔𝕢𝕦𝕚𝕤𝕚𝕥𝕚𝕠𝕟 𝕊𝕖𝕥-𝕌𝕡 𝕗𝕣𝕠𝕞 𝕄𝕣 𝕂𝕚𝕔?

% data{protocol, EMG}(sweep, data_num)

[data{1,1:4}] = load_EMG(filepath_CTL); clear filepath_CTL; 
[data{2,1:4}] = load_EMG(filepath_VER); clear filepath_VER; 
[data{3,1:4}] = load_EMG(filepath_HOR); clear filepath_HOR; 

[data{1,5}] = load(filepath_CTL_foot).A; clear filepath_CTL_foot; 
[data{2,5}] = load(filepath_VER_foot).A; clear filepath_VER_foot; 
[data{3,5}] = load(filepath_HOR_foot).A; clear filepath_HOR_foot; 


% Acquisition Set-Up
sweep_length = 10;              % Signal length in second
Fs = 2000;                      % Samples per second
dt = 1/Fs;                      % Seconds per sample
pre_trig = 4;                   % Pre-trigger 
N = Fs*sweep_length;            % Total number of samples per signal


%% Perturbation and exclude trails
% Functions used: [sort_sweeps()]

% Exclude data 
exclude_CTL = [];               % excluded control sweeps
exclude_VER = [7,45];           % excluded vertical sweeps
exclude_HOR = [];              % excluded horizontal sweeps

VER_perturbation = [1,2,4,6,14,19,21,23,30,33,34,35,39,40,43,46,48,50,51,54,56,58]; 
HOR_perturbation = [4,9,10,12,15,19,26,28,29,35,36,44,55,61,78,87,88,91,94]; 


VER_perturbation_delay    = VER_perturbation([1,2,3,5,6,8,11,12,13,14,15,16,17,18,19,20]);
VER_perturbation_notDelay = VER_perturbation([4,7,9,10]);

[VER_yes, VER_no] = sort_sweeps(size(data{VER,SOL},1), VER_perturbation,  exclude_VER); 
[HOR_yes, HOR_no] = sort_sweeps(size(data{HOR,SOL},1), HOR_perturbation,  exclude_HOR); 
[VER_delay]    = sort_sweeps(size(data{VER,SOL},1), VER_perturbation_delay   ,  exclude_VER); 
[VER_notDelay] = sort_sweeps(size(data{VER,SOL},1), VER_perturbation_notDelay,  exclude_VER); 



for i = [SOL, TA, FSR, ANG]
    data{CTL,i}(exclude_CTL,:) = []; 
    data{VER,i}(exclude_VER,:) = []; 
    data{HOR,i}(exclude_HOR,:) = []; 
end 

%% Load offset 
raw = data;                     % save raw data 
load('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_bene_ver'); 
offset_ver = offset; 
load('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_bene_hor'); 
offset_hor = offset; 

%offset_hor = zeros(1,size(data{HOR,SOL},1)); 
%offset_ver = zeros(1,size(data{VER,SOL},1)); 


%% Filtrering and detrend (similar to MR. kick)
% Functions used: [rectify_filter()], [filt_FSR()]

fc = 40;                            % Cutoff frequency for LowPass filter
order = 1;                          % Filter order 
[b,a] = butter(order,fc/(Fs/2));    % Filter coefficient

% rectify and filter EMG. Remove noise in FSR 
for proto = ProtoAll
    [data{proto,SOL}, data{proto,TA}] = rectify_filter(data{proto,SOL}, data{proto,TA}, b, a);  
    [data{proto,FSR}] = func_filt_FSR(data{proto,FSR}, "test", 1 , "limit_pct", 0.95); 
end


%% 𝔹𝕖𝕘𝕚𝕟𝕚𝕟𝕘 𝕠𝕗 𝕊𝕥𝕒𝕟𝕕- 𝕒𝕟𝕕 𝕊𝕨𝕚𝕟𝕘𝕡𝕙𝕒𝕤?
% Functions used: [step_index_pos()], [step_index_corr()], [step_corr()]

% find change in FSR signal
step_index = cell(3,1);
error_index = cell(3,1);

for proto = ProtoAll
    [step_index{proto}, error_index{proto}] = func_step_index(data{proto,FSR});
end


if true 
    % >>>> TEST CODE <<<<
    proto = VER; % CTL[x], HOR[X], VER[x]
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
