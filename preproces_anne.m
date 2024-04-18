clc; clear all; close all; 

SubjectName = "Andreas"

% Folderpath to main folder 
main_folderpath = strrep(fileparts(matlab.desktop.editor.getActiveFilename),'\','/');

% Folders path to data 
folderpath_data_part1 = main_folderpath + "/Data_MrKick/" + "Andreas_21feb.mat";

%% Abbreviation
fprintf('Script section: Abbreviation \n'); 

% protocol abbreviation types
CTL  = 1;  % Control nr 1
VER  = 2;  % Vertical 
HOR  = 3;  % Horizontal
CTL2 = 4; % Control nr 2  

Protocol_All = [CTL];
steps_tested = [2,4,6];

% Sensor abbreviation
SOL = 1; % Soleus 
TA  = 2; % Tibialis
ANG = 3; % Ankle position 
FSR = 4; % Force sensitiv resistor 

% include function folder
addpath(main_folderpath +"/FunctionFiles")

%% Load data and Acquisition Set-Up from Mr Kick
fprintf('Script section: Load data and Acquisition Set-Up from Mr Kick \n'); 
% Functions used: [load_EMG_v2()]

% load pre-control
[SOL_CTL1, TA_CTL1, angle_CTL1, FSR_CTL1] = load_EMG_v2(folderpath_data_part1); 

data    = cell(3,4);
% example:
%   data{protocol, sensor}(sweep, data number)

data{CTL,SOL} = [TA_CTL1];       
data{CTL,TA}  = [SOL_CTL1];        
data{CTL,ANG}  = [angle_CTL1]; 
data{CTL,FSR}  = [FSR_CTL1];      

% Acquisition Set-Up
sweep_length = 10;              % Signal length in second
Fs = 2000;                      % Samples per second
dt = 1/Fs;                      % Seconds per sample
pre_trig = 4;                   % Pre-trigger 
N = Fs*sweep_length;            % Total number of samples per signal


if ~(N == length(data{CTL,SOL}(1,:)))
    error_message =  "Wrong sweep length";
    errordlg(error_message , 'Error');
    error(error_message)
end


% Exclude sweep 
exclude_CTL = [5];               % excluded control sweeps
exclude_CTL2= [];               % excluded control sweeps
exclude_VER = [];               % excluded horizontal sweeps
exclude_HOR = [];               % excluded horizontal sweeps

fprintf(2,'\n     Files excluded. Check Mr Kick for files to exclude \n')
fprintf(2,'\n     %s\n', num2str(exclude_CTL));

for proto = Protocol_All
    for i = [SOL, TA, FSR, ANG]
        data{CTL,i}(exclude_CTL,:) = []; 
    end 
end

%% Preproces data 
run("Preproces.m")