%% Andreas stair, Unload 

clc;
clear; 
close all; 

SubjectName = "Christian"; 
addpath("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/FunctionFiles")

% plots/function enabled or disabled
normalize = false; 

%% Data path
folderpath = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/data/Christian_unload(05_08_2022)/";
filepath_CTL = folderpath + "control.mat"; 
filepath_Unload = folderpath + "unload.mat"; 

%% Abbreviation
CTL = 1; VER = 2; HOR = 3; CTL2 = 4; time = 5; unload = 6; 
SOL = 1; TA = 2; ANG = 3; FSR = 4; FOO = 5;  VEL = 6; ACC = 7;
ProtoAll = [CTL, unload];

%% Load data and Acquisition Set-Up from Mr Kick
[data{CTL,1:4}] = load_EMG(filepath_CTL);       clear filepath_CTL
[data{unload,1:4}] = load_EMG(filepath_Unload); clear filepath_Unload

% Acquisition Set-Up
sweep_length = 10;              % Signal length in second
Fs = 2000;                      % Samples per second
dt = 1/Fs;                      % Seconds per sample
pre_trig = 4;                   % Pre-trigger 
N = Fs*sweep_length;            % Total number of samples per signal

% Exclude data 
exclude_CTL = [];               % excluded control sweeps
exclude_unload = [23]; 

unload_perturbation = [2, 9, 12, 15, 17, 21, 25, 32, 35, 38, 40, 44, 46, 49, 53, 57, 59, 61, 63, 66]; 
[unload_yes, unload_no] = sort_sweeps(size(data{unload,SOL},1), unload_perturbation,  exclude_unload); 

for i = [SOL, TA, FSR, ANG]
    data{CTL,i}(exclude_CTL,:) = []; 
    data{unload,i}(exclude_unload,:) = []; 
end 

%% Offsets
% raw = data;                     % save raw data 
% load('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_andreas_ver'); 
% offset_ver = offset; 
% load('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_andreas_hor'); 
% offset_hor = offset; 

%% Filtrering and detrend 
fc = 40;                            % Cutoff frequency for LowPass filter
order = 1;                          % Filter order 
[b,a] = butter(order,fc/(Fs/2));    % Filter coefficient

for PROTO = ProtoAll
    [data{PROTO,SOL}, data{PROTO,TA}] = rectify_filter(data{PROTO,SOL}, data{PROTO,TA}, b, a);  % rectify and filter EMG
    [data{PROTO,FSR}] = filt_FSR( data{PROTO,FSR} );  
end


%% Begining of Stand- and Swingphase
step_index    = cell(3,1);
for PROTO = ProtoAll
    [step_index{PROTO}] = step_index_pos(data{PROTO,FSR});     % find change in FSR signal
end


% Manually remove error
for sweep = [9] % Unload frem
    temp = step_index{unload}(sweep,:); 
    step_corr{unload}(sweep,4)= temp(2); 
    step_corr{unload}(sweep,5)= temp(3);         
end


for sweep = [1,3,4,6,10,11,12,14,19,20,21,22,26,29,36,37,40,42,46,51,53,54,55,57,59,60] % Unload tilbage
    temp = step_index{unload}(sweep,:); 
    step_corr{unload}(sweep,4)= temp(6); 
    step_corr{unload}(sweep,5)= temp(7);         
end

[step_index{unload}] = step_index_corr(data{unload,FSR}, step_index{unload}, step_corr{unload}); 
sweep = 17;  step_index{unload}(sweep,[7,6,5,4,3,2]) = [9671,10810,11503, 12776, 13545, 14841];
sweep = 34;  step_index{unload}(sweep,[7,6,5,4,3,2]) = [9725,10864,11503, 12851, 13658, 15129];

for sweep = [54,59] % Unload tilbage (extra)
    temp = step_index{unload}(sweep,:); 
    step_corr{unload}(sweep,4)= temp(6); 
    step_corr{unload}(sweep,5)= temp(7);         
end
[step_index{unload}] = step_index_corr(data{unload,FSR}, step_index{unload}, step_corr{unload}); 



% Test for Error 
if false 
    % >>>> TEST CODE <<<<
    proto = unload; % Checked CTL and Unload
    loop = true; sweep = 1; 
    prompt = "Continue, press >c<" + newline + "Quite, press >q<"+ newline + "Change sweep number, press >t<"+ newline;
    
    while loop == true
        clc
        figure; hold on; sgtitle("sweep: " + sweep)
        plot(data{proto, ANG}(sweep,:))
        plot(data{proto, FSR}(sweep,:))
        
         % ground contact
        plot([8000,8000],[0 5], "LineWidth",3, "Color", "red")

        % step 6
        plot([step_index{proto}(sweep,2), step_index{proto}(sweep,2)],[0 5], "LineWidth",2, 'color', 'black')
        plot([step_index{proto}(sweep,3), step_index{proto}(sweep,3)],[0 5], "LineWidth",2, "Color", 'black')

        % step 4
        plot([step_index{proto}(sweep,4), step_index{proto}(sweep,4)],[0 5], "LineWidth",2, "Color", "green")
        plot([step_index{proto}(sweep,5), step_index{proto}(sweep,5)],[0 5], "LineWidth",2, "Color", "blue")
        
        % step 2
        plot([step_index{proto}(sweep,6), step_index{proto}(sweep,6)],[0 5], "LineWidth",2, 'color', 'black')
        plot([step_index{proto}(sweep,7), step_index{proto}(sweep,7)],[0 5], "LineWidth",2, "Color", 'black')
        
    

        
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
        close all; 
        sweep = sweep + 1;
        if sweep > size(data{proto,SOL},1)
            loop = false; 
        end 
    end
end

%% ℕ𝕠𝕣𝕞𝕒𝕝𝕚𝕫𝕖 𝔼𝕄𝔾 𝕤𝕚𝕫𝕖
% Metode: find the maximum EMG point during the second step of a moving
% average. 
span = 10; 


if normalize == true
    for PROTO = ProtoAll
        for i = 1:size(data{PROTO,FSR},1) %sweeps
            SOL_max(i) = max(smooth(data{PROTO,SOL}((i), step_index{PROTO}((i),7):step_index{PROTO}((i),6)),span, 'moving'));
            TA_max(i)  = max(smooth(data{PROTO,TA} ((i), step_index{PROTO}((i),7):step_index{PROTO}((i),6)),span, 'moving'));
        end

        SOL_max_gns = mean(SOL_max);    clear SOL_max
        TA_max_gns  = mean(TA_max);     clear TA_max
    
        for i = 1:size(data{VER,SOL},1) %sweeps
            data{PROTO,SOL}(i,:) =(data{PROTO,SOL}(i,:) )/SOL_max_gns;
            data{PROTO,TA}(i,:) = (data{PROTO,TA}(i,:) ) / TA_max_gns;
        end
        clear SOL_max SOL_max_gns TA_max TA_max_gns
    end 
end


%% 𝔸𝕝𝕚𝕘𝕟 𝕕𝕒𝕥𝕒
msToSec = @(x) x*10^-3; % ms to sec 
secToMs = @(x) x*10^3; % ms to sec 

data_stand = cell(numel(ProtoAll),5); 
for PROTO = ProtoAll
    [data_stand{PROTO,:}] = align_data_vol2(step_index{PROTO}, data{PROTO,1:4}, 'alignStep', 'four_begin');
end

before = 0; 
after = 0; 
data_align = cell(3,5); 
raw_align  = cell(3,5); 

for PROTO = ProtoAll
    [data_align{PROTO,:}] = align_data_vol2(step_index{PROTO}, data{PROTO,1:4}, 'sec_before', msToSec(before), 'sec_after', msToSec(after), 'alignStep', 'four_begin');
end

%% 𝔸𝕟𝕜𝕝𝕖 𝕕𝕖𝕘𝕣𝕖𝕖 𝕒𝕟𝕕 𝕊𝕡𝕖𝕖𝕕
 
span = 10;  % sets the span of the moving average to span.

for PROTO = ProtoAll
    for i = 1:size(data_stand{PROTO,ANG},1) %sweeps
        data_stand{PROTO,ANG}(i,:) = smooth(data_stand{PROTO,ANG}(i,:), span, 'moving'); % [ankle angle]
        data_align{PROTO,ANG}(i,:) = smooth(data_align{PROTO,ANG}(i,:), span, 'moving'); % [ankle angle]
    end
end 

angle_calibrate = mean(data_stand{CTL,ANG}(:,:),1);
minSensor = min(angle_calibrate);
maxSensor = max(angle_calibrate);
minVideo  = 86.5-90;
maxVideo  = 117.6-90;

x1 = minSensor; x2 = maxSensor; y1 = minVideo; y2 = maxVideo; 
a = (y2-y1)/(x2 - x1);
b = y1 - a*x1; 

span = 10; 
for PROTO = ProtoAll
    for i = 1:size(data_stand{PROTO,ANG},1) %sweeps
        %         data_stand{PROTO,ANG}(i,:) = data_stand{PROTO,ANG}(i,:)*a+b; % [ankle angle]
        %         data_align{PROTO,ANG}(i,:) = data_align{PROTO,ANG}(i,:)*a+b; % [ankle angle]

        data_align{PROTO,VEL}(i,:) = diff(data_align{PROTO,ANG}(i,:))/(dt*10^3);     % [deg/sec]
        data_align{PROTO,ACC}(i,:) = diff(smooth(data_align{PROTO,VEL}(i,:), span, 'moving'))/(dt*10^3);     % [deg/sec^2]
    end
end 


%% ℙ𝕝𝕠𝕥𝕤

if true
    figure; 
    proto = unload; 
        if proto == unload;  no = unload_no; yes = unload_yes; sgtitle("Unload (300ms) - subject " + SubjectName); end

    offset = 332; % [ms]
    %3
    startRange = []
    no_startRange = []

    x_range = [-400 400];
    subplot(511);hold on; ylabel("Position" + newline + "[Deg]"), xlim(x_range)
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,ANG}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,ANG}(yes,:),1), "LineWidth",1, "color","black")
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,ANG}(yes(startRange),:),1), "LineWidth",1, "color","red")
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,ANG}(no(no_startRange),:),1), "LineWidth",1, "color","blue")

        legend("control (N=52)", "Unload (N=19)")


        %%plot(data_align{CTL,time}*10^3-offset, mean(data_align{CTL,ANG}(:,:),1), "LineWidth",1, "color","red")


     subplot(512);hold on; ylabel("Velocity" + newline + "[Deg/ms]"), xlim(x_range)
        plot(data_align{proto,time}(1:end-1)*10^3-offset, mean(data_align{PROTO,VEL}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_align{proto,time}(1:end-1)*10^3-offset, mean(data_align{PROTO,VEL}(yes,:),1), "LineWidth",1, "color","black")

     subplot(513);hold on; ylabel("Aceleration" + newline + "[Deg/ms^2]"), xlim(x_range)
        plot(data_align{proto,time}(3:end)*10^3-offset, mean(data_align{PROTO,ACC}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
       plot(data_align{proto,time}(3:end)*10^3-offset, mean(data_align{PROTO,ACC}(yes,:),1), "LineWidth",1, "color","black")

     subplot(514);hold on; ylabel("Soleus" + newline + "EMG [\muV]"), xlim(x_range);
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
       plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(yes,:),1), "LineWidth",1, "color","black")
       plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(yes(startRange),:),1), "LineWidth",1, "color","red")
      plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(no(no_startRange),:),1), "LineWidth",1, "color","blue")


    subplot(515);hold on; ylabel("Tibialis" + newline + "EMG [\muV]"), xlim(x_range); xlabel("Timer after unload onset [ms]"); 
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,TA}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
       plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,TA}(yes,:),1), "LineWidth",1, "color","black")
       plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,TA}(yes(startRange),:),1), "LineWidth",1, "color","red")

end

%%
figure; hold on; ylabel("Soleus" + newline + "EMG [\muV]"), xlim(x_range);
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
       plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(yes,:),1), "LineWidth",1, "color","black")
       plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(yes(startRange),:),1), "LineWidth",1, "color","red")
    %  plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(no(no_startRange),:),1), "LineWidth",1, "color","blue")
