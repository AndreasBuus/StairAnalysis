clc; 
clear all; 
close all; 

SubjectName = "Trine2"; 

% Folders path 
folderpath_10sem = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/data/Trine2 21.04.2022/";

% File Path 
filepath_CTL = folderpath_10sem + "control1.mat";
filepath_CTL_2 = folderpath_10sem + "control2.mat"; 
filepath_HOR = folderpath_10sem + "horisontal1.mat"; 
filepath_VER = folderpath_10sem + "vertical1.mat"; 

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


%% Load data and Acquisition Set-Up from Mr Kick
% Functions used: [load_EMG()]

data    = cell(numel(ProtoAll),4); % data{protocol, EMG}(sweep, data_num)

% Load control 
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



%% Perturbation and exclude trails
% Functions used: [sort_sweeps()]

% Exclude data 
exclude_CTL = [5,6,7,19,20];               % excluded control sweeps
exclude_CTL2= [8,13];               % excluded control sweeps
exclude_VER = [36];               % excluded horizontal sweeps
exclude_HOR = [];               % excluded horizontal sweeps

HOR_perturbation = [6,11,12,16,18,21,25,29,31,34,36,37,40,42,44,52,55,57,59,62]; % Checked

VER_perturbation = [1,7,10,12,13,15,19,21,24,28,32,34,38,41,43,45,47,49,56,59]; % Checked 
VER_perturbation_delay    = [1,7,10,15,19,24,28,32,34,38,41,43,45,47,56,59]; % delayed
VER_perturbation_notDelay = [12,13,21,49]; % not delayed


[VER_yes, VER_no] = sort_sweeps(size(data{VER,SOL},1), VER_perturbation,  exclude_VER); 
[VER_delay]    = sort_sweeps(size(data{VER,SOL},1), VER_perturbation_delay   ,  exclude_VER);
[VER_notDelay] = sort_sweeps(size(data{VER,SOL},1), VER_perturbation_notDelay,  exclude_VER); 

[HOR_yes, HOR_no] = sort_sweeps(size(data{HOR,SOL},1), HOR_perturbation,  exclude_HOR); 

for i = [SOL, TA, FSR, ANG]
    data{CTL,i}(exclude_CTL,:) = []; 
    data{CTL2,i}(exclude_CTL2,:) = []; 

    data{VER,i}(exclude_VER,:) = []; 
    data{HOR,i}(exclude_HOR,:) = []; 
end 

%% Load offset 

raw = data;                     % save raw data 
load('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_trine2_ver'); 
offset_ver = offset; 
load('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_trine2_hor'); 
offset_hor = offset; 

%% Filtrering and detrend (similar to MR. kick)
% Functions used: [rectify_filter()], [filt_FSR()]

fc = 40;                            % Cutoff frequency for LowPass filter
order = 1;                          % Filter order 
[b,a] = butter(order,fc/(Fs/2));    % Filter coefficient

for proto = ProtoAll
    [data{proto,SOL}, data{proto,TA}] = rectify_filter(data{proto,SOL}, data{proto,TA}, b, a);  % rectify and filter EMG
    [data{proto,FSR}] = func_filt_FSR(data{proto,FSR}, "test", 0, "limit_pct", 0.95); 
end


%% Begining of Stand- and Swingphase
% find change in FSR signal
step_index = cell(3,1);
error_index = cell(3,1);

for proto = ProtoAll
    [step_index{proto}, error_index{proto}] = func_step_index(data{proto,FSR});
end

sweep = 4; [step_index{CTL}(sweep,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',6, 'move_num', 2, 'sweep',sweep);
sweep = 6; [step_index{HOR}(sweep,:)] = func_step_index_corr('FSR', data{HOR,FSR}, 'direction', 'right', 'edge_num',6, 'move_num', 2, 'sweep',sweep);
sweep = 11; [step_index{HOR}(sweep,:)] = func_step_index_corr('FSR', data{HOR,FSR}, 'direction', 'right', 'edge_num',6, 'move_num', 2, 'sweep',sweep);
sweep = 22; [step_index{HOR}(sweep,:)] = func_step_index_corr('FSR', data{HOR,FSR}, 'direction', 'right', 'edge_num',8, 'move_num', 2, 'sweep',sweep);
sweep = 34; [step_index{HOR}(sweep,:)] = func_step_index_corr('FSR', data{HOR,FSR}, 'direction', 'right', 'edge_num',8, 'move_num', 2, 'sweep',sweep);


% sweep = 4; [step_index{CTL}(sweep,:)] = func_step_index_corr('FSR', data{CTL,FSR}, 'direction', 'right', 'edge_num',8, 'move_num', 2, 'sweep',sweep);
% 
% sweep = 11; [step_index{HOR}(sweep,:)] = func_step_index_corr('FSR', data{HOR,FSR}, 'direction', 'right', 'edge_num',8, 'move_num', 2, 'sweep',sweep);
% sweep = 14; [step_index{HOR}(sweep,:)] = func_step_index_corr('FSR', data{HOR,FSR}, 'direction', 'right', 'edge_num',8, 'move_num', 2, 'sweep',sweep);
% sweep = 31; [step_index{HOR}(sweep,:)] = func_step_index_corr('FSR', data{HOR,FSR}, 'direction', 'right', 'edge_num',8, 'move_num', 2, 'sweep',sweep);


if true 
    % >>>> TEST CODE <<<<
    proto = CTL2; % Checked: CTL[x], HOR[x], VER[x], CTL2[]
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
% step_index    = cell(3,1);
% for proto = ProtoAll
%     [step_index{proto}] = step_index_pos(data{proto,FSR});     % find change in FSR signal
% end
% 
% for sweep = [2,4,5,9,10,12,14,17,18,19,23,24,26,28,30,31,32,34,35,36,38,39,41,42,45,47,48,50,56,59]
%     temp = step_index{HOR}(sweep,:);
%     step_corr{HOR}(sweep,4)= temp(6); 
%     step_corr{HOR}(sweep,5)= temp(7);         
% end
% 
% sweep = 1;  step_corr{HOR}(sweep,5)=11576; step_corr{HOR}(sweep,4)=12416;
% sweep = 6;  step_corr{HOR}(sweep,5)=11066; step_corr{HOR}(sweep,4)=12610;
% sweep = 7;  step_corr{HOR}(sweep,5)=11291; step_corr{HOR}(sweep,4)=12141;
% sweep = 8;  step_corr{HOR}(sweep,5)=11068; step_corr{HOR}(sweep,4)=11924;
% sweep = 11;  step_corr{HOR}(sweep,5)=10918; step_corr{HOR}(sweep,4)=12222;
% sweep = 13;  step_corr{HOR}(sweep,5)=10995; step_corr{HOR}(sweep,4)=11981;
% sweep = 15;  step_corr{HOR}(sweep,5)=11105; step_corr{HOR}(sweep,4)=11921;
% sweep = 20;  step_corr{HOR}(sweep,5)=10819; step_corr{HOR}(sweep,4)=11640;
% sweep = 22;  step_corr{HOR}(sweep,5)=10947; step_corr{HOR}(sweep,4)=11734;
% sweep = 27;  step_corr{HOR}(sweep,5)=10788; step_corr{HOR}(sweep,4)=11595;
% sweep = 33;  step_corr{HOR}(sweep,5)=10871; step_corr{HOR}(sweep,4)=11664;
% sweep = 42;  step_corr{HOR}(sweep,5)=10845; step_corr{HOR}(sweep,4)=12076;
% sweep = 43;  step_corr{HOR}(sweep,5)=10859; step_corr{HOR}(sweep,4)=11715;
% sweep = 46;  step_corr{HOR}(sweep,5)=10780; step_corr{HOR}(sweep,4)=11626;
% sweep = 49;  step_corr{HOR}(sweep,5)=10882; step_corr{HOR}(sweep,4)=11712;
% sweep = 51;  step_corr{HOR}(sweep,5)=10930; step_corr{HOR}(sweep,4)=11652;
% sweep = 53;  step_corr{HOR}(sweep,5)=10699; step_corr{HOR}(sweep,4)=11412;
% sweep = 54;  step_corr{HOR}(sweep,5)=10718; step_corr{HOR}(sweep,4)=11506;
% sweep = 58;  step_corr{HOR}(sweep,5)=10769; step_corr{HOR}(sweep,4)=11519;
% sweep = 60;  step_corr{HOR}(sweep,5)=10756; step_corr{HOR}(sweep,4)=11639;
% sweep = 61;  step_corr{HOR}(sweep,5)=10616; step_corr{HOR}(sweep,4)=11292;
% 
% for sweep = [2,4,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,32,33,35,37,38,39,40,42,43,47,49,50,53,54,56,59]
%     temp = step_index{VER}(sweep,:);
%     step_corr{VER}(sweep,4)= temp(6); 
%     step_corr{VER}(sweep,5)= temp(7);         
% end
% %sweep = 6; step_corr{HOR}(sweep,[5,4]) = [12516, 16143]; 
% 
% sweep = 5;   step_corr{VER}(sweep,[5,4]) = [11850, 13048];
% sweep = 26;  step_corr{VER}(sweep,[5,4]) = [11025, 11855];
% sweep = 31;  step_corr{VER}(sweep,[5,4]) = [10847, 11699];
% sweep = 33;  step_corr{VER}(sweep,[5,4]) = [10678, 11415];
% sweep = 39;  step_corr{VER}(sweep,[5,4]) = [11008, 11974];
% sweep = 44;  step_corr{VER}(sweep,[5,4]) = [11089, 11866];
% sweep = 46;  step_corr{VER}(sweep,[5,4]) = [10821, 11490];
% sweep = 51;  step_corr{VER}(sweep,[5,4]) = [11028, 11912];
% sweep = 52;  step_corr{VER}(sweep,[5,4]) = [10990, 11715];
% sweep = 55;  step_corr{VER}(sweep,[5,4]) = [10789, 11528];
% sweep = 57;  step_corr{VER}(sweep,[5,4]) = [11221, 12030];
% 
% for sweep = [3,4,5,6,7,8,9,11,12,13,14,15,16,17]
%     temp = step_index{CTL}(sweep,:); 
%     step_corr{CTL}(sweep,4)= temp(6); 
%     step_corr{CTL}(sweep,5)= temp(7);         
% end
% 
% %CTL2
% for sweep = [1,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20] %CTL2
%     temp = step_index{CTL2}(sweep,:); 
%     step_corr{CTL2}(sweep,4)= temp(6); 
%     step_corr{CTL2}(sweep,5)= temp(7);         
% end
% sweep = 2;  step_corr{CTL2}(sweep,[5,4]) = [10878, 11623];
% sweep = 6;  step_corr{CTL2}(sweep,[5,4]) = [10694, 11382];
% sweep = 12;  step_corr{CTL2}(sweep,[5,4]) = [10635, 11272];
% sweep = 13;  step_corr{CTL2}(sweep,[5,4]) = [10078, 10794];
% sweep = 18;  step_corr{CTL2}(sweep,[5,4]) = [10548, 11320];
% 
% 
% %first time
% [step_index{HOR}] = step_index_corr(data{HOR,FSR}, step_index{HOR}, step_corr{HOR}); 
% [step_index{VER}] = step_index_corr(data{VER,FSR}, step_index{VER}, step_corr{VER}); 
% [step_index{CTL}] = step_index_corr(data{CTL,FSR}, step_index{CTL}, step_corr{CTL}); 
% [step_index{CTL2}] = step_index_corr(data{CTL2,FSR}, step_index{CTL2}, step_corr{CTL2}); 
% 
% 
% for sweep = [12,14,17]
%     temp = step_index{CTL}(sweep,:); 
%     step_corr{CTL}(sweep,4)= temp(6); 
%     step_corr{CTL}(sweep,5)= temp(7);         
% end
% 
% %CTL2
% for sweep = [3,4,5,9,10,11,14,15,16,17] 
%     temp = step_index{CTL2}(sweep,:); 
%     step_corr{CTL2}(sweep,4)= temp(6); 
%     step_corr{CTL2}(sweep,5)= temp(7);         
% end
% 
% %second time
% [step_index{CTL}] = step_index_corr(data{CTL,FSR}, step_index{CTL}, step_corr{CTL}); 
% [step_index{CTL2}] = step_index_corr(data{CTL2,FSR}, step_index{CTL2}, step_corr{CTL2}); 
% 
% % -- var: "Stand_duration" 
% % Duration of stand phase of tread 4
% % stand_duration{proto}(sweep)
% stand_DUR    = cell(3,1);
% for proto = ProtoAll
%     for i = 1:size(data{proto,FSR},1)
%         stand_DUR{proto}(i) = (step_index{proto}(i,4) - step_index{proto}(i,5))*dt; 
%     end
% end
% 
% 
% if false 
%     % >>>> TEST CODE <<<<
%     proto = CTL2; % Checked HOR, VER
%     loop = true; sweep = 1; 
%     prompt = "Continue, press >c<" + newline + "Quite, press >q<"+ newline + "Change sweep number, press >t<"+ newline;
%     
%     while loop == true
%         clc
%         figure; hold on; sgtitle("sweep: " + sweep)
%         plot(data{proto, ANG}(sweep,:))
%         plot(data{proto, FSR}(sweep,:))
%         plot([8000, 8000],[-2 6], "LineWidth",3, "Color", "black")
% 
%         plot([step_index{proto}(sweep,4), step_index{proto}(sweep,4)],[0 5], "LineWidth",2, "Color", "green")
%         plot([step_index{proto}(sweep,5), step_index{proto}(sweep,5)],[0 5], "LineWidth",2, "Color", "blue")
%         plot([step_index{proto}(sweep,6), step_index{proto}(sweep,6)],[0 5], "LineWidth",2, 'color', 'black')
%         plot([step_index{proto}(sweep,7), step_index{proto}(sweep,7)],[0 5], "LineWidth",2, "Color", 'black')
% 
%         
%         correctInput = false; 
%         while correctInput == false
%             str = input(prompt, 's');
%             if strcmp(str,"q")
%                 disp("Loop stopped")
%                 loop = false; correctInput = true; 
%             elseif strcmp(str,"t")
%                 sweep = input("New sweep number: ")-1; 
%                 correctInput = true; 
%             elseif strcmp(str,"c") %, sweep == size(data{proto, SOL}))
%                 correctInput = true; 
%             end 
%     
%             if correctInput == false
%                 warning("Input not accepted")
%             end
%         end
%         close all; 
%         sweep = sweep + 1;
%         if sweep > size(data{proto,SOL},1)
%             loop = false; 
%         end 
%     end
% end


%% Save Data 

type{1} = VER_yes;   type{2} = VER_no; 
type{3} = HOR_yes;   type{4} = HOR_no; 
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName + "_data", 'data')
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName+"_type",'type')
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/data_preprocessed/" + SubjectName+"_step",'step_index')

disp("Processed: " + SubjectName)
