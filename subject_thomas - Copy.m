clc; 
clear all; 
close all; 

SubjectName = "Thomas"; 

% plots/function enabled or disabled
normalize = true; 

% Folders path 
folderpath_10sem = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/data/Thomas 16.03.2022/"; 

% File Path 
filepath_CTL = folderpath_10sem + "thomas_control_(16_3_2022).mat";
filepath_CTL_2=folderpath_10sem + "thomas_control2_(16_3_2022).mat"; 
filepath_HOR = folderpath_10sem + "thomas_horizontal_(16_3_2022).mat"; 
filepath_VER = folderpath_10sem + "thomas_vertical_(16_3_2022).mat"; 


%% Abbreviation
CTL = 1; VER = 2; HOR = 3; CTL2 = 4; time = 5;
SOL = 1; TA = 2; ANG = 3; FSR = 4; FOO = 5; 
VEL = 1; ACC = 2; 

ProtoAll = [CTL, VER, HOR, CTL2];

%% 𝕃𝕠𝕒𝕕 𝕕𝕒𝕥𝕒 𝕒𝕟𝕕 𝔸𝕔𝕢𝕦𝕚𝕤𝕚𝕥𝕚𝕠𝕟 𝕊𝕖𝕥-𝕌𝕡 𝕗𝕣𝕠𝕞 𝕄𝕣 𝕂𝕚𝕔𝕜

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

% Exclude data 
exclude_CTL = [];               % excluded control sweeps
exclude_CTL2= [];               % excluded control sweeps
exclude_VER = [];               % excluded horizontal sweeps
exclude_HOR = [];               % excluded horizontal sweeps

HOR_perturbation = [3,4,6,8,9,12,14,16,24,29,32,33,35,42,45,46,47,51,52,56]; 
VER_perturbation = [2,3,4,5,7,8,9,14,17,19,27,30,31,34,35,38,40,46,47,59]; % checked

VER_perturbation_delay     = VER_perturbation([1,2,12,13,14,17,19]); % delayed
VER_perturbation_1notDelay = VER_perturbation([3,4,5,6,7,8,9,10,11]); 
VER_perturbation_2notDelay = VER_perturbation([15,16,18,20]); 

[VER_yes, VER_no] = sort_sweeps(size(data{VER,SOL},1), VER_perturbation,  exclude_VER); 
[VER_delay]    = sort_sweeps(size(data{VER,SOL},1), VER_perturbation_delay   ,  exclude_VER); 
[VER_notDelay1] = sort_sweeps(size(data{VER,SOL},1), VER_perturbation_1notDelay,  exclude_VER); 
[VER_notDelay2] = sort_sweeps(size(data{VER,SOL},1), VER_perturbation_2notDelay,  exclude_VER); 
[HOR_yes, HOR_no] = sort_sweeps(size(data{HOR,SOL},1), HOR_perturbation,  exclude_HOR); 

VER_notDelay = [VER_notDelay1, VER_notDelay2]; 

for i = [SOL, TA, FSR, ANG]
    data{CTL,i}(exclude_CTL,:) = []; 
    data{CTL2,i}(exclude_CTL2,:) = []; 

    data{VER,i}(exclude_VER,:) = []; 
    data{HOR,i}(exclude_HOR,:) = []; 
end 

raw = data;                 
%load("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_thomas_ver_test"); 
load("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_thomas_ver"); 

offset_ver = offset; 
load("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_thomas_hor"); 
offset_hor = offset; 

%% 𝔽𝕚𝕝𝕥𝕣𝕖𝕣𝕚𝕟𝕘 𝕒𝕟𝕕 𝕕𝕖𝕥𝕣𝕖𝕟𝕕
fc = 40;                            % Cutoff frequency for LowPass filter
order = 1;                          % Filter order 
[b,a] = butter(order,fc/(Fs/2));    % Filter coefficient

for PROTO = ProtoAll
    [data{PROTO,SOL}, data{PROTO,TA}] = rectify_filter(data{PROTO,SOL}, data{PROTO,TA}, b, a);  % rectify and filter EMG
    [data{PROTO,FSR}] = filt_FSR( data{PROTO,FSR} );  
end


%% 𝔹𝕖𝕘𝕚𝕟𝕚𝕟𝕘 𝕠𝕗 𝕊𝕥𝕒𝕟𝕕- 𝕒𝕟𝕕 𝕊𝕨𝕚𝕟𝕘𝕡𝕙𝕒𝕤𝕖
step_index    = cell(3,1);
for PROTO = ProtoAll
    [step_index{PROTO}] = step_index_pos(data{PROTO,FSR});     % find change in FSR signal
end

% -- var: "Stand_duration" 
% Duration of stand phase of tread 4
% stand_duration{PROTO}(sweep)
stand_DUR    = cell(3,1);
for PROTO = ProtoAll
    for i = 1:size(data{PROTO,FSR},1)
        stand_DUR{PROTO}(i) = (step_index{PROTO}(i,4) - step_index{PROTO}(i,5))*dt; 
    end
end

step_corr{HOR}(1,5)=11574; step_corr{HOR}(1,4)=12627; 
step_corr{HOR}(2,5)=11650; step_corr{HOR}(2,4)=12892; 
step_corr{HOR}(7,5)=11801; step_corr{HOR}(7,4)=13056; 
step_corr{HOR}(17,5)=11604; step_corr{HOR}(17,4)=12749; 
step_corr{HOR}(20,5)=11628; step_corr{HOR}(20,4)=12851; 
step_corr{HOR}(32,5)=12009; step_corr{HOR}(32,4)=13349; 
step_corr{HOR}(33,5)=11862; step_corr{HOR}(33,4)=13487; 
step_corr{HOR}(38,5)=11855; step_corr{HOR}(38,4)=13005; 
step_corr{HOR}(40,5)=11963; step_corr{HOR}(40,4)=12973; 
step_corr{HOR}(47,5)=11937; step_corr{HOR}(47,4)=13461; 
step_corr{HOR}(49,5)=11876; step_corr{HOR}(49,4)=13021; 

step_corr{VER}(1,5)=11539; step_corr{VER}(1,4)=12701; 
step_corr{VER}(4,5)=11770; step_corr{VER}(4,4)=12896; 
step_corr{VER}(10,5)=11694; step_corr{VER}(10,4)=12855; 
step_corr{VER}(12,5)=11674; step_corr{VER}(12,4)=12866; 
step_corr{VER}(18,5)=11959; step_corr{VER}(18,4)=13101; 
step_corr{VER}(21,5)=11681; step_corr{VER}(21,4)=12898; 
step_corr{VER}(22,5)=11969; step_corr{VER}(22,4)=13181; 
step_corr{VER}(24,5)=11770; step_corr{VER}(24,4)=12978; 
step_corr{VER}(25,5)=11753; step_corr{VER}(25,4)=12932; 
step_corr{VER}(26,5)=11610; step_corr{VER}(26,4)=12837; 
step_corr{VER}(28,5)=11632; step_corr{VER}(28,4)=12839; 
step_corr{VER}(29,5)=11658; step_corr{VER}(29,4)=12818; 
step_corr{VER}(41,5)=11641; step_corr{VER}(41,4)=12796; 
step_corr{VER}(44,5)=11676; step_corr{VER}(44,4)=12973; 


for sweep = [2]
    temp = step_index{CTL}(sweep,:); 
    step_corr{CTL}(sweep,4)= temp(6); 
    step_corr{CTL}(sweep,5)= temp(7);         
end

for sweep = [1,2]
    temp = step_index{CTL2}(sweep,:); 
    step_corr{CTL2}(sweep,4)= temp(6); 
    step_corr{CTL2}(sweep,5)= temp(7);         
end

[step_index{HOR}] = step_index_corr(data{HOR,FSR}, step_index{HOR}, step_corr{HOR}); 
[step_index{VER}] = step_index_corr(data{VER,FSR}, step_index{VER}, step_corr{VER}); 
[step_index{CTL}] = step_index_corr(data{CTL,FSR}, step_index{CTL}, step_corr{CTL}); 
[step_index{CTL2}] = step_index_corr(data{CTL2,FSR}, step_index{CTL2}, step_corr{CTL2}); 


if false 
    % >>>> TEST CODE <<<<
    proto = CTL2; %CTL [x], VER [x], HOR [x], CTL2[x]
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

if normalize == true
    disp("Normalizing enable")
    % (sweep,6) -  Second step, falling  
    % (sweep,7) -  Second step, rising  


    % VERTICAL 
    span = 15; 
    for i = 1:size(data{VER,SOL},1) %sweeps
        SOL_max(i) = max(smooth(data{VER,SOL}((i), step_index{VER}((i),7):step_index{VER}((i),6)),span, 'moving'));
        TA_max(i)  = max(smooth(data{VER,TA} ((i), step_index{VER}((i),7):step_index{VER}((i),6)),span, 'moving'));
    end
    SOL_max_gns = mean(SOL_max);    clear SOL_max
    TA_max_gns  = mean(TA_max);     clear TA_max
    
    for PROTO = VER %protocol
        for i = 1:size(data{VER,SOL},1) %sweeps
            data{PROTO,SOL}(i,:) =(data{PROTO,SOL}(i,:) )/SOL_max_gns;
            data{PROTO,TA}(i,:) = (data{PROTO,TA}(i,:) ) / TA_max_gns;
        end
    end 

    % HORISONTAL 
    span = 15; 
    for i = 1:size(data{HOR,SOL},1) %sweeps
        SOL_max(i) = max(smooth(data{HOR,SOL}((i), step_index{HOR}((i),7):step_index{HOR}((i),6)),span, 'moving'));
        TA_max(i)  = max(smooth(data{HOR,TA} ((i), step_index{HOR}((i),7):step_index{HOR}((i),6)),span, 'moving'));
    end
    
    SOL_max_gns = mean(SOL_max);    clear SOL_max
    TA_max_gns  = mean(TA_max);     clear TA_max
    
    for PROTO = HOR %protocol
        for i = 1:size(data{HOR,SOL},1) %sweeps
            data{PROTO,SOL}(i,:) = (data{PROTO,SOL}(i,:) )/SOL_max_gns;
            data{PROTO,TA}(i,:)  = (data{PROTO,TA}(i,:) ) / TA_max_gns;
        end
    end 
end


%% 𝔸𝕝𝕚𝕘𝕟 𝕕𝕒𝕥𝕒
msToSec = @(x) x*10^-3; % ms to sec 
secToMs = @(x) x*10^3; % ms to sec 

% 1: data_stand{protocol, EMG}(sweep, data_num)
data_stand = cell(3,5); time = 5;
for PROTO = ProtoAll
    [data_stand{PROTO,:}] = align_data_vol2(step_index{PROTO}, data{PROTO,1:4}, 'alignStep', 'four_begin');
end

% 2: data_align{protocol, EMG}(sweep, data_num)
before = 1200; 
after = 2000; 
data_align = cell(3,5); 
raw_align  = cell(3,5); 

for PROTO = ProtoAll
    [data_align{PROTO,:}] = align_data_vol2(step_index{PROTO}, data{PROTO,1:4}, 'sec_before', msToSec(before), 'sec_after', msToSec(after), 'alignStep', 'four_begin');
    [raw_align{PROTO,:}] = align_data_vol2(step_index{PROTO}, raw{PROTO,1:4}, 'sec_before', msToSec(before), 'sec_after', msToSec(after), 'alignStep', 'four_begin' );
end

%% pre-baseline vs post-baseline 
if false 
    figure; sgtitle("");
    gray = [0.7,0.7,0.7];
    offset = 0; % [ms]
    x_range = [-100 2000]; 

    x4 = [1660 1013 1013 1660];
    x2 = [0 600 600 0];
    y = [-10 -10 1000 1000];

    subplot(311);hold on; ylabel("Position" +newline+ "[Deg]"), xlim(x_range); ylim([-10, 5])
        patch(x2,y,"blue",'FaceAlpha',.2, 'EdgeColor', "none")
        patch(x4,y,"red",'FaceAlpha',.2, 'EdgeColor', "none")
        plot(data_align{CTL,time}*10^3-offset, mean(data_align{CTL,ANG}(:,:),1), "LineWidth",3, "color",gray)
        plot(data_align{VER,time}*10^3-offset, mean(data_align{VER,ANG}(VER_no(:),:),1), "LineWidth",1, "color","red")
        plot(data_align{HOR,time}*10^3-offset, mean(data_align{HOR,ANG}(HOR_no(:),:),1), "LineWidth",1, "color","blue")
        plot(data_align{CTL2,time}*10^3-offset, mean(data_align{CTL2,ANG}(:,:),1), "LineWidth",1, "color","black")

        % plot(data_align{VER,time}*10^3-offset, mean(data_align{VER,ANG}(VER_yes(:),:),1), "LineWidth",1, "color","green")
       %plot(data_align{HOR,time}*10^3-offset, mean(data_align{HOR,ANG}(HOR_yes(:),:),1), "LineWidth",1, "color","magenta")

        legend("","","(1) pre-baseline","(2) control-vertical","(3) control-horizontal","(4) post-baseline")

        
    subplot(312);hold on; ylabel("Soleus"+newline+"[\muV]"), xlim(x_range); ylim([0 800])
        patch(x2,y,"blue",'FaceAlpha',.2, 'EdgeColor', "none")
        patch(x4,y,"red",'FaceAlpha',.2, 'EdgeColor', "none")
        plot(data_align{CTL,time}*10^3-offset, mean(data_align{CTL,SOL}(:,:),1), "LineWidth",1, "color",gray)
        plot(data_align{VER,time}*10^3-offset, mean(data_align{VER,SOL}(VER_no(:),:),1), "LineWidth",1, "color","red")
        plot(data_align{HOR,time}*10^3-offset, mean(data_align{HOR,SOL}(HOR_no(:),:),1), "LineWidth",1, "color","blue")
        plot(data_align{CTL2,time}*10^3-offset, mean(data_align{CTL2,SOL}(:,:),1), "LineWidth",1, "color","black")

    subplot(313);hold on; ylabel("Tibialis"+newline+"[\muV]"), xlim(x_range); ylim([0 400])
        patch(x2,y,"blue",'FaceAlpha',.2, 'EdgeColor', "none")
        patch(x4,y,"red",'FaceAlpha',.2, 'EdgeColor', "none")
        plot(data_align{CTL,time}*10^3-offset, mean(data_align{CTL,TA}(:,:),1), "LineWidth",1, "color",gray)
        plot(data_align{VER,time}*10^3-offset, mean(data_align{VER,TA}(VER_no(:),:),1), "LineWidth",1, "color","red")
        plot(data_align{HOR,time}*10^3-offset, mean(data_align{HOR,TA}(HOR_no(:),:),1), "LineWidth",1, "color","blue")
        plot(data_align{CTL2,time}*10^3-offset, mean(data_align{CTL2,TA}(:,:),1), "LineWidth",1, "color","black")

    xlabel("Time after foot-strike with step four [ms]")
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

for PROTO = ProtoAll
    for i = 1:size(data_stand{PROTO,ANG},1) %sweeps
        data{PROTO,ANG}(i,:) = data{PROTO,ANG}(i,:)*a+b; % [ankle angle]
        data_stand{PROTO,ANG}(i,:) = data_stand{PROTO,ANG}(i,:)*a+b; % [ankle angle]
        data_align{PROTO,ANG}(i,:) = data_align{PROTO,ANG}(i,:)*a+b; % [ankle angle]

        velocity_stand{PROTO}(i,:) = diff(data_stand{PROTO,ANG}(i,:))/(dt*10^3);     % [deg/sec]
        velocity_align{PROTO}(i,:) = diff(data_align{PROTO,ANG}(i,:))/(dt*10^3);     % [deg/sec]
        span = 10; 
        aceleration_stand{PROTO}(i,:) = diff(smooth(velocity_stand{PROTO}(i,:), span, 'moving'))/(dt*10^3);     % [deg/sec]
        aceleration_align{PROTO}(i,:) = diff(smooth(velocity_align{PROTO}(i,:), span, 'moving'))/(dt*10^3);     % [deg/sec]
    end
end 
        

%% ℙ𝕝𝕠𝕥𝕤 - false

if true
    figure; 
    proto = VER; 
        if proto == HOR;  no = HOR_no; yes = HOR_yes; sgtitle("Horizontal Pertubation - subject " + SubjectName); end
        if proto == VER;  no = VER_no; yes = VER_yes; sgtitle(""); end

    offset = 0; % [ms]
    x_range = [-200 200]; 
    offset_delay = 0; 
    subplot(311);hold on; ylabel("Position"+newline+"[Deg]"), xlim(x_range)
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,ANG}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,ANG}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_align{proto,time}*10^3-offset-offset_delay, mean(data_align{proto,ANG}(VER_notDelay,:),1), "LineWidth",1, "color","black", "LineStyle","--")
        legend("Control", "Vertical perturbation, type 1", "Vertical perturbation, type 2")

     subplot(312);hold on; ylabel("Velocity"+newline+"[Deg/ms]"), xlim(x_range)
        plot(data_align{proto,time}(1:end-1)*10^3-offset, mean(velocity_align{proto}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_align{proto,time}(1:end-1)*10^3-offset, mean(velocity_align{proto}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_align{proto,time}(1:end-1)*10^3-offset-offset_delay, mean(velocity_align{proto}(VER_notDelay,:),1), "LineWidth",1, "color","black","LineStyle","--")

     subplot(313);hold on; ylabel("Soleus"+newline+"[\muV]"), xlim(x_range)
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(VER_notDelay,:),1), "LineWidth",1, "color","black","LineStyle","--")


        %      subplot(513);hold on; ylabel("Aceleration"+newline+"[Deg/ms^2]"), xlim(x_range)
%         plot(data_align{proto,time}(3:end)*10^3-offset, mean(aceleration_align{proto}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
%         plot(data_align{proto,time}(3:end)*10^3-offset, mean(aceleration_align{proto}(VER_delay,:),1), "LineWidth",1, "color","black")
%         plot(data_align{proto,time}(3:end)*10^3-offset, mean(aceleration_align{proto}(VER_notDelay,:),1), "LineWidth",1, "color","black","LineStyle","--")
%         legend("Control", "Vertical perturbation, type 1", "Vertical perturbation, type 2")
% 
%      subplot(514);hold on; ylabel("Soleus"+newline+"[\muV]"), xlim(x_range)
%         plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
%         plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(VER_delay,:),1), "LineWidth",1, "color","black")
%         plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,SOL}(VER_notDelay,:),1), "LineWidth",1, "color","black","LineStyle","--")
% 
%     subplot(515);hold on; ylabel("Tibialis"+newline+"[\muV]"), xlim(x_range);
%         plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,TA}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
%         plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,TA}(VER_delay,:),1), "LineWidth",1, "color","black")
%         plot(data_align{proto,time}*10^3-offset, mean(data_align{proto,TA}(VER_notDelay,:),1), "LineWidth",1, "color","black","LineStyle","--")
    
    xlabel("Time after foot-strike with fourth step [ms]");
end


%% Individual offset 
data_indiv = cell(3,5); 
viewBefore = 1000; % [ms]
viewAfter = 1000; % [ms]
for PROTO = [VER, HOR]
    for i = 1:size(data_stand{PROTO,ANG},1) %sweeps 
        if PROTO == VER 
            offs = offset_ver; 
        elseif PROTO == HOR
            offs = offset_hor;
        end 
        indexZero = Fs*before*10^-3;    % zero position [sample]
        off = Fs*offs(i)*10^-3;   % offset [sample]
        befo = Fs*viewBefore*10^-3;     % view before [sample]
        afte = Fs*viewAfter*10^-3;      % view after [sample]
        inc_arr = [round(indexZero+off-befo) : round(indexZero+off+afte)]; 
        
        for mode = [SOL, TA, FSR, ANG]
            data_indiv{PROTO,mode}(i,:) = data_align{PROTO,mode}(i,inc_arr);
        end
    end 
end

% Time array
for PROTO = ProtoAll
    data_indiv{PROTO,time} = rescale(inc_arr,-viewBefore,viewAfter);    % time [ms]
end

% Speed and aceleration
for PROTO = [VER, HOR]
    for i = 1:size(data_stand{PROTO,ANG},1) %sweeps
        velocity_indiv{PROTO}(i,:) = diff(data_indiv{PROTO,ANG}(i,:))/(dt*10^3);     % [deg/sec]
        span = 10; 
        aceleration_indiv{PROTO}(i,:) = diff(smooth(velocity_indiv{PROTO}(i,:), span, 'moving'))/(dt*10^3);     % [deg/sec]
    end
end 

if true
    figure; % New alignment 
    proto = HOR; 
        if proto == HOR;  no = HOR_no; yes = HOR_yes; sgtitle("Horizontal Pertubation - new align - subject " + SubjectName); end
        if proto == VER;  no = VER_no; yes = VER_yes; sgtitle("Vertical Pertubation - new align - subject " + SubjectName); end
    
    % ([1,2, 12, 13, 14, 17,19])
    x_range = [-800 1000]; 
    subplot(511);hold on; ylabel("Position" + newline + "[Deg]"), xlim(x_range)
        plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(yes,:),1), "LineWidth",1, "color","black")

     subplot(512);hold on; ylabel("Velocity" + newline + "[Deg/ms]"), xlim(x_range)
        plot(data_indiv{proto,time}(1:end-1), mean(velocity_indiv{proto}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}(1:end-1), mean(velocity_indiv{proto}(yes,:),1), "LineWidth",1, "color","black")

     subplot(513);hold on; ylabel("Aceleration" + newline + "[Deg/ms^2]"), xlim(x_range)
        plot(data_indiv{proto,time}(3:end), mean(aceleration_indiv{proto}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}(3:end), mean(aceleration_indiv{proto}(yes,:),1), "LineWidth",1, "color","black")

     subplot(514);hold on; ylabel("Soleus EMG" + newline + "[\muV]"), xlim(x_range)
        plot(data_indiv{proto,time}, mean(data_indiv{proto,SOL}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}, mean(data_indiv{proto,SOL}(yes,:),1), "LineWidth",1, "color","black")

    subplot(515);hold on; ylabel("Tibialis EMG" + newline + "[\muV]"), xlim(x_range); xlabel("Time [ms]");
        plot(data_indiv{proto,time}, mean(data_indiv{proto,TA}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}, mean(data_indiv{proto,TA}(yes,:),1), "LineWidth",1, "color","black")
end

type{1} = VER_yes; type{2} = VER_no; type{3} = HOR_yes; type{4} = HOR_no; 
%type{5} = VER_delay; % VER_notDelay

save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/Indiv_data/"+SubjectName,'data_indiv')
save("C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/Indiv_data/"+SubjectName+"_Type",'type')

%%  INDIV
idx1 = [1,2,12,13,14,17,19]; 
idx2 = [3,4,5,6,7,8,9,10,11]; 
if true
    figure; 
    proto = VER; 
        if proto == HOR;  no = HOR_no; yes = HOR_yes; sgtitle("Horizontal Pertubation - isolating - subject " + SubjectName); end
        if proto == VER;  no = VER_no; yes = VER_yes; sgtitle(""); end
    
    %x = [-0 1000 1000 -0];
    y = [-10 -10 1000 1000];
    x = [-150 150 150 -150];

    patchcolor = "black"; 
    FaceAlpha = 0.1; 
    % ([1,2, 12, 13, 14, 17,19])
    x_range = [-180 180]; 
    subplot(511);hold on; ylabel("Position" + newline + "[Deg]"), xlim(x_range); ylim([-5 11]) %ylim([-5 33])
            patch(x,y,patchcolor,'FaceAlpha',FaceAlpha, 'EdgeColor', "none")
        plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(VER_no,:),1), "LineWidth",3, "color",[0.7, 0.7, 0.7])
        plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(VER_notDelay,:),1), "LineWidth",1, "color","black", "LineStyle","--")

        %plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(VER_notDelay1,:),1), "LineWidth",1, "color","red")
        %plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(VER_notDelay2,:),1), "LineWidth",1, "color","blue")
        % excluder 3,4,5,6,7,8,9,10,11

        %plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(yes([3,4,5,6,7,8,9,10,11]),:),1), "LineWidth",1, "color","green")


     subplot(512);hold on; ylabel("Velocity" + newline + "[Deg/ms]"), xlim(x_range), ylim([-0.45 0.3])
             patch(x,y,patchcolor,'FaceAlpha',FaceAlpha, 'EdgeColor', "none")
        plot(data_indiv{proto,time}(1:end-1), mean(velocity_indiv{proto}(VER_no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}(1:end-1), mean(velocity_indiv{proto}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_indiv{proto,time}(1:end-1), mean(velocity_indiv{proto}(VER_notDelay,:),1), "LineWidth",1, "color","black", "LineStyle","--")
        %plot(data_indiv{proto,time}(1:end-1), mean(velocity_indiv{proto}(VER_notDelay2,:),1), "LineWidth",1, "color","blue")

     subplot(513);hold on; ylabel("Aceleration" + newline + "[Deg/ms^2]"), xlim(x_range); ylim([-0.12 0.12])
        plot(data_indiv{proto,time}(3:end), mean(aceleration_indiv{proto}(VER_no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}(3:end), mean(aceleration_indiv{proto}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_indiv{proto,time}(3:end), mean(aceleration_indiv{proto}(VER_notDelay,:),1), "LineWidth",1, "color","black", "LineStyle","--")
        %plot(data_indiv{proto,time}(3:end), mean(aceleration_indiv{proto}(VER_notDelay2,:),1), "LineWidth",1, "color","blue")

      % legend(["", "Control (n = 39)", "Perturbation type 1 (n=7)", "Perturbation type 2 (n=13)"])

     subplot(514);hold on; ylabel("Soleus EMG" + newline + "[\muV]"), xlim(x_range);  % ylim([0 1000])
        plot(data_indiv{proto,time}, mean(data_indiv{proto,SOL}(VER_no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}, mean(data_indiv{proto,SOL}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_indiv{proto,time}, mean(data_indiv{proto,SOL}(VER_notDelay,:),1), "LineWidth",1, "color","black", "LineStyle","--")
        %plot(data_indiv{proto,time}, mean(data_indiv{proto,SOL}(VER_notDelay2,:),1), "LineWidth",1, "color","blue")

    subplot(515);hold on; ylabel("Tibialis EMG" + newline + "[\muV]"), xlim(x_range);
        plot(data_indiv{proto,time}, mean(data_indiv{proto,TA}(VER_no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}, mean(data_indiv{proto,TA}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_indiv{proto,time}, mean(data_indiv{proto,TA}(VER_notDelay,:),1), "LineWidth",1, "color","black", "LineStyle","--")
        %plot(data_indiv{proto,time}, mean(data_indiv{proto,TA}(VER_notDelay2,:),1), "LineWidth",1, "color","blue")

    xlabel("Time after estimated perturbation onset [ms]");
end

%%
figure; 
x_range = [-800 800];
hold on; ylabel("Position" + newline + "[Deg]"), xlim(x_range); ylim([-5 11]) %ylim([-5 33])
        plot(data_align{proto,time}*10^3, mean(data_align{proto,ANG}(VER_no,:),1), "LineWidth",3, "color",[0.7, 0.7, 0.7])
        plot(data_align{proto,time}*10^3, mean(data_align{proto,ANG}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_align{proto,time}*10^3, mean(data_align{proto,ANG}(VER_notDelay1,:),1), "LineWidth",1, "color","red")
        plot(data_align{proto,time}*10^3, mean(data_align{proto,ANG}(VER_notDelay2,:),1), "LineWidth",1, "color","blue")


    subplot(511);hold on; ylabel("Position" + newline + "[Deg]"), xlim(x_range); ylim([-5 11]) %ylim([-5 33])
            patch(x,y,patchcolor,'FaceAlpha',FaceAlpha, 'EdgeColor', "none")
        plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(VER_no,:),1), "LineWidth",3, "color",[0.7, 0.7, 0.7])
        plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(VER_notDelay1,:),1), "LineWidth",1, "color","red")
        plot(data_indiv{proto,time}, mean(data_indiv{proto,ANG}(VER_notDelay2,:),1), "LineWidth",1, "color","blue")



%% INDIV test
idx1 = [1,2,12,13,14,17,19]; 
idx2 = [3,4,5,6,7,8,9,10,11]; 
if true
    figure; 
    proto = VER; 
        if proto == HOR;  no = HOR_no; yes = HOR_yes; sgtitle("Horizontal Pertubation - isolating - subject " + SubjectName); end
        if proto == VER;  no = VER_no; yes = VER_yes; sgtitle("Vertical Pertubation, subject 1"); end
    
    % ([1,2, 12, 13, 14, 17,19])
    x_range = [-500 800]; 
    subplot(511);hold on; ylabel("Position" + newline + "[Deg]"), xlim(x_range)
        plot(data_align{proto,time}*10^3, mean(data_align{proto,ANG}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_align{proto,time}*10^3, mean(data_align{proto,ANG}(yes(idx1),:),1), "LineWidth",1, "color","black")
        plot(data_align{proto,time}*10^3, mean(data_align{proto,ANG}(yes(idx2),:),1), "LineWidth",1, "color","red")
        legend(["Control (n=39) ", "Perturbation type 1 (n=7)", "Perturbation type 2 (n=9)"])


     subplot(512);hold on; ylabel("Velocity" + newline + "[Deg/ms]"), xlim(x_range)
        plot(data_align{proto,time}(1:end-1)*10^3, mean(velocity_align{proto}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_align{proto,time}(1:end-1)*10^3, mean(velocity_align{proto}(yes(idx1),:),1), "LineWidth",1, "color","black")
        plot(data_align{proto,time}(1:end-1)*10^3, mean(velocity_align{proto}(yes(idx2),:),1), "LineWidth",1, "color","red")

     subplot(513);hold on; ylabel("Aceleration" + newline + "[Deg/ms^2]"), xlim(x_range)
        plot(data_align{proto,time}(3:end)*10^3, mean(aceleration_align{proto}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_align{proto,time}(3:end)*10^3, mean(aceleration_align{proto}(yes(idx1),:),1), "LineWidth",1, "color","black")
        plot(data_align{proto,time}(3:end)*10^3, mean(aceleration_align{proto}(yes(idx2),:),1), "LineWidth",1, "color","red")

     subplot(514);hold on; ylabel("Soleus EMG" + newline + "[\muV]"), xlim(x_range)
        plot(data_align{proto,time}*10^3, mean(data_align{proto,SOL}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_align{proto,time}*10^3, mean(data_align{proto,SOL}(yes(idx1),:),1), "LineWidth",1, "color","black")
        plot(data_align{proto,time}*10^3, mean(data_align{proto,SOL}(yes(idx2),:),1), "LineWidth",1, "color","red")

    subplot(515);hold on; ylabel("Tibialis EMG" + newline + "[\muV]"), xlim(x_range); xlabel("Time [ms]");
        plot(data_align{proto,time}*10^3, mean(data_align{proto,TA}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_align{proto,time}*10^3, mean(data_align{proto,TA}(yes(idx1),:),1), "LineWidth",1, "color","black")
        plot(data_align{proto,time}*10^3, mean(data_align{proto,TA}(yes(idx2),:),1), "LineWidth",1, "color","red")
        xlabel("Time after foot-strike of fourth step [ms]" + newline + "(detected by FSR sensor)");
end



%% APPDESIGNER 
% APPDESIGNER
%save('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_thomas_ver_test','offset')

%save('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_thomas_ver','offset')
%save('C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/offset/offset_thomas_hor','offset')


%% Control plotting: Avg. Control sweep, (control with variance)
for i = 1:size(data{CTL,SOL},2)
    solSTD = std(data{CTL,SOL}(:,i)) ;
    solMean = mean(data{CTL,SOL}(:,i));
    solUpper(i) = solMean + solSTD; 
    solLower(i) = solMean - solSTD; 

    angSTD = std(data{CTL,ANG}(:,i));
    angMean = mean(data{CTL,ANG}(:,i));
    angUpper(i) = angMean + angSTD; 
    angLower(i) = angMean - angSTD; 

    taSTD = std(data{CTL,TA}(:,i));
    taMean = mean(data{CTL,TA}(:,i));
    taUpper(i) = taMean + taSTD; 
    taLower(i) = taMean - taSTD; 
end 

x = linspace(-4, 6-dt, N); 
X=[x,fliplr(x)];                %#create continuous x value array for plotting
YSol=[solUpper,fliplr(solLower)];              %#create y values for out and then back
YTa=[taUpper,fliplr(taLower)];              %#create y values for out and then back
YAng=[angUpper,fliplr(angLower)];              %#create y values for out and then back

if true
    figure;  sgtitle("Avg. Control sweep, Single subject (n = 20)")
    subplot(311); hold on; title("Left Soleus"); ylabel("[\muV]")
    fill(X,YSol,  [0.7,0.7,0.7], "EdgeColor","none");                  %#plot filled area
    plot(x, mean(data{CTL,SOL},1), "LineWidth",1, "color","black")
    yyaxis right; ylabel("Phase"); ylim([-0.1 1.1])
    plot(x, rescale(mean(data{CTL,FSR},1)), "color",	"yellow");
    text(6.3,1,"[Stand]",'FontSize',5)
    text(6.3,0,"[Swing]",'FontSize',5)
    
    subplot(312); hold on; title("Left Tibialis"); ylabel("[\muV]"); ylim([-80 500])
    fill(X,YTa,  [0.7,0.7,0.7], "EdgeColor","none");                  %#plot filled area
    plot(x, mean(data{CTL,TA},1), "LineWidth",1, "color","black")
    yyaxis right; ylabel("Phase"); ylim([-0.1 1.1])
    plot(x, rescale(mean(data{CTL,FSR},1)), "color",	"yellow")
    text(6.3,1,"[Stand]",'FontSize',5)
    text(6.3,0,"[Swing]",'FontSize',5)
    
    subplot(313); hold on; title("Left Ankle Position"); ylabel("[Deg]"); xlabel("Time [s]")
    fill(X,YAng,  [0.7,0.7,0.7], "EdgeColor", [0.65,0.65,0.65]);                  %#plot filled area
    plot(x, mean(data{CTL,ANG},1), "LineWidth",1, "color","black")
    yyaxis right; ylabel("Phase"); ylim([-0.1 1.1])
    plot(x, rescale(mean(data{CTL,FSR},1)), "color",	"yellow")
    
    text(6.3,1,"[Stand]",'FontSize',5)
    text(6.3,0,"[Swing]",'FontSize',5)
end 


%% Aligned with fourth step (control with variance)
taUpper = []; taLower = [];
solLower = []; solUpper = [];
angUpper = []; angLower = [];

for i = 1:size(data_align{CTL,SOL},2)
    solSTD = std(data_align{CTL,SOL}(:,i)) ;
    solMean = mean(data_align{CTL,SOL}(:,i));
    solUpper(i) = solMean + solSTD; 
    solLower(i) = solMean - solSTD; 

    angSTD = std(data_align{CTL,ANG}(:,i));
    angMean = mean(data_align{CTL,ANG}(:,i));
    angUpper(i) = angMean + angSTD; 
    angLower(i) = angMean - angSTD; 

    taSTD = std(data_align{CTL,TA}(:,i));
    taMean = mean(data_align{CTL,TA}(:,i));
    taUpper(i) = taMean + taSTD; 
    taLower(i) = taMean - taSTD; 
end 

x = data_align{CTL,time}; 

X=[x,fliplr(x)];                %#create continuous x value array for plotting
YSol=[solUpper,fliplr(solLower)];              %#create y values for out and then back
YTa=[taUpper,fliplr(taLower)];              %#create y values for out and then back
YAng=[angUpper,fliplr(angLower)];              %#create y values for out and then back

if true
    figure;  sgtitle("Avg. Control sweep, Single subject (n = 20)")
    subplot(311); hold on;  ylabel("Left Soleus EMG" + newline + "[\muV]")
    fill(X,YSol,  [0.7,0.7,0.7], "EdgeColor","none");                  %#plot filled area
    plot(x, mean(data_align{CTL,SOL},1), "LineWidth",1, "color","black")
    yyaxis right; ylabel("Phase"); ylim([-0.1 1.1])
    plot(x, rescale(mean(data_align{CTL,FSR},1)), "color",	"yellow");
    text(2.1,1,"[Stand]",'FontSize',5)
    text(2.1,0,"[Swing]",'FontSize',5)
    
    subplot(312); hold on;  ylabel("Left Tibialis EMG" + newline + "[\muV]"); ylim([-80 500])
    fill(X,YTa,  [0.7,0.7,0.7], "EdgeColor","none");                  %#plot filled area
    plot(x, mean(data_align{CTL,TA},1), "LineWidth",1, "color","black")
    yyaxis right; ylabel("Phase"); ylim([-0.1 1.1])
    plot(x, rescale(mean(data_align{CTL,FSR},1)), "color",	"yellow")
    text(2.1,1,"[Stand]",'FontSize',5)
    text(2.1,0,"[Swing]",'FontSize',5)
    
    subplot(313); hold on;  ylabel("Left Ankle Position"+ newline +"[Deg]"); xlabel("Time [s]")
    fill(X,YAng,  [0.7,0.7,0.7], "EdgeColor", [0.9,0.9,0.9]);                  %#plot filled area
    plot(x, mean(data_align{CTL,ANG},1), "LineWidth",1, "color","black")
    yyaxis right; ylabel("Phase"); ylim([-0.1 1.1])
    plot(x, rescale(mean(data_align{CTL,FSR},1)), "color",	"yellow")
    text(2.1,1,"[Stand]",'FontSize',5)
    text(2.1,0,"[Swing]",'FontSize',5)
end 








%%

if true
    figure; 
    proto = VER; 
        if proto == HOR;  no = HOR_no; yes = HOR_yes; sgtitle("Horizontal Pertubation - subject " + SubjectName); end
        if proto == VER;  no = VER_no; yes = VER_yes; sgtitle(""); end

    offset = 0; % [ms]
    x_range = [-200 200]; 
    offset_delay = 0; 
    subplot(311);hold on; ylabel("Position"+newline+"[Deg]"), xlim(x_range)
        plot(data_indiv{proto,time}-50, mean(data_indiv{proto,ANG}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}-offset, mean(data_indiv{proto,ANG}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_indiv{proto,time}-offset-offset_delay, mean(data_indiv{proto,ANG}(VER_notDelay,:),1), "LineWidth",1, "color","black", "LineStyle","--")
        legend("Vertical Control (n=39)", "Vertical perturbation, type 1 (n=7)", "Vertical perturbation, type 2 (n=13)")

     subplot(312);hold on; ylabel("Velocity"+newline+"[Deg/ms]"), xlim(x_range)
        plot(data_indiv{proto,time}(1:end-1)-50, mean(velocity_indiv{proto}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}(1:end-1)-offset, mean(velocity_indiv{proto}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_indiv{proto,time}(1:end-1)-offset-offset_delay, mean(velocity_indiv{proto}(VER_notDelay,:),1), "LineWidth",1, "color","black","LineStyle","--")

     subplot(313);hold on; ylabel("Soleus"+newline+"[\muV]"), xlim(x_range)
        plot(data_indiv{proto,time}-50, mean(data_indiv{proto,SOL}(no,:),1), "LineWidth",3, "color",[0.75, 0.75, 0.75])
        plot(data_indiv{proto,time}-offset, mean(data_indiv{proto,SOL}(VER_delay,:),1), "LineWidth",1, "color","black")
        plot(data_indiv{proto,time}-offset, mean(data_indiv{proto,SOL}(VER_notDelay,:),1), "LineWidth",1, "color","black","LineStyle","--")
end

