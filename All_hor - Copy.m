clc 
clear all; 
close all; 

%% Folders 
folderpath_10sem = "C:/Users/BuusA/OneDrive - Aalborg Universitet/10. semester (Kandidat)/Matlab files/Indiv_data/"; 

load(folderpath_10sem + "Thomas.mat");  Thomas = data_indiv;                TotalData{:,:,1} = data_indiv; 
load(folderpath_10sem + "Thomas_Type.mat");  Thomas_type = type;            TotalType{:,:,1} = type; 
load(folderpath_10sem + "Andreas.mat"); Andreas = data_indiv;               TotalData{:,:,2} = data_indiv; 
load(folderpath_10sem + "Andreas_Type.mat");  Andreas_type = type;          TotalType{:,:,2} = type; 
load(folderpath_10sem + "Maria.mat"); Maria = data_indiv;                   TotalData{:,:,3} = data_indiv; 
load(folderpath_10sem + "Maria_Type.mat");  Maria_type = type;              TotalType{:,:,3} = type; 
load(folderpath_10sem + "Benedikte.mat"); Benedikte = data_indiv;           TotalData{:,:,4} = data_indiv; 
load(folderpath_10sem + "Benedikte_Type.mat");  Benedikte_type = type;      TotalType{:,:,4} = type; 
load(folderpath_10sem + "Trine2.mat"); Trine2 = data_indiv;                 TotalData{:,:,5} = data_indiv; 
load(folderpath_10sem + "Trine2_Type.mat");  Trine2_type = type;            TotalType{:,:,5} = type; 
load(folderpath_10sem + "Gritt.mat"); Gritt = data_indiv;                   TotalData{:,:,6} = data_indiv; 
load(folderpath_10sem + "Gritt_Type.mat");  Gritt_type = type;              TotalType{:,:,6} = type; 
load(folderpath_10sem + "Mia.mat"); Mia = data_indiv;                       TotalData{:,:,7} = data_indiv; 
load(folderpath_10sem + "Mia_Type.mat");  Mia_type = type;                  TotalType{:,:,7} = type; 
load(folderpath_10sem + "Trine.mat"); Trine = data_indiv;                   TotalData{:,:,8} = data_indiv; 
load(folderpath_10sem + "Trine_Type.mat");  Trine_type = type;              TotalType{:,:,8} = type; 
load(folderpath_10sem + "Andrew.mat"); Andrew = data_indiv;                 TotalData{:,:,9} = data_indiv; 
load(folderpath_10sem + "Andrew_Type.mat");  Andrew_type = type;            TotalType{:,:,9} = type; 

subject{1,1} = "Thomas";    subject{1,2} = "Thomas_type"; 
subject{2,1} = "Andreas";   subject{2,2} = "Andreas_type"; 
subject{3,1} = "Beneditke"; subject{3,2} = "Beneditke_type"; %
subject{4,1} = "Maria";     subject{4,2} = "Maria_type"; 
subject{5,1} = "Trine2";    subject{5,2} = "Trine2_type"; 
subject{6,1} = "Griit";     subject{6,2} = "Gritt_Type.mat";
subject{7,1} = "Mia";       subject{7,2} = "Mia_Type.mat"; % probabaly exclude
subject{8,1} = "Trine";     subject{8,2} = "Trine_Type.mat"; % probala
subject{9,1} = "Andrew";     subject{9,2} = "Andrew_Type.mat"; % probala

NSub = size(subject,1); 

%type{1} = VER_yes; type{2} = VER_no; type{3} = HOR_yes; type{4} = HOR_no; 
clear data_indiv type folderpath_10sem
%% Abbreviation
    CTL = 1; VER = 2; HOR = 3; CTL2 = 4; time = 5;
    SOL = 1; TA = 2; ANG = 3; FSR = 4; FOO = 5; 
    VEL = 6; ACC = 7; test = 8; SOL_diff = 9; SOL_diff_no_delay = 10; TA_diff = 11;
    ProtoAll = [CTL, VER, HOR, CTL2];

%% Functions
    msToSec = @(x) x*10^-3; % ms to sec 
    secToMs = @(x) x*10^3; % ms to sec 

%% Acquisition Set-Up
    sweep_length = 10;              % Signal length in second
    Fs = 2000;                      % Samples per second
    dt = 1/Fs;                      % Seconds per sample
    pre_trig = 4;                   % Pre-trigger 
    N = Fs*sweep_length;            % Total number of samples per signal

%% Speed and aceleration
span = 10; 
for sub = 1:NSub 
    for PROTO = [HOR]
        data = TotalData{1,1,sub}; 
        type = TotalType{1,1,sub};
        for i = 1:size(data{PROTO,1},1) %sweeps
            data{PROTO,VEL}(i,:) = diff(data{PROTO,ANG}(i,:))/(dt*10^3);     % [deg/ms]
            data{PROTO,ACC}(i,:) = diff(smooth(data{PROTO,VEL}(i,:), span, 'moving'))/(dt*10^3);     % [deg/ms^2]
        end
        TotalData{1,1,sub} = data; 
        TotalType{1,1,sub} = type;
    end 
end

%% Plot
SLR_x = [39 59 59 39];
MLR_x = [60 80 80 60];
LLR_x = [81 100 100 81];

proto = HOR; 
x_range = [-800 800]; 
%subjecsInc = [1,2,3,4,5,6,7,8];
subjecsInc = [8]


if true    
    figure;
    if proto == HOR;  no = type{4}; yes = type{3}; sgtitle(""); end
    % set(gcf,'Position',[100 100 1400 500])
    offset = 0%-40;
    graaColor = [0.7, 0.7, 0.7]; 
    x = [0 650 650 0];
    y = [-10 -10 10 10];
    
    addedOffset = 0; 

    
    subplot(411);hold on; ylabel("Normalized"+newline+"Soleus"), xlim(x_range)        
        for sub = subjecsInc
            data = TotalData{1,1,sub}; 
            type = TotalType{1,1,sub}; no = type{4}; yes = type{3};
            yes_mean = mean((data{proto,SOL}(yes,:)),1)+addedOffset; 
            no_mean  = mean((data{proto,SOL}(no,:)),1)+addedOffset; 


            % Find Difference for each YES sweeps
            for i = 1:numel(yes)
                signal_diff_s = 100*((data{proto,SOL}(yes(i),:)+addedOffset)-no_mean)./no_mean; 
                TF = isinf(signal_diff_s); 
                signal_diff_s(TF) = 0; 
                data{PROTO,SOL_diff}(i,:) = signal_diff_s; 
            end 
            TotalData{1,1,sub} = data; % Save data 

            plot(data{proto,time}+offset, no_mean, "color", [0.7, 0.7, 0.7], "LineWidth",3)
            plot(data{proto,time}+offset, yes_mean, "color", "black")
            %plot(data{proto,time}, rescale((yes_mean-no_mean)./no_mean), "color", "red", "LineWidth",1)
            %plot(data{proto,time}, rescale(mean(data{PROTO,SOL_diff}(:,:),1)), "color", "blue", "LineWidth",1)
        end 
    
    subplot(412);hold on; ylabel("Normalized"+newline+"Tibialis"), xlim(x_range), ylim([0 1.1])
            patch(x,y,"red",'FaceAlpha',.2, 'EdgeColor', "none")

%         y = [0 0 1 1];
%         patch(SLR_x,y,[0.75, 0.75, 0.75], 'EdgeColor', 'none','FaceAlpha',0.3)
%         patch(MLR_x,y,[0.60, 0.60, 0.60], 'EdgeColor', 'none','FaceAlpha',0.3)
%         patch(LLR_x,y,[0.75, 0.75, 0.75], 'EdgeColor', 'none','FaceAlpha',0.3)
        
        for sub = subjecsInc
            data = TotalData{1,1,sub}; 
            type = TotalType{1,1,sub}; no = type{4}; yes = type{3};
            yes_mean = mean(data{proto,TA}(yes,:),1); 
            no_mean  = mean(data{proto,TA}(no,:),1); 


            % Find Difference for each YES sweeps
            for i = 1:numel(yes)
                signal_diff_s = (data{proto,TA}(yes(i),:)-no_mean)./no_mean; 
                TF = isinf(signal_diff_s); 
                signal_diff_s(TF) = 0; 
                data{PROTO,TA_diff}(i,:) = signal_diff_s; 
            end 
            TotalData{1,1,sub} = data; % Save data 

            plot(data{proto,time}+offset, rescale(no_mean), "color", [0.7, 0.7, 0.7], "LineWidth",3)
            plot(data{proto,time}+offset, rescale(yes_mean), "color", "black")
            %plot(data{proto,time}, rescale((yes_mean-no_mean)./no_mean), "color", "red", "LineWidth",1)
            %plot(data{proto,time}, rescale(mean(data{PROTO,SOL_diff}(:,:),1)), "color", "blue", "LineWidth",1)
        end 


    subplot(413);hold on; ylabel(""),ylabel("Position"+newline+"[Deg]"), xlim(x_range), ylim([-2 10])
        patch(x,y,"red",'FaceAlpha',.2, 'EdgeColor', "none")

        for sub = subjecsInc
            data = TotalData{1,1,sub}; 
            type = TotalType{1,1,sub}; no = type{4}; yes = type{3};

            yes_mean = mean(data{proto,ANG}(yes,:),1); 
            no_mean  = mean(data{proto,ANG}(no,:),1); 
            
            plot(data{proto,time}+offset, no_mean, "color", [0.7, 0.7, 0.7], "LineWidth",3)
            plot(data{proto,time}+offset, yes_mean, "color", "black")
        end 

     subplot(414);hold on; ylabel("Velocity"+newline+"[Deg/ms]"), xlim(x_range), ylim([-0.08 0.08])
       patch(x,y,"red",'FaceAlpha',.2, 'EdgeColor', "none")

        for sub = subjecsInc
            data = TotalData{1,1,sub}; 
            type = TotalType{1,1,sub}; no = type{4}; yes = type{3};

            yes_mean = mean(data{PROTO,VEL}(yes,:),1); 
            no_mean  = mean(data{PROTO,VEL}(no,:),1); 
            
            plot(data{proto,time}(1,1:end-1)+offset, no_mean, "color", [0.7, 0.7, 0.7], "LineWidth",3)
            plot(data{proto,time}(1,1:end-1)+offset, yes_mean, "color", "black")
           
        end 
    xlabel("Timer after estimated perturbation [ms]")
end

%% BOKS plot 

zero_idx = 2001; 
SLR = [39,59]; % ms
MLR = [60 80]; % ms

offsets = [17.5,16.5,31.5,32.5,35.5,25.0,25.0,25.0]; %ORIGINAL
subjecsInc = [1,2,3,4,5,8,9]; %ORIGINAL!

%offsets = [17.5, 31.5, 32.5, 35.5]; POWERPUFF
%subjecsInc = [1,3,4,5]; <-- POWERPUFF!

% offsets =    [21,35,20,30,20,29];
% subjecsInc = [ 1, 2, 3, 4, 5,8];

for i = 1:numel(subjecsInc)
    data = TotalData{1,1,subjecsInc(i)}; 
    type = TotalType{1,1,subjecsInc(i)}; no = type{4}; yes = type{3};

    zero_aligned_idx = zero_idx + (offsets(i)*Fs*10^-3); 
    SLR_idx = [SLR(1)*Fs*10^-3+zero_aligned_idx, SLR(2)*Fs*10^-3+zero_aligned_idx]; 
    MLR_idx = [MLR(1)*Fs*10^-3+zero_aligned_idx, MLR(2)*Fs*10^-3+zero_aligned_idx];
    
    
    kage(i,1) = mean(data{HOR,SOL}(yes,SLR_idx(1):SLR_idx(2)), "all");
    kage(i,2) = mean(data{HOR,SOL}(no, SLR_idx(1):SLR_idx(2)), "all");
    
    kage(i,3) = mean(data{HOR,SOL}(yes,MLR_idx(1):MLR_idx(2)), "all");
    kage(i,4) = mean(data{HOR,SOL}(no, MLR_idx(1):MLR_idx(2)), "all");

    kage(i,5) = mean(data{HOR,SOL}(yes,SLR_idx(1):MLR_idx(2)), "all");
    kage(i,6) = mean(data{HOR,SOL}(no, SLR_idx(1):MLR_idx(2)), "all");

    maage(i,1) = mean(data{HOR,TA}(yes,SLR_idx(1):SLR_idx(2)), "all");
    maage(i,2) = mean(data{HOR,TA}(no, SLR_idx(1):SLR_idx(2)), "all");
    
    maage(i,3) = mean(data{HOR,TA}(yes,MLR_idx(1):MLR_idx(2)), "all");
    maage(i,4) = mean(data{HOR,TA}(no, MLR_idx(1):MLR_idx(2)), "all");

    maage(i,5) = mean(data{HOR,TA}(yes,SLR_idx(1):MLR_idx(2)), "all");
    maage(i,6) = mean(data{HOR,TA}(no, SLR_idx(1):MLR_idx(2)), "all");

end

%[p,h] = signrank(kage(:,1),kage(:,2)) % not significant. SLR SOL
[p,h] = signrank(kage(:,3),kage(:,4)) % not significant, MLR, SOL
%[p,h] = signrank(kage(:,5),kage(:,6)) % not significant, SLR+MLR SOL

%[p,h] = signrank(maage(:,1),maage(:,2)) % not significant. SLR SOL
%[p,h] = signrank(maage(:,3),maage(:,4)) % not significant, MLR, SOL
%[p,h] = signrank(maage(:,5),maage(:,6)) % not significant, SLR+MLR SOL


%%
HOR8 = ["Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation"]; 
CTL8 = ["Horizontal Control";"Horizontal Control";"Horizontal Control";"Horizontal Control";"Horizontal Control";"Horizontal Control";"Horizontal Control";"Horizontal Control"]; 
base8 = ["baseline";"baseline";"baseline";"baseline";"baseline";"baseline";"baseline";"baseline"];

SLR8 = ["SLR";"SLR";"SLR";"SLR";"SLR";"SLR";"SLR";"SLR"];
MLR8 = ["MLR";"MLR";"MLR";"MLR";"MLR";"MLR";"MLR";"MLR"]; 
SLRMLR8 = ["SLR+MLR";"SLR+MLR";"SLR+MLR";"SLR+MLR";"SLR+MLR";"SLR+MLR";"SLR+MLR";"SLR+MLR"];

HOR4=HOR8(1:4); 
CTL4=CTL8(1:4);
base4=base8(1:4);
SLR4 = SLR8(1:4);
MLR4= MLR8(1:4);
SLRMLR4 = SLRMLR8(1:4);

HOR7 = HOR8(1:7); 
CTL7 = CTL8(1:7);
base7= base8(1:7);
SLR7 = SLR8(1:7);
MLR7 = MLR8(1:7);
SLRMLR7 = SLRMLR8(1:7);


HOR6 = HOR8(1:6); 
CTL6 = CTL8(1:6);
base6= base8(1:6);
SLR6 = SLR8(1:6);
MLR6 = MLR8(1:6);
SLRMLR6 = SLRMLR8(1:6);


% type = [SLR4; SLR4; MLR4; MLR4; SLRMLR4; SLRMLR4];
% protocol = [HOR4; CTL4;HOR4; CTL4;HOR4; CTL4];

% type = [SLR8; SLR8; MLR8; MLR8; SLRMLR8; SLRMLR8];
% protocol = [HOR8; CTL8; HOR8; CTL8; HOR8; CTL8];

type = [SLR7; SLR7; MLR7; MLR7; SLRMLR7; SLRMLR7];
protocol = [HOR7; CTL7; HOR7; CTL7; HOR7; CTL7];

% type = [SLR6; SLR6; MLR6; MLR6; SLRMLR6; SLRMLR6];
% protocol = [HOR6; CTL6; HOR6; CTL6; HOR6; CTL6];

averagesSOL = [kage(:,1); kage(:,2);kage(:,3);kage(:,4);kage(:,5);kage(:,6)];
averagesTA = [maage(:,1); maage(:,2);maage(:,3);maage(:,4);maage(:,5);maage(:,6)];

kagen = table(type,protocol,averagesSOL,averagesTA);

monthOrder = {'SLR','MLR','SLR+MLR'};
kagen.type = categorical(kagen.type,monthOrder);


figure; sgtitle("All subject (N=8)")

subplot(211)
boxchart(kagen.type, kagen.averagesSOL, 'GroupByColor', kagen.protocol)
ylabel("Average normalized"+newline+"soleus activity")
legend

subplot(212)
boxchart(kagen.type, kagen.averagesTA, 'GroupByColor', kagen.protocol)
ylabel("Average normalized"+newline+"tibialis activity")



%%
% HOR8 = ["Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation";"Horizontal perturbation"]; 
% CTL8 = ["Horizontal Control";"Horizontal Control";"Horizontal Control";"Horizontal Control";"Horizontal Control";"Horizontal Control";"Horizontal Control"]; 
% base8 = ["baseline";"baseline";"baseline";"baseline";"baseline";"baseline";"baseline"];
% 
% SLR8 = ["SLR";"SLR";"SLR";"SLR";"SLR";"SLR";"SLR"];
% MLR8 = ["MLR";"MLR";"MLR";"MLR";"MLR";"MLR";"MLR"]; 
% SLRMLR8 = ["SLR+MLR";"SLR+MLR";"SLR+MLR";"SLR+MLR";"SLR+MLR";"SLR+MLR";"SLR+MLR"];
% 
% type = [SLR8; SLR8; MLR8; MLR8; SLRMLR8; SLRMLR8];
% protocol = [HOR8; CTL8;HOR8; CTL8;HOR8; CTL8];
% averages = [kage(:,1); kage(:,2);kage(:,3);kage(:,4);kage(:,5);kage(:,6)];
% 
% kagen = table(type,protocol,averages);
% 
% monthOrder = {'SLR','MLR','SLR+MLR'};
% kagen.type = categorical(kagen.type,monthOrder);
% 
% 
% figure;
% boxchart(kagen.type, kagen.averages, 'GroupByColor', kagen.protocol)
% ylabel("Average normalized"+newline+"soleus activity")
% legend
% title("All subject (N=7)")

%% Plot all 
%offsets = [17.5,16.5,31.5,32.5,35.5,25.0,25.0,25.0]; 
% offsets = [32.5,25.0]; 
% 
% subjecsInc = [1,2,3,4,5,6,8,9];
% 
% xrange = [-150 150];
% test = 1; 
% figure; 
% for i = 1:numel(subjecsInc)
%     data = TotalData{1,1,subjecsInc(i)}; 
%     type = TotalType{1,1,subjecsInc(i)}; no = type{4}; yes = type{3};
% 
%     subplot(211); hold on; ylabel("Soleus"+newline+"(%back)"); xlim(xrange)
%     if i == test
%         plot(data{HOR,time}-offsets(i), mean(data{HOR,SOL_diff}(:,:),1), "LineWidth",1, "Color","red"); 
%     else 
%         plot(data{HOR,time}-offsets(i), mean(data{HOR,SOL_diff}(:,:),1), "LineWidth",1, "Color","black"); 
%     end 
% 
% 
%     subplot(212); hold on; ylabel("Soleus"+newline+"(-back)"); xlim(xrange)    
%     
%      if i == test
%         plot(data{HOR,time}-offsets(i), mean(data{HOR,SOL}(yes,:),1)-mean(data{HOR,SOL}(no,:),1), "LineWidth",1, "Color","red"); 
%     else 
%         plot(data{HOR,time}-offsets(i), mean(data{HOR,SOL}(yes,:),1)-mean(data{HOR,SOL}(no,:),1), "LineWidth",1, "Color","black"); 
%     end 
% end






%% Linear Regression
% Hvad er soleus aktivitet indenfor 38 ms efter onset til 70 ms efter onset

viewBefore = 1000; % [ms]
viewAfter  = 1000; % [ms]
zero_idx = find(data{proto,time} == 0);     %[sample]

offsets = [17.5,16.5,31.5,32.5,35.5,25.0,25.0,25.0]; 
subjecsInc = [1,3,4,5];

% SOLEUS diff : 44,-5 and 55,9

% Unit [ms] , OBS must align 0.5
delay =      44% 10:1:100;  %5:1:30 %12.5 ; %[10:5:50]; % X X X
offset =     -5% -40:1:30; %27.5; % [-100:5:100]; %  Y Y Y 
depIdxDiff = 30; %[20, 30, 40];
preIdxDiff = 30; %[20, 30, 40];

for off = 1:numel(offset) 
for de = 1:numel(depIdxDiff)
for pr = 1:numel(preIdxDiff)
for del = 1:numel(delay)


testSubject = [4]; 
plt = 1; 
pltAllSubject = 1; 
surfplt = 0; 

    % Predictor
preIdx1 = zero_idx + msToSec(offset(off))*Fs;
preIdx2 = preIdx1 + floor(Fs*msToSec(preIdxDiff(pr)));
preIdx = [preIdx1:preIdx2];

    % Dependent
depIdx1 = zero_idx + msToSec(offset(off))*Fs + msToSec(delay(del))*Fs;
depIdx2 = depIdx1 + floor(Fs*msToSec(depIdxDiff(de)));
depIdx = [depIdx1:depIdx2];

pre = []; dep = [];
pre1 = []; dep1 = []; 
for sub = subjecsInc
    data = TotalData{1,1,sub}; 
    type = TotalType{1,1,sub}; HOR_yes = type{3}; HOR_no = type{4}; 
    for PROTO = [HOR]
        
        data_dep{SOL_diff} = []; 
        for i = 1:size(data{HOR,SOL_diff},1)
            data_dep{SOL_diff}(i) = mean(data{HOR,SOL_diff}(i,depIdx + offsets(sub)*Fs*10^-3)); 
        end 

        data_dep{SOL} = [];
        data_pre{VEL} = [];
        for i = 1:size(data{PROTO,SOL},1) % Sweeps  
            % Dependent Soleus
            %data_dep{SOL}(i) = mean(maxk(data{PROTO,SOL}(i,depIdx),30)); 
            data_dep{SOL}(i) = mean(data{PROTO,SOL}(i,depIdx + offsets(sub)*Fs*10^-3)); 

            % Predictors
            data_pre{VEL}(i) = mean(abs(data{PROTO,VEL}(i,preIdx + offsets(sub)*Fs*10^-3))); 
            %data_pre{VEL}(i) = mean(abs(mink(data{PROTO,VEL}(i,preIdx),20))); 
        end
    end 
    predictor1{sub} = data_pre{VEL}(HOR_yes) - mean(data_pre{VEL}(HOR_yes));   % Velocity, removed mean 
    %predictor1{sub} = rescale(data_pre{VEL}(HOR_yes));                        % Velocity, rescaled
  
    dependent1{sub} = data_dep{SOL_diff}(:) - mean(data_dep{SOL_diff});       % Soleus Diff, removed mean
    %dependent1{sub} = rescale(data_dep{SOL_diff}(:));                          % Soleus Diff, rescaled
    %dependent1{sub} = data_dep{SOL}(HOR_yes) - mean(data_dep{SOL}(HOR_yes)); 
        pre1 = [pre1, predictor1{sub}];  % HERE
        dep1 = [dep1, dependent1{sub}']; % HERE 

    predictor{sub} = data_pre{VEL}(HOR_yes) - mean(data_pre{VEL}(HOR_yes)) ;     
    dependent{sub} = data_dep{SOL}(HOR_yes) - mean(data_dep{SOL}(HOR_yes)) ; 
        pre = [pre, predictor{sub}];
        dep = [dep, dependent{sub}]; 
        
end 


% plot variables (SOL,ANG, VEL)
if plt 
    sub = testSubject; 
    data = TotalData{1,1,sub}; 
    type = TotalType{1,1,sub}; HOR_yes = type{3}; HOR_no = type{4}; 
    figure; 
    sweep = 1; % perturbation

    arrDep = depIdx; 
    arrPre = preIdx; 
     subplot(511); hold on; title("Mean SOL. "+ (depIdx1-zero_idx)*dt*1000 + "[ms] - "+ (depIdx2-zero_idx)*dt*1000+"[ms]") 
        plot(data{proto,time}-offsets(sub), mean(data{HOR,SOL}(HOR_yes(:),:),1), "color", [0.7,0.7,0.7], "LineWidth",2)
        plot(data{proto,time}(1,arrDep)-offsets(sub), mean(data{HOR,SOL}(HOR_yes(:),arrDep),1), "color", "black"); 
    subplot(512); hold on; title("Sweep SOL")
        plot(data{proto,time}+offsets(sub), data{HOR,SOL}(HOR_yes(sweep),:), "color", [0.7,0.7,0.7], "LineWidth",2)
        plot(data{proto,time}(1,arrDep)+offsets(sub), data{HOR,SOL}(HOR_yes(sweep),arrDep), "color", "black"); 
    subplot(513); hold on; title("Sweep SOL diff")
        plot(data{proto,time}+offsets(sub),data{HOR,SOL_diff}(sweep,:), "color", [0.7,0.7,0.7], "LineWidth",2); 
        plot(data{proto,time}(arrDep)+offsets(sub), data{HOR,SOL_diff}(sweep,arrDep), "color", "black"); 
    subplot(514); hold on; title("mean Velocity. "+ (preIdx1-zero_idx)*dt*1000 + "[ms]. - "+ (preIdx2-zero_idx)*dt*1000+"[ms]")
        plot(data{proto,time}(1,2:end)+offsets(sub), mean(data{HOR,VEL}(HOR_yes(:),:),1), "color", [0.7,0.7,0.7], "LineWidth",2); 
        plot(data{proto,time}(arrPre)+offsets(sub), mean(data{HOR,VEL}(HOR_yes(:),arrPre),1), "color", "black"); 
        plot(data{proto,time}(1,2:end)+offsets(sub), data{HOR,VEL}(HOR_yes(sweep),:), "color", "red"); 
    subplot(515); hold on; xlim([0 4001]); title("mean ankle")
        plot(data{proto,time}, mean(data{HOR,ANG}(HOR_no,:),1)+offsets(sub), "color", [0.5,0.5,0.5], "LineWidth",2); 
        plot(data{proto,time}, mean(data{HOR,ANG}(HOR_yes,:),1)+offsets(sub), "color", [0.7,0.7,0.7], "LineWidth",1); 
       % plot(arrPre, mean(data{HOR,ANG}(HOR_yes,arrPre),1), "color", "black"); 
end

sub = testSubject; 
%mdl = fitlm(predictor{sub},dependent{sub}); 
if plt
   % mdl = fitlm(predictor{sub},dependent{sub}) 
   % mdl = fitlm(rescale(predictor{sub}),rescale(dependent{sub}))
   mdl = fitlm(predictor1{sub}, dependent1{sub})
     figure; plot(mdl); ylabel("Soleus"); xlabel("predictor"); title("subject " + sub)
end

% Linear regression, all subject 
if pltAllSubject 
    mdl = fitlm(pre1,dep1) 
    b = table2array(mdl.Coefficients(1,1)); 
    a = table2array(mdl.Coefficients(2,1)); 
    linearReg = @(x) x*a + b; 
    figure; hold on; title("pre1")
    for i = subjecsInc
        plot(predictor1{i}, dependent1{i},'o')
    end
    plot(pre1, linearReg(pre1)); ylabel("Soleus"); xlabel("Predictor")
end

if pltAllSubject 
    mdl = fitlm(pre,dep) 
    b = table2array(mdl.Coefficients(1,1)); 
    a = table2array(mdl.Coefficients(2,1)); 
    linearReg = @(x) x*a + b; 
    figure; hold on; title("pre")
    for i = subjecsInc
        plot(predictor{i}, dependent{i},'o')
    end
    plot(pre, linearReg(pre)); ylabel("Soleus"); xlabel("Predictor")
end


mdl = fitlm(pre, dep);
resultSig(off,del) = table2array(mdl.Coefficients(2,4)); 
resultSize(off,del)= table2array(mdl.Coefficients(2,1));
resultRsquare(off,del)= mdl.Rsquared.Adjusted; 


end; end; end; end

%% SURF PLOT
if surfplt 

    [X,Y] = meshgrid(delay, offset);
    Z = delay + offset'
    [x,y] = find(Z > 36 & Z < 44)
    k = zeros(size(Z))
    for i = 1:numel(x)
        k(x(i),y(i)) = 1;
    end
    
    Zpower = resultSize;
    Zsig = resultSig*(-1); 
    ZRsquare = resultRsquare; 
    Zpos = k; 
    
    figure
        subplot(221); surf(X,Y,Zpower); xlabel("x - delay [ms]"); ylabel("y - offset [ms]"); zlabel("z - power ")
        subplot(222); surf(X,Y,Zsig);   xlabel("x - delay [ms]"); ylabel("y - offset [ms]"); zlabel("z - sig "); zlim([-0.05 0])
        subplot(223); surf(X,Y,ZRsquare);   xlabel("x - delay [ms]"); ylabel("y - offset [ms]"); zlabel("z - Rsquare ")
        subplot(224); surf(X,Y,Zpos);   xlabel("x - delay [ms]"); ylabel("y - offset [ms]"); zlabel("z - pos ")
    
    [X,Y] = meshgrid(delay, offset);
    Z = delay + offset'
    [x,y] = find(or(Z < 35, Z > 40))
    for i = 1:numel(x)
        Zpower(x(i),y(i)) = 0;
        Zsig(x(i),y(i)) = -1;
        ZRsquare(x(i),y(i)) = 0;
    end
    
    figure 
        subplot(221); surf(X,Y,Zpower); xlabel("x - delay [ms]"); ylabel("y - offset [ms]"); zlabel("z - power "); title("Coefficients x1")
        subplot(222); surf(X,Y,Zsig);   xlabel("x - delay [ms]"); ylabel("y - offset [ms]"); zlabel("z - pValue "); zlim([-0.05 0]); title("p Value")
        subplot(223); surf(X,Y,ZRsquare);   xlabel("x - delay [ms]"); ylabel("y - offset [ms]"); zlabel("z - R^2 "); title("Adjusted R-Squared:")
        subplot(224); surf(X,Y,Zpos);   xlabel("x - delay [ms]"); ylabel("y - offset [ms]"); zlabel("z - pos "); title("Variables")
end



