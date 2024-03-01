clear 
clc
close all
% 
% x = [-18, -15, -10, -5, 0, 5, 10, 19]
% y = [ 5.2,  4.7, 4.5, 3.6, 3.4, 3.1, 2.7, 1.8]
% 
% mdl = fitlm(x', y')
% tbl = anova(mdl)


%% Levene's test
% Kommer samples fra populationer ned ens variance ?
% if sig > 0.05 kommer fra x grupper med samme variance 

load carsmall


vartestn(MPG,Model_Year,'TestType','LeveneAbsolute')
[p,tbl,stats] = anova1(MPG,Model_Year);

% population variances are not equal if “Sig.” or p < 0.05.


%% One way anova

% create some data for the plot
x = 1:10;
y = randn(1,10);

% plot the data
plot(x,y);

% add text below the plot
text(5,-2,'This is some additional information about the data','HorizontalAlignment','center','FontSize',12);



