function [yes, no, array] = sort_sweeps(N, perturbation_arr,  exclude_arr)
    
yes = zeros(1,N); 
no = ones(1,N); 

yes(perturbation_arr) = 1;  % mark perturbation sweeps as One
yes(exclude_arr) = [];      % remove failed sweeps
no(exclude_arr) = [];       % remove failed sweeps
array = yes;                % array were 1 indicate yes and 0 indicate no
yes = find(yes == 1);       % index positions
no(yes) = 0;                % mark all control sweeps as One
no = find(no == 1); 

% % test code
% k  = mean(data{HOR,SOL}(HOR_yes,:),1);
% k1 = mean(test{HOR,SOL}(HOR_perturbation(2:end),:),1);
% 
% k(1:10)
% k1(1:10)