function [EMG_norm, factor] = func_normalize_EMG(step_index, data, varargin)
% MYFUNCTION - This function returns the EMG signal normalized to specified
% step
%
% EMG_norm = func_normalize_EMG(step_index, data, 'normalize_to_step')
% EMG_norm = func_normalize_EMG(step_index, data, 'span')
% EMG_norm = func_normalize_EMG(step_index, data, 'protocols')

    % Abbreviation
    CTL = 1; VER = 2; HOR = 3; CTL2 = 4; % protocols 
    SOL = 1; TA = 2; ANG = 3; FSR = 4; time = 5; VEL = 6; ACC = 7; % sensor types

    % Default values and validiation 
    defaultStep = 0;
    defaultSpan = 15; 
    expectedStepsObtions = [0,2,4,6];
    expectedProtocolObtions = [CTL, VER, HOR, CTL2]; 
    ecpectedProtocol = CTL;
    
    validationSpanFcn = @(x) assert(numel(x) <= 1, 'Assertion failed: x contains more than 1 element.');
    validationStepFcn = @(x) assert(ismember(x, expectedStepsObtions), 'Assertion failed.');
    validationProtoFcn = @(x) assert(all(ismember(x, expectedProtocolObtions)), 'Error: Invalid input options.');

    % Variable input argument list
    p = inputParser;
    addParameter(p, 'normalize_to_step',defaultStep, validationStepFcn); 
    addParameter(p, 'span', defaultSpan, validationSpanFcn)
    addParameter(p, 'protocols',expectedProtocolObtions, validationProtoFcn)
    parse(p,varargin{:})

    % Find rise of FSR sensor to specified step
    [rise, fall] = func_find_edge(p.Results.normalize_to_step); 
   
    for proto = p.Results.protocols % loop through protocols
        clear SOL_max TA_max
        for i = 1:size(data{proto,1},1) % loop through sweeps      
            if p.Results.normalize_to_step == 0
                % groundcontact : fall 
                index_array = 6000 : step_index{proto}((i),fall); 
            else 
                % rise + (standsize / 2) : fall 
                index_array = step_index{proto}((i),rise) + ceil((step_index{proto}((i),fall)-step_index{proto}((i),rise))/2)  : step_index{proto}((i),fall); 
            end 
            SOL_max(i) = max(smooth(data{proto,SOL}((i), index_array),p.Results.span, 'moving'));
            TA_max(i)  = max(smooth(data{proto,TA} ((i), index_array),p.Results.span, 'moving'));
        end
    
        % Divide every element with found average
        data{proto,SOL} = data{proto,SOL} / mean(SOL_max);
        data{proto,TA} = data{proto,TA} / mean(TA_max);

        if proto == CTL             
            factor(SOL) = mean(SOL_max);
            factor(TA) = mean(TA_max);
        end 
    end 
    EMG_norm = data; 
end 
