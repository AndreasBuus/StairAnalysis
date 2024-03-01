function [SOL_align, TA_align, angle_align, FSR_align, time_align] = align_data_vol2(step_index, SOL, TA, angle, FSR, varargin)

    defaultAlign = 'four_begin';
    expectedAlignObtions = {'four_begin','second_end', 'second_begin'};
    defaultBefore = 0;
    defaultAfter = 0;
    defaultFs = 2000; 

    p = inputParser;
    %addParameter(p,'mode',defaultMode, @(x) any(validatestring(x,expectedModes)))    
    addParameter(p,'sec_before',defaultBefore)
    addParameter(p,'sec_after',defaultAfter)
    addParameter(p,'samplingFrequence',defaultFs)
    addParameter(p,'alignStep',defaultAlign, @(x) any(validatestring(x,expectedAlignObtions))); 


    parse(p,varargin{:})
    
    Fs = p.Results.samplingFrequence;  % [sample/sec]
    dt = 1/Fs;                         % [sec/samples]

    samplesBefore = floor(p.Results.sec_before * Fs);  % [sample]
    samplesAfter  = floor(p.Results.sec_after * Fs);   % [sample]

    if strcmp(p.Results.alignStep, 'four_begin')
        % (sweep,4) -  Fouth step, falling  
        % (sweep,5) -  Fouth step, rising  
        for i = 1:size(FSR,1)
            fourth_stand_DUR(i) = step_index(i, 4) - step_index(i, 5); % [array(:)]
        end 
   
        for i = 1:size(FSR,1) 
            array1   = step_index(i,5)-samplesBefore; % [index]
            arrayEnd = (step_index(i,5) + floor(mean(fourth_stand_DUR))) + samplesAfter; % [index]
    
            fouth_stand_arr = [array1:arrayEnd]; % [array]
            SOL_align(i,:)  = SOL(i, fouth_stand_arr); 
            TA_align(i,:)   = TA(i, fouth_stand_arr); 
            FSR_align(i,:)  = FSR(i, fouth_stand_arr);  
            angle_align(i,:) = angle(i, fouth_stand_arr); 
        end
        time_align = linspace(-p.Results.sec_before, (floor(mean(fourth_stand_DUR)) + samplesAfter)*dt, length(fouth_stand_arr));
    end


    if strcmp(p.Results.alignStep, 'second_begin')
        % (sweep,6) -  Second step, falling  
        % (sweep,7) -  Second step, rising  
        for i = 1:size(FSR,1)
            fourth_stand_DUR(i) = step_index(i, 6) - step_index(i, 7); % [array(:)]
        end 
    

        for i = 1:size(FSR,1) 
            array1   = step_index(i,7)-samplesBefore; % [index]
            arrayEnd = (step_index(i,7) + floor(mean(fourth_stand_DUR))) + samplesAfter; % [index]
    
            fouth_stand_arr = [array1:arrayEnd]; % [array]
            SOL_align(i,:)  = SOL(i, fouth_stand_arr); 
            TA_align(i,:)   = TA(i, fouth_stand_arr); 
            FSR_align(i,:)  = FSR(i, fouth_stand_arr);  
            angle_align(i,:) = angle(i, fouth_stand_arr); 
        end
        time_align = linspace(-p.Results.sec_before, (floor(mean(fourth_stand_DUR)) + samplesAfter)*dt, length(fouth_stand_arr));
    end

    if strcmp(p.Results.alignStep, 'second_end')
        % (sweep,6) -  Second step, falling  <----
        % (sweep,7) -  Second step, rising  
     
        for i = 1:size(FSR,1) 
            array1   = step_index(i,6)-samplesBefore; % [index]
            arrayEnd = step_index(i,6) + 600 + samplesAfter; % [index]
    
            fouth_stand_arr = [array1:arrayEnd]; % [array]
            SOL_align(i,:)  = SOL(i, fouth_stand_arr); 
            TA_align(i,:)   = TA(i, fouth_stand_arr); 
            FSR_align(i,:)  = FSR(i, fouth_stand_arr);  
            angle_align(i,:) = angle(i, fouth_stand_arr); 
        end
        time_align = linspace(-p.Results.sec_before, (600 + samplesAfter)*dt, length(fouth_stand_arr));
    end


% -- var "Step_index" 
% Index position of change in FSR record, 
% (sweep,1) -  Seventh  step, rising 
% (sweep,2) -  Sixth step, falling
% (sweep,3) -  Sixth step, rising
% (sweep,4) -  Fouth step, falling  <--  
% (sweep,5) -  Fouth step, rising  <-- 
% (sweep,6) -  Second step, falling  



end 