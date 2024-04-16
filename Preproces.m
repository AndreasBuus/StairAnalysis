%% Filtrering and detrend (similar to MR. kick)
fprintf('Script section: Filtrering and detrend (similar to MR. kick) \n'); 

fc = 40;                            % Cutoff frequency for LowPass filter
order = 1;                          % Filter order 
[b,a] = butter(order,fc/(Fs/2));    % Filter coefficient


% rectify and filter EMG. Remove noise in FSR 
for proto = Protocol_All
    [data{proto,SOL}, data{proto,TA}] = rectify_filter(data{proto,SOL}, data{proto,TA}, b, a);  
    [data{proto,FSR}] = func_filt_FSR(data{proto,FSR}, "test", false , "limit_pct", 0.95, "gap_size2", 150); 
end


%% Find correct position for Stand - and Swingphase
fprintf('Script section: Find correct position for Stand - and Swingphase \n'); 

step_index = cell(3,1);  % Index for position 
error_index = cell(3,1);
setup_overall = 8500; 

filename =  "/step_indexs/" + SubjectName + "_step_index.mat";
filepath = fullfile(main_folderpath, filename);

% Check if an already defined offset file exist for subject 
if exist(filepath, 'file') == 2 % Step_index already exist 
    fprintf('\n     A >pre< file already exist and will be loaded'); 
    load(filepath) % return 4x1 cell array called 'offset'
    step_index = offset; 
else 
    fprintf(2,'\n     No >pre< file exist and a automaticly will be generated '); 
    for proto = Protocol_All
        [step_index{proto}, error_index{proto}] = func_step_index(data{proto,FSR}, 'offset', setup_overall);
    end
end


if ~(exist(filepath, 'file') == 2)
    enable_gui = true; 
else 
    prompt = newline + "\n     Run >Correct FSR position GUI< ?.\n     YES: press >y<. NO, press >n< \n"+ newline;

    correctInput = false; 
    while correctInput == false     % Wait for correct user input
        switch input(prompt, 's')   % Save user input
            case "y"
                correctInput = true; 
                enable_gui = true; 
            case "n"
                correctInput = true; 
                enable_gui = false;
            otherwise
                correctInput = false;
                warning("     Input not accepted")
        end 
    end 
end 

if enable_gui
    fprintf('\n     Running >Correct FSR position GUI< \n')
    offset = [];
    findStepOnFSR     % Open gui and return offset if exported
    pause

    if ~isempty(offset)
        step_index = offset; 
        if exist(filepath, 'file') == 2 
            prompt = newline + "\n     Want to overwrite. YES: press >y<. NO, press >n<"+ newline;
            correctInput = false; 
            while correctInput == false     % Wait for correct user input
                switch input(prompt, 's')   % Save user input
                    case "y"
                        correctInput = true; 
                        oversave = true; 
                    case "n"
                        correctInput = true; 
                        oversave = false;
                    otherwise
                        correctInput = false;
                        warning("     Input not accepted")
                end 
            end 
        else 
            oversave = true; 
        end 
    end
else 
    oversave = false; 
end

% GUI has been open but nothing saved
if isempty(offset)
    oversave = false; 
end

if oversave
    save(filepath,'offset')
    fprintf('\n     Data saved or oversaved \n')
end

if ~(exist(filepath, 'file') == 2)
    error("No verified step index has been created. The program will now terminate.")
end 


%% Manually readjust 
fprintf('Script section: Readjust data to Local Peak instead of FSR .  .  .  .  .  '); tic

correct_fall_edge = true; 

% Find falling edge and correct it
if correct_fall_edge 
    fprintf('\n     Finding stand-off / fall-edge by ankel trajectory instead of FSR \n')
    for sweep = 1:size(data{CTL, ANG},1)
        y = data{CTL, ANG}(sweep,:); 
        [pks,locs] = findpeaks(y, 'MinPeakProminence',3,'MinPeakDistance',500, 'Annotate','extents'); 
        rise = [3,5,7]; 
        fall = [2,4,6]; 
        offset_from_rise = 200; 
        for i = 1:3
            distancefromval = locs - (step_index{CTL}(sweep, rise(i))+offset_from_rise);  
            distancefromval(distancefromval < 0) = +Inf; 
            [~, indexToClosestMatch] = min(distancefromval);
            if(indexToClosestMatch < length(locs)) % check if not last value 
                if(pks(indexToClosestMatch) < pks(indexToClosestMatch+1)*0.5) % half size of next pk
                    indexToClosestMatch = indexToClosestMatch +1;
                end 
            end
            step_index{CTL}(sweep, fall(i)) = locs(indexToClosestMatch);
        end 
    end 
end 


%% Run offset GUI 

% Check if an already defined offset file exist for subject 
filename =  "/step_indexs/" + SubjectName + "_offset.mat";
filepath = fullfile(main_folderpath, filename);
if exist(filepath, 'file') == 2
    fprintf('\n     A >post< file already exist and will be loaded'); 
    load(filepath) % return 4x1 cell array called 'offset'
    step_index = offset; 
else 
    fprintf(2,'\n     No >post< file exist'); 
end

 

fprintf('\n     Running >Re-adjust GUI< \n')

offset = [];    % preparer for new input    
readjustFSR     % open gui
pause           % wait for user input 
    
if ~isempty(offset)
    if exist(filepath, 'file') == 2 
        prompt = newline + "     Want to overwrite. YES: press >y<. NO, press >n<"+ newline;
        correctInput = false; 
        while correctInput == false     % Wait for correct user input
            switch input(prompt, 's')   % Save user input
                case "y"
                    correctInput = true; 
                    oversave = true; 
                case "n"
                    correctInput = true; 
                    oversave = false;
                otherwise
                    correctInput = false;
                    warning("     Input not accepted")
            end 
        end 
    else 
        oversave = true; 
    end 

    if oversave
        step_index = offset; 
        save(filepath,'offset')
        disp("Data saved or oversaved")
    end         
end 


%% Save Data 
fprintf('Script section: Save data'); 

if ~exist(main_folderpath, 'dir') == 7
    newfolder = fullfile(main_folderpath, "data_preprocessed");
    mkdir(newfolder); 
end

save(main_folderpath + "/data_preprocessed/" + SubjectName + "_data", 'data')

