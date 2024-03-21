function [SOL_raw, TA_raw, angle, FSR, FSR3, Trig, ForceZ] = load_EMG_v2(str_filename)
    arguments 
        str_filename string
    end


    if ~(exist(str_filename, 'file') == 2)
        main_folderpath = strrep(fileparts(matlab.desktop.editor.getActiveFilename),'\','/');
        dir(char(main_folderpath + "/Data_MrKick"))

        error_message =  "Unable to find file or directory " +  str_filename;
        errordlg(error_message , 'Error');
        error(error_message)
    end

    % Load data
    var = load(str_filename); % Load data 


    
    % Create string array to call data from 'var'
    emg_varStr = strings(1,var.Nsweep);        % Preallocation
    data_varStr = strings(1,var.Nsweep);       % Preallocation
    
    for i = 1:9
        emg_varStr(i) = "dath00" + i;
        data_varStr(i) = "datl00" + i;
    end
    for i = 10:99
        emg_varStr(i) = "dath0" + i;
        data_varStr(i) = "datl0" + i;
    end 
    for i = 100:var.Nsweep
        emg_varStr(i) = "dath" + i;
        data_varStr(i) = "datl" + i;
    end


    % Call and out EMG data from 'var'
    TA = 1; SOL = 2;                    % var{i}(:, TA=1 / SOL=2) 
    



    for i = 1:var.Nsweep
        SOL_raw(i,:) = var.(emg_varStr(i))(:,SOL)*10^6;  % unit [µV]
        TA_raw(i,:) = var.(emg_varStr(i))(:,TA)*10^6;    % unit [µV]
        
        data{i} = var.(data_varStr(i));           

        angle(i,:) = data{i}(:,5);         % correct 
        FSR(i,:)   = data{i}(:,1);         % correct      
        FSR3(i,:)  = data{i}(:,2);         % correct
        Trig(i,:)  = data{i}(:,3);         % 
        ForceZ(i,:)= data{i}(:,4);
    end
end
