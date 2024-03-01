function [SOL_raw, TA_raw, angle, FSR] = load_EMG(str_filename)
    arguments 
        str_filename string
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
    if var.Nsweep < 100
        for i = 10:var.Nsweep
            emg_varStr(i) = "dath0" + i;
            data_varStr(i) = "datl0" + i;
        end 
    else 
        for i = 10:99
            emg_varStr(i) = "dath0" + i;
            data_varStr(i) = "datl0" + i;
        end 
        for i = 100:var.Nsweep
            emg_varStr(i) = "dath" + i;
            data_varStr(i) = "datl" + i;
        end
    end


    % Call and out EMG data from 'var'
    TA = 1; SOL = 2;                    % var{i}(:, TA=1 / SOL=2) 
    fsr_select = 1; angle_select = 4;   % data{i}(:, FSR=1 / angle=4)

    for i = 1:var.Nsweep
        SOL_raw(i,:) = var.(emg_varStr(i))(:,SOL)*10^6; % unit [µV]
        TA_raw(i,:) = var.(emg_varStr(i))(:,TA)*10^6;   % unit [µV]
        %SOL_raw(i,:) = var.(emg_varStr(i))(:,SOL); 


        data{i} = var.(data_varStr(i));                 
        angle(i,:) = data{i}(:, angle_select);          % Do not know
        FSR(i,:) = data{i}(:,fsr_select);               % Binary
    end
end
