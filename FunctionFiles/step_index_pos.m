function [FSR_edge_index] = step_index_pos(FSR)
    
    % FSR (sweep, data num)

    for i = 1:size(FSR,1) % loop through sweeps
        %disp("Sweep " + i)
        edge_indexes = find(edge(FSR(i,:))); % Index pos of change in FSR
        a = FSR(i, end-2000:end); 
         if FSR(i,end) == 0  % remove index if FSR end on a decrease
            edge_indexes(end) = []; 
         elseif numel(unique(a)) > 1
            edge_indexes(end) = [];
            edge_indexes(end) = [];
        end

        for k = 0:8           
            FSR_edge_index(i,k+1) = edge_indexes(end-k); % load from end to mid
        end 
    end

% (sweep,1) -  Seventh  step, rising 
% (sweep,2) -  Sixth step, falling
% (sweep,3) -  Sixth step, rising
% (sweep,4) -  Fouth step, falling  <--  
% (sweep,5) -  Fouth step, rising  <-- 
% (sweep,6) -  Second step, falling  
% (sweep,7) -  Second step, rising  



end  