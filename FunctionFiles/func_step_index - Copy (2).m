function [edge_index, error_index] = func_step_index(FSR, varargin)
% func_step_index - This function returns the individual steps as a index
% position in an array. The first (1) array postion return index postion of
% rising edge of the force plate. The last index (9) position return the
% the seven step rising edge. 
%
% [edge_index(sweep, data num)] = func_step_index(FSR)

    defaultOffset = 7800; 
    p = inputParser;
    addOptional(p, 'offset', defaultOffset);
    parse(p, varargin{:});
    offset = p.Results.offset;

    error_num = 0; 
    error_index = []; 

    for sweep = 1:size(FSR,1) % loop through sweeps
        edge_indexes = find(edge(FSR(sweep,offset:end)))+offset; %  Index pos of change in FSR
        % ensure that 9 index are return or an error is returned
        data_length = length(edge_indexes);
        if data_length < 9 
            error_num = error_num + 1; 
            error_index(error_num) = sweep;
            pad_length = 9 - data_length;
            edge_indexes = padarray(edge_indexes', pad_length, 'post');
        end 
        
        edge_index(sweep,:) = flip(edge_indexes(1:9)); 
    
    end
% (sweep,1) -  Seventh  step, rising 
% (sweep,2) -  Sixth step, falling
% (sweep,3) -  Sixth step, rising
% (sweep,4) -  Fouth step, falling  <--  
% (sweep,5) -  Fouth step, rising  <-- 
% (sweep,6) -  Second step, falling  
% (sweep,7) -  Second step, rising  
% (sweep,8) -  Zero step, falling  
% (sweep,9) -  Zero step, rising  
end  