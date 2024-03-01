function [rise, fall] = func_find_edge(step) 
   
    switch step 
        case 0 
            rise = 9; fall = 8; 
        case 2
            rise = 7; fall = 6; 
        case 4
            rise = 5; fall = 4; 
        case 6
            rise = 3; fall = 2; 
        case 7
            rise = 1; fall = []; 
        otherwise
            msg = "Not valid step obtion. Possible inputs: [0, 2, 4, 6]";
            error(msg)
    end
end

