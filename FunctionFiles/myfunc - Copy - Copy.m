function [output] = myfunc(input) 
    [rise, fall] = func_find_edge(input);

    output = rise + fall; 
end

