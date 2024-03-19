function [step_index] = func_step_index_corr(varargin)  

%function [step_index] = func_step_index_corr(FSR, edge_num, move_num, sweep)  

inputNames = {'FSR', 'direction', 'edge_num', 'move_num', 'sweep', 'offset'};
p = inputParser;

defaultOffset = 7950;


for i = 1:length(inputNames)-1
    addOptional(p, inputNames{i}, []);
end

addParameter(p,'offset',defaultOffset)

parse(p, varargin{:});

offset = p.Results.offset;
FSR = p.Results.FSR;
edge_num = p.Results.edge_num;
move_num = p.Results.move_num;
sweep = p.Results.sweep;

 
edges_index_total = find(edge(FSR(sweep,offset:end)))+offset; %  Index pos of change in FSR

edges_index_removed = edges_index_total([1:edge_num-1, (edge_num+move_num):end]);

step_index = flip(edges_index_removed(1:9));
  
end
