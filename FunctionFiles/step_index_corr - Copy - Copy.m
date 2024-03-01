function [step_index] = step_index_corr(FSR, step_index, step_corr)  

    for sweep = 1:size(step_corr,1) % loop through sweeps
        if step_corr(sweep, 4) > 0
            %disp("sweep: " + sweep)           

            edge_indexes = find(edge(FSR(sweep,:))); % Index pos of change in FSR
            
            for index_pos_four = [step_corr(sweep, 4)-5:step_corr(sweep, 4)+5]
                 a = find(edge_indexes == index_pos_four);
                 if a > 0
                     step_index(sweep, 4) = edge_indexes(a); 
                     step_index(sweep, 3) = edge_indexes(a+1);
                     step_index(sweep, 2) = edge_indexes(a+2);
                 end
            end 

            for index_pos_five = [step_corr(sweep, 5)-5:step_corr(sweep, 5)+5]
                 b = find(edge_indexes == index_pos_five);
                 if b > 0
                     step_index(sweep, 5) = edge_indexes(b); 
                     step_index(sweep, 6) = edge_indexes(b-1);
                     step_index(sweep, 7) = edge_indexes(b-2);
                 end
            end 
        end
    end 
end


