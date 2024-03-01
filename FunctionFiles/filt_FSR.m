% Tested 28/02 - 2022

function [FSR] = filt_FSR(FSR)
    % FSR(sweep, data num) 
    
    Nsweep = size(FSR,1); 

    % Remove small fluxuations
    for n = 1:Nsweep
        for i = 1:size(FSR,2)
            if FSR(n,i) > 2.5
                FSR(n,i) = 5; 
            elseif FSR(n,i) < 2.5
                FSR(n,i) = 0; 
            end
        end
    end
    % Test code 
    % unique(FSR_control) 
    
    % Remove noise
    for sweep = 1:Nsweep % loop through sweeps
        edge_indexes = find(edge(FSR(sweep,end/2:end))); % Index pos of change in FSR
    
        for i = 1:numel(edge_indexes)-1
            dur = edge_indexes(i+1) - edge_indexes(i); 
            if dur < 300
                index = edge_indexes(i)+10000;
                
                if FSR(sweep, index+10) < 2.5
                    %  figure; hold on; % Test code 
                    %  sgtitle("sweep " + sweep + " Index pos " + index ) % Test code 
                    %  subplot(211); title("before") % Test code 
                    %  plot(FSR(sweep,:)) % Test code 
                    FSR(sweep,10000+edge_indexes(i)-1:10000+edge_indexes(i+1)+1) = 5;           
                    %  subplot(212); title("after") % Test code 
                    %  plot(FSR(sweep,:)) % Test code 
                    %  pause; % Test code 
                    %  close all; % Test code 
                end
            end
        end
    end 




    %for i = 1:size(FSR,1)
    %        test(i) = sum(edge(FSR(i,10000:end))) ; 
%         if 7 < sum(edge(FSR(i,10000:end)))
%             
%             edgeValues = edge(FSR(i,10000:end)); 
%             index = find(edgeValues); 
%             index = index + 10000;
% 
%             for k = 1:length(index) - 1
%                 distance(k) = index(k+1) - index (k); 
%             end 
%             result = find(distance == min(distance)); 
%             
%             if distance(result+1) < distance(result-1)
%                 FSR(i, index(result)-1:index(result+2) ) = 5; 
%             elseif distance(result+1) > distance(result-1)
%                 FSR(i, index(result-1)-1:index(result+1))=5;
%             end
%         end
    

end  