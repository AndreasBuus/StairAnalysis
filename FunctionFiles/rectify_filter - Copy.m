function [SOL, TA] = rectify_filter(SOL_raw, TA_raw, b, a) 

    for i = 1:size(SOL_raw,1)
        SOL(i,:) = filter(b,a, abs( detrend(SOL_raw(i,:),'constant')));       
        TA(i,:) = filter(b,a, abs( detrend(TA_raw(i,:),'constant')));
    end

end
