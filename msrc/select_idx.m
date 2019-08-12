function [selected_idx] = select_idx(idx, ratio, k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
n = size(idx,1);
if (k > 0) 
    selected_idx = zeros(n,1);
    z = randperm(n,n);
    for i=1:n
        if (idx(z(i)))
            if (k > 0) 
              k = k-1;
              selected_idx(z(i)) = 1;
            end 
        end
    end
else
    c = rand(size(idx));
    idx = idx.*c;
    selected_idx = idx>(1-ratio);
end 
selected_idx = logical(selected_idx);
end

