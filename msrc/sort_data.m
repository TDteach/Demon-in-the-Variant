function [ sorted_X, sorted_Y, seq_Y, index] = sort_data( X, Y )
%INORDER Summary of this function goes here
%   Detailed explanation goes here
  [N,M] = size(X);  
  [sorted_Y,index] = sort(Y);
  sorted_X = X(index);
  
  seq_Y = zeros(N,1);
  k = 1;
  for i=1:N
    if (i>1) && (sorted_Y(i) ~= sorted_Y(i-1))
        k = k+1;
    end
    seq_Y(i) = k;
  end
end

