function [ sorted_X, sorted_Y, ctg, lbs] = sort_data( X, Y )
%INORDER Summary of this function goes here
%   Detailed explanation goes here
  [N,M] = size(X);  
  [sorted_Y,index] = sort(Y);
  sorted_X = X(index,:);
  
  lbs = zeros(N,1);
  ctg = zeros(1,3);
  k = 1; lb = sorted_Y(1); lbs(1,1) = 1;
  ctg(k,1) = 1; ctg(k,2) = 1; ctg(k,3) = 1;
  for i=2:N
    if sorted_Y(i) == lb
        ctg(k,2) = i; ctg(k,3) = ctg(k,3)+1;
    else 
        k = k+1; lb = sorted_Y(i);
        ctg(k,1) = i; ctg(k,2) = i; ctg(k,3) = 1;
    end
    lbs(i,1) = k;
  end
end

