function [ X_mean, Y_cnt ] = calc_mean( X, Y )
%CALC_MEAN Summary of this function goes here
%   Detailed explanation goes here

  [N,M] = size(X);
  L = max(Y)+1;
  Y_cnt = zeros(L,1);
  X_mean = zeros(L,M);
  for i=1:N
      k = Y(i)+1;
      X_mean(k,:) = X_mean(k,:)+X(i,:);
      Y_cnt(k) = Y_cnt(k)+1;
  end
  for i = 1:L
      if Y_cnt(i) > 0
        X_mean(i,:) = X_mean(i,:)/Y_cnt(i);
      else
        X_mean(i,:) = zeros(1,M);
      end
  end

end

