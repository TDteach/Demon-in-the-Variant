function [u] = statistic_mean(X,Su, Se, mean_a)
%STATISTIC_MEAN Summary of this function goes here
%   Detailed explanation goes here
    [N,M] = size(X);
    X = X-repmat(mean_a,[N,1]);
    G = -inv(N*Su+Se);
    u = zeros(1,M);
    for i=1:N
        vec = X(i,:);
        dd = Se*G*vec';
        u = u - dd';
    end
    u = u+mean_a;
end

