function [sl_X, sl_Y, rs_X, rs_Y] = select_k_from_every_class(features, labels, K_every)
%SELECT_K_FROM_EVERY_CLASS Summary of this function goes here
%   Detailed explanation goes here
[ X, Y, ctg, lbs] = sort_data( features, labels);
[N,M] = size(X);
L = size(ctg,1);
K_every = min(K_every,min(ctg(:,3)));

idx = false(N,1);
for k=1:L
    st = randperm(ctg(k,3),K_every);
    for i=1:K_every
        idx(ctg(k,1)+st(i)-1) = true;
    end
end

sl_X = X(idx,:);
sl_Y = Y(idx,:);
ndx = (~idx);
rs_X = X(ndx,:);
rs_Y = Y(ndx,:);
end

