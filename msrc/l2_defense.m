function [ scores, tpr, fpr, thr ] = l2_defense( X, poisoned_labels, true_labels)
%L2_DEFENSE Summary of this function goes here
%   Detailed explanation goes here
    
    [N,M] = size(X);
    [X_mean, Y_cnt] = calc_mean(X,poisoned_labels);
    l = true_labels(1:N,:);
    
    scores = zeros(N,1);
    for i = 1:N
        k = poisoned_labels(i)+1;
        scores(i,1) = norm(X(i,:)-X_mean(k,:));
    end
    
    
%     [mean_sc, cnt_sc] = calc_mean(scores,poisoned_labels);
%     lb = zeros(max(poisoned_labels)+1,1);
%     lb(1,1) = 1;
%     [tpr,fpr,thr] = roc(mean_sc',lb');
   
    y=(poisoned_labels==l);
    [tpr,fpr,thr] = roc(y',scores');

end

