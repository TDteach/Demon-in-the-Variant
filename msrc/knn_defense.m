function [ scores, tpr, fpr, thr ] = knn_defense( X, poisoned_labels, true_labels )
%KNN_DEFENSE Summary of this function goes here
%   Detailed explanation goes here
    [N,M] = size(X);
    l = true_labels(1:N,:);
    
    k = 5;
    
    [idx,D] = knnsearch(X,X,'k',k);
    scores = D(:,k);
    y=(poisoned_labels==l);
    [tpr,fpr,thr] = roc(y',scores');
end

