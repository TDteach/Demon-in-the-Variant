function [scores, gp, tpr, fpr, thr] = kmeans_defense(features,labels, ori_labels)
%KMEANS_DEFENSE Summary of this function goes here
%   Detailed explanation goes here

L = max(labels);
gp = zeros(L+1,2);
ret = zeros(size(labels,1),2);

m = 10;
% [coeff, XX] = pca(features);
mdl = rica(features, m);
XX = features*mdl.TransformWeights;
%XX = features;

for k=0:L
    disp(k)
%     idx = (labels==0)+(labels==3)+(labels==5);
    idx = labels==k;
    idx = logical(idx);
    if (sum(idx) == 0)
        continue;
    end
    
    
    X = XX(idx,:);

    [clust, C] = kmeans(X,2);
    score = silhouette(X, clust);
    ret(idx,1) = score;
    ret(idx,2) = k;
    gp(k+1,1) = mean(score);
    gp(k+1,2) = std(score);
      
end
scores = ret;
y=(labels==ori_labels);
[tpr,fpr,thr] = roc(y',scores');



end
