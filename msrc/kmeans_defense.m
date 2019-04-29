function [ret, gp] = kmeans_defense(features,labels, ori_labels)
%KMEANS_DEFENSE Summary of this function goes here
%   Detailed explanation goes here

L = max(labels);
gp = zeros(L+1,2);
ret = zeros(size(labels,1),2);
[coeff, score] = pca(features);
XX = score(:,1:2);

for k=0:max(labels)
    idx = labels==k;
    if (sum(idx) == 0)
        continue;
    end
    X = XX(idx,:);
    y = labels(idx,:);
    
    %draw picture
    oy = ori_labels(idx,:);
    s = zeros(size(y));
    c = zeros(size(y));
    for i=1:numel(y)
        if y(i) == oy(i)
            s(i) = 50;
            c(i) = 1;
        else 
            s(i) = 25;
            c(i) = 2;
        end
    end
    figure;
    if size(X,2) == 3
      scatter3(X(:,1), X(:,2), X(:,3),s,c);
    else
      scatter(X(:,1), X(:,2),s,c);
    end 
    break;


    clust = kmeans(X,2);
    score = silhouette(X, clust);
    ret(idx,1) = score;
    ret(idx,2) = k;
    gp(k+1,1) = mean(score);
    gp(k+1,2) = std(score);
end

end
