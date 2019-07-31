function [ scores, tpr, fpr, thr ] = pca_defense( X, poisoned_labels, true_labels)
%L2_DEFENSE Summary of this function goes here
%   Detailed explanation goes here
    
    [N,M] = size(X);
    
    [pc, sc, lt] = pca(X);
    sq = lt.^2;
    sq = sq./sum(sq);
    z = 0;
    for i=1:size(lt,1)
        z = z+sq(i);
        if (z > 0.95)
            break;
        end
    end
    
    sp = pc(:,1:i); %first i components
    t = sp*sp';  %spanned space
    dif = X - X*t;
    
    scores = zeros(N,1);
    for i = 1:N
        scores(i,1) = norm(dif(i,:));
    end
    
%     [mean_sc, cnt_sc] = calc_mean(scores,poisoned_labels);
%     lb = zeros(max(poisoned_labels)+1,1);
%     lb(1,1) = 1;
%     [tpr,fpr,thr] = roc(mean_sc',lb');
    
    y=(poisoned_labels==true_labels);
    [tpr,fpr,thr] = roc(y',scores');

end

