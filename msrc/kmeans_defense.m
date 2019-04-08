function [ret] = kmeans_defense(features,labels)
%KMEANS_DEFENSE Summary of this function goes here
%   Detailed explanation goes here

L = max(labels);
ret = zeros(size(labels,1),2);
for k=0:L
    idx = labels==k;
    if (sum(idx) == 0)
        continue;
    end
    X = features(idx,:);
    clust = kmeans(X,2);
    score = silhouette(X, clust);
    ret(idx,1) = score;
    ret(idx,2) = k;
end

end

% [X, y, Y, ind] = sort_data(features,labels);
% 
% [N,M] = size(X);
% ret_std = zeros(N,2);
% last_i = 0;
% k = 0;
% for i=1:N
%    if (i==N) || (Y(i) ~= Y(i+1))
%         [score] = calc_kmeans(X(last_i+1:i,:));
%         
%         
%         if (y(i)==26)
%             last_i
%             i
%             break;
%         end
%         
%         k = k+1;
%         ret_std(last_i+1:i,1) = score;
%         ret_std(last_i+1:i,2) = y(i);
%         last_i = i;
%         
% %         ret_std(k,2) = y(i);
%    end
% end
% 
% 
% 
% end
% 
% function [score] = calc_kmeans(X)
%   clust = kmeans(X,2);
%   score = silhouette(X, clust);
%  
% %   boxplot(score);
% %   ylim([-1,1]);
% %   pause;
% %   ret = std(score);
% end