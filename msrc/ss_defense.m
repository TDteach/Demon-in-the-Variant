function [ret_sc, score, v] = ss_defense(features,labels, ori_labels)
%KMEANS_DEFENSE Summary of this function goes here
%   Detailed explanation goes here

L = max(labels);
[N,M] = size(features);
gp = zeros(L+1,2);
ret = zeros(N,2);
XX = features;
ret_sc = zeros(N,1);
dis_sc = zeros(L+1,1);

for k=0:L
    idx = labels==k;
    if (sum(idx) < 2)
        continue;
    end
    X = XX(idx,:);
    y = labels(idx,:);
    oy = ori_labels(idx,:);
    yy = (y==oy);
    

    n = size(X,1);
    X = X- mean(X);
    [U, S, V] = svd(X);
    v = V(:,1);
    v = v./norm(v);

    score = X*v;
    ret_sc(idx,:) = score;
    
    
    [cc, C] = kmeans(score,2);
    
    u1 = mean(score(cc==1));
    s1 = std(score(cc==1));
    u2 = mean(score(cc==2));
    s2 = std(score(cc==2));
    u = mean(score);
    s = std(score);

    
    ans = 0;
    for i = 1:n
        x = score(i);
        ans = ans-(- log(s) - 0.5* (x-u)^2*s^(-2));
        if cc(i) == 1
            ans = ans+(-log(s1) - 0.5* (x-u1)^2*s1^(-2));
        else
            ans = ans+(-log(s2) - 0.5* (x-u2)^2*s2^(-2));
        end
    end
    dis_sc(k+1,1) = 2*ans-log(n)*(1+1);
    
    
%     plotroc(yy',cc');
%     break;
    
%     figure;
%     plot(yy,score,'xr');
%     
%     figure;
%     plotroc(yy',score');
end

figure;
plot(0:L, dis_sc');
a = calc_anomaly_index(dis_sc/max(dis_sc));
figure;
plot(0:L, a);

end
