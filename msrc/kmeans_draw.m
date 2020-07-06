function [ret] = kmeans_draw(features,labels, ori_labels)
%KMEANS_DEFENSE Summary of this function goes here
%   Detailed explanation goes here

L = max(labels);
gp = zeros(L+1,2);
ret = zeros(size(labels,1),2);
dis_sc = zeros(L+1,1);
% [coeff, XX] = pca(features);
XX = features;
m = 2;

for k=0:L
%     idx = (labels==0)+(labels==3)+(labels==5);
    idx = labels==k;
    idx = logical(idx);
    if (sum(idx) == 0)
        continue;
    end
    
    y = labels(idx,:);
    oy = ori_labels(idx,:);
    
%     X = XX(idx,:);
    x = XX(idx,:);
    x = x-mean(x);   
    [coeff, X] = pca(x);

    m = min(size(X,2),m);
    X = X(:,1:m);
    
    %draw picture
    
%     iid = (oy~=1);
%     X = X(iid,:);
%     y = y(iid,:);
%     oy = oy(iid,:);
    
    s = zeros(size(y));
    cc = zeros(size(y));
    c = cell(size(y));
    for i=1:numel(y)
        cc(i) = oy(i);
        if y(i) == oy(i);
            s(i) = 50;
            c{i} = 'intact';
        else 
            s(i) = 25;
            c{i} = 'infected';
        end
    end
    figure;
    if size(X,2) == 3
      h = scatter3(X(:,1), X(:,2), X(:,3),s,cc);
    else
      h = gscatter(X(:,1), X(:,2),cc);
    end 
    for i=1:size(h,1)
       if h(i).DisplayName == '0'
           h(i).DisplayName = 'Normal';
           h(i).Color=[0,0,1];
           h(i).Marker = 'o';
       else
%            h(i).DisplayName = ['Infected ',h(i).DisplayName];
           h(i).DisplayName = 'Infected';
           h(i).Marker = '+';
           h(i).Color(3)=0;
           h(i).Color(2)= 0;
%            h(i).Color(2)= (i-1)/(size(h,1)-1)/2;
           h(i).Color(1)=(i-1)/(size(h,1)-1)/2+0.5;
       end
    end
    set(gcf,'Position',[100 100 260 200])
    
    
    idx = y == oy;
    ma = mean(X(idx,:));
    ca = cov(X);
    mb = mean(X(~idx,:));
    
    did = sqrt((ma-mb) * pinv(ca) * (ma-mb)');
    
%     break;

    [clust, C] = kmeans(X,2);
    score = silhouette(X, clust);
    ret(idx,1) = score;
    ret(idx,2) = k;
    gp(k+1,1) = mean(score);
    gp(k+1,2) = std(score);
    
    n1 = sum(clust==1);
    n2 = sum(clust==2);
    n = size(X,1);
    u1 = mean(X(clust==1,:));
    s1 = cov(X(clust==1,:));
    ds1 = det(s1); ps1 = pinv(s1);
    u2 = mean(X(clust==2,:)); s2 = cov(X(clust==2,:));
    ds2 = det(s2); ps2 = pinv(s2);
    u = mean(X);
    s = cov(X);
    ds = det(s);
    ps = pinv(s);
    
    ans = 0;
    for i = 1:n
        x = X(i,:);
        ans = ans-(-0.5*log(ds) - 0.5* (x-u)*ps*(x-u)');
        if clust(i) == 1
            ans = ans+(-0.5*log(ds1) - 0.5* (x-u1)*ps1*(x-u1)');
        else
            ans = ans+(-0.5*log(ds2) - 0.5* (x-u2)*ps2*(x-u2)');
        end
    end
    dis_sc(k+1,1) = 2*ans-log(n)*(m+m*m);
    
end

% figure;
% plot(0:L, dis_sc');
% a = calc_anomaly_index(dis_sc/max(dis_sc));
% figure;
% plot(0:L, a);


end
