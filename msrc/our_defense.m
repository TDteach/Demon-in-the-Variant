function [Su, Se, mean_a, a, gidx] = our_defense(features, labels, ori_labels, known_ratio, gidx)
%OUR_DEFENSE Summary of this function goes here
%   Detailed explanation goes here

    switch nargin
        case 2
            gidx = (labels>=0);
        case 3
            gidx = (labels==ori_labels);
        case 4
            gidx = (labels==ori_labels);
            c = rand(size(gidx));
            gidx = gidx.*c;
            gidx = gidx>(1-known_ratio);
    end
    
    
%     gidx = (labels>=5);
    
    gX = features(gidx,:);
    gY = labels(gidx,:);
    [Su, Se, mean_a] = global_model(gX, gY);
    
    a = 0;
    
    lidx = (labels<10);
    lX = features(lidx,:);
    lY = labels(lidx,:);
    [ class_score, u1, u2, split_rst] = local_model(lX, lY, Su, Se, mean_a);
    
    x = class_score(:,1);
    y = class_score(:,2)
    figure;
    plot(x, y/max(y));
    hold on;
    a = calc_anomaly_index(y);
    plot(x, a);
    figure;
    n = size(u1,1);
    dis_u = zeros(n,1);
    F = inv(Se);
    for i=1:n
        d = u1(i,:)-u2(i,:);
        dis_u(i,1) = d * F * d';
    end
    b = log(det(Se));
    dis_u = dis_u+b;
    plot(x,dis_u);
end

