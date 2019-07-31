function [Su, Se, mean_a, mean_l, class_score, a] = our_defense(in_X, in_Y, known_ratio, local_limit, draw_figure)
%OUR_DEFENSE Summary of this function goes here
%   Detailed explanation goes here

    switch nargin
        case 2
            local_limit = max(labels(:))+1;
            known_ratio = 0.03;
            draw_figure = false;
        case 3
            local_limit = max(labels(:))+1;
            draw_figure = false;
        case 4
            draw_figure = false;
    end
    
    gidx = (labels==ori_labels);
    c = rand(size(gidx));
    gidx = gidx.*c;
    gidx = gidx>(1-known_ratio);
    gX = in_X(gidx,:);
    gY = in_Y(gidx,:);
    [Su, Se, mean_a, mean_l] = global_model(gX, gY);
    
    % to fastly see the initial results
    lidx = (gY<local_limit);
    lX = in_X(lidx,:);
    lY = in_Y(lidx,:);
    [ class_score, u1, u2, split_rst] = local_model(lX, lY, Su, Se, mean_a);
    
    x = class_score(:,1);
    y = class_score(:,2);
    a = calc_anomaly_index(y/max(y)); 
    if draw_figure
        figure;
        plot(x, y/max(y));
        figure;
        plot(x, a);
        
        n = size(u1,1);
        dis_u = zeros(n,1);
        F = inv(Se);
        for i=1:n
            d = u1(i,:)-u2(i,:);
            dis_u(i,1) = d * F * d';
        end
        b = log(det(Se));
        dis_u = dis_u+b;
        figure;
        plot(x,dis_u);
    end
end

