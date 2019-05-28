function [Su, Se, mean_a, class_score, a] = our_defense(gX, gY, local_limit, draw_figure)
%OUR_DEFENSE Summary of this function goes here
%   Detailed explanation goes here

    switch nargin
        case 2
            local_limit = max(labels(:))+1;
            draw_figure = false;
        case 3
            draw_figure = false;
    end
    
    [Su, Se, mean_a] = global_model(gX, gY);
    
    lidx = (gY<local_limit);
    lX = gX(lidx,:);
    lY = gY(lidx,:);
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

