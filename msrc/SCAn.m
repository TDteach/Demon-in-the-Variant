function [gb_model, lc_model, ai] = SCAn(features, labels, ori_labels, ratio, draw_figure)
%SCAN Summary of this function goes here
%   Detailed explanation goes here

    if nargin < 4
        ratio = 0.03
    end    
    if nargin < 5
        draw_figure = true;
    end

    [gb_model,features,labels] = build_global(features, labels, ori_labels, ratio);
    lc_model = local_model(features,labels,gb_model);
    
    x = lc_model.sts(:,1);
    y = lc_model.sts(:,2);
    ai = calc_anomaly_index(y/max(y)); 
    if draw_figure
        figure;
        plot(x, y/max(y));
        figure;
        plot(x, ai);
        
%         u1 = lc_model.mu1;
%         u2 = lc_model.mu2;
%         
%         n = size(u1,1);
%         dis_u = zeros(n,1);
%         Se = gb_model.Se;
%         F = inv(Se);
%         for i=1:n
%             d = u1(i,:)-u2(i,:);
%             dis_u(i,1) = d * F * d';
%         end
%         b = log(det(Se));
%         dis_u = dis_u+b;
%         figure;
%         plot(x,dis_u);
    end
end


function [gb_model, lX, lY] = build_global(features,labels,ori_labels, ratio)
    gidx = (labels==ori_labels);
    c = rand(size(gidx));
    gidx = gidx.*c;
    gidx = gidx>(1-ratio);
    lidx = (~gidx);
    gX = features(gidx,:); gY = labels(gidx,:);
    lX = features(lidx,:); lY = labels(lidx,:);
    gb_model = global_model(gX, gY);
end