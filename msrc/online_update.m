function [lc_model] = online_update(x,y,gb_model, lc_model)
%ONLINE_UPDATE Summary of this function goes here
%   Detailed explanation goes here

    N = lc_model.N;
    mean_a = lc_model.mean;
    mean_a = (mean_a*N+x)/(N+1);
    lc_model.N = N+1;
    
    Su = gb_model.Su;
    Se = gb_model.Se;
    F = pinv(Se);
    
    k = lc_model.lb_map(y);
 
    X = lc_model.lX{k};
    X = X-repmat(mean_a,[lc_model.sts(k,3),1]);
    subg = lc_model.subg{k};
    u1 = lc_model.mu1(k,:);
    u2 = lc_model.mu2(k,:);
    [subg, u1, u2, X] = find_split_online(x-mean_a, X, F, subg, u1, u2);
    [sc] = calc_test(X,Su,Se,F,subg,u1,u2);
    lc_model.mu1(k,:) = u1;
    lc_model.mu2(k,:) = u2;
    lc_model.subg{k} = subg;
    lc_model.sts(k,2) = sc;
    lc_model.sts(k,3) = lc_model.sts(k,3)+1;
    X = X+repmat(mean_a,[lc_model.sts(k,3),1]);
    lc_model.lX{k} = X;
   
    disp('update done');
end

function [subg, u1, u2, X] = find_split_online(x, X, F, subg, u1, u2)
    
    X = [X;x];
    subg = [subg;rand(1)];
    [N,M] = size(X);
    
    last_z1 = -ones(N,1);
    
    %EM
    steps = 0;
    while (norm(subg-last_z1) > 0.01) && (norm((1-subg)-last_z1) > 0.01)  && (steps < 100)
        steps = steps+1;
        last_z1 = subg;

        %max-step
        %calc u1 and u2
        idx1 = (subg >= 0.5); idx2 = (~idx1);
        if (sum(idx1) == 0) || (sum(idx2) == 0)
            break;
        end
        if sum(idx1) == 1
            u1 = X(idx1,:);
        else
            u1 = mean(X(idx1,:));
        end
        if sum(idx2) == 1
            u2 = X(idx2,:);
        else
            u2 = mean(X(idx2,:));
        end
        
        bias = u1*(F)*u1'-u2*(F)*u2';
        e2 = u1-u2;
        for i = 1:N
            e1 = X(i,:);
            delta = e1*F*e2';
            if bias-2*delta < 0
                subg(i,1) = 1;
            else
                subg(i,1) = 0;
            end
        end

    end
    
end

function [sc] = calc_test(X,Su,Se,F,subg,u1,u2)
    [N,M] = size(X);
    
    G = -pinv(N*Su+Se);
    mu = zeros(1,M);
    for i=1:N
        vec = X(i,:);
        dd = Se*G*vec';
        mu = mu - dd';
    end
    
    
    b1 = mu*(F)*mu'-u1*(F)*u1';
    b2 = mu*(F)*mu'-u2*(F)*u2';
    n1 = sum(subg>=0.5);
    n2 = N-n1;
    sc = n1*b1+n2*b2;
   
    for i=1:N
        e1 = X(i,:);
        if (subg(i) >= 0.5)
            e2 = mu-u1;
        else
            e2 = mu-u2;
        end
        sc = sc-2*e1*F*e2';
        
    end
    
    sc = sc/N; 
       
end