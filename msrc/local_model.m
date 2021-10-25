function [ lc_model ] = local_model(features, labels, gb_model, online, verbose)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%     idx = (lbs <= 10);
%     fM = fM(idx,:);
%     lbs = lbs(idx,:);
    
     %%==========================%%
     
     if nargin < 4
         online=false;
     end
     if nargin < 5
         verbose = false;
     end
     
     Su = gb_model.Su;
     Se = gb_model.Se;
%      mean_a = gb_model.mean;
     mean_a = mean(features);
     F = pinv(Se);
    
    [ X, Y, ctg, lbs] = sort_data( features, labels );
    [N,M] = size(X);
    L = size(ctg,1);
    
    if online
        lc_model.N = N;
        lc_model.mean = mean_a;
        lc_model.lX = cell(L,1);
        for k=1:L
            lc_model.lX{k} = X(ctg(k,1):ctg(k,2),:);
        end
        
    end
    
    X = X-repmat(mean_a,[N,1]);
    
    class_score = zeros(L,3);
    u1 = zeros(L,M);
    u2 = zeros(L,M);
    split_rst = cell(L,1);
    lb_map = zeros(L, 1);
    
    for k=1:L
        o_lb = Y(ctg(k,1),1);
        lb_map(k,1) = o_lb;
        map_k = gb_model.lb_map(o_lb);
        if verbose
            disp(['submodel for label ',num2str(o_lb)]);
            tic
        end
        [subg, i_u1, i_u2] = find_split(X(ctg(k,1):ctg(k,2),:), F);
        [i_sc] = calc_test(X(ctg(k,1):ctg(k,2),:), Su, Se, F, subg, i_u1, i_u2);
%         [subg, i_u1, i_u2] = gaussian_mixture(X(ctg(k,1):ctg(k,2),:), F);
%         [i_sc] = calc_stat(X(ctg(k,1):ctg(k,2),:), Su, Se, F, subg, i_u1, i_u2);
        if verbose
            toc
        end
        split_rst{k} =  subg;
        u1(k,:) = i_u1;
        u2(k,:) = i_u2;
        class_score(k,1) = o_lb;
        class_score(k,2) = i_sc;
        class_score(k,3) = ctg(k,3);
    end
    
    disp('local model done');
    
    lc_model.sts = class_score;
    lc_model.mu1 = u1;
    lc_model.mu2 = u2;
    lc_model.subg = split_rst;
    lc_model.lb_map = containers.Map(lb_map,1:L);
    
end

function [subg, u1, u2] = find_split(X, F)
    [N,M] = size(X);
    subg = rand(N,1);
    
    if (N==1)
        u1 = X;
        u2 = X;
        subg(1,1) = 0;
        return
    end
    
    if sum(subg >= 0.5) == 0
        subg(1,1) = 1;
    end
    if sum(subg < 0.5) == 0
        subg(1,1) = 0;
    end
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
%     %Hotelling's T-squared
%     n_z1 = sum(z1>=0.5);
%     n_z2 = N-n_z1;
%     sc = (n_z1*n_z2)/N ;
%     diff_u = u(1,:)-u(2,:);
%     sc = sc * diff_u * F * diff_u';
%     t2 = sc*(N-M-1)/((N-2)*M);
%     sc = t2/finv(0.95, M, N-M-1);
    
    [N,M] = size(X);
    
    SuF = Su*F;    
    G = -pinv(N*Su+Se);
    G = G*SuF;
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
%     sc = n1*b1+n2*b2+N*log(n1/N)+N*log(n2/N);
    sc = n1*b1+n2*b2;
   
    for i=1:N
        e1 = X(i,:);
        if (subg(i) >= 0.5)
            e2 = mu-u1;
        else
            e2 = mu-u2;
        end
        sc = sc-2*e1*F*e2';
        
%         e3 = 2*mu-u1-u2;
%         sc = sc-2*e1*F*e3';
    end
    
    
%     sc = sc / sqrt(2*(M*M+M+1));
    
    %for KL-divergence test and Beyainsian Information Cretera
    sc = sc/N; 
    
%     (sc-k)/sqrt(2*k);
%     sc = sc-log(N)*(M+M*M+1);
       
end

function [subg, u1, u2] = gaussian_mixture(X, F)
    [N,M] = size(X);
    
    if (N==1)
        u1 = X;
        u2 = X;
        subg(1,1) = 0;
        return
    end
    
    subg = rand(N,1);
        
    %EM
    last_z1 = -ones(N,1);
    steps = 0;
    while (norm(subg-last_z1) > 0.01) && (norm((1-subg)-last_z1) > 0.01)  && (steps < 100)
        steps = steps+1;
        last_z1 = subg;

        %max-step
        %calc p1 and p2
        n1 = sum(subg); n2 = N-n1;
        p1 = n1/N; p2 = 1-p1;
        %calc u1 and u2
        u1 = (subg'*X)/n1;
        u2 = ((1-subg)'*X)/(N-n1);
        
        %exp-step
        %calc subg
        for i=1:N
            r1 = X(i,:)-u1; r2 = X(i,:)-u2;
            m1 = r1*F*r1'; m2 = r2*F*r2';
            up = log(p1)+m1;
            %dn = log(p2)+m2+log(1+p1/p2*exp(m1-m2));
            dn = log(p2)+m2;
            if m1-m2 < -30                         %exp(-30) < 1e-13
                z = log(p1)-log(p2)+(m1-m2);       %log(1+x) -> exp(log(x)) when x->0
                if z < -30
                    dn = dn + 0;                       
                else
                    dn = dn+exp(z);
                end
            elseif m1-m2 > 30                      %exp(30) > 1e13
                dn = dn + log(p1)-log(p2)+(m1-m2); %log(1+x) -> log(x) when x->\infinity
            else
                dn = dn + log(1+p1/p2*exp(m1-m2));
            end
                     
            subg(i,1)= exp(up-dn);
        end
    end
end

function [sc] = calc_stat(X,Su,Se,F,subg,u1,u2)    
    [N,M] = size(X);
    
    SuF = Su*F;    
    G = -pinv(N*Su+Se);
    G = G*SuF;
    mu = zeros(1,M);
    for i=1:N
        vec = X(i,:);
        dd = Se*G*vec';
        mu = mu - dd';
    end
    
    n1 = sum(subg); n2 = N-n1;
    p1 = n1/N; p2 = 1-p1;
    sc = 0;
    
    for i=1:N
        r0 = X(i,:)-mu; m0 = r0*F*r0';
        r1 = X(i,:)-u1; r2 = X(i,:)-u2;
        m1 = r1*F*r1'; m2 = r2*F*r2';
        
        up = m0;
        %dn = log(p2)+m2+log(1+p1/p2*exp(m1-m2));
        dn = log(p2)+m2;
        if m1-m2 < -30                         %exp(-30) < 1e-13
            z = log(p1)-log(p2)+(m1-m2);       %log(1+x) -> exp(log(x)) when x->0
            if z < -30
                dn = dn + 0;                       
            else
                dn = dn+exp(z);
            end
        elseif m1-m2 > 30                      %exp(30) > 1e13
            dn = dn + log(p1)-log(p2)+(m1-m2); %log(1+x) -> log(x) when x->\infinity
        else
            dn = dn + log(1+p1/p2*exp(m1-m2));
        end
        sc = sc+(up-dn);
    end
    sc = sc/N;
       
end




