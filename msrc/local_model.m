function [ class_score, u1, u2, split_rst ] = local_model(fM, lbs, Su, Se, mean_a)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%     idx = (lbs <= 10);
%     fM = fM(idx,:);
%     lbs = lbs(idx,:);
    
     %%==========================%%
    
    [ X, Y, seq_Y, index] = sort_data( fM, lbs );
    [N,M] = size(X);
   
    X = X-repmat(mean_a,[N,1]);
    
    L = max(seq_Y); % count from 0
    class_score = zeros(L,2);
    u1 = zeros(L,M);
    u2 = zeros(L,M);
    split_rst = zeros(N,1);
    
    last_i = 0;
    for i=1:N
        k = seq_Y(i);
        if (i < N) && (k == seq_Y(i+1))
            continue
        end
        disp(k);
        [i_z1, i_u1, i_u2, i_sc] = find_LDA_split(X(last_i+1:i,:), Su, Se);
        split_rst(last_i+1:i,:) = i_z1;
        u1(k,:) = i_u1;
        u2(k,:) = i_u2;
        class_score(k,1) = Y(i);
        class_score(k,2) = i_sc;
        last_i = i;
    end
end

function [z1, u1, u2, sc] = find_LDA_split(X, Su, Se)
    [N,M] = size(X);
    F = inv(Se);
    
    z1 = rand(N,1);
    last_z1 = -ones(N,1);
    
    steps = 0;
    while (norm(z1-last_z1) > 0.01) && (norm((1-z1)-last_z1) > 0.01)  && (steps < 100)
        steps = steps+1;
        last_z1 = z1;

        idx = z1 >= 0.5;
        u1 = mean(X(idx,:));
        idx = z1 < 0.5;
        u2 = mean(X(idx,:));
        
%         n_z1 = sum(z1>=0.5);
%         n_z2 = N-n_z1;
%         G1 = -inv(n_z1*Su+Se);
%         G2 = -inv(n_z2*Su+Se);
%         u1 = zeros(1,M);
%         u2 = zeros(1,M);
%         for i = 1:N
%             vec = X(i,:);
%             if (z1 >= 0.5)
%                 dd = Se*G1*vec';
%                 u1 = u1-dd';
%             else
%                 dd = Se*G2*vec';
%                 u2 = u2-dd';
%             end
%         end

        bias = u1*(F)*u1'-u2*(F)*u2';

        for i = 1:N
            e1 = X(i,:);
            e2 = u1-u2;
            delta = e1*F*e2';
            if bias-2*delta < 0
                z1(i) = 1;
            else
                z1(i) = 0;
            end
        end


%         n_z1 = sum(z1);
%         n_z2 = N-n_z1;
%         u1 = zeros(1,M);
%         u2 = zeros(1,M);
%         for i=1:N
%             u1 = u1+z1(i)*X(i,:);
%             u2 = u2+(1-z1(i))*X(i,:);
%         end
%         u1 = u1/n_z1;
%         u2 = u2/n_z2;
%         
% %         n_z1
% %         pause
%         
%         v = (u1-u2)*inv_S;
%         
%         c1 = zeros(M,M);
%         c2 = zeros(M,M);
%         for i = 1:N
%             vec = X(i,:)-u1;
%             c1 = c1+z1(i)*vec*vec';
%             vec = X(i,:)-u2;
%             c2 = c2+(1-z1(i))*vec*vec';
%         end
%         ca = c1+c2;
%         c1 = c1/n_z1;
%         c2 = c2/n_z2;
%         ca = ca/N;
%         
%         sig1 = v*c1*v';
%         sig2 = v*c2*v';
%         siga = v*ca*v';
%         
%         bet1 = (trace(c1)-sig1)/(M-1);
%         bet2 = (trace(c2)-sig2)/(M-1);
%         beta = (trace(ca)-siga)/(M-1);
%         
%         for i = 1:N
%             x1 = X(i,:)-u1;
%             x2 = X(i,:)-u2;
%             
%             z = log(n_z2)-log(n_z1);
% %             z = z+(M-1)*(log(bet1)-log(bet2));
%             z = z+log(sig1)-log(sig2);
%             z = z+x1*x1'/beta-x2*x2'/beta;
%             
%             x1 = x1*v';
%             x2 = x2*v';
%             
%             z = z+(1/sig1-1/beta)*x1*x1-(1/sig2-1/beta)*x2*x2;
%             
%             z1(i) = 1/(1+exp(0.5*z));
%         end
        

    end
    
%     %Hotelling's T-squared
%     n_z1 = sum(z1>=0.5);
%     n_z2 = N-n_z1;
%     sc = (n_z1*n_z2)/N ;
%     diff_u = u(1,:)-u(2,:);
%     sc = sc * diff_u * F * diff_u';
%     t2 = sc*(N-M-1)/((N-2)*M);
%     sc = t2/finv(0.95, M, N-M-1);
    
    G = -inv(N*Su+Se);
    u_ori = zeros(1,M);
    for i=1:N
        vec = X(i,:);
        dd = Se*G*vec';
        u_ori = u_ori - dd';
    end
    
    b1 = u_ori*(F)*u_ori'-u1*(F)*u1';
    b2 = u_ori*(F)*u_ori'-u2*(F)*u2';
    sc = sum(z1>=0.5)*b1+sum(z1<0.5)*b2;

    for i=1:N
        e1 = X(i,:);
        if (z1(i) >= 0.5)
            e2 = u_ori-u1;
        else
            e2 = u_ori-u2;
        end
        sc = sc-2*e1*F*e2';
    end
    
    sc = sc;
       
end



