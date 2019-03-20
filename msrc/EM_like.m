function [ class_score, u1, u2, split_rst ] = EM_like(fM, lbs, Su, Se, A_bi, G_bi)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    [N,M] = size(fM);
    [y,I] = sort(lbs);
    mean_a = mean(fM);
    fM = fM-repmat(mean_a,[N,1]);
    
    X = zeros(size(fM));
    for i = 1:N
        X(i,:) = fM(I(i),:);
    end
    
    uni_lbs = unique(lbs);
    L = size(uni_lbs,1);
    class_score = zeros(L,1);
    u1 = zeros(L,M);
    u2 = zeros(L,M);
    split_rst = zeros(N,1);
    
    ct = 0; k = 0;
    for i=1:N
        ct = ct+1;
        if ((i<N) && (y(i) ~= y(i+1))) || (i==N)
            k = k+1;
            if (k > 10)
                break;
            end
            if ct < 2
                split_rst(i) = 1;
                class_score(k,1) = 0;
                u1(k,:) = X(i,:);
                u2(k,:) = X(i,:);
                ct = 0;
                continue;
            end
            
            
            disp(k);
            beg_i = i-ct+1;
            
            [z1,u,sc] = find_split(X(beg_i:i,:), Su, Se, G_bi);
            split_rst(beg_i:i) = z1;
            class_score(k,1) = sc;
            u1(k,:) = u(1,:);
            u2(k,:) = u(2,:);
%             t(k,1) = sum(z1)/size(z1,1);
%             disp(z1);
%             pause;
            
            
%             mean_f = mean(X(beg_i:i,:));
%             l = beg_i;
%             r = 0;
%             while r < i
%                 r = min(i, l+limit-1);
%                 l = max(r-limit+1,beg_i);
%                 
% %                 tx = X(l:r,:)-repmat(mean_f,[r-l+1,1]);
%                 tx = X(l:r,:);
%                 [z1, z2, u1, u2, e] = deal(tx, Su, Se);
%                 
% %                 disp(u1'*A*u1+mean_f*A*mean_f'-2*u1'*G*mean_f');
% %                 disp(u2'*A*u2+mean_f*A*mean_f'-2*u2'*G*mean_f');
% %                 disp(u1'*A*u1+u2'*A*u2-2*u1'*G*u2);
%                 
%                 disp(z1);
%                 
%                 l = r+1;
%                 
%                 if r-beg_i+1 > 100
%                     break;
%                 end
%             end 
%           
            ct = 0;
%             break;
        end
    end
  
end

function [z1,u, sc] = find_split(X, Su, Se, G_bi)
    [N,M] = size(X);
    F = inv(Se);
    SuF = Su*F;
    
    z1 = rand(N,1);
    last_z1 = -ones(N,1);
    
    
    steps = 0;
    while (norm(z1-last_z1) > 0.01) && (norm((1-z1)-last_z1) > 0.01)  && (steps < 100)
        steps = steps+1;
        last_z1 = z1;
        
        n_z1 = sum(z1>=0.5);
        G1 = -inv(n_z1*Su+Se);
        G1 = G1*SuF;
        G2 = -inv((N-n_z1)*Su+Se);
        G2 = G2*SuF;
        
        u = zeros(2,M);
        
        for i = 1:N
            if z1(i) >= 0.5
                k = 1;
                G = G1;
            else
                k = 2;
                G = G2;
            end
            vec = X(i,:);
            dd = Se*G*vec';
            u(k,:) = u(k,:) - dd';
        end
        
%         bias = u(1,:)*(F+G_bi)*u(1,:)'-u(2,:)*(F+G_bi)*u(2,:)';
        bias = u(1,:)*(F)*u(1,:)'-u(2,:)*(F)*u(2,:)';
%         bias = 0;
%         disp(bias);
        for i = 1:N
            e1 = X(i,:);
            e2 = u(1,:)-u(2,:);
%             delta = e1*G_bi*e2';
            delta = e1*F*e2';
            if bias-2*delta < 0
                z1(i) = 1;
            else
                z1(i) = 0;
            end
        end
        
%         disp(z1(1:10));
    end
    
    G = -inv(N*Su+Se);
    u_ori = zeros(1,M);
    for i=1:N
        vec = X(i,:);
        dd = Se*G*vec';
        u_ori = u_ori - dd';
    end
    
    b1 = u_ori*(F)*u_ori'-u(1,:)*(F)*u(1,:)';
    b2 = u_ori*(F)*u_ori'-u(2,:)*(F)*u(2,:)';
    sc = sum(z1>=0.5)*b1+sum(z1<0.5)*b2;
%     disp(sc);
    for i=1:N
        e1 = X(i,:);
        if (z1(i) >= 0.5)
            e2 = u_ori-u(1,:);
        else
            e2 = u_ori-u(2,:);
        end
        sc = sc-2*e1*F*e2';
    end
       
end

function [z1, z2, u1,u2,e] = deal(x, Su, Se)
    [n,M] = size(x);
%     z1 = rand(n,1);
    z1 = ones(n,1);
    z2 = 1-z1;
    last_1 = rand(n,1);
    
    k = 1;
    
    while (norm(z1-last_1) > 0.01) && (norm(z2-last_1) > 0.01)  && (k <= 100)
%         disp([num2str(k),':  ',num2str(norm(z1-last_1))]);
%         disp([num2str(k),':  ',num2str(norm(z2-last_1))]);
        k = k+1;
        
        last_1 = z1;
        
        Sx = zeros(n*M, n*M);
        for i = 1:n
            for j = 1:n
                d = z1(i)*z1(j)+z2(i)*z2(j);
                for u = 1:M
                    for v = 1:M
                        ii = (i-1)*M+u;
                        jj = (j-1)*M+v;
                        Sx(ii,jj) = Su(u,v)*d;
                        if i == j
                            Sx(ii,jj) = Sx(ii,jj)+Se(u,v);
                        end
                    end
                end
            end
        end

        xx = x';
        xx = reshape(xx, [n*M,1]);
        xx = inv(Sx)*xx;
        tx = reshape(xx,[M,n]);

        tu = Su*tx;
        u1 = zeros(M,1);
        u2 = zeros(M,1);
        for i = 1:n
            u1 = u1+tu(:,i)*z1(i);
            u2 = u2+tu(:,i)*z2(i);
        end
        e = Se*tx;

        for i = 1:n
            S = (z1(i)*z1(i)+z2(i)*z2(i))*Su+Se;
            inv_S = inv(S);
            vec_a = u1-u2;
            vec_b = u2+e(:,i);
            a = vec_a'*inv_S*vec_a;
            b = vec_a'*inv_S*vec_b;
            jug = -b/a;
            if jug < 0
                z1(i) = 0;
            elseif jug > 1
                z1(i) = 1;
            else
                z1(i) = jug;
            end
            z2(i) = 1-z1(i);
        end
        
    end
end

