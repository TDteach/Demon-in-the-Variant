function [Su, Se, A, G, score,u,e] = joint_bayesian(fM, lbs, Su, Se)
    idx = (lbs ~= 6);
    fM = fM(idx,:);
    lbs = lbs(idx,:);

    [N,M] = size(fM);
    [y,I] = sort(lbs);
    mean_a = mean(fM);
    
    
    fM = fM-repmat(mean_a,[N,1]);
    
    uni_lbs = unique(lbs);
    L = size(uni_lbs,1);
    
    rank_l = zeros(N,1);
    k = 1;
    for i=1:N
        if (i>1) && (y(i) ~= y(i-1))
            k = k+1;
        end
        rank_l(I(i)) = k;
    end
    
    cnt_l = zeros(L,1);
    mean_f = zeros(L, M);
    
    for i =1:N
        vec = fM(i,:);
        k = rank_l(i);
        mean_f(k,:) = mean_f(k,:)+vec;
        cnt_l(k) = cnt_l(k)+1;
    end
    for k =1:L
        mean_f(k,:) = mean_f(k,:)/cnt_l(k); 
    end
   
    
    u = zeros(N,M);
    e = zeros(N,M);
    if nargin<3
        for i = 1:N
            k = rank_l(i);
            vec = fM(i,:)-mean_f(k,:);
            u(i,:) = mean_f(k,:);
            e(i,:) = fM(i,:) - u(i,:);
        end
        Su = cov(u);
        Se = cov(e);        
    end
    
    A = zeros(1,1);
    G = zeros(1,1);
    score = zeros(1,1);
    
    
    dist_Su = 1e5;
    dist_Se = 1e5;
    
    while dist_Su+dist_Se > 0.1;
        last_Su = Su;
        last_Se = Se;
     
        F = inv(Se);
        SuF = Su*F;
        
        G_set = cell(L,1);
        for k = 1:L
            G = -pinv(cnt_l(k)*Su+Se);
            G = G*SuF;
            G_set{k} = G;
        end
        
        u_m = zeros(L, M);
%         e_m = zeros(L,M);
        e = zeros(N,M);
        u = zeros(N,M);

        for i = 1:N
            vec = fM(i,:);
            k = rank_l(i);
            G = G_set{k};
            dd = Se*G*vec';
            u_m(k,:) = u_m(k,:) - dd';
%             u_m(k,:) = u_m(k,:) + (Su*(F+cnt_l(k)*G)*vec')';
%             e_m(k,:) = e_m(k,:) + dd';
        end

        for i = 1:N
            vec = fM(i,:);
            k = rank_l(i);
            e(i,:) = vec-u_m(k);
            u(i,:) = u_m(k,:);
        end
        
        Su = cov(u);
        Se = cov(e);
        
        dif_Su = Su-last_Su;
        dif_Se = Se-last_Se;
        dist_Su = norm(dif_Su(:))
        dist_Se = norm(dif_Se(:))
        
        
    end
   
    score = zeros(N,1);
    A = zeros(M,M);
    
    inv_Su_Se = inv(Su+Se);
    tM = zeros(M*2,M*2);
    tM(1:M,1:M) = Su+Se;
    tM(M+1:M+M,M+1:M+M) = Su+Se;
    tM(M+1:M+M,1:M) = Su;
    tM(1:M, M+1:M+M) = Su;
    inv_tM = inv(tM);
    G = inv_tM(M+1:M+M, 1:M);
    F_G = inv_tM(1:M,1:M);
    A = inv_Su_Se - (F_G);
    
    
    for i=1:N
        va = fM(i,:);
        vb = mean_f(rank_l(i),:);
        score(i) = va*A*va'+vb*A*vb'-2*va*G*vb';
%         score(i) = va*Z*va'+vb*Z*vb'+(va+vb)*G*(va+vb)';
%         score(i) = -score(i);
    end
end