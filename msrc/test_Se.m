function [dist_Se, norm_e, dist_u, mean_f, dist_r] = test_Se(u,e,lbs,Su,Se)
    [N,M] = size(e);
    [y,I] = sort(lbs);
    
    uni_lbs = unique(lbs);
    L = size(uni_lbs,1);
    
    dist_Se = zeros(L,1);
    mean_f = zeros(L, M);
    dist_u = zeros(L,1);
    
    k = 0;
    ct = 0;
    E = zeros(1,M);
    U = zeros(1,M);
    
    try_Se = cell(L,1);
    cnt_l = zeros(L,1);
    
    for i = 1:N
        if (i>1) && (y(i) ~= y(i-1)) || (i==N)
            k = k+1;
            cnt_l(k,1) = ct;
            if ct < 2
                ct = 0;
                E = zeros(1,M);
                U = zeros(1,M);
                continue
            end
            
            mean_f(k,:) = mean(U+E);
            S = cov(E);
            dif = S-Se;
            dist_Se(k,1) = norm(dif(:));
            
            z = u(i-1,:) - mean_f(k,:);
            dist_u(k,1) = norm(z);
            
            R = U+E-repmat(mean_f(k,:),[ct,1]);
            try_Se{k} = cov(R);
            
            ct = 0;
            E = zeros(1,M);
            U = zeros(1,M);
        end
        ct = ct+1;
        E(ct,:) = e(I(i),:);
        U(ct,:) = u(I(i),:);
    end
    
    norm_e = zeros(N,1);
    for i=1:N
        norm_e(i,1) = norm(e(i,:));
    end
    
    mean_R = zeros(M,M);
    for k=1:L
        if cnt_l(k) < 2
            continue;
        end
        mean_R = mean_R+try_Se{k};
    end
    mean_R = mean_R/sum(cnt_l>1);
%     mean_R = mean_R/L;
    dist_r = zeros(L,1);
    for k = 1:L
        if cnt_l(k) < 2
            continue;
        end
        z = try_Se{k}-mean_R;
        dist_r(k,1) = norm(z(:));
    end
    
end