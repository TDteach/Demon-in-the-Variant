function [Su, Se, mean_a, u_m] = global_model(fM, lbs, Su, Se)
%     idx = (lbs > 3);
%     fM = fM(idx,:);
%     lbs = lbs(idx,:);

    
%     N = size(lbs,1);
%     n = floor(N*0.1);
%     idx = randperm(N,n);
%     fM = fM(idx,:);
%     lbs = lbs(idx,:);
    
    %%==========================%%
    
    
    [ X, Y, seq_Y, index] = sort_data( fM, lbs );
    [N,M] = size(X);
    
    % mean removed
    mean_a = mean(X);
    X = X-repmat(mean_a,[N,1]);
    
    L = max(seq_Y);
    cnt_l = zeros(L,1);
    mean_f = zeros(L, M);
    last_i = 0;
    for i = 1:N
        k = seq_Y(i);
        if (i < N) && (k == seq_Y(i+1))
            continue
        end
        mean_f(k,:) = mean(X(last_i+1:i,:));
        cnt_l(k,1) = i-last_i;
        last_i = i;
    end
    
    u = zeros(N,M);
    e = zeros(N,M);
    if nargin<3
        for i = 1:N
            k = seq_Y(i);
            u(i,:) = mean_f(k,:);
            e(i,:) = X(i,:) - u(i,:);
        end
        Su = cov(u);
        Se = cov(e);        
    end
    
    dist_Su = 1e5;
    dist_Se = 1e5;
    
    while dist_Su+dist_Se > 0.1
        last_Su = Su;
        last_Se = Se;
     
        F = pinv(Se);
        SuF = Su*F;
        
        G_set = cell(L,1);
        for k = 1:L
            G = -pinv(cnt_l(k)*Su+Se);
            G = G*SuF;
            G_set{k} = G;
        end
        
        u_m = zeros(L, M);
        e = zeros(N,M);
        u = zeros(N,M);

        for i = 1:N
            vec = X(i,:);
            k = seq_Y(i);
            G = G_set{k};
            dd = Se*G*vec';
            u_m(k,:) = u_m(k,:) - dd';
%             u_m(k,:) = u_m(k,:) + (Su*(F+cnt_l(k)*G)*vec')';
        end

        for i = 1:N
            vec = X(i,:);
            k = seq_Y(i);
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
    
%     ret_u = zeros(N,M);
%     ret_e = zeros(N,M);
%     ret_u(index,:) = u;
%     ret_e(index,:) = e;
    
    'done'
end