function [gb_model] = global_model(features, labels, verbose, Su, Se)
    
    if nargin < 3
        verbose = true
    end

    [ X, Y, ctg, lbs] = sort_data( features, labels );
    [N,M] = size(X);
    
    % mean removed
    mean_a = mean(X);
    X = X-repmat(mean_a,[N,1]);
    
    % calc mean for each category
    L =size(ctg,1);
    mean_f = zeros(L, M);
    lb_map = zeros(L, 1);
    for k = 1:L
        mean_f(k,:) = mean(X(ctg(k,1):ctg(k,2),:));
        lb_map(k,1) = Y(ctg(k,1),1);
    end
    
    % initialize mean and residual for each example
    u = zeros(N,M);
    e = zeros(N,M);
    if nargin<4
        for i = 1:N
            k = lbs(i);
            u(i,:) = mean_f(k,:);
            e(i,:) = X(i,:) - u(i,:);
        end
        Su = cov(u);
        Se = cov(e);        
    end
    
    %EM
    dist_Su = 1e5;
    dist_Se = 1e5;
    n_iters = 0;
    while (dist_Su+dist_Se > 0.01) && (n_iters < 100)
        n_iters = n_iters+1;
        last_Su = Su;
        last_Se = Se;
     
        F = pinv(Se);
        SuF = Su*F;
        
        G_set = cell(L,1);
        for k = 1:L
            G = -pinv(ctg(k,3)*Su+Se);
            G = G*SuF;
            G_set{k} = G;
        end
        
        u_m = zeros(L, M);
        e = zeros(N,M);
        u = zeros(N,M);

        %calc \mu for each category
        for i = 1:N
            vec = X(i,:);
            k = lbs(i);
            G = G_set{k};
            dd = Se*G*vec';
            u_m(k,:) = u_m(k,:) - dd';
            %u_m(k,:) = u_m(k,:) + (Su*(F+ctg(k,3)*G)*vec')';
        end

        %calc \epsilon for each example
        for i = 1:N
            vec = X(i,:);
            k = lbs(i);
            e(i,:) = vec-u_m(k,:);
            u(i,:) = u_m(k,:);
        end
        
        %max-step
        Su = cov(u);
        Se = cov(e);
        
        dif_Su = Su-last_Su;
        dif_Se = Se-last_Se;
        dist_Su = norm(dif_Su(:)); 
        dist_Se = norm(dif_Se(:));
        
        %show results of this step
        if verbose
            disp(['iteration ',num2str(n_iters),', dist_Su=',num2str(dist_Su),' dist_Se=',num2str(dist_Se)]);
        end
    end
    
    if n_iters >= 100
        Su = 0;
        Se = 0;
        disp('global model fail');
    else
        disp('global model done');
    end
    
    gb_model.Su = Su;
    gb_model.Se = Se;
    gb_model.mean = mean_a;
%     gb_model.mu = u;
    gb_model.lb_map = containers.Map(lb_map,1:L);

end