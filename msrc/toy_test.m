n = 100;
m = 2;
S = Se(1:m,1:m);
ca = rand(1,m)*10; sp_a = mvnrnd(ca,S, 100);
cb = rand(1,m)*10; sp_b = mvnrnd(cb,S, 100);

h1 =plot(ca(1), ca(2), '+b', 'MarkerSize',10, 'LineWidth',3);
hold on;
h2 =plot(cb(1), cb(2), '+r', 'MarkerSize',10, 'LineWidth',3);
hold on;
h3 =plot(sp_a(:,1), sp_a(:,2), '^b');
hold on;
h4 =plot(sp_b(:,1), sp_b(:,2), '*r');
norm(ca-cb)
legend([h1,h2,h3,h4],{'Center 1','Center 2', 'Catogery 1','Catogery 2'});

save('toy_examples','ca','cb','S','sp_a','sp_b');

X = [sp_a;sp_b];
Y = ones(size(X,1),1);

% function [subg, u1, u2] = find_split(X, F)
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
    
% end

hold on;
h1 =plot(u1(1), u1(2), '+b', 'MarkerSize',10, 'LineWidth',3);
hold on;
h2 =plot(u2(1), u2(2), '+r', 'MarkerSize',10, 'LineWidth',3);
F = pinv(S);

% function calc_test
    mu = mean(X);
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
%
%%

