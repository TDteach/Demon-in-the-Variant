addpath(genpath('msrc'))

%initial trigger
mu = 0.5;
sigma = 0.13;
width = 32;
pt = zeros(width,width);
m = 7*7;
for i=1:m
    z = -1;
    while (z < 0) || (1 < z)
        z = normrnd(mu,sigma);
    end
    i = ceil(z*width);
    z = -1;
    while (z < 0) || (1 < z)
        z = normrnd(mu,sigma);
    end
    j = ceil(z*width);
    pt(i,j) = 1;
end
% imshow(pt);
% imwrite(pt,'trigger_try.png');
%%
%parameter settings
args.sigma = 1e-3;
args.epsilon = 0.05;
args.samples_per_draw = 50;
args.max_queries = 10000;
args.plateau_drop=2.0;
args.plateau_length=5;
args.momentum = 0.9;
args.max_lr=1e-2;
args.min_lr=5e-5;
args.adv_threshold = -1.0;
args.conservative = 2;


%%
initial_img = pt;
epsilon = args.epsilon;
lower = 0; %max(0,min(1,initial_img-args.epsilon));
upper = 1; %max(0,min(1,initial_img+args.epsilon));
queries_per_iter = args.samples_per_draw;
max_iters = floor(args.max_queries/queries_per_iter);
max_lr = args.max_lr;
goal_epsilon = epsilon;
adv_threshold = args.adv_threshold;
adv = initial_img;

%HISTROP VARIABLES
num_queries = 0;
grad = 0;
prev_adv = adv;
last_ls = zeros(2,1); n_ls = 0;
sc_record = [];
tg_record = [];
tm_record = [];

max_iters = 1;

%MAIN LOOP
for i =1:max_iters
    disp(['iter: ',num2str(i)]);
    
    prev_grad = grad;
    [rcd_loss, rcd_imag, rcd_time, grad] = get_grad(adv, queries_per_iter, args.sigma);
    sc_record = [sc_record;rcd_loss];
    tg_record = [tg_record;rcd_imag];
    tm_record = [tm_record;rcd_time];
    
    save(['record_',num2str(i)],'rcd_loss','tg_record','tg_record');
    
    min_loss = min(rcd_loss);
    loss = mean(rcd_loss);
    
    % CHECK IF STOP
    if min_loss < exp(2)
        disp(['[log] early stopping at iteration ',num2str(i)]);
        break;
    end
    
    %SIMPLE MOMENTUM
    grad = args.momentum * prev_grad + (1.0-args.momentum) * grad;
    
    % PLATEAU LR ANNEALING
    last_ls(n_ls+1) = loss;
    n_ls = n_ls+1;
    if (n_ls >= args.plateau_length) && (last_ls(n_ls) > last_ls(n_ls-args.plateau_length+1))
        max_lr = max(max_lr/args.plateau_drop, args.min_lr);
        disp(['[log] Annealing max_lr to ',num2str(max_lr)]);
        n_ls = 0; last_ls = zeros(2,1);
    end
    
    % SEARCH FOR LR AND EPSILON DECAY
    current_lr = max_lr;
    proposed_adv = adv- current_lr * sign(grad);
    prop_de = 0;
    if (loss < adv_threshold) && (epsilon > gloal_epsilon)
        prop_de = delta_epsilon;
    end
    while current_lr >= args.min_lr
        % PARTIAL INFORMATION ONLY
%         if k < NUM_LABELS
%             proposed_epsilon = max(epsilon - prop_de, goal_epsilon)
%             lower = max(0,min(1,initial_img-proposed_epsilon));
%             upper = max(0,min(1,initial_img+proposed_epsilon));
%         end
        
        proposed_adv = adv - current_lr*sign(grad);
        proposed_adv = max(lower,min(upper,proposed_adv));
        num_queries = num_queries+1;
%         if robust_in_top_k(target_class, proposed_adv, k)
        if true
            if prop_de > 0
                delta_epsilon = max(prop_de,0.1);
            end
            prev_adv = adv;
            adv = proposed_adv;
            epsilon = max(epsilon-prop_de/args.conservative, goal_epsilon);
            break;
        elseif current_lr >= args.min_lr * 2
            current_lr = curret_lr /2;
        else
            prop_de = prop_de / 2;
            if prop_de < 2e-3
                disp('Did not converge');
            end
            current_lr = max_lr;
            disp(['[log] backtracking eps to ',num2str(epsilon-prop_de)]);
        end
    end
    
    
    
end
%%
for i=1:size(tg_record,1)
    disp(i);
    imshow(tg_record{i});
    pause;
end

%%


function [sc] = do_one_iter(pt)
    target_label = 0;
    
    imwrite(pt,'trigger_try.png');
    disp('training begins');
    system('python3 pysrc/train_gtsrb.py > /dev/null');
    disp('training ends');

    home_folder = getenv('HOME');
    fo = fullfile(home_folder,'/data/npys');
    fn = 'out';
    [features,labels,ori_labels] = read_features(fn,fo);
    [gb_model, lc_model, ai] = SCAn(features, labels, ori_labels, 0.1, false);
    sc = ai(target_label+1);
    disp(['sc = ',num2str(ai(target_label+1))]);
end

function [rcd_loss, rcd_imag, rcd_time, grad] = get_grad(pt, num_iters, sigma)
    sum_grad = zeros(size(pt));
    rcd_loss = zeros(num_iters,1);
    rcd_imag = cell(num_iters,1);
    rcd_time = cell(num_iters,1);
    num_loop = num_iters/2;
    tic;
    for i = 1:num_loop
        noise = normrnd(0,1,size(pt));
        x_a = pt+noise*sigma; x_s = pt-noise*sigma;
        rcd_imag{i*2-1} = x_a; rcd_imag{i*2} = x_s;
        sc_a = do_one_iter(x_a);
        rcd_time{i*2-1} = datetime;
        sc_s = do_one_iter(x_s);
        rcd_time{i*2} = datetime;
        sum_grad = sum_grad+sc_a*x_a-sc_s*x_s;
        rcd_loss(i*2-1) = sc_a; rcd_loss(i*2) = sc_s;
    end
    toc;
    
    grad = sum_grad./(2*num_iters*sigma);
end


